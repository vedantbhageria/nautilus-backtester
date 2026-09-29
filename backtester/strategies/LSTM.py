from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os.path
import sys
from collections import deque
from functools import partial

import bt_compat as bt
import numpy as np
import jax
import jax.numpy as jnp

from strategy_base import PortfolioStrategy

def _init_cell_params(key, hidden_size):
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        'W_ih': jax.random.normal(k1, (4 * hidden_size, 1)) * 0.1,
        'W_hh': jax.random.normal(k2, (4 * hidden_size, hidden_size)) * 0.1,
        'b': jnp.zeros(4 * hidden_size),
        'W_fc': jax.random.normal(k3, (1, hidden_size)) * 0.1,
        'b_fc': jnp.zeros(1),
    }


def _forward_single(params, x, hidden_size):
    """x: (seq_len, 1) normalized prices -> scalar normalized prediction."""
    h0 = jnp.zeros(hidden_size)
    c0 = jnp.zeros(hidden_size)

    def step(carry, xt):
        h, c = carry
        gates = params['W_ih'] @ xt + params['W_hh'] @ h + params['b']
        i, f, g, o = jnp.split(gates, 4)
        i, f, o = jax.nn.sigmoid(i), jax.nn.sigmoid(f), jax.nn.sigmoid(o)
        g = jnp.tanh(g)
        c = f * c + i * g
        h = o * jnp.tanh(c)
        return (h, c), None

    (h, _), _ = jax.lax.scan(step, (h0, c0), x)
    return (params['W_fc'] @ h + params['b_fc'])[0]


def _loss_single(params, x, y, hidden_size):
    pred = _forward_single(params, x, hidden_size)
    return (pred - y) ** 2


def _bias_correct(x, b, step):
    denom = 1 - b ** step
    if jnp.ndim(step) > 0:
        denom = denom.reshape(denom.shape + (1,) * (x.ndim - denom.ndim))
    return x / denom


def _adam_step(params, grads, m, v, step, lr, b1=0.9, b2=0.999, eps=1e-8):
    m = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g, m, grads)
    v = jax.tree.map(lambda v, g: b2 * v + (1 - b2) * g ** 2, v, grads)
    mhat = jax.tree.map(lambda m: _bias_correct(m, b1, step), m)
    vhat = jax.tree.map(lambda v: _bias_correct(v, b2, step), v)
    params = jax.tree.map(lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps),
                          params, mhat, vhat)
    return params, m, v


def _zeros_like_params(params):
    return jax.tree.map(jnp.zeros_like, params)


class LSTMTest(PortfolioStrategy):

    params = (
        ('k', 2.0),             # threshold in std devs of the innovation
        ('a', 7.0),             # kept for parity with KalmanTest (unused here)
        ('warmup', 60),         # bars collected before the initial fit
        ('seq_len', 20),        # lookback window fed to the LSTM each step
        ('hidden_size', 16),
        ('num_layers', 1),      # kept for parity with the old torch version
                                 # (unused — the hand-rolled cell is single-layer)
        ('lr', 1e-3),
        ('online_steps', 1),    # gradient steps taken per bar on the newest sample
        ('vol_window', 50),     # rolling window used to estimate innovation std (S)
        ('init_epochs', 150),   # epochs used to pretrain on the warmup window
        ('reversion', True),    # True: fade the deviation (default — matches
                                 # every other strategy's convention); False:
                                 # follow it (breakout)
        ('seed', 42),
        # ---- position management (ported from EMNonLinTest, each a toggle) --
        ('long_only', False),   # True: no short ENTRIES — an opposite (short)
                                 # signal flattens an open long to cash instead
                                 # of reversing into a short.
        ('max_hold', 0),        # time stop (bars). 0 = off. Flatten any position
                                 # that has been open >= max_hold bars; the clock
                                 # resets on every entry/flip (a reversal is a
                                 # new trade).
        ('dead', 0),            # settling period (bars). 0 = off. Suppress trade
                                 # ENTRIES for the first `dead` bars after this
                                 # symbol goes live, so the freshly-pretrained
                                 # model adapts online a bit before it's trusted.
                                 # (max_hold flattens still apply during it.)
        ('exit_band', 0.0),     # dead-zone exit. 0 = off. When |innovation| <
                                 # exit_band * band (price has reverted close to
                                 # the model's prediction, so the edge is gone),
                                 # exit to CASH instead of holding. This is the
                                 # main lever against "always in the market": at
                                 # 0 the strategy never voluntarily flattens and
                                 # only ever flips long<->short.
    )

    def setup(self):
        self._key = jax.random.PRNGKey(self.params.seed)
        self.models = {}
        self._chart = {}
        self._bar_cache = {}   # d -> (innov, S, y_pred, band), filled once per bar
        hs = self.params.hidden_size
        seq_len = self.params.seq_len

        self._fwd_single = jax.jit(partial(_forward_single, hidden_size=hs))
        self._grad_batch = jax.jit(
            jax.vmap(jax.grad(partial(_loss_single, hidden_size=hs)),
                     in_axes=(0, 0, 0)))
        self._fwd_batch = jax.jit(
            jax.vmap(partial(_forward_single, hidden_size=hs), in_axes=(0, 0)))
        n_windows = self.params.warmup - seq_len   # constant: every symbol's
                                                    # pretrain window count
        if n_windows > 0:
            self._fwd_pretrain = jax.jit(jax.vmap(self._fwd_single, in_axes=(None, 0)))
            self._grad_pretrain = jax.jit(jax.grad(
                lambda p, X, Y: jnp.mean((self._fwd_pretrain(p, X) - Y) ** 2)))

        self._all_ds = list(self.datas)
        self._slot = {d: i for i, d in enumerate(self._all_ds)}
        n = len(self._all_ds)
        init_keys = jax.random.split(self._key, n + 1)
        self._key = init_keys[0]
        self._P = jax.vmap(lambda k: _init_cell_params(k, hs))(init_keys[1:])
        self._M = _zeros_like_params(self._P)
        self._V = _zeros_like_params(self._P)
        self._adam_steps = np.zeros(n, dtype=np.int64)

        for d in self._all_ds:
            self.models[d] = {
                'warm': [],
                'ready': False,
                'mu': 0.0, 'sigma': 1.0,                    # placeholder until _init_model
                'buf': deque([0.0] * seq_len, maxlen=seq_len),   # pre-seeded: always full
                'innovs': deque(maxlen=self.params.vol_window),
                'last_pred': 0.0,
                # position management bookkeeping (see on_bar)
                'pos_sign': 0,        # -1/0/+1 sign of the position held last bar
                'bars_in_pos': 0,     # bars the current position has been open
                'bars_since_ready': 0,  # bars since this symbol went live (for `dead`)
            }
            self._chart[d._name] = []

    def _normalize(self, st, price):
        return (price - st['mu']) / st['sigma']

    def _denormalize(self, st, z):
        return z * st['sigma'] + st['mu']

    def _init_model(self, d, st):
        i = self._slot[d]
        buf = np.array(st['warm'], dtype=float)
        st['mu'] = float(buf.mean())
        st['sigma'] = float(buf.std())
        if st['sigma'] < 1e-8:
            st['sigma'] = 1.0

        norm = (buf - st['mu']) / st['sigma']
        seq_len = self.params.seq_len

        self._key, subkey = jax.random.split(self._key)
        p = _init_cell_params(subkey, self.params.hidden_size)
        m, v = _zeros_like_params(p), _zeros_like_params(p)
        step = 0

        if len(norm) > seq_len:
            X, Y = [], []
            for j in range(len(norm) - seq_len):
                X.append(norm[j:j + seq_len])
                Y.append(norm[j + seq_len])
            X = jnp.asarray(np.array(X), dtype=jnp.float32)[..., None]   # (n, seq_len, 1)
            Y = jnp.asarray(np.array(Y), dtype=jnp.float32)              # (n,)

            for _ in range(self.params.init_epochs):
                step += 1
                g = self._grad_pretrain(p, X, Y)
                p, m, v = _adam_step(p, g, m, v, step, self.params.lr)

        self._P = jax.tree.map(lambda whole, new: whole.at[i].set(new), self._P, p)
        self._M = jax.tree.map(lambda whole, new: whole.at[i].set(new), self._M, m)
        self._V = jax.tree.map(lambda whole, new: whole.at[i].set(new), self._V, v)
        self._adam_steps[i] = step

        for val in norm[-seq_len:]:
            st['buf'].append(float(val))
        st['ready'] = True

        seq = jnp.asarray(np.array(st['buf']), dtype=jnp.float32)[:, None]
        st['last_pred'] = float(self._fwd_single(p, seq))

    def _batch_online_step(self):
        ds = self._all_ds
        sts = [self.models[d] for d in ds]
        prices = [d.close[0] if len(d) else 0.0 for d in ds]
        zs = [self._normalize(st, p) for st, p in zip(sts, prices)]

        # 1) innovation from the prediction made LAST bar (before this bar's update)
        y_preds = [self._denormalize(st, st['last_pred']) for st in sts]
        innovs = [p - yp for p, yp in zip(prices, y_preds)]

        # 2) batched online-training step, directly on the persistent stacked state
        X = jnp.stack([jnp.asarray(np.array(st['buf']), dtype=jnp.float32)[:, None]
                       for st in sts])                       # (N, seq_len, 1)
        Y = jnp.asarray(zs, dtype=jnp.float32)                # (N,)
        for i in range(self.params.online_steps):
            self._adam_steps = self._adam_steps + 1
            steps = jnp.asarray(self._adam_steps)
            G = self._grad_batch(self._P, X, Y)
            self._P, self._M, self._V = _adam_step(
                self._P, G, self._M, self._V, steps, self.params.lr)

        # 3) roll each symbol's window forward, batched prediction for next bar
        for st, z in zip(sts, zs):
            st['buf'].append(float(z))
        Xnext = jnp.stack([jnp.asarray(np.array(st['buf']), dtype=jnp.float32)[:, None]
                           for st in sts])
        preds = np.asarray(self._fwd_batch(self._P, Xnext))   # ONE host sync for all N

        # 4) persist — ONLY for symbols that are ready and have a fresh bar
        # this tick; a not-yet-ready symbol's placeholder result is simply
        # discarded here (its real state got set by _init_model instead)
        for i, (d, st) in enumerate(zip(ds, sts)):
            if not st['ready']:
                continue
            if len(d) == 0 or len(d) == self._last_len.get(d, 0):
                continue    # no fresh bar for this symbol this tick

            st['last_pred'] = float(preds[i])

            innov, y_pred, price = innovs[i], y_preds[i], prices[i]
            st['innovs'].append(innov)
            if len(st['innovs']) >= 5:
                S = float(np.var(st['innovs']))
            else:
                S = (0.01 * price) ** 2
            band = self.params.k * (S ** 0.5)
            self._chart[d._name].append((self.bar_epoch(d), round(y_pred, 8),
                                         round(y_pred + band, 8),
                                         round(y_pred - band, 8)))
            self._diag_record(d, price, y_pred, innov, S, band, st)
            self._bar_cache[d] = (innov, S, y_pred, band)

    def next(self):
        self._batch_online_step()
        super().next()

    def prenext(self):
        self.next()

    def on_bar(self, d, price):
        st = self.models[d]

        overdue = False
        pos = self.getposition(d).size
        sign = 1 if pos > 0 else (-1 if pos < 0 else 0)
        if sign != st['pos_sign']:
            st['bars_in_pos'] = 0
        st['pos_sign'] = sign
        if sign != 0:
            st['bars_in_pos'] += 1
            if self.params.max_hold and st['bars_in_pos'] >= self.params.max_hold:
                overdue = True

        if not st['ready']:
            st['warm'].append(price)
            if len(st['warm']) >= self.params.warmup:
                self._init_model(d, st)
            return (0, 0) if overdue else 0

        cached = self._bar_cache.pop(d, None)
        if cached is None:
            return (0, 0) if overdue else 0   # buffer wasn't full yet this bar
        innov, S, y_pred, band = cached
        if band <= 0:
            return (0, 0) if overdue else 0

        # `dead` settling period: for the first `dead` bars after this symbol
        # went live, suppress trade ENTRIES (the freshly-pretrained model is
        # still adapting online). Position management (max_hold) still applies.
        st['bars_since_ready'] += 1
        if st['bars_since_ready'] <= self.params.dead:
            return (0, 0) if overdue else 0

        if overdue:
            return (0, 0)

        if self.params.exit_band and abs(innov) < self.params.exit_band * band:
            if pos != 0:
                return (0, 0)
            return 0

        long_sig = innov > band      # price above the upper band
        short_sig = innov < -band    # price below the lower band

        if self.params.reversion:
            long_sig, short_sig = short_sig, long_sig

        if long_sig:
            if pos > 0:
                return 0
            self.log('%s breakout ABOVE +band (innov %.6f, band %.6f) -> LONG'
                     % (d._name, innov, band))
            return 1
        if short_sig:
            if self.params.long_only:
                # no short ENTRIES — but an opposite signal still flattens an
                # open long (otherwise a long could only exit via exit_band/
                # max_hold/trailing-stop)
                if pos > 0:
                    self.log('%s short signal -> flatten long (long_only)' % d._name)
                    return (0, 0)
                return 0
            if pos < 0:
                return 0
            self.log('%s breakdown BELOW -band (innov %.6f, band %.6f) -> SHORT'
                     % (d._name, innov, band))
            return -1

        return 0

    def build_chart_lines(self):
        k = self.params.k
        out = {}
        for d in self.datas:
            rows = self._chart[d._name]
            out[d._name] = [
                {'name': 'LSTM', 'color': '#58a6ff',
                 'points': [{'time': t, 'value': e} for t, e, u, l in rows]},
                {'name': '+%g sigma' % k, 'color': '#8b949e',
                 'points': [{'time': t, 'value': u} for t, e, u, l in rows]},
                {'name': '-%g sigma' % k, 'color': '#8b949e',
                 'points': [{'time': t, 'value': l} for t, e, u, l in rows]},
            ]
        return out

    def stop(self):
        self.log('(k=%g warmup=%d seq_len=%d hidden=%d) Ending Value %.2f'
                 % (self.params.k, self.params.warmup, self.params.seq_len,
                    self.params.hidden_size, self.broker.getvalue()), doprint=True)
