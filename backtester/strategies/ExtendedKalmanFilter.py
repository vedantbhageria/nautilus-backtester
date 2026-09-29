from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os.path
import sys
from collections import deque

import bt_compat as bt
import numpy as np

from strategy_base import PortfolioStrategy


def estimatefuture(X, P, Q, f_func,F_jac, drag=0, price=1.0):
    Xfuture = f_func(X, drag, price)
    F = F_jac(X, drag, price)
    Pfuture = F @ P @ np.transpose(F) + Q
    Pfuture = 0.5 * (Pfuture + Pfuture.T)      # force symmetry
    return Xfuture, Pfuture


def KalmanGain(Pold, R, H_jac, X_pred):
    H = H_jac(X_pred)
    K = Pold @ np.transpose(H) @ np.linalg.inv(H @ Pold @ np.transpose(H) + R)
    return K, H


def consolidatecurrent(Xold, Pold, K, Z, H, h_func):
    X = Xold + K @ (Z - h_func(Xold))
    P = (np.identity(np.shape(H)[1]) - K @ H) @ Pold
    P = 0.5 * (P + P.T)      # force symmetry
    return X, P
        
import math
        
class ExtendedKalmanTest(PortfolioStrategy):

    params = (
        ('k', 2.0),            # threshold in std devs of the innovation
        ('a', 7.0),             # stop loss this much above value
        ('warmup', 30),        # bars used to initialize each filter
        ('q_level', 0.2e-3),     # process noise on position (relative to est. R)
        ('q_vel', 0.2e-6),       # process noise on velocity (relative to est. R)
        ('q_acc', 0.2e-9),       # process noise on acceleration (relative to est. R)
        ('reversion', True),   # True: fade the deviation (default — matches
                                # every other strategy's convention); False:
                                # follow it (breakout)
        ('k_exit', 0.5),       # dead zone: exit to FLAT when |innov| < k_exit*sigma
        ('min_hold', 5),       # hold >= this many bars before exit/reverse
        ('cost_mult', 1.0),    # only enter if k*sigma > cost_mult * round-trip cost
        ('commission', 0.0002),# per-side fee fraction (fallback if broker lookup fails)
        ('drag_m', 1.0),       # drag "mass": smaller => stronger quadratic drag
        ('c_d_window', 180),   # rolling window (bars) for the live c_d regime estimate
        ('trend_bias', True),  # only long when Kalman velocity>0, short when <0
    )

    def setup(self):
        self.kf = {}       # data -> filter state
        self._chart = {}   # symbol -> [(epoch, est, upper, lower)]
        self.pause = {}
        for d in self.datas:
            self.kf[d] = {'warm': [], 'ready': False,
                          'X': None, 'P': None, 'Xf': None, 'Pf': None,
                          'Q': None, 'R': None, 'entry_len': None,
                          'c_d': 0.0, 'pbuf': None}
            self._chart[d._name] = []
            self.pause[d] = False
        self.vol_avg = {d: bt.indicators.SimpleMovingAverage(d.volume, period=self.params.warmup-1)
                for d in self.datas}
        self.vol_std = {d: bt.indicators.StandardDeviation(d.close, period=self.params.warmup - 1)
                for d in self.datas}
        

    @staticmethod
    def f_func(Xf, drag, price):
        velocity = Xf[1, 0]
        F_lin = np.array([[1.0, 1.0, 0.5],
                        [0.0, 1.0, 1.0],
                        [0.0, 0.0, 1.0]])
        drag_accel = drag * velocity * np.abs(velocity) / price
        drag_term = np.array([[0.0],
                            [0.0],
                            [drag_accel]])
        return F_lin @ Xf - drag_term
    @staticmethod
    def h_func(X):
        return np.array([[1.0, 0.0, 0.0]]) @ X
    @staticmethod
    def F_jac(X, drag, price):
        vel = X[1, 0]
        return np.array([[1.0, 1.0, 0.5],
                        [0.0, 1.0, 1.0],
                        [0.0, -2*drag*abs(vel)/price, 1.0  ]])
    @staticmethod
    def H_jac(X):
        return np.array([[1.0, 0.0, 0.0]])

    @staticmethod
    def _calc_cd(prices):
        """Regime measure = -lag1 autocorrelation of returns over `prices`.
        c_d>0 => mean-reverting, c_d<0 => trending, 0 => undefined."""
        buf = np.asarray(prices, dtype=float)
        if len(buf) < 5:
            return 0.0
        returns = np.diff(buf) / buf[:-1]
        with np.errstate(invalid='ignore', divide='ignore'):
            cc = np.corrcoef(returns[:-1], returns[1:])[0, 1]
        return float(-cc) if np.isfinite(cc) else 0.0

    def _init_filter(self, st):
        buf = np.array(st['warm'], dtype=float)
        idx = np.arange(len(buf))
        # Quadratic fit gives position, velocity AND acceleration:
        #   price(t) ~= c2*t^2 + c1*t + c0
        #   velocity      = d/dt  = 2*c2*t + c1
        #   acceleration  = d2/dt2 = 2*c2   (constant)
        c2, c1, c0 = np.polyfit(idx, buf, 2)
        resid = buf - (c2 * idx ** 2 + c1 * idx + c0)
        R = float(np.var(resid))
        R = max(R, (buf[-1] * 1e-6) ** 2, 1e-12)    # floor so S is never ~0
        st['R'] = np.array([[R]])
        st['Q'] = np.diag([R * self.params.q_level,
                           R * self.params.q_vel,
                           R * self.params.q_acc])
        n = len(buf) - 1                            # last warmup index
        pos = buf[-1]
        vel = 2.0 * c2 * n + c1
        acc = 2.0 * c2
        X0 = np.array([[pos], [vel], [acc]])
        P0 = np.diag([R, R, R]).astype(float)
        st['X'], st['P'] = X0, P0
        st['Xf'], st['Pf'] = estimatefuture(X0, P0, st['Q'], self.f_func, self.F_jac)
        st['ready'] = True

        # Rolling price buffer (per symbol) that keeps the regime estimate live:
        # seeded with the warmup prices, then updated each bar in on_bar so c_d
        # tracks the CURRENT regime instead of being frozen at warmup.
        st['pbuf'] = deque(buf.tolist(), maxlen=self.params.c_d_window)
        st['c_d'] = self._calc_cd(st['pbuf'])
        return st['c_d']

    def _step(self, st, price, avg_vol, std_vol, cur_vol, c_d):

        R = st['R']
        Xf, Pf = st['Xf'], st['Pf']                  # a-priori, from last bar
        y_pred = (self.h_func(Xf)).item()                     # predicted price for this bar


        rel_std = (std_vol / price) if price else 0.0
        vol_ratio = (cur_vol / avg_vol) if avg_vol else 1.0
        C = max(c_d, 0.0)
        drag = 0.5 * rel_std * C * vol_ratio / self.params.drag_m

        innov = price - y_pred
        # update with the actual measurement
        Z = np.array([[price]])
        K, H = KalmanGain(Pf, R, self.H_jac, Xf)
        Xc, Pc = consolidatecurrent(Xf, Pf, K, Z, H, self.h_func)
        S = (H @ Pf @ np.transpose(H) + R).item()    # innovation covariance
        S = max(S, 1e-12)     # numerical floor — S must stay positive
        # predict next bar
        st['X'], st['P'] = Xc, Pc
        st['Xf'], st['Pf'] = estimatefuture(Xc, Pc, st['Q'], self.f_func, self.F_jac, drag, price)
        return innov, S, y_pred

    def on_bar(self, d, price):
        st = self.kf[d]

        # Warmup: collect prices until the filter can be initialized.
        if not st['ready']:
            st['warm'].append(price)
            if len(st['warm']) >= self.params.warmup:
                self._init_filter(st)          # sets st['c_d'] for THIS symbol
            return 0

        raw_avg = self.vol_avg[d][0]
        avg_vol = raw_avg if raw_avg and not math.isnan(raw_avg) else 1.0

        raw_std = self.vol_std[d][0]
        std_vol = raw_std if raw_std and not math.isnan(raw_std) else 0.0

        # Rolling regime update: keep the last c_d_window prices and refresh the
        # live c_d estimate. Recomputed every 5 bars (regimes drift slowly; this
        # keeps the per-bar autocorr cost down across 100 symbols).
        st['pbuf'].append(price)
        if len(d) % 5 == 0:
            st['c_d'] = self._calc_cd(st['pbuf'])

        cur_vol = d.volume[0]
        c_d = st['c_d']
        innov, S, y_pred = self._step(st, price, avg_vol, std_vol, cur_vol, c_d)
        sigma = S ** 0.5
        band = self.params.k * sigma
        self._chart[d._name].append((self.bar_epoch(d), round(y_pred, 8),
                                     round(y_pred + band, 8),
                                     round(y_pred - band, 8)))
        self._diag_record(d, price, y_pred, innov, S, band, st)
        if band <= 0:
            return 0

        pos_size = self.getposition(d).size
        held = (len(d) - st['entry_len']) if st['entry_len'] is not None else 0

        # --- Dead zone (hysteresis): once in a trade, exit to FLAT when price
        #     snaps back inside k_exit*sigma of the estimate. Respect min_hold
        #     so a single noisy bar can't bounce us straight out. ---
        if pos_size != 0 and abs(innov) < self.params.k_exit * sigma and held >= self.params.min_hold:
            st['entry_len'] = None
            self.log('%s dead zone (|innov| %.6f < %.6f) -> FLAT'
                     % (d._name, abs(innov), self.params.k_exit * sigma))
            return (0, 0)

        # --- Cost-aware gate: only OPEN/REVERSE when the band edge we trade
        #     (k*sigma, in price) beats the round-trip fee. Exits above are
        #     never gated. ---
        try:
            comm = self.broker.getcommissioninfo(d).p.commission
        except Exception:
            comm = self.params.commission
        if band <= self.params.cost_mult * (2.0 * comm * price):
            return 0                          # edge too small to cover costs

        # innov = price - prediction. FOLLOW (reversion=False): trade in the
        # direction of the surprise (price broke ABOVE the band -> LONG).
        # REVERSION (reversion=True, default): fade it. Same convention as
        # every other strategy in this project — keep it consistent.
        long_sig = innov > band      # price above the upper band
        short_sig = innov < -band    # price below the lower band

        if self.params.reversion:
            long_sig, short_sig = short_sig, long_sig

        # --- Trend-direction regime bias: don't fight the Kalman velocity.
        #     Only allow longs in an up-trend (vel>0), shorts in a down-trend. ---
        if self.params.trend_bias:
            vel = st['X'][1, 0]
            if vel > 0:
                short_sig = False
            elif vel < 0:
                long_sig = False

        if long_sig:
            if pos_size > 0:                  # already long -> don't re-enter
                return 0
            if pos_size < 0 and held < self.params.min_hold:
                return 0                      # too soon to reverse a short
            st['entry_len'] = len(d)
            self.log('%s LONG (innov %.6f, band %.6f)' % (d._name, innov, band))
            return 1

        if short_sig:
            if pos_size < 0:                  # already short -> don't re-enter
                return 0
            if pos_size > 0 and held < self.params.min_hold:
                return 0                      # too soon to reverse a long
            st['entry_len'] = len(d)
            self.log('%s SHORT (innov %.6f, band %.6f)' % (d._name, innov, band))
            return -1

        return 0

    def build_chart_lines(self):
        k = self.params.k
        out = {}
        for d in self.datas:
            rows = self._chart[d._name]
            out[d._name] = [
                {'name': 'Kalman', 'color': '#58a6ff',
                 'points': [{'time': t, 'value': e} for t, e, u, l in rows]},
                {'name': '+%g sigma' % k, 'color': '#8b949e',
                 'points': [{'time': t, 'value': u} for t, e, u, l in rows]},
                {'name': '-%g sigma' % k, 'color': '#8b949e',
                 'points': [{'time': t, 'value': l} for t, e, u, l in rows]},
            ]
        return out

    def stop(self):
        self.log('(k=%g warmup=%d) Ending Value %.2f'
                 % (self.params.k, self.params.warmup,
                    self.broker.getvalue()), doprint=True)
