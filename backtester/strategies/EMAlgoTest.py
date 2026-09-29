from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os.path
import sys
from collections import deque
import bt_compat as bt
import numpy as np

from strategy_base import PortfolioStrategy

_F = np.array([[1.0, 1.0], [0.0, 1.0]])   # position += velocity
_H = np.array([[1.0, 0.0]])               # measure position only

_LOG2PI = float(np.log(2.0 * np.pi))

try:
    from numba import njit
    _NUMBA = True
except ImportError:                     # graceful fallback: no-op decorator
    _NUMBA = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f


@njit(cache=True)
def _inv2(A):
    """Explicit 2x2 inverse (np.linalg.inv goes through LAPACK -- pure
    overhead at this size)."""
    det = A[0, 0]*A[1, 1] - A[0, 1]*A[1, 0]
    out = np.empty((2, 2))
    out[0, 0] = A[1, 1]/det; out[0, 1] = -A[0, 1]/det
    out[1, 0] = -A[1, 0]/det; out[1, 1] = A[0, 0]/det
    return out


@njit(cache=True)
def _cv_forward(Z, x0, P0, F, Q, r):
    """Kalman forward pass, H = [1, 0] (measure the level), scalar R = r.
    Same recursion as _kalman_forward below."""
    T = Z.shape[0]
    X_f = np.empty((T, 2, 1)); P_f = np.empty((T, 2, 2))
    X_p = np.empty((T, 2, 1)); P_p = np.empty((T, 2, 2))
    x, P = x0.copy(), P0.copy()
    K = np.zeros((2, 1))
    loglik = 0.0
    for t in range(T):
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        S = P_pred[0, 0] + r                 # H P_pred H^T + R, H = [1,0]
        innov = Z[t] - x_pred[0, 0]
        K = P_pred[:, :1] / S                # P_pred H^T S^-1
        x = x_pred + K * innov
        P = P_pred - K * P_pred[0, :]        # (I - K H) P_pred  (K (2,1) x row (2,) broadcasts to the 2x2 outer product K H P_pred)
        P = 0.5 * (P + P.T)
        X_p[t] = x_pred; P_p[t] = P_pred
        X_f[t] = x; P_f[t] = P
        loglik += -0.5 * (np.log(S) + innov*innov/S + _LOG2PI)
    return X_f, P_f, X_p, P_p, K, loglik


@njit(cache=True)
def _cv_smoother(X_f, P_f, X_p, P_p, F):
    """RTS backward pass -- same recursion as _rts_smoother below."""
    T = X_f.shape[0]
    X_s = X_f.copy(); P_s = P_f.copy()
    J = np.zeros((T, 2, 2))
    for t in range(T - 2, -1, -1):
        J[t] = P_f[t] @ F.T @ _inv2(P_p[t + 1])
        X_s[t] = X_f[t] + J[t] @ (X_s[t + 1] - X_p[t + 1])
        Pt = P_f[t] + J[t] @ (P_s[t + 1] - P_p[t + 1]) @ J[t].T
        P_s[t] = 0.5 * (Pt + Pt.T)
    return X_s, P_s, J


@njit(cache=True)
def _cv_lag(P_f, J, K_last, F, P0, P_p0):

    T = P_f.shape[0]
    P_lag = np.zeros((T, 2, 2))
    J_prior = P0 @ F.T @ _inv2(P_p0)
    IKH = np.eye(2)
    IKH[:, 0] -= K_last[:, 0]                # I - K H, H = [1,0]
    if T >= 2:
        P_lag[T - 1] = IKH @ F @ P_f[T - 2]
        for t in range(T - 2, 0, -1):
            P_lag[t] = P_f[t] @ J[t - 1].T + J[t] @ (P_lag[t + 1] - F @ P_f[t]) @ J[t - 1].T
        P_lag[0] = P_f[0] @ J_prior.T + J[0] @ (P_lag[1] - F @ P_f[0]) @ J_prior.T
    else:
        P_lag[0] = IKH @ F @ P0
    return P_lag, J_prior


def _em_fit_cv(prices, x0, P0, Q0, R0, n_iter, r_floor):

    Z = np.ascontiguousarray(prices, dtype=float)
    T = len(Z)
    F = _F
    x0 = np.ascontiguousarray(x0, dtype=float)
    P0 = np.ascontiguousarray(P0, dtype=float)
    Q = np.ascontiguousarray(Q0, dtype=float)
    r = float(R0[0, 0])
    loglik_hist = []

    for _ in range(n_iter):
        X_f, P_f, X_p, P_p, K_last, ll = _cv_forward(Z, x0, P0, F, Q, r)
        loglik_hist.append(ll)
        X_s, P_s, J = _cv_smoother(X_f, P_f, X_p, P_p, F)
        P_lag, J_prior = _cv_lag(P_f, J, K_last, F, P0, P_p[0])

        # prior's smoothed posterior (see the note in the general path)
        x0_s = x0 + J_prior @ (X_s[0] - X_p[0])
        P0_s = P0 + J_prior @ (P_s[0] - P_p[0]) @ J_prior.T
        P0_s = 0.5 * (P0_s + P0_s.T)

        # M-step sufficient statistics, vectorised over t
        X_prev = np.concatenate((x0_s[None], X_s[:-1]))
        P_prev = np.concatenate((P0_s[None], P_s[:-1]))
        S11 = P_s.sum(0) + np.einsum('tia,tja->ij', X_s, X_s)
        S10 = P_lag.sum(0) + np.einsum('tia,tja->ij', X_s, X_prev)
        S00 = P_prev.sum(0) + np.einsum('tia,tja->ij', X_prev, X_prev)

        Q = (S11 - F @ S10.T - S10 @ F.T + F @ S00 @ F.T) / T
        Q = 0.5 * (Q + Q.T)
        w, v = np.linalg.eigh(Q)
        Q = np.ascontiguousarray((v * np.clip(w, r_floor, None)) @ v.T)

        innov = Z - X_s[:, 0, 0]                       # z_t - H x_s
        r = max(float((innov @ innov + P_s[:, 0, 0].sum()) / T), r_floor)

    X_f, P_f, _, _, _, ll = _cv_forward(Z, x0, P0, F, Q, r)
    loglik_hist.append(ll)
    return Q, np.array([[r]]), X_f[-1].copy(), P_f[-1].copy(), loglik_hist


def em_fit(prices, x0, P0, Q0, R0, F=_F, H=_H, n_iter=5, r_floor=1e-12):
    # Fast scalar path for this module's fixed constant-velocity model; the
    # general numpy path below stays as reference and for custom F/H.
    if (F is _F or np.array_equal(F, _F)) and (H is _H or np.array_equal(H, _H)):
        return _em_fit_cv(prices, x0, P0, Q0, R0, n_iter, r_floor)

    """Z = [np.array([[p]], dtype=float) for p in prices]
    T = len(Z)
    n = x0.shape[0]
    Q, R = Q0.copy(), R0.copy()
    loglik_hist = []

    for _ in range(n_iter):
        X_f, P_f, X_p, P_p, K_last, ll = _kalman_forward(Z, x0, P0, F, H, Q, R)
        loglik_hist.append(ll)
        X_s, P_s, J = _rts_smoother(X_f, P_f, X_p, P_p, F)
        P_lag = _lag_one_covariance(P_f, J, K_last, F, H, P0, P_p[0])

        J_prior = P0 @ F.T @ np.linalg.inv(P_p[0])
        x0_s = x0 + J_prior @ (X_s[0] - X_p[0])
        P0_s = P0 + J_prior @ (P_s[0] - P_p[0]) @ J_prior.T
        P0_s = 0.5 * (P0_s + P0_s.T)

        S11 = np.zeros((n, n)); S10 = np.zeros((n, n)); S00 = np.zeros((n, n))
        R_new = np.zeros_like(R)
        for t in range(T):
            xt, Pt = X_s[t], P_s[t]
            xtm1 = x0_s if t == 0 else X_s[t - 1]
            Ptm1 = P0_s if t == 0 else P_s[t - 1]
            Plag_t = P_lag[t]

            S11 += Pt + xt @ xt.T
            S10 += Plag_t + xt @ xtm1.T
            S00 += Ptm1 + xtm1 @ xtm1.T

            innov = Z[t] - H @ xt
            R_new += innov @ innov.T + H @ Pt @ H.T

        # F is held FIXED (not re-estimated), so the ML Q given fixed F is
        # the residual covariance of (x_t - F x_{t-1}), not the usual
        # S11 - F@S10^T form (that formula assumes F is also being updated).
        Q_new = (S11 - F @ S10.T - S10 @ F.T + F @ S00 @ F.T) / T
        Q_new = 0.5 * (Q_new + Q_new.T)
        R_new = R_new / T

        # numerical floor, same pattern used elsewhere in this codebase
        # (KalmanFilter._init_filter's R floor) so S never collapses to ~0.
        w, v = np.linalg.eigh(Q_new)
        w = np.clip(w, r_floor, None)
        Q_new = (v * w) @ v.T
        R_new = np.clip(R_new, r_floor, None)

        Q, R = Q_new, R_new

    X_f, P_f, _, _, _, ll = _kalman_forward(Z, x0, P0, F, H, Q, R)
    loglik_hist.append(ll)
    return Q, R, X_f[-1], P_f[-1], loglik_hist"""


class EMTest(PortfolioStrategy):

    params = (
        ('k', 2.0),             # threshold in std devs of the innovation
        ('a', 7.0),             # stop loss this much above value 
        ('warmup', 180),        # bars used for the initial polyfit seed
        ('q_level', 0.1e-3),    # INITIAL process noise on level (relative to est. R)
        ('q_vel', 0.1e-6),      # INITIAL process noise on slope (relative to est. R)
        ('reversion', True),    # True: fade the deviation (default — matches
                                 # every other strategy's convention); False:
                                 # follow it (breakout)
        ('em_window', 720),     # bars of rolling history the EM refit uses
        ('em_interval', 1440),   # refit Q,R every this many bars
        ('em_iters', 5),        # EM iterations per refit
    )

    F = _F
    H = _H

    def setup(self):
        self.kf = {}
        self._chart = {}
        for d in self.datas:
            self.kf[d] = {'ready': False, 'warm': [],
                          'X': None, 'P': None, 'Xf': None, 'Pf': None,
                          'Q': None, 'R': None,
                          'pbuf': deque(maxlen=self.params.em_window),
                          'bars_since_em': 0}
            self._chart[d._name] = []

    def _init_filter(self, st):
        buf = np.array(st['warm'], dtype=float)
        idx = np.arange(len(buf))
        m, b = np.polyfit(idx, buf, 1)
        resid = buf - (m * idx + b)
        R = float(np.var(resid))
        R = max(R, (buf[-1] * 1e-6) ** 2, 1e-12)
        st['R'] = np.array([[R]])
        st['Q'] = np.array([[R * self.params.q_level, 0.0],
                            [0.0, R * self.params.q_vel]])
        X0 = np.array([[buf[-1]], [m]])
        P0 = np.array([[R, 0.0], [0.0, R]])
        st['X'], st['P'] = X0, P0
        st['Xf'] = self.F @ X0
        st['Pf'] = self.F @ P0 @ self.F.T + st['Q']
        st['pbuf'].extend(buf.tolist())
        st['ready'] = True

    def _refit_em(self, d, st):

        prices = list(st['pbuf'])
        x0 = np.array([[prices[0]], [0.0]])
        P0 = st['P']            # current uncertainty as the batch's prior
        Q, R, x_last, P_last, ll = em_fit(prices, x0, P0, st['Q'], st['R'],
                                          F=self.F, H=self.H,
                                          n_iter=self.params.em_iters)
        self.log('%s EM refit: Q=diag(%.3g,%.3g) R=%.3g  loglik %.1f -> %.1f'
                 % (d._name, Q[0, 0], Q[1, 1], R[0, 0], ll[0], ll[-1]))
        st['Q'], st['R'] = Q, R
        st['X'], st['P'] = x_last, P_last
        st['Xf'] = self.F @ x_last
        st['Pf'] = self.F @ P_last @ self.F.T + Q

    def _step(self, st, price):
        # Scalar fast path of _kalman_step for the fixed 2-state CV model
        # (identical math; drops ~25 tiny numpy calls per bar). Q may be a
        # full symmetric matrix after an EM refit, so keep its off-diagonal.
        X, P, Q = st['X'], st['P'], st['Q']
        x0, x1 = X[0, 0], X[1, 0]
        p00, p01, p11 = P[0, 0], P[0, 1], P[1, 1]
        q00, q01, q11 = Q[0, 0], Q[0, 1], Q[1, 1]
        r = st['R'][0, 0]

        # predict
        xp0 = x0 + x1
        b00 = p00 + 2.0*p01 + p11 + q00
        b01 = p01 + p11 + q01
        b11 = p11 + q11
        S = b00 + r
        innov = price - xp0
        # update
        K0 = b00 / S; K1 = b01 / S
        nx0 = xp0 + K0*innov
        nx1 = x1 + K1*innov
        omk = 1.0 - K0
        n00 = omk*b00
        n11 = b11 - K1*b01
        n01 = 0.5*(omk*b01 + (b01 - K1*b00))    # symmetrized, as before

        y_pred = st['Xf'][0, 0]                 # last bar's forecast for THIS bar
        st['X'] = np.array([[nx0], [nx1]])
        st['P'] = np.array([[n00, n01], [n01, n11]])
        # roll the one-bar-ahead forecast (Xf/Pf), same as F@x / F@P@F.T + Q
        st['Xf'] = np.array([[nx0 + nx1], [nx1]])
        st['Pf'] = np.array([[n00 + 2.0*n01 + n11 + q00, n01 + n11 + q01],
                             [n01 + n11 + q01,           n11 + q11]])
        return innov, S, y_pred

    def on_bar(self, d, price):
        st = self.kf[d]

        if not st['ready']:
            st['warm'].append(price)
            if len(st['warm']) >= self.params.warmup:
                self._init_filter(st)
            return 0

        st['pbuf'].append(price)
        st['bars_since_em'] += 1
        if (st['bars_since_em'] >= self.params.em_interval
                and len(st['pbuf']) >= self.params.em_window):
            self._refit_em(d, st)
            st['bars_since_em'] = 0

        innov, S, y_pred = self._step(st, price)
        band = self.params.k * (S ** 0.5)
        self._chart[d._name].append((self.bar_epoch(d), round(y_pred, 8),
                                     round(y_pred + band, 8),
                                     round(y_pred - band, 8)))
        self._diag_record(d, price, y_pred, innov, S, band, st)
        if band <= 0:
            return 0

        # innov = price - prediction. FOLLOW (reversion=False): trade in the
        # direction of the surprise (price broke ABOVE the band -> LONG).
        # REVERSION (reversion=True, default): fade it. Same convention as
        # every other strategy in this project — keep it consistent.
        long_sig = innov > band
        short_sig = innov < -band

        if self.params.reversion:
            long_sig, short_sig = short_sig, long_sig

        pos_size = self.getposition(d).size

        if long_sig:
            if pos_size > 0:
                return 0
            self.log('%s breakout ABOVE +band (innov %.6f, band %.6f) -> LONG'
                     % (d._name, innov, band))
            return 1
        if short_sig:
            if pos_size < 0:
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
                {'name': 'EM-Kalman', 'color': '#58a6ff',
                 'points': [{'time': t, 'value': e} for t, e, u, l in rows]},
                {'name': '+%g sigma' % k, 'color': '#8b949e',
                 'points': [{'time': t, 'value': u} for t, e, u, l in rows]},
                {'name': '-%g sigma' % k, 'color': '#8b949e',
                 'points': [{'time': t, 'value': l} for t, e, u, l in rows]},
            ]
        return out

    def stop(self):
        self.log('(k=%g warmup=%d em_window=%d) Ending Value %.2f'
                 % (self.params.k, self.params.warmup, self.params.em_window,
                    self.broker.getvalue()), doprint=True)
