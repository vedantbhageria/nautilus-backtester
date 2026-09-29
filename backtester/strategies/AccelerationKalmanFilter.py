
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os.path
import sys

import bt_compat as bt
import numpy as np

from strategy_base import PortfolioStrategy

def estimatefuture(X, P, Q, f_func,F_jac):
    Xfuture = f_func(X)
    F = F_jac(X)
    Pfuture = F @ P @ np.transpose(F) + Q
    return Xfuture, Pfuture


def KalmanGain(Pold, R, H_jac, X_pred):
    H = H_jac(X_pred)
    K = Pold @ np.transpose(H) @ np.linalg.inv(H @ Pold @ np.transpose(H) + R)
    return K, H


def consolidatecurrent(Xold, Pold, K, Z, H, h_func):
    X = Xold + K @ (Z - h_func(Xold))
    P = (np.identity(np.shape(H)[1]) - K @ H) @ Pold
    return X, P


# 3-state constant-ACCELERATION model, state X = [position, velocity, accel],
# one bar per step (t = 1):
#   position_{k+1} = position + velocity + 0.5*accel   (x + u t + ½ a t²)
#   velocity_{k+1} = velocity + accel                  (v = u + a t)
#   accel_{k+1}    = accel                             (constant)

_F = np.array([[1.0, 1.0, 0.5],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0]])
_H = np.array([[1.0, 0.0, 0.0]])   # measure position only

class AccelerationKalmanTest(PortfolioStrategy):

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
    )

    F = _F   # constant-acceleration transition (position, velocity, accel)
    H = _H   # measure position only

    def setup(self):
        self.kf = {}       # data -> filter state
        self._chart = {}   # symbol -> [(epoch, est, upper, lower)]
        self.pause = {}
        for d in self.datas:
            self.kf[d] = {'warm': [], 'ready': False,
                          'X': None, 'P': None, 'Xf': None, 'Pf': None,
                          'Q': None, 'R': None}
            self._chart[d._name] = []
            self.pause[d] = False

    @staticmethod
    def f_func(Xf):
        return _F @ Xf
    @staticmethod
    def h_func(X):
        return _H @ X
    @staticmethod
    def F_jac(X):
        return _F
    @staticmethod
    def H_jac(X):
        return _H

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

    def _step(self, st, price):

        R = st['R']
        Xf, Pf = st['Xf'], st['Pf']                  # a-priori, from last bar
        y_pred = (self.h_func(Xf)).item()                     # predicted price for this bar
        innov = price - y_pred
        # update with the actual measurement
        Z = np.array([[price]])
        K, H = KalmanGain(Pf, R, self.H_jac, Xf)
        Xc, Pc = consolidatecurrent(Xf, Pf, K, Z, H, self.h_func)
        S = (H @ Pf @ np.transpose(H) + R).item()    # innovation covariance
        # predict next bar
        st['X'], st['P'] = Xc, Pc
        st['Xf'], st['Pf'] = estimatefuture(Xc, Pc, st['Q'], self.f_func, self.F_jac)
        return innov, S, y_pred

    def on_bar(self, d, price):
        st = self.kf[d]

        # Warmup: collect prices until the filter can be initialized.
        if not st['ready']:
            st['warm'].append(price)
            if len(st['warm']) >= self.params.warmup:
                self._init_filter(st)
            return 0

        innov, S, y_pred = self._step(st, price)
        band = self.params.k * (S ** 0.5)
        self._chart[d._name].append((self.bar_epoch(d), round(y_pred, 8),
                                     round(y_pred + band, 8),
                                     round(y_pred - band, 8)))
        if band <= 0:
            return 0
        

        # innov = price - prediction. FOLLOW (reversion=False): trade in the
        # direction of the surprise (price broke ABOVE the band -> LONG).
        # REVERSION (reversion=True, default): fade it. Same convention as
        # every other strategy in this project — keep it consistent.
        long_sig = innov > band      # price above the upper band
        short_sig = innov < -band    # price below the lower band

        if self.params.reversion:
            long_sig, short_sig = short_sig, long_sig

        pos_size = self.getposition(d).size

        if long_sig:
            if pos_size > 0:                  # already long -> don't re-enter
                return 0
            self.log('%s breakout ABOVE +band (innov %.6f, band %.6f) -> LONG'
                     % (d._name, innov, band))
            
            return 1
        
        if short_sig:
            if pos_size < 0:                  # already short -> don't re-enter
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