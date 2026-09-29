from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os.path
import sys

import bt_compat as bt
import numpy as np

from strategy_base import PortfolioStrategy


def estimatefuture(X, P, F, Q):
    Xfuture = F @ X
    Pfuture = F @ P @ np.transpose(F) + Q
    return Xfuture, Pfuture


def KalmanGain(Pold, R, H):
    K = Pold @ np.transpose(H) @ np.linalg.inv(H @ Pold @ np.transpose(H) + R)
    return K


def consolidatecurrent(Xold, Pold, K, Z, H):
    X = Xold + K @ (Z - H @ Xold)
    P = (np.identity(np.shape(H)[1]) - K @ H) @ Pold
    return X, P


class KalmanTest(PortfolioStrategy):

    params = (
        ('k', 2.0),            # threshold in std devs of the innovation
        ('a', 7.0),             # stop loss this much above value
        ('warmup', 30),        # bars used to initialize each filter
        ('q_level', 0.2e-3),     # process noise on level (relative to est. R)
        ('q_vel', 0.2e-6),       # process noise on slope (relative to est. R)
        ('reversion', True),   # True: fade the deviation (default — matches
                                # every other strategy's convention); False:
                                # follow it (breakout)
    )

    F = np.array([[1.0, 1.0], [0.0, 1.0]])   # constant-velocity transition
    H = np.array([[1.0, 0.0]])               # measure the level only

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

    def _init_filter(self, st):
        buf = np.array(st['warm'], dtype=float)
        idx = np.arange(len(buf))
        m, b = np.polyfit(idx, buf, 1)              # slope per bar, intercept
        resid = buf - (m * idx + b)
        R = float(np.var(resid))
        R = max(R, (buf[-1] * 1e-6) ** 2, 1e-12)    # floor so S is never ~0
        st['R'] = np.array([[R]])
        st['Q'] = np.array([[R * self.params.q_level, 0.0],
                            [0.0, R * self.params.q_vel]])
        X0 = np.array([[buf[-1]], [m]])
        P0 = np.array([[R, 0.0], [0.0, R]])
        st['X'], st['P'] = X0, P0
        st['Xf'], st['Pf'] = estimatefuture(X0, P0, self.F, st['Q'])
        st['ready'] = True

    def _step(self, st, price):
        """Advance one bar. Returns (innovation, S, predicted_price)."""
        H, R = self.H, st['R']
        Xf, Pf = st['Xf'], st['Pf']                  # a-priori, from last bar
        y_pred = (H @ Xf).item()                     # predicted price for this bar
        S = (H @ Pf @ np.transpose(H) + R).item()    # innovation covariance
        innov = price - y_pred
        # update with the actual measurement
        Z = np.array([[price]])
        K = KalmanGain(Pf, R, H)
        Xc, Pc = consolidatecurrent(Xf, Pf, K, Z, H)
        # predict next bar
        st['X'], st['P'] = Xc, Pc
        st['Xf'], st['Pf'] = estimatefuture(Xc, Pc, self.F, st['Q'])
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
        self._diag_record(d, price, y_pred, innov, S, band, st)
        if band <= 0:
            return 0
        

        """if innov < band/4 and innov > -band/4:
            exit_sig = True
        else:
            exit_sig = False
            long_sig = innov < -band     # price below the lower band
            short_sig = innov > band     # price above the upper band"""
        

        # innov = price - prediction. FOLLOW (reversion=False): trade in the
        # direction of the surprise (price broke ABOVE the band -> LONG).
        # REVERSION (reversion=True, default): fade it (price above the band
        # -> SHORT, betting it comes back). Same convention as every other
        # strategy in this project — keep it consistent when editing.
        long_sig = innov > band      # price above the upper band
        short_sig = innov < -band    # price below the lower band

        if self.params.reversion:
            long_sig, short_sig = short_sig, long_sig

        pos_size = self.getposition(d).size
        
        """if self.pause[d]:
            if innov < band * 5/3 and innov > -band * 5/3:
                self.pause[d] == False"""
            
        if long_sig:
            if pos_size > 0:                  # already long -> don't re-enter
                """if innov < -band * self.params.a/self.params.k:
                    self.pause[d] = True
                    return (1,0)"""
                return 0
            self.log('%s breakout ABOVE +band (innov %.6f, band %.6f) -> LONG'
                     % (d._name, innov, band))
            
            return 1
        if short_sig:
            if pos_size < 0:                  # already short -> don't re-enter
                """if innov > band * self.params.a/self.params.k:
                    self.pause[d] = True
                    return (1,0)"""
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
