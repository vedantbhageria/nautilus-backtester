from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os.path
import sys

import bt_compat as bt

from strategy_base import PortfolioStrategy, epoch


class EMACrossShortTest(PortfolioStrategy):
    """EMA crossover, stop-and-reverse, inverted variant.

    Bearish cross (fast below slow) -> LONG; bullish cross -> SHORT.
    """

    # backtrader merges these with the base class params (trade_usd, printlog).
    params = (
        ('fast_ema_period', 15),
        ('slow_ema_period', 30),
    )

    def setup(self):
        self.fast, self.slow, self.cross = {}, {}, {}
        for d in self.datas:
            self.fast[d] = bt.indicators.ExponentialMovingAverage(
                d, period=self.params.fast_ema_period)
            self.slow[d] = bt.indicators.ExponentialMovingAverage(
                d, period=self.params.slow_ema_period)
            # +1 fast crosses above slow (bull), -1 fast crosses below (bear)
            self.cross[d] = bt.indicators.CrossOver(self.fast[d], self.slow[d])

    def on_bar(self, d, price):
        cross = self.cross[d][0]
        if cross < 0:
            self.log('%s BEAR CROSS -> reverse LONG, %.6f' % (d._name, price))
            return 1        # bearish cross -> go LONG (inverted variant)
        if cross > 0:
            self.log('%s BULL CROSS -> reverse SHORT, %.6f' % (d._name, price))
            return -1       # bullish cross -> go SHORT
        return 0

    def build_chart_lines(self):
        fp, sp = self.params.fast_ema_period, self.params.slow_ema_period
        out = {}
        for d in self.datas:
            times = [epoch(bt.num2date(v)) for v in d.datetime.array]
            fast = list(self.fast[d].array)
            slow = list(self.slow[d].array)
            out[d._name] = [
                {'name': 'EMA %d' % fp, 'color': '#58a6ff',
                 'points': [{'time': t, 'value': round(fast[j], 8)}
                            for j, t in enumerate(times)
                            if j < len(fast) and fast[j] == fast[j]]},  # skip NaN
                {'name': 'EMA %d' % sp, 'color': '#ff9800',
                 'points': [{'time': t, 'value': round(slow[j], 8)}
                            for j, t in enumerate(times)
                            if j < len(slow) and slow[j] == slow[j]]},
            ]
        return out

    def stop(self):
        self.log('(fast %d / slow %d) Ending Value %.2f'
                 % (self.params.fast_ema_period, self.params.slow_ema_period,
                    self.broker.getvalue()), doprint=True)
