"""The small slice of the backtrader API the ported strategies touch.

The strategies in this folder were written against backtrader. The engine
underneath is now NautilusTrader (see backend/nt_engine.py). Strategies
still do ``import bt_compat as bt`` and call ``bt.indicators.*`` and
``bt.num2date``, so they port with a one-line change. The ``d`` feeds they
receive are ``strategy_base.Feed`` objects that mirror backtrader's line
interface (``d.close[0]``, ``d.close[-1]``, ``len(d)``,
``d.datetime.datetime(0)``, ``.array``).

Indicators are incremental. Each one registers with its source feed and is
updated once per bar, in creation order, before the strategy's next() runs.
That matches backtrader's precomputed lines bar for bar.
"""
import math
from collections import deque
from datetime import datetime, timezone

NaN = float('nan')


def num2date(v):
    """Feed datetime values are epoch seconds -> naive UTC datetime."""
    return datetime.fromtimestamp(v, timezone.utc).replace(tzinfo=None)


def _src_line(src):
    # a Feed passed directly means "its close line", as in backtrader
    return getattr(src, 'close', None) if hasattr(src, '_name') else src


class _Indicator:
    """Base: a growing ``array`` indexed like a backtrader line."""

    def __init__(self, *srcs):
        self.srcs = [_src_line(s) for s in srcs]
        self.array = []
        self.feed = self.srcs[0].feed
        self.feed._indicators.append(self)

    def __getitem__(self, ago):
        return self.array[len(self.array) - 1 + ago]

    def __len__(self):
        return len(self.array)

    def _update(self):
        raise NotImplementedError


class SimpleMovingAverage(_Indicator):
    def __init__(self, src, period=30):
        super().__init__(src)
        self.period = int(period)
        self._win = deque(maxlen=self.period)
        self._sum = 0.0

    def _update(self):
        x = self.srcs[0][0]
        if len(self._win) == self.period:
            self._sum -= self._win[0]
        self._win.append(x)
        self._sum += x
        self.array.append(self._sum / self.period
                          if len(self._win) == self.period else NaN)


class ExponentialMovingAverage(_Indicator):
    """backtrader EMA: seeded with the SMA of the first ``period`` values,
    then ema += alpha * (x - ema), alpha = 2 / (period + 1)."""

    def __init__(self, src, period=30):
        super().__init__(src)
        self.period = int(period)
        self.alpha = 2.0 / (self.period + 1)
        self._seed = []
        self._ema = None

    def _update(self):
        x = self.srcs[0][0]
        if self._ema is None:
            self._seed.append(x)
            if len(self._seed) < self.period:
                self.array.append(NaN)
                return
            self._ema = sum(self._seed) / self.period
            self._seed = None
        else:
            self._ema += self.alpha * (x - self._ema)
        self.array.append(self._ema)


class StandardDeviation(_Indicator):
    """Population std over ``period`` (backtrader's default: sqrt(E[x^2] - E[x]^2))."""

    def __init__(self, src, period=20):
        super().__init__(src)
        self.period = int(period)
        self._win = deque(maxlen=self.period)

    def _update(self):
        self._win.append(self.srcs[0][0])
        if len(self._win) < self.period:
            self.array.append(NaN)
            return
        n = self.period
        mean = sum(self._win) / n
        var = sum(v * v for v in self._win) / n - mean * mean
        self.array.append(math.sqrt(var) if var > 0 else 0.0)


class CrossOver(_Indicator):
    """+1 when a crosses above b, -1 when it crosses below, else 0. Like
    backtrader, the comparison is against the last NON-zero difference,
    so touching and then continuing still counts as one cross."""

    def __init__(self, a, b):
        super().__init__(a, b)
        self._last_nz = None

    def _update(self):
        a, b = self.srcs[0][0], self.srcs[1][0]
        if a != a or b != b:
            self.array.append(0.0)
            return
        diff = a - b
        out = 0.0
        if self._last_nz is not None:
            if self._last_nz < 0 < diff:
                out = 1.0
            elif self._last_nz > 0 > diff:
                out = -1.0
        if diff != 0:
            self._last_nz = diff
        self.array.append(out)


class indicators:          # noqa: N801  (mirrors the `bt.indicators` namespace)
    SimpleMovingAverage = SimpleMovingAverage
    SMA = SimpleMovingAverage
    ExponentialMovingAverage = ExponentialMovingAverage
    EMA = ExponentialMovingAverage
    StandardDeviation = StandardDeviation
    StdDev = StandardDeviation
    CrossOver = CrossOver
