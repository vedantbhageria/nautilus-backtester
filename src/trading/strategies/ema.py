"""Fast/slow EMA pair per instrument: the indicator plumbing both EMA strategies
share. Subclasses only decide what to do with the two values (``on_ema``)."""
from __future__ import annotations

from decimal import Decimal

from nautilus_trader.indicators import ExponentialMovingAverage
from nautilus_trader.model.data import Bar
from nautilus_trader.model.identifiers import InstrumentId

from trading.strategies.base import DashboardStrategy, DashboardStrategyConfig


class EMAConfig(DashboardStrategyConfig, frozen=True):
    trade_usd: Decimal = Decimal("2000")
    fast_ema_period: int = 5
    slow_ema_period: int = 20
    warmup_factor: int = 2            # history requested = slow_ema_period * warmup_factor bars


class EMAStrategy(DashboardStrategy):
    def __init__(self, config: EMAConfig):
        super().__init__(config)
        if config.fast_ema_period >= config.slow_ema_period:
            raise ValueError(f"fast_ema_period ({config.fast_ema_period}) must be shorter "
                             f"than slow_ema_period ({config.slow_ema_period})")
        self._fast = {i: ExponentialMovingAverage(config.fast_ema_period) for i in config.instrument_ids}
        self._slow = {i: ExponentialMovingAverage(config.slow_ema_period) for i in config.instrument_ids}
        self._snapshot: dict[str, dict] = {}

    # -- DashboardStrategy hooks --------------------------------------------------

    def reset_instrument(self, iid: InstrumentId) -> None:
        self._fast[iid].reset()
        self._slow[iid].reset()
        self._snapshot.pop(str(iid).split(".")[0], None)

    def update_indicators(self, iid: InstrumentId, bar: Bar) -> None:
        self._fast[iid].handle_bar(bar)
        self._slow[iid].handle_bar(bar)

    def is_warm(self, iid: InstrumentId) -> bool:
        return self._fast[iid].initialized and self._slow[iid].initialized

    def warmup_bars(self) -> int:
        return self.config.slow_ema_period * self.config.warmup_factor

    def chart_values(self, iid: InstrumentId) -> dict:
        if not self.is_warm(iid):
            return {}
        return {"fast_ema": self._fast[iid].value, "slow_ema": self._slow[iid].value}

    def on_signal_bar(self, iid: InstrumentId, bar: Bar) -> None:
        price = float(bar.close)
        if price <= 0:
            return
        fast, slow = self._fast[iid].value, self._slow[iid].value
        self._snapshot[str(iid).split(".")[0]] = {
            "fast_ema": round(fast, 6), "slow_ema": round(slow, 6), "last_close": round(price, 6),
        }
        self.publish_metrics({"emas": self._snapshot})
        self.on_ema(iid, fast, slow, price)

    def on_ema(self, iid: InstrumentId, fast: float, slow: float, price: float) -> None:
        raise NotImplementedError

    # -- chart hooks used by the backtest runner (see chart_protocol.py) ----------

    @classmethod
    def chart_indicators(cls, config: EMAConfig) -> dict:
        return {"fast_ema": ExponentialMovingAverage(config.fast_ema_period),
                "slow_ema": ExponentialMovingAverage(config.slow_ema_period)}

    @classmethod
    def chart_spec(cls, config: EMAConfig) -> list:
        return [
            {"name": "fast_ema", "color": "#a855f7", "panel": "overlay", "label": f"EMA {config.fast_ema_period}"},
            {"name": "slow_ema", "color": "#22d3ee", "panel": "overlay", "label": f"EMA {config.slow_ema_period}"},
        ]
