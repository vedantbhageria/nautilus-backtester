import json
import os
from decimal import Decimal
from nautilus_trader.config import StrategyConfig
from nautilus_trader.indicators import ExponentialMovingAverage
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import Strategy

STRATEGY_SET = "dashboard:strategies"     # live registry of running strategies for the dashboard
METRICS_PREFIX = "dashboard:metrics:"     # per-strategy free-form metrics for the dashboard


class EMACrossConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    trade_size: Decimal
    fast_ema_period: int = 5
    slow_ema_period: int = 20


class EMACross(Strategy):
    def __init__(self, config: EMACrossConfig):
        super().__init__(config)
        self.fast_ema = ExponentialMovingAverage(config.fast_ema_period)
        self.slow_ema = ExponentialMovingAverage(config.slow_ema_period)
        self._redis = None

    def on_start(self):
        self.register_indicator_for_bars(self.config.bar_type, self.fast_ema)
        self.register_indicator_for_bars(self.config.bar_type, self.slow_ema)
        self.subscribe_bars(self.config.bar_type)
        # Register in the dashboard strategy set so the name shows before the
        # first trade (cache.strategy_ids() is empty until an order is placed).
        try:
            import redis
            self._redis = redis.Redis.from_url(
                os.getenv("REDIS_URL", "redis://localhost:6379"), decode_responses=True,
            )
            self._redis.sadd(STRATEGY_SET, str(self.id))
            self.publish_metrics({"description": self._description()})
        except Exception as e:
            self._redis = None
            self.log.warning(f"Strategy registry unavailable: {e}")

    def _description(self) -> str:
        return (
            f"EMA crossover on {self.config.instrument_id}. "
            f"Buys when the {self.config.fast_ema_period}-period EMA crosses above "
            f"the {self.config.slow_ema_period}-period EMA; sells when it crosses below."
        )

    def publish_metrics(self, metrics: dict) -> None:
        if self._redis is None:
            return
        try:
            self._redis.set(f"{METRICS_PREFIX}{self.id}", json.dumps(metrics))
        except Exception:
            pass

    def on_bar(self, bar: Bar):
        if not self.indicators_initialized():
            return
        self.publish_metrics({
            "description": self._description(),
            "fast_ema": self.fast_ema.value,
            "slow_ema": self.slow_ema.value,
            "last_close": float(bar.close),
        })
        if self.fast_ema.value >= self.slow_ema.value:
            if self.portfolio.is_flat(self.config.instrument_id):
                self.buy()
            elif self.portfolio.is_net_short(self.config.instrument_id):
                self.close_all_positions(self.config.instrument_id)
                self.buy()
        elif self.fast_ema.value < self.slow_ema.value:
            if self.portfolio.is_flat(self.config.instrument_id):
                self.sell()
            elif self.portfolio.is_net_long(self.config.instrument_id):
                self.close_all_positions(self.config.instrument_id)
                self.sell()

    def buy(self):
        instrument = self.cache.instrument(self.config.instrument_id)
        order = self.order_factory.market(
            self.config.instrument_id,
            OrderSide.BUY,
            instrument.make_qty(self.config.trade_size),
        )
        self.submit_order(order)

    def sell(self):
        instrument = self.cache.instrument(self.config.instrument_id)
        order = self.order_factory.market(
            self.config.instrument_id,
            OrderSide.SELL,
            instrument.make_qty(self.config.trade_size),
        )
        self.submit_order(order)
    

    def on_stop(self):
        self.close_all_positions(self.config.instrument_id)
        if self._redis is not None:
            try:
                self._redis.srem(STRATEGY_SET, str(self.id))
                self._redis.delete(f"{METRICS_PREFIX}{self.id}")
            except Exception:
                pass