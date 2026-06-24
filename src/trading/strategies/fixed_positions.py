import json
import os
import time
from decimal import Decimal

from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import Strategy

STRATEGY_SET = "dashboard:strategies"
METRICS_PREFIX = "dashboard:metrics:"

_MIN_ORDER_USD = 5
_REBALANCE_COOLDOWN = 5.0


class FixedNotionalConfig(StrategyConfig, frozen=True):
    instrument_ids: tuple[InstrumentId, ...]
    target_usd: Decimal = Decimal("2000")
    rebalance_threshold: float = 0.001


class FixedNotional(Strategy):

    def __init__(self, config: FixedNotionalConfig):
        super().__init__(config)
        self._pending: set[str] = set()
        self._last_rebalance: dict[str, float] = {}
        self._redis = None

    def on_start(self) -> None:
        for iid in self.config.instrument_ids:
            self.subscribe_trade_ticks(iid)

        try:
            import redis as _redis
            self._redis = _redis.Redis.from_url(
                os.getenv("REDIS_URL", "redis://localhost:6379"), decode_responses=True,
            )
            self._redis.sadd(STRATEGY_SET, str(self.id))
            self._publish_metrics()
        except Exception as e:
            self._redis = None
            self.log.warning(f"Dashboard registry unavailable: {e}")

    def on_trade_tick(self, tick: TradeTick) -> None:
        self._maybe_rebalance(tick.instrument_id)

    def _maybe_rebalance(self, iid: InstrumentId) -> None:
        key = str(iid)

        if key in self._pending:
            return

        """
        now = time.monotonic()
        if now - self._last_rebalance.get(key, 0) < _REBALANCE_COOLDOWN:
            return
        """

        tick = self.cache.trade_tick(iid)
        if tick is None:
            return
        price = float(tick.price)
        if price <= 0:
            return

        exposure = self.portfolio.net_exposure(iid)
        current_value = float(exposure) if exposure else 0.0
        target = float(self.config.target_usd)
        deviation = (current_value - target) / target

        if abs(deviation) < self.config.rebalance_threshold:
            return

        diff_usd = target - current_value
        if abs(diff_usd) < _MIN_ORDER_USD:
            return

        instrument = self.cache.instrument(iid)
        if instrument is None:
            return

        raw_qty = abs(diff_usd) / price
        qty = instrument.make_qty(
            Decimal(str(raw_qty)).quantize(
                Decimal(10) ** -instrument.size_precision
            )
        )
        if float(qty) <= 0:
            return

        side = OrderSide.BUY if diff_usd > 0 else OrderSide.SELL
        order = self.order_factory.market(iid, side, qty)
        self._pending.add(key)

        """self._last_rebalance[key] = now"""

        self.log.info(
            f"Rebalance {key}: value={current_value:.2f} target={target:.2f} "
            f"deviation={deviation*100:.1f}% -> {side.name} {qty}"
        )
        self._publish_order_event("submitted", iid, side.name, str(qty))
        self.submit_order(order)

    def on_order_filled(self, event) -> None:
        self._pending.discard(str(event.instrument_id))
        self._publish_order_event("filled", event.instrument_id)
        self._publish_metrics()

    def on_order_rejected(self, event) -> None:
        self._pending.discard(str(event.instrument_id))
        self._publish_order_event("rejected", event.instrument_id)

    def on_order_canceled(self, event) -> None:
        self._pending.discard(str(event.instrument_id))
        self._publish_order_event("canceled", event.instrument_id)

    def _publish_order_event(self, status: str, instrument_id, side: str = "", qty: str = "") -> None:
        if self._redis is None:
            return
        try:
            self._redis.xadd("dashboard:order_events", {
                "strategy": str(self.id),
                "instrument": str(instrument_id),
                "status": status,
                "side": side,
                "qty": qty,
                "ts": str(time.time()),
            }, maxlen=500, approximate=True)
        except Exception:
            pass

    def _publish_metrics(self) -> None:
        if self._redis is None:
            return
        target = float(self.config.target_usd)
        holdings = {}
        for iid in self.config.instrument_ids:
            exposure = self.portfolio.net_exposure(iid)
            holdings[str(iid)] = round(float(exposure) if exposure else 0.0, 2)
        try:
            self._redis.set(
                f"{METRICS_PREFIX}{self.id}",
                json.dumps({
                    "description": (
                        f"Holds ${target:,.0f} notional of each of "
                        f"{len(self.config.instrument_ids)} instruments. "
                        f"Rebalances when drift exceeds "
                        f"{self.config.rebalance_threshold*100:.0f}%."
                    ),
                    "target_usd": target,
                    "holdings": holdings,
                }),
            )
        except Exception:
            pass

    def on_stop(self) -> None:
        self.close_all_positions()
        if self._redis is not None:
            try:
                self._redis.srem(STRATEGY_SET, str(self.id))
                self._redis.delete(f"{METRICS_PREFIX}{self.id}")
            except Exception:
                pass
