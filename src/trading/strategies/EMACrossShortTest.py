import json
import os
import time
from decimal import Decimal
from nautilus_trader.config import StrategyConfig
from nautilus_trader.indicators import ExponentialMovingAverage
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import Strategy

STRATEGY_SET = "dashboard:strategies"     # live registry of running strategies for the dashboard
METRICS_PREFIX = "dashboard:metrics:"     # per-strategy free-form metrics for the dashboard

_MIN_ORDER_USD = 5  # Binance minimum notional per order


class EMACrossSARConfig(StrategyConfig, frozen=True):
    instrument_ids: tuple[InstrumentId, ...]
    trade_usd: Decimal = Decimal("2000")
    bar_spec: str = "10-TICK-LAST"   # 10-tick bars, aggregated internally from trades
    fast_ema_period: int = 5
    slow_ema_period: int = 20


class EMACrossStopReverse(Strategy):
    def __init__(self, config: EMACrossSARConfig):
        super().__init__(config)
        # Per-instrument bar types and EMA pairs.
        self._bar_types: dict[InstrumentId, BarType] = {
            iid: BarType.from_str(f"{iid}-{config.bar_spec}-INTERNAL")
            for iid in config.instrument_ids
        }
        self._fast: dict[InstrumentId, ExponentialMovingAverage] = {
            iid: ExponentialMovingAverage(config.fast_ema_period)
            for iid in config.instrument_ids
        }
        self._slow: dict[InstrumentId, ExponentialMovingAverage] = {
            iid: ExponentialMovingAverage(config.slow_ema_period)
            for iid in config.instrument_ids
        }
        self._redis = None
        self._ema_snapshot: dict[str, dict] = {}
        self._prev_signal: dict[InstrumentId, str] = {}  # "bull" | "bear"
        self._pending_entries: set[InstrumentId] = set()  # submitted but not yet filled
        self._active = False

    def on_start(self):
        # Always register with the dashboard so the strategy is visible
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

        # Only subscribe to bars (and thus start trading) when armed.
        if self._active:
            for iid in self.config.instrument_ids:
                bt = self._bar_types[iid]
                self.register_indicator_for_bars(bt, self._fast[iid])
                self.register_indicator_for_bars(bt, self._slow[iid])
                self.subscribe_bars(bt)
            self.log.info(f"Armed: subscribed to {len(self.config.instrument_ids)} x {self.config.bar_spec} bars")
        else:
            self.log.info("Disarmed on start — idle until armed by controller")

    def _description(self) -> str:
        return (
            f"EMA crossover on {len(self.config.instrument_ids)} instruments using "
            f"{self.config.bar_spec} bars. Buys ${float(self.config.trade_usd):,.0f} notional when the "
            f"{self.config.fast_ema_period}-period EMA crosses above the "
            f"{self.config.slow_ema_period}-period EMA; exits the entire position when it crosses below."
        )

    def publish_metrics(self, metrics: dict) -> None:
        if self._redis is None:
            return
        try:
            self._redis.set(f"{METRICS_PREFIX}{self.id}", json.dumps(metrics))
        except Exception:
            pass

    def _publish_order_event(self, status: str, instrument_id, side: str = "", qty: str = "", price: str = "") -> None:
        if self._redis is None:
            return
        try:
            self._redis.xadd("dashboard:order_events", {
                "strategy": str(self.id),
                "instrument": str(instrument_id),
                "status": status,
                "side": side,
                "qty": qty,
                "price": price,
                "ts": str(time.time()),
            }, maxlen=500, approximate=True)
        except Exception:
            pass

    def on_order_filled(self, event) -> None:
        self._pending_entries.discard(event.instrument_id)
        side = getattr(event, "order_side", None)
        px = getattr(event, "last_px", None)
        qty = getattr(event, "last_qty", None)
        self._publish_order_event(
            "filled", event.instrument_id,
            side=side.name if side is not None else "",
            qty=str(qty) if qty is not None else "",
            price=str(px) if px is not None else "",
        )

    def on_order_rejected(self, event) -> None:
        self._pending_entries.discard(event.instrument_id)
        self._publish_order_event("rejected", event.instrument_id)

    def on_order_canceled(self, event) -> None:
        self._pending_entries.discard(event.instrument_id)
        self._publish_order_event("canceled", event.instrument_id)

    def on_bar(self, bar: Bar):
        if not self._active:
            return
        iid = bar.bar_type.instrument_id
        fast = self._fast.get(iid)
        slow = self._slow.get(iid)
        if fast is None or slow is None or not (fast.initialized and slow.initialized):
            return

        price = float(bar.close)
        if price <= 0:
            return

        # Record this instrument's EMAs and republish the full live snapshot.
        sym = str(iid).split(".")[0]
        self._ema_snapshot[sym] = {
            "fast_ema": round(fast.value, 6),
            "slow_ema": round(slow.value, 6),
            "last_close": round(price, 6),
        }
        self.publish_metrics({
            "description": self._description(),
            "emas": self._ema_snapshot,
        })

        if fast.value > slow.value:
            signal = "bull"
        elif slow.value > fast.value:
            signal = "bear"
        else:
            signal = None

        prev = self._prev_signal.get(iid)
        self._prev_signal[iid] = signal

        if prev is None or signal == prev:  # first bar or no crossover, skip
            return
        
        effective_open = len(self.cache.positions_open(strategy_id=self.id)) + len(self._pending_entries)
        if signal == "bull":
            if self.portfolio.is_flat(iid) or self.portfolio.is_net_short(iid):
                self.close_all_positions(iid)
                if effective_open < 30 and iid not in self._pending_entries:
                    self._pending_entries.add(iid)
                    self._trade(iid, OrderSide.BUY, price)
        elif signal == "bear":
            if self.portfolio.is_flat(iid) or self.portfolio.is_net_long(iid):
                self.close_all_positions(iid)
                if effective_open < 30 and iid not in self._pending_entries:
                    self._pending_entries.add(iid)
                    self._trade(iid, OrderSide.SELL, price)

    def _trade(self, iid: InstrumentId, side: OrderSide, price: float) -> None:
        target_usd = float(self.config.trade_usd)
        if target_usd < _MIN_ORDER_USD:
            return
        instrument = self.cache.instrument(iid)
        if instrument is None:
            return
        raw_qty = target_usd / price
        qty = instrument.make_qty(
            Decimal(str(raw_qty)).quantize(Decimal(10) ** -instrument.size_precision)
        )
        if float(qty) <= 0:
            return
        order = self.order_factory.market(iid, side, qty)
        self.log.info(f"EMA {side.name} {iid}: ${target_usd:,.0f} -> {qty} @ {price}")
        self._publish_order_event("submitted", iid, side.name, str(qty))
        self.submit_order(order)

    def on_stop(self):
        pass
