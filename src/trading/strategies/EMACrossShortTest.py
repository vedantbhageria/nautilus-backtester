import json
import os
import time
from datetime import datetime, timedelta, timezone
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
_UNIT_SECONDS = {"SECOND": 1, "MINUTE": 60, "HOUR": 3600, "DAY": 86400, "WEEK": 604800}


def _spec_interval_seconds(bar_spec: str) -> int:
    parts = bar_spec.split("-")
    try:
        return int(parts[0]) * _UNIT_SECONDS.get(parts[1].upper(), 60)
    except (ValueError, IndexError):
        return 60


def _bar_source_for_spec(bar_spec: str) -> str:
    """Choose the aggregation source from the bar spec, matching the dashboard.

    Minute+ time bars use official Binance klines (EXTERNAL) — these are also what
    the dashboard's 1m/5m/15m/1h views read, so the strategy and chart share one
    Redis stream. Sub-minute and count-based bars have no klines, so Nautilus
    aggregates them from trades (INTERNAL).
    """
    parts = bar_spec.split("-")
    unit = parts[1].upper() if len(parts) >= 2 else ""
    if unit in ("MINUTE", "HOUR", "DAY", "WEEK"):
        return "EXTERNAL"
    return "INTERNAL"


class EMACrossSARConfig(StrategyConfig, frozen=True):
    instrument_ids: tuple[InstrumentId, ...]
    trade_usd: Decimal = Decimal("2000")
    bar_spec: str = "10-TICK-LAST"   # 10-tick bars, aggregated internally from trades
    fast_ema_period: int = 5
    slow_ema_period: int = 20
    backtest: bool = False   # run inside a BacktestEngine (no live Redis/quotes/warmup)


class EMACrossStopReverse(Strategy):
    def __init__(self, config: EMACrossSARConfig):
        super().__init__(config)
        # Per-instrument bar types and EMA pairs.
        _src = _bar_source_for_spec(config.bar_spec)
        self._bar_types: dict[InstrumentId, BarType] = {
            iid: BarType.from_str(f"{iid}-{config.bar_spec}-{_src}")
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
        self._open_count: int = 0   # open + pending positions; authoritative cap counter
        self._active = False

    def on_start(self):
        # Backtest mode: bars are supplied by the engine; just register indicators
        # and subscribe. No Redis, no quote subscription, no warmup request — the
        # 4 days of historical bars warm the EMAs naturally, and the runner extracts
        # positions/EMA afterwards (so no per-bar Redis writes during the run).
        if getattr(self.config, "backtest", False):
            self._active = True
            self._redis = None
            for iid in self.config.instrument_ids:
                bt = self._bar_types[iid]
                self.register_indicator_for_bars(bt, self._fast[iid])
                self.register_indicator_for_bars(bt, self._slow[iid])
                self.subscribe_bars(bt)
            self.log.info(f"Backtest armed: {len(self.config.instrument_ids)} instruments")
            return

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

        if self._active:
            for iid in self.config.instrument_ids:
                bt = self._bar_types[iid]
                self.register_indicator_for_bars(bt, self._fast[iid])
                self.register_indicator_for_bars(bt, self._slow[iid])
                self.subscribe_bars(bt)
                # The sandbox execution client fills orders by processing live
                # quote/trade ticks for the venue. EXTERNAL (kline) bars don't
                # produce ticks, so without this the sandbox has no market to fill
                # against and rejects orders. Quote ticks keep its L1 book populated.
                self.subscribe_quote_ticks(iid)
            self._reenter_handoff_positions()
            self._warmup_emas()
            self.log.info(f"Armed: subscribed to {len(self.config.instrument_ids)} x {self.config.bar_spec} bars + quotes")
        else:
            self.log.info("Disarmed on start — idle until armed by controller")

    def _reenter_handoff_positions(self) -> None:
        if self._redis is None:
            return
        try:
            raw = self._redis.get("dashboard:backtest:handoff_positions")
            if not raw:
                return
            handoff = json.loads(raw)
            self._redis.delete("dashboard:backtest:handoff_positions")
        except Exception as e:
            self.log.warning(f"Could not read handoff positions: {e}")
            return

        iid_map = {str(iid): iid for iid in self.config.instrument_ids}
        entered = 0
        for p in handoff:
            iid = iid_map.get(p.get("instrument"))
            if iid is None:
                continue
            if self._open_count >= 50 or iid in self._pending_entries:
                continue
            side_str = p.get("side", "")
            if side_str in ("LONG", "BUY"):
                order_side = OrderSide.BUY
            elif side_str in ("SHORT", "SELL"):
                order_side = OrderSide.SELL
            else:
                continue
            self._pending_entries.add(iid)
            self._open_count += 1
            self._trade(iid, order_side, p.get("avg_px", 0) or p.get("avg_px_open", 0))
            entered += 1

        if entered:
            self.log.info(f"Handoff: re-entered {entered} positions from backtest")

    def _warmup_emas(self) -> None:
        # Pre-initialize the EMAs from historical bars so the strategy can signal
        # immediately. Registered indicators are fed automatically via handle_bar
        # (historical=True); on_bar is NOT called for historical bars so there are
        # no warmup trades. Only safe for EXTERNAL klines.
        #
        # We always use slow_ema * 2 bars regardless of any user-supplied start
        # date — that date is only used for chart visualization (fetched separately
        # in binance_data.py). Keeping warmup short avoids Binance REST rate-limit
        # bans when the strategy has many instruments.
        if _bar_source_for_spec(self.config.bar_spec) != "EXTERNAL":
            self.log.info("EMA warmup skipped (INTERNAL bars aggregate live)")
            return
        warmup_n = self.config.slow_ema_period * 2
        secs = warmup_n * _spec_interval_seconds(self.config.bar_spec)
        start = datetime.now(timezone.utc) - timedelta(seconds=secs)
        self.log.info(f"EMA warmup: requesting ~{warmup_n} bars × {len(self.config.instrument_ids)} instruments")
        for iid in self.config.instrument_ids:
            self.request_bars(self._bar_types[iid], start=start)

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
        was_pending = event.instrument_id in self._pending_entries
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
        # If this was a closing fill (not an opening fill we submitted), decrement count.
        # Opening fills: was_pending=True (we added iid to pending before submitting).
        # Closing fills (from close_all_positions): was_pending=False.
        if not was_pending:
            self._open_count = max(0, self._open_count - 1)

    def on_order_rejected(self, event) -> None:
        if event.instrument_id in self._pending_entries:
            self._pending_entries.discard(event.instrument_id)
            self._open_count = max(0, self._open_count - 1)
        reason = getattr(event, "reason", "")
        self.log.warning(f"Order REJECTED {event.instrument_id}: {reason}")
        self._publish_order_event("rejected", event.instrument_id, price=str(reason))

    def on_order_canceled(self, event) -> None:
        if event.instrument_id in self._pending_entries:
            self._pending_entries.discard(event.instrument_id)
            self._open_count = max(0, self._open_count - 1)
        self._publish_order_event("canceled", event.instrument_id)

    def _publish_indicator(self, bar: Bar, fast_v: float, slow_v: float) -> None:
        # Push this strategy's (already warmed-up) EMAs into the dashboard indicator
        # stream so the chart draws them directly — no cold-start recompute in the
        # ControlActor. Same stream/keys the ControlActor uses for its other
        # indicators; the dashboard merges fields per timestamp.
        if self._redis is None:
            return
        try:
            self._redis.xadd(
                f"dashboard:indicators:{bar.bar_type.instrument_id}",
                {
                    "ts": str(bar.ts_event // 1_000_000_000),
                    "tf": str(bar.bar_type.spec),
                    "fast_ema": str(round(fast_v, 6)),
                    "slow_ema": str(round(slow_v, 6)),
                },
                maxlen=50000, approximate=True,
            )
        except Exception:
            pass

    def on_historical_data(self, data) -> None:
        # Warmup klines feed the registered EMAs (handle_bar historical=True) before
        # this runs, so publish their values too — gives the chart an EMA line over
        # the warmup history, not just from the first live bar.
        bars = data if isinstance(data, list) else [data]
        for bar in bars:
            if not isinstance(bar, Bar):
                continue
            fast = self._fast.get(bar.bar_type.instrument_id)
            slow = self._slow.get(bar.bar_type.instrument_id)
            if fast and slow and fast.initialized and slow.initialized:
                self._publish_indicator(bar, fast.value, slow.value)

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
        self._publish_indicator(bar, fast.value, slow.value)

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
        
        if signal == "bull":
            if self.portfolio.is_flat(iid) or self.portfolio.is_net_short(iid):
                self.close_all_positions(iid)
                if self._open_count < 50 and iid not in self._pending_entries:
                    self._pending_entries.add(iid)
                    self._open_count += 1
                    self._trade(iid, OrderSide.BUY, price)
        elif signal == "bear":
            if self.portfolio.is_flat(iid) or self.portfolio.is_net_long(iid):
                self.close_all_positions(iid)
                if self._open_count < 50 and iid not in self._pending_entries:
                    self._pending_entries.add(iid)
                    self._open_count += 1
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
