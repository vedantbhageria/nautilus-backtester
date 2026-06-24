import json
import os
import time
from datetime import datetime, timedelta, timezone

_TICK_CHUNK_SECS = 3600  # 1 hour per request — Binance aggTrades window limit
_IST = timezone(timedelta(hours=5, minutes=30))
_INTERNAL_SPECS = [("1-SECOND-LAST", 1), ("5-SECOND-LAST", 5), ("15-SECOND-LAST", 15)]
_TICK_QUIET_SECS = 2.0  # flush a buffer once no new historical ticks arrive for this long

from nautilus_trader.common.actor import Actor
from nautilus_trader.config import ActorConfig
from nautilus_trader.indicators import (
    BollingerBands,
    ExponentialMovingAverage,
    IchimokuCloud,
    RelativeStrengthIndex,
    SimpleMovingAverage,
    VolumeWeightedAveragePrice,
)
from nautilus_trader.model.data import Bar, BarType, TradeTick
from nautilus_trader.model.enums import AggregationSource, BarAggregation
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price, Quantity

COMMAND_STREAM = "dashboard:commands"
PORTFOLIO_KEY = "dashboard:portfolio"
STRATEGY_SET = "dashboard:strategies"
INDICATORS_PREFIX = "dashboard:indicators:"


class ControlActorConfig(ActorConfig, frozen=True):
    redis_url: str = "redis://localhost:6379"
    command_stream: str = COMMAND_STREAM
    command_poll_ms: int = 300


class ControlActor(Actor):
    def __init__(self, config: ControlActorConfig) -> None:
        super().__init__(config)
        self._redis = None
        self._cmd_cursor = "$"
        # Per-instrument accumulation for backfill aggregation. Ticks for a
        # request arrive in multiple batches with unpredictable gaps, so we
        # accumulate the whole backfill and re-aggregate the full set once the
        # stream goes quiet — a late flush re-publishes complete bars that
        # supersede any partial bars from an earlier (premature) flush.
        self._tick_buffer: dict[str, list] = {}      # instrument_id -> all ticks this backfill
        self._tick_last_seen: dict[str, float] = {}  # instrument_id -> monotonic time of last tick
        self._tick_flushed: dict[str, int] = {}      # instrument_id -> tick count at last flush
        self._indicators: dict[str, dict] = {}       # "{instrument_id}:{spec}" -> indicator instances

    def on_start(self) -> None:
        try:
            import redis
            self._redis = redis.Redis.from_url(
                os.getenv("REDIS_URL", self.config.redis_url), decode_responses=True,
            )
            last = self._redis.xrevrange(self.config.command_stream, count=1)
            self._cmd_cursor = last[0][0] if last else "0-0"
            self.clock.set_timer(
                name="poll_commands",
                interval=timedelta(milliseconds=self.config.command_poll_ms),
                callback=self._poll_commands,
            )
            self.log.info(f"Command channel on '{self.config.command_stream}'", color=4)
        except Exception as e:
            self._redis = None
            self.log.warning(f"Command channel unavailable: {e}")
        # Periodic flush of debounced historical-tick buffers.
        self.clock.set_timer(
            name="flush_hist_ticks",
            interval=timedelta(milliseconds=500),
            callback=self._flush_quiet_buffers,
        )
        # Periodic portfolio/PnL/positions snapshot for the dashboard.
        if self._redis is not None:
            self.clock.set_timer(
                name="portfolio_snapshot",
                interval=timedelta(seconds=1),
                callback=self._snapshot_portfolio,
            )
        self.log.info("ControlActor started", color=4)

    def _snapshot_portfolio(self, event) -> None:
        if self._redis is None:
            return
        try:
            venues = {p.instrument_id.venue for p in self.cache.positions()}
            pnl: dict[str, dict] = {}
            for venue in venues:
                for ccy, money in (self.portfolio.realized_pnls(venue) or {}).items():
                    pnl.setdefault(ccy.code, {"realized": 0.0, "unrealized": 0.0})["realized"] += money.as_double()
                for ccy, money in (self.portfolio.unrealized_pnls(venue) or {}).items():
                    pnl.setdefault(ccy.code, {"realized": 0.0, "unrealized": 0.0})["unrealized"] += money.as_double()
            for v in pnl.values():
                v["total"] = v["realized"] + v["unrealized"]
            positions = []
            for p in self.cache.positions_open():
                up = self.portfolio.unrealized_pnl(p.instrument_id)
                positions.append({
                    "instrument": str(p.instrument_id),
                    "strategy": str(p.strategy_id),
                    "side": p.side.name,
                    "qty": p.quantity.as_double(),
                    "avg_px": p.avg_px_open,
                    "realized": p.realized_pnl.as_double() if p.realized_pnl is not None else 0.0,
                    "unrealized": up.as_double() if up is not None else 0.0,
                    "ccy": up.currency.code if up is not None else "",
                })
            prices = {}
            for p in self.cache.positions_open():
                tick = self.cache.trade_tick(p.instrument_id)
                if tick is not None:
                    prices[str(p.instrument_id)] = float(tick.price)
            names = {str(s) for s in self.cache.strategy_ids()}
            try:
                names |= set(self._redis.smembers(STRATEGY_SET) or [])
            except Exception:
                pass
            snap = {
                "ts": int(self.clock.timestamp_ns() // 1_000_000),
                "strategies": sorted(names),
                "positions": positions,
                "prices": prices,
                "pnl": pnl,
            }
            self._redis.set(PORTFOLIO_KEY, json.dumps(snap))
        except Exception as e:
            self.log.error(f"Portfolio snapshot failed: {e}")

    def _get_indicators(self, key: str) -> dict:
        if key not in self._indicators:
            self._indicators[key] = {
                "sma":  SimpleMovingAverage(20),
                "ema":  ExponentialMovingAverage(20),
                "rsi":  RelativeStrengthIndex(14),
                "vwap": VolumeWeightedAveragePrice(),
                "bb":   BollingerBands(20, 2.0),
                "ichi": IchimokuCloud(9, 26, 52),
            }
        return self._indicators[key]

    def _feed_and_publish_indicator(self, bar: Bar) -> None:
        if self._redis is None:
            return
        spec = str(bar.bar_type.spec)
        key = f"{bar.bar_type.instrument_id}:{spec}"
        inds = self._get_indicators(key)
        for ind in inds.values():
            try:
                ind.handle_bar(bar)
            except Exception as e:
                self.log.debug(f"Indicator feed error ({key}): {e}")
        fields: dict[str, str] = {"ts": str(bar.ts_event // 1_000_000_000), "tf": spec}
        try:
            if inds["sma"].initialized:  fields["sma"]  = str(round(inds["sma"].value, 6))
            if inds["ema"].initialized:  fields["ema"]  = str(round(inds["ema"].value, 6))
            if inds["rsi"].initialized:  fields["rsi"]  = str(round(inds["rsi"].value, 4))
            if inds["vwap"].initialized: fields["vwap"] = str(round(inds["vwap"].value, 6))
            if inds["bb"].initialized:
                fields["bb_upper"] = str(round(inds["bb"].upper, 6))
                fields["bb_mid"]   = str(round(inds["bb"].middle, 6))
                fields["bb_lower"] = str(round(inds["bb"].lower, 6))
            if inds["ichi"].initialized:
                fields["ichi_tenkan"] = str(round(inds["ichi"].tenkan_sen, 6))
                fields["ichi_kijun"]  = str(round(inds["ichi"].kijun_sen, 6))
                fields["ichi_span_a"] = str(round(inds["ichi"].senkou_span_a, 6))
                fields["ichi_span_b"] = str(round(inds["ichi"].senkou_span_b, 6))
        except Exception as e:
            self.log.debug(f"Indicator read error ({key}): {e}")
        if len(fields) > 2:
            try:
                self._redis.xadd(
                    f"{INDICATORS_PREFIX}{bar.bar_type.instrument_id}",
                    fields, maxlen=50000, approximate=True,
                )
            except Exception as e:
                self.log.debug(f"Indicator xadd error: {e}")

    def on_bar(self, bar: Bar) -> None:
        self._feed_and_publish_indicator(bar)

    def _poll_commands(self, event) -> None:
        if self._redis is None:
            return
        try:
            results = self._redis.xread({self.config.command_stream: self._cmd_cursor}, count=50)
        except Exception as e:
            self.log.error(f"Command poll failed: {e}")
            return
        if not results:
            return
        _, entries = results[0]
        for entry_id, fields in entries:
            self._cmd_cursor = entry_id
            raw = fields.get("json")
            if not raw:
                continue
            try:
                self._on_command(json.loads(raw))
            except Exception as e:
                self.log.error(f"Bad command {raw!r}: {e}")

    def _on_command(self, command: dict) -> None:
        action = command.get("action")
        try:
            if action == "subscribe":
                iid = InstrumentId.from_str(command["instrument_id"])
                self.subscribe_quote_ticks(iid)
                self.subscribe_trade_ticks(iid)
                if command.get("bar_type"):
                    self.subscribe_bars(BarType.from_str(command["bar_type"]), update_catalog=False)
            elif action == "unsubscribe":
                iid = InstrumentId.from_str(command["instrument_id"])
                self.unsubscribe_quote_ticks(iid)
                self.unsubscribe_trade_ticks(iid)
                if command.get("bar_type"):
                    self.unsubscribe_bars(BarType.from_str(command["bar_type"]))
            elif action == "backfill":
                if command.get("bar_type"):
                    self._backfill_bars(
                        BarType.from_str(command["bar_type"]),
                        start=command["start"],
                        end=command.get("end"),
                    )
                else:
                    self._backfill_ticks(
                        InstrumentId.from_str(command["instrument_id"]),
                        start=command["start"],
                        end=command.get("end"),
                    )
            else:
                self.log.warning(f"Unknown command action: {action!r}")
        except Exception as e:
            self.log.error(f"Command failed {command!r}: {e}")
            

    def _backfill_bars(self, bar_type: BarType, start, end=None) -> None:
        start_dt = self._parse_ts(start)
        end_dt = self._parse_ts(end) if end else datetime.now(timezone.utc)
        if start_dt is None:
            self.log.warning(f"Backfill ignored ({bar_type}): bad start={start!r}")
            return
        if bar_type.spec.aggregation in (BarAggregation.SECOND, BarAggregation.MILLISECOND):
            self.log.warning(f"Backfill skipped ({bar_type}): Binance has no sub-minute klines")
            return
        ext_bt = BarType(bar_type.instrument_id, bar_type.spec, AggregationSource.EXTERNAL)
        self.log.info(f"Backfill bars {ext_bt} [{start_dt} -> {end_dt}]", color=4)
        self.request_bars(ext_bt, start=start_dt, end=end_dt, update_catalog=False)

    def _backfill_ticks(self, instrument_id: InstrumentId, start, end=None) -> None:
        start_dt = self._parse_ts(start)
        end_dt = self._parse_ts(end) if end else datetime.now(timezone.utc)
        if start_dt is None:
            self.log.warning(f"Tick backfill ignored ({instrument_id}): bad start={start!r}")
            return
        self.log.info(f"Tick backfill {instrument_id} [{start_dt} -> {end_dt}]", color=4)
        # Fresh accumulation for this backfill session.
        key = str(instrument_id)
        self._tick_buffer.pop(key, None)
        self._tick_last_seen.pop(key, None)
        self._tick_flushed.pop(key, None)
        chunk_ns = int(_TICK_CHUNK_SECS * 1e9)
        cur, n = self._ns(start_dt), 0
        end_ns = self._ns(end_dt)
        while cur < end_ns:
            chunk_end = min(cur + chunk_ns, end_ns)
            self.request_trade_ticks(
                instrument_id, start=self._dt(cur), end=self._dt(chunk_end), update_catalog=False,
            )
            cur = chunk_end
            n += 1
        self.log.info(f"Tick request: {n} chunk(s) of ≤{_TICK_CHUNK_SECS // 3600}h", color=4)

    def on_historical_data(self, data) -> None:
        # Historical TradeTicks (from request_trade_ticks backfill): buffer per
        # instrument; the flush timer aggregates once the stream goes quiet.
        if isinstance(data, TradeTick):
            ticks = [data]
        elif isinstance(data, list) and data and isinstance(data[0], TradeTick):
            ticks = data
        else:
            ticks = None
        if ticks is not None:
            key = str(ticks[0].instrument_id)
            self._tick_buffer.setdefault(key, []).extend(ticks)
            self._tick_last_seen[key] = time.monotonic()
            return
        # Historical Bars (from request_bars / EXTERNAL backfill): feed indicators.
        if isinstance(data, Bar):
            self._feed_and_publish_indicator(data)
        elif isinstance(data, list) and data and isinstance(data[0], Bar):
            for bar in data:
                self._feed_and_publish_indicator(bar)

    def _flush_quiet_buffers(self, event) -> None:
        # Re-aggregate the FULL accumulated buffer (not just new ticks) once a
        # backfill goes quiet. Re-publishing complete bars supersedes any partial
        # bars from an earlier premature flush. The dirty check (count changed)
        # prevents re-publishing the same complete set every tick.
        now = time.monotonic()
        for key, last in list(self._tick_last_seen.items()):
            if now - last < _TICK_QUIET_SECS:
                continue
            ticks = self._tick_buffer.get(key, [])
            if len(ticks) == self._tick_flushed.get(key, 0):
                continue  # nothing new since last flush
            self._tick_flushed[key] = len(ticks)
            if ticks:
                self._aggregate_and_publish(ticks)

    def _aggregate_and_publish(self, ticks: list) -> None:
        instrument_id = ticks[0].instrument_id
        instrument = self.cache.instrument(instrument_id)
        if instrument is None:
            return
        pp, sp = instrument.price_precision, instrument.size_precision
        for spec, tfsec in _INTERNAL_SPECS:
            bt = BarType.from_str(f"{instrument_id}-{spec}-INTERNAL")
            buckets: dict[int, dict] = {}
            for tick in ticks:
                ts_s = tick.ts_event // 1_000_000_000
                bucket_ns = (ts_s // tfsec) * tfsec * 1_000_000_000
                price = float(tick.price)
                size = float(tick.size)
                if bucket_ns not in buckets:
                    buckets[bucket_ns] = {"o": price, "h": price, "l": price, "c": price, "v": size}
                else:
                    b = buckets[bucket_ns]
                    b["c"] = price
                    b["v"] += size
                    if price > b["h"]: b["h"] = price
                    if price < b["l"]: b["l"] = price
            if not buckets:
                continue
            # Forward-fill empty periods with a flat 0-volume bar at the prior
            # close, matching Nautilus' live INTERNAL aggregation (which emits a
            # bar every period regardless of trade activity).
            step_ns = tfsec * 1_000_000_000
            first_ns, last_ns = min(buckets), max(buckets)
            n_pub = 0
            prev_close = buckets[first_ns]["o"]
            ts_ns = first_ns
            while ts_ns <= last_ns:
                b = buckets.get(ts_ns)
                if b is None:
                    b = {"o": prev_close, "h": prev_close, "l": prev_close, "c": prev_close, "v": 0.0}
                prev_close = b["c"]
                bar = Bar(
                    bar_type=bt,
                    open=Price(b["o"], pp),
                    high=Price(b["h"], pp),
                    low=Price(b["l"], pp),
                    close=Price(b["c"], pp),
                    volume=Quantity(b["v"], sp),
                    ts_event=ts_ns,
                    ts_init=ts_ns,
                )
                self._msgbus.publish(topic=f"data.bars.{bt}", msg=bar)
                n_pub += 1
                ts_ns += step_ns
            self.log.info(f"Published {n_pub} {spec} bars ({len(buckets)} with trades) for {instrument_id}", color=4)


    @staticmethod
    def _ns(dt: datetime) -> int:
        return int(dt.timestamp() * 1e9)

    @staticmethod
    def _dt(ns: int) -> datetime:
        return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc)

    @staticmethod
    def _parse_ts(ts) -> datetime | None:
        if ts is None:
            return None
        if isinstance(ts, datetime):
            return ts if ts.tzinfo else ts.replace(tzinfo=_IST)
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(float(ts), tz=timezone.utc)
        try:
            s = str(ts).replace("Z", "+00:00")
            dt = datetime.fromisoformat(s)
            return dt if dt.tzinfo else dt.replace(tzinfo=_IST)
        except ValueError:
            return None
