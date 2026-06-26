import json
import os
from datetime import datetime, timedelta, timezone

_IST = timezone(timedelta(hours=5, minutes=30))

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
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import AggregationSource, BarAggregation
from nautilus_trader.model.identifiers import InstrumentId

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
        self._indicators: dict[str, dict] = {}       # "{instrument_id}:{spec}" -> indicator instances
        # Closed positions copied out of the cache before it purges them (~1 min),
        # so the dashboard's closed-positions list stays stable. position_id -> record.
        self._closed_positions: dict[str, dict] = {}

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
            positions = []
            prices = {}
            for p in self.cache.positions_open():
                try:
                    up = self.portfolio.unrealized_pnl(p.instrument_id)
                except Exception:
                    up = None
                try:
                    exposure = self.portfolio.net_exposure(p.instrument_id)
                    value = float(exposure) if exposure is not None else None
                except Exception:
                    value = None
                positions.append({
                    "id": str(p.id),
                    "instrument": str(p.instrument_id),
                    "strategy": str(p.strategy_id),
                    "side": p.side.name,
                    "qty": p.quantity.as_double(),
                    "avg_px": p.avg_px_open,
                    "value": value,
                    "realized": p.realized_pnl.as_double() if p.realized_pnl is not None else 0.0,
                    "unrealized": up.as_double() if up is not None else 0.0,
                    "ccy": up.currency.code if up is not None else "",
                    "ts_opened": int(p.ts_opened // 1_000_000) if p.ts_opened else 0,
                })
                tick = self.cache.trade_tick(p.instrument_id)
                if tick is not None:
                    prices[str(p.instrument_id)] = float(tick.price)
            # Capture newly-closed positions into a persistent map (the cache
            # purges closed positions after ~1 min). Keyed by position id so each
            # is recorded once; updates are idempotent.
            for p in self.cache.positions_closed():
                rp = p.realized_pnl
                self._closed_positions[str(p.id)] = {
                    "id": str(p.id),
                    "instrument": str(p.instrument_id),
                    "strategy": str(p.strategy_id),
                    "side": "LONG" if p.entry.name == "BUY" else "SHORT",
                    "qty": p.peak_qty.as_double(),
                    "avg_px_open": p.avg_px_open,
                    "avg_px_close": p.avg_px_close,
                    "realized": rp.as_double() if rp is not None else 0.0,
                    "ccy": rp.currency.code if rp is not None else "",
                    "ts_opened": int(p.ts_opened // 1_000_000) if p.ts_opened else 0,
                    "ts_closed": int(p.ts_closed // 1_000_000) if p.ts_closed else 0,
                }
            # Held for the entire session (no cap) so the dashboard keeps the full
            # position history, matching the now-unpurged exec-engine cache.
            closed_positions = sorted(
                self._closed_positions.values(), key=lambda c: c["ts_closed"], reverse=True,
            )
            # Aggregate PnL from the same position data the per-strategy view uses,
            # so the total equals the sum of the per-strategy PnLs. Realized includes
            # retained closed positions (the portfolio's own realized PnL drops them
            # once the cache purges closed positions).
            pnl: dict[str, dict] = {}
            for rec in positions:
                d = pnl.setdefault(rec["ccy"] or "USDT", {"realized": 0.0, "unrealized": 0.0})
                d["realized"] += rec["realized"]
                d["unrealized"] += rec["unrealized"]
            for c in self._closed_positions.values():
                d = pnl.setdefault(c["ccy"] or "USDT", {"realized": 0.0, "unrealized": 0.0})
                d["realized"] += c["realized"]
            for v in pnl.values():
                v["total"] = v["realized"] + v["unrealized"]
            try:
                names = set(self._redis.smembers(STRATEGY_SET) or [])
            except Exception:
                names = set()
            snap = {
                "ts": int(self.clock.timestamp_ns() // 1_000_000),
                "strategies": sorted(names),
                "positions": positions,
                "closed_positions": closed_positions,
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
            # Sub-minute; Binance has no klines, so let Nautilus download the
            # trades and aggregate them internally into the requested bars.
            int_bt = BarType(bar_type.instrument_id, bar_type.spec, AggregationSource.INTERNAL)
            self.log.info(f"Backfill internal bars {int_bt} [{start_dt} -> {end_dt}]", color=4)
            self.request_aggregated_bars([int_bt], start=start_dt, end=end_dt, update_catalog=False)
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
        self.request_trade_ticks(instrument_id, start=start_dt, end=end_dt, update_catalog=False)

    def on_historical_data(self, data) -> None:
        # Nautilus streams aggregated bars natively to historical.data.bars, which
        # the dashboard now tails continuously — so no republish is needed here.
        # We only feed the indicator overlays (a custom dashboard feature).
        if isinstance(data, Bar):
            self._feed_and_publish_indicator(data)
        elif isinstance(data, list) and data and isinstance(data[0], Bar):
            for bar in data:
                self._feed_and_publish_indicator(bar)


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
