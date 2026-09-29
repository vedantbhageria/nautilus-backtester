"""The node's one channel to the dashboard.

Publishes (every second): portfolio snapshot, real strategy status, heartbeat,
equity points. Publishes (as they happen): every order event, keyed by client
order id, and every closed position into a persistent ledger. Runs the node-side
consistency checks every 15s. Executes data commands (subscribe / backfill).

Design notes
* Order and position events come from one message-bus subscription for all
  strategies, so the dashboard's order history can't disagree with the engine
  and no strategy can forget to report one (denied orders used to vanish).
* The closed-position ledger is the single source for realized PnL, both this
  session's and all-time. The old code restored the ledger on restart *and*
  added a persisted "overall" base that already contained it, double-counting
  every previous session.
* Unrealized PnL is per position, marked the way Nautilus's Portfolio marks
  (bid for longs, ask for shorts, then last trade, then last bar). The old code
  assigned the instrument-wide figure to each position, multiplying it whenever
  two positions shared an instrument.
* A Redis outage never loses order events or notifications: failed writes wait
  in an outbox and are flushed, in order, on reconnect.
"""
from __future__ import annotations

import json
import os
import time
from collections import deque
from datetime import datetime, timedelta, timezone

import redis
from nautilus_trader.common.actor import Actor
from nautilus_trader.config import ActorConfig
from nautilus_trader.indicators import (
    BollingerBands,
    IchimokuCloud,
    RelativeStrengthIndex,
    SimpleMovingAverage,
    VolumeWeightedAveragePrice,
)
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import AggregationSource, BarAggregation, PositionSide, PriceType
from nautilus_trader.model.identifiers import InstrumentId, Venue

from trading import consistency
from trading import redis_io as K

_IST = timezone(timedelta(hours=5, minutes=30))
_EQUITY_INTERVAL_MS = 5000
_EQUITY_CAP = 100_000           # ~5.8 days of 5s points per mode
_OUTBOX_CAP = 20_000

_ORDER_STATUS = {
    "OrderSubmitted": "submitted", "OrderAccepted": "accepted", "OrderFilled": "filled",
    "OrderCanceled": "canceled", "OrderRejected": "rejected", "OrderDenied": "denied",
    "OrderExpired": "expired", "OrderCancelRejected": "cancel_rejected",
    "OrderModifyRejected": "modify_rejected", "OrderTriggered": "triggered", "OrderUpdated": "updated",
}
_ORDER_FAILURES = {"rejected", "denied", "cancel_rejected", "modify_rejected"}


class ControlActorConfig(ActorConfig, frozen=True):
    command_poll_ms: int = 300
    consistency_interval_s: int = 15
    venue: str = "BINANCE_FUTURES"
    sandbox: bool = True            # sandbox execution restarts flat every session


class ControlActor(Actor):
    def __init__(self, config: ControlActorConfig) -> None:
        super().__init__(config)
        self.strategies: list = []  # DashboardStrategy instances, injected by the node
        self._r = None
        self._down_since: float | None = None
        self._outbox: deque = deque(maxlen=_OUTBOX_CAP)
        self._cmd_cursor = "$"
        self._indicators: dict[str, dict] = {}
        self._ledger: dict[str, dict] = {}       # closed positions, all sessions
        self._ledger_rev = 0
        self._ledger_saved_rev = -1
        self._session_ns = 0
        self._session_ms = 0
        self._start_balance: float | None = None
        self._peak = {"live": K.ACCOUNT_START, "test": K.ACCOUNT_START}
        self._last_equity_ms = 0
        self._check_status: dict[str, str] = {}

    # -- lifecycle -------------------------------------------------------------

    def on_start(self) -> None:
        self._session_ns = self.clock.timestamp_ns()
        self._session_ms = self._session_ns // 1_000_000
        self._r = K.connect()
        try:
            last = self._r.xrevrange(K.DATA_CMDS, count=1)
            self._cmd_cursor = last[0][0] if last else "0-0"
            ledger = K.read_json(self._r, K.CLOSED_POSITIONS, {}) or {}
            self._ledger = {k: v for k, v in ledger.items() if isinstance(v, dict)}
            self._ledger_saved_rev = self._ledger_rev
            for mode in ("live", "test"):
                pk = self._r.get(K.EQUITY_PEAK[mode])
                if pk is not None:
                    self._peak[mode] = max(self._peak[mode], float(pk))
            # The registry is exactly what this node runs: drops ids left by old
            # configs (e.g. "EMACross-001-None") that would never answer.
            pipe = self._r.pipeline(transaction=False)
            pipe.delete(K.STRATEGIES)
            ids = [str(s.id) for s in self.strategies]
            if ids:
                pipe.sadd(K.STRATEGIES, *ids)
            pipe.execute()
        except redis.RedisError as e:
            self.log.error(f"Redis unavailable at start ({e}); will keep retrying")
            self._down_since = time.time()
        self.msgbus.subscribe(topic="events.order.*", handler=self._on_order_event)
        self.msgbus.subscribe(topic="events.position.*", handler=self._on_position_event)
        self.clock.set_timer(name="ControlActor:commands",
                             interval=timedelta(milliseconds=self.config.command_poll_ms),
                             callback=self._poll_commands)
        self.clock.set_timer(name="ControlActor:snapshot", interval=timedelta(seconds=1),
                             callback=self._snapshot)
        self.clock.set_timer(name="ControlActor:checks",
                             interval=timedelta(seconds=self.config.consistency_interval_s),
                             callback=self._run_checks)
        self._notify("info", "Trading node started",
                     f"{len(self.strategies)} strateg(ies) idle; "
                     f"{'sandbox' if self.config.sandbox else 'LIVE VENUE'} execution on {self.config.venue}")

    def on_stop(self) -> None:
        self._notify("warn", "Trading node stopping", "Strategies stop; open positions are left as they are.")
        try:
            self._flush_outbox()
        except redis.RedisError as e:
            self.log.error(f"{len(self._outbox)} held event(s) not delivered at shutdown: {e}")

    # -- redis plumbing ----------------------------------------------------------

    def _redis(self, what: str, fn):
        """Run one Redis operation. Outages are logged once and reported on recovery."""
        if self._r is None:
            return None
        try:
            if self._down_since is not None or self._outbox:
                self._flush_outbox()
            return fn(self._r)
        except redis.RedisError as e:
            if self._down_since is None:
                self._down_since = time.time()
                self.log.error(f"Redis unavailable during {what}: {e}. Holding writes; retrying.")
            return None

    def _flush_outbox(self) -> None:
        pipe = self._r.pipeline(transaction=False)
        batch = list(self._outbox)
        for stream, fields, maxlen in batch:
            pipe.xadd(stream, fields, maxlen=maxlen, approximate=True)
        if batch:
            pipe.execute()                           # raises -> caller counts it as still down
            for _ in batch:
                self._outbox.popleft()
        if self._down_since is not None:
            down = time.time() - self._down_since
            self._down_since = None
            self.log.warning(f"Redis restored after {down:.0f}s; flushed {len(batch)} held event(s)")
            self._r.xadd(K.NOTIFICATIONS, K.notification(
                "warn", "node", f"Redis connection restored after {down:.0f}s",
                f"{len(batch)} order event(s)/notification(s) held during the outage were delivered. "
                "Snapshots resumed. If Redis runs in WSL, keep launch.bat's keep-alive running."),
                maxlen=1000, approximate=True)

    def _xadd(self, stream: str, fields: dict, maxlen: int) -> None:
        """Write that must not be lost: queued if Redis is down."""
        if self._redis(f"xadd {stream}",
                       lambda r: r.xadd(stream, fields, maxlen=maxlen, approximate=True)) is None:
            self._outbox.append((stream, fields, maxlen))

    def _notify(self, level, title, detail="", retry=None, key="", cmd_id="") -> None:
        log = self.log.error if level == "error" else self.log.warning if level == "warn" else self.log.info
        log(f"{title}{': ' + detail if detail else ''}")
        self._xadd(K.NOTIFICATIONS, K.notification(level, "node", title, detail, retry, key, cmd_id), 1000)

    # -- order / position events --------------------------------------------------

    def _on_order_event(self, event) -> None:
        status = _ORDER_STATUS.get(type(event).__name__)
        if status is None:
            return
        order = self.cache.order(event.client_order_id)
        if status == "filled" and order is not None and not order.is_closed:
            status = "partially_filled"
        side = order.side.name if order is not None else ""
        qty = str(order.quantity) if order is not None else ""
        price = str(getattr(event, "last_px", "") or (order.price if order is not None and order.has_price else ""))
        reason = str(getattr(event, "reason", "") or "")
        iid = str(event.instrument_id)
        self._xadd(K.ORDER_EVENTS, {
            "order_id": str(event.client_order_id), "strategy": str(event.strategy_id),
            "instrument": iid, "status": status, "side": side, "qty": qty,
            "filled": str(order.filled_qty) if order is not None else "",
            "price": price, "reason": reason, "ts": f"{time.time():.3f}",
        }, 5000)
        if status in _ORDER_FAILURES:
            sym = iid.split(".")[0]
            self._notify("error", f"Order {status.replace('_', ' ')}: {side} {qty} {sym}",
                         f"{reason or 'no reason given'} ({event.strategy_id}, {event.client_order_id})",
                         key=f"order:{status}:{event.strategy_id}:{sym}")

    def _on_position_event(self, event) -> None:
        if type(event).__name__ != "PositionClosed":
            return
        p = self.cache.position(event.position_id)
        if p is not None:
            self._record_closed(p)

    def _record_closed(self, p) -> bool:
        rp = p.realized_pnl
        rec = {
            "id": str(p.id), "instrument": str(p.instrument_id), "strategy": str(p.strategy_id),
            "side": "LONG" if p.entry.name == "BUY" else "SHORT",
            "qty": p.peak_qty.as_double(),
            "avg_px_open": p.avg_px_open, "avg_px_close": p.avg_px_close,
            "realized": rp.as_double() if rp is not None else 0.0,
            "ccy": rp.currency.code if rp is not None else "USDT",
            "ts_opened": p.ts_opened // 1_000_000, "ts_closed": p.ts_closed // 1_000_000,
            "session": self._session_ms,
        }
        key = rec["id"]
        old = self._ledger.get(key)
        if old is not None and old.get("ts_opened") != rec["ts_opened"]:
            key = f"{key}#{rec['ts_opened']}"      # NETTING reuses position ids
            rec["id"] = key
            old = self._ledger.get(key)
        if old == rec:
            return False
        if old is not None:
            rec["session"] = old.get("session", self._session_ms)
        self._ledger[key] = rec
        self._ledger_rev += 1
        return True

    # -- snapshot ---------------------------------------------------------------

    def _mark(self, p):
        """Price a position the way Nautilus's Portfolio does, plus book/bar fallbacks."""
        iid = p.instrument_id
        px = self.cache.price(iid, PriceType.BID if p.side == PositionSide.LONG else PriceType.ASK) \
            or self.cache.price(iid, PriceType.LAST)
        if px is None:
            book = self.cache.order_book(iid)
            if book is not None:
                px = book.best_bid_price() if p.side == PositionSide.LONG else book.best_ask_price()
        if px is None:
            for s in self.strategies:
                bt = getattr(s, "_bar_types", {}).get(iid)
                bar = self.cache.bar(bt) if bt is not None else None
                if bar is not None:
                    return bar.close
        return px

    def _open_positions(self) -> tuple[list, dict]:
        out, prices = [], {}
        for p in self.cache.positions_open():
            px = self._mark(p)
            upnl = p.unrealized_pnl(px) if px is not None else None
            rp = p.realized_pnl
            out.append({
                "id": str(p.id), "instrument": str(p.instrument_id), "strategy": str(p.strategy_id),
                "side": p.side.name, "qty": p.quantity.as_double(), "avg_px": p.avg_px_open,
                "mark": float(px) if px is not None else None,
                "value": p.quantity.as_double() * float(px) if px is not None else None,
                "realized": rp.as_double() if rp is not None else 0.0,
                "unrealized": upnl.as_double() if upnl is not None else 0.0,
                "ccy": (rp or upnl).currency.code if (rp or upnl) is not None else "USDT",
                "ts_opened": p.ts_opened // 1_000_000,
            })
            if px is not None:
                prices[str(p.instrument_id)] = float(px)
        return out, prices

    def _snapshot(self, event) -> None:
        try:
            self._snapshot_inner()
        except Exception as e:                    # a bug here must not stop the timer
            self.log.exception("Snapshot failed", e)

    def _snapshot_inner(self) -> None:
        now_ms = self.clock.timestamp_ns() // 1_000_000
        positions, prices = self._open_positions()
        ledger = list(self._ledger.values())
        session = [c for c in ledger if c.get("session") == self._session_ms]
        pnl = consistency.pnl_from(positions, session)
        overall = {ccy: v["realized"] for ccy, v in consistency.pnl_from(positions, ledger).items()}
        status = {}
        for s in self.strategies:
            try:
                status[str(s.id)] = s.status()
            except Exception as e:
                status[str(s.id)] = {"state": "ERROR", "armed": False, "error": str(e)}
        self._capture_start_balance()
        snap = {
            "ts": now_ms, "session": self._session_ms,
            "strategies": [str(s.id) for s in self.strategies],
            "positions": positions, "prices": prices, "pnl": pnl, "overall_pnl": overall,
            "closed_rev": self._ledger_rev, "closed_count": len(self._ledger),
        }
        rev = self._ledger_rev
        equity = self._equity_points(now_ms, positions, ledger)

        def write(r):
            pipe = r.pipeline(transaction=False)
            pipe.set(K.PORTFOLIO, json.dumps(snap))
            pipe.set(K.STRATEGY_STATUS, json.dumps(status))
            pipe.set(K.OVERALL_PNL, json.dumps(overall))
            pipe.set(K.HEARTBEAT, json.dumps({"ts": now_ms, "pid": os.getpid(), "session": self._session_ms,
                                              "sandbox": self.config.sandbox, "venue": self.config.venue,
                                              "held_writes": len(self._outbox)}))
            if rev != self._ledger_saved_rev:
                pipe.set(K.CLOSED_POSITIONS, json.dumps(self._ledger))
            for mode, point in equity:
                pipe.rpush(K.EQUITY[mode], json.dumps(point))
                pipe.ltrim(K.EQUITY[mode], -_EQUITY_CAP, -1)
                pipe.set(K.EQUITY_PEAK[mode], self._peak[mode])
            pipe.execute()
            return True

        if self._redis("snapshot", write):
            self._ledger_saved_rev = rev

    def _equity_points(self, now_ms: int, positions: list, ledger: list) -> list:
        if now_ms - self._last_equity_ms < _EQUITY_INTERVAL_MS:
            return []
        modes = self._redis("read modes", lambda r: r.hgetall(K.STRATEGY_MODES))
        if modes is None:
            return []
        self._last_equity_ms = now_ms
        out = []
        for mode in ("live", "test"):
            # untagged strategies count as live, matching the server's filter
            mine = lambda sid: modes.get(sid, "live") == mode  # noqa: E731
            realized = sum(p["realized"] for p in positions if mine(p["strategy"])) + \
                sum(c.get("realized", 0.0) for c in ledger if mine(c.get("strategy")))
            unreal = sum(p["unrealized"] for p in positions if mine(p["strategy"]))
            nav = K.ACCOUNT_START + realized + unreal
            self._peak[mode] = max(self._peak[mode], nav)
            out.append((mode, {"ts": now_ms, "realized": round(realized, 4), "unrealized": round(unreal, 4),
                               "total": round(realized + unreal, 4), "nav": round(nav, 4)}))
        return out

    def _capture_start_balance(self) -> None:
        # start = balance now - realized so far this session, so the account
        # check is exact even if the first fill beat the first snapshot.
        if self._start_balance is not None:
            return
        account = self.portfolio.account(Venue(self.config.venue))
        if account is None:
            return
        bal = account.balance_total(USDT)
        if bal is None:
            return
        realized = sum(p.realized_pnl.as_double() for p in self.cache.positions()
                       if p.ts_opened >= self._session_ns and p.realized_pnl is not None)
        self._start_balance = bal.as_double() - realized

    # -- consistency ------------------------------------------------------------

    def _run_checks(self, event) -> None:
        try:
            rep = consistency.node_checks(
                cache=self.cache, portfolio=self.portfolio, strategies=self.strategies,
                venue=Venue(self.config.venue), currency=USDT, session_start_ns=self._session_ns,
                start_balance=self._start_balance, ledger=self._ledger, sandbox=self.config.sandbox,
                now_ns=self.clock.timestamp_ns())
        except Exception as e:
            self.log.exception("Consistency checks crashed", e)
            return
        # Repair the ledger *after* reporting, so a gap shows up once as a failure
        # instead of being silently patched.
        repaired = sum(self._record_closed(p) for p in self.cache.positions_closed()
                       if p.ts_opened >= self._session_ns)
        if repaired:
            self.log.warning(f"Ledger repaired: {repaired} closed position(s) re-recorded")
        self._redis("write report", lambda r: r.set(K.NODE_CONSISTENCY, json.dumps(rep)))
        for c in rep["checks"]:
            prev = self._check_status.get(c["id"])
            self._check_status[c["id"]] = c["status"]
            if prev == c["status"] or (prev is None and c["status"] in ("pass", "skip")):
                continue
            items = "\n".join(c["items"][:12])
            if c["status"] == "fail":
                self._notify("error", f"Consistency check failed: {c['title']}",
                             f"{c['detail']}\n{items}".strip(), key=f"check:{c['id']}")
            elif c["status"] == "warn":
                self._notify("warn", c["title"], f"{c['detail']}\n{items}".strip(), key=f"check:{c['id']}")
            elif prev in ("fail", "warn"):
                self._notify("success", f"Resolved: {c['title']}", c["detail"], key=f"check:{c['id']}")

    # -- data commands (subscribe / backfill) ---------------------------------------

    def _poll_commands(self, event) -> None:
        results = self._redis("command poll",
                              lambda r: r.xread({K.DATA_CMDS: self._cmd_cursor}, count=50))
        if not results:
            return
        for entry_id, fields in results[0][1]:
            self._cmd_cursor = entry_id
            raw = fields.get("json")
            try:
                command = json.loads(raw) if raw else None
            except ValueError:
                command = None
            if not isinstance(command, dict):
                self._notify("warn", "Malformed data command ignored", repr(raw)[:300])
                continue
            try:
                self._on_command(command)
            except Exception as e:
                self._notify("error", f"Data command '{command.get('action')}' failed",
                             f"{type(e).__name__}: {e}",
                             retry={"mode": "live", "command": command},
                             key=f"datacmd:{command.get('action')}:{command.get('instrument_id') or command.get('bar_type')}")

    def _on_command(self, command: dict) -> None:
        action = command.get("action")
        if action == "subscribe":
            iid = InstrumentId.from_str(command["instrument_id"])
            if self.cache.instrument(iid) is None:
                raise ValueError(f"unknown instrument {iid}")
            # Trades only: quotes would reach the sandbox matcher, whose book is L2.
            self.subscribe_trade_ticks(iid)
            if command.get("bar_type"):
                self.subscribe_bars(BarType.from_str(command["bar_type"]), update_catalog=False)
        elif action == "unsubscribe":
            iid = InstrumentId.from_str(command["instrument_id"])
            self.unsubscribe_trade_ticks(iid)
            if command.get("bar_type"):
                self.unsubscribe_bars(BarType.from_str(command["bar_type"]))
        elif action == "backfill":
            start = self._parse_ts(command.get("start"))
            end = self._parse_ts(command.get("end")) or datetime.now(timezone.utc)
            if start is None:
                raise ValueError(f"bad start time {command.get('start')!r}")
            if start >= end:
                raise ValueError(f"start {start:%Y-%m-%d %H:%M} is not before end {end:%Y-%m-%d %H:%M}")
            if command.get("bar_type"):
                self._backfill_bars(BarType.from_str(command["bar_type"]), start, end)
            else:
                iid = InstrumentId.from_str(command["instrument_id"])
                self.log.info(f"Tick backfill {iid} [{start} -> {end}]")
                self.request_trade_ticks(iid, start=start, end=end, update_catalog=False)
        else:
            raise ValueError(f"unknown action {action!r}")

    def _backfill_bars(self, bar_type: BarType, start: datetime, end: datetime) -> None:
        if bar_type.spec.aggregation in (BarAggregation.SECOND, BarAggregation.MILLISECOND):
            # Binance has no sub-minute klines: Nautilus downloads the trades and
            # aggregates them into the requested bars.
            bt = BarType(bar_type.instrument_id, bar_type.spec, AggregationSource.INTERNAL)
            self.log.info(f"Backfill internal bars {bt} [{start} -> {end}]")
            self.request_aggregated_bars([bt], start=start, end=end, update_catalog=False)
            return
        bt = BarType(bar_type.instrument_id, bar_type.spec, AggregationSource.EXTERNAL)
        self.log.info(f"Backfill bars {bt} [{start} -> {end}]")
        self.request_bars(bt, start=start, end=end, update_catalog=False)

    @staticmethod
    def _parse_ts(ts) -> datetime | None:
        if ts is None or ts == "":
            return None
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(float(ts), tz=timezone.utc)
        try:
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        except ValueError:
            return None
        return dt if dt.tzinfo else dt.replace(tzinfo=_IST)

    # -- dashboard indicators (SMA/RSI/VWAP/BB/Ichimoku on charted bars) -------------

    def on_bar(self, bar: Bar) -> None:
        self._publish_indicators(bar)

    def on_historical_data(self, data) -> None:
        if isinstance(data, Bar):
            self._publish_indicators(data)

    def _publish_indicators(self, bar: Bar) -> None:
        spec = str(bar.bar_type.spec)
        key = f"{bar.bar_type.instrument_id}:{spec}"
        inds = self._indicators.get(key)
        if inds is None:
            inds = self._indicators[key] = {
                "sma": SimpleMovingAverage(20), "rsi": RelativeStrengthIndex(14),
                "vwap": VolumeWeightedAveragePrice(), "bb": BollingerBands(20, 2.0),
                "ichi": IchimokuCloud(9, 26, 52),
            }
        for ind in inds.values():
            ind.handle_bar(bar)
        f = {"ts": str(bar.ts_event // 1_000_000_000), "tf": spec}
        if inds["sma"].initialized:
            f["sma"] = str(round(inds["sma"].value, 6))
        if inds["rsi"].initialized:
            f["rsi"] = str(round(inds["rsi"].value, 4))
        if inds["vwap"].initialized:
            f["vwap"] = str(round(inds["vwap"].value, 6))
        if inds["bb"].initialized:
            f.update(bb_upper=str(round(inds["bb"].upper, 6)), bb_mid=str(round(inds["bb"].middle, 6)),
                     bb_lower=str(round(inds["bb"].lower, 6)))
        if inds["ichi"].initialized:
            i = inds["ichi"]
            f.update(ichi_tenkan=str(round(i.tenkan_sen, 6)), ichi_kijun=str(round(i.kijun_sen, 6)),
                     ichi_span_a=str(round(i.senkou_span_a, 6)), ichi_span_b=str(round(i.senkou_span_b, 6)))
        if len(f) > 2:   # chart points are replaceable: drop, don't queue, during an outage
            self._redis("indicator", lambda r: r.xadd(f"{K.INDICATORS}{bar.bar_type.instrument_id}", f,
                                                      maxlen=50000, approximate=True))
