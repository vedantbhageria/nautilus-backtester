"""Plumbing shared by every strategy the dashboard drives.

Lifecycle
---------
The node starts every strategy at boot and they stay RUNNING for the life of the
node. ``armed`` is the trading switch:

    arm()        subscribe bars + the execution book, warm up, start trading
    disarm()     stop entering, cancel working orders, drop idle data feeds
    close_all()  disarm + Nautilus ``market_exit`` (cancel, close, retry until flat)

Keeping the component RUNNING means ``market_exit`` (which requires RUNNING)
always works, and there is no stop/reset/start state machine to get wrong. The
old controller stopped strategies at boot and restarted them with reset+start,
which raised on any double-click and left EMAs stale across a pause.

Bars
----
Live bars and warm-up history reach the subclass through one path
(``_route_bar``), strictly in time order. Live bars that arrive while an
instrument's history request is outstanding are buffered and replayed after it;
anything at or before the last fed timestamp is skipped. Before this, a history
response landing after a live bar fed the EMAs out of order, and a failed
request (Binance -1003 rate limit) left the instrument cold forever without a
word. Requests are now paced, retried, and reported.

Positions
---------
Every position/order query is scoped to this strategy. ``portfolio.is_flat``
is portfolio-wide, so with two strategies on the same instrument one would see
the other's position and never trade.

Order and position events reach the dashboard through the ControlActor's single
message-bus subscription, so strategies don't report them (and can't disagree).
"""
from __future__ import annotations

import json
from collections import deque
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import msgspec
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import BookType, OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import Strategy

from trading import redis_io

MIN_ORDER_USD = 5.0                 # Binance futures minimum notional
WARMUP_SEND_MS = 200                # pace history requests at 5/s (Binance weight budget)
WARMUP_CHECK_S = 20                 # re-request history that never answered
WARMUP_MAX_ATTEMPTS = 3
_UNIT_SECONDS = {"SECOND": 1, "MINUTE": 60, "HOUR": 3600, "DAY": 86400, "WEEK": 604800}


def spec_seconds(bar_spec: str) -> int | None:
    """Interval of a time bar spec ("1-MINUTE-LAST" -> 60); None for tick/volume bars."""
    parts = bar_spec.split("-")
    try:
        unit = _UNIT_SECONDS.get(parts[1].upper())
        return int(parts[0]) * unit if unit else None
    except (ValueError, IndexError):
        return None


def bar_source(bar_spec: str) -> str:
    """Minute+ bars are Binance klines (EXTERNAL, shared with the dashboard's
    1m+ charts and warm-able over REST); anything shorter is aggregated from
    trades inside Nautilus (INTERNAL) and warms live."""
    unit = bar_spec.split("-")[1].upper() if bar_spec.count("-") >= 2 else ""
    return "EXTERNAL" if unit in ("MINUTE", "HOUR", "DAY", "WEEK") else "INTERNAL"


class DashboardStrategyConfig(StrategyConfig, frozen=True):
    instrument_ids: tuple[InstrumentId, ...] = ()
    bar_spec: str = "1-MINUTE-LAST"
    backtest: bool = False            # inside a BacktestEngine: no Redis, no book, armed at start
    book_depth: int = 20              # L2 levels fed to the sandbox matcher
    # A real venue can take longer than the 10s Nautilus default to flatten.
    market_exit_interval_ms: int = 500
    market_exit_max_attempts: int = 120


class DashboardStrategy(Strategy):
    """Subclasses implement:

        reset_instrument(iid)          forget indicator/signal state for one instrument
        update_indicators(iid, bar)    feed one bar (live or history, in time order)
        is_warm(iid) -> bool           indicators ready to signal
        warmup_bars() -> int           history length to request (0 = warm live)
        on_signal_bar(iid, bar)        trading logic; only called armed + warm + live
        description() -> str
        chart_values(iid) -> dict      optional: series drawn on the dashboard chart
    """

    def __init__(self, config: DashboardStrategyConfig):
        super().__init__(config)
        src = bar_source(config.bar_spec)
        self._bar_types: dict[InstrumentId, BarType] = {
            iid: BarType.from_str(f"{iid}-{config.bar_spec}-{src}") for iid in config.instrument_ids
        }
        self.armed = False
        self.after_close_all = None          # callable set by the node (CSV export)
        self._r = None
        self._tradeable: list[InstrumentId] = []
        self._missing: list[InstrumentId] = []
        self._bar_subs: set[InstrumentId] = set()
        self._book_subs: set[InstrumentId] = set()
        self._fed_until: dict[InstrumentId, int] = {}
        self._last_live_ns: dict[InstrumentId, int] = {}
        self._armed_at_ns = 0
        # warm-up state
        self._warming: set[InstrumentId] = set()
        self._warm_queue: deque[InstrumentId] = deque()
        self._warm_attempts: dict[InstrumentId, int] = {}
        self._warm_sent_ns: dict[InstrumentId, int] = {}
        self._warm_failed: list[InstrumentId] = []
        self._warm_requests: dict = {}       # request id -> instrument
        self._buffer: dict[InstrumentId, list[Bar]] = {}
        self._close_after_exit = False
        self._warned: set[str] = set()

    # -- subclass hooks (defaults) ----------------------------------------------

    def reset_instrument(self, iid: InstrumentId) -> None: ...
    def update_indicators(self, iid: InstrumentId, bar: Bar) -> None: ...
    def is_warm(self, iid: InstrumentId) -> bool: return True
    def warmup_bars(self) -> int: return 0
    def on_signal_bar(self, iid: InstrumentId, bar: Bar) -> None: ...
    def description(self) -> str: return type(self).__name__
    def chart_values(self, iid: InstrumentId) -> dict: return {}

    @classmethod
    def backtest_config(cls, cfg: DashboardStrategyConfig) -> DashboardStrategyConfig:
        """Config for running this strategy inside the BacktestEngine on 1m klines."""
        name = (cfg.strategy_id or cls.__name__).replace("-", "")
        return msgspec.structs.replace(cfg, bar_spec="1-MINUTE-LAST", backtest=True,
                                       strategy_id=f"{name}BT", order_id_tag="BT")

    # -- lifecycle -------------------------------------------------------------

    def on_start(self) -> None:
        if self.config.backtest:
            self._tradeable = list(self.config.instrument_ids)
            for iid in self._tradeable:
                self.subscribe_bars(self._bar_types[iid])
            self.armed = True
            return
        self._r = redis_io.connect()
        try:
            self._r.sadd(redis_io.STRATEGIES, str(self.id))
            self.publish_metrics({})
        except Exception as e:
            self.log.warning(f"Dashboard registry unavailable: {e}")
        self.log.info("Idle until armed from the dashboard")

    def on_stop(self) -> None:
        # Node shutdown. Leave positions alone (a restart must not flatten a book),
        # but disarm first: _cancel_warmup replays buffered bars.
        self.armed = False
        self._cancel_warmup()

    def on_reset(self) -> None:
        for iid in self.config.instrument_ids:
            self.reset_instrument(iid)
        self._fed_until.clear()
        self._buffer.clear()

    # -- dashboard controls (always called on the node's event loop) -------------

    def arm(self) -> str:
        if not self.is_running:
            raise RuntimeError(f"strategy is {self.state.name}, not RUNNING")
        if self.armed:
            return "already running"
        if self.is_exiting():
            raise RuntimeError("a close-all is still in progress; wait for it to finish")
        self._tradeable = [i for i in self.config.instrument_ids if self.cache.instrument(i) is not None]
        self._missing = [i for i in self.config.instrument_ids if self.cache.instrument(i) is None]
        if not self._tradeable:
            raise RuntimeError("none of the configured instruments exist on the venue")
        for iid in self._tradeable:
            self.reset_instrument(iid)
        self._fed_until.clear()
        self._buffer.clear()
        self._last_live_ns.clear()
        self._warned.clear()
        self._armed_at_ns = self.clock.timestamp_ns()
        self._ensure_feeds(self._tradeable, bars=True)
        self.armed = True
        self._start_warmup(self._tradeable)
        msg = f"trading {len(self._tradeable)} instruments on {self.config.bar_spec} bars"
        if self._missing:
            self.notify("warn", f"{len(self._missing)} configured instruments don't exist on the venue",
                        ", ".join(str(i) for i in self._missing), key=f"missing:{self.id}")
        return msg

    def disarm(self) -> str:
        was = self.armed
        self.armed = False
        self._cancel_warmup()
        for iid in {o.instrument_id for o in self.working_orders()}:
            self.cancel_all_orders(iid)
        keep = self.instruments_with_exposure()
        self._drop_feeds(keep)
        held = f"; {len(keep)} open position(s) kept (unmanaged until resumed)" if keep else ""
        return ("paused" if was else "already paused") + held

    def close_all(self) -> str:
        if not self.is_running:
            raise RuntimeError(f"strategy is {self.state.name}, not RUNNING")
        if self.is_exiting():
            return "close-all already in progress"
        self.armed = False
        self._cancel_warmup()
        targets = self.instruments_with_exposure() | {o.instrument_id for o in self.working_orders()}
        if not targets:
            self._drop_feeds(set())
            self._after_exit(0)
            return "nothing to close"
        # The sandbox matcher fills against the L2 book; a paused strategy may
        # have dropped it, so make sure every instrument we need to exit has one.
        self._ensure_feeds(targets, bars=False)
        self._close_after_exit = True
        self.market_exit()
        return f"closing {len(self.cache.positions_open(strategy_id=self.id))} position(s)"

    def post_market_exit(self) -> None:
        if not self._close_after_exit:
            return
        self._close_after_exit = False
        left = self.cache.positions_open(strategy_id=self.id)
        self._drop_feeds({p.instrument_id for p in left})
        self._after_exit(len(left))

    def _after_exit(self, left: int) -> None:
        if left:
            self.notify("error", f"Close-all finished with {left} position(s) still open",
                        "The venue did not fill every close order within the retry window. "
                        "Retry, or close them manually.",
                        retry=redis_io.strategy_retry(str(self.id), "close_strategy"),
                        key=f"close:{self.id}")
        else:
            self.notify("success", "All positions closed", key=f"close:{self.id}")
        if self.after_close_all is not None:
            try:
                self.after_close_all()
            except Exception as e:
                self.notify("error", "Post close-all export failed", str(e))

    def rewarm(self) -> str:
        """Re-request history for every instrument that isn't warm (after a failure)."""
        if not self.armed:
            raise RuntimeError("strategy is not running")
        cold = [i for i in self._tradeable if not self.is_warm(i)]
        if not cold:
            return "every instrument is already warm"
        for iid in cold:
            self.reset_instrument(iid)
            self._fed_until.pop(iid, None)
        self._start_warmup(cold)
        return f"re-warming {len(cold)} instrument(s)"

    # -- bars ------------------------------------------------------------------

    def on_bar(self, bar: Bar) -> None:
        self._route_bar(bar, live=True)

    def on_historical_data(self, data) -> None:
        if isinstance(data, Bar):
            self._route_bar(data, live=False)

    def _route_bar(self, bar: Bar, live: bool) -> None:
        iid = bar.bar_type.instrument_id
        if iid not in self._bar_types:
            return
        if live and iid in self._warming:
            buf = self._buffer.setdefault(iid, [])
            buf.append(bar)
            if len(buf) > 5000:
                del buf[:-5000]
            return
        if bar.ts_event <= self._fed_until.get(iid, 0):
            return                                   # already fed (history/live overlap)
        self._fed_until[iid] = bar.ts_event
        self.update_indicators(iid, bar)
        if not live:
            return
        self._last_live_ns[iid] = self.clock.timestamp_ns()
        vals = self.chart_values(iid)
        if vals:
            self._publish_indicator(bar, vals)
        if self.armed and self.is_warm(iid) and not self.is_exiting():
            self.on_signal_bar(iid, bar)

    # -- warm-up ---------------------------------------------------------------

    def _start_warmup(self, iids) -> None:
        n = self.warmup_bars()
        if self.config.backtest or n <= 0 or bar_source(self.config.bar_spec) != "EXTERNAL":
            return
        for iid in iids:
            self._warming.add(iid)
            self._warm_attempts[iid] = 0
            self._warm_queue.append(iid)
        self._warm_failed = []
        self._set_timer("warmup-send", timedelta(milliseconds=WARMUP_SEND_MS), self._warmup_send)
        self._set_timer("warmup-check", timedelta(seconds=WARMUP_CHECK_S), self._warmup_check)

    def _warmup_send(self, event) -> None:
        while self._warm_queue:
            iid = self._warm_queue.popleft()
            if iid not in self._warming:
                continue
            n = self.warmup_bars()
            step = spec_seconds(self.config.bar_spec) or 60
            start = datetime.now(timezone.utc) - timedelta(seconds=(n + 2) * step)
            self._warm_attempts[iid] = self._warm_attempts.get(iid, 0) + 1
            self._warm_sent_ns[iid] = self.clock.timestamp_ns()
            # limit < 500 keeps each call at the lowest Binance request weight
            rid = self.request_bars(self._bar_types[iid], start=start, limit=min(n + 2, 499),
                                    callback=self._warmup_done)
            self._warm_requests[rid] = iid
            return
        self._cancel_timer("warmup-send")

    def _warmup_done(self, request_id) -> None:
        iid = self._warm_requests.pop(request_id, None)
        if iid is not None and iid in self._warming:
            self._finish_warming(iid)

    def _finish_warming(self, iid: InstrumentId) -> None:
        self._warming.discard(iid)
        for bar in self._buffer.pop(iid, []):
            self._route_bar(bar, live=True)

    def _warmup_check(self, event) -> None:
        now = self.clock.timestamp_ns()
        queued = set(self._warm_queue)
        for iid in list(self._warming):
            if iid in queued or now - self._warm_sent_ns.get(iid, now) < 15_000_000_000:
                continue
            if self._warm_attempts.get(iid, 0) < WARMUP_MAX_ATTEMPTS:
                self._warm_queue.append(iid)         # never answered: ask again
            else:
                self._warm_failed.append(iid)
                self._finish_warming(iid)            # give up; warms from live bars instead
        if self._warm_queue:
            self._set_timer("warmup-send", timedelta(milliseconds=WARMUP_SEND_MS), self._warmup_send)
        if self._warming:
            return
        self._cancel_timer("warmup-check")
        if self._warm_failed:
            n = self.warmup_bars()
            step = spec_seconds(self.config.bar_spec) or 60
            self.notify("warn", f"{len(self._warm_failed)} instrument(s) failed to warm up",
                        f"History requests got no answer after {WARMUP_MAX_ATTEMPTS} attempts "
                        f"(usually a Binance rate limit). They will start signalling after "
                        f"~{n * step // 60} min of live bars, or retry now: "
                        + ", ".join(str(i).split(".")[0] for i in self._warm_failed),
                        retry=redis_io.strategy_retry(str(self.id), "rewarm"),
                        key=f"warmup:{self.id}")
        else:
            warm = sum(1 for i in self._tradeable if self.is_warm(i))
            self.notify("success", f"Warm-up complete: {warm}/{len(self._tradeable)} instruments ready",
                        key=f"warmup:{self.id}")

    def _cancel_warmup(self) -> None:
        self._cancel_timer("warmup-send")
        self._cancel_timer("warmup-check")
        self._warm_queue.clear()
        for iid in list(self._warming):
            self._finish_warming(iid)

    # -- feeds -----------------------------------------------------------------

    def _ensure_feeds(self, iids, bars: bool) -> None:
        for iid in iids:
            if bars and iid not in self._bar_subs and iid in self._bar_types:
                self.subscribe_bars(self._bar_types[iid])
                self._bar_subs.add(iid)
            if iid not in self._book_subs:
                # The sandbox runs an L2_MBP book: it ignores quotes, so without
                # deltas a market order has no liquidity and is rejected.
                self.subscribe_order_book_deltas(iid, book_type=BookType.L2_MBP,
                                                 depth=self.config.book_depth,
                                                 params={"update_speed": 100})
                self._book_subs.add(iid)

    def _drop_feeds(self, keep) -> None:
        for iid in list(self._bar_subs):
            if iid not in keep:
                self.unsubscribe_bars(self._bar_types[iid])
                self._bar_subs.discard(iid)
        for iid in list(self._book_subs):
            if iid not in keep:
                self.unsubscribe_order_book_deltas(iid)
                self._book_subs.discard(iid)

    # -- positions / orders (always scoped to this strategy) ---------------------

    def net_qty(self, iid: InstrumentId) -> float:
        return sum(p.signed_qty for p in self.cache.positions_open(instrument_id=iid, strategy_id=self.id))

    def working_orders(self, iid: InstrumentId | None = None) -> list:
        """Every order not in a terminal state, INITIALIZED included (orders_open
        and orders_inflight both miss the gap before the risk engine sees it)."""
        return [o for o in self.cache.orders(instrument_id=iid, strategy_id=self.id) if not o.is_closed]

    def has_working_order(self, iid: InstrumentId) -> bool:
        return bool(self.working_orders(iid))

    def instruments_with_exposure(self) -> set[InstrumentId]:
        return {p.instrument_id for p in self.cache.positions_open(strategy_id=self.id)}

    def order_qty(self, iid: InstrumentId, usd: float, price: float):
        """(Quantity, "") or (None, reason) for a market order worth ``usd``."""
        inst = self.cache.instrument(iid)
        if inst is None:
            return None, "instrument not loaded"
        if price <= 0:
            return None, "no valid price"
        if usd < MIN_ORDER_USD:
            return None, f"${usd:,.2f} is below the ${MIN_ORDER_USD:.0f} minimum notional"
        try:
            # make_qty raises (not returns 0) when the size rounds to zero; inside
            # on_bar that exception would propagate through Nautilus's handler
            qty = inst.make_qty(Decimal(str(usd / price)))
        except ValueError:
            qty = None
        if qty is None or qty.as_double() <= 0:
            return None, f"${usd:,.2f} rounds to zero at size precision {inst.size_precision}"
        if inst.min_quantity is not None and qty < inst.min_quantity:
            return None, f"{qty} is below the venue minimum {inst.min_quantity}"
        if inst.max_quantity is not None and qty > inst.max_quantity:
            return None, f"{qty} exceeds the venue maximum {inst.max_quantity}"
        if inst.min_notional is not None and qty.as_double() * price < inst.min_notional.as_double():
            return None, f"notional below the venue minimum {inst.min_notional}"
        return qty, ""

    def submit_market(self, iid: InstrumentId, side: OrderSide, usd: float, price: float) -> bool:
        qty, why = self.order_qty(iid, usd, price)
        sym = str(iid).split(".")[0]
        if qty is None:
            self._warn_once(f"size:{iid}", f"Skipped {side.name} {sym}", why)
            return False
        self.log.info(f"{side.name} {iid}: ${usd:,.0f} -> {qty} @ ~{price}")
        self.submit_order(self.order_factory.market(iid, side, qty))
        return True

    # -- dashboard output --------------------------------------------------------

    def status(self) -> dict:
        now = self.clock.timestamp_ns()
        step = spec_seconds(self.config.bar_spec)
        grace = max(3 * (step or 60), 90) * 1_000_000_000
        cold = [i for i in self._tradeable if not self.is_warm(i)]
        stale = []
        if self.armed and step and now - self._armed_at_ns > grace:
            stale = [i for i in self._tradeable if now - self._last_live_ns.get(i, self._armed_at_ns) > grace]
        short = lambda xs: [str(i).split(".")[0] for i in xs][:60]  # noqa: E731
        return {
            "state": self.state.name,
            "armed": self.armed,
            "exiting": self.is_exiting(),
            "bar_spec": self.config.bar_spec,
            "instruments": len(self.config.instrument_ids),
            "tradeable": len(self._tradeable) if self.armed else len(self.config.instrument_ids),
            "missing": short(self._missing),
            "warm": len(self._tradeable) - len(cold) if self.armed else 0,
            "warming": len(self._warming),
            "cold": short(cold) if self.armed else [],
            "stale": short(stale),
            "open_positions": len(self.cache.positions_open(strategy_id=self.id)),
            "working_orders": len(self.working_orders()),
            "armed_at": self._armed_at_ns // 1_000_000 if self.armed else 0,
        }

    def publish_metrics(self, metrics: dict) -> None:
        if self._r is None:
            return
        try:
            self._r.set(f"{redis_io.METRICS}{self.id}",
                        json.dumps({"description": self.description(), **metrics}))
        except Exception as e:
            self._warn_log(f"metrics publish failed: {e}")

    def _publish_indicator(self, bar: Bar, values: dict) -> None:
        if self._r is None:
            return
        try:
            self._r.xadd(f"{redis_io.INDICATORS}{bar.bar_type.instrument_id}",
                         {"ts": str(bar.ts_event // 1_000_000_000), "tf": str(bar.bar_type.spec),
                          **{k: str(round(v, 8)) for k, v in values.items()}},
                         maxlen=50000, approximate=True)
        except Exception as e:
            self._warn_log(f"indicator publish failed: {e}")

    def notify(self, level: str, title: str, detail: str = "", retry: dict | None = None,
               key: str = "") -> None:
        log = self.log.error if level == "error" else self.log.warning if level == "warn" else self.log.info
        log(f"{title}{': ' + detail if detail else ''}")
        if not self.config.backtest:
            redis_io.notify(self._r, level, f"strategy:{self.id}", title, detail, retry=retry, key=key)

    def _warn_once(self, key: str, title: str, detail: str) -> None:
        if key in self._warned:
            return
        self._warned.add(key)
        self.notify("warn", title, detail, key=key)

    def _warn_log(self, msg: str) -> None:
        if msg not in self._warned:
            self._warned.add(msg)
            self.log.warning(msg)

    # -- timers ------------------------------------------------------------------

    def _set_timer(self, name: str, interval: timedelta, callback) -> None:
        full = f"{self.id}:{name}"
        if full not in self.clock.timer_names:
            self.clock.set_timer(name=full, interval=interval, callback=callback)

    def _cancel_timer(self, name: str) -> None:
        full = f"{self.id}:{name}"
        if full in self.clock.timer_names:
            self.clock.cancel_timer(full)
