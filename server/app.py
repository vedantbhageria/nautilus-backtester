"""Dashboard server: WebSocket + REST bridge between the browser and Redis.

Live mode: market data + state from Redis DB 0.
Test mode: the same market data; positions/PnL filtered to strategies tagged
"test", merged with backtest results from DB 1.

Reliability contract with the browser
* Every command is validated here and refused with a reason (4xx/503) instead
  of being queued blindly. Strategy commands are refused while the node's
  heartbeat is stale: a queued "start" executing an hour later is worse than an
  error now. Accepted commands get a ``cmd_id``; the node answers with a
  notification carrying it.
* Notifications (``dashboard:notifications``) are pushed to every client and
  replayed on connect. A watchdog adds its own for Redis outages and a silent
  node, delivered directly if Redis itself is the thing that's down.
* ``/api/health`` and ``/api/consistency`` expose component health and the full
  consistency report (see ``trading.consistency``).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime

import redis.asyncio as aioredis
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response
from redis.exceptions import RedisError

from server import pg_history
from trading import consistency
from trading import redis_io as K

log = logging.getLogger("dashboard")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s dashboard: %(message)s"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)

HTML_PATH = os.path.join(os.path.dirname(__file__), "dashboard.html")
MODES = ("live", "test")

QUICK_HISTORY_COUNT = 50000
HIST_FETCH_CAP = 50000
XREAD_BLOCK_MS = 50
XREAD_COUNT = 500
DEFAULT_WINDOW = "10m"
WINDOWS = {"2m", "10m", "30m", "1h", "4h", "1d", "3d"}
TRADES_INFIX = ":data.trades."
BARS_INFIX = ":data.bars."
HIST_BARS_INFIX = ":historical.data.bars."
HIST_TRADES_INFIX = ":historical.data.trades."
NODE_STALE_S = 10               # refuse strategy commands when the heartbeat is older

STRATEGY_ACTIONS = {"start_strategy", "stop_strategy", "close_strategy", "export_csv",
                    "start_test_strategy", "rewarm"}
DATA_ACTIONS = {"subscribe", "unsubscribe", "backfill"}

TIMEFRAMES = {"1s": "1-SECOND", "5s": "5-SECOND", "15s": "15-SECOND",
              "1m": "1-MINUTE", "5m": "5-MINUTE", "15m": "15-MINUTE", "1h": "1-HOUR"}
TF_SOURCE = {"1s": "INTERNAL", "5s": "INTERNAL", "15s": "INTERNAL",
             "1m": "EXTERNAL", "5m": "EXTERNAL", "15m": "EXTERNAL", "1h": "EXTERNAL"}

app = FastAPI()

# Market data always comes from the live node (DB 0); state is per mode.
r = aioredis.from_url(K.live_url(), decode_responses=True, socket_keepalive=True,
                      socket_connect_timeout=3, max_connections=300)
r_state = {"live": r, "test": aioredis.from_url(K.test_url(), decode_responses=True,
                                                socket_connect_timeout=3, max_connections=50)}
r_sync = {"live": K.connect(K.live_url()), "test": K.connect(K.test_url())}

_BT_EXEC = ThreadPoolExecutor(max_workers=4, thread_name_prefix="history")
_PF_EXEC = ThreadPoolExecutor(max_workers=2, thread_name_prefix="portfolio")


# ── helpers ──────────────────────────────────────────────────────────────────
def symbol_from_trade_key(key: str) -> str | None:
    i = key.find(TRADES_INFIX)
    if i < 0:
        return None
    venue, _, sym = key[i + len(TRADES_INFIX):].partition(".")
    return f"{sym}.{venue}" if sym else None


async def _first_key(pattern: str) -> str | None:
    async for key in r.scan_iter(match=pattern, count=1000):
        return key
    return None


def _trade_pattern(symbol: str, infix: str) -> str:
    sym, _, venue = symbol.rpartition(".")
    return f"*{infix}{venue}.{sym}"


def _to_epoch_seconds(ts) -> float:
    if isinstance(ts, (int, float)):
        return float(ts) / 1e9
    s = str(ts)
    if s.isdigit():
        return int(s) / 1e9
    s = re.sub(r"(\.\d{6})\d+", r"\1", s.replace("Z", "+00:00"))
    return datetime.fromisoformat(s).timestamp()


def _payload_value(fields: dict) -> str | None:
    if not fields:
        return None
    return fields.get("payload") or next(iter(fields.values()))


def _tick(fields: dict) -> dict | None:
    try:
        d = json.loads(_payload_value(fields))
        return {"t": _to_epoch_seconds(d["ts_event"]), "price": float(d["price"]),
                "qty": float(d.get("size", 0) or 0)}
    except (KeyError, ValueError, TypeError):
        return None


def _bar(fields: dict) -> dict | None:
    try:
        d = json.loads(_payload_value(fields))
        # ms-rounded so Redis and Postgres bars share one key (float ns -> s
        # otherwise gives ...59.9990001 vs ...59.999 and duplicates candles)
        return {"t": round(_to_epoch_seconds(d["ts_event"]), 3), "o": float(d["open"]),
                "h": float(d["high"]), "l": float(d["low"]), "c": float(d["close"]),
                "v": float(d.get("volume", 0) or 0)}
    except (KeyError, ValueError, TypeError):
        return None


def _indicator(fields: dict) -> dict | None:
    try:
        return {"ts": int(fields["ts"]), "tf": fields["tf"],
                **{k: float(v) for k, v in fields.items() if k not in ("ts", "tf")}}
    except (KeyError, ValueError, TypeError):
        return None


def bar_type_for(symbol: str, timeframe: str) -> str | None:
    spec = TIMEFRAMES.get(timeframe)
    return f"{symbol}-{spec}-LAST-{TF_SOURCE[timeframe]}" if spec else None


def _tf_of(symbol: str, bar_type: str) -> str | None:
    return next((tf for tf in TIMEFRAMES if bar_type_for(symbol, tf) == bar_type), None)


# ── client state ─────────────────────────────────────────────────────────────
clients: dict[WebSocket, str] = {}                        # ws -> mode
mode_clients: dict[str, set[WebSocket]] = {"live": set(), "test": set()}
tick_clients: dict[str, set[WebSocket]] = defaultdict(set)   # symbol -> ws
bar_clients: dict[str, set[WebSocket]] = defaultdict(set)    # bar_type -> ws
ind_clients: dict[str, set[WebSocket]] = defaultdict(set)    # symbol -> ws
client_bars: dict[WebSocket, set[str]] = defaultdict(set)
tails: dict[str, asyncio.Task] = {}                        # tail name -> task
packet_clients: set[WebSocket] = set()
_pkt_counts: dict[str, int] = defaultdict(int)


async def _safe_send(ws: WebSocket, payload) -> None:
    try:
        await ws.send_text(payload if isinstance(payload, str) else json.dumps(payload))
    except Exception:
        pass            # socket closing; its handler cleans up


async def _broadcast(targets, payload) -> None:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    for ws in list(targets):
        await _safe_send(ws, text)


# ── stream tails ───────────────────────────────────────────────────────────────
async def _tail(name: str, pattern: str | None, key: str | None, subscribers, parse, frame,
                start: str = "$", backlog: int = 0) -> None:
    """Forward new entries of one Redis stream to its subscribers until none are
    left. Waits (polling) for the stream to exist: an instrument's first trade
    can take minutes. One implementation for ticks, bars and indicators."""
    try:
        while subscribers() and key is None:
            key = await _first_key(pattern)
            if key is None:
                await asyncio.sleep(1.0)
        if key is None:
            return
        cursor = start
        if backlog:
            recent = await r.xrevrange(key, count=backlog)
            cursor = recent[-1][0] if len(recent) == backlog else "0-0"
        while subscribers():
            try:
                res = await r.xread({key: cursor}, count=XREAD_COUNT, block=XREAD_BLOCK_MS)
            except RedisError as e:
                log.warning(f"tail {name}: {e}")
                await asyncio.sleep(1.0)
                continue
            if not res or not res[0][1]:
                continue
            cursor = res[0][1][-1][0]
            points = [p for _id, f in res[0][1] if (p := parse(f))]
            if points:
                await _broadcast(subscribers(), frame(points))
    except asyncio.CancelledError:
        pass
    except Exception:
        log.exception(f"tail {name} crashed")
    finally:
        tails.pop(name, None)


def _ensure(name: str, coro_factory) -> None:
    t = tails.get(name)
    if t is None or t.done():
        tails[name] = asyncio.create_task(coro_factory())


def _ensure_symbol_tails(sym: str) -> None:
    subs = lambda: tick_clients.get(sym)  # noqa: E731

    def live_frame(points):
        _pkt_counts[sym] += len(points)
        return {"type": "ticks", "symbol": sym, "points": points}
    _ensure(f"ticks:{sym}", lambda: _tail(f"ticks:{sym}", _trade_pattern(sym, TRADES_INFIX), None, subs,
                                          _tick, live_frame))
    _ensure(f"hticks:{sym}", lambda: _tail(f"hticks:{sym}", _trade_pattern(sym, HIST_TRADES_INFIX), None,
                                           subs, _tick,
                                           lambda p: {"type": "hist_ticks", "symbol": sym, "points": p}))
    _ensure(f"ind:{sym}", lambda: _tail(f"ind:{sym}", None, f"{K.INDICATORS}{sym}", lambda: ind_clients.get(sym),
                                        _indicator, lambda p: {"type": "indicators", "symbol": sym, "points": p},
                                        start="0-0"))


def _ensure_bar_tails(sym: str, bt: str) -> None:
    subs = lambda: bar_clients.get(bt)  # noqa: E731
    frame = lambda p: {"type": "bars", "symbol": sym, "bar_type": bt, "points": p}  # noqa: E731
    _ensure(f"bars:{bt}", lambda: _tail(f"bars:{bt}", f"*{BARS_INFIX}{bt}", None, subs, _bar, frame, backlog=2))
    _ensure(f"hbars:{bt}", lambda: _tail(f"hbars:{bt}", f"*{HIST_BARS_INFIX}{bt}", None, subs, _bar, frame,
                                         start="0-0"))


# ── history on subscribe ──────────────────────────────────────────────────────
async def _send_history(ws: WebSocket, symbol: str, window: str) -> None:
    seen, ticks = set(), []
    for infix, cap in ((HIST_TRADES_INFIX, HIST_FETCH_CAP), (TRADES_INFIX, QUICK_HISTORY_COUNT)):
        key = await _first_key(_trade_pattern(symbol, infix))
        if key is None:
            continue
        for _id, fields in await r.xrevrange(key, "+", "-", count=cap):
            pt = _tick(fields)
            if pt and (k := (pt["t"], pt["price"], pt["qty"])) not in seen:
                seen.add(k)
                ticks.append(pt)
    ticks.sort(key=lambda d: d["t"])
    await _safe_send(ws, {"type": "history", "symbol": symbol, "ticks": ticks, "complete": True, "window": window})


_catchup_at: dict[str, float] = {}
CATCHUP_COOLDOWN_S = 600
CATCHUP_MAX_HOLES = 3


async def _catch_up(bar_type: str, tf: str, bars: list[dict], pg_last: float | None) -> None:
    """Ask the node to backfill the holes between the last Postgres bar and now
    (normally one: last sync -> first live bar). Older holes are left to the
    backtester's Data tab."""
    if pg_last is None or time.monotonic() - _catchup_at.get(bar_type, -1e9) < CATCHUP_COOLDOWN_S:
        return
    step, now = pg_history.TF_SECONDS[tf], time.time()
    holes, prev = [], pg_last
    for t in (b["t"] for b in bars if pg_last - step <= b["t"] <= now):
        if t - prev > 1.5 * step:
            holes.append((prev, t))
        prev = max(prev, t)
    if now - prev > 2 * step:
        holes.append((prev, None))
    if not holes:
        return
    _catchup_at[bar_type] = time.monotonic()
    for start, end in holes[:CATCHUP_MAX_HOLES]:
        await _data_command({"action": "backfill", "bar_type": bar_type, "start": start,
                             **({"end": end} if end is not None else {})})


async def _send_bar_history(ws: WebSocket, symbol: str, bar_type: str) -> None:
    by_t: dict[float, dict] = {}
    for infix in (HIST_BARS_INFIX, BARS_INFIX):
        key = await _first_key(f"*{infix}{bar_type}")
        if key is None:
            continue
        for _id, fields in await r.xrevrange(key, "+", "-", count=HIST_FETCH_CAP):
            if (bar := _bar(fields)) and bar["t"] not in by_t:
                by_t[bar["t"]] = bar
    if clients.get(ws) == "test":
        # backtest chart bars (live bars win on the same timestamp)
        for bar in K.read_json(r_sync["test"], f"{K.TEST_CHART}{bar_type}", []) or []:
            if bar.get("t") is not None:
                by_t.setdefault(round(bar["t"], 3), bar)
    tf = _tf_of(symbol, bar_type)
    pg_last = None
    if tf in pg_history.TF_SECONDS:
        pg = await asyncio.get_running_loop().run_in_executor(_BT_EXEC, pg_history.load_bars, symbol, tf)
        now = time.time()
        for bar in pg:
            by_t.setdefault(bar["t"], bar)
            if bar["t"] <= now:            # skip the still-forming aggregate bucket
                pg_last = bar["t"]
    bars = [by_t[t] for t in sorted(by_t)]
    await _safe_send(ws, {"type": "bar_history", "symbol": symbol, "bar_type": bar_type, "bars": bars})
    if tf in pg_history.TF_SECONDS:
        await _catch_up(bar_type, tf, bars, pg_last)


async def _send_indicator_history(ws: WebSocket, symbol: str) -> None:
    entries = await r.xrange(f"{K.INDICATORS}{symbol}", "-", "+", count=HIST_FETCH_CAP)
    points = [p for _id, f in entries if (p := _indicator(f))]
    if points:
        await _safe_send(ws, {"type": "indicator_history", "symbol": symbol, "points": points})


# ── portfolio frames ──────────────────────────────────────────────────────────
class _LedgerCache:
    rev = None
    records: list = []


_ledger_cache = _LedgerCache()


def _ledger(rev) -> list:
    """Closed-position ledger, re-read only when the node's revision changes."""
    if rev != _ledger_cache.rev:
        led = K.read_json(r_sync["live"], K.CLOSED_POSITIONS, {}) or {}
        _ledger_cache.records = sorted(led.values(), key=lambda c: c.get("ts_closed", 0), reverse=True)
        _ledger_cache.rev = rev
    return _ledger_cache.records


def build_portfolio(mode: str) -> dict | None:
    rc = r_sync["live"]
    snap = K.read_json(rc, K.PORTFOLIO)
    if not snap:
        return None
    modes = rc.hgetall(K.STRATEGY_MODES) or {}
    status = K.read_json(rc, K.STRATEGY_STATUS, {}) or {}
    strategies = sorted(snap.get("strategies", []))
    mine = lambda sid: modes.get(sid, "live") == mode  # noqa: E731  (untagged count as live)
    positions = [p for p in snap.get("positions", []) if mine(p.get("strategy"))]
    closed = [c for c in _ledger(snap.get("closed_rev")) if mine(c.get("strategy"))]
    session_closed = [c for c in closed if c.get("session") == snap.get("session")]
    if mode == "test":
        bt = K.read_json(r_sync["test"], f"{K.BACKTEST}:positions", {}) or {}
        have = {c.get("id") for c in closed}
        extra = [{**c, "session": "backtest"} for c in bt.get("closed_positions", []) if c.get("id") not in have]
        closed = sorted(closed + extra, key=lambda c: c.get("ts_closed", 0), reverse=True)
        session_closed += extra        # a test run = backtest + its live hand-off
    keys = [f"{K.METRICS}{sid}" for sid in strategies]
    metrics = {}
    for sid, raw in zip(strategies, rc.mget(keys) if keys else []):
        try:
            metrics[sid] = json.loads(raw) if raw else {}
        except ValueError:
            metrics[sid] = {}
    hb = K.read_json(rc, K.HEARTBEAT) or {}
    overall = {ccy: v["realized"] for ccy, v in consistency.pnl_from(positions, closed).items()}
    return {
        "type": "portfolio", "ts": snap.get("ts"), "session": snap.get("session"),
        "strategies": strategies,
        "strategy_status": status,
        "strategy_states": {sid: "RUNNING" if status.get(sid, {}).get("armed") else "STOPPED" for sid in strategies},
        "strategy_modes": modes,
        "positions": positions, "closed_positions": closed,
        "pnl": consistency.pnl_from(positions, session_closed),
        "overall_pnl": overall,
        "prices": snap.get("prices", {}), "metrics": metrics,
        "node": {"age": round(time.time() - hb["ts"] / 1000, 1) if hb.get("ts") else None,
                 "session": hb.get("session"), "held_writes": hb.get("held_writes", 0)},
    }


async def _portfolio_loop(mode: str) -> None:
    loop = asyncio.get_running_loop()
    last_ts, failures = None, 0
    while True:
        await asyncio.sleep(0.5)
        if not mode_clients[mode]:
            continue
        try:
            frame = await loop.run_in_executor(_PF_EXEC, build_portfolio, mode)
            failures = 0
        except Exception as e:
            failures += 1
            if failures == 1:
                log.warning(f"portfolio ({mode}) failed: {e}")
            continue
        if frame is None or frame["ts"] == last_ts:
            continue
        last_ts = frame["ts"]
        await _broadcast(mode_clients[mode], frame)


# ── order events + notifications ───────────────────────────────────────────────
def _order_frame(fields: dict) -> dict:
    return {"type": "order_event", **fields, "ts": float(fields.get("ts") or 0)}


def _notification_frame(nid: str, fields: dict) -> dict:
    try:
        retry = json.loads(fields["retry"]) if fields.get("retry") else None
    except ValueError:
        retry = None
    return {"type": "notification", "id": nid, **fields, "ts": float(fields.get("ts") or 0), "retry": retry}


async def _tail_order_events() -> None:
    cursor = "$"
    while True:
        try:
            res = await r.xread({K.ORDER_EVENTS: cursor}, count=200, block=1000)
            if not res:
                continue
            cursor = res[0][1][-1][0]
            modes = await r.hgetall(K.STRATEGY_MODES)
            for _id, fields in res[0][1]:
                await _broadcast(mode_clients[modes.get(fields.get("strategy", ""), "live")], _order_frame(fields))
        except asyncio.CancelledError:
            return
        except Exception as e:
            log.debug(f"order tail: {e}")
            await asyncio.sleep(1.0)


async def _tail_notifications() -> None:
    cursor = "$"
    while True:
        try:
            res = await r.xread({K.NOTIFICATIONS: cursor}, count=100, block=1000)
            if not res:
                continue
            cursor = res[0][1][-1][0]
            for nid, fields in res[0][1]:
                await _broadcast(clients, _notification_frame(nid, fields))
        except asyncio.CancelledError:
            return
        except Exception as e:
            log.debug(f"notification tail: {e}")
            await asyncio.sleep(1.0)


async def server_notify(level: str, title: str, detail: str = "", key: str = "", retry: dict | None = None) -> None:
    """Notification from the server itself; broadcast directly if Redis is down."""
    rec = K.notification(level, "server", title, detail, retry, key)
    log.log(logging.ERROR if level == "error" else logging.WARNING if level == "warn" else logging.INFO,
            f"{title} {detail}")
    try:
        await r.xadd(K.NOTIFICATIONS, rec, maxlen=1000, approximate=True)
    except Exception:
        await _broadcast(clients, _notification_frame(f"local-{uuid.uuid4().hex[:8]}", rec))


# ── watchdog + health ───────────────────────────────────────────────────────────
health: dict = {"redis": {"live": None, "test": None}, "node": {}, "postgres": None, "checked": 0}


async def _watchdog() -> None:
    state = {"redis_live": True, "redis_test": True, "node": None, "session": None}
    pg_checked = 0.0
    while True:
        await asyncio.sleep(2.0)
        for name in MODES:
            try:
                await r_state[name].ping()
                ok, err = True, None
            except Exception as e:
                ok, err = False, str(e)
            health["redis"][name] = err
            if ok != state[f"redis_{name}"]:
                state[f"redis_{name}"] = ok
                if ok:
                    await server_notify("success", f"Redis ({name} DB) reachable again", key=f"redis:{name}")
                else:
                    await server_notify("error", f"Redis ({name} DB) unreachable",
                                        f"{err}. If Redis runs in WSL, the distro may have shut down: "
                                        "run launch.bat (it keeps WSL alive).", key=f"redis:{name}")
        if health["redis"]["live"]:
            continue
        try:
            hb = await r.get(K.HEARTBEAT)
            hb = json.loads(hb) if hb else None
        except (RedisError, ValueError):
            continue
        age = time.time() - hb["ts"] / 1000 if hb else None
        alive = age is not None and age < NODE_STALE_S
        health["node"] = {"alive": alive, "age": round(age, 1) if age is not None else None,
                          "pid": (hb or {}).get("pid"), "session": (hb or {}).get("session"),
                          "held_writes": (hb or {}).get("held_writes", 0), "venue": (hb or {}).get("venue"),
                          "sandbox": (hb or {}).get("sandbox")}
        if state["node"] is not None and alive != state["node"]:
            if alive:
                await server_notify("success", "Trading node is responding again", key="node")
            else:
                await server_notify("error", "Trading node stopped responding",
                                    f"Last heartbeat {age:.0f}s ago. Strategy commands are refused until it's back."
                                    if age is not None else "No heartbeat found.", key="node")
        state["node"] = alive
        session = (hb or {}).get("session")
        if state["session"] and session and session != state["session"]:
            await server_notify("warn", "Trading node restarted",
                                "Strategies start idle after a restart; press Start to resume them.", key="node-restart")
        state["session"] = session or state["session"]
        if time.time() - pg_checked > 30:
            pg_checked = time.time()
            health["postgres"] = await asyncio.get_running_loop().run_in_executor(_BT_EXEC, pg_history.probe)
        health["checked"] = time.time()


# ── packet rate ─────────────────────────────────────────────────────────────────
async def _packet_loop() -> None:
    while True:
        await asyncio.sleep(1.0)
        rates = dict(_pkt_counts)
        _pkt_counts.clear()
        if packet_clients:
            await _broadcast(packet_clients, {"type": "packets", "t": time.time(),
                                              "total": sum(rates.values()), "rates": rates})


@app.on_event("startup")
async def _startup() -> None:
    for coro in (_packet_loop(), _portfolio_loop("live"), _portfolio_loop("test"),
                 _tail_order_events(), _tail_notifications(), _watchdog()):
        asyncio.create_task(coro)
    log.info(f"dashboard up: live={K.live_url()} test={K.test_url()}")


# ── commands ────────────────────────────────────────────────────────────────────
class CommandError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def _validate_dates(cmd: dict) -> None:
    try:
        start = date.fromisoformat(cmd.get("start_date") or "")
    except ValueError:
        raise CommandError(400, "start_date is required (YYYY-MM-DD)")
    end = None
    if cmd.get("end_date"):
        try:
            end = date.fromisoformat(cmd["end_date"])
        except ValueError:
            raise CommandError(400, f"end_date {cmd['end_date']!r} is not YYYY-MM-DD")
    if start >= date.today() and not end:
        raise CommandError(400, "start_date must be before today")
    if end is not None and end <= start:
        raise CommandError(400, "end_date must be after start_date")


async def _node_alive() -> tuple[bool, str]:
    raw = await r.get(K.HEARTBEAT)
    try:
        hb = json.loads(raw) if raw else None
    except ValueError:
        hb = None
    if not hb:
        return False, "the trading node has not started (no heartbeat)"
    age = time.time() - hb["ts"] / 1000
    if age > NODE_STALE_S:
        return False, f"the trading node is not responding (last heartbeat {age:.0f}s ago)"
    return True, ""


async def _data_command(cmd: dict) -> None:
    await r.xadd(K.DATA_CMDS, {"json": json.dumps(cmd)}, maxlen=1000, approximate=True)


async def handle_command(mode: str, cmd) -> dict:
    """Validate and send one command. Raises CommandError(status, reason)."""
    if mode not in MODES:
        raise CommandError(400, f"unknown mode {mode!r}")
    if not isinstance(cmd, dict) or not cmd.get("action"):
        raise CommandError(400, "missing 'action'")
    action = cmd["action"]
    try:
        if action in STRATEGY_ACTIONS:
            sid = cmd.get("strategy_id")
            if not sid:
                raise CommandError(400, "missing 'strategy_id'")
            if not await r.sismember(K.STRATEGIES, sid):
                raise CommandError(404, f"unknown strategy {sid!r}")
            if action == "start_test_strategy":
                _validate_dates(cmd)
            alive, why = await _node_alive()
            if not alive:
                raise CommandError(503, f"Command not sent: {why}.")
            cmd_id = uuid.uuid4().hex[:12]
            fields = {k: str(v) for k, v in cmd.items() if v is not None}
            fields.update(mode=mode, cmd_id=cmd_id)
            await r.xadd(K.STRATEGY_CMDS, fields, maxlen=200, approximate=True)
            return {"ok": True, "cmd_id": cmd_id, "sent": fields}
        if action in DATA_ACTIONS:
            await _data_command(cmd)
            return {"ok": True, "sent": cmd}
    except RedisError as e:
        raise CommandError(503, f"Redis unavailable: {e}")
    raise CommandError(400, f"unknown action {action!r}")


async def _command_response(mode: str, cmd) -> JSONResponse:
    try:
        return JSONResponse(await handle_command(mode, cmd))
    except CommandError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=e.status)


@app.post("/api/command")
async def command(cmd: dict, mode: str = "live"):
    return await _command_response(mode, cmd)


@app.post("/api/{mode}/command")
async def command_mode(mode: str, cmd: dict):
    return await _command_response(mode, cmd)


# ── WebSocket ───────────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket, mode: str = Query("live")):
    mode = mode if mode in MODES else "live"
    await websocket.accept()
    clients[websocket] = mode
    mode_clients[mode].add(websocket)
    loop = asyncio.get_running_loop()
    try:
        frame = await loop.run_in_executor(_PF_EXEC, build_portfolio, mode)
        if frame:
            await _safe_send(websocket, frame)
        modes = await r.hgetall(K.STRATEGY_MODES)
        for _id, fields in reversed(await r.xrevrange(K.ORDER_EVENTS, count=500)):
            if modes.get(fields.get("strategy", ""), "live") == mode:
                await _safe_send(websocket, _order_frame(fields))
        for nid, fields in reversed(await r.xrevrange(K.NOTIFICATIONS, count=60)):
            await _safe_send(websocket, {**_notification_frame(nid, fields), "replay": True})
    except Exception as e:
        log.warning(f"ws init ({mode}): {e}")
        await _safe_send(websocket, _notification_frame("local-init", K.notification(
            "error", "server", "Dashboard could not load current state", str(e), key="ws-init")))

    try:
        while True:
            try:
                msg = json.loads(await websocket.receive_text())
            except ValueError:
                continue
            action = msg.get("action")
            try:
                if action == "subscribe":
                    window = msg.get("window") if msg.get("window") in WINDOWS else DEFAULT_WINDOW
                    for sym in [s for s in msg.get("symbols", []) if isinstance(s, str)]:
                        tick_clients[sym].add(websocket)
                        ind_clients[sym].add(websocket)
                        await _data_command({"action": "subscribe", "instrument_id": sym})
                        await _send_history(websocket, sym, window)
                        await _send_indicator_history(websocket, sym)
                        _ensure_symbol_tails(sym)
                        for tf in TIMEFRAMES:
                            bt = bar_type_for(sym, tf)
                            bar_clients[bt].add(websocket)
                            client_bars[websocket].add(bt)
                            await _data_command({"action": "subscribe", "instrument_id": sym, "bar_type": bt})
                            await _send_bar_history(websocket, sym, bt)
                            _ensure_bar_tails(sym, bt)
                elif action == "set_window":
                    sym = msg.get("symbol")
                    if isinstance(sym, str) and websocket in tick_clients.get(sym, ()):
                        window = msg.get("window") if msg.get("window") in WINDOWS else DEFAULT_WINDOW
                        await _send_history(websocket, sym, window)
                elif action == "refresh_bars":
                    sym, bt = msg.get("symbol"), msg.get("bar_type")
                    if isinstance(sym, str) and websocket in bar_clients.get(bt, ()):
                        await _send_bar_history(websocket, sym, bt)
                elif action == "refresh_indicators":
                    sym = msg.get("symbol")
                    if isinstance(sym, str) and websocket in ind_clients.get(sym, ()):
                        await _send_indicator_history(websocket, sym)
                elif action == "unsubscribe":
                    for sym in [s for s in msg.get("symbols", []) if isinstance(s, str)]:
                        tick_clients.get(sym, set()).discard(websocket)
                        ind_clients.get(sym, set()).discard(websocket)
                        for bt in [b for b in client_bars.get(websocket, set()) if b.startswith(f"{sym}-")]:
                            client_bars[websocket].discard(bt)
                            bar_clients.get(bt, set()).discard(websocket)
                elif action in STRATEGY_ACTIONS:
                    try:
                        res = await handle_command(mode, msg)
                        await _safe_send(websocket, {"type": "command_result", **res})
                    except CommandError as e:
                        await _safe_send(websocket, {"type": "command_result", "ok": False, "error": str(e)})
            except RedisError as e:
                await _safe_send(websocket, _notification_frame("local-ws", K.notification(
                    "error", "server", f"'{action}' failed: Redis unavailable", str(e), key="ws-redis")))
    except WebSocketDisconnect:
        pass
    finally:
        mode_clients[mode].discard(websocket)
        clients.pop(websocket, None)
        for subs in (tick_clients, ind_clients):
            for s in subs.values():
                s.discard(websocket)
        for bt in client_bars.pop(websocket, set()):
            bar_clients.get(bt, set()).discard(websocket)


@app.websocket("/ws/packets")
async def ws_packets(websocket: WebSocket):
    await websocket.accept()
    packet_clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        packet_clients.discard(websocket)


# ── REST ─────────────────────────────────────────────────────────────────────────
@app.get("/api/symbols")
async def symbols():
    out = set()
    try:
        async for key in r.scan_iter(match="*:instruments:*", count=2000):
            _, _, iid = key.partition(":instruments:")
            if iid:
                out.add(iid)
        if not out:
            async for key in r.scan_iter(match=f"*{TRADES_INFIX}*", count=1000):
                if sym := symbol_from_trade_key(key):
                    out.add(sym)
    except RedisError as e:
        return JSONResponse({"ok": False, "error": f"Redis unavailable: {e}"}, status_code=503)
    return JSONResponse(sorted(out))


@app.get("/api/health")
async def api_health():
    return JSONResponse({**health, "now": time.time(), "ws_clients": len(clients)})


def _consistency_sync() -> dict:
    return consistency.server_checks(consistency.gather(r_sync["live"], r_sync["test"]))


@app.get("/api/consistency")
async def api_consistency():
    rep = await asyncio.get_running_loop().run_in_executor(_PF_EXEC, _consistency_sync)
    return JSONResponse(rep)


@app.get("/api/{mode}/consistency")
async def api_consistency_mode(mode: str):
    return await api_consistency()


async def _json_from(fn, *args) -> Response:
    try:
        body = await asyncio.get_running_loop().run_in_executor(_BT_EXEC, fn, *args)
    except RedisError as e:
        return JSONResponse({"ok": False, "error": f"Redis unavailable: {e}"}, status_code=503)
    return Response(content=body, media_type="application/json")


def _account_equity_sync(mode: str) -> str:
    rc_live, rc_test = r_sync["live"], r_sync["test"]
    pts = [json.loads(e) for e in rc_live.lrange(K.EQUITY[mode], 0, -1)]
    if mode == "test":
        # Backtest curve first, then the live hand-off offset to continue from the
        # backtest's ending NAV: live_nav + (bt_end_nav - ACCOUNT_START).
        bt = K.read_json(rc_test, f"{K.BACKTEST}:equity", []) or []
        offset = float(rc_test.get(f"{K.BACKTEST}:equity:end_nav") or K.ACCOUNT_START) - K.ACCOUNT_START
        if bt:
            pts = [p for p in pts if p["ts"] > bt[-1]["ts"]]
        pts = bt + [{**p, "nav": round(p["nav"] + offset, 4)} for p in pts]
        peak = max((p["nav"] for p in pts), default=K.ACCOUNT_START)
    else:
        peak = float(rc_live.get(K.EQUITY_PEAK["live"]) or K.ACCOUNT_START)
    if len(pts) > 5000:
        stride = len(pts) // 5000 + 1
        pts = pts[::stride] + ([pts[-1]] if (len(pts) - 1) % stride else [])
    return json.dumps({"points": pts, "peak": peak})


@app.get("/api/{mode}/account/equity")
async def account_equity(mode: str):
    return await _json_from(_account_equity_sync, mode if mode in MODES else "live")


@app.get("/api/{mode}/backtest/meta")
async def backtest_meta(mode: str):
    return await _json_from(lambda: r_sync["test"].get(f"{K.BACKTEST}:meta") or '{"status": "none"}')


@app.get("/api/{mode}/backtest/positions")
async def backtest_positions(mode: str):
    return await _json_from(lambda: r_sync["test"].get(f"{K.BACKTEST}:positions")
                            or '{"positions": [], "closed_positions": [], "pnl": {}}')


def _positions_sync(mode: str) -> str:
    f = build_portfolio(mode) or {}
    return json.dumps({"positions": f.get("positions", []), "closed_positions": f.get("closed_positions", []),
                       "pnl": f.get("pnl", {})})


@app.get("/api/{mode}/positions")
async def positions(mode: str):
    return await _json_from(_positions_sync, mode if mode in MODES else "live")


def _chart_bars_sync(symbol: str) -> str:
    """Backtest chart bars (all timeframes) + indicator series for one symbol."""
    rc = r_sync["test"]
    bars = []
    for tf, spec in (("1m", "1-MINUTE"), ("5m", "5-MINUTE"), ("15m", "15-MINUTE"), ("1h", "1-HOUR")):
        bt = f"{symbol}-{spec}-LAST-EXTERNAL"
        if (data := K.read_json(rc, f"{K.TEST_CHART}{bt}")) is not None:
            bars.append({"tf": tf, "bar_type": bt, "bars": data})
    inds = [{**pt, "tf": "1-MINUTE-LAST"} for pt in K.read_json(rc, f"{K.BACKTEST}:indicators:{symbol}", []) or []]
    spec = K.read_json(rc, f"{K.BACKTEST}:chart_spec", []) or []
    return json.dumps({"bars": bars, "indicators": inds, "spec": spec})


@app.get("/api/{mode}/chart_bars/{symbol}")
async def chart_bars(mode: str, symbol: str):
    return await _json_from(_chart_bars_sync, symbol)


@app.get("/", response_class=HTMLResponse)
async def index():
    with open(HTML_PATH, "r", encoding="utf-8") as fh:
        return HTMLResponse(fh.read())
