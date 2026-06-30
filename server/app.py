"""Dashboard server — live + test mode separation.

Live mode : market data + state from LIVE_REDIS_URL (db 0, default).
Test mode : market data from LIVE_REDIS_URL (shared live feed),
            state (portfolio / positions / strategy) from TEST_REDIS_URL (db 1).

Both modes use identical WebSocket and REST APIs; the ?mode= query param
selects which backend each client talks to.
"""

import os
import re
import json
import asyncio
import time
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import redis as syncredis
import redis.asyncio as aioredis
from redis.exceptions import RedisError
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response

load_dotenv()

LIVE_REDIS_URL = os.getenv("LIVE_REDIS_URL", os.getenv("REDIS_URL", "redis://localhost:6379/0"))
TEST_REDIS_URL = os.getenv("TEST_REDIS_URL", "redis://localhost:6379/1")
HTML_PATH = os.path.join(os.path.dirname(__file__), "dashboard.html")

QUICK_HISTORY_COUNT = 50000
HIST_FETCH_CAP = 50000
XREAD_BLOCK_MS = 50
XREAD_COUNT = 500
DEFAULT_WINDOW = "10m"
WINDOWS = {"2m", "10m", "30m", "1h", "4h", "1d", "3d"}
TRADES_INFIX = ":data.trades."
COMMAND_STREAM = "dashboard:commands"
STRATEGY_CMDS_STREAM = "dashboard:strategy_cmds"
STRATEGY_STATES_KEY = "dashboard:strategy_states"
INDICATORS_PREFIX = "dashboard:indicators:"
ORDER_EVENTS_KEY = "dashboard:order_events"
STRATEGY_ACTIONS = {"start_strategy", "stop_strategy", "close_strategy",
                    "export_csv", "start_test_strategy"}
BARS_INFIX = ":data.bars."
HIST_BARS_INFIX = ":historical.data.bars."
HIST_TRADES_INFIX = ":historical.data.trades."

TIMEFRAMES = {
    "1s": "1-SECOND", "5s": "5-SECOND", "15s": "15-SECOND",
    "1m": "1-MINUTE", "5m": "5-MINUTE", "15m": "15-MINUTE", "1h": "1-HOUR",
}
TF_SOURCE = {
    "1s": "INTERNAL", "5s": "INTERNAL", "15s": "INTERNAL",
    "1m": "EXTERNAL", "5m": "EXTERNAL", "15m": "EXTERNAL", "1h": "EXTERNAL",
}

app = FastAPI()

# ── Redis connections ────────────────────────────────────────────────────────
# Market data: always from the live node (shared feed for both UI modes).
r = aioredis.from_url(
    LIVE_REDIS_URL, decode_responses=True,
    socket_timeout=None, socket_keepalive=True, max_connections=300,
)

# State Redis (portfolio / positions / strategy): one per UI mode.
_MODE_URLS = {"live": LIVE_REDIS_URL, "test": TEST_REDIS_URL}

r_state: dict[str, aioredis.Redis] = {
    m: aioredis.from_url(url, decode_responses=True,
                         socket_timeout=None, socket_keepalive=True, max_connections=100)
    for m, url in _MODE_URLS.items()
}

# Sync variants for thread-pool work (backtest serving + portfolio dedup).
r_sync: dict[str, syncredis.Redis] = {
    m: syncredis.from_url(url, decode_responses=True)
    for m, url in _MODE_URLS.items()
}

_BT_EXEC = ThreadPoolExecutor(max_workers=4, thread_name_prefix="backtest")
_PF_EXEC = ThreadPoolExecutor(max_workers=2, thread_name_prefix="portfolio")


# ── helpers ─────────────────────────────────────────────────────────────────
def symbol_from_trade_key(key: str) -> str | None:
    i = key.find(TRADES_INFIX)
    if i < 0:
        return None
    suffix = key[i + len(TRADES_INFIX):]
    venue, _, sym = suffix.partition(".")
    return f"{sym}.{venue}" if sym else None


async def find_trade_key(symbol: str) -> str | None:
    sym, _, venue = symbol.rpartition(".")
    pattern = f"*{TRADES_INFIX}{venue}.{sym}"
    async for key in r.scan_iter(match=pattern, count=1000):
        return key
    return None


async def find_hist_trade_key(symbol: str) -> str | None:
    sym, _, venue = symbol.rpartition(".")
    pattern = f"*{HIST_TRADES_INFIX}{venue}.{sym}"
    async for key in r.scan_iter(match=pattern, count=1000):
        return key
    return None


def _to_epoch_seconds(ts) -> float:
    if isinstance(ts, (int, float)):
        return float(ts) / 1e9
    s = str(ts)
    if s.isdigit():
        return int(s) / 1e9
    s = s.replace("Z", "+00:00")
    s = re.sub(r"(\.\d{6})\d+", r"\1", s)
    return datetime.fromisoformat(s).timestamp()


def _tick_from_payload(raw: str) -> dict | None:
    try:
        d = json.loads(raw)
        return {"t": _to_epoch_seconds(d["ts_event"]), "price": float(d["price"]),
                "qty": float(d.get("size", 0) or 0)}
    except (KeyError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _payload_value(fields: dict) -> str | None:
    if not fields:
        return None
    return fields.get("payload") or next(iter(fields.values()))


def bar_type_for(symbol: str, timeframe: str) -> str | None:
    spec = TIMEFRAMES.get(timeframe)
    src = TF_SOURCE.get(timeframe, "EXTERNAL")
    return f"{symbol}-{spec}-LAST-{src}" if spec else None


async def find_bar_key(bar_type: str) -> str | None:
    async for key in r.scan_iter(match=f"*{BARS_INFIX}{bar_type}", count=1000):
        return key
    return None


async def find_hist_bar_key(bar_type: str) -> str | None:
    async for key in r.scan_iter(match=f"*{HIST_BARS_INFIX}{bar_type}", count=1000):
        return key
    return None


def _bar_from_payload(raw: str) -> dict | None:
    try:
        d = json.loads(raw)
        return {"t": _to_epoch_seconds(d["ts_event"]),
                "o": float(d["open"]), "h": float(d["high"]),
                "l": float(d["low"]), "c": float(d["close"]),
                "v": float(d.get("volume", 0) or 0)}
    except (KeyError, ValueError, TypeError, json.JSONDecodeError):
        return None


# ── client state ─────────────────────────────────────────────────────────────
# Market data state (shared — market data is always from the live node).
clients: dict[WebSocket, str] = {}          # ws -> mode
symbol_clients: dict[str, set[WebSocket]] = defaultdict(set)
tail_tasks: dict[str, asyncio.Task] = {}
hist_tail_tasks: dict[str, asyncio.Task] = {}
bar_clients: dict[str, set[WebSocket]] = defaultdict(set)
bar_tail_tasks: dict[str, asyncio.Task] = {}
hist_bar_tail_tasks: dict[str, asyncio.Task] = {}
client_bars: dict[WebSocket, set[str]] = defaultdict(set)
ind_clients: dict[str, set[WebSocket]] = defaultdict(set)
ind_tail_tasks: dict[str, asyncio.Task] = {}
_pkt_counts: dict[str, int] = defaultdict(int)
packet_clients: set[WebSocket] = set()

# Mode-segregated client sets (for portfolio / order event broadcasts).
mode_clients: dict[str, set[WebSocket]] = {"live": set(), "test": set()}


# ── send helpers ─────────────────────────────────────────────────────────────
async def _safe_send(ws: WebSocket, payload: dict) -> None:
    try:
        await ws.send_text(json.dumps(payload))
    except Exception:
        pass


async def _safe_send_text(ws: WebSocket, text: str) -> None:
    try:
        await ws.send_text(text)
    except Exception:
        pass


async def _send_command(cmd: dict) -> None:
    await r.xadd(COMMAND_STREAM, {"json": json.dumps(cmd)}, maxlen=1000, approximate=True)


async def _send_strategy_command(cmd: dict, mode: str = "live") -> None:
    fields = {k: str(v) for k, v in cmd.items()}
    # Strategy commands always go to DB 0 — that's where binance_data.py listens.
    # The mode is passed as a field so the node knows which Redis to write results to.
    fields["mode"] = mode
    await r_state["live"].xadd(STRATEGY_CMDS_STREAM, fields, maxlen=100, approximate=True)


# ── market-data tails (shared, always live Redis) ────────────────────────────
async def _tail_symbol(symbol: str) -> None:
    key = None
    for _ in range(40):
        if not symbol_clients.get(symbol):
            return
        key = await find_trade_key(symbol)
        if key:
            break
        await asyncio.sleep(0.5)
    if key is None:
        return
    cursor = "$"
    try:
        while symbol_clients.get(symbol):
            try:
                results = await r.xread({key: cursor}, count=XREAD_COUNT, block=XREAD_BLOCK_MS)
            except RedisError:
                await asyncio.sleep(0.5)
                continue
            if not results:
                continue
            _, entries = results[0]
            if not entries:
                continue
            cursor = entries[-1][0]
            points = []
            for _id, fields in entries:
                pv = _payload_value(fields)
                pt = _tick_from_payload(pv) if pv else None
                if pt:
                    points.append(pt)
            if not points:
                continue
            _pkt_counts[symbol] += len(points)
            frame = {"type": "ticks", "symbol": symbol, "points": points}
            for ws in list(symbol_clients.get(symbol, ())):
                await _safe_send(ws, frame)
    except asyncio.CancelledError:
        pass
    finally:
        tail_tasks.pop(symbol, None)


def _ensure_tail(symbol: str) -> None:
    if symbol not in tail_tasks or tail_tasks[symbol].done():
        tail_tasks[symbol] = asyncio.create_task(_tail_symbol(symbol))


async def _tail_hist_symbol(symbol: str) -> None:
    sym, _, venue = symbol.rpartition(".")
    pattern = f"*{HIST_TRADES_INFIX}{venue}.{sym}"
    key = None
    cursor = "$"
    try:
        while symbol_clients.get(symbol):
            if key is None:
                async for k in r.scan_iter(match=pattern, count=1000):
                    key = k
                    break
                if key is None:
                    await asyncio.sleep(2.0)
                    continue
            try:
                results = await r.xread({key: cursor}, count=XREAD_COUNT, block=XREAD_BLOCK_MS)
            except RedisError:
                await asyncio.sleep(0.5)
                continue
            if not results:
                continue
            _, entries = results[0]
            if not entries:
                continue
            cursor = entries[-1][0]
            points = []
            for _id, fields in entries:
                pv = _payload_value(fields)
                pt = _tick_from_payload(pv) if pv else None
                if pt:
                    points.append(pt)
            if not points:
                continue
            frame = {"type": "hist_ticks", "symbol": symbol, "points": points}
            for ws in list(symbol_clients.get(symbol, ())):
                await _safe_send(ws, frame)
    except asyncio.CancelledError:
        pass
    finally:
        hist_tail_tasks.pop(symbol, None)


def _ensure_hist_tail(symbol: str) -> None:
    if symbol not in hist_tail_tasks or hist_tail_tasks[symbol].done():
        hist_tail_tasks[symbol] = asyncio.create_task(_tail_hist_symbol(symbol))


async def _send_history(ws: WebSocket, symbol: str, window: str) -> None:
    seen: set = set()
    ticks: list[dict] = []
    for finder, cap in ((find_hist_trade_key, HIST_FETCH_CAP), (find_trade_key, QUICK_HISTORY_COUNT)):
        key = await finder(symbol)
        if key is None:
            continue
        try:
            entries = await r.xrevrange(key, "+", "-", count=cap)
        except Exception:
            entries = []
        for _id, fields in entries:
            pv = _payload_value(fields)
            pt = _tick_from_payload(pv) if pv else None
            if pt:
                k = (pt["t"], pt["price"], pt["qty"])
                if k not in seen:
                    seen.add(k)
                    ticks.append(pt)
    ticks.sort(key=lambda d: d["t"])
    await _safe_send(ws, {"type": "history", "symbol": symbol, "ticks": ticks,
                          "complete": True, "window": window})


async def _tail_bars(symbol: str, bar_type: str) -> None:
    key = None
    cursor = "$"
    try:
        while bar_clients.get(bar_type):
            if key is None:
                key = await find_bar_key(bar_type)
                if key is None:
                    await asyncio.sleep(2.0)
                    continue
                recent = await r.xrevrange(key, count=2)
                cursor = recent[1][0] if len(recent) > 1 else "0-0"
            try:
                results = await r.xread({key: cursor}, count=XREAD_COUNT, block=XREAD_BLOCK_MS)
            except RedisError:
                await asyncio.sleep(0.5)
                continue
            if not results:
                continue
            _, entries = results[0]
            if not entries:
                continue
            cursor = entries[-1][0]
            points = []
            for _id, fields in entries:
                pv = _payload_value(fields)
                bar = _bar_from_payload(pv) if pv else None
                if bar:
                    points.append(bar)
            if not points:
                continue
            frame = {"type": "bars", "symbol": symbol, "bar_type": bar_type, "points": points}
            for ws in list(bar_clients.get(bar_type, ())):
                await _safe_send(ws, frame)
    except asyncio.CancelledError:
        pass
    finally:
        bar_tail_tasks.pop(bar_type, None)


async def _tail_hist_bars(symbol: str, bar_type: str) -> None:
    key = None
    cursor = "0-0"
    try:
        while bar_clients.get(bar_type):
            if key is None:
                key = await find_hist_bar_key(bar_type)
                if key is None:
                    await asyncio.sleep(2.0)
                    continue
            try:
                results = await r.xread({key: cursor}, count=XREAD_COUNT, block=XREAD_BLOCK_MS)
            except RedisError:
                await asyncio.sleep(0.5)
                continue
            if not results:
                continue
            _, entries = results[0]
            if not entries:
                continue
            cursor = entries[-1][0]
            points = []
            for _id, fields in entries:
                pv = _payload_value(fields)
                bar = _bar_from_payload(pv) if pv else None
                if bar:
                    points.append(bar)
            if not points:
                continue
            frame = {"type": "bars", "symbol": symbol, "bar_type": bar_type, "points": points}
            for ws in list(bar_clients.get(bar_type, ())):
                await _safe_send(ws, frame)
    except asyncio.CancelledError:
        pass
    finally:
        hist_bar_tail_tasks.pop(bar_type, None)


def _ensure_bar_tail(symbol: str, bar_type: str) -> None:
    if bar_type not in bar_tail_tasks or bar_tail_tasks[bar_type].done():
        bar_tail_tasks[bar_type] = asyncio.create_task(_tail_bars(symbol, bar_type))
    if bar_type not in hist_bar_tail_tasks or hist_bar_tail_tasks[bar_type].done():
        hist_bar_tail_tasks[bar_type] = asyncio.create_task(_tail_hist_bars(symbol, bar_type))


async def _send_bar_history(ws: WebSocket, symbol: str, bar_type: str) -> None:
    by_t: dict[float, dict] = {}
    for finder in (find_hist_bar_key, find_bar_key):
        key = await finder(bar_type)
        if key is None:
            continue
        try:
            entries = await r.xrevrange(key, "+", "-", count=HIST_FETCH_CAP)
        except Exception:
            entries = []
        seen: set = set()
        for _id, fields in entries:
            pv = _payload_value(fields)
            bar = _bar_from_payload(pv) if pv else None
            if not bar or bar["t"] in seen:
                continue
            seen.add(bar["t"])
            by_t[bar["t"]] = bar
    # In test mode, also merge backtest chart bars stored by the backtest runner.
    # These are stored as JSON arrays under test:chart:{bar_type} in the test Redis.
    ws_m = clients.get(ws, "live")
    if ws_m == "test":
        try:
            raw = r_sync["test"].get(f"test:chart:{bar_type}")
            if raw:
                for bar in json.loads(raw):
                    t = bar.get("t")
                    if t and t not in by_t:   # live bars take precedence for same timestamp
                        by_t[t] = bar
        except Exception:
            pass
    bars = [by_t[t] for t in sorted(by_t)]
    await _safe_send(ws, {"type": "bar_history", "symbol": symbol, "bar_type": bar_type, "bars": bars})


def _drop_bars_for_symbol(ws: WebSocket, symbol: str) -> None:
    prefix = f"{symbol}-"
    for bt in [b for b in client_bars.get(ws, set()) if b.startswith(prefix)]:
        client_bars[ws].discard(bt)
        bar_clients.get(bt, set()).discard(ws)


def _ind_fields_to_point(fields: dict) -> dict | None:
    try:
        ts = int(fields["ts"])
        tf = fields["tf"]
        snap = {k: float(v) for k, v in fields.items() if k not in ("ts", "tf")}
        return {"ts": ts, "tf": tf, **snap}
    except (KeyError, ValueError, TypeError):
        return None


async def _send_indicator_history(ws: WebSocket, symbol: str) -> None:
    key = f"{INDICATORS_PREFIX}{symbol}"
    try:
        entries = await r.xrange(key, "-", "+", count=HIST_FETCH_CAP)
    except Exception:
        return
    points = [p for (_id, f) in entries if (p := _ind_fields_to_point(f))]
    if points:
        await _safe_send(ws, {"type": "indicator_history", "symbol": symbol, "points": points})


async def _tail_indicators(symbol: str) -> None:
    key = f"{INDICATORS_PREFIX}{symbol}"
    cursor = "0-0"
    try:
        while ind_clients.get(symbol):
            try:
                results = await r.xread({key: cursor}, count=XREAD_COUNT, block=XREAD_BLOCK_MS)
            except RedisError:
                await asyncio.sleep(0.5)
                continue
            if not results:
                continue
            _, entries = results[0]
            if not entries:
                continue
            cursor = entries[-1][0]
            points = [p for (_id, f) in entries if (p := _ind_fields_to_point(f))]
            if not points:
                continue
            frame = {"type": "indicators", "symbol": symbol, "points": points}
            for ws in list(ind_clients.get(symbol, ())):
                await _safe_send(ws, frame)
    except asyncio.CancelledError:
        pass
    finally:
        ind_tail_tasks.pop(symbol, None)


def _ensure_indicator_tail(symbol: str) -> None:
    if symbol not in ind_tail_tasks or ind_tail_tasks[symbol].done():
        ind_tail_tasks[symbol] = asyncio.create_task(_tail_indicators(symbol))


# ── packet rate loop ─────────────────────────────────────────────────────────
async def _packet_loop() -> None:
    while True:
        await asyncio.sleep(1.0)
        if not packet_clients:
            _pkt_counts.clear()
            continue
        rates = dict(_pkt_counts)
        _pkt_counts.clear()
        frame = {"type": "packets", "t": time.time(),
                 "total": sum(rates.values()), "rates": rates}
        for ws in list(packet_clients):
            await _safe_send(ws, frame)


# ── portfolio loops (one per mode) ───────────────────────────────────────────
def _build_portfolio_sync(mode: str, last_sig):
    # Portfolio always lives in DB 0 (live node). Filter by strategy mode.
    rc = r_sync["live"]
    raw = rc.get("dashboard:portfolio")
    if not raw:
        return (None, None)
    try:
        states = rc.hgetall(STRATEGY_STATES_KEY)
    except Exception:
        states = {}
    try:
        mode_map = rc.hgetall("dashboard:strategy_modes")  # sid -> "live"|"test"
    except Exception:
        mode_map = {}
    sig = (raw, tuple(sorted(states.items())), tuple(sorted(mode_map.items())))
    if sig == last_sig:
        return (sig, None)
    try:
        snap = json.loads(raw)
    except Exception:
        return (sig, None)

    all_strats = snap.get("strategies", [])

    # Strategy list: always show ALL strategies in both tabs so users can
    # select and start them regardless of which mode they're currently in.
    # Positions/PnL are filtered to only show this mode's trades.
    pos_strats = {sid for sid, m in mode_map.items() if m == mode}
    untagged   = {sid for sid in all_strats if sid not in mode_map}
    if mode == "live":
        pos_visible = pos_strats | untagged   # live + untagged for positions
    else:
        pos_visible = pos_strats              # only tagged-test positions

    positions = [p for p in snap.get("positions", []) if p.get("strategy") in pos_visible]
    closed    = [p for p in snap.get("closed_positions", []) if p.get("strategy") in pos_visible]

    # Recompute PnL from filtered positions only.
    pnl: dict = {}
    for p in positions:
        d = pnl.setdefault(p.get("ccy") or "USDT", {"realized": 0.0, "unrealized": 0.0})
        d["realized"]   += p.get("realized", 0.0)
        d["unrealized"] += p.get("unrealized", 0.0)
    for p in closed:
        d = pnl.setdefault(p.get("ccy") or "USDT", {"realized": 0.0, "unrealized": 0.0})
        d["realized"] += p.get("realized", 0.0)
    for v in pnl.values():
        v["total"] = v["realized"] + v["unrealized"]

    metrics = {}
    for sid in all_strats:
        try:
            mraw = rc.get(f"dashboard:metrics:{sid}")
            if mraw:
                metrics[sid] = json.loads(mraw)
        except Exception:
            pass

    merged = {"positions": positions, "closed_positions": closed, "pnl": pnl}
    if mode == "test":
        merged = _merge_backtest_into_snap(merged)

    frame = {
        "type": "portfolio",
        **snap,
        "strategies": sorted(all_strats),
        "positions": merged["positions"],
        "closed_positions": merged["closed_positions"],
        "pnl": merged["pnl"],
        "metrics": metrics,
        "strategy_states": states,
    }
    return (sig, json.dumps(frame))


async def _portfolio_loop(mode: str) -> None:
    last_sig = None
    loop = asyncio.get_running_loop()
    while True:
        await asyncio.sleep(0.25)
        ws_set = mode_clients.get(mode, set())
        if not ws_set:
            continue
        try:
            sig, text = await loop.run_in_executor(
                _PF_EXEC, _build_portfolio_sync, mode, last_sig)
        except Exception:
            continue
        last_sig = sig
        if text is None:
            continue
        for ws in list(ws_set):
            await _safe_send_text(ws, text)


# ── order events tail (single stream, routed per mode) ───────────────────────
def _event_mode(strategy: str, mode_map: dict) -> str:
    # All strategies (live and test) run in the live node and write order events
    # to DB 0, tagged by strategy. Route each to its mode; untagged → live,
    # matching the portfolio filter (_build_portfolio_sync).
    return mode_map.get(strategy, "live")


async def _tail_order_events() -> None:
    rc = r_state["live"]   # all order events live in DB 0, tagged by strategy
    cursor = "$"
    while True:
        try:
            results = await rc.xread({ORDER_EVENTS_KEY: cursor}, count=100, block=1000)
            if not results:
                continue
            _, entries = results[0]
            if not entries:
                continue
            cursor = entries[-1][0]
            try:
                mode_map = await rc.hgetall("dashboard:strategy_modes")
            except Exception:
                mode_map = {}
            for _id, fields in entries:
                ev_mode = _event_mode(fields.get("strategy", ""), mode_map)
                frame = {
                    "type": "order_event",
                    "strategy": fields.get("strategy", ""),
                    "instrument": fields.get("instrument", ""),
                    "status": fields.get("status", ""),
                    "side": fields.get("side", ""),
                    "qty": fields.get("qty", ""),
                    "price": fields.get("price", ""),
                    "ts": float(fields.get("ts", 0)),
                }
                for ws in list(mode_clients.get(ev_mode, ())):
                    await _safe_send(ws, frame)
        except RedisError:
            await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            break
        except Exception:
            await asyncio.sleep(0.5)


@app.on_event("startup")
async def _startup() -> None:
    asyncio.create_task(_packet_loop())
    asyncio.create_task(_portfolio_loop("live"))
    asyncio.create_task(_portfolio_loop("test"))
    asyncio.create_task(_tail_order_events())


# ── WebSocket ────────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket, mode: str = Query("live")):
    if mode not in _MODE_URLS:
        mode = "live"
    await websocket.accept()
    clients[websocket] = mode
    mode_clients[mode].add(websocket)

    # Send current portfolio snapshot so a fresh client isn't blank.
    # Always read from DB 0 (live node); _build_portfolio_sync filters by mode.
    try:
        loop = asyncio.get_running_loop()
        sig, text = await loop.run_in_executor(_PF_EXEC, _build_portfolio_sync, mode, None)
        if text:
            await _safe_send(websocket, json.loads(text))
    except Exception:
        pass

    # Replay recent order events (all in DB 0, tagged by strategy). Only replay
    # the ones belonging to this client's mode so live/test queues stay separate.
    try:
        mode_map = await r_state["live"].hgetall("dashboard:strategy_modes")
    except Exception:
        mode_map = {}
    try:
        past = await r_state["live"].xrevrange(ORDER_EVENTS_KEY, count=500)
        for _id, fields in reversed(past):
            if _event_mode(fields.get("strategy", ""), mode_map) != mode:
                continue
            await _safe_send(websocket, {
                "type": "order_event",
                "strategy": fields.get("strategy", ""),
                "instrument": fields.get("instrument", ""),
                "status": fields.get("status", ""),
                "side": fields.get("side", ""),
                "qty": fields.get("qty", ""),
                "price": fields.get("price", ""),
                "ts": float(fields.get("ts", 0)),
            })
    except Exception:
        pass

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            action = msg.get("action")

            if action == "subscribe":
                window = msg.get("window") if msg.get("window") in WINDOWS else DEFAULT_WINDOW
                for sym in [s for s in msg.get("symbols", []) if isinstance(s, str)]:
                    clients.setdefault(websocket, mode)
                    symbol_clients[sym].add(websocket)
                    ind_clients[sym].add(websocket)
                    await _send_command({"action": "subscribe", "instrument_id": sym})
                    await _send_history(websocket, sym, window)
                    await _send_indicator_history(websocket, sym)
                    _ensure_tail(sym)
                    _ensure_hist_tail(sym)
                    _ensure_indicator_tail(sym)
                    for tf in TIMEFRAMES:
                        bt = bar_type_for(sym, tf)
                        if bt:
                            bar_clients[bt].add(websocket)
                            client_bars[websocket].add(bt)
                            await _send_command({"action": "subscribe", "instrument_id": sym, "bar_type": bt})
                            await _send_bar_history(websocket, sym, bt)
                            _ensure_bar_tail(sym, bt)

            elif action == "set_window":
                sym = msg.get("symbol")
                window = msg.get("window") if msg.get("window") in WINDOWS else DEFAULT_WINDOW
                if isinstance(sym, str) and websocket in symbol_clients.get(sym, ()):
                    await _send_history(websocket, sym, window)

            elif action == "refresh_bars":
                sym = msg.get("symbol")
                bt = msg.get("bar_type")
                if isinstance(sym, str) and isinstance(bt, str) and websocket in bar_clients.get(bt, ()):
                    await _send_bar_history(websocket, sym, bt)

            elif action == "refresh_indicators":
                sym = msg.get("symbol")
                if isinstance(sym, str) and websocket in ind_clients.get(sym, ()):
                    await _send_indicator_history(websocket, sym)

            elif action == "unsubscribe":
                for sym in [s for s in msg.get("symbols", []) if isinstance(s, str)]:
                    symbol_clients.get(sym, set()).discard(websocket)
                    ind_clients.get(sym, set()).discard(websocket)
                    _drop_bars_for_symbol(websocket, sym)

            elif action in STRATEGY_ACTIONS:
                ws_mode = clients.get(websocket, "live")
                await _send_strategy_command(msg, ws_mode)

    except WebSocketDisconnect:
        pass
    finally:
        mode_clients.get(mode, set()).discard(websocket)
        clients.pop(websocket, None)
        for sym in list(symbol_clients):
            symbol_clients[sym].discard(websocket)
            ind_clients.get(sym, set()).discard(websocket)
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


# ── REST ─────────────────────────────────────────────────────────────────────
@app.get("/api/symbols")
async def symbols():
    out = set()
    async for key in r.scan_iter(match="*:instruments:*", count=2000):
        _, _, iid = key.partition(":instruments:")
        if iid:
            out.add(iid)
    if not out:
        async for key in r.scan_iter(match=f"*{TRADES_INFIX}*", count=1000):
            sym = symbol_from_trade_key(key)
            if sym:
                out.add(sym)
    return JSONResponse(sorted(out))


@app.post("/api/command")
async def command(cmd: dict, mode: str = "live"):
    if not isinstance(cmd, dict) or "action" not in cmd:
        return JSONResponse({"ok": False, "error": "missing 'action'"}, status_code=400)
    if cmd.get("action") in STRATEGY_ACTIONS:
        await _send_strategy_command(cmd, mode)
    else:
        await _send_command(cmd)
    return JSONResponse({"ok": True, "sent": cmd})


@app.post("/api/{mode}/command")
async def command_mode(mode: str, cmd: dict):
    if not isinstance(cmd, dict) or "action" not in cmd:
        return JSONResponse({"ok": False, "error": "missing 'action'"}, status_code=400)
    if cmd.get("action") in STRATEGY_ACTIONS:
        await _send_strategy_command(cmd, mode)
    else:
        await _send_command(cmd)
    return JSONResponse({"ok": True, "sent": cmd})


async def _bt_json(fn, *args) -> Response:
    loop = asyncio.get_running_loop()
    body = await loop.run_in_executor(_BT_EXEC, fn, *args)
    return Response(content=body, media_type="application/json")


# ── Account equity (per mode) ────────────────────────────────────────────────
def _account_equity_sync(mode: str) -> str:
    rc_live = r_sync["live"]   # DB 0 — live node state
    rc_test = r_sync["test"]   # DB 1 — backtest results

    if mode == "test":
        # Prepend backtest equity curve (DB 1) then live test points (DB 0).
        # The live test points start from ACCOUNT_START=100000 since ControlActor
        # doesn't know the backtest ending NAV. We offset them so the chart is
        # continuous: live_nav_adjusted = bt_end_nav + (live_nav - ACCOUNT_START).
        bt_raw = rc_test.get("dashboard:backtest:equity")
        bt_pts = json.loads(bt_raw) if bt_raw else []
        live_entries = rc_live.lrange("dashboard:equity:test", 0, -1)
        live_pts = [json.loads(e) for e in live_entries]
        bt_end_nav = float(rc_test.get("dashboard:backtest:equity:end_nav") or 100_000)
        nav_offset = bt_end_nav - 100_000.0
        if bt_pts and live_pts:
            bt_end_ts = bt_pts[-1]["ts"]
            live_pts = [p for p in live_pts if p["ts"] > bt_end_ts]
        # Apply offset so live portion is continuous with backtest
        if nav_offset != 0.0:
            live_pts = [{**p, "nav": round(p["nav"] + nav_offset, 4)} for p in live_pts]
        all_pts = bt_pts + live_pts
        peak_val = max((p["nav"] for p in all_pts), default=100_000.0)
        entries = [json.dumps(p) for p in all_pts]
        peak = str(peak_val)
    else:
        entries = rc_live.lrange("dashboard:equity:live", 0, -1)
        peak = rc_live.get("dashboard:equity:live:peak") or "null"

    MAXP = 5000
    n = len(entries)
    if n > MAXP:
        stride = n // MAXP + 1
        kept = entries[::stride]
        if kept[-1] != entries[-1]:
            kept.append(entries[-1])
        entries = kept
    return '{"points": [%s], "peak": %s}' % (",".join(entries), peak)


@app.get("/api/{mode}/account/equity")
async def account_equity(mode: str):
    if mode not in _MODE_URLS:
        mode = "live"
    return await _bt_json(_account_equity_sync, mode)


# ── Backtest meta / positions (test mode only) ────────────────────────────────
BACKTEST_PREFIX = "dashboard:backtest"


def _bt_meta_sync(mode: str) -> str:
    return r_sync[mode].get(f"{BACKTEST_PREFIX}:meta") or '{"status": "none"}'


def _bt_positions_sync(mode: str) -> str:
    return (r_sync[mode].get(f"{BACKTEST_PREFIX}:positions")
            or '{"positions": [], "closed_positions": [], "pnl": {}}')


@app.get("/api/{mode}/backtest/meta")
async def backtest_meta(mode: str):
    if mode not in _MODE_URLS:
        mode = "live"
    return await _bt_json(_bt_meta_sync, mode)


@app.get("/api/{mode}/backtest/positions")
async def backtest_positions(mode: str):
    if mode not in _MODE_URLS:
        mode = "live"
    return await _bt_json(_bt_positions_sync, mode)


def _positions_sync(mode: str) -> str:
    """Mode-filtered portfolio: live positions merged with backtest history."""
    _, text = _build_portfolio_sync(mode, None)
    live_snap = json.loads(text) if text else {}
    positions = list(live_snap.get("positions", []))
    closed    = list(live_snap.get("closed_positions", []))
    pnl       = dict(live_snap.get("pnl", {}))

    # For test mode, also merge backtest closed positions from the engine run.
    if mode == "test":
        try:
            raw = r_sync["test"].get("dashboard:backtest:positions")
            if raw:
                bt = json.loads(raw)
                bt_closed = bt.get("closed_positions", [])
                bt_open   = bt.get("positions", [])
                # Merge: backtest positions first (oldest), live on top
                existing_ids = {p.get("id") for p in closed}
                for p in bt_closed:
                    if p.get("id") not in existing_ids:
                        closed.append(p)
                for p in bt_open:
                    if p.get("id") not in existing_ids:
                        positions.append(p)
                # Recompute merged PnL
                for p in bt_closed:
                    d = pnl.setdefault(p.get("ccy") or "USDT", {"realized": 0.0, "unrealized": 0.0, "total": 0.0})
                    d["realized"] += p.get("realized", 0.0)
                    d["total"]     = d["realized"] + d["unrealized"]
        except Exception:
            pass

    closed.sort(key=lambda c: c.get("ts_closed", 0), reverse=True)
    return json.dumps({"positions": positions, "closed_positions": closed, "pnl": pnl})


def _merge_backtest_into_snap(snap: dict) -> dict:
    """Merge dashboard:backtest:positions (test Redis) into a portfolio snapshot."""
    try:
        raw = r_sync["test"].get("dashboard:backtest:positions")
        if not raw:
            return snap
        bt = json.loads(raw)
        existing_ids = {p.get("id") for p in snap.get("closed_positions", [])}
        extra_closed = [p for p in bt.get("closed_positions", []) if p.get("id") not in existing_ids]
        extra_open   = [p for p in bt.get("positions", [])         if p.get("id") not in existing_ids]
        if not extra_closed and not extra_open:
            return snap
        positions = snap.get("positions", []) + extra_open
        closed    = snap.get("closed_positions", []) + extra_closed
        closed.sort(key=lambda c: c.get("ts_closed", 0), reverse=True)
        pnl = dict(snap.get("pnl", {}))
        for p in extra_closed:
            d = pnl.setdefault(p.get("ccy") or "USDT", {"realized": 0.0, "unrealized": 0.0})
            d["realized"] += p.get("realized", 0.0)
        for p in extra_open:
            d = pnl.setdefault(p.get("ccy") or "USDT", {"realized": 0.0, "unrealized": 0.0})
            d["unrealized"] += p.get("unrealized", 0.0)
        for v in pnl.values():
            v["total"] = v["realized"] + v["unrealized"]
        return {**snap, "positions": positions, "closed_positions": closed, "pnl": pnl}
    except Exception:
        return snap


@app.get("/api/{mode}/positions")
async def positions(mode: str):
    if mode not in _MODE_URLS:
        mode = "live"
    return await _bt_json(_positions_sync, mode)


# ── Test-mode chart bars: refresh bar history for a symbol after backtest ─────
def _refresh_chart_bars_sync(mode: str, symbol: str) -> str:
    """Return backtest chart bars (all timeframes) + EMA indicators for a symbol.

    Response shape: {"bars": [{tf, bar_type, bars}], "indicators": [{ts, tf, fast_ema, slow_ema}]}
    """
    rc = r_sync["test"]   # backtest data always in DB 1
    bars_result = []
    for tf, spec in [("1m", "1-MINUTE"), ("5m", "5-MINUTE"), ("15m", "15-MINUTE"), ("1h", "1-HOUR")]:
        src = "EXTERNAL"
        bar_type = f"{symbol}-{spec}-LAST-{src}"
        key = f"test:chart:{bar_type}"
        raw = rc.get(key)
        if raw:
            try:
                bars_result.append({"tf": tf, "bar_type": bar_type, "bars": json.loads(raw)})
            except Exception:
                pass

    # Backtest EMA indicators (written by backtest_runner as {ts, fast_ema, slow_ema} lists)
    indicators = []
    raw_ind = rc.get(f"dashboard:backtest:indicators:{symbol}")
    if raw_ind:
        try:
            for pt in json.loads(raw_ind):
                pt["tf"] = "1-MINUTE-LAST"   # backtest always runs on 1m bars
                indicators.append(pt)
        except Exception:
            pass

    return json.dumps({"bars": bars_result, "indicators": indicators})


@app.get("/api/{mode}/chart_bars/{symbol}")
async def chart_bars(mode: str, symbol: str):
    if mode not in _MODE_URLS:
        mode = "live"
    return await _bt_json(_refresh_chart_bars_sync, mode, symbol)


@app.get("/", response_class=HTMLResponse)
async def index():
    with open(HTML_PATH, "r", encoding="utf-8") as fh:
        return HTMLResponse(fh.read())
