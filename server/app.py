"""Dashboard server — pure pass-through between Nautilus' native Redis streams
and the browser. Performs NO calculations.

Nautilus' MessageBus (configured with database=redis + encoding="json" +
stream_per_topic) already publishes every subscribed trade/quote/bar to Redis:

    key     : {streams_prefix}:trader-{trader_id}:{topic}
              e.g. stream:trader-TESTER-001:data.trades.BINANCE_FUTURES.ETHUSDT-PERP
    topics  : data.trades.{venue}.{symbol}
              data.quotes.{venue}.{symbol}
              data.bars.{bar_type}
    payload : one stream field holding the object's JSON to_dict(), e.g. a
              TradeTick -> {type, instrument_id, price, size, ts_event, ...}

This server discovers those streams by topic pattern (so it doesn't care about
the exact key prefix), decodes the JSON payload, and forwards. The dashboard's
symbol vocabulary is the instrument id (symbol.venue). MA / metric panels stay
blank because nothing produces those — intentional.
"""

import os
import re
import json
import asyncio
import time
from datetime import datetime
from collections import defaultdict

import redis.asyncio as aioredis
from redis.exceptions import RedisError
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
HTML_PATH = os.path.join(os.path.dirname(__file__), "dashboard.html")

QUICK_HISTORY_COUNT = 500
HIST_FETCH_CAP = 50000   # max backfilled entries pulled into history (dense tick ranges)
XREAD_BLOCK_MS = 5000
XREAD_COUNT = 500
DEFAULT_WINDOW = "10m"
WINDOWS = {"2m", "10m", "30m", "1h", "4h", "1d", "3d"}
TRADES_INFIX = ":data.trades."
COMMAND_STREAM = "dashboard:commands"   # ControlActor polls this (see control_actor.py)
STRATEGY_CMDS_STREAM = "dashboard:strategy_cmds"
STRATEGY_STATES_KEY = "dashboard:strategy_states"
INDICATORS_PREFIX = "dashboard:indicators:"
ORDER_EVENTS_KEY = "dashboard:order_events"
STRATEGY_ACTIONS = {"start_strategy", "stop_strategy", "close_strategy"}

app = FastAPI()
# socket_timeout=None: blocking XREAD must not be killed by a read timeout when
# data is briefly idle (that was crashing the tail tasks).
r = aioredis.from_url(REDIS_URL, decode_responses=True, socket_timeout=None, socket_keepalive=True)


async def _send_command(cmd: dict) -> None:
    """Publish a command to the engine's command stream; the ControlActor polls it."""
    await r.xadd(COMMAND_STREAM, {"json": json.dumps(cmd)}, maxlen=1000, approximate=True)


async def _send_strategy_command(cmd: dict) -> None:
    """Publish a strategy control command; the strategy manager thread in binance_data.py polls it."""
    fields = {k: str(v) for k, v in cmd.items()}
    await r.xadd(STRATEGY_CMDS_STREAM, fields, maxlen=100, approximate=True)


# ── topic <-> symbol helpers ────────────────────────────────────────────────
def symbol_from_trade_key(key: str) -> str | None:
    i = key.find(TRADES_INFIX)
    if i < 0:
        return None
    suffix = key[i + len(TRADES_INFIX):]          # 'BINANCE_FUTURES.ETHUSDT-PERP'
    venue, _, sym = suffix.partition(".")
    return f"{sym}.{venue}" if sym else None


async def find_trade_key(symbol: str) -> str | None:
    sym, _, venue = symbol.rpartition(".")
    pattern = f"*{TRADES_INFIX}{venue}.{sym}"
    async for key in r.scan_iter(match=pattern, count=1000):
        return key
    return None


# request_trade_ticks (tick backfill) responses stream natively to:
#   trader-{id}:stream:historical.data.trades.{venue}.{symbol}
HIST_TRADES_INFIX = ":historical.data.trades."


async def find_hist_trade_key(symbol: str) -> str | None:
    sym, _, venue = symbol.rpartition(".")
    pattern = f"*{HIST_TRADES_INFIX}{venue}.{sym}"
    async for key in r.scan_iter(match=pattern, count=1000):
        return key
    return None


def _to_epoch_seconds(ts) -> float:
    """ts_event is either ns-int or an ISO8601 string (timestamps_as_iso8601)."""
    if isinstance(ts, (int, float)):
        return float(ts) / 1e9
    s = str(ts)
    if s.isdigit():
        return int(s) / 1e9
    # ISO8601: 'Z' -> '+00:00'; truncate >6 fractional digits (ns) for fromisoformat
    s = s.replace("Z", "+00:00")
    s = re.sub(r"(\.\d{6})\d+", r"\1", s)
    return datetime.fromisoformat(s).timestamp()


def _tick_from_payload(raw_field_value: str) -> dict | None:
    try:
        d = json.loads(raw_field_value)
        return {
            "t": _to_epoch_seconds(d["ts_event"]),   # epoch seconds (true event time)
            "price": float(d["price"]),
            "qty": float(d.get("size", 0) or 0),
        }
    except (KeyError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _payload_value(fields: dict) -> str | None:
    """Nautilus writes the payload under one field; be tolerant of its name."""
    if not fields:
        return None
    return fields.get("payload") or next(iter(fields.values()))


# ── candles / timeframes ─────────────────────────────────────────────────────
# Dashboard timeframe -> Nautilus bar step+unit. "tick" = raw line (no bar).
# Bars are aggregated INTERNAL (Nautilus builds standard Bars from trades).
TIMEFRAMES = {
    "1s": "1-SECOND", "5s": "5-SECOND", "15s": "15-SECOND",
    "1m": "1-MINUTE", "5m": "5-MINUTE", "15m": "15-MINUTE", "1h": "1-HOUR",
}
BARS_INFIX = ":data.bars."

TF_SOURCE = {
    "1s": "INTERNAL", "5s": "INTERNAL", "15s": "INTERNAL",
    "1m": "EXTERNAL", "5m": "EXTERNAL", "15m": "EXTERNAL", "1h": "EXTERNAL",
}


def bar_type_for(symbol: str, timeframe: str) -> str | None:
    spec = TIMEFRAMES.get(timeframe)
    src = TF_SOURCE.get(timeframe, "EXTERNAL")
    return f"{symbol}-{spec}-LAST-{src}" if spec else None


async def find_bar_key(bar_type: str) -> str | None:
    async for key in r.scan_iter(match=f"*{BARS_INFIX}{bar_type}", count=1000):
        return key
    return None


# request_bars (backfill) responses are streamed natively to a SEPARATE topic:
#   trader-{id}:stream:historical.data.bars.{bar_type}
# so backfilled candles live here, while live aggregated candles live under
# data.bars. We merge both for history.
HIST_BARS_INFIX = ":historical.data.bars."


async def find_hist_bar_key(bar_type: str) -> str | None:
    async for key in r.scan_iter(match=f"*{HIST_BARS_INFIX}{bar_type}", count=1000):
        return key
    return None


def _bar_from_payload(raw: str) -> dict | None:
    """Decode a Nautilus Bar payload -> {t, o, h, l, c, v}."""
    try:
        d = json.loads(raw)
        return {
            "t": _to_epoch_seconds(d["ts_event"]),
            "o": float(d["open"]), "h": float(d["high"]),
            "l": float(d["low"]), "c": float(d["close"]),
            "v": float(d.get("volume", 0) or 0),
        }
    except (KeyError, ValueError, TypeError, json.JSONDecodeError):
        return None


# ── client / subscription state ─────────────────────────────────────────────
clients: dict[WebSocket, set[str]] = {}
symbol_clients: dict[str, set[WebSocket]] = defaultdict(set)
tail_tasks: dict[str, asyncio.Task] = {}
_pkt_counts: dict[str, int] = defaultdict(int)
packet_clients: set[WebSocket] = set()

# Bars: keyed by bar_type string (per symbol+timeframe).
bar_clients: dict[str, set[WebSocket]] = defaultdict(set)
bar_tail_tasks: dict[str, asyncio.Task] = {}
client_bars: dict[WebSocket, set[str]] = defaultdict(set)   # ws -> bar_types it watches

# Historical trade ticks (backfill arrivals) — tailed separately so they stream
# in automatically without the dashboard needing to poll after a backfill request.
hist_tail_tasks: dict[str, asyncio.Task] = {}

# Per-instrument indicator snapshots published by ControlActor.
ind_clients: dict[str, set[WebSocket]] = defaultdict(set)
ind_tail_tasks: dict[str, asyncio.Task] = {}


async def _safe_send(ws: WebSocket, payload: dict) -> None:
    try:
        await ws.send_text(json.dumps(payload))
    except Exception:
        pass


async def _tail_symbol(symbol: str) -> None:
    # For a freshly-requested symbol the node may not have created the stream
    # yet (it just got the subscribe command). Wait for it to appear.
    key = None
    for _ in range(40):  # up to ~20s
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
    """Tail historical.data.trades.* and stream new backfill ticks as hist_ticks messages.

    The key may not exist until the first backfill runs, so we poll for it
    indefinitely (every 2s) rather than giving up after N retries.
    Starting the cursor at '$' ensures we only forward entries that arrive
    AFTER the initial _send_history call (which already covers older entries).
    """
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
    # Merge backfilled (historical.data.trades) + live (data.trades) ticks,
    # dedup by (time, price, qty), sort ascending.
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
    await _safe_send(ws, {
        "type": "history", "symbol": symbol, "ticks": ticks,
        "complete": True, "window": window,
    })


async def _tail_bars(symbol: str, bar_type: str) -> None:
    # Wait for the bar stream (the node may have just been told to aggregate it).
    key = None
    for _ in range(60):  # up to ~30s (1m bars take a minute for the first one)
        if not bar_clients.get(bar_type):
            return
        key = await find_bar_key(bar_type)
        if key:
            break
        await asyncio.sleep(0.5)
    if key is None:
        return
    # Start from the beginning of the stream so backfill bars already written
    # before this tail task started are not missed. Duplicates with bar_history
    # are harmless — the dashboard deduplicates by timestamp.
    cursor = "0-0"
    try:
        while bar_clients.get(bar_type):
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


def _ensure_bar_tail(symbol: str, bar_type: str) -> None:
    if bar_type not in bar_tail_tasks or bar_tail_tasks[bar_type].done():
        bar_tail_tasks[bar_type] = asyncio.create_task(_tail_bars(symbol, bar_type))


async def _send_bar_history(ws: WebSocket, symbol: str, bar_type: str) -> None:
    # Merge backfilled (historical.data.bars) + live (data.bars) candles,
    # dedup by event time, sort ascending.
    by_t: dict[float, dict] = {}
    for finder in (find_hist_bar_key, find_bar_key):
        key = await finder(bar_type)
        if key is None:
            continue
        try:
            entries = await r.xrevrange(key, "+", "-", count=HIST_FETCH_CAP)   # newest first
        except Exception:
            entries = []
        # Within one stream, keep the NEWEST entry per bucket (first seen, since
        # xrevrange is newest-first) — a backfill may publish a bucket twice
        # (partial then complete); the complete one is newest and must win.
        seen: set = set()
        for _id, fields in entries:
            pv = _payload_value(fields)
            bar = _bar_from_payload(pv) if pv else None
            if not bar or bar["t"] in seen:
                continue
            seen.add(bar["t"])
            by_t[bar["t"]] = bar   # later finder (live) overrides historical for same bucket
    bars = [by_t[t] for t in sorted(by_t)]
    await _safe_send(ws, {
        "type": "bar_history", "symbol": symbol, "bar_type": bar_type, "bars": bars,
    })


def _drop_bars_for_symbol(ws: WebSocket, symbol: str) -> None:
    """Remove ws from any bar_type belonging to this symbol (timeframe switch/leave)."""
    prefix = f"{symbol}-"
    for bt in [b for b in client_bars.get(ws, set()) if b.startswith(prefix)]:
        client_bars[ws].discard(bt)
        bar_clients.get(bt, set()).discard(ws)


def _ind_fields_to_point(fields: dict) -> dict | None:
    try:
        ts  = int(fields["ts"])
        tf  = fields["tf"]
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
    # Start from 0-0 (stream beginning) so entries written before or during the
    # subscribe roundtrip are not missed — same pattern as _tail_bars.
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


async def _portfolio_frame() -> dict | None:
    # Build the portfolio frame: account snapshot (ControlActor) + each running
    # strategy's free-form metrics (dashboard:metrics:{id}, strategy-defined).
    try:
        raw = await r.get("dashboard:portfolio")
    except Exception:
        return None
    if not raw:
        return None
    try:
        snap = json.loads(raw)
    except Exception:
        return None
    metrics: dict = {}
    for sid in snap.get("strategies", []):
        try:
            mraw = await r.get(f"dashboard:metrics:{sid}")
            if mraw:
                metrics[sid] = json.loads(mraw)
        except Exception:
            pass
    try:
        strategy_states = await r.hgetall(STRATEGY_STATES_KEY)
    except Exception:
        strategy_states = {}
    return {"type": "portfolio", **snap, "metrics": metrics, "strategy_states": strategy_states}


async def _tail_order_events() -> None:
    cursor = "$"
    while True:
        try:
            results = await r.xread({ORDER_EVENTS_KEY: cursor}, count=100, block=1000)
            if not results:
                continue
            _, entries = results[0]
            if not entries:
                continue
            cursor = entries[-1][0]
            for _id, fields in entries:
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
                for ws in list(clients):
                    await _safe_send(ws, frame)
        except RedisError:
            await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            break
        except Exception:
            await asyncio.sleep(0.5)


async def _portfolio_loop() -> None:
    # Broadcast the portfolio frame to all clients, only when it changes.
    last = None
    while True:
        await asyncio.sleep(0.25)
        if not clients:
            continue
        frame = await _portfolio_frame()
        if frame is None:
            continue
        key = json.dumps(frame, sort_keys=True)
        if key == last:
            continue
        last = key
        for ws in list(clients):
            await _safe_send(ws, frame)


@app.on_event("startup")
async def _startup() -> None:
    asyncio.create_task(_packet_loop())
    asyncio.create_task(_portfolio_loop())
    asyncio.create_task(_tail_order_events())


# ── WebSocket: market data ──────────────────────────────────────────────────
@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients[websocket] = set()
    # Send the current portfolio frame once so a fresh client isn't blank
    # until the next change (the broadcast loop only sends on change).
    frame = await _portfolio_frame()
    if frame:
        await _safe_send(websocket, frame)
    # Replay recent order events so the client sees history from before it connected.
    try:
        past = await r.xrevrange(ORDER_EVENTS_KEY, count=500)
        for _id, fields in reversed(past):
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
                    clients[websocket].add(sym)
                    symbol_clients[sym].add(websocket)
                    ind_clients[sym].add(websocket)
                    await _send_command({"action": "subscribe", "instrument_id": sym})
                    await _send_history(websocket, sym, window)
                    await _send_indicator_history(websocket, sym)
                    _ensure_tail(sym)
                    _ensure_hist_tail(sym)
                    _ensure_indicator_tail(sym)
                    # Subscribe + tail ALL timeframes immediately so switching
                    # the candle slider is instant (no server roundtrip needed).
                    for tf, _spec in TIMEFRAMES.items():
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
                # Re-pull bar history (hist + live merged) after an EXTERNAL backfill,
                # whose request_bars output lands in historical.data.bars (not tailed).
                sym = msg.get("symbol")
                bt = msg.get("bar_type")
                if isinstance(sym, str) and isinstance(bt, str) and websocket in bar_clients.get(bt, ()):
                    await _send_bar_history(websocket, sym, bt)

            elif action == "refresh_indicators":
                # Re-pull full indicator history after a backfill so the browser
                # receives entries written before the live tail started.
                sym = msg.get("symbol")
                if isinstance(sym, str) and websocket in ind_clients.get(sym, ()):
                    await _send_indicator_history(websocket, sym)

            elif action == "unsubscribe":
                for sym in [s for s in msg.get("symbols", []) if isinstance(s, str)]:
                    clients[websocket].discard(sym)
                    symbol_clients.get(sym, set()).discard(websocket)
                    ind_clients.get(sym, set()).discard(websocket)
                    _drop_bars_for_symbol(websocket, sym)

    except WebSocketDisconnect:
        pass
    finally:
        for sym in clients.pop(websocket, set()):
            symbol_clients.get(sym, set()).discard(websocket)
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


# ── REST ────────────────────────────────────────────────────────────────────
@app.get("/api/symbols")
async def symbols():
    """Full loaded universe from the instrument cache (key ...:instruments:{id}),
    so the dashboard can browse/subscribe to any symbol load_all brought in.
    Falls back to active trade streams if the instrument cache is empty."""
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
async def command(cmd: dict):
    """Dashboard -> engine commands (subscribe / unsubscribe / backfill).

    Published to the COMMAND_STREAM Redis stream; the node's ControlActor polls
    it and dispatches. Note: 'backfill' won't fully work until bars are
    re-enabled (the BinanceBar serialization issue) — the command is delivered,
    but request_bars on -EXTERNAL bars would crash the engine.
    """
    if not isinstance(cmd, dict) or "action" not in cmd:
        return JSONResponse({"ok": False, "error": "missing 'action'"}, status_code=400)
    if cmd.get("action") in STRATEGY_ACTIONS:
        await _send_strategy_command(cmd)
    else:
        await _send_command(cmd)
    return JSONResponse({"ok": True, "sent": cmd})


@app.get("/", response_class=HTMLResponse)
async def index():
    with open(HTML_PATH, "r", encoding="utf-8") as fh:
        return HTMLResponse(fh.read())
