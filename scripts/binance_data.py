"""Trading node: Binance market data, sandbox execution, dashboard control.

Run via launch.bat (or `.venv\\Scripts\\python.exe scripts\\binance_data.py`).

Strategy commands arrive on the ``dashboard:strategy_cmds`` Redis stream. Each is
executed on the node's event loop and answered with a notification carrying the
command's id: success with what happened, or an error with the reason and a
Retry payload. Nothing is written optimistically; the dashboard shows the state
each strategy actually reports (``DashboardStrategy.status``).
"""
import csv
import json
import os
import threading
import time
import traceback
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import redis
from nautilus_trader.adapters.binance import BinanceLiveDataClientFactory
from nautilus_trader.adapters.sandbox.factory import SandboxLiveExecClientFactory
from nautilus_trader.live.node import TradingNode
from nautilus_trader.model.identifiers import InstrumentId

from backtest_runner import run_backtest
from trading import redis_io as K
from trading.actors.control_actor import ControlActor, ControlActorConfig
from trading.configs.binance_config import BINANCE_FUTURES, BINANCE_SPOT, config_node
from trading.strategies.EMACross import EMACross, EMACrossConfig
from trading.strategies.EMACrossShortTest import EMACrossSARConfig, EMACrossStopReverse

EXPORT_DIR = os.getenv("POSITION_EXPORT_DIR", "exports")
_IST = timezone(timedelta(hours=5, minutes=30))


def perps(symbols):
    return tuple(InstrumentId.from_str(f"{s}-PERP.{BINANCE_FUTURES}") for s in symbols)


# Top-20 USDT perpetuals by open interest.
PERP_INSTRUMENTS = perps([
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT",
    "LINKUSDT", "SUIUSDT", "DOTUSDT", "NEARUSDT", "APTUSDT", "LTCUSDT", "UNIUSDT", "ATOMUSDT",
    "INJUSDT", "AAVEUSDT", "ARBUSDT", "RENDERUSDT",
])

PERP_INSTRUMENTS_SAR = perps([
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT", "TRXUSDT", "HYPEUSDT", "DOGEUSDT", "ZECUSDT",
    "LABUSDT", "XLMUSDT", "XMRUSDT", "CCUSDT", "LINKUSDT", "ADAUSDT", "BCHUSDT", "LTCUSDT", "HBARUSDT",
    "SUIUSDT", "AVAXUSDT", "1000SHIBUSDT", "NEARUSDT", "TAOUSDT", "WLFIUSDT", "PAXGUSDT", "UNIUSDT",
    "ASTERUSDT", "WLDUSDT", "ONDOUSDT", "DOTUSDT", "AAVEUSDT", "SKYUSDT", "MUSDT", "ETCUSDT",
    "MORPHOUSDT", "DEXEUSDT", "1000PEPEUSDT", "QNTUSDT", "ATOMUSDT", "RENDERUSDT", "POLUSDT", "KASUSDT",
    "ALGOUSDT", "ENAUSDT", "JUPUSDT", "JSTUSDT", "BEATUSDT", "VVVUSDT", "FILUSDT", "NIGHTUSDT", "APTUSDT",
    "ARBUSDT", "AEROUSDT", "INJUSDT", "DASHUSDT", "CAKEUSDT", "TRUMPUSDT", "VETUSDT", "FETUSDT",
    "PENGUUSDT", "SEIUSDT", "JTOUSDT", "1000BONKUSDT", "1000LUNCUSDT", "ETHFIUSDT", "VIRTUALUSDT",
    "KITEUSDT", "TIAUSDT", "SUNUSDT", "SKYAIUSDT", "STXUSDT", "SPXUSDT", "CRVUSDT", "XPLUSDT", "GRASSUSDT",
    "GWEIUSDT", "PYTHUSDT", "XTZUSDT", "OPUSDT", "MONUSDT", "CFXUSDT", "JASMYUSDT", "BSVUSDT", "BUSDT",
    "1000FLOKIUSDT", "PENDLEUSDT", "VELVETUSDT", "LDOUSDT", "ZROUSDT", "KAIAUSDT", "AKTUSDT", "GRTUSDT",
    "STRKUSDT", "CHZUSDT", "UBUSDT", "AXSUSDT", "IOTAUSDT", "ENSUSDT", "EIGENUSDT", "COMPUSDT",
])

# strategy_id is the name and order_id_tag the suffix; the StrategyId is
# "{strategy_id}-{order_id_tag}". Tags must be unique: without them Nautilus
# silently renamed "EMACross-001" to "EMACross-000" at registration.
STRATEGIES = [
    EMACross(EMACrossConfig(
        strategy_id="EMACross", order_id_tag="000",
        instrument_ids=PERP_INSTRUMENTS, trade_usd=Decimal("2000"),
        bar_spec="5-SECOND-LAST", fast_ema_period=5, slow_ema_period=10,
    )),
    EMACrossStopReverse(EMACrossSARConfig(
        strategy_id="EMACrossStop&Reverse", order_id_tag="001",
        instrument_ids=PERP_INSTRUMENTS_SAR, trade_usd=Decimal("2000"),
        bar_spec="1-MINUTE-LAST", fast_ema_period=60, slow_ema_period=120,
    )),
]

def refuse_if_another_node_running() -> None:
    """Two nodes on one Redis both obey every strategy command and interleave
    their orders, positions and status on the dashboard. Refuse to be second."""
    hb = K.read_json(K.connect(), K.HEARTBEAT)
    if hb and time.time() - hb["ts"] / 1000 < 15 and hb.get("pid") != os.getpid():
        raise SystemExit(
            f"[node] REFUSING TO START: another trading node (pid {hb.get('pid')}) is already running "
            f"against this Redis (heartbeat {time.time() - hb['ts'] / 1000:.0f}s ago). Stop it first.")


if __name__ == "__main__":
    refuse_if_another_node_running()

node = TradingNode(config_node)
control = ControlActor(ControlActorConfig(venue=BINANCE_FUTURES, sandbox=True))
control.strategies = STRATEGIES
node.trader.add_actor(control)
for s in STRATEGIES:
    node.trader.add_strategy(s)
STRATS = {str(s.id): s for s in STRATEGIES}
for sid, s in STRATS.items():
    print(f"[node] strategy registered: {sid} ({len(s.config.instrument_ids)} instruments, {s.config.bar_spec})")

for name in (BINANCE_SPOT, BINANCE_FUTURES):
    node.add_data_client_factory(name, BinanceLiveDataClientFactory)
node.add_exec_client_factory(BINANCE_FUTURES, SandboxLiveExecClientFactory)
node.build()


# ---------------------------------------------------------------------------
# position export
# ---------------------------------------------------------------------------

def export_positions_csv(r, sid: str, mode: str) -> str:
    snap = K.read_json(r, K.PORTFOLIO, {}) or {}
    ledger = K.read_json(r, K.CLOSED_POSITIONS, {}) or {}
    open_ = [p for p in snap.get("positions", []) if p.get("strategy") == sid]
    closed = [c for c in ledger.values() if c.get("strategy") == sid]
    if mode == "test":
        bt = K.read_json(K.connect(K.test_url()), f"{K.BACKTEST}:positions", {}) or {}
        seen = {c.get("id") for c in closed}
        closed += [c for c in bt.get("closed_positions", []) if c.get("id") not in seen]
    closed.sort(key=lambda c: c.get("ts_closed", 0))

    def ist(ms):
        return datetime.fromtimestamp(ms / 1000, tz=_IST).strftime("%Y-%m-%d %H:%M:%S") if ms else ""

    def hold(a, b):
        if not a or not b:
            return ""
        s = int((b - a) / 1000)
        return f"{s // 60}m {s % 60}s" if s >= 60 else f"{s}s"

    os.makedirs(EXPORT_DIR, exist_ok=True)
    path = os.path.join(EXPORT_DIR, f"positions_{sid.replace('&', 'and')}_{datetime.now():%Y%m%d-%H%M%S}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["status", "symbol", "strategy", "side", "qty", "entry", "exit",
                    "entry_px", "exit_px", "pnl", "ccy", "hold"])
        for p in closed:
            w.writerow(["closed", p.get("instrument", "").split(".")[0], sid, p.get("side"), p.get("qty"),
                        ist(p.get("ts_opened")), ist(p.get("ts_closed")), p.get("avg_px_open"),
                        p.get("avg_px_close"), p.get("realized"), p.get("ccy"),
                        hold(p.get("ts_opened"), p.get("ts_closed"))])
        for p in open_:
            w.writerow(["open", p.get("instrument", "").split(".")[0], sid, p.get("side"), p.get("qty"),
                        ist(p.get("ts_opened")), "", p.get("avg_px"), p.get("mark"),
                        p.get("unrealized"), p.get("ccy"), ""])
    return f"{len(closed)} closed + {len(open_)} open position(s) -> {os.path.abspath(path)}"


# ---------------------------------------------------------------------------
# command dispatcher
# ---------------------------------------------------------------------------

class Dispatcher:
    """Reads strategy commands from Redis on its own thread; runs them on the
    node's event loop (Nautilus objects are not thread-safe) and replies."""

    def __init__(self, loop):
        self.loop = loop
        self.r = K.connect()
        self.stop = threading.Event()
        self.backtest_lock = threading.Lock()

    def reply(self, level, title, detail="", cmd=None, retry=False):
        cmd = cmd or {}
        sid = cmd.get("strategy_id", "")
        payload = K.strategy_retry(sid, cmd.get("action", ""), cmd.get("mode", "live"),
                                   **{k: v for k, v in cmd.items()
                                      if k not in ("action", "strategy_id", "mode", "cmd_id")}) if retry else None
        K.notify(self.r, level, f"strategy:{sid}" if sid else "node", title, detail,
                 retry=payload, key=f"cmd:{cmd.get('action')}:{sid}", cmd_id=cmd.get("cmd_id", ""))

    def on_loop(self, cmd, fn, label):
        """Run fn() on the event loop; reply with its message or the error."""
        def run():
            try:
                msg = fn()
                self.reply("success", f"{label}: {msg}", cmd=cmd)
            except Exception as e:
                traceback.print_exc()
                self.reply("error", f"{label} failed", f"{type(e).__name__}: {e}", cmd=cmd, retry=True)
        self.loop.call_soon_threadsafe(run)

    def run(self):
        cursor = "$"
        print(f"[dispatcher] listening on {K.STRATEGY_CMDS}")
        outage = None
        while not self.stop.is_set():
            try:
                res = self.r.xread({K.STRATEGY_CMDS: cursor}, count=20, block=1000)
                if outage is not None:
                    print(f"[dispatcher] redis back after {time.time() - outage:.0f}s")
                    outage = None
            except redis.RedisError as e:
                if outage is None:
                    outage = time.time()
                    print(f"[dispatcher] redis unavailable ({e}); retrying")
                time.sleep(1)
                continue
            for eid, fields in (res[0][1] if res else []):
                cursor = eid
                try:
                    self.handle(eid, dict(fields))
                except Exception as e:
                    traceback.print_exc()
                    self.reply("error", f"Command '{fields.get('action')}' crashed", str(e), cmd=fields, retry=True)

    def handle(self, eid: str, cmd: dict):
        action, sid = cmd.get("action"), cmd.get("strategy_id")
        age = time.time() - int(eid.split("-")[0]) / 1000
        print(f"[dispatcher] {action} {sid} (age {age:.1f}s, id {cmd.get('cmd_id')})")
        if age > K.STALE_COMMAND_S:
            # Queued while the node or Redis was down: acting on it now could
            # start trading hours after the click.
            self.reply("warn", f"Ignored stale '{action}' command",
                       f"Sent {age:.0f}s ago while the node was unreachable. Retry if you still want it.",
                       cmd=cmd, retry=True)
            return
        strat = STRATS.get(sid)
        if strat is None:
            self.reply("error", f"Unknown strategy '{sid}'", f"Running strategies: {', '.join(STRATS)}", cmd=cmd)
            return
        mode = cmd.get("mode", "live")
        if action == "start_strategy":
            self.r.hset(K.STRATEGY_MODES, sid, "live" if mode != "test" else "test")
            self.on_loop(cmd, strat.arm, f"Started {sid}")
        elif action == "stop_strategy":
            self.on_loop(cmd, strat.disarm, f"Paused {sid}")
        elif action == "close_strategy":
            strat.after_close_all = lambda: self.reply(
                "success", f"Exported {sid}", export_positions_csv(self.r, sid, mode), cmd=cmd)
            self.on_loop(cmd, strat.close_all, f"Close-all {sid}")
        elif action == "rewarm":
            self.on_loop(cmd, strat.rewarm, f"Re-warm {sid}")
        elif action == "export_csv":
            try:
                self.reply("success", f"Exported {sid}", export_positions_csv(self.r, sid, mode), cmd=cmd)
            except Exception as e:
                self.reply("error", f"Export for {sid} failed", f"{type(e).__name__}: {e}", cmd=cmd, retry=True)
        elif action == "start_test_strategy":
            self.start_test(cmd, strat)
        else:
            self.reply("error", f"Unknown action '{action}'", cmd=cmd)

    def start_test(self, cmd, strat):
        sid = str(strat.id)
        start_date, end_date = cmd.get("start_date") or None, cmd.get("end_date") or None
        if not self.backtest_lock.acquire(blocking=False):
            self.reply("error", "A backtest is already running", "Wait for it to finish, then retry.",
                       cmd=cmd, retry=True)
            return
        self.r.hset(K.STRATEGY_MODES, sid, "test")
        live_handoff = not end_date
        self.reply("info", f"Backtest started for {sid}",
                   f"{start_date} -> {end_date or 'now'}" + ("; hands off to live when done" if live_handoff else ""),
                   cmd=cmd)

        def work():
            rt = K.connect(K.test_url())
            try:
                insts = [node.cache.instrument(i) for i in strat.config.instrument_ids]
                result = run_backtest(rt, strat, [i for i in insts if i is not None],
                                      start_date=start_date, end_date=end_date)
                self.reply("success" if not result["skipped"] else "warn",
                           f"Backtest finished for {sid}: {result['closed']} closed, "
                           f"PnL {result['pnl']:+,.2f} USDT",
                           (f"No data for {len(result['skipped'])} instrument(s): "
                            + ", ".join(result["skipped"])) if result["skipped"] else "", cmd=cmd)
                if live_handoff:
                    self.on_loop(cmd, strat.arm, f"Live hand-off {sid}")
            except Exception as e:
                traceback.print_exc()
                self.reply("error", f"Backtest failed for {sid}", f"{type(e).__name__}: {e}", cmd=cmd, retry=True)
            finally:
                self.backtest_lock.release()
                rt.close()

        threading.Thread(target=work, name=f"backtest-{sid}", daemon=True).start()


if __name__ == "__main__":
    dispatcher = Dispatcher(node.get_event_loop())
    threading.Thread(target=dispatcher.run, name="dispatcher", daemon=True).start()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        dispatcher.stop.set()
        try:
            node.stop()
        except Exception:
            pass
        node.dispose()
