import os
import csv
import json
import time
import threading
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import redis
import redis as syncredis
from dotenv import load_dotenv
from nautilus_trader.live.node import TradingNode
from nautilus_trader.adapters.binance import BinanceLiveDataClientFactory
from nautilus_trader.adapters.binance import BinanceLiveExecClientFactory
from nautilus_trader.adapters.sandbox.factory import SandboxLiveExecClientFactory
from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.examples.algorithms.twap import TWAPExecAlgorithm

from trading.configs.binance_config import config_node, BINANCE_SPOT, BINANCE_FUTURES
from trading.actors.control_actor import ControlActor, ControlActorConfig
from trading.strategies.EMACross import EMACross, EMACrossConfig
from trading.strategies.EMACrossShortTest import EMACrossStopReverse, EMACrossSARConfig
from trading.strategies.fixed_positions import FixedNotional, FixedNotionalConfig, STRATEGY_SET
from backtest_runner import run_backtest

load_dotenv()
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
STRATEGY_CMDS_STREAM = "dashboard:strategy_cmds"

_strats: dict[str, object] = {}   # str(strategy.id) -> Strategy instance
_stop_event = threading.Event()


STRATEGY_STATES_KEY = "dashboard:strategy_states"
PORTFOLIO_KEY = "dashboard:portfolio"
EXPORT_DIR = os.getenv("POSITION_EXPORT_DIR", "exports")


def _iso(ms):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat() if ms else ""


def _export_positions_csv(r, sid):
    try:
        raw = r.get(PORTFOLIO_KEY)
        if not raw:
            print(f"[export] no portfolio snapshot available for {sid}")
            return
        snap = json.loads(raw)
        open_ = [p for p in snap.get("positions", []) if p.get("strategy") == sid]
        closed = [p for p in snap.get("closed_positions", []) if p.get("strategy") == sid]
        os.makedirs(EXPORT_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = os.path.join(EXPORT_DIR, f"positions_{sid}_{ts}.csv")
        def _ist(ms):
            if not ms: return ""
            return (datetime.fromisoformat(_iso(ms).replace("Z", "")) + timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d %H:%M:%S")
        def _hold(a, b):
            if not a or not b: return ""
            s = int((b - a) / 1000)
            m = s // 60
            return f"{m}m {s % 60}s" if m else f"{s}s"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["symbol", "strategy", "side", "qty", "entry", "exit",
                        "entry_px", "exit_px", "pnl", "ccy", "hold"])
            for p in closed:
                w.writerow([
                    p.get("instrument", "").split(".")[0],
                    (p.get("strategy") or "").split("-None")[0],
                    p.get("side"), p.get("qty"),
                    _ist(p.get("ts_opened")), _ist(p.get("ts_closed")),
                    p.get("avg_px_open"), p.get("avg_px_close"),
                    p.get("realized"), p.get("ccy"),
                    _hold(p.get("ts_opened"), p.get("ts_closed")),
                ])
                
        print(f"[export] wrote {len(closed)} closed + {len(open_)} open positions to {path}")
    except Exception as e:
        print(f"[export] failed for {sid}: {e}")


def _safe_call(fn, label):
    def wrapper():
        try:
            fn()
        except Exception as e:
            print(f"[strategy_manager] {label} raised: {e}")
    return wrapper


def _arm(strat):
    return lambda: setattr(strat, "_active", True)


def _disarm(strat):
    return lambda: setattr(strat, "_active", False)


def _cancel_all(strat):
    # Cancel every open order for this strategy (cancel_all_orders needs a
    # per-instrument id, so iterate the strategy's open orders from the cache).
    def fn():
        for o in list(strat.cache.orders_open(strategy_id=strat.id)):
            strat.cancel_order(o)
    return fn


def _close_all(strat):
    # Close every open position for this strategy (close_all_positions needs a
    # per-instrument id, so iterate the strategy's open positions from the cache).
    def fn():
        for p in list(strat.cache.positions_open(strategy_id=strat.id)):
            strat.close_position(p)
    return fn


def _strategy_manager(loop):
    r = syncredis.Redis.from_url(REDIS_URL, decode_responses=True)
    cursor = "$"
    # Strategies auto-start with the node. We want them OFF until the user
    # presses Start, so stop each one the instant it reaches RUNNING.

    to_stop = set(_strats.keys())
    print(f"[strategy_manager] started, watching {STRATEGY_CMDS_STREAM!r}, strats={list(_strats.keys())}")
    while not _stop_event.is_set():
        try:
            for sid in list(to_stop):
                strat = _strats[sid]
                if strat.is_running:
                    loop.call_soon_threadsafe(_safe_call(strat.stop, f"initial-stop({sid})"))
                    r.hset(STRATEGY_STATES_KEY, sid, "STOPPED")
                    to_stop.discard(sid)
                    print(f"[strategy_manager] auto-stopped {sid} on startup")
            block_ms = 100 if to_stop else 1000
            results = r.xread({STRATEGY_CMDS_STREAM: cursor}, count=10, block=block_ms)
            if not results:
                continue
            _, entries = results[0]
            for eid, fields in entries:
                cursor = eid
                action = fields.get("action")
                sid = fields.get("strategy_id")
                print(f"[strategy_manager] received action={action!r} strategy_id={sid!r}")
                strat = _strats.get(sid)
                if not strat:
                    print(f"[strategy_manager] unknown strategy_id={sid!r}, known={list(_strats.keys())}")
                    continue
                try:
                    if action == "stop_strategy":
                        loop.call_soon_threadsafe(_disarm(strat))
                        loop.call_soon_threadsafe(_safe_call(_cancel_all(strat), f"cancel-orders({sid})"))
                        loop.call_soon_threadsafe(_safe_call(strat.stop, f"stop({sid})"))
                        r.hset(STRATEGY_STATES_KEY, sid, "STOPPED")
                        print(f"[strategy_manager] stop scheduled for {sid}")
                    elif action == "start_strategy":
                        loop.call_soon_threadsafe(_arm(strat))
                        loop.call_soon_threadsafe(_safe_call(strat.reset, f"reset({sid})"))
                        loop.call_soon_threadsafe(_safe_call(strat.start, f"start({sid})"))
                        r.hset(STRATEGY_STATES_KEY, sid, "RUNNING")
                        print(f"[strategy_manager] start scheduled for {sid}")
                    elif action == "export_csv":
                        _export_positions_csv(r, sid)
                        print(f"[strategy_manager] CSV exported for {sid}")
                    elif action == "backtest_strategy":
                        # Run a 4-day BacktestEngine for this strategy in a background
                        # thread; results land in dashboard:backtest:* (separate from live).
                        def _run_bt(sid=sid, strat=strat):
                            rr = syncredis.Redis.from_url(REDIS_URL, decode_responses=True)
                            try:
                                insts = [node.cache.instrument(iid)
                                         for iid in strat.config.instrument_ids]
                                insts = [i for i in insts if i is not None]
                                run_backtest(rr, strat, insts, days=4)
                            finally:
                                rr.close()
                        threading.Thread(target=_run_bt, daemon=True).start()
                        print(f"[strategy_manager] backtest started for {sid}")
                    elif action == "close_strategy":
                        loop.call_soon_threadsafe(_disarm(strat))
                        loop.call_soon_threadsafe(_safe_call(_cancel_all(strat), f"cancel-orders({sid})"))
                        loop.call_soon_threadsafe(_safe_call(_close_all(strat), f"close_all({sid})"))
                        r.hset(STRATEGY_STATES_KEY, sid, "STOPPED")
                        print(f"[strategy_manager] close scheduled for {sid}")
                        # Retry close until all positions fill; then export CSV and stop.
                        def _drain_and_stop(strat=strat, sid=sid, r=r):
                            for _ in range(30):
                                time.sleep(1)
                                remaining = strat.cache.positions_open(strategy_id=strat.id)
                                if not remaining:
                                    break
                                loop.call_soon_threadsafe(
                                    _safe_call(_close_all(strat), f"retry-close({sid})")
                                )
                            _export_positions_csv(r, sid)
                            loop.call_soon_threadsafe(_safe_call(strat.stop, f"stop({sid})"))
                            print(f"[strategy_manager] all positions closed, stopped {sid}")
                        threading.Thread(target=_drain_and_stop, daemon=True).start()
                except Exception as e:
                    print(f"[strategy_manager] error scheduling {action} for {sid}: {e}")
        except Exception as e:
            print(f"[strategy_manager] poll error: {e}")
            time.sleep(0.5)

node = TradingNode(config_node)

# Top 20 most popular USDT-margined perpetual futures on Binance by open interest.
_PERP_SYMS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "SUIUSDT",
    "DOTUSDT", "NEARUSDT", "APTUSDT", "LTCUSDT", "UNIUSDT",
    "ATOMUSDT", "INJUSDT", "AAVEUSDT", "ARBUSDT", "RENDERUSDT",
]
PERP_INSTRUMENTS = tuple(
    InstrumentId.from_str(f"{sym}-PERP.{BINANCE_FUTURES}")
    for sym in _PERP_SYMS
)

_PERP_SYMS_STOP_REVERSES = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT",
    "TRXUSDT", "HYPEUSDT", "DOGEUSDT", "ZECUSDT", "LABUSDT",
    "XLMUSDT", "XMRUSDT", "CCUSDT", "LINKUSDT", "ADAUSDT",
    "BCHUSDT", "LTCUSDT", "HBARUSDT", "SUIUSDT", "AVAXUSDT",
    "1000SHIBUSDT", "NEARUSDT", "TAOUSDT", "WLFIUSDT", "PAXGUSDT",
    "UNIUSDT", "ASTERUSDT", "WLDUSDT", "ONDOUSDT", "DOTUSDT",
    "AAVEUSDT", "SKYUSDT", "MUSDT", "ETCUSDT", "MORPHOUSDT",
    "DEXEUSDT", "1000PEPEUSDT", "QNTUSDT", "ATOMUSDT", "RENDERUSDT",
    "POLUSDT", "KASUSDT", "ALGOUSDT", "ENAUSDT", "JUPUSDT",
    "JSTUSDT", "BEATUSDT", "VVVUSDT", "FILUSDT", "NIGHTUSDT",
    "APTUSDT", "ARBUSDT", "AEROUSDT", "INJUSDT", "DASHUSDT",
    "CAKEUSDT", "TRUMPUSDT", "VETUSDT", "FETUSDT", "PENGUUSDT",
    "SEIUSDT", "JTOUSDT", "1000BONKUSDT", "1000LUNCUSDT", "ETHFIUSDT",
    "VIRTUALUSDT", "KITEUSDT", "TIAUSDT", "SUNUSDT", "SKYAIUSDT",
    "STXUSDT", "SPXUSDT", "CRVUSDT", "XPLUSDT", "GRASSUSDT",
    "GWEIUSDT", "PYTHUSDT", "XTZUSDT", "OPUSDT", "MONUSDT",
    "CFXUSDT", "JASMYUSDT", "BSVUSDT", "BUSDT", "1000FLOKIUSDT",
    "PENDLEUSDT", "VELVETUSDT", "LDOUSDT", "ZROUSDT", "KAIAUSDT",
    "AKTUSDT", "GRTUSDT", "STRKUSDT", "CHZUSDT", "UBUSDT",
    "AXSUSDT", "IOTAUSDT", "ENSUSDT", "EIGENUSDT", "COMPUSDT",
]

PERP_INSTRUMENTS_SAR = tuple(
    InstrumentId.from_str(f"{sym}-PERP.{BINANCE_FUTURES}")
    for sym in _PERP_SYMS_STOP_REVERSES
)

node.trader.add_actor(ControlActor(ControlActorConfig()))

# _fn = FixedNotional(FixedNotionalConfig(
#     strategy_id="FixedNotional-001",
#     instrument_ids=PERP_INSTRUMENTS,
#     target_usd=Decimal("2000"),
#     rebalance_threshold=0.001,
# ))
# node.trader.add_strategy(_fn)
# _strats[str(_fn.id)] = _fn
# print(f"[binance_data] strategy registered: id={_fn.id!r}")

_ema = EMACross(EMACrossConfig(
    strategy_id="EMACross-001",
    instrument_ids=PERP_INSTRUMENTS,
    trade_usd=Decimal("2000"),
    bar_spec="5-SECOND-LAST",   # 5-second bars per instrument
    fast_ema_period=5,  #60 180
    slow_ema_period=10,
))
_ema_SAR = EMACrossStopReverse(EMACrossSARConfig(
    strategy_id="EMACrossStop&Reverse-001",
    instrument_ids=PERP_INSTRUMENTS_SAR,
    trade_usd=Decimal("2000"),
    bar_spec="1-MINUTE-LAST",   # 5-second bars per instrument
    fast_ema_period=5,  
    slow_ema_period=10,
))

node.trader.add_strategy(_ema)
node.trader.add_strategy(_ema_SAR)
_strats[str(_ema.id)] = _ema
_strats[str(_ema_SAR.id)] = _ema_SAR

print(f"[binance_data] strategy registered: id={_ema.id!r}")
print(f"[binance_data] strategy registered: id={_ema_SAR.id!r}")

#exec_algorithm = TWAPExecAlgorithm()
#node.trader.add_exec_algorithm(exec_algorithm)


for name in (BINANCE_SPOT, BINANCE_FUTURES):
    node.add_data_client_factory(name, BinanceLiveDataClientFactory)
node.add_exec_client_factory(BINANCE_FUTURES, SandboxLiveExecClientFactory)

node.build()

# Write initial RUNNING state for all registered strategies.

"""
_r = syncredis.Redis.from_url(REDIS_URL, decode_responses=True)
for _sid in _strats:
    _r.hset(STRATEGY_STATES_KEY, _sid, "RUNNING")
_r.close()
"""

OVERALL_PNL_KEY = "dashboard:overall_pnl"

if __name__ == "__main__":
    r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
    # Preserve the persisted overall (cross-session) PnL across the flush so it
    # survives node restarts; everything else is session state and gets wiped.
    _saved_overall = r.get(OVERALL_PNL_KEY)
    if _saved_overall:
        r.set(OVERALL_PNL_KEY, _saved_overall)
    # Show strategies on the dashboard immediately as STOPPED, they don't run
    # until the user presses Start (the manager auto-stops them on startup).
    for _sid in _strats:
        r.sadd(STRATEGY_SET, _sid)
        r.hset(STRATEGY_STATES_KEY, _sid, "STOPPED")
    r.close()
    loop = node.get_event_loop()
    t = threading.Thread(target=_strategy_manager, args=(loop,), daemon=True)
    t.start()
    try:
        node.run()
    except KeyboardInterrupt:
        node.stop()
    finally:
        _stop_event.set()
        node.dispose()
