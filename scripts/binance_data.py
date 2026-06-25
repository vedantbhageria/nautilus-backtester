import os
import time
import threading
from decimal import Decimal

import redis as syncredis
from dotenv import load_dotenv
from nautilus_trader.live.node import TradingNode
from nautilus_trader.adapters.binance import BinanceLiveDataClientFactory
from nautilus_trader.adapters.binance import BinanceLiveExecClientFactory
from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.examples.algorithms.twap import TWAPExecAlgorithm

from trading.configs.binance_config import config_node, BINANCE_SPOT, BINANCE_FUTURES
from trading.actors.control_actor import ControlActor, ControlActorConfig
from trading.strategies.EMACross import EMACross, EMACrossConfig
from trading.strategies.fixed_positions import FixedNotional, FixedNotionalConfig, STRATEGY_SET

load_dotenv()
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
STRATEGY_CMDS_STREAM = "dashboard:strategy_cmds"

_strats: dict[str, object] = {}   # str(strategy.id) -> Strategy instance
_stop_event = threading.Event()


STRATEGY_STATES_KEY = "dashboard:strategy_states"


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
                        loop.call_soon_threadsafe(_safe_call(strat.stop, f"stop({sid})"))
                        r.hset(STRATEGY_STATES_KEY, sid, "STOPPED")
                        print(f"[strategy_manager] stop scheduled for {sid}")
                    elif action == "start_strategy":
                        loop.call_soon_threadsafe(_arm(strat))
                        loop.call_soon_threadsafe(_safe_call(strat.reset, f"reset({sid})"))
                        loop.call_soon_threadsafe(_safe_call(strat.start, f"start({sid})"))
                        r.hset(STRATEGY_STATES_KEY, sid, "RUNNING")
                        print(f"[strategy_manager] start scheduled for {sid}")
                    elif action == "close_strategy":
                        loop.call_soon_threadsafe(_disarm(strat))
                        loop.call_soon_threadsafe(_safe_call(strat.close_all_positions, f"close_all({sid})"))
                        loop.call_soon_threadsafe(_safe_call(strat.stop, f"stop({sid})"))
                        r.hset(STRATEGY_STATES_KEY, sid, "STOPPED")
                        print(f"[strategy_manager] close+stop scheduled for {sid}")
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
    fast_ema_period=5,
    slow_ema_period=10,
))
node.trader.add_strategy(_ema)
_strats[str(_ema.id)] = _ema
print(f"[binance_data] strategy registered: id={_ema.id!r}")
#exec_algorithm = TWAPExecAlgorithm()
#node.trader.add_exec_algorithm(exec_algorithm)


for name in (BINANCE_SPOT, BINANCE_FUTURES):
    node.add_data_client_factory(name, BinanceLiveDataClientFactory)
    node.add_exec_client_factory(name, BinanceLiveExecClientFactory)

node.build()

# Write initial RUNNING state for all registered strategies.
"""
_r = syncredis.Redis.from_url(REDIS_URL, decode_responses=True)
for _sid in _strats:
    _r.hset(STRATEGY_STATES_KEY, _sid, "RUNNING")
_r.close()
"""

if __name__ == "__main__":
    r = syncredis.Redis.from_url(REDIS_URL, decode_responses=True)
    r.flushall()
    # Show strategies on the dashboard immediately as STOPPED — they don't run
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
