from decimal import Decimal

from nautilus_trader.live.node import TradingNode
from nautilus_trader.adapters.binance import BinanceLiveDataClientFactory
from nautilus_trader.adapters.binance import BinanceLiveExecClientFactory
from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.examples.algorithms.twap import TWAPExecAlgorithm


from trading.configs.binance_config import config_node, BINANCE_SPOT, BINANCE_FUTURES
from trading.actors.control_actor import ControlActor, ControlActorConfig
from trading.strategies.EMACross import EMACross, EMACrossConfig
from trading.strategies.fixed_positions import FixedNotional, FixedNotionalConfig

node = TradingNode(config_node)

# Top 20 most popular USDT-margined perpetual futures on Binance by open interest.
_PERP_SYMS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "SUIUSDT",
    "DOTUSDT", "NEARUSDT", "APTUSDT", "LTCUSDT", "UNIUSDT",
    "ATOMUSDT", "INJUSDT", "AAVEUSDT", "ARBUSDT", "MNTUSDT",
]
PERP_INSTRUMENTS = tuple(
    InstrumentId.from_str(f"{sym}-PERP.{BINANCE_FUTURES}")
    for sym in _PERP_SYMS
)

node.trader.add_actor(ControlActor(ControlActorConfig()))
node.trader.add_strategy(FixedNotional(FixedNotionalConfig(
    instrument_ids=PERP_INSTRUMENTS,
    target_usd=Decimal("2000"),
    rebalance_threshold=0.001,
)))
#exec_algorithm = TWAPExecAlgorithm()
"""
node.trader.add_strategy(EMACross(EMACrossConfig(
    instrument_id=BTC,
    bar_type=BTC_BAR,
    trade_size=Decimal("0.001"),   # BTC perp min order size
    fast_ema_period=5,
    slow_ema_period=20,
)))
"""
#node.trader.add_exec_algorithm(exec_algorithm)


for name in (BINANCE_SPOT, BINANCE_FUTURES):
    node.add_data_client_factory(name, BinanceLiveDataClientFactory)
    node.add_exec_client_factory(name, BinanceLiveExecClientFactory)

node.build()

if __name__ == "__main__":
    try:
        node.run()
    except KeyboardInterrupt:
        node.stop()
    finally:
        node.dispose()
