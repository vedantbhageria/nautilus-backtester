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

node = TradingNode(config_node)

# BTC perpetual on the Futures testnet (DEMO) — execution venue.
BTC = InstrumentId.from_str(f"BTCUSDT-PERP.{BINANCE_FUTURES}")
# Tick data → INTERNAL tick-aggregation bars (built locally from trades).
# 100-TICK = one bar per 100 trades. Drop to 1-TICK for a bar per trade (noisier).
BTC_BAR = BarType.from_str(f"BTCUSDT-PERP.{BINANCE_FUTURES}-10-TICK-LAST-INTERNAL")

node.trader.add_actor(ControlActor(ControlActorConfig()))
#exec_algorithm = TWAPExecAlgorithm()
node.trader.add_strategy(EMACross(EMACrossConfig(
    instrument_id=BTC,
    bar_type=BTC_BAR,
    trade_size=Decimal("0.001"),   # BTC perp min order size
    fast_ema_period=5,
    slow_ema_period=20,
)))
#node.trader.add_exec_algorithm(exec_algorithm)


for name in (BINANCE_SPOT, BINANCE_FUTURES):
    node.add_data_client_factory(name, BinanceLiveDataClientFactory)
    node.add_exec_client_factory(name, BinanceLiveExecClientFactory)

node.build()

if __name__ == "__main__":
    try:
        node.run()
    finally:
        node.dispose()
