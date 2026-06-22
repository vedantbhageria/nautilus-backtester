from nautilus_trader.config import TradingNodeConfig
from nautilus_trader.live.node import TradingNode
from nautilus_trader.model.identifiers import TraderId
from nautilus_trader.model.identifiers import ClientId
from nautilus_trader.config import LoggingConfig
from nautilus_trader.config import LiveExecEngineConfig
from nautilus_trader.config import InstrumentProviderConfig
from nautilus_trader.config import LiveDataEngineConfig
from nautilus_trader.config import CacheConfig
from nautilus_trader.adapters.binance import BINANCE
from nautilus_trader.adapters.binance import BinanceDataClientConfig
from nautilus_trader.adapters.binance import BinanceExecClientConfig
from nautilus_trader.adapters.binance import BinanceAccountType
from nautilus_trader.adapters.binance import BinanceInstrumentProviderConfig
from nautilus_trader.adapters.binance.common.enums import BinanceEnvironment


config_node = TradingNodeConfig(
    trader_id=TraderId("TESTER-001"),
    logging=LoggingConfig(
        log_level="INFO",
        # log_level_file="DEBUG",
        # log_file_format="json",
        log_colors=True,
        use_pyo3=True,
    ),
    data_engine=LiveDataEngineConfig(
        external_clients=[ClientId(BINANCE)],
    ),
    exec_engine=LiveExecEngineConfig(
        reconciliation=True,
        # open_check_interval_secs=5.0,
        open_check_open_only=False,
        # snapshot_orders=True,
        # snapshot_positions=True,
        # snapshot_positions_interval_secs=5.0,
        purge_closed_orders_interval_mins=1,
        purge_closed_orders_buffer_mins=0,
        purge_closed_positions_interval_mins=1,
        purge_closed_positions_buffer_mins=0,
        purge_account_events_interval_mins=1,
        purge_account_events_lookback_mins=0,
        purge_from_database=True,
        graceful_shutdown_on_exception=True,
    ),
    cache=CacheConfig(
        # database=DatabaseConfig(),
        timestamps_as_iso8601=True,
        flush_on_start=False,
    ),
    data_clients={
        BINANCE: BinanceDataClientConfig(
            account_type=BinanceAccountType.USDT_FUTURES,
            environment=BinanceEnvironment.DEMO,
            instrument_provider=BinanceInstrumentProviderConfig(
                load_all=True,
                query_commission_rates=True,
            ),
        ),
    },
    exec_clients={
        BINANCE: BinanceExecClientConfig(
            account_type=BinanceAccountType.USDT_FUTURES,
            environment=BinanceEnvironment.DEMO,
            instrument_provider=BinanceInstrumentProviderConfig(
                load_all=True,
                query_commission_rates=True,
            ),
            max_retries=3,
            log_rejected_due_post_only_as_warning=False,
        ),
    },
    timeout_connection=30.0,
    timeout_reconciliation=10.0,
    timeout_portfolio=10.0,
    timeout_disconnection=10.0,
    timeout_post_stop=5.0,
)