import os
from urllib.parse import urlparse

from dotenv import load_dotenv

from nautilus_trader.config import TradingNodeConfig
from nautilus_trader.model.identifiers import TraderId
from nautilus_trader.config import LoggingConfig
from nautilus_trader.config import LiveExecEngineConfig
from nautilus_trader.config import LiveDataEngineConfig
from nautilus_trader.config import CacheConfig
from nautilus_trader.config import MessageBusConfig
from nautilus_trader.config import DatabaseConfig
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.instruments import CryptoPerpetual, CryptoFuture, CurrencyPair
from nautilus_trader.adapters.binance import BinanceDataClientConfig
from nautilus_trader.adapters.binance import BinanceExecClientConfig
from nautilus_trader.adapters.binance import BinanceAccountType
from nautilus_trader.adapters.binance import BinanceInstrumentProviderConfig
from nautilus_trader.adapters.binance.common.enums import BinanceEnvironment

BINANCE_SPOT = "BINANCE_SPOT"          # instruments -> e.g. BTCUSDT.BINANCE_SPOT
BINANCE_FUTURES = "BINANCE_FUTURES"    # instruments -> e.g. BTCUSDT-PERP.BINANCE_FUTURES

load_dotenv()
_redis = urlparse(os.getenv("REDIS_URL", "redis://localhost:6379"))
redis_db = DatabaseConfig(
    type="redis",
    host=_redis.hostname or "localhost",
    port=_redis.port or 6379,
    username=_redis.username,
    password=_redis.password,
)

_retention = int(os.getenv("STREAM_RETENTION_MINS", "4320"))  # 3 days
STREAM_RETENTION_MINS = _retention if _retention > 0 else None
CATALOG_PATH = os.getenv("CATALOG_PATH", "C:/nautilus/catalog")


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
        database=redis_db,
        timestamps_as_iso8601=True,
        flush_on_start=False,
    ),
    message_bus=MessageBusConfig(
        database=redis_db,
        encoding="json",
        timestamps_as_iso8601=True,
        stream_per_topic=True,
        streams_prefix="stream",
        autotrim_mins=STREAM_RETENTION_MINS,
        heartbeat_interval_secs=1,
        types_filter=[CryptoPerpetual, CryptoFuture, CurrencyPair],
    ),
    data_clients={
        BINANCE_SPOT: BinanceDataClientConfig(
            venue=Venue(BINANCE_SPOT),
            account_type=BinanceAccountType.SPOT,
            environment=BinanceEnvironment.DEMO,
            use_agg_trade_ticks=True,  
            instrument_provider=BinanceInstrumentProviderConfig(
                load_all=True,
            ),
        ),
        BINANCE_FUTURES: BinanceDataClientConfig(
            venue=Venue(BINANCE_FUTURES),
            account_type=BinanceAccountType.USDT_FUTURES,
            environment=BinanceEnvironment.DEMO,
            use_agg_trade_ticks=True,
            instrument_provider=BinanceInstrumentProviderConfig(
                load_all=True,
            ),
        ),
    },
    exec_clients={},
    timeout_connection=30.0,
    timeout_reconciliation=10.0,
    timeout_portfolio=10.0,
    timeout_disconnection=10.0,
    timeout_post_stop=5.0,
)
