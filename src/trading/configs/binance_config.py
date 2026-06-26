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
from nautilus_trader.adapters.sandbox.config import SandboxExecutionClientConfig

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
        # No purging: hold closed orders/positions/account events in the cache for
        # the entire session so the dashboard keeps the full position history.
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
        buffer_interval_ms=50,
        types_filter=[CryptoPerpetual, CryptoFuture, CurrencyPair],
    ),
    data_clients={
        BINANCE_SPOT: BinanceDataClientConfig(
            venue=Venue(BINANCE_SPOT),
            account_type=BinanceAccountType.SPOT,
            environment=BinanceEnvironment.LIVE,
            use_agg_trade_ticks=True,  
            instrument_provider=BinanceInstrumentProviderConfig(
                load_all=True,
            ),
        ),
        BINANCE_FUTURES: BinanceDataClientConfig(
            venue=Venue(BINANCE_FUTURES),
            account_type=BinanceAccountType.USDT_FUTURES,
            environment=BinanceEnvironment.LIVE,
            use_agg_trade_ticks=True,
            instrument_provider=BinanceInstrumentProviderConfig(
                load_all=True,
            ),
        ),
    },

    exec_clients={
        BINANCE_FUTURES: SandboxExecutionClientConfig(
        venue=BINANCE_FUTURES,
        starting_balances=["100000 USDT"],
        ),
    },
    timeout_connection=30.0,
    timeout_reconciliation=10.0,
    timeout_portfolio=10.0,
    timeout_disconnection=30.0,
    timeout_post_stop=30.0,
)

"""        BINANCE_FUTURES: BinanceExecClientConfig(
            venue=Venue(BINANCE_FUTURES),
            account_type=BinanceAccountType.USDT_FUTURES,
            environment=BinanceEnvironment.DEMO,
            instrument_provider=BinanceInstrumentProviderConfig(
                load_all=True,
            ),
        ),"""