import os
from decimal import Decimal
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
from nautilus_trader.model.data import OrderBookDelta, OrderBookDeltas
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
        # Keep order book deltas OFF the Redis backbone: the L2 subscription streams
        # ~20k book updates/sec across all instruments, and with stream_per_topic +
        # 3-day retention that floods/OOMs Redis (crashes the node's data feed). The
        # in-process matcher still receives them (types_filter only affects external
        # streaming), so L2 fills keep working — the dashboard never reads books anyway.
        types_filter=[CryptoPerpetual, CryptoFuture, CurrencyPair,
                      OrderBookDelta, OrderBookDeltas],
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
            # HEDGING to match the backtest venue (each entry is a separate position,
            # not netted into one per instrument).
            oms_type="HEDGING",
            # L2 (market-by-price) book so market orders walk real depth and fill the
            # FULL size with realistic slippage, instead of being capped at the L1
            # top-of-book size (which left $2000 orders only partially filled on thin
            # names). The strategy feeds the book via order book deltas (Binance
            # partial depth @20 levels). Trades aren't needed for execution and arrive
            # with offset timestamps that trigger "Skipping stale trade", so keep them
            # off. (L3/per-order depth isn't published by Binance.)
            book_type="L2_MBP",
            # 10x leverage to match the backtest venue (and give the 50-position cap
            # headroom: 50 x $2000 = $100k notional -> only $10k margin at 10x, vs the
            # full $100k balance at 1x which would start rejecting orders at the cap).
            default_leverage=Decimal(10),
            trade_execution=False,
        ),
    },
    timeout_connection=30.0,
    timeout_reconciliation=10.0,
    timeout_portfolio=10.0,
    timeout_disconnection=30.0,
    timeout_post_stop=30.0,
)

"""       
         BINANCE_FUTURES: BinanceExecClientConfig(
            venue=Venue(BINANCE_FUTURES),
            account_type=BinanceAccountType.USDT_FUTURES,
            environment=BinanceEnvironment.DEMO,
            instrument_provider=BinanceInstrumentProviderConfig(
                load_all=True,
            ),
        ),
"""