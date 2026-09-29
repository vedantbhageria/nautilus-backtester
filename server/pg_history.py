"""Price history for the live charts, read from the backtester's Postgres.

The backtester keeps Binance USDT-futures klines in Postgres (``bars``, 1m;
filled by its Data > Sync). The live node only has what it has streamed or
backfilled into Redis. So when a live chart opens, its candles start from the
same Postgres history the backtests ran on, and the node only has to fetch the
gap from the last stored bar to now (see ``app._catch_up``).

Bars use the Redis convention: ``t`` = bar close in epoch seconds
(``...:59.999`` for 1m), so both sources dedupe on the same key.

Connection settings come from PG* env vars. The defaults are the backtester's
own (backtester/backend/db.py), so there's one place to change them.
"""
import importlib.util
import os
import threading

_DB_PY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "backtester", "backend", "db.py")
_db = None
_db_lock = threading.Lock()

# how far back each chart timeframe loads (1m covers the widest chart window, 3d)
HISTORY_DAYS = {"1m": 3, "5m": 7, "15m": 14, "1h": 30}
TF_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600}


def _dbmod():
    global _db
    with _db_lock:
        if _db is None:
            spec = importlib.util.spec_from_file_location("backtester_db", _DB_PY)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _db = mod
    return _db


def pg_symbol(instrument_id: str) -> str | None:
    """BTCUSDT-PERP.BINANCE_FUTURES -> BTCUSDT. Postgres only holds USDT-futures
    klines, so spot (and anything else) has no stored history."""
    sym, _, venue = instrument_id.partition(".")
    if venue != "BINANCE_FUTURES" or not sym.endswith("-PERP"):
        return None
    return sym[:-len("-PERP")]


def probe() -> str | None:
    """None if Postgres answers, else why not (for /api/health). Postgres is
    optional: without it the charts show only what Redis has."""
    try:
        conn = _dbmod().get_conn(_retries=1)
    except Exception as e:
        return f"{type(e).__name__}: {str(e).strip().splitlines()[0] if str(e).strip() else ''}"
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        return None
    except Exception as e:
        return f"{type(e).__name__}: {e}"
    finally:
        conn.close()


def load_bars(instrument_id: str, timeframe: str) -> list[dict]:
    """Stored bars for the chart, oldest first. Returns [] when the symbol or
    timeframe has no Postgres history or the database is unreachable (the
    chart then just shows what Redis has, as before)."""
    sym = pg_symbol(instrument_id)
    step = TF_SECONDS.get(timeframe)
    if sym is None or step is None:
        return []
    days = HISTORY_DAYS[timeframe]
    try:
        conn = _dbmod().get_conn(_retries=1)
    except Exception:
        return []
    try:
        with conn.cursor() as cur:
            if step == 60:
                cur.execute(
                    "SELECT extract(epoch FROM ts), open, high, low, close, volume FROM bars "
                    "WHERE symbol = %s AND ts >= now() - make_interval(days => %s) ORDER BY ts",
                    (sym, days))
            else:
                # aggregate 1m -> 5m/15m/1h on bar OPEN time (close ts - 59.999s),
                # then label each bucket by its close, like the klines Nautilus streams
                cur.execute(
                    """
                    WITH b AS (
                      SELECT to_timestamp(floor(extract(epoch FROM ts) / 60) * 60) AS open_ts,
                             ts, open, high, low, close, volume
                      FROM bars
                      WHERE symbol = %(s)s AND ts >= now() - make_interval(days => %(d)s)
                    )
                    SELECT extract(epoch FROM date_bin(make_interval(secs => %(st)s), open_ts,
                                                       TIMESTAMPTZ '1970-01-01'))
                             + %(st)s - 0.001,
                           (array_agg(open ORDER BY ts))[1], max(high), min(low),
                           (array_agg(close ORDER BY ts DESC))[1], sum(volume)
                    FROM b GROUP BY 1 ORDER BY 1
                    """,
                    {"s": sym, "d": days, "st": step})
            rows = cur.fetchall()
    except Exception:
        return []
    finally:
        conn.close()
    return [{"t": round(float(t), 3), "o": float(o), "h": float(h), "l": float(lo),
             "c": float(c), "v": float(v or 0)} for t, o, h, lo, c, v in rows]
