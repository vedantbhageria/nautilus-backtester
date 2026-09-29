

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import csv
import json
import os
import time

import psycopg2
from psycopg2.extras import Json, execute_values
# Start-Service postgresql-x64-18
DB = {
    'host': os.getenv('PGHOST', 'localhost'),
    'port': int(os.getenv('PGPORT', '5432')),
    'user': os.getenv('PGUSER', 'postgres'),
    'password': os.getenv('PGPASSWORD', 'Forctix@0609'),
}
DB_NAME = os.getenv('PGDATABASE', 'backtest')

# Bars (bars / bars_1h) are shared with the backtrader app: same database,
# same market data. Run history goes to nt_-prefixed tables so Nautilus
# results never mix with backtrader's runs/trades/equity/per_symbol.
#
# Hourly bars live in their own table (bars_1h) rather than an `interval`
# column on `bars`, so the existing 26M-row minute table needs no PK
# migration and 1h/1m never collide on a shared (symbol, ts) key.
_INTERVAL_TABLE = {'1m': 'bars', '1h': 'bars_1h'}


def _bars_table(interval):
    try:
        return _INTERVAL_TABLE[interval or '1m']
    except KeyError:
        raise ValueError('unsupported interval %r (want one of %s)'
                         % (interval, list(_INTERVAL_TABLE)))


_SCHEMA = """
CREATE TABLE IF NOT EXISTS bars (
    symbol  text             NOT NULL,
    ts      timestamptz      NOT NULL,
    open    double precision NOT NULL,
    high    double precision NOT NULL,
    low     double precision NOT NULL,
    close   double precision NOT NULL,
    volume  double precision NOT NULL,
    PRIMARY KEY (symbol, ts)
);

CREATE TABLE IF NOT EXISTS bars_1h (
    symbol  text             NOT NULL,
    ts      timestamptz      NOT NULL,
    open    double precision NOT NULL,
    high    double precision NOT NULL,
    low     double precision NOT NULL,
    close   double precision NOT NULL,
    volume  double precision NOT NULL,
    PRIMARY KEY (symbol, ts)
);

CREATE TABLE IF NOT EXISTS nt_runs (
    id            serial PRIMARY KEY,
    generated     timestamptz NOT NULL DEFAULT now(),
    run_tag       text,
    params        jsonb,
    summary       jsonb,
    pyfolio_stats jsonb
);
-- runs predating the run_tag column (added for durable artifact linking —
-- timestamp matching cross-linked parallel runs that finished microseconds
-- apart); NULL run_tag rows fall back to timestamp matching
ALTER TABLE nt_runs ADD COLUMN IF NOT EXISTS run_tag text;

CREATE TABLE IF NOT EXISTS nt_trades (
    run_id  integer REFERENCES nt_runs(id) ON DELETE CASCADE,
    symbol  text NOT NULL,
    ts      timestamptz NOT NULL,
    pnl     double precision,
    pnlcomm double precision
);
CREATE INDEX IF NOT EXISTS nt_trades_run_idx ON nt_trades (run_id, symbol);

CREATE TABLE IF NOT EXISTS nt_equity (
    run_id integer REFERENCES nt_runs(id) ON DELETE CASCADE,
    ts     timestamptz NOT NULL,
    value  double precision NOT NULL
);
CREATE INDEX IF NOT EXISTS nt_equity_run_idx ON nt_equity (run_id, ts);

CREATE TABLE IF NOT EXISTS nt_per_symbol (
    run_id integer REFERENCES nt_runs(id) ON DELETE CASCADE,
    symbol text NOT NULL,
    trades integer,
    pnl    double precision,
    won    integer,
    PRIMARY KEY (run_id, symbol)
);
"""


def get_conn(_retries=3, _retry_delay=0.6):
    """Connect to Postgres, creating DB_NAME if it doesn't exist yet.

    On Windows the postgresql service can still be finishing its own startup
    (or hasn't been started at all) at the moment this app boots, so a
    'connection refused' here doesn't necessarily mean postgres is down for
    good — retry a few times with a short backoff before giving up. If it's
    genuinely not running, this still raises, and the caller's message should
    tell you to start it (Windows: `Start-Service postgresql-x64-18` from an
    elevated PowerShell, or set the service to Automatic in services.msc)."""
    last_err = None
    for attempt in range(_retries):
        try:
            return psycopg2.connect(dbname=DB_NAME, **DB)
        except psycopg2.OperationalError as e:
            if 'does not exist' in str(e):
                break   # DB missing (not a connectivity problem) -> create it below
            last_err = e
            if attempt < _retries - 1:
                time.sleep(_retry_delay)
    else:
        raise last_err
    # Target DB missing: connect to the maintenance DB and create it.
    admin = psycopg2.connect(dbname='postgres', **DB)
    admin.autocommit = True
    with admin.cursor() as cur:
        cur.execute('CREATE DATABASE "%s"' % DB_NAME)
    admin.close()
    return psycopg2.connect(dbname=DB_NAME, **DB)


def init_schema(conn):
    
    with conn.cursor() as cur:
        cur.execute("CREATE SCHEMA IF NOT EXISTS public")
        cur.execute("SET search_path TO public")
        conn.commit()

        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")
            conn.commit()
            timescale = True
        except psycopg2.Error:
            conn.rollback()
            timescale = False

        cur.execute(_SCHEMA)
        conn.commit()

        if timescale:
            for table, col in (('bars', 'ts'), ('bars_1h', 'ts'), ('nt_equity', 'ts')):
                try:
                    cur.execute(
                        "SELECT create_hypertable(%s, %s, "
                        "if_not_exists => TRUE, migrate_data => TRUE)",
                        (table, col))
                    conn.commit()
                except psycopg2.Error:
                    conn.rollback()  # plain table still works
    return timescale


def load_bars_csv(conn, symbol, csv_path):

    from datetime import datetime as _dt, timezone as _tz
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            ts = _dt.strptime(r['datetime'], '%Y-%m-%d %H:%M:%S').replace(
                tzinfo=_tz.utc, second=59, microsecond=999000)
            rows.append((symbol, ts, float(r['open']), float(r['high']),
                         float(r['low']), float(r['close']), float(r['volume'])))
    if not rows:
        return 0
    with conn.cursor() as cur:
        execute_values(
            cur,
            "INSERT INTO bars (symbol, ts, open, high, low, close, volume) "
            "VALUES %s ON CONFLICT (symbol, ts) DO NOTHING",
            rows, page_size=2000)
    conn.commit()
    return len(rows)


def symbols(conn, interval='1m'):
    tbl = _bars_table(interval)
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT symbol FROM %s ORDER BY symbol" % tbl)
        return [r[0] for r in cur.fetchall()]


def bars_span(conn, interval='1m'):

    tbl = _bars_table(interval)
    with conn.cursor() as cur:
        cur.execute("SELECT min(ts), max(ts) FROM %s" % tbl)
        return cur.fetchone()


def clear_run_history(conn):

    with conn.cursor() as cur:
        for tbl in ('nt_trades', 'nt_equity', 'nt_per_symbol', 'nt_runs'):
            cur.execute('DELETE FROM %s' % tbl)
    conn.commit()


def delete_run(conn, run_id):
    """Remove one run (and its trades/equity/per_symbol rows) from postgres."""
    with conn.cursor() as cur:
        for tbl in ('nt_trades', 'nt_equity', 'nt_per_symbol'):
            cur.execute('DELETE FROM %s WHERE run_id=%%s' % tbl, (run_id,))
        cur.execute('DELETE FROM nt_runs WHERE id=%s', (run_id,))
        deleted = cur.rowcount
    conn.commit()
    return deleted > 0


def per_symbol_span(conn, interval='1m'):

    tbl = _bars_table(interval)
    with conn.cursor() as cur:
        cur.execute("SELECT symbol, min(ts), max(ts), count(*) FROM %s "
                    "GROUP BY symbol ORDER BY symbol" % tbl)
        return cur.fetchall()


def coverage(conn, symbol, start, end, interval='1m'):

    tbl = _bars_table(interval)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT min(ts), max(ts), count(*) FROM %s "
            "WHERE symbol=%%s AND ts >= %%s AND ts <= %%s" % tbl,
            (symbol, start, end))
        return cur.fetchone()


def data_integrity(conn, interval='1m', spike_pct=0.5):
    """Full per-symbol data-quality scan for `interval`. Beyond the span/gap
    counts (per_symbol_span only covers those), this also catches BAD VALUES
    that a gap check can't see — non-positive/null OHLC, high<low, zero/negative
    volume, and suspicious >spike_pct single-bar close jumps — plus the stored
    price range. Returns {symbol: {span/value metrics}}. One pass of a few
    grouped queries (window function for the spike detect)."""
    tbl = _bars_table(interval)
    out = {}
    with conn.cursor() as cur:
        # span + total bars
        cur.execute("SELECT symbol, min(ts), max(ts), count(*) FROM %s "
                    "GROUP BY symbol" % tbl)
        for sym, mn, mx, cnt in cur.fetchall():
            out[sym] = {'first': mn, 'last': mx, 'bars': cnt,
                        'bad_ohlc': 0, 'hl_viol': 0, 'zero_vol': 0,
                        'neg_vol': 0, 'spikes': 0,
                        'min_close': None, 'max_close': None}
        # value integrity + price range (aggregate, cheap)
        cur.execute("""SELECT symbol,
              count(*) FILTER (WHERE open<=0 OR high<=0 OR low<=0 OR close<=0
                    OR open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL),
              count(*) FILTER (WHERE high < low),
              count(*) FILTER (WHERE volume = 0),
              count(*) FILTER (WHERE volume < 0),
              min(close), max(close)
            FROM %s GROUP BY symbol""" % tbl)
        for sym, bad, hl, zv, nv, mnc, mxc in cur.fetchall():
            if sym in out:
                out[sym].update(bad_ohlc=bad, hl_viol=hl, zero_vol=zv,
                                neg_vol=nv, min_close=mnc, max_close=mxc)
        # suspicious single-bar jumps (window function)
        cur.execute("""WITH j AS (
              SELECT symbol, close,
                     lag(close) OVER (PARTITION BY symbol ORDER BY ts) AS prev
              FROM %s)
            SELECT symbol, count(*) FILTER (
                WHERE prev IS NOT NULL AND prev > 0 AND abs(close/prev - 1) > %%s)
            FROM j GROUP BY symbol""" % tbl, (spike_pct,))
        for sym, sp in cur.fetchall():
            if sym in out:
                out[sym]['spikes'] = sp
    return out


def gap_ranges(conn, symbol, interval='1m', limit=500):
    """Actual [start, end] gap spans for `symbol` — not just a count — so a
    chart can shade the holes directly. `start`/`end` are the two bars that
    bracket the hole (the missing bars sit strictly between them). One pass,
    same lag-window shape as spike_bars. A hole must be > 1.5 bar-intervals
    to count (skips ordinary single-bar jitter)."""
    tbl = _bars_table(interval)
    bar_s = 3600 if interval == '1h' else 60
    with conn.cursor() as cur:
        cur.execute("""WITH j AS (
              SELECT ts, lag(ts) OVER (ORDER BY ts) AS prev
              FROM %s WHERE symbol = %%s)
            SELECT prev, ts FROM j
            WHERE prev IS NOT NULL AND EXTRACT(EPOCH FROM (ts - prev)) > %%s * 1.5
            ORDER BY prev LIMIT %%s""" % tbl, (symbol, bar_s, limit))
        rows = cur.fetchall()
    out = []
    for prev, ts in rows:
        missing = int((ts - prev).total_seconds() // bar_s) - 1
        out.append((prev, ts, missing))
    return out


def bars_range(conn, symbol, start, end, interval='1m'):
    """Stored bars for `symbol` between start/end (inclusive) as plain rows
    — used by the data-fix chart and the redownload/compare endpoint (the
    JSON-friendly sibling of export_bars_csv)."""
    tbl = _bars_table(interval)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT ts, open, high, low, close, volume FROM %s "
            "WHERE symbol=%%s AND ts >= %%s AND ts <= %%s ORDER BY ts" % tbl,
            (symbol, start, end))
        return cur.fetchall()


def spike_bars(conn, symbol, interval='1m', spike_pct=0.5, limit=500):
    """The actual bars where |close/prev_close - 1| > spike_pct for one symbol
    — so a flagged symbol can be inspected bar-by-bar (real move vs bad tick).
    Returns [{ts, prev_close, close, pct, high, low, volume}] oldest-first."""
    tbl = _bars_table(interval)
    with conn.cursor() as cur:
        cur.execute("""WITH j AS (
              SELECT ts, high, low, close, volume,
                     lag(close) OVER (ORDER BY ts) AS prev
              FROM %s WHERE symbol = %%s)
            SELECT ts, prev, close, high, low, volume FROM j
            WHERE prev IS NOT NULL AND prev > 0 AND abs(close/prev - 1) > %%s
            ORDER BY ts LIMIT %%s""" % tbl, (symbol, spike_pct, limit))
        return cur.fetchall()


def insert_bars(conn, symbol, rows, interval='1m'):

    if not rows:
        return 0
    tbl = _bars_table(interval)
    with conn.cursor() as cur:
        execute_values(
            cur,
            "INSERT INTO %s (symbol, ts, open, high, low, close, volume) "
            "VALUES %%s ON CONFLICT (symbol, ts) DO UPDATE SET "
            "open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low, "
            "close=EXCLUDED.close, volume=EXCLUDED.volume" % tbl,
            [(symbol, r[0].replace(second=59, microsecond=999000)) + tuple(r[1:])
             for r in rows], page_size=10000)
    conn.commit()
    return len(rows)


def export_bars_csv(conn, symbol, start, end, path,
                    dt_format='%Y-%m-%d %H:%M:%S', interval='1m'):
 
    tbl = _bars_table(interval)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT ts, open, high, low, close, volume FROM %s "
            "WHERE symbol=%%s AND ts >= %%s AND ts <= %%s ORDER BY ts" % tbl,
            (symbol, start, end))
        rows = cur.fetchall()
    if not rows:
        return 0
    from datetime import timezone as _tz
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['datetime', 'open', 'high', 'low', 'close', 'volume'])
        for ts, o, h, l, c, v in rows:
            w.writerow([ts.astimezone(_tz.utc).strftime(dt_format),
                        repr(o), repr(h), repr(l), repr(c), repr(v)])
    return len(rows)


def save_run(conn, results, trade_log, equity):
    """Persist one backtest run (results.json content + full trade/equity)."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO nt_runs (generated, run_tag, params, summary, pyfolio_stats) "
            "VALUES (%s, %s, %s, %s, %s) RETURNING id",
            (results['generated'], results.get('run_tag'),
             Json(results['params']),
             Json(results['summary']),
             Json(results.get('pyfolio', {}).get('stats') or {})))
        run_id = cur.fetchone()[0]

        if trade_log:
            # `ts` = position close time (trade_log now uses exit_dt).
            execute_values(
                cur,
                "INSERT INTO nt_trades (run_id, symbol, ts, pnl, pnlcomm) VALUES %s",
                [(run_id, t['symbol'], t.get('exit_dt') or t.get('dt'),
                  t['pnl'], t['pnlcomm'])
                 for t in trade_log], page_size=2000)

        if equity:
            execute_values(
                cur,
                "INSERT INTO nt_equity (run_id, ts, value) VALUES %s",
                [(run_id, ts, v) for ts, v in equity], page_size=2000)

        per = results.get('per_symbol') or {}
        if per:
            execute_values(
                cur,
                "INSERT INTO nt_per_symbol (run_id, symbol, trades, pnl, won) VALUES %s",
                [(run_id, s, v['trades'], v['pnl'], v['won'])
                 for s, v in per.items()])
    conn.commit()
    return run_id


def list_runs(conn, limit=50):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, generated, run_tag, params, summary FROM nt_runs "
            "ORDER BY id DESC LIMIT %s", (limit,))
        return [{'id': i, 'generated': g.isoformat(), 'run_tag': rt,
                 'params': p, 'summary': s}
                for i, g, rt, p, s in cur.fetchall()]


if __name__ == '__main__':
    conn = get_conn()
    ts = init_schema(conn)
    print('schema ready (timescaledb=%s) at %s:%s/%s'
          % (ts, DB['host'], DB['port'], DB_NAME))
    print(json.dumps(list_runs(conn, 5), indent=1, default=str))
    conn.close()
