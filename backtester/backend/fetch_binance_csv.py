
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import csv
import io
import os
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import requests

FAPI_KLINES = "https://fapi.binance.com/fapi/v1/klines"
VISION_BASE = "https://data.binance.vision/data/futures/um"
DAY_MS = 86_400_000
import _paths
DATA_DIR = os.path.join(_paths.ROOT, "datas")
CSV_DT_FORMAT = "%Y-%m-%d %H:%M:%S"
MAX_WORKERS = 16  # vision zips have no rate limit; REST is only the tiny tail

# One requests.Session per worker thread (connection reuse + thread safety).
_local = threading.local()


def _session():
    s = getattr(_local, "session", None)
    if s is None:
        s = _local.session = requests.Session()
    return s

# Symbols from the Nautilus EMACrossStopReverse strategy (_PERP_SYMS_STOP_REVERSES).
SYMBOLS = [
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

# Start-Service postgresql-x64-18


def _fetch_klines(symbol, start_ms, end_ms, interval="1m"):
    out = []
    cur = start_ms
    sess = _session()
    while cur < end_ms:
        data = None
        # Retry with backoff on rate limits (429) / IP bans (418) and transient
        # errors, so parallel workers don't silently give up mid-download.
        for attempt in range(6):
            try:
                resp = sess.get(FAPI_KLINES, params={
                    "symbol": symbol, "interval": interval,
                    "startTime": cur, "endTime": end_ms, "limit": 1500,
                }, timeout=20)
                if resp.status_code in (429, 418):
                    wait = int(resp.headers.get("Retry-After", 2 ** attempt))
                    print("  %s rate-limited (%s), waiting %ds"
                          % (symbol, resp.status_code, wait))
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as e:
                if attempt == 5:
                    print("  %s request failed at %s: %s" % (symbol, cur, e))
                else:
                    time.sleep(2 ** attempt)
        if not data:
            break
        out.extend(data)
        if len(data) < 1500:
            break
        cur = int(data[-1][0]) + 60_000   # next open time
        time.sleep(0.05)                  # be gentle on the REST weight limit
    return out


def fetch_to_csv(symbol, start_ms, end_ms, out_dir=DATA_DIR):
    print("Fetching %s 1m klines %s -> %s ..." % (
        symbol,
        datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).date(),
        datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc).date(),
    ))
    klines = _fetch_klines(symbol, start_ms, end_ms)
    if not klines:
        print("  no data returned for %s" % symbol)
        return None

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "%s-1m.csv" % symbol)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["datetime", "open", "high", "low", "close", "volume"])
        for k in klines:
            # k = [openTime, open, high, low, close, volume, closeTime, ...]
            # CLOSE-timestamped to match the Nautilus EXTERNAL bar convention.
            dt = datetime.fromtimestamp(int(k[6]) / 1000, tz=timezone.utc)
            w.writerow([dt.strftime(CSV_DT_FORMAT), k[1], k[2], k[3], k[4], k[5]])
    print("  wrote %d bars -> %s" % (len(klines), path))
    return path


def _ms_to_dt(ms):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def _missing_ranges(cov, start_ms, end_ms, interval='1m'):

    bar_ms = 3_600_000 if interval == '1h' else 60_000
    mn, mx, cnt = cov
    if mn is None:
        return [(start_ms, end_ms)]
    mn_ms = int(mn.timestamp() * 1000)
    mx_ms = int(mx.timestamp() * 1000)
    expected = (mx_ms - mn_ms) // bar_ms + 1
    if cnt < expected:
        return [(start_ms, end_ms)]
    out = []
    if mn_ms - start_ms > bar_ms:
        out.append((start_ms, mn_ms - 1))
    if end_ms - mx_ms > bar_ms:
        out.append((mx_ms - bar_ms, end_ms))
    return out


def _vision_get(url):

    sess = _session()
    for attempt in range(4):
        try:
            resp = sess.get(url, timeout=60)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.content
        except Exception:
            if attempt == 3:
                raise
            time.sleep(1.5 ** attempt)


def _zip_rows(content, now_ms):

    rows = []
    zf = zipfile.ZipFile(io.BytesIO(content))
    with zf.open(zf.namelist()[0]) as f:
        for line in io.TextIOWrapper(f, 'utf-8'):
            parts = line.rstrip('\n').split(',')
            if len(parts) < 7 or not parts[0][:1].isdigit():
                continue                       # header / blank line
            close_ms = int(parts[6])
            if close_ms > 10**14:              # microseconds -> ms
                close_ms //= 1000
            if close_ms > now_ms:
                continue
            rows.append((_ms_to_dt(close_ms), float(parts[1]), float(parts[2]),
                         float(parts[3]), float(parts[4]), float(parts[5])))
    return rows


def _month_days(year, month, today0_ms):
  
    d = datetime(year, month, 1, tzinfo=timezone.utc)
    out = []
    while d.month == month and int(d.timestamp() * 1000) + DAY_MS <= today0_ms:
        out.append(d.date())
        d += timedelta(days=1)
    return out


def _plan_range(s_ms, e_ms, now_ms):

    today0 = (now_ms // DAY_MS) * DAY_MS
    months, days, rest = [], [], []
    v_end = min(e_ms, today0 - 1)
    d = (s_ms // DAY_MS) * DAY_MS
    while d <= v_end:
        dt = _ms_to_dt(d)
        if dt.day == 1:
            nxt = (dt.replace(day=28) + timedelta(days=4)).replace(day=1)
            m_end = int(nxt.timestamp() * 1000) - 1
            if m_end <= v_end:                 # whole month inside the range
                months.append((dt.year, dt.month))
                d = m_end + 1
                continue
        days.append(dt.date())
        d += DAY_MS
    if e_ms >= today0:
        rest.append((max(s_ms, today0), e_ms))
    return months, days, rest


def _download(symbol, ranges, now_ms, interval="1m"):

    today0 = (now_ms // DAY_MS) * DAY_MS
    rows = []
    for s, e in ranges:
        months, days, rest = _plan_range(s, e, now_ms)
        rest = list(rest)
        for (y, m) in months:
            content = _vision_get(
                "%s/monthly/klines/%s/%s/%s-%s-%04d-%02d.zip"
                % (VISION_BASE, symbol, interval, symbol, interval, y, m))
            if content is None:
                # Monthly not published (or listed mid-month): use dailies.
                days.extend(_month_days(y, m, today0))
            else:
                rows.extend(_zip_rows(content, now_ms))
        for day in days:
            content = _vision_get(
                "%s/daily/klines/%s/%s/%s-%s-%s.zip"
                % (VISION_BASE, symbol, interval, symbol, interval, day.isoformat()))
            if content is not None:
                rows.extend(_zip_rows(content, now_ms))
            else:
                # Vision hasn't published this day yet 
                d0 = int(datetime(day.year, day.month, day.day,
                                  tzinfo=timezone.utc).timestamp() * 1000)
                rest.append((d0, min(d0 + DAY_MS - 1, now_ms)))
        for rs, re_ in rest:
            for k in _fetch_klines(symbol, rs, re_, interval):
                close_ms = int(k[6])
                if close_ms > now_ms:          # drop the still-forming candle
                    continue
                rows.append((_ms_to_dt(close_ms), float(k[1]), float(k[2]),
                             float(k[3]), float(k[4]), float(k[5])))
    return rows


def main(days=180, interval='1m'):

    days = max(1, int(days))
    interval = interval if interval in ('1m', '1h') else '1m'
    started = time.time()
    now = datetime.now(timezone.utc)
    end_ms = int(now.timestamp() * 1000)
    start_ms = end_ms - days * 86_400_000
    start_dt, end_dt = _ms_to_dt(start_ms), _ms_to_dt(end_ms)

    conn = None
    try:
        import db
        conn = db.get_conn()
        db.init_schema(conn)
    except Exception as e:
        print("DB unavailable (%s) — falling back to direct CSV download" % e)
        if conn is not None:
            try:
                conn.close()   # a half-initialized/aborted connection is unusable
            except Exception:
                pass
            conn = None

    if conn is None:
        # No database: old behavior, full download straight to CSVs.
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(fetch_to_csv, s.upper(), start_ms, end_ms): s
                       for s in SYMBOLS}
            for fut in as_completed(futures):
                fut.result()
        print("Done (no DB) in %.1fs" % (time.time() - started))
        return {'db': False, 'days': days, 'interval': interval,
                'elapsed': round(time.time() - started, 1)}


    need = {}
    for s in SYMBOLS:
        sym = s.upper()
        ranges = _missing_ranges(db.coverage(conn, sym, start_dt, end_dt, interval),
                                 start_ms, end_ms, interval)
        if ranges:
            need[sym] = ranges
    print("[%s] %d/%d symbols need downloading (rest already in postgres)"
          % (interval, len(need), len(SYMBOLS)))


    new_bars, done = 0, 0
    if need:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(_download, sym, ranges, end_ms, interval): sym
                       for sym, ranges in need.items()}
            for fut in as_completed(futures):
                sym = futures[fut]
                done += 1
                try:
                    n = db.insert_bars(conn, sym, fut.result(), interval)
                    new_bars += n
                    print("  [%d/%d] %s: +%d bars" % (done, len(need), sym, n))
                except Exception as e:
                    print("  [%d/%d] %s FAILED: %s" % (done, len(need), sym, e))


    short_history = []
    try:
        for sym, mn, _mx, _cnt in db.per_symbol_span(conn, interval):
            if mn is not None and (mn - start_dt).total_seconds() > 86400:
                short_history.append([sym, mn.strftime('%Y-%m-%d')])
    except Exception as e:
        print("short-history scan failed: %s" % e)
    conn.close()


    print("Inserted/updated %d bars in postgres in %.1fs"
          % (new_bars, time.time() - started))
    if short_history:
        print("NOTE: %d symbol(s) have no data before their listing date: %s"
              % (len(short_history),
                 ', '.join('%s(%s)' % (s, d) for s, d in short_history[:10])))
    return {'db': True, 'days': days, 'interval': interval, 'symbols_needed': len(need),
            'symbols_total': len(SYMBOLS), 'new_bars': new_bars,
            'short_history': short_history,
            'elapsed': round(time.time() - started, 1)}


if __name__ == "__main__":
    main()
