"""Backtest a live strategy over Binance 1m klines and publish the results to the
``dashboard:backtest:*`` namespace (Redis DB 1) for the dashboard's test mode.

Runs in a background thread inside the node so it can reuse the live cache's
instrument definitions. Separate engine, separate Redis DB: it never touches
live trading state.

Robustness notes
* Klines are paced by the weight Binance reports (X-MBX-USED-WEIGHT-1M) and
  retried with backoff. The old fetch stopped at the first error and ran the
  backtest on whatever it had, so a rate limit silently truncated history.
  Instruments that still fail are listed in the result and the run's meta.
* The strategy under test is the live strategy's own class, configured by its
  ``backtest_config`` hook. It used to be hard-coded to EMACrossStopReverse no
  matter which strategy was chosen.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import requests
from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import AccountType, BookType, OmsType
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Money

from trading import redis_io as K
from trading.strategies.chart_protocol import backtest_config_for, chart_indicators_for, chart_spec_for

FAPI_KLINES = "https://fapi.binance.com/fapi/v1/klines"
BT = K.BACKTEST
_VENUE = "BINANCE_FUTURES"
_WEIGHT_BUDGET = 1800          # of Binance's 2400/min, leaving room for the live node
_session = requests.Session()


class KlineError(RuntimeError):
    pass


def _get(params: dict, attempts: int = 5) -> list:
    delay = 1.0
    for attempt in range(1, attempts + 1):
        try:
            resp = _session.get(FAPI_KLINES, params=params, timeout=20)
        except requests.RequestException as e:
            err = f"network: {e}"
        else:
            used = int(resp.headers.get("X-MBX-USED-WEIGHT-1M", 0) or 0)
            if resp.status_code in (418, 429):
                wait = float(resp.headers.get("Retry-After", 0) or 0) or 60 - datetime.now().second + 1
                err = f"HTTP {resp.status_code} rate limited; waiting {wait:.0f}s"
                time.sleep(min(wait, 120))
                continue
            if resp.status_code == 400:
                raise KlineError(f"rejected: {resp.text[:200]}")     # bad symbol: retrying won't help
            if resp.ok:
                if used > _WEIGHT_BUDGET:
                    time.sleep(60 - datetime.now().second + 1)       # let the 1-minute window roll
                return resp.json()
            err = f"HTTP {resp.status_code}: {resp.text[:200]}"
        if attempt < attempts:
            time.sleep(delay)
            delay = min(delay * 2, 30)
    raise KlineError(f"gave up after {attempts} attempts ({err})")


def fetch_klines(symbol: str, start_ms: int, end_ms: int) -> list:
    """All USDT-M 1m klines in [start, end]. Raises KlineError instead of
    returning a silently short series."""
    out: list = []
    cur = start_ms
    while cur < end_ms:
        data = _get({"symbol": symbol, "interval": "1m", "startTime": cur, "endTime": end_ms, "limit": 1500})
        if not data:
            break
        out.extend(data)
        if len(data) < 1500:
            break
        cur = int(data[-1][0]) + 60_000
    return out


def _kline_to_bar(bar_type: BarType, instrument, k: list) -> Bar:
    ts = int(k[6]) * 1_000_000          # close time: klines are close-stamped, like live
    return Bar(bar_type=bar_type,
               open=instrument.make_price(Decimal(k[1])), high=instrument.make_price(Decimal(k[2])),
               low=instrument.make_price(Decimal(k[3])), close=instrument.make_price(Decimal(k[4])),
               volume=instrument.make_qty(Decimal(k[5])), ts_event=ts, ts_init=ts)


def _meta(r, **kw) -> None:
    r.set(f"{BT}:meta", json.dumps(kw))


def run_backtest(r, live_strategy, instruments, days: int = 4,
                 start_date: str | None = None, end_date: str | None = None) -> dict:
    """Fetch, run, publish. Returns {"closed", "pnl", "skipped"}; raises on failure
    (after recording status=error in the meta so the dashboard shows it)."""
    sid = str(live_strategy.id)
    started = time.time()
    try:
        return _run(r, live_strategy, instruments, days, start_date, end_date, sid, started)
    except Exception as e:
        _meta(r, status="error", strategy=sid, error=f"{type(e).__name__}: {e}", started=started)
        raise


def _run(r, live_strategy, instruments, days, start_date, end_date, sid, started) -> dict:
    now = datetime.now(timezone.utc)
    start = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc) if start_date else now - timedelta(days=days)
    end = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc) if end_date else now
    if end_date and end <= start:
        raise ValueError(f"end date {end_date} is not after start date {start_date}")
    if start >= now:
        raise ValueError(f"start date {start_date} is in the future")
    end = min(end, now)
    start_ms, end_ms = int(start.timestamp() * 1000), int(end.timestamp() * 1000)
    if not instruments:
        raise ValueError("none of the strategy's instruments are loaded in the node")

    cfg = backtest_config_for(live_strategy)
    cls = type(live_strategy)
    meta = dict(status="running", strategy=sid, n=len(instruments), done=0, started=started,
                start_date=start.date().isoformat(), end_date=end.date().isoformat(),
                live_handoff=end_date is None)
    pipe = r.pipeline(transaction=False)
    for pattern in (f"{BT}:bars:*", f"{BT}:indicators:*", f"{K.TEST_CHART}*"):
        for key in r.scan_iter(match=pattern, count=1000):
            pipe.delete(key)
    pipe.execute()
    _meta(r, **meta)

    engine = BacktestEngine(config=BacktestEngineConfig(trader_id="BACKTESTER-001",
                                                        logging=LoggingConfig(log_level="ERROR")))
    engine.add_venue(venue=Venue(_VENUE), oms_type=OmsType.HEDGING, account_type=AccountType.MARGIN,
                     base_currency=None, starting_balances=[Money(K.ACCOUNT_START, USDT)],
                     default_leverage=Decimal(10), book_type=BookType.L1_MBP, bar_execution=True)

    all_bars, skipped, last_px, price_series = [], [], {}, {}
    for i, inst in enumerate(instruments):
        sym = inst.id.symbol.value.replace("-PERP", "")
        try:
            klines = fetch_klines(sym, start_ms, end_ms)
        except KlineError as e:
            skipped.append(f"{sym} ({e})")
            klines = []
        if not klines:
            if not any(s.startswith(sym + " ") for s in skipped):
                skipped.append(f"{sym} (no klines in range)")
            meta.update(done=i + 1, skipped=skipped)
            _meta(r, **meta)
            continue
        bar_type = BarType.from_str(f"{inst.id}-1-MINUTE-LAST-EXTERNAL")
        engine.add_instrument(inst)
        inds = chart_indicators_for(live_strategy)
        bars_json, ind_json = [], []
        for k in klines:
            bar = _kline_to_bar(bar_type, inst, k)
            all_bars.append(bar)
            # bars: close time to the ms (...:59.999), the key live Redis and
            # Postgres bars use, so the chart merge dedupes instead of doubling
            # candles; indicator points: whole seconds, like the live stream
            bars_json.append({"t": round(int(k[6]) / 1000, 3), "o": float(k[1]), "h": float(k[2]),
                              "l": float(k[3]), "c": float(k[4]), "v": float(k[5])})
            for ind in inds.values():
                ind.handle_bar(bar)
            vals = {name: round(ind.value, 8) for name, ind in inds.items() if ind.initialized}
            if vals:
                ind_json.append({"ts": int(k[6]) // 1000, **vals})
        last_px[str(inst.id)] = bars_json[-1]["c"]
        price_series[str(inst.id)] = [(b["t"], b["c"]) for b in bars_json]
        blob = json.dumps(bars_json)
        pipe = r.pipeline(transaction=False)
        pipe.set(f"{BT}:bars:{inst.id}", blob)
        pipe.set(f"{BT}:indicators:{inst.id}", json.dumps(ind_json))
        pipe.set(f"{K.TEST_CHART}{inst.id}-1-MINUTE-LAST-EXTERNAL", blob)
        pipe.execute()
        meta.update(done=i + 1, skipped=skipped)
        _meta(r, **meta)

    if not all_bars:
        raise RuntimeError("no kline data for any instrument: " + "; ".join(skipped[:10]))
    r.set(f"{BT}:chart_spec", json.dumps(chart_spec_for(live_strategy)))
    engine.add_data(all_bars, sort=True)
    engine.add_strategy(cls(cfg))
    meta.update(phase="engine")
    _meta(r, **meta)
    engine.run()

    closed_pos, pnl = [], 0.0
    for p in engine.cache.positions_open():      # force-close at the last bar so every trade is realized
        px = last_px.get(str(p.instrument_id), p.avg_px_open)
        realized = round((px - p.avg_px_open) * p.signed_qty, 4)
        closed_pos.append({
            "id": f"BT-{p.id}", "instrument": str(p.instrument_id), "strategy": sid,
            "side": "LONG" if p.entry.name == "BUY" else "SHORT", "qty": p.quantity.as_double(),
            "avg_px_open": p.avg_px_open, "avg_px_close": round(px, 8), "realized": realized, "ccy": "USDT",
            "ts_opened": p.ts_opened // 1_000_000, "ts_closed": end_ms, "forced_close": True})
        pnl += realized
    for p in engine.cache.positions_closed():
        rp = p.realized_pnl.as_double() if p.realized_pnl is not None else 0.0
        closed_pos.append({
            "id": f"BT-{p.id}", "instrument": str(p.instrument_id), "strategy": sid,
            "side": "LONG" if p.entry.name == "BUY" else "SHORT", "qty": p.peak_qty.as_double(),
            "avg_px_open": p.avg_px_open, "avg_px_close": p.avg_px_close, "realized": rp,
            "ccy": p.realized_pnl.currency.code if p.realized_pnl is not None else "USDT",
            "ts_opened": p.ts_opened // 1_000_000, "ts_closed": p.ts_closed // 1_000_000})
        pnl += rp
    closed_pos.sort(key=lambda c: c["ts_closed"], reverse=True)
    pnl_map = {"USDT": {"realized": pnl, "unrealized": 0.0, "total": pnl}}
    r.set(f"{BT}:positions", json.dumps({"positions": [], "closed_positions": closed_pos, "pnl": pnl_map}))

    equity = _equity_curve(closed_pos, price_series, start_ms, end_ms)
    r.set(f"{BT}:equity", json.dumps(equity))
    r.set(f"{BT}:equity:end_nav", str(equity[-1]["nav"]))
    try:
        engine.dispose()
    except Exception:
        pass
    meta.update(status="done", finished=time.time(), open=0, closed=len(closed_pos), pnl=pnl_map,
                start_ms=start_ms, end_ms=end_ms, phase="done")
    _meta(r, **meta)
    print(f"[backtest] {sid}: {len(closed_pos)} closed, pnl {pnl:+.2f}, skipped {len(skipped)}")
    return {"closed": len(closed_pos), "pnl": pnl, "skipped": skipped}


def _equity_curve(closed_pos: list, price_series: dict, start_ms: int, end_ms: int) -> list:
    """Mark-to-market NAV (realized + unrealized) at every 1m close, so the test
    Account tab draws drawdowns while positions are open, like the live curve."""
    sweep = sorted(({"inst": c["instrument"], "signed": c["qty"] if c["side"] == "LONG" else -c["qty"],
                     "open_px": c["avg_px_open"], "realized": c["realized"],
                     "t_open": c["ts_opened"] // 1000, "t_close": c["ts_closed"] // 1000}
                    for c in closed_pos), key=lambda p: p["t_open"])
    all_ts = sorted({t for series in price_series.values() for t, _ in series})
    ptr = dict.fromkeys(price_series, 0)
    last = dict.fromkeys(price_series)
    pts = [{"ts": start_ms, "nav": K.ACCOUNT_START, "realized": 0.0, "unrealized": 0.0, "total": 0.0}]
    realized, open_now, oi = 0.0, [], 0
    for t in all_ts:
        for iid, series in price_series.items():
            p = ptr[iid]
            while p < len(series) and series[p][0] <= t:
                last[iid] = series[p][1]
                p += 1
            ptr[iid] = p
        while oi < len(sweep) and sweep[oi]["t_open"] <= t:
            open_now.append(sweep[oi])
            oi += 1
        still = []
        for pos in open_now:
            if pos["t_close"] <= t:
                realized += pos["realized"]
            else:
                still.append(pos)
        open_now = still
        unreal = sum(pos["signed"] * (last[pos["inst"]] - pos["open_px"])
                     for pos in open_now if last.get(pos["inst"]) is not None)
        pts.append({"ts": int(t * 1000), "nav": round(K.ACCOUNT_START + realized + unreal, 4),
                    "realized": round(realized, 4), "unrealized": round(unreal, 4),
                    "total": round(realized + unreal, 4)})
    final = sum(p["realized"] for p in sweep)
    last_pt = {"ts": end_ms, "nav": round(K.ACCOUNT_START + final, 4), "realized": round(final, 4),
               "unrealized": 0.0, "total": round(final, 4)}
    if pts[-1]["ts"] >= end_ms:
        pts[-1] = last_pt               # keep timestamps strictly increasing
    else:
        pts.append(last_pt)
    return pts
