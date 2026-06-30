"""Run a Nautilus BacktestEngine for a strategy over the past N days of Binance
1-minute klines, and write the results to a `dashboard:backtest:*` Redis namespace
for the dashboard to render (chart overlay + positions tab + PnL).

Runs in a background thread inside the live node process, so it can pull instrument
definitions from the live cache. It never touches the live trading state — separate
engine, separate Redis keys.
"""
import json
import time
import traceback
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import requests
from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.indicators import ExponentialMovingAverage
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import AccountType, BookType, OmsType
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Money

from trading.strategies.EMACrossShortTest import EMACrossStopReverse, EMACrossSARConfig

FAPI_KLINES = "https://fapi.binance.com/fapi/v1/klines"
BT = "dashboard:backtest"          # Redis key namespace for backtest results
_VENUE = "BINANCE_FUTURES"


def _fetch_klines(symbol: str, start_ms: int, end_ms: int) -> list:
    """Page through Binance USDT-M 1m klines [start, end]. symbol e.g. 'ETHUSDT'."""
    out: list = []
    cur = start_ms
    while cur < end_ms:
        try:
            resp = requests.get(FAPI_KLINES, params={
                "symbol": symbol, "interval": "1m",
                "startTime": cur, "endTime": end_ms, "limit": 1500,
            }, timeout=20)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            break
        if not data:
            break
        out.extend(data)
        if len(data) < 1500:
            break
        cur = int(data[-1][0]) + 60_000   # next open time
        time.sleep(0.05)                  # be gentle on the REST weight limit
    return out


def _kline_to_bar(bar_type: BarType, instrument, k: list) -> Bar:
    # k = [openTime, open, high, low, close, volume, closeTime, ...]; EXTERNAL klines
    # are CLOSE-timestamped (matches the live convention the dashboard expects).
    ts = int(k[6]) * 1_000_000   # closeTime ms -> ns
    return Bar(
        bar_type=bar_type,
        open=instrument.make_price(Decimal(k[1])),
        high=instrument.make_price(Decimal(k[2])),
        low=instrument.make_price(Decimal(k[3])),
        close=instrument.make_price(Decimal(k[4])),
        volume=instrument.make_qty(Decimal(k[5])),
        ts_event=ts,
        ts_init=ts,
    )


def _set_meta(r, **kw):
    try:
        r.set(f"{BT}:meta", json.dumps(kw))
    except Exception:
        pass


def run_backtest(r, live_strategy, instruments,
                 days: int = 4,
                 start_date: str | None = None,
                 end_date: str | None = None) -> None:
    """Fetch data, run the engine, write results. `r` is a sync Redis client.

    start_date / end_date: ISO-8601 date strings (YYYY-MM-DD).
    If end_date is None the range ends NOW and the caller is responsible for
    starting the live strategy after this function returns (live handoff).
    """
    cfg_live = live_strategy.config
    sid = str(live_strategy.id)
    started = time.time()
    try:
        if start_date:
            start = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
        else:
            start = datetime.now(timezone.utc) - timedelta(days=days)
        if end_date:
            end = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)
        else:
            end = datetime.now(timezone.utc)
        start_ms, end_ms = int(start.timestamp() * 1000), int(end.timestamp() * 1000)
        actual_days = max(1, int((end - start).total_seconds() / 86400))

        # Clear any previous run's per-instrument series.
        for key in r.scan_iter(match=f"{BT}:bars:*", count=1000):
            r.delete(key)
        for key in r.scan_iter(match=f"{BT}:indicators:*", count=1000):
            r.delete(key)

        _set_meta(r, status="running", strategy=sid, days=actual_days,
                  n=len(instruments), done=0, started=started,
                  start_date=start.date().isoformat(), end_date=end.date().isoformat(),
                  live_handoff=(end_date is None))

        engine = BacktestEngine(config=BacktestEngineConfig(
            trader_id="BACKTESTER-001",
            logging=LoggingConfig(log_level="ERROR"),
        ))
        engine.add_venue(
            venue=Venue(_VENUE),
            oms_type=OmsType.HEDGING,
            account_type=AccountType.MARGIN,
            base_currency=None,
            starting_balances=[Money(100_000, USDT)],
            default_leverage=Decimal(10),
            book_type=BookType.L1_MBP,
            bar_execution=True,
        )

        all_bars: list = []
        pipe = r.pipeline()
        for i, inst in enumerate(instruments):
            sym = inst.id.symbol.value.replace("-PERP", "")   # ETHUSDT-PERP -> ETHUSDT
            klines = _fetch_klines(sym, start_ms, end_ms)
            if not klines:
                _set_meta(r, status="running", strategy=sid, days=actual_days,
                          n=len(instruments), done=i + 1, started=started)
                continue
            bar_type = BarType.from_str(f"{inst.id}-1-MINUTE-LAST-EXTERNAL")
            engine.add_instrument(inst)
            fast = ExponentialMovingAverage(cfg_live.fast_ema_period)
            slow = ExponentialMovingAverage(cfg_live.slow_ema_period)
            bars_json, ema_json = [], []
            for k in klines:
                bar = _kline_to_bar(bar_type, inst, k)
                all_bars.append(bar)
                fast.handle_bar(bar); slow.handle_bar(bar)
                t = int(k[6]) // 1000   # close time seconds (dashboard floors to open)
                bars_json.append({"t": t, "o": float(k[1]), "h": float(k[2]),
                                  "l": float(k[3]), "c": float(k[4]), "v": float(k[5])})
                if fast.initialized and slow.initialized:
                    ema_json.append({"ts": t, "fast_ema": round(fast.value, 6),
                                     "slow_ema": round(slow.value, 6)})
            bars_json_str = json.dumps(bars_json)
            pipe.set(f"{BT}:bars:{inst.id}", bars_json_str)
            pipe.set(f"{BT}:indicators:{inst.id}", json.dumps(ema_json))
            # Write chart bars so the dashboard can merge them into the live chart.
            # Key: test:chart:{instrument_id}-1-MINUTE-LAST-EXTERNAL
            pipe.set(f"test:chart:{inst.id}-1-MINUTE-LAST-EXTERNAL", bars_json_str)
            if i % 10 == 0:
                pipe.execute(); pipe = r.pipeline()
                _set_meta(r, status="running", strategy=sid, days=actual_days,
                          n=len(instruments), done=i + 1, started=started)
        pipe.execute()

        # Run the strategy over all the data.
        engine.add_data(all_bars, sort=True)
        bt_cfg = EMACrossSARConfig(
            strategy_id="EMACrossSAR-BACKTEST-000",
            instrument_ids=cfg_live.instrument_ids,
            trade_usd=cfg_live.trade_usd,
            bar_spec="1-MINUTE-LAST",
            fast_ema_period=cfg_live.fast_ema_period,
            slow_ema_period=cfg_live.slow_ema_period,
            backtest=True,
        )
        engine.add_strategy(EMACrossStopReverse(bt_cfg))
        _set_meta(r, status="running", strategy=sid, days=actual_days,
                  n=len(instruments), done=len(instruments), started=started, phase="engine")
        engine.run()

        # Last close per instrument (for force-close pricing) and the full 1m close
        # series (seconds -> close) used by the mark-to-market equity curve below.
        last_px = {}
        price_series = {}
        for inst in instruments:
            raw = r.get(f"{BT}:bars:{inst.id}")
            if raw:
                arr = json.loads(raw)
                if arr:
                    last_px[str(inst.id)] = arr[-1]["c"]
                    price_series[str(inst.id)] = [(b["t"], b["c"]) for b in arr]

        open_pos, closed_pos = [], []
        pnl = {}
        # Force-close open positions at last bar price so they appear in closed history.
        # They can't be ported to live trading anyway, so showing them as open is misleading.
        for p in engine.cache.positions_open():
            px = last_px.get(str(p.instrument_id), p.avg_px_open)
            realized = round((px - p.avg_px_open) * p.signed_qty, 4) if px else 0.0
            ccy = "USDT"
            closed_pos.append({
                "instrument": str(p.instrument_id), "strategy": sid,
                "side": "LONG" if p.entry.name == "BUY" else "SHORT",
                "qty": p.quantity.as_double(),
                "avg_px_open": p.avg_px_open, "avg_px_close": round(px, 8) if px else p.avg_px_open,
                "realized": realized, "ccy": ccy,
                "ts_opened": int(p.ts_opened // 1_000_000),
                "ts_closed": end_ms,
            })
            pnl.setdefault(ccy, {"realized": 0.0, "unrealized": 0.0})["realized"] += realized
        for p in engine.cache.positions_closed():
            rp = p.realized_pnl
            ccy = rp.currency.code if rp is not None else "USDT"
            closed_pos.append({
                "instrument": str(p.instrument_id), "strategy": sid,
                "side": "LONG" if p.entry.name == "BUY" else "SHORT",
                "qty": p.peak_qty.as_double(),
                "avg_px_open": p.avg_px_open, "avg_px_close": p.avg_px_close,
                "realized": rp.as_double() if rp is not None else 0.0, "ccy": ccy,
                "ts_opened": int(p.ts_opened // 1_000_000),
                "ts_closed": int(p.ts_closed // 1_000_000),
            })
            pnl.setdefault(ccy, {"realized": 0.0, "unrealized": 0.0})["realized"] += (
                rp.as_double() if rp is not None else 0.0)
        for v in pnl.values():
            v["total"] = v["realized"] + v["unrealized"]

        closed_pos.sort(key=lambda c: c["ts_closed"], reverse=True)
        r.set(f"{BT}:positions", json.dumps({
            "positions": open_pos, "closed_positions": closed_pos, "pnl": pnl}))

        # Build a MARK-TO-MARKET equity curve (realized + unrealized per bar) so the
        # Account tab's NAV / return / drawdown reflect TOTAL PnL throughout the run —
        # exactly like the live curve, which is `realized + unrealized` every 5s. A
        # realized-only curve hides open positions' drawdown until they close, which
        # made every still-open position's loss land on one timestamp (end_ms) when
        # force-closed, producing an artificial cliff at the right edge.
        #
        # All positions are now in closed_pos (open ones were force-closed above), each
        # with a +qty (LONG) / -qty (SHORT) signed size and ms open/close timestamps.
        sweep_pos = []
        for cp in closed_pos:
            signed = cp["qty"] if cp["side"] == "LONG" else -cp["qty"]
            sweep_pos.append({
                "inst": cp["instrument"], "signed": signed,
                "open_px": cp["avg_px_open"], "realized": cp.get("realized", 0.0),
                "t_open": cp["ts_opened"] // 1000,    # ms -> s (bar grid is seconds)
                "t_close": cp["ts_closed"] // 1000,
            })
        sweep_pos.sort(key=lambda p: p["t_open"])

        # Sweep every 1m close timestamp in order; forward-fill each instrument's price,
        # open/close positions as their entries/exits pass, and mark the open book to
        # the current price. O(bars + ts·open_positions) — open book is capped at 50.
        all_ts = sorted({t for series in price_series.values() for (t, _) in series})
        ptr = {iid: 0 for iid in price_series}
        last_close = {iid: None for iid in price_series}
        equity_pts = [{"ts": start_ms, "nav": 100_000.0, "realized": 0.0, "unrealized": 0.0, "total": 0.0}]
        realized_running = 0.0
        open_now, oi = [], 0
        for t in all_ts:
            for iid, series in price_series.items():
                p = ptr[iid]
                while p < len(series) and series[p][0] <= t:
                    last_close[iid] = series[p][1]
                    p += 1
                ptr[iid] = p
            while oi < len(sweep_pos) and sweep_pos[oi]["t_open"] <= t:
                open_now.append(sweep_pos[oi]); oi += 1
            still_open = []
            for pos in open_now:
                if pos["t_close"] <= t:
                    realized_running += pos["realized"]
                else:
                    still_open.append(pos)
            open_now = still_open
            unreal = 0.0
            for pos in open_now:
                px = last_close.get(pos["inst"])
                if px is not None:
                    unreal += pos["signed"] * (px - pos["open_px"])
            total = realized_running + unreal
            equity_pts.append({
                "ts": t * 1000,
                "nav": round(100_000.0 + total, 4),
                "realized": round(realized_running, 4),
                "unrealized": round(unreal, 4),
                "total": round(total, 4),
            })
        # Final point at end_ms with everything realized. Force-close realized equals
        # the last-bar unrealized, so this is continuous with the sweep (no new cliff)
        # and gives the live-handoff offset a clean realized-only ending NAV.
        final_realized = sum(p["realized"] for p in sweep_pos)
        equity_pts.append({
            "ts": end_ms,
            "nav": round(100_000.0 + final_realized, 4),
            "realized": round(final_realized, 4),
            "unrealized": 0.0,
            "total": round(final_realized, 4),
        })
        r.set(f"{BT}:equity", json.dumps(equity_pts))
        # Store the ending NAV as the test equity starting point for the live handoff
        r.set(f"{BT}:equity:end_nav", str(round(100_000.0 + final_realized, 4)))
        _set_meta(r, status="done", strategy=sid, days=actual_days, n=len(instruments),
                  done=len(instruments), started=started, finished=time.time(),
                  open=0, closed=len(closed_pos), pnl=pnl,
                  start_ms=start_ms, end_ms=end_ms,
                  live_handoff=(end_date is None))
        try:
            engine.dispose()
        except Exception:
            pass
        print(f"[backtest] done: {len(open_pos)} open, {len(closed_pos)} closed, pnl={pnl}")
    except Exception as e:
        print(f"[backtest] FAILED: {e}\n{traceback.format_exc()}")
        _set_meta(r, status="error", strategy=sid, error=str(e), started=started)
