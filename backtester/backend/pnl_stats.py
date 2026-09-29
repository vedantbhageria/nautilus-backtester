# Shared trade-level P&L math for per-symbol / per-class attribution — used
# by both server.py (dashboard's Testing/Compare endpoints) and
# build_report.py (the external HTML report), so the numbers always agree.

import csv
import math
import os


def read_position_pnls(pdir):
    """{symbol: [pnlcomm, ...]} from a job's positions/<SYMBOL>.csv folder."""
    out = {}
    for fn in os.listdir(pdir):
        if not fn.endswith('.csv'):
            continue
        sym = fn[:-4]
        pnls = []
        try:
            with open(os.path.join(pdir, fn), newline='', encoding='utf-8') as f:
                for r in csv.DictReader(f):
                    try:
                        pnls.append(float(r.get('pnlcomm') or 0.0))
                    except (TypeError, ValueError):
                        pass
        except OSError:
            continue
        if pnls:
            out[sym] = pnls
    return out


def pnl_stats(pnls):
    """trades/pnl/win_rate/sharpe/max_dd from a flat list of per-trade
    pnlcomm values — same trade-level math whether it's one symbol or several
    pooled together for a class."""
    n = len(pnls)
    total = sum(pnls)
    wins = sum(1 for p in pnls if p > 0)
    sharpe = None
    if n >= 2:
        mu = total / n
        var = sum((p - mu) ** 2 for p in pnls) / (n - 1)
        sd = math.sqrt(var)
        if sd > 0:
            sharpe = round(mu / sd * math.sqrt(n), 4)
    cum = 0.0; peak = 0.0; maxdd = 0.0
    for p in pnls:
        cum += p
        peak = max(peak, cum)
        maxdd = min(maxdd, cum - peak)
    return {'trades': n, 'pnl': round(total, 4),
            'win_rate': round(100.0 * wins / n, 2) if n else None,
            'sharpe': sharpe, 'max_dd': round(maxdd, 4)}
