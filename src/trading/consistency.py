"""Consistency checks: does everything the system reports agree with itself?

Two layers, one report format:

* ``node_checks`` run inside the trading node against Nautilus's own cache,
  portfolio and account (ground truth). The ControlActor runs them every 15s
  and publishes the report to Redis.
* ``server_checks`` run anywhere that can read Redis (dashboard server, CLI),
  against what the node *published*: heartbeat, snapshots, the closed-position
  ledger, the order-event stream, equity series.

Every check returns ``{"id", "title", "status", "detail", "items"}`` with status
``pass | warn | fail | skip``. A check that raises is reported as ``fail`` with
the exception, so one broken check can never hide the others.

Run from a shell: ``python scripts/verify_consistency.py`` (exit code 1 on fail).
"""
from __future__ import annotations

import time
from collections import defaultdict

PASS, WARN, FAIL, SKIP = "pass", "warn", "fail", "skip"
TERMINAL_ORDER = {"filled", "canceled", "rejected", "denied", "expired"}


def check(cid: str, title: str, status: str, detail: str = "", items=None) -> dict:
    return {"id": cid, "title": title, "status": status, "detail": detail, "items": list(items or [])[:50]}


def _run(checks: list, cid: str, title: str, fn) -> None:
    try:
        res = fn()
        checks.append(check(cid, title, *res) if isinstance(res, tuple) else res)
    except Exception as e:                        # a broken check must not hide the rest
        checks.append(check(cid, title, FAIL, f"check raised {type(e).__name__}: {e}"))


def report(checks: list, source: str) -> dict:
    summary = {s: sum(1 for c in checks if c["status"] == s) for s in (PASS, WARN, FAIL, SKIP)}
    overall = FAIL if summary[FAIL] else WARN if summary[WARN] else PASS
    return {"ts": time.time(), "source": source, "status": overall, "summary": summary, "checks": checks}


# ---------------------------------------------------------------------------
# node side (Nautilus objects)
# ---------------------------------------------------------------------------

def node_checks(*, cache, portfolio, strategies, venue, currency, session_start_ns: int,
                start_balance: float | None, ledger: dict, sandbox: bool, now_ns: int) -> dict:
    checks: list = []
    session_positions = [p for p in cache.positions() if p.ts_opened >= session_start_ns]

    _run(checks, "cache_integrity", "Nautilus cache integrity",
         lambda: (PASS, "cache indexes consistent") if cache.check_integrity()
         else (FAIL, "cache.check_integrity() failed; the node log names the broken index"))

    def account_balance():
        account = portfolio.account(venue)
        if account is None:
            return FAIL, f"no account for {venue}: the execution client never reported one"
        if start_balance is None:
            return SKIP, "session starting balance not captured yet"
        bal = account.balance_total(currency)
        bal = bal.as_double() if bal is not None else 0.0
        realized = sum(p.realized_pnl.as_double() for p in session_positions if p.realized_pnl is not None)
        expected = start_balance + realized
        diff = bal - expected
        detail = (f"balance {bal:,.4f} vs start {start_balance:,.4f} + realized {realized:,.4f} "
                  f"= {expected:,.4f} (diff {diff:+.6f})")
        if abs(diff) <= 0.01 + 1e-7 * abs(bal):
            return PASS, detail
        return FAIL, detail + ". Fills and positions disagree with the account ledger."
    _run(checks, "account_balance", "Account balance = start + realized PnL", account_balance)

    def carried_positions():
        if not sandbox:
            return SKIP, "live venue: carried positions are reconciled by Nautilus"
        old = [p for p in cache.positions_open() if p.ts_opened < session_start_ns]
        if not old:
            return PASS, "no open positions from a previous session"
        return (FAIL, "the sandbox restarts flat, so these cached positions don't exist at the venue "
                      "and distort PnL/NAV", [f"{p.id} {p.instrument_id} {p.signed_qty:+g}" for p in old])
    _run(checks, "carried_positions", "No phantom positions from previous sessions", carried_positions)

    def net_positions():
        by_iid = defaultdict(float)
        for p in cache.positions_open():
            by_iid[p.instrument_id] += p.signed_qty
        bad = []
        for iid, qty in by_iid.items():
            port = float(portfolio.net_position(iid))
            if abs(port - qty) > 1e-9 * max(1.0, abs(qty)):
                bad.append(f"{iid}: portfolio {port:g} vs positions {qty:g}")
        return (FAIL, "portfolio net position disagrees with open positions", bad) if bad \
            else (PASS, f"{len(by_iid)} instrument(s) agree")
    _run(checks, "net_positions", "Portfolio net position = sum of open positions", net_positions)

    def stuck_orders():
        stuck, lingering = [], []
        for o in cache.orders():
            if o.is_closed:
                continue
            age_s = (now_ns - o.ts_last) / 1e9
            name = o.status.name
            label = f"{o.client_order_id} {o.instrument_id} {o.side.name} {name} {age_s:.0f}s ({o.strategy_id})"
            if name in ("INITIALIZED", "SUBMITTED", "PENDING_UPDATE", "PENDING_CANCEL") and age_s > 15:
                stuck.append(label)
            elif o.order_type.name == "MARKET" and age_s > 15:
                lingering.append(label)
        if stuck:
            return FAIL, "orders with no venue response for >15s", stuck
        if lingering:
            return WARN, "market orders accepted but unfilled for >15s (no liquidity?)", lingering
        return PASS, "no stuck orders"
    _run(checks, "stuck_orders", "No orders stuck in flight", stuck_orders)

    def idle_exposure():
        bad, warn = [], []
        for s in strategies:
            if s.armed or s.is_exiting():
                continue
            orders = s.working_orders()
            pos = s.cache.positions_open(strategy_id=s.id)
            if orders:
                bad.append(f"{s.id}: {len(orders)} working order(s) while paused")
            if pos:
                warn.append(f"{s.id}: {len(pos)} open position(s) while paused (unmanaged)")
        if bad:
            return FAIL, "paused strategies must not have working orders", bad + warn
        if warn:
            return WARN, "paused strategies are holding positions nothing is managing", warn
        return PASS, "no unmanaged exposure"
    _run(checks, "idle_exposure", "No unmanaged orders or positions", idle_exposure)

    def warmup():
        cold = []
        for s in strategies:
            st = s.status()
            if st["armed"] and not st["warming"] and st["cold"]:
                cold.append(f"{s.id}: {len(st['cold'])} cold: {', '.join(st['cold'][:15])}")
        return (WARN, "armed instruments that can't signal yet", cold) if cold \
            else (PASS, "every armed instrument is warm")
    _run(checks, "warmup", "Armed instruments are warmed up", warmup)

    def feeds():
        stale = []
        for s in strategies:
            st = s.status()
            if st["stale"]:
                stale.append(f"{s.id}: no bars from {', '.join(st['stale'][:15])}")
        return (WARN, "no live bar for 3+ intervals", stale) if stale else (PASS, "live bars arriving")
    _run(checks, "data_feeds", "Live data feeds are flowing", feeds)

    def missing():
        miss = []
        for s in strategies:
            st = s.status()
            if st["missing"]:
                miss.append(f"{s.id}: {', '.join(st['missing'])}")
        return (WARN, "configured instruments that don't exist on the venue (never traded)", miss) if miss \
            else (PASS, "every configured instrument exists")
    _run(checks, "instruments", "Configured instruments exist", missing)

    def ledger_matches():
        bad, missing_ids = [], []
        for p in session_positions:
            if not p.is_closed:
                continue
            rec = ledger.get(str(p.id))
            if rec is None:
                missing_ids.append(str(p.id))
                continue
            rp = p.realized_pnl.as_double() if p.realized_pnl is not None else 0.0
            if abs(rec.get("realized", 0.0) - rp) > 1e-6:
                bad.append(f"{p.id}: ledger {rec.get('realized')} vs engine {rp}")
        if missing_ids or bad:
            return (FAIL, f"{len(missing_ids)} closed position(s) missing from the dashboard ledger, "
                          f"{len(bad)} with different PnL", missing_ids + bad)
        return PASS, f"{sum(1 for p in session_positions if p.is_closed)} closed position(s) match"
    _run(checks, "ledger", "Dashboard ledger = engine closed positions", ledger_matches)

    return report(checks, "node")


# ---------------------------------------------------------------------------
# server side (plain dicts read from Redis)
# ---------------------------------------------------------------------------

def gather(r_live, r_test=None) -> dict:
    """Everything server_checks needs, read in one place (shared with the CLI)."""
    from trading import redis_io as K
    data = {"now": time.time(), "redis": {}}
    for name, r in (("live", r_live), ("test", r_test)):
        if r is None:
            continue
        try:
            r.ping()
            data["redis"][name] = None
        except Exception as e:
            data["redis"][name] = str(e)
    if data["redis"].get("live"):
        return data
    data["heartbeat"] = K.read_json(r_live, K.HEARTBEAT)
    data["portfolio"] = K.read_json(r_live, K.PORTFOLIO)
    data["ledger"] = K.read_json(r_live, K.CLOSED_POSITIONS, {}) or {}
    data["status"] = K.read_json(r_live, K.STRATEGY_STATUS, {}) or {}
    data["modes"] = r_live.hgetall(K.STRATEGY_MODES) or {}
    data["node_report"] = K.read_json(r_live, K.NODE_CONSISTENCY)
    data["order_events"] = [f for _id, f in r_live.xrevrange(K.ORDER_EVENTS, count=5000)][::-1]
    data["equity"] = {}
    for mode, key in K.EQUITY.items():
        data["equity"][mode] = [_loads(x) for x in r_live.lrange(key, -3, -1)]
    if r_test is not None and not data["redis"].get("test"):
        data["backtest_meta"] = K.read_json(r_test, f"{K.BACKTEST}:meta")
    return data


def _loads(raw):
    import json
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


def pnl_from(positions: list, closed: list) -> dict:
    """{ccy: {realized, unrealized, total}}: the one definition of PnL shared by
    the node snapshot, the server frames and these checks."""
    out: dict = {}
    for p in positions:
        d = out.setdefault(p.get("ccy") or "USDT", {"realized": 0.0, "unrealized": 0.0})
        d["realized"] += p.get("realized") or 0.0
        d["unrealized"] += p.get("unrealized") or 0.0
    for c in closed:
        d = out.setdefault(c.get("ccy") or "USDT", {"realized": 0.0, "unrealized": 0.0})
        d["realized"] += c.get("realized") or 0.0
    for v in out.values():
        v["total"] = v["realized"] + v["unrealized"]
    return out


def server_checks(data: dict) -> dict:
    checks: list = []
    now = data["now"]

    for name, err in data.get("redis", {}).items():
        checks.append(check(f"redis_{name}", f"Redis {name} DB reachable", FAIL if err else PASS,
                            err or "PING ok"))
    if data.get("redis", {}).get("live"):
        return report(checks, "server")

    hb = data.get("heartbeat")
    node_age = now - hb["ts"] / 1000 if hb else None

    def node_alive():
        if hb is None:
            return FAIL, "no heartbeat: the trading node has never run against this Redis"
        detail = f"last heartbeat {node_age:.1f}s ago (pid {hb.get('pid')})"
        if node_age < 5:
            return PASS, detail
        return (WARN if node_age < 30 else FAIL), detail + ": node stalled or stopped"
    _run(checks, "node_alive", "Trading node is alive", node_alive)

    snap = data.get("portfolio") or {}
    ledger = data.get("ledger") or {}
    closed = list(ledger.values())

    def snapshot_fresh():
        if not snap:
            return FAIL, "no portfolio snapshot published"
        age = now - snap.get("ts", 0) / 1000
        return (PASS if age < 5 else WARN if age < 30 else FAIL), f"snapshot {age:.1f}s old"
    _run(checks, "snapshot_fresh", "Portfolio snapshot is current", snapshot_fresh)

    def pnl_totals():
        if not snap:
            return SKIP, "no snapshot"
        session = snap.get("session")
        expected = pnl_from(snap.get("positions", []),
                            [c for c in closed if c.get("session") == session])
        got = snap.get("pnl", {})
        bad = []
        for ccy in set(expected) | set(got):
            for k in ("realized", "unrealized", "total"):
                a, b = got.get(ccy, {}).get(k, 0.0), expected.get(ccy, {}).get(k, 0.0)
                if abs(a - b) > 1e-6 + 1e-9 * abs(b):
                    bad.append(f"{ccy} {k}: snapshot {a:.6f} vs recomputed {b:.6f}")
        return (FAIL, "published PnL != sum of its positions", bad) if bad \
            else (PASS, "session PnL = open positions + this session's closed positions")
    _run(checks, "pnl_totals", "Published PnL = sum of positions", pnl_totals)

    def ledger_sane():
        bad = []
        ids = set()
        for key, c in ledger.items():
            pid = c.get("id")
            if pid != key:
                bad.append(f"{key}: record id {pid!r} doesn't match its key")
            if pid in ids:
                bad.append(f"{pid}: duplicate")
            ids.add(pid)
            o, cl = c.get("ts_opened") or 0, c.get("ts_closed") or 0
            if not (0 < o <= cl):
                bad.append(f"{pid}: opened {o} / closed {cl} out of order")
            qty, po, pc = c.get("qty") or 0, c.get("avg_px_open") or 0, c.get("avg_px_close") or 0
            if qty <= 0 or po <= 0 or pc <= 0:
                bad.append(f"{pid}: non-positive qty/price ({qty}, {po}, {pc})")
                continue
            gross = (pc - po) * qty * (1 if c.get("side") == "LONG" else -1)
            fees = 0.002 * qty * (po + pc)          # generous: 0.1% taker each side
            if abs((c.get("realized") or 0.0) - gross) > fees + 0.01:
                bad.append(f"{pid}: realized {c.get('realized'):.4f} but price move gives {gross:.4f}")
        return (FAIL, "closed-position records are internally inconsistent", bad) if bad \
            else (PASS, f"{len(ledger)} record(s) internally consistent")
    _run(checks, "ledger_sane", "Closed positions add up (price x qty = PnL)", ledger_sane)

    def single_node():
        # Every event the current node publishes carries an order id; recent
        # events without one come from a second, older node sharing this Redis.
        foreign = [ev for ev in data.get("order_events", [])
                   if not ev.get("order_id") and now - float(ev.get("ts", 0)) < 600]
        if foreign:
            strategies = sorted({ev.get("strategy", "?") for ev in foreign})
            return (FAIL, f"{len(foreign)} order event(s) in the last 10 min came from another trading node "
                          "(old code, no order ids). Two nodes obey the same commands and trade twice; "
                          "stop the extra python binance_data.py process.",
                    [f"strategies: {', '.join(strategies)}"])
        return PASS, "all recent order events come from this node"
    _run(checks, "single_node", "Only one trading node is running", single_node)

    def order_lifecycle():
        last: dict = {}
        for ev in data.get("order_events", []):
            oid = ev.get("order_id")
            if oid:
                last[oid] = ev
        stuck = [f"{oid} {ev.get('instrument')} {ev.get('status')} {now - float(ev.get('ts', now)):.0f}s"
                 for oid, ev in last.items()
                 if ev.get("status") not in TERMINAL_ORDER and ev.get("status") != "accepted"
                 and now - float(ev.get("ts", now)) > 30]
        rejected = [f"{ev.get('instrument')} {ev.get('status')}: {ev.get('reason')}"
                    for ev in last.values() if ev.get("status") in ("rejected", "denied")
                    and now - float(ev.get("ts", 0)) < 3600]
        if stuck:
            return FAIL, "orders whose last event is non-terminal for >30s", stuck
        if rejected:
            return WARN, f"{len(rejected)} order(s) rejected/denied in the last hour", rejected
        return PASS, f"{len(last)} recent order(s) all resolved"
    _run(checks, "order_lifecycle", "Every order reached a final state", order_lifecycle)

    def equity():
        out = []
        status = PASS
        for mode, pts in (data.get("equity") or {}).items():
            pts = [p for p in pts if p]
            if len(pts) >= 2 and any(b["ts"] <= a["ts"] for a, b in zip(pts, pts[1:])):
                return FAIL, f"{mode} equity timestamps not increasing", [str(pts)]
            if pts and node_age is not None and node_age < 5:
                age = now - pts[-1]["ts"] / 1000
                if age > 20:
                    status = WARN
                    out.append(f"{mode}: last equity point {age:.0f}s old")
        return status, "equity series current" if status == PASS else "equity series lagging", out
    _run(checks, "equity", "Equity series recording", equity)

    def strategy_status():
        status = data.get("status") or {}
        strategies = (snap or {}).get("strategies", [])
        missing = [s for s in strategies if s not in status]
        faulted = [f"{sid}: {st.get('state')}" for sid, st in status.items() if st.get("state") != "RUNNING"]
        if faulted:
            return FAIL, "strategies not in RUNNING state (can't trade or exit)", faulted
        if missing:
            return WARN, "registered strategies without status (stale registry?)", missing
        return PASS, f"{len(status)} strateg(ies) reporting"
    _run(checks, "strategy_status", "Strategies report their real state", strategy_status)

    def mode_tags():
        modes = data.get("modes") or {}
        traded = {p.get("strategy") for p in (snap or {}).get("positions", [])} | \
                 {c.get("strategy") for c in closed if c.get("session") == (snap or {}).get("session")}
        untagged = sorted(s for s in traded if s and s not in modes)
        return (WARN, "traded this session without a live/test tag (counted as live)", untagged) if untagged \
            else (PASS, "every traded strategy is tagged live or test")
    _run(checks, "mode_tags", "Positions are attributed to live/test", mode_tags)

    def backtest():
        meta = data.get("backtest_meta")
        if not meta or meta.get("status") in (None, "none"):
            return SKIP, "no backtest has run"
        st = meta.get("status")
        if st == "error":
            return FAIL, f"last backtest failed: {meta.get('error')}"
        if st == "running" and now - (meta.get("started") or now) > 3600:
            return WARN, "backtest 'running' for over an hour (thread died?)"
        skipped = meta.get("skipped") or []
        if skipped:
            return WARN, f"backtest ran without {len(skipped)} instrument(s) (no data)", skipped
        return PASS, f"last backtest {st}"
    _run(checks, "backtest", "Last backtest completed cleanly", backtest)

    rep = report(checks, "server")
    node = data.get("node_report")
    if node is None:
        rep["checks"].append(check("node_report", "Node self-checks reported", WARN if hb else SKIP,
                                   "the node hasn't published a consistency report"))
    else:
        age = now - node.get("ts", 0)
        for c in node.get("checks", []):
            rep["checks"].append({**c, "id": "node." + c["id"], "title": c["title"] + " (node)"})
        if age > 60 and node_age is not None and node_age < 5:
            rep["checks"].append(check("node_report", "Node self-checks reported", WARN,
                                       f"node report is {age:.0f}s old"))
    return report(rep["checks"], "combined")
