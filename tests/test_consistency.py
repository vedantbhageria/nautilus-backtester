import time
from types import SimpleNamespace as NS

from trading import consistency as C


def _healthy(now=None):
    now = now or time.time()
    ms = int(now * 1000)
    closed = {
        "P-1": {"id": "P-1", "instrument": "BTCUSDT-PERP.BINANCE_FUTURES", "strategy": "S-000", "side": "LONG",
                "qty": 0.02, "avg_px_open": 100000.0, "avg_px_close": 101000.0, "realized": 19.2, "ccy": "USDT",
                "ts_opened": ms - 60000, "ts_closed": ms - 30000, "session": 1},
    }
    positions = [{"id": "P-2", "instrument": "ETHUSDT-PERP.BINANCE_FUTURES", "strategy": "S-000", "side": "SHORT",
                  "qty": 1.0, "avg_px": 3000.0, "realized": -0.6, "unrealized": 5.0, "ccy": "USDT", "ts_opened": ms}]
    pnl = C.pnl_from(positions, list(closed.values()))
    return {
        "now": now, "redis": {"live": None, "test": None},
        "heartbeat": {"ts": ms - 1000, "pid": 1, "session": 1},
        "portfolio": {"ts": ms - 800, "session": 1, "strategies": ["S-000"], "positions": positions, "pnl": pnl},
        "ledger": closed,
        "status": {"S-000": {"state": "RUNNING", "armed": True}},
        "modes": {"S-000": "live"},
        "node_report": {"ts": now - 5, "checks": [C.check("cache_integrity", "Nautilus cache integrity", C.PASS)]},
        "order_events": [{"order_id": "O-1", "instrument": "X", "status": "submitted", "ts": str(now - 40)},
                         {"order_id": "O-1", "instrument": "X", "status": "filled", "ts": str(now - 39)}],
        "equity": {"live": [{"ts": ms - 6000}, {"ts": ms - 1000}]},
    }


def by_id(rep):
    return {c["id"]: c for c in rep["checks"]}


def test_healthy_system_passes():
    rep = C.server_checks(_healthy())
    bad = [c for c in rep["checks"] if c["status"] in ("fail", "warn")]
    assert bad == [], bad
    assert rep["status"] == "pass"
    assert "node.cache_integrity" in by_id(rep)            # node self-checks are merged in


def test_redis_down_short_circuits():
    rep = C.server_checks({"now": time.time(), "redis": {"live": "Connection refused"}})
    assert rep["status"] == "fail"
    assert by_id(rep)["redis_live"]["detail"] == "Connection refused"


def test_stale_heartbeat_fails():
    d = _healthy()
    d["heartbeat"]["ts"] -= 60_000
    assert by_id(C.server_checks(d))["node_alive"]["status"] == "fail"


def test_pnl_mismatch_is_caught():
    d = _healthy()
    d["portfolio"]["pnl"]["USDT"]["realized"] += 1.0     # published figure drifted from its positions
    assert by_id(C.server_checks(d))["pnl_totals"]["status"] == "fail"


def test_other_sessions_excluded_from_session_pnl():
    d = _healthy()
    d["ledger"]["P-0"] = {**d["ledger"]["P-1"], "id": "P-0", "session": 0}   # previous session
    assert by_id(C.server_checks(d))["pnl_totals"]["status"] == "pass"


def test_ledger_arithmetic():
    d = _healthy()
    d["ledger"]["P-1"]["realized"] = -500.0               # price rose on a LONG but PnL says loss
    c = by_id(C.server_checks(d))["ledger_sane"]
    assert c["status"] == "fail" and "P-1" in c["items"][0]


def test_stuck_and_rejected_orders():
    d = _healthy()
    now = d["now"]
    d["order_events"].append({"order_id": "O-2", "instrument": "Y", "status": "submitted", "ts": str(now - 90)})
    assert by_id(C.server_checks(d))["order_lifecycle"]["status"] == "fail"
    d = _healthy()
    d["order_events"].append({"order_id": "O-3", "instrument": "Z", "status": "rejected", "reason": "no market",
                              "ts": str(now - 10)})
    c = by_id(C.server_checks(d))["order_lifecycle"]
    assert c["status"] == "warn" and "no market" in c["items"][0]


def test_untagged_strategy_warns():
    d = _healthy()
    d["modes"] = {}
    assert by_id(C.server_checks(d))["mode_tags"]["status"] == "warn"


def test_broken_check_reported_not_raised():
    checks = []
    C._run(checks, "boom", "Boom", lambda: 1 / 0)
    assert checks[0]["status"] == "fail" and "ZeroDivisionError" in checks[0]["detail"]


# -- node checks with stand-ins for Nautilus objects -----------------------------

class Money(float):
    def as_double(self):
        return float(self)


def _pos(pid, iid, qty, realized, opened, closed=False, strategy="S-000"):
    return NS(id=pid, instrument_id=iid, signed_qty=qty, realized_pnl=Money(realized), ts_opened=opened,
              is_closed=closed, strategy_id=strategy)


class FakeCache:
    def __init__(self, positions, orders=()):
        self._p, self._o = positions, list(orders)

    def positions(self):
        return self._p

    def positions_open(self, strategy_id=None):
        return [p for p in self._p if not p.is_closed and (strategy_id is None or p.strategy_id == strategy_id)]

    def orders(self):
        return self._o

    def check_integrity(self):
        return True


def _node(positions, balance, net=None, orders=(), strategies=(), ledger=None, sandbox=True):
    cache = FakeCache(positions, orders)
    account = NS(balance_total=lambda ccy: Money(balance))
    portfolio = NS(account=lambda v: account,
                   net_position=lambda iid: (net or {}).get(iid, sum(p.signed_qty for p in cache.positions_open()
                                                                   if p.instrument_id == iid)))
    return C.node_checks(cache=cache, portfolio=portfolio, strategies=list(strategies), venue="V", currency="USDT",
                         session_start_ns=100, start_balance=100_000.0, ledger=ledger or {}, sandbox=sandbox,
                         now_ns=10**12)


def test_account_balance_reconciles():
    ps = [_pos("A", "BTC", 0, 25.0, 200, closed=True), _pos("B", "ETH", 1.0, -0.8, 300)]
    ledger = {"A": {"realized": 25.0}}
    assert by_id(_node(ps, 100_024.2, ledger=ledger))["account_balance"]["status"] == "pass"
    assert by_id(_node(ps, 100_030.0, ledger=ledger))["account_balance"]["status"] == "fail"


def test_phantom_positions_from_previous_session():
    ps = [_pos("OLD", "BTC", 0.5, 0.0, 50)]               # opened before session start (100)
    assert by_id(_node(ps, 100_000.0))["carried_positions"]["status"] == "fail"
    assert by_id(_node(ps, 100_000.0, sandbox=False))["carried_positions"]["status"] == "skip"


def test_net_position_mismatch():
    ps = [_pos("A", "BTC", 0.5, 0.0, 200)]
    assert by_id(_node(ps, 100_000.0, net={"BTC": 0.4}))["net_positions"]["status"] == "fail"


def test_stuck_order_detected():
    o = NS(is_closed=False, ts_last=10**12 - 30 * 10**9, status=NS(name="SUBMITTED"), client_order_id="O-1",
           instrument_id="BTC", side=NS(name="BUY"), strategy_id="S", order_type=NS(name="MARKET"))
    assert by_id(_node([], 100_000.0, orders=[o]))["stuck_orders"]["status"] == "fail"


def test_ledger_gap_detected():
    ps = [_pos("A", "BTC", 0, 12.0, 200, closed=True)]
    rep = _node(ps, 100_012.0, ledger={})
    assert by_id(rep)["ledger"]["status"] == "fail"


def test_paused_strategy_with_orders_fails():
    s = NS(id="S-000", armed=False, is_exiting=lambda: False, working_orders=lambda: [1],
           cache=FakeCache([]), status=lambda: {"armed": False, "warming": 0, "cold": [], "stale": [], "missing": []})
    assert by_id(_node([], 100_000.0, strategies=[s]))["idle_exposure"]["status"] == "fail"
