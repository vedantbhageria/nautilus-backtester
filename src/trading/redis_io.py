"""Redis wiring shared by the node, the strategies, the dashboard server and the
consistency checker.

Everything that crosses the Redis boundary is named here once: the connection
URLs, every ``dashboard:*`` key, and the notification channel. These used to be
re-declared in six files, where a typo in any one of them silently split the
data instead of failing.
"""
from __future__ import annotations

import json
import os
import time
from urllib.parse import urlparse, urlunparse

from dotenv import load_dotenv

load_dotenv()


# -- connection --------------------------------------------------------------

def _normalise(url: str, db: int | None = None) -> str:
    # "localhost" resolves to ::1 first on Windows; Redis in WSL only listens on
    # IPv4, so every new connection paid for a failed IPv6 attempt first.
    p = urlparse(url)
    host = "127.0.0.1" if p.hostname in (None, "localhost") else p.hostname
    netloc = host + (f":{p.port}" if p.port else ":6379")
    if p.username or p.password:
        netloc = f"{p.username or ''}:{p.password or ''}@{netloc}"
    path = f"/{db}" if db is not None else (p.path or "/0")
    return urlunparse((p.scheme or "redis", netloc, path, "", "", ""))


def live_url() -> str:
    """DB 0: the node's cache/message bus plus all live dashboard state."""
    raw = os.getenv("LIVE_REDIS_URL") or os.getenv("REDIS_URL") or "redis://127.0.0.1:6379/0"
    return _normalise(raw, None if urlparse(raw).path not in ("", "/") else 0)


def test_url() -> str:
    """DB 1: backtest results for the dashboard's test mode."""
    raw = os.getenv("TEST_REDIS_URL")
    return _normalise(raw) if raw else _normalise(live_url(), 1)


def host_port() -> tuple[str, int, str | None, str | None]:
    p = urlparse(live_url())
    return p.hostname or "127.0.0.1", p.port or 6379, p.username, p.password


def connect(url: str | None = None):
    """A sync client that fails fast instead of hanging a caller's thread.

    The timeouts matter more than they look: without them a Redis restart leaves
    a blocking XREAD waiting forever, and the command loop silently stops.
    """
    import redis
    return redis.Redis.from_url(
        url or live_url(),
        decode_responses=True,
        socket_connect_timeout=3,
        socket_timeout=10,          # > the longest XREAD block we use (2s)
        health_check_interval=15,
        retry_on_timeout=True,
    )


# -- keys (DB 0 unless noted) -------------------------------------------------

DATA_CMDS = "dashboard:commands"            # stream: dashboard -> ControlActor (subscribe/backfill)
STRATEGY_CMDS = "dashboard:strategy_cmds"   # stream: dashboard -> node command dispatcher
NOTIFICATIONS = "dashboard:notifications"   # stream: anyone -> dashboard notification centre
ORDER_EVENTS = "dashboard:order_events"     # stream: every order event, keyed by client order id
INDICATORS = "dashboard:indicators:"        # stream per instrument: chart indicator points

STRATEGIES = "dashboard:strategies"         # set: registered strategy ids
STRATEGY_STATUS = "dashboard:strategy_status"  # JSON: actual state per strategy (written by the node)
STRATEGY_MODES = "dashboard:strategy_modes"    # hash: strategy id -> "live" | "test"
METRICS = "dashboard:metrics:"              # JSON per strategy: description + live metrics
HEARTBEAT = "dashboard:heartbeat"           # JSON: node liveness, written every second

PORTFOLIO = "dashboard:portfolio"           # JSON: open positions + pnl, every second
CLOSED_POSITIONS = "dashboard:closed_positions"  # JSON: {position_id: record}, survives restarts
OVERALL_PNL = "dashboard:overall_pnl"       # JSON: {ccy: realized}, banked across sessions
EQUITY = {"live": "dashboard:equity:live", "test": "dashboard:equity:test"}
EQUITY_PEAK = {"live": "dashboard:equity:live:peak", "test": "dashboard:equity:test:peak"}

NODE_CONSISTENCY = "dashboard:consistency:node"  # JSON report from the node's own checks

# DB 1 (test mode)
BACKTEST = "dashboard:backtest"             # :meta :positions :equity :equity:end_nav :bars:* :indicators:*
TEST_CHART = "test:chart:"                  # + bar_type -> JSON bar list

ACCOUNT_START = float(os.getenv("ACCOUNT_START_EQUITY", "100000"))
STALE_COMMAND_S = 60        # the node ignores commands older than this (queued while it was down)


# -- notifications ------------------------------------------------------------

LEVELS = ("info", "success", "warn", "error")


def notification(level: str, source: str, title: str, detail: str = "",
                 retry: dict | None = None, key: str = "", cmd_id: str = "") -> dict:
    """One notification record.

    ``retry`` is the exact command to resend, as ``{"mode": ..., "command": {...}}``;
    the dashboard shows a Retry button for it. ``key`` groups repeats of the same
    condition so the dashboard can collapse them. ``cmd_id`` ties a result back to
    the command that caused it.
    """
    if level not in LEVELS:
        level = "info"
    return {
        "ts": f"{time.time():.3f}",
        "level": level,
        "source": source,
        "title": title[:300],
        "detail": (detail or "")[:4000],
        "retry": json.dumps(retry) if retry else "",
        "key": key,
        "cmd_id": cmd_id,
    }


def notify(r, level: str, source: str, title: str, detail: str = "",
           retry: dict | None = None, key: str = "", cmd_id: str = "") -> bool:
    """Publish a notification. Never raises: a failing notifier must not take
    down the code path that is trying to report a failure. Returns delivery."""
    rec = notification(level, source, title, detail, retry, key, cmd_id)
    if r is None:
        print(f"[notify:{level}] {source}: {title} {detail}".rstrip())
        return False
    try:
        r.xadd(NOTIFICATIONS, rec, maxlen=1000, approximate=True)
        return True
    except Exception as e:
        print(f"[notify:{level}] (redis unavailable: {e}) {source}: {title} {detail}".rstrip())
        return False


def strategy_retry(sid: str, action: str, mode: str = "live", **extra) -> dict:
    """Retry payload that resends a strategy command."""
    return {"mode": mode, "command": {"action": action, "strategy_id": sid, **extra}}


def read_json(r, key: str, default=None):
    raw = r.get(key)
    if not raw:
        return default
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return default
