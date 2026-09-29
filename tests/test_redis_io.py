import json

from trading import redis_io as K


def test_localhost_becomes_ipv4(monkeypatch):
    monkeypatch.delenv("LIVE_REDIS_URL", raising=False)
    monkeypatch.delenv("TEST_REDIS_URL", raising=False)
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")
    assert K.live_url() == "redis://127.0.0.1:6379/0"
    assert K.test_url() == "redis://127.0.0.1:6379/1"
    assert K.host_port()[:2] == ("127.0.0.1", 6379)


def test_explicit_db_and_credentials_kept(monkeypatch):
    monkeypatch.setenv("LIVE_REDIS_URL", "redis://user:pw@10.0.0.5:6380/3")
    monkeypatch.setenv("TEST_REDIS_URL", "redis://10.0.0.5:6380/4")
    assert K.live_url() == "redis://user:pw@10.0.0.5:6380/3"
    assert K.test_url() == "redis://10.0.0.5:6380/4"


def test_notification_record_is_flat_strings():
    rec = K.notification("bogus", "node", "t" * 400, "d", retry=K.strategy_retry("S-1", "rewarm"), key="k")
    assert rec["level"] == "info"                      # unknown levels are coerced
    assert len(rec["title"]) == 300
    assert json.loads(rec["retry"]) == {"mode": "live", "command": {"action": "rewarm", "strategy_id": "S-1"}}
    assert all(isinstance(v, str) for v in rec.values())   # Redis stream fields


def test_notify_never_raises():
    class Broken:
        def xadd(self, *a, **k):
            raise ConnectionError("down")
    assert K.notify(Broken(), "error", "x", "y") is False
    assert K.notify(None, "error", "x", "y") is False
