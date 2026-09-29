import pytest

import backtest_runner as br


class Resp:
    def __init__(self, status, body=None, headers=None, text=""):
        self.status_code, self._body, self.headers, self.text = status, body, headers or {}, text

    @property
    def ok(self):
        return 200 <= self.status_code < 300

    def json(self):
        return self._body


def _kline(open_ms):
    return [open_ms, "1", "1", "1", "1", "1", open_ms + 59_999]


def test_transient_errors_are_retried(monkeypatch):
    seq = iter([Resp(500, text="boom"), Resp(200, [_kline(0), _kline(60_000)])])
    monkeypatch.setattr(br._session, "get", lambda *a, **k: next(seq))
    monkeypatch.setattr(br.time, "sleep", lambda s: None)
    assert len(br.fetch_klines("BTCUSDT", 0, 120_000)) == 2


def test_persistent_failure_raises_instead_of_truncating(monkeypatch):
    # first page ok and full, second page keeps failing: the old code returned
    # the first page as if it were the whole range
    full = [_kline(i * 60_000) for i in range(1500)]
    calls = {"n": 0}

    def get(*a, **k):
        calls["n"] += 1
        return Resp(200, full) if calls["n"] == 1 else Resp(502, text="bad gateway")
    monkeypatch.setattr(br._session, "get", get)
    monkeypatch.setattr(br.time, "sleep", lambda s: None)
    with pytest.raises(br.KlineError):
        br.fetch_klines("BTCUSDT", 0, 3000 * 60_000)


def test_bad_symbol_fails_fast(monkeypatch):
    monkeypatch.setattr(br._session, "get", lambda *a, **k: Resp(400, text='{"code":-1121,"msg":"Invalid symbol."}'))
    with pytest.raises(br.KlineError, match="rejected"):
        br.fetch_klines("NOPEUSDT", 0, 60_000)


def test_rate_limit_waits_then_succeeds(monkeypatch):
    waits = []
    seq = iter([Resp(429, headers={"Retry-After": "7"}), Resp(200, [_kline(0)])])
    monkeypatch.setattr(br._session, "get", lambda *a, **k: next(seq))
    monkeypatch.setattr(br.time, "sleep", waits.append)
    assert len(br.fetch_klines("BTCUSDT", 0, 60_000)) == 1
    assert waits == [7.0]
