"""Strategy decision rules and warm-up ordering, on real Nautilus strategies
registered against a test clock/cache/portfolio (no network, no Redis)."""
from decimal import Decimal

import pytest
from nautilus_trader.cache.cache import Cache
from nautilus_trader.common.component import MessageBus, TestClock
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId, TraderId
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.portfolio.portfolio import Portfolio

from trading.strategies.EMACross import EMACross, EMACrossConfig
from trading.strategies.EMACrossShortTest import EMACrossSARConfig, EMACrossStopReverse

IID = InstrumentId.from_str("BTCUSDT-PERP.BINANCE_FUTURES")


def _register(strategy):
    clock = TestClock()
    trader = TraderId("TESTER-000")
    msgbus = MessageBus(trader_id=trader, clock=clock)
    cache = Cache()
    portfolio = Portfolio(msgbus=msgbus, cache=cache, clock=clock)
    strategy.register(trader_id=trader, portfolio=portfolio, msgbus=msgbus, cache=cache, clock=clock)
    return strategy


class Recorder:
    """Stand-in for order submission: records decisions, tracks net quantity."""

    def __init__(self, s, busy=False):
        self.calls, self.net, self.busy = [], 0.0, busy
        s.submit_market = self.submit
        s.close_all_positions = self.close
        s.net_qty = lambda iid: self.net
        s.has_working_order = lambda iid: self.busy
        s.instruments_with_exposure = lambda: set()
        s.working_orders = lambda iid=None: []

    def submit(self, iid, side, usd, price):
        self.calls.append(side.name)
        self.net += 1 if side == OrderSide.BUY else -1
        return True

    def close(self, iid):
        self.calls.append("CLOSE")
        self.net = 0.0


def _sar():
    return _register(EMACrossStopReverse(EMACrossSARConfig(
        strategy_id="SAR", order_id_tag="001", instrument_ids=(IID,), bar_spec="1-MINUTE-LAST",
        fast_ema_period=2, slow_ema_period=4, trade_usd=Decimal("2000"))))


def test_sar_contrarian_reversals():
    s = _sar()
    rec = Recorder(s)
    s.on_ema(IID, 10, 9, 100)            # first reading: no trade, just remembers "bull"
    assert rec.calls == []
    s.on_ema(IID, 9, 10, 100)            # bearish cross while flat -> go long (contrarian)
    assert rec.calls == ["BUY"]
    s.on_ema(IID, 8, 10, 100)            # still bear: nothing
    assert rec.calls == ["BUY"]
    s.on_ema(IID, 11, 10, 100)           # bullish cross while long -> close, go short
    assert rec.calls == ["BUY", "CLOSE", "SELL"]


def test_sar_does_not_consume_cross_while_order_in_flight():
    s = _sar()
    rec = Recorder(s)
    s.on_ema(IID, 10, 9, 100)
    rec.busy = True
    s.on_ema(IID, 9, 10, 100)            # crossover arrives while an order is unresolved
    assert rec.calls == []
    rec.busy = False
    s.on_ema(IID, 9, 10, 100)            # ...so it is acted on at the next bar, not lost
    assert rec.calls == ["BUY"]


def test_emacross_long_only_regime():
    s = _register(EMACross(EMACrossConfig(strategy_id="EMA", order_id_tag="000", instrument_ids=(IID,),
                                          bar_spec="5-SECOND-LAST", fast_ema_period=2, slow_ema_period=4)))
    rec = Recorder(s)
    s.on_ema(IID, 10, 10, 100)           # fast >= slow and flat -> buy
    s.on_ema(IID, 11, 10, 100)           # already long -> hold
    assert rec.calls == ["BUY"]
    s.on_ema(IID, 9, 10, 100)            # fast < slow and long -> exit
    assert rec.calls == ["BUY", "CLOSE"]
    s.on_ema(IID, 9, 10, 100)            # flat and bearish -> nothing
    assert rec.calls == ["BUY", "CLOSE"]


def _bar(bt, minute, close):
    ts = (1_700_000_000 + minute * 60) * 1_000_000_000
    p = Price.from_str(f"{close:.2f}")
    return Bar(bt, p, p, p, p, Quantity.from_str("1"), ts, ts)


def test_warmup_feeds_in_time_order_and_replays_buffer():
    s = _sar()
    fed = []
    s.update_indicators = lambda iid, bar: fed.append(bar.ts_event)
    signals = []
    s.on_signal_bar = lambda iid, bar: signals.append(bar.ts_event)
    s.is_warm = lambda iid: True
    s.armed = True
    bt = s._bar_types[IID]
    s._warming.add(IID)
    s.on_bar(_bar(bt, 10, 100))          # live bar arrives before the history answer
    s.on_bar(_bar(bt, 11, 101))
    assert fed == []                      # buffered, not fed out of order
    for m in range(0, 11):                # history up to and including minute 10 (overlap)
        s.on_historical_data(_bar(bt, m, 99))
    s._finish_warming(IID)                # completion callback
    minutes = [(t // 1_000_000_000 - 1_700_000_000) // 60 for t in fed]
    assert minutes == list(range(0, 12))  # strictly increasing, the overlapping bar fed once
    assert len(signals) == 1              # only the genuinely new live bar may signal


def test_order_sizing_explains_skips():
    from nautilus_trader.test_kit.providers import TestInstrumentProvider
    s = _sar()
    qty, why = s.order_qty(IID, 2000, 100)
    assert qty is None and "not loaded" in why       # instrument not in this test cache
    inst = TestInstrumentProvider.btcusdt_perp_binance()
    s.cache.add_instrument(inst)
    qty, why = s.order_qty(inst.id, 2000, 100_000)
    assert why == "" and abs(qty.as_double() - 0.02) < 1e-9
    assert "minimum notional" in s.order_qty(inst.id, 1.0, 100_000)[1]
    assert "rounds to zero" in s.order_qty(inst.id, 5.0, 100_000_000)[1]


def test_config_guard():
    with pytest.raises(ValueError):
        EMACrossStopReverse(EMACrossSARConfig(strategy_id="X", order_id_tag="9", instrument_ids=(IID,),
                                              fast_ema_period=10, slow_ema_period=10))
