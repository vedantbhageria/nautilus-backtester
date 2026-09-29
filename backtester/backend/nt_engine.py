"""NautilusTrader engine behind the backtester's strategy interface.

``run_backtest.py`` decides the window, loads bars, and writes the artifacts
the dashboard reads. This module does the simulation itself:

  * one ``CryptoPerpetual`` per symbol on a simulated BINANCE venue
    (NETTING, margin account, 10x leverage, taker fee = COMMISSION),
  * bars go into a ``BacktestEngine``,
  * ``PortfolioDriver`` (a Nautilus ``Strategy``) receives every bar, pushes it
    into the matching ``strategy_base.Feed`` and schedules a time alert 1 ns
    after that bar timestamp. The alert fires once all instruments' bars for
    that timestamp are in, so the user strategy's ``next()`` sees a complete
    cross-section, just as backtrader's did,
  * ``order_target_size`` becomes a market order for the delta to the
    target. Nautilus fills it at the current book price, and the fill events
    feed the strategy's trade log.

Account value comes from the Nautilus margin account (``balance_total``: cash
plus realised PnL minus fees) plus open positions marked to the latest bar
close.
"""
from decimal import Decimal

import pandas as pd

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.config import LoggingConfig, RiskEngineConfig, StrategyConfig
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import (AccountType, BookType, CurrencyType,
                                         OmsType, OrderSide)
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import CryptoPerpetual
from nautilus_trader.model.objects import Currency, Money, Price, Quantity
from nautilus_trader.persistence.wranglers import BarDataWrangler
from nautilus_trader.trading.strategy import Strategy

from strategy_base import Feed

VENUE = 'BINANCE'
MAX_PRICE_PRECISION = 8
SIZE_PRECISION = 6
# Book depth given to the simulated exchange for every bar. backtrader fills a
# market order in full at the bar price whatever the volume. Nautilus sizes the
# L1 book from bar volume and splits larger orders into partial fills at worse
# ticks. A deep book gives backtrader's all-at-one-price fills. The real
# volume still reaches the strategy's feeds unchanged.
BOOK_VOLUME = 1_000_000_000


def _decimals(x):
    s = repr(float(x))
    if 'e' in s or 'E' in s:
        mant, exp = s.lower().split('e')
        frac = len(mant.split('.')[1]) if '.' in mant else 0
        return max(0, frac - int(exp))
    return len(s.split('.')[1].rstrip('0')) if '.' in s else 0


def price_precision(df):
    """Tick precision inferred from the data (bars are stored as doubles)."""
    sample = pd.concat([df[c].head(3000) for c in ('open', 'high', 'low', 'close')])
    return min(MAX_PRICE_PRECISION, max((_decimals(v) for v in sample), default=2))


def make_instrument(symbol, pp, commission):
    base = symbol[:-4] if symbol.endswith('USDT') else symbol
    return CryptoPerpetual(
        instrument_id=InstrumentId(Symbol('%s-PERP' % symbol), Venue(VENUE)),
        raw_symbol=Symbol(symbol),
        base_currency=Currency(base, 8, 0, base, CurrencyType.CRYPTO),
        quote_currency=USDT,
        settlement_currency=USDT,
        is_inverse=False,
        price_precision=pp,
        size_precision=SIZE_PRECISION,
        price_increment=Price(10 ** -pp, pp),
        size_increment=Quantity(10 ** -SIZE_PRECISION, SIZE_PRECISION),
        margin_init=Decimal(1),     # margin = notional / leverage
        margin_maint=Decimal('0.5'),
        maker_fee=Decimal(str(commission)),
        taker_fee=Decimal(str(commission)),
        ts_event=0,
        ts_init=0,
    )


class _CommInfo:
    def __init__(self, commission):
        self.p = type('P', (), {'commission': commission})()


class PortfolioDriver(Strategy):
    """Feeds bars to the backtrader-style strategy and executes its orders."""

    def __init__(self, feeds, bar_types, commission):
        super().__init__(StrategyConfig(strategy_id='PORTFOLIO-001',
                                        order_id_tag='001'))
        self.feeds = feeds                    # InstrumentId -> Feed
        self.volumes = {}                     # symbol -> real bar volumes, in order
        self.bar_types = bar_types
        self.commission = commission
        self.user = None                      # set by run()
        self._alerted_ts = -1
        self._order_feed = {}                 # client_order_id str -> Feed
        self.fills = 0
        self.rejects = []
        self.step_hook = None                 # optional progress callback(ts_ns)

    # ---- lifecycle ------------------------------------------------------
    def on_start(self):
        for bt_ in self.bar_types:
            self.subscribe_bars(bt_)

    def on_bar(self, bar):
        feed = self.feeds[bar.bar_type.instrument_id]
        ts = bar.ts_event
        vols = self.volumes[feed._name]
        feed._push(ts // 1_000_000_000, float(bar.open), float(bar.high),
                   float(bar.low), float(bar.close), float(vols[len(feed)]))
        if ts != self._alerted_ts:
            self._alerted_ts = ts
            self.clock.set_time_alert_ns('step-%d' % ts, ts + 1, self._on_step)

    def _on_step(self, event):
        self.user.next()
        if self.step_hook is not None:
            self.step_hook(event.ts_event)

    # ---- broker interface used by strategy_base ------------------------
    def position(self, d):
        size = float(self.portfolio.net_position(d.instrument_id))
        if not size:
            return 0.0, 0.0
        tr = self.user._trade.get(d)
        return size, (tr['price'] if tr else 0.0)

    def getvalue(self):
        acct = self.portfolio.account(Venue(VENUE))
        value = acct.balance_total(USDT).as_double()
        for d, tr in self.user._trade.items():
            if len(d):
                value += (d.close[0] - tr['price']) * tr['size']
        return value

    def getcash(self):
        return self.portfolio.account(Venue(VENUE)).balance_free(USDT).as_double()

    def getcommissioninfo(self, d):
        return _CommInfo(self.commission)

    def submit_target(self, d, target):
        inst = self.cache.instrument(d.instrument_id)
        cur = float(self.portfolio.net_position(d.instrument_id))
        delta = float(target) - cur
        qty = inst.make_qty(abs(delta))
        if qty.as_double() <= 0:
            return None
        order = self.order_factory.market(
            instrument_id=d.instrument_id,
            order_side=OrderSide.BUY if delta > 0 else OrderSide.SELL,
            quantity=qty,
        )
        ref = order.client_order_id.value
        self._order_feed[ref] = d
        self.submit_order(order)
        return ref

    # ---- execution events ------------------------------------------------
    def on_order_filled(self, event):
        ref = event.client_order_id.value
        d = self._order_feed.get(ref)
        if d is None:
            return
        order = self.cache.order(event.client_order_id)
        done = order is None or order.is_closed
        if done:
            self._order_feed.pop(ref, None)
        self.fills += 1
        comm = event.commission.as_double() if event.commission is not None else 0.0
        self.user._on_fill(
            d, ref, 'BUY' if event.order_side == OrderSide.BUY else 'SELL',
            event.last_px.as_double(), event.last_qty.as_double(), comm,
            event.ts_event // 1_000_000_000, done)

    def _dead(self, event, why):
        ref = event.client_order_id.value
        d = self._order_feed.pop(ref, None)
        if d is not None:
            self.rejects.append((d._name, why, getattr(event, 'reason', '')))
            self.user._on_order_dead(d, ref)

    def on_order_rejected(self, event):
        self._dead(event, 'rejected')

    def on_order_denied(self, event):
        self._dead(event, 'denied')

    def on_order_canceled(self, event):
        self._dead(event, 'canceled')

    def on_order_expired(self, event):
        self._dead(event, 'expired')


def build_and_run(frames, user_cls, user_params, starting_cash, leverage,
                  commission, bar_minutes=1, step_hook=None, log=print):
    """frames: {symbol: DataFrame[open, high, low, close, volume] indexed by
    UTC timestamp}. Returns (driver, user_strategy, engine). The caller must
    call engine.dispose() when done with the results."""
    engine = BacktestEngine(config=BacktestEngineConfig(
        trader_id='BACKTESTER-001',
        logging=LoggingConfig(log_level='ERROR'),
        risk_engine=RiskEngineConfig(bypass=True),
    ))
    engine.add_venue(
        venue=Venue(VENUE),
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=USDT,
        starting_balances=[Money(starting_cash, USDT)],
        default_leverage=Decimal(leverage),
        book_type=BookType.L1_MBP,
        bar_execution=True,
    )

    spec = '%d-HOUR' % (bar_minutes // 60) if bar_minutes >= 60 else '%d-MINUTE' % bar_minutes
    feeds, bar_types, volumes, n_bars, last_ns = {}, [], {}, 0, 0
    for sym, df in frames.items():
        pp = price_precision(df)
        inst = make_instrument(sym, pp, commission)
        engine.add_instrument(inst)
        bar_type = BarType.from_str('%s-%s-LAST-EXTERNAL' % (inst.id, spec))
        clean = df[['open', 'high', 'low', 'close', 'volume']].copy()
        for c in ('open', 'high', 'low', 'close'):
            clean[c] = clean[c].round(pp)
        clean['high'] = clean[['open', 'high', 'low', 'close']].max(axis=1)
        clean['low'] = clean[['open', 'high', 'low', 'close']].min(axis=1)
        real_volume = clean['volume'].clip(lower=0).tolist()
        clean['volume'] = float(BOOK_VOLUME)
        bars = BarDataWrangler(bar_type, inst).process(clean)
        volumes[sym] = real_volume
        engine.add_data(bars, validate=False, sort=False)
        n_bars += len(bars)
        last_ns = max(last_ns, bars[-1].ts_init)
        feeds[inst.id] = Feed(sym, inst.id)
        bar_types.append(bar_type)
    engine.sort_data()
    log('[nautilus] %d instruments, %d bars loaded' % (len(feeds), n_bars))

    driver = PortfolioDriver(feeds, bar_types, commission)
    driver.volumes = volumes
    driver.step_hook = step_hook
    # feeds in symbol order (backtrader added them alphabetically)
    ordered = sorted(feeds.values(), key=lambda f: f._name)
    user = user_cls(ordered, driver, **user_params)
    driver.user = user
    engine.add_strategy(driver)
    # run a hair past the last bar so its step alert (last_ns + 1) fires too
    engine.run(end=last_ns + 1_000)
    return driver, user, engine
