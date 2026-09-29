"""Portfolio strategy base: the backtrader-style interface on a Nautilus engine.

A strategy subclasses ``PortfolioStrategy`` and implements:
  setup()               create per-feed state and indicators
  on_bar(d, price)      return a signal (see next() for the accepted shapes)
  build_chart_lines()   overlay series for the dashboard's candle chart

This is the same contract the backtrader version had, so strategy files port
unchanged apart from ``import bt_compat as bt``. Execution goes through
NautilusTrader: ``backend/nt_engine.py`` owns a Nautilus ``Strategy`` that
receives the bars, calls ``next()`` once per timestamp after every
instrument's bar for that timestamp has arrived, and routes the target-size
orders placed here to the simulated exchange as market orders. Fills come back
through ``_on_fill``, which keeps the per-symbol position lifecycle log
(``trade_log``) the dashboard and reports read.

Fill timing differs from backtrader. backtrader filled a market order at the
NEXT bar's open. Nautilus fills it immediately, at the close of the bar the
decision was made on, so ``signal_dt`` and the fill ``dt`` are the same bar.
"""
from datetime import datetime, timezone


def epoch(dt):
    """Naive-UTC datetime -> integer epoch seconds (lightweight-charts time)."""
    return int(dt.replace(tzinfo=timezone.utc).timestamp())


def _iso(ts_s):
    return datetime.fromtimestamp(ts_s, timezone.utc).replace(tzinfo=None).isoformat()


# ---------------------------------------------------------------------------
# feeds: backtrader-style lines over bars pushed in by the engine
# ---------------------------------------------------------------------------
class Line:
    __slots__ = ('array', 'feed')

    def __init__(self, feed):
        self.array = []
        self.feed = feed

    def __getitem__(self, ago):
        # backtrader indexing: [0] = current bar, [-1] = previous, ...
        return self.array[len(self.array) - 1 + ago]

    def __len__(self):
        return len(self.array)

    def get(self, ago=0, size=1):
        end = len(self.array) + ago
        return self.array[max(0, end - size):end]


class DateTimeLine(Line):
    __slots__ = ()

    def datetime(self, ago=0):
        return datetime.fromtimestamp(self[ago], timezone.utc).replace(tzinfo=None)

    def date(self, ago=0):
        return self.datetime(ago).date()


class Feed:
    """One instrument's bar series. ``_name`` is the plain symbol (BTCUSDT);
    ``instrument_id`` is the Nautilus id it trades under."""

    def __init__(self, name, instrument_id=None):
        self._name = name
        self.instrument_id = instrument_id
        self.datetime = DateTimeLine(self)
        self.open = Line(self)
        self.high = Line(self)
        self.low = Line(self)
        self.close = Line(self)
        self.volume = Line(self)
        self._indicators = []

    def __len__(self):
        return len(self.close.array)

    def __repr__(self):
        return '<Feed %s bars=%d>' % (self._name, len(self))

    def _push(self, ts_s, o, h, low, c, v):
        self.datetime.array.append(ts_s)
        self.open.array.append(o)
        self.high.array.append(h)
        self.low.array.append(low)
        self.close.array.append(c)
        self.volume.array.append(v)
        for ind in self._indicators:
            ind._update()


# ---------------------------------------------------------------------------
# params: backtrader's merged `params = (('name', default), ...)` tuples
# ---------------------------------------------------------------------------
class Params:
    def __init__(self, values):
        self.__dict__.update(values)

    def _getkeys(self):
        return list(self.__dict__.keys())

    def _getitems(self):
        return list(self.__dict__.items())


class _ParamsMeta(type):
    def __new__(mcs, name, bases, ns):
        own = ns.get('params', ())
        own = list(own.items()) if isinstance(own, dict) else list(own)
        merged = {}
        for b in reversed(bases):
            merged.update(getattr(b, '_param_defaults', {}))
        merged.update(dict(own))
        ns['_param_defaults'] = merged
        return super().__new__(mcs, name, bases, ns)


class _Position:
    __slots__ = ('size', 'price')

    def __init__(self, size=0.0, price=0.0):
        self.size = size
        self.price = price


class PortfolioStrategy(metaclass=_ParamsMeta):

    params = (
        ('trade_usd', 2000.0),
        ('printlog', False),
        ('diag', False),      # record per-bar filter internals into self._diag
        # ---- trailing exit (opt-in) -----------------------------------
        # A ratcheting percent stop, independent of on_bar()'s own signal:
        # long  -> stop sits trail_pct BELOW price, only ever moves UP.
        # short -> stop sits trail_pct ABOVE price, only ever moves DOWN.
        ('use_trail', False),
        ('trail_pct', 0.05),
    )

    # ---- to override ----------------------------------------------------
    def setup(self):
        pass

    def on_bar(self, d, price):
        return 0

    def build_chart_lines(self):
        """{symbol: [{'name', 'color', 'points': [{'time', 'value'}]}]}"""
        return {}

    # ---- plumbing ---------------------------------------------------------
    def __init__(self, datas, broker, **kwargs):
        unknown = set(kwargs) - set(self._param_defaults)
        if unknown:
            raise TypeError('%s got unexpected keyword argument(s): %s'
                            % (type(self).__name__, ', '.join(sorted(unknown))))
        self.params = self.p = Params(dict(self._param_defaults, **kwargs))
        self.datas = list(datas)
        self.data = self.datas[0] if self.datas else None
        self.broker = broker

        self.orders = {}       # data -> pending order ref
        self._last_len = {}    # data -> last seen bar count
        self.executed = {}     # symbol -> [{signal_dt, dt, side, price, size}]
        self.trade_log = []    # one entry per CLOSED position (open->close)
        self.equity = []       # [(iso_dt, account_value)]
        self._order_sig = {}   # order ref -> signal-bar datetime (when decided)
        self._trail_exit_price = {}   # data -> stop price of an in-flight trail-exit close
        self._diag = {}        # symbol -> [per-bar filter internals] (when p.diag)
        self._trail_stop = {}  # data -> current ratcheted trailing-stop price
        self._trade = {}       # data -> open position lifecycle (entry side of the log)

        for d in self.datas:
            self._last_len[d] = 0
            self.executed[d._name] = []

        self.setup()

    def log(self, txt, dt=None, doprint=False):
        if self.params.printlog or doprint:
            if dt is None:
                ref = next((d for d in self.datas if len(d)), None)
                dt = ref.datetime.datetime(0) if ref is not None else datetime.utcnow()
            print('%s, %s' % (dt.isoformat(), txt))

    def _diag_record(self, d, price, y_pred, innov, S, band, st):
        """One bar of filter internals for offline analysis. No-op unless the
        `diag` param is on. Kalman/EM strategies also carry a state vector X
        and covariance P, captured as x0.. / P00.. when present."""
        if not self.params.diag:
            return
        rec = {'t': self.bar_epoch(d), 'price': price, 'pred': y_pred,
               'upper': y_pred + band, 'lower': y_pred - band,
               'innov': innov, 'S': S, 'band': band}
        X, P = st.get('X'), st.get('P')
        if X is not None and P is not None:
            n = X.shape[0]
            for i in range(n):
                rec['x%d' % i] = float(X[i, 0])
            for i in range(n):
                for j in range(i, n):
                    rec['P%d%d' % (i, j)] = float(P[i, j])
        self._diag.setdefault(d._name, []).append(rec)

    def bar_epoch(self, d):
        """Epoch seconds of the current bar on feed `d` (for chart points)."""
        return int(d.datetime[0])

    # ---- broker-facing helpers (same names as backtrader) -----------------
    def getposition(self, d):
        size, price = self.broker.position(d)
        return _Position(size, price)

    def order_target_size(self, data, target):
        return self.broker.submit_target(data, target)

    # ---- fills from the Nautilus engine -------------------------------
    def _on_fill(self, d, ref, side, price, qty, commission, ts_s, order_done):
        """One fill. ``qty`` is unsigned; ``side`` is 'BUY'/'SELL'. Mirrors
        backtrader's notify_order + notify_trade bookkeeping."""
        sig_dt = self._order_sig.get(ref)
        if order_done:
            self._order_sig.pop(ref, None)
            self.orders.pop(d, None)
        signed = qty if side == 'BUY' else -qty
        self.executed[d._name].append({
            'signal_dt': sig_dt, 'dt': _iso(ts_s), 'side': side,
            'price': price, 'size': signed,
        })
        self.log('%s %s EXECUTED, Price: %.6f, Size: %.6f' % (d._name, side, price, signed))

        remaining, comm_left = signed, commission
        while abs(remaining) > 1e-12:
            tr = self._trade.get(d)
            if tr is None:                       # flat -> open a new lifecycle
                self._trade[d] = {'size': remaining, 'price': price, 'pnl': 0.0,
                                  'comm': comm_left, 'entry_dt': _iso(ts_s),
                                  'entry_signal_dt': sig_dt, 'open_len': len(d),
                                  'orig_size': remaining}
                return
            if (tr['size'] > 0) == (remaining > 0):   # adding to the position
                new = tr['size'] + remaining
                tr['price'] = (tr['price'] * tr['size'] + price * remaining) / new
                tr['size'] = new
                tr['comm'] += comm_left
                return
            # reducing / closing / flipping
            close_qty = min(abs(remaining), abs(tr['size']))
            frac = close_qty / abs(remaining)
            direction = 1.0 if tr['size'] > 0 else -1.0
            tr['pnl'] += close_qty * (price - tr['price']) * direction
            tr['comm'] += comm_left * frac
            tr['size'] += close_qty * (-direction)
            remaining += close_qty * direction
            comm_left *= (1.0 - frac)
            if abs(tr['size']) < 1e-12:
                self._close_trade(d, tr, ts_s, sig_dt)
            else:
                return

    def _close_trade(self, d, tr, ts_s, sig_dt):
        self._trade.pop(d, None)
        size = tr['orig_size']
        entry = tr['price']
        pnl = tr['pnl']
        # like backtrader: exit price backed out of gross pnl on the entry size
        exit_price = (entry + pnl / size) if size else None
        trail_price = self._trail_exit_price.pop(d, None)
        self.trade_log.append({
            'symbol': d._name,
            'side': 'LONG' if size > 0 else 'SHORT',
            'size': abs(size),
            'entry_signal_dt': tr['entry_signal_dt'],
            'entry_dt': tr['entry_dt'],
            'exit_signal_dt': sig_dt,
            'exit_dt': _iso(ts_s),
            'entry_price': round(entry, 8),
            'exit_price': round(exit_price, 8) if exit_price is not None else None,
            'bars_held': len(d) - tr['open_len'],
            'pnl': round(pnl, 6),
            'pnlcomm': round(pnl - tr['comm'], 6),
            'exit_reason': 'trail' if trail_price is not None else 'signal',
            'trail_stop_price': round(trail_price, 8) if trail_price is not None else None,
        })
        self.log('%s %s CLOSED, size %.6f, GROSS %.2f, NET %.2f'
                 % (d._name, 'LONG' if size > 0 else 'SHORT', abs(size), pnl, pnl - tr['comm']))

    def _on_order_dead(self, d, ref):
        """Rejected / denied / canceled: the order never (fully) happened."""
        self._order_sig.pop(ref, None)
        self._trail_exit_price.pop(d, None)
        self.orders.pop(d, None)

    # ---- one portfolio step (called once per bar timestamp) ----------
    def next(self):
        ref = next((d for d in self.datas if len(d)), None)
        if ref is None:
            return
        self.equity.append((
            ref.datetime.datetime(0).isoformat(),
            round(self.broker.getvalue(), 4),
        ))

        for d in self.datas:
            if len(d) == 0 or len(d) == self._last_len[d]:
                continue
            self._last_len[d] = len(d)

            price = d.close[0]
            if price <= 0:
                continue

            # Always advance the model, even if an order is in flight.
            result = self.on_bar(d, price)

            if d in self.orders:      # pending order on this instrument -> wait
                continue

            # on_bar returns a scalar signal, (signal, size), or
            # (signal, size, stop_pct). stop_pct is a SIGNED whole-number
            # percent requesting a trailing exit for this bar: negative ->
            # protective side (below price for longs, above for shorts),
            # positive -> the opposite side. Omitted -> the run's
            # use_trail/trail_pct params (protective side, fraction).
            stop_pct = None
            if isinstance(result, (tuple, list)):
                signal = result[0]
                size = result[1] if len(result) > 1 else None
                if len(result) > 2 and result[2] is not None:
                    stop_pct = float(result[2]) / 100.0
            else:
                signal, size = result, None

            pos_size = self.getposition(d).size
            if pos_size == 0:
                self._trail_stop.pop(d, None)
            else:
                signed_pct = stop_pct if stop_pct is not None else (
                    -self.params.trail_pct if self.params.use_trail else None)
                if signed_pct:
                    below = (pos_size > 0) != (signed_pct > 0)
                    pct_abs = abs(signed_pct)
                    cur = self._trail_stop.get(d)
                    if below:
                        candidate = price * (1 - pct_abs)
                        new_stop = candidate if cur is None else max(cur, candidate)
                        hit = price <= new_stop
                    else:
                        candidate = price * (1 + pct_abs)
                        new_stop = candidate if cur is None else min(cur, candidate)
                        hit = price >= new_stop
                    self._trail_stop[d] = new_stop
                    if hit:
                        order = self.order_target_size(data=d, target=0)
                        if order is not None:
                            self.orders[d] = order
                            self._order_sig[order] = d.datetime.datetime(0).isoformat()
                            self._trail_exit_price[d] = new_stop
                        self._trail_stop.pop(d, None)
                        continue   # trailing exit overrides this bar's signal
                else:
                    self._trail_stop.pop(d, None)

            if signal != signal:      # NaN signal (indicator not ready)
                continue

            # Explicit flatten: (0, 0) -> go to cash. A bare 0 means hold.
            if signal == 0 and size == 0:
                if self.getposition(d).size != 0:
                    order = self.order_target_size(data=d, target=0)
                    if order is not None:
                        self.orders[d] = order
                        self._order_sig[order] = d.datetime.datetime(0).isoformat()
                continue

            if not signal:
                continue

            magnitude = size if size is not None else (self.params.trade_usd / price)
            target = signal * magnitude
            order = self.order_target_size(data=d, target=target)
            if order is not None:
                self.orders[d] = order
                self._order_sig[order] = d.datetime.datetime(0).isoformat()

    def prenext(self):
        self.next()

    def stop(self):
        self.log('%s ending value %.2f'
                 % (type(self).__name__, self.broker.getvalue()), doprint=True)
