from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId

from trading.strategies.ema import EMAConfig, EMAStrategy


class EMACrossSARConfig(EMAConfig, frozen=True):
    bar_spec: str = "10-TICK-LAST"
    fast_ema_period: int = 15
    slow_ema_period: int = 30
    warmup_factor: int = 2
    max_open_instruments: int = 50


class EMACrossStopReverse(EMAStrategy):
    """Contrarian stop-and-reverse on EMA crossovers: a bearish cross closes any
    short and goes long; a bullish cross closes any long and goes short."""

    def __init__(self, config: EMACrossSARConfig):
        super().__init__(config)
        self._prev_signal: dict[InstrumentId, str | None] = {}

    def description(self) -> str:
        c = self.config
        return (f"Contrarian EMA stop-and-reverse on {len(c.instrument_ids)} instruments using {c.bar_spec} "
                f"bars, ${float(c.trade_usd):,.0f} per entry. When the {c.fast_ema_period}-period EMA crosses "
                f"below the {c.slow_ema_period}-period EMA it closes any short and goes long; when it crosses "
                f"above it closes any long and goes short. At most {c.max_open_instruments} instruments open at once.")

    def reset_instrument(self, iid: InstrumentId) -> None:
        super().reset_instrument(iid)
        self._prev_signal.pop(iid, None)

    def on_ema(self, iid: InstrumentId, fast: float, slow: float, price: float) -> None:
        if self.has_working_order(iid):
            return      # don't consume the crossover: decide again next bar
        signal = "bull" if fast > slow else "bear" if slow > fast else None
        prev = self._prev_signal.get(iid)
        self._prev_signal[iid] = signal
        if prev is None or signal is None or signal == prev:
            return
        net = self.net_qty(iid)
        usd = float(self.config.trade_usd)
        if signal == "bear" and net <= 0:
            if net < 0:
                self.close_all_positions(iid)
            if self._can_enter(iid):
                self.submit_market(iid, OrderSide.BUY, usd, price)
        elif signal == "bull" and net >= 0:
            if net > 0:
                self.close_all_positions(iid)
            if self._can_enter(iid):
                self.submit_market(iid, OrderSide.SELL, usd, price)

    def _can_enter(self, iid: InstrumentId) -> bool:
        # A reversal reuses the instrument's slot; only new instruments count
        # against the cap (open positions + entries still in flight).
        busy = self.instruments_with_exposure() | {o.instrument_id for o in self.working_orders()}
        if iid in busy:
            return True
        if len(busy) < self.config.max_open_instruments:
            return True
        self._warn_once(f"cap:{self.id}", "Entry skipped: instrument cap reached",
                        f"{len(busy)} instruments already open or entering (cap "
                        f"{self.config.max_open_instruments}). Further entries are skipped until one closes.")
        return False
