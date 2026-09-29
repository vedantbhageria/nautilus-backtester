from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId

from trading.strategies.ema import EMAConfig, EMAStrategy


class EMACrossConfig(EMAConfig, frozen=True):
    bar_spec: str = "10-TICK-LAST"
    warmup_factor: int = 4


class EMACross(EMAStrategy):
    """Long-only EMA regime: hold a long while fast >= slow, flat otherwise."""

    def description(self) -> str:
        c = self.config
        return (f"Long-only EMA crossover on {len(c.instrument_ids)} instruments using {c.bar_spec} bars. "
                f"Buys ${float(c.trade_usd):,.0f} notional while the {c.fast_ema_period}-period EMA is at "
                f"or above the {c.slow_ema_period}-period EMA; exits the whole position when it drops below.")

    def on_ema(self, iid: InstrumentId, fast: float, slow: float, price: float) -> None:
        if self.has_working_order(iid):
            return                          # last decision still in flight; re-evaluate next bar
        net = self.net_qty(iid)
        if fast >= slow:
            if net == 0:
                self.submit_market(iid, OrderSide.BUY, float(self.config.trade_usd), price)
        elif net > 0:
            self.close_all_positions(iid)
