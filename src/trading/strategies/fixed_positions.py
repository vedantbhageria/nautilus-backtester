from decimal import Decimal

from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId

from trading.strategies.base import MIN_ORDER_USD, DashboardStrategy, DashboardStrategyConfig


class FixedNotionalConfig(DashboardStrategyConfig, frozen=True):
    target_usd: Decimal = Decimal("2000")
    rebalance_threshold: float = 0.001


class FixedNotional(DashboardStrategy):
    """Hold ``target_usd`` of each instrument; rebalance on every trade tick
    once the position drifts more than ``rebalance_threshold``."""

    def description(self) -> str:
        c = self.config
        return (f"Holds ${float(c.target_usd):,.0f} notional of each of {len(c.instrument_ids)} instruments. "
                f"Rebalances when drift exceeds {c.rebalance_threshold * 100:.1f}%.")

    def arm(self) -> str:
        msg = super().arm()
        for iid in self._tradeable:
            self.subscribe_trade_ticks(iid)
        return msg

    def disarm(self) -> str:
        for iid in self._tradeable:
            self.unsubscribe_trade_ticks(iid)
        return super().disarm()

    def close_all(self) -> str:
        for iid in self._tradeable:
            self.unsubscribe_trade_ticks(iid)
        return super().close_all()

    def on_trade_tick(self, tick: TradeTick) -> None:
        if self.armed and not self.is_exiting():
            self._rebalance(tick.instrument_id, float(tick.price))

    def _rebalance(self, iid: InstrumentId, price: float) -> None:
        if price <= 0 or self.has_working_order(iid):
            return
        value = self.net_qty(iid) * price
        target = float(self.config.target_usd)
        if abs(value - target) / target < self.config.rebalance_threshold:
            return
        diff = target - value
        if abs(diff) < MIN_ORDER_USD:
            return                          # drift too small to trade; not an error
        side = OrderSide.BUY if diff > 0 else OrderSide.SELL
        if self.submit_market(iid, side, abs(diff), price):
            self.publish_metrics({"target_usd": target})
