from nautilus_trader.common.actor import Actor
from nautilus_trader.config import ActorConfig
from nautilus_trader.model import InstrumentId
from nautilus_trader.model.events import OrderFilled

#Actors can subscribe to order fill events for specific instruments using subscribe_order_fills(). This is useful for monitoring trading activity, fill analysis, or tracking execution quality.

#When subscribed, the handler on_order_filled() receives all fills for the specified instrument, regardless of which strategy or component generated the original orde

class MyActorConfig(ActorConfig):
    instrument_id: InstrumentId  # example value: "ETHUSDT-PERP.BINANCE"


class FillMonitorActor(Actor):
    def __init__(self, config: MyActorConfig) -> None:
        super().__init__(config)
        self.fill_count = 0
        self.total_volume = 0.0

    def on_start(self) -> None:
        # Subscribe to all fills for the instrument
        self.subscribe_order_fills(self.config.instrument_id)

    def on_order_filled(self, event: OrderFilled) -> None:
        # Handle order fill events
        self.fill_count += 1
        self.total_volume += float(event.last_qty)

        self.log.info(
            f"Fill received: {event.order_side} {event.last_qty} @ {event.last_px}, "
            f"Total fills: {self.fill_count}, Volume: {self.total_volume}"
        )

    def on_stop(self) -> None:
        # Unsubscribe from fills
        self.unsubscribe_order_fills(self.config.instrument_id)