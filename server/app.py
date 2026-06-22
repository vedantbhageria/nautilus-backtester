"""FastAPI dashboard backend.

Responsibilities (implementation TBD):
- Serve the static dashboard page (server/static/index.html).
- Consume engine events from Redis streams -> push to clients over websocket.
- Read historical bars from the Parquet ParquetDataCatalog for backfill.
- Publish dashboard commands (load symbol, subscribe bars, ...) to Redis,
  where the ControlActor picks them up.
"""
