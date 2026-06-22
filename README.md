# nautilus-trading

NautilusTrader pipeline: backtesting now, live/paper soon, with a streaming
browser dashboard for historical + live data and interactive control.

## Layout

```
src/trading/            importable package (pip install -e .)
  strategies/           trading strategies (EMACross, ...)
  indicators/           custom indicators
  actors/               ControlActor — dashboard -> engine command surface
  configs/              TradingNode / backtest configs
server/                 FastAPI dashboard backend
  app.py                websockets + Redis consumer + Parquet reader
  static/               frontend (lightweight-charts)
scripts/                build_catalog / run_backtest / run_live entrypoints
examples/               reference snippets (not imported)
tests/
```

## Architecture

- **One shared Redis.** Nautilus' message bus and the dashboard use the same
  instance. The cache/message bus are Nautilus abstractions — Redis is the
  optional backend, not the bus itself.
- **Two stream directions:** engine -> dashboard (events) and dashboard ->
  engine (commands, handled by `ControlActor`).
- **History from the Parquet catalog; live tail from Redis streams.**
- Interactive symbol-loading is a **live/sandbox node** feature; backtests are
  configure-then-run.

## Setup

```bash
cp .env.example .env        # set CATALOG_PATH (outside OneDrive), REDIS_URL, keys
pip install -e .[dev]
```

The Parquet catalog lives at `$CATALOG_PATH`, kept outside the repo and
OneDrive sync (see `.gitignore`).
