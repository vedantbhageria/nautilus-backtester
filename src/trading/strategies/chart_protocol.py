"""Pluggable chart + backtest hooks, so strategies can be swapped without editing
the backtest runner or the dashboard.

Everything strategy-specific about
  (a) how the strategy is instantiated inside the BacktestEngine, and
  (b) what lines it draws on the chart
lives on the strategy class itself, behind four hooks. `backtest_runner` and the
dashboard consume them generically.

These are resolved by duck typing (plain `getattr`) rather than a base class, so
there's no multiple-inheritance interaction with Nautilus's Cython `Strategy`.

Implement on your Strategy class:

    @classmethod
    def backtest_config(cls, cfg_live):
        '''REQUIRED. The config used to run THIS strategy inside the BacktestEngine.
        Usually the live config with bar_spec="1-MINUTE-LAST" and backtest=True.'''

    @classmethod
    def chart_indicators(cls, config) -> dict[str, object]:
        '''OPTIONAL. name -> indicator exposing .handle_bar(bar), .initialized, .value.
        The runner feeds historical klines through these to recompute the strategy's
        chart lines. A fresh set is built per instrument. Return {} to draw nothing.'''

    @classmethod
    def chart_spec(cls, config) -> list[dict]:
        '''OPTIONAL. Cosmetics for the dashboard:
        [{"name": "fast_ema", "color": "#a855f7", "panel": "overlay", "label": "EMA 60"}]
        `panel` is "overlay" (on the price chart) or "oscillator" (sub-panel).
        The dashboard renders any series it finds even without a spec — the spec just
        supplies colours/labels/panel placement.'''

    def chart_values(self, instrument_id) -> dict[str, float]:
        '''OPTIONAL. Live per-bar values for this instrument, e.g.
        {"fast_ema": 1.234, "slow_ema": 1.230}. Return {} while warming up.'''

Adding a new strategy therefore means: write the class, implement `backtest_config`
(+ the chart hooks if it draws anything), and register it in `binance_data.py`.
Nothing in `backtest_runner.py` or `dashboard.html` needs to change.
"""

from __future__ import annotations


def backtest_config_for(strategy):
    """Build the BacktestEngine config for `strategy`'s class from its live config."""
    cls = type(strategy)
    fn = getattr(cls, "backtest_config", None)
    if fn is None:
        raise TypeError(
            f"{cls.__name__} must define a `backtest_config(cls, cfg_live)` classmethod "
            f"so the backtest runner knows how to instantiate it. See chart_protocol.py."
        )
    return fn(strategy.config)


def chart_indicators_for(strategy) -> dict:
    """A FRESH set of chart indicators for one instrument. {} if the strategy declares none."""
    fn = getattr(type(strategy), "chart_indicators", None)
    if fn is None:
        return {}
    try:
        return dict(fn(strategy.config) or {})
    except Exception:
        return {}


def chart_spec_for(strategy) -> list:
    """Series cosmetics for the dashboard. [] if the strategy declares none."""
    fn = getattr(type(strategy), "chart_spec", None)
    if fn is None:
        return []
    try:
        return list(fn(strategy.config) or [])
    except Exception:
        return []


def chart_values_for(strategy, instrument_id) -> dict:
    """Current live values of the strategy's chart series for one instrument."""
    fn = getattr(strategy, "chart_values", None)
    if fn is None:
        return {}
    try:
        return dict(fn(instrument_id) or {})
    except Exception:
        return {}
