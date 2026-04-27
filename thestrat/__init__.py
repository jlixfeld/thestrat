"""TheStrat — portable aggregator and bar classifier for Strat methodology.

Public API:
- TimeframeAggregator: 1m → {5m, 15m, 30m, 1h, D, W, M, Q, Y} OHLCV aggregation
- classify_bar / classify_bars_df / classify_bars_multi_symbol: scenario, color, shape, in-force
- BarDict, ClassifiedBar, Color, Scenario, Shape, Timeframe: domain types
"""

from importlib.metadata import PackageNotFoundError, version

from thestrat.aggregator import EQUITY_OFFSET_MINUTES, TimeframeAggregator
from thestrat.classifier import (
    SHAPE_BODY_ZONE,
    classify_bar,
    classify_bars_df,
    classify_bars_multi_symbol,
    classify_color,
    classify_scenario,
)
from thestrat.types import (
    BarDict,
    ClassifiedBar,
    Color,
    Scenario,
    Shape,
    Timeframe,
)

try:
    __version__ = version("thestrat")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "TimeframeAggregator",
    "EQUITY_OFFSET_MINUTES",
    "classify_bar",
    "classify_bars_df",
    "classify_bars_multi_symbol",
    "classify_color",
    "classify_scenario",
    "SHAPE_BODY_ZONE",
    "BarDict",
    "ClassifiedBar",
    "Color",
    "Scenario",
    "Shape",
    "Timeframe",
    "__version__",
]
