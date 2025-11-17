"""
TheStrat Python Module

Standalone module for vectorized Strat technical analysis and OHLC timeframe aggregation.
Supports historical and real-time market data processing across all asset classes.
"""

from .aggregation import Aggregation
from .base import Component
from .factory import Factory
from .indicators import Indicators
from .precision import (  # noqa: F401
    PrecisionError,
    apply_precision,
    get_comparison_tolerance,
    get_field_decimal_places,
    get_field_precision_type,
)
from .schemas import (
    AggregationConfig,
    FactoryConfig,
    GapDetectionConfig,
    IndicatorSchema,
    IndicatorsConfig,
    SwingPointsConfig,
    TimeframeItemConfig,
)
from .signals import SIGNALS, PriceChange, SignalBias, SignalCategory, SignalMetadata, SignalStatus

try:
    from importlib.metadata import version

    __version__ = version("thestrat")
except Exception:
    # Fallback if package not found (development mode)
    __version__ = "ERROR: VERSION NOT FOUND"
__all__ = [
    "Factory",
    "Component",
    "Aggregation",
    "Indicators",
    "AggregationConfig",
    "FactoryConfig",
    "GapDetectionConfig",
    "IndicatorsConfig",
    "SwingPointsConfig",
    "TimeframeItemConfig",
    "SignalMetadata",
    "SignalCategory",
    "SignalBias",
    "SignalStatus",
    "PriceChange",
    "SIGNALS",
    "IndicatorSchema",
    "PrecisionError",
    "apply_precision",
    "get_field_decimal_places",
    "get_field_precision_type",
    "get_comparison_tolerance",
]
