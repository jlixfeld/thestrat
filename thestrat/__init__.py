"""
TheStrat Python Module

Standalone module for vectorized Strat technical analysis and OHLC timeframe aggregation.
Supports historical and real-time market data processing across all asset classes.
"""

from .aggregation import Aggregation
from .base import Component
from .factory import Factory
from .indicators import Indicators
from .schemas import (
    AggregationConfig,
    FactoryConfig,
    GapDetectionConfig,
    IndicatorsConfig,
    SwingPointsConfig,
    TimeframeItemConfig,
)
from .signals import SIGNALS, PriceChange, SignalBias, SignalCategory, SignalMetadata, SignalStatus

__version__ = "1.0.0"
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
]
