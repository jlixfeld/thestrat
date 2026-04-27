"""Strat-domain types: bar shapes, scenarios, colors, timeframes."""

from datetime import datetime
from enum import StrEnum
from typing import TypedDict


class BarDict(TypedDict):
    """OHLCV bar dictionary — the standard shape used across fetching, streaming, and storage."""

    symbol: str
    timestamp: datetime
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class Color(StrEnum):
    """Bar color based on open vs close."""

    GREEN = "green"
    RED = "red"
    NEUTRAL = "neutral"


class Scenario(StrEnum):
    """Strat scenario classification."""

    ONE = "1"
    TWO_UP = "2U"
    TWO_DOWN = "2D"
    THREE = "3"


class Shape(StrEnum):
    """Bar shape classification — hammer or shooter."""

    HAMMER = "hammer"
    SHOOTER = "shooter"


class ClassifiedBar(BarDict):
    """OHLCV bar with Strat classification fields."""

    scenario3: Scenario | None
    scenario2: Scenario | None
    scenario1: Scenario | None
    scenario0: Scenario | None
    color3: Color | None
    color2: Color | None
    color1: Color | None
    color0: Color
    shape: str | None
    in_force: bool | None


class Timeframe(StrEnum):
    """Bar timeframe values."""

    FIVE_SEC = "5sec"
    ONE_MIN = "1min"
    FIVE_MIN = "5min"
    FIFTEEN_MIN = "15min"
    THIRTY_MIN = "30min"
    ONE_HOUR = "1h"
    FOUR_HOUR = "4h"
    SIX_HOUR = "6h"
    TWELVE_HOUR = "12h"
    ONE_DAY = "1d"
    ONE_WEEK = "1w"
    ONE_MONTH = "1m"  # Calendar month aggregation, not 1 minute (see ONE_MIN)
    ONE_QUARTER = "1q"
    ONE_YEAR = "1y"
