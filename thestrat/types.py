"""Public bar values and supported timeframe identifiers."""

from datetime import datetime
from enum import StrEnum
from typing import NotRequired, TypedDict


class BarDict(TypedDict):
    symbol: str | None
    timestamp: datetime
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: NotRequired[object]


class Color(StrEnum):
    GREEN = "green"
    RED = "red"
    NEUTRAL = "neutral"


class Scenario(StrEnum):
    ONE = "1"
    TWO_UP = "2U"
    TWO_DOWN = "2D"
    THREE = "3"


class Shape(StrEnum):
    HAMMER = "hammer"
    SHOOTER = "shooter"


class CurrentBar(BarDict):
    scenario: Scenario | None
    color: Color
    shape: Shape | None
    in_force: bool | None


class Timeframe(StrEnum):
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
    ONE_MONTH = "1m"
    ONE_QUARTER = "1q"
    ONE_YEAR = "1y"


DEFAULT_TIMEFRAMES = tuple(Timeframe)
