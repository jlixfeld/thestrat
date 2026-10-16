"""TheStrat: explicit OHLCV aggregation and composed bar classification."""

from importlib.metadata import PackageNotFoundError, version

from thestrat.api import InsufficientSourceDataError, TheStrat
from thestrat.sessions import SESSIONS, InstrumentType, Schedule, ScheduleRangeError, Session, session_for
from thestrat.types import DEFAULT_TIMEFRAMES, BarDict, Color, CurrentBar, Scenario, Shape, Timeframe

try:
    __version__ = version("thestrat")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "TheStrat",
    "InsufficientSourceDataError",
    "Session",
    "Schedule",
    "ScheduleRangeError",
    "InstrumentType",
    "SESSIONS",
    "session_for",
    "Timeframe",
    "DEFAULT_TIMEFRAMES",
    "BarDict",
    "CurrentBar",
    "Color",
    "Scenario",
    "Shape",
    "__version__",
]
