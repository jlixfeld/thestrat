"""Eligible exchange sessions and continuous UTC sessions.

Presets use pandas-market-calendars NYSE and CME_Equity (CME equity index
futures only). Calendar queries are supported from 1900 through 2100.
FX has no preset because no product-specific calendar has been selected.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Protocol, cast
from zoneinfo import ZoneInfo

import pandas as pd
import pandas_market_calendars as mcal
import polars as pl

SCHEDULE_SCHEMA = {
    name: pl.Datetime("us", "UTC") for name in ("session_open", "session_close", "segment_open", "segment_close")
}


class ScheduleRangeError(ValueError):
    """The requested calendar date range is unsupported."""


class Schedule(Protocol):
    def __call__(self, *, start: datetime, end: datetime) -> pl.DataFrame: ...


def _bounds(start: datetime, end: datetime) -> None:
    if (
        start.tzinfo is None
        or end.tzinfo is None
        or start.utcoffset() != timedelta(0)
        or end.utcoffset() != timedelta(0)
        or start >= end
    ):
        raise ValueError("schedule bounds must be aware UTC with start < end")
    if start < datetime(1900, 1, 1, tzinfo=UTC) or end > datetime(2101, 1, 1, tzinfo=UTC):
        raise ScheduleRangeError("calendar supports 1900 through 2100")


@dataclass(frozen=True)
class CalendarSchedule:
    """pandas-market-calendars rule adapter, bounded to 1900–2100."""

    calendar: str

    def __call__(self, *, start: datetime, end: datetime) -> pl.DataFrame:
        _bounds(start, end)
        calendar = mcal.get_calendar(self.calendar)
        table = calendar.schedule(
            start_date=(start - timedelta(days=2)).date(), end_date=(end + timedelta(days=2)).date()
        )
        rows = []
        for _, row in table.iterrows():
            opened, closed = (
                cast(pd.Timestamp, row["market_open"]).to_pydatetime(),
                cast(pd.Timestamp, row["market_close"]).to_pydatetime(),
            )
            segments = [(opened, closed)]
            if "break_start" in row and not pd.isna(cast(pd.Timestamp, row["break_start"])):
                segments = [
                    (opened, cast(pd.Timestamp, row["break_start"]).to_pydatetime()),
                    (cast(pd.Timestamp, row["break_end"]).to_pydatetime(), closed),
                ]
            for left, right in segments:
                left, right = max(opened, left), min(closed, right)
                if left < right and left < end and right > start:
                    rows.append((opened, closed, left, right))
        return pl.DataFrame(rows, schema=SCHEDULE_SCHEMA, orient="row").sort("session_open", "segment_open")


def _crypto_schedule(*, start: datetime, end: datetime) -> pl.DataFrame:
    _bounds(start, end)
    day = start.replace(hour=0, minute=0, second=0, microsecond=0)
    rows = []
    while day < end:
        close = day + timedelta(days=1)
        rows.append((day, close, day, close))
        day = close
    return pl.DataFrame(rows, schema=SCHEDULE_SCHEMA, orient="row")


@dataclass(frozen=True)
class Session:
    timezone: str
    anchor_minutes: int
    schedule: Schedule

    def __post_init__(self) -> None:
        ZoneInfo(self.timezone)
        if (
            isinstance(self.anchor_minutes, bool)
            or not isinstance(self.anchor_minutes, int)
            or not 0 <= self.anchor_minutes < 1440
        ):
            raise ValueError("anchor_minutes must be an integer from 0 through 1439")
        if not isinstance(self.schedule, Callable):
            raise ValueError("schedule must be callable")


class InstrumentType(StrEnum):
    EQUITY_US = "equity_us"
    FUTURES_CME = "futures_cme"
    CRYPTO = "crypto"
    FX = "fx"


SESSIONS = {
    InstrumentType.EQUITY_US: Session("America/New_York", 570, CalendarSchedule("NYSE")),
    InstrumentType.FUTURES_CME: Session("America/New_York", 1080, CalendarSchedule("CME_Equity")),
    InstrumentType.CRYPTO: Session("UTC", 0, _crypto_schedule),
}


def session_for(instrument_type: InstrumentType | str) -> Session:
    try:
        return SESSIONS[InstrumentType(instrument_type)]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"no scheduled preset for {instrument_type}; supply an explicit Session") from exc


def segments(session: Session, start: datetime, end: datetime) -> pl.DataFrame:
    result = session.schedule(start=start, end=end)
    if not isinstance(result, pl.DataFrame) or result.schema != SCHEDULE_SCHEMA:
        raise ValueError("schedule must return the exact UTC microsecond schedule schema")
    if result.null_count().select(pl.sum_horizontal(pl.all())).item():
        raise ValueError("schedule boundaries must not be null")
    rows = result.rows(named=True)
    if rows != sorted(rows, key=lambda row: (row["session_open"], row["segment_open"])):
        raise ValueError("schedule must be sorted")
    prior = None
    for row in rows:
        so, sc, left, right = (row[n] for n in SCHEDULE_SCHEMA)
        if not so <= left < right <= sc:
            raise ValueError("invalid schedule segment boundaries")
        if not left < end or not right > start:
            raise ValueError("schedule returned a non-overlapping segment")
        if prior is not None:
            if left < prior["segment_close"]:
                raise ValueError("schedule segments overlap")
            if so == prior["session_open"] and sc != prior["session_close"]:
                raise ValueError("inconsistent session boundaries")
            if so != prior["session_open"] and so < prior["session_close"]:
                raise ValueError("schedule sessions overlap")
        if anchor(session, so) > so:
            raise ValueError("anchor is later than actual session open")
        prior = row
    return result


def localize(naive: datetime, timezone: str) -> datetime:
    """Earlier fold; shift a nonexistent wall boundary forward one hour."""
    zone = ZoneInfo(timezone)
    value = naive.replace(tzinfo=zone, fold=0)
    if value.astimezone(UTC).astimezone(zone).replace(tzinfo=None) != naive:
        value = (naive + timedelta(hours=1)).replace(tzinfo=zone, fold=0)
    return value.astimezone(UTC)


def anchor(session: Session, opened: datetime) -> datetime:
    date = opened.astimezone(ZoneInfo(session.timezone)).date()
    return localize(
        datetime.combine(date, datetime.min.time()) + timedelta(minutes=session.anchor_minutes), session.timezone
    )
