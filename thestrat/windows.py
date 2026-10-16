"""Shared eligible window boundaries for aggregation and lookup."""

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl

from thestrat.sessions import (
    CalendarSchedule,
    ScheduleRangeError,
    Session,
    _crypto_schedule,
    anchor,
    localize,
    segments,
)

MINUTES = {"1min": 1, "5min": 5, "15min": 15, "30min": 30, "1h": 60, "4h": 240, "6h": 360, "12h": 720}


def _period_start(session: Session, opened: datetime, timeframe: str) -> datetime:
    day = opened.astimezone(ZoneInfo(session.timezone)).replace(tzinfo=None, hour=0, minute=0, second=0, microsecond=0)
    if timeframe == "1w":
        day -= timedelta(days=day.weekday())
    elif timeframe == "1m":
        day = day.replace(day=1)
    elif timeframe == "1q":
        day = day.replace(month=1 + 3 * ((day.month - 1) // 3), day=1)
    elif timeframe == "1y":
        day = day.replace(month=1, day=1)
    return localize(day + timedelta(minutes=session.anchor_minutes), session.timezone)


def windows(
    session: Session, start: datetime, end: datetime, timeframes: tuple[str, ...]
) -> dict[str, list[tuple[datetime, datetime, datetime]]]:
    """Each tuple is (label, eligible segment start, eligible segment end)."""
    schedule = segments(session, start, end)
    result = {tf: [] for tf in timeframes}
    zone = ZoneInfo(session.timezone)
    for row in schedule.iter_rows(named=True):
        opened, closed, left, right = (
            row[n] for n in ("session_open", "session_close", "segment_open", "segment_close")
        )
        for tf in timeframes:
            if tf not in MINUTES:
                label = anchor(session, opened) if tf == "1d" else _period_start(session, opened, tf)
                result[tf].append((label, left, right))
                continue
            step = timedelta(minutes=MINUTES[tf])
            if MINUTES[tf] < 60:
                local_left = left.astimezone(zone)
                floor = local_left.replace(
                    minute=(local_left.minute // MINUTES[tf]) * MINUTES[tf], second=0, microsecond=0
                )
                physical = floor.astimezone(UTC)
                while physical < right:
                    # Source minutes retain physical snapshot keys. Higher subhour
                    # intervals merge into their matching earlier-fold wall label.
                    label = (
                        physical
                        if tf == "1min"
                        else localize(physical.astimezone(zone).replace(tzinfo=None), session.timezone)
                    )
                    result[tf].append((label, max(left, physical), min(right, physical + step)))
                    physical += step
                continue
            else:
                naive = anchor(session, opened).astimezone(zone).replace(tzinfo=None)
                local_left = left.astimezone(zone).replace(tzinfo=None)
                naive += ((local_left - naive) // step) * step
            # Wall-clock stepping merges repeated fall-back windows by choosing fold 0.
            while True:
                label = localize(naive, session.timezone)
                next_label = localize(naive + step, session.timezone)
                if label >= right:
                    break
                if next_label > left and next_label > label:
                    result[tf].append((label, max(left, label), min(right, next_label, closed)))
                naive += step
    return result


def lookup(
    session: Session, timestamp: datetime, timeframes: tuple[str, ...], count: int, direction: str, inclusive: bool
) -> pl.DataFrame:
    if timestamp.tzinfo is None:
        raise ValueError("window_starts timestamp must be timezone aware")
    timestamp = timestamp.astimezone(UTC)
    span = 4
    preset = isinstance(session.schedule, CalendarSchedule) or session.schedule is _crypto_schedule
    lower, upper = datetime(1900, 1, 1, tzinfo=UTC), datetime(2101, 1, 1, tzinfo=UTC)
    if preset and not lower <= timestamp < upper:
        raise ScheduleRangeError("calendar supports 1900 through 2100")
    previous_bounds = None
    while True:
        # Start with nearby sessions and widen only when requested starts are
        # unavailable. Calendar labels may precede their first eligible session.
        left, right = timestamp - timedelta(days=span), timestamp + timedelta(days=span)
        if preset:
            left, right = max(left, lower), min(right, upper)
        if previous_bounds == (left, right):
            raise ScheduleRangeError("requested window count exceeds supported calendar range")
        previous_bounds = (left, right)
        boundaries = windows(session, left, right, timeframes)
        rows = []
        complete = True
        for tf, intervals in boundaries.items():
            labels = sorted({label for label, _, _ in intervals})
            if direction == "backward":
                labels = [v for v in labels if v < timestamp or (inclusive and v == timestamp)][-count:]
            else:
                labels = [v for v in labels if v > timestamp or (inclusive and v == timestamp)][:count]
            complete &= len(labels) == count
            rows.extend((tf, value) for value in labels)
        if complete:
            return pl.DataFrame(
                rows, schema={"timeframe": pl.String, "timestamp": pl.Datetime("us", "UTC")}, orient="row"
            )
        span *= 2
