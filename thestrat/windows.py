"""Shared eligible window boundaries for aggregation and lookup."""

from bisect import bisect_right
from collections.abc import Sequence
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
    session: Session,
    start: datetime,
    end: datetime,
    timeframes: tuple[str, ...],
    *,
    schedule: pl.DataFrame | None = None,
) -> dict[str, list[tuple[datetime, datetime, datetime]]]:
    """Each tuple is (label, eligible segment start, eligible segment end)."""
    schedule = segments(session, start, end) if schedule is None else schedule
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


_GEOMETRY_SCHEMA = {
    "input_timestamp": pl.Datetime("us", "UTC"),
    "timeframe": pl.String,
    "position": pl.Int64,
    "window_start": pl.Datetime("us", "UTC"),
    "window_end": pl.Datetime("us", "UTC"),
    "segments": pl.List(pl.Struct({"start": pl.Datetime("us", "UTC"), "end": pl.Datetime("us", "UTC")})),
}


def _shift(value: datetime, days: int) -> datetime:
    try:
        return value + timedelta(days=days)
    except OverflowError:
        return datetime.min.replace(tzinfo=UTC) if days < 0 else datetime.max.replace(tzinfo=UTC)


def _query_bounds(session: Session, left: datetime, right: datetime, tf: str) -> tuple[datetime, datetime]:
    """Fetch whole local calendar periods; short targets need only nearby sessions."""
    if tf in ("1w", "1m", "1q", "1y"):
        try:
            left = _period_start(session, left, tf)
        except (OverflowError, ValueError):
            left = datetime.min.replace(tzinfo=UTC)
        try:
            last = _period_start(session, right, tf).astimezone(ZoneInfo(session.timezone)).replace(tzinfo=None)
            if tf == "1w":
                last += timedelta(days=7)
            elif tf == "1y":
                last = last.replace(year=last.year + 1)
            else:
                months = 3 if tf == "1q" else 1
                month = last.month - 1 + months
                last = last.replace(year=last.year + month // 12, month=month % 12 + 1)
            right = localize(last, session.timezone)
        except (OverflowError, ValueError):
            right = datetime.max.replace(tzinfo=UTC)
    else:
        left = left.replace(hour=0, minute=0, second=0, microsecond=0)
        right = _shift(right.replace(hour=0, minute=0, second=0, microsecond=0), 1)
    return left, right


def _canonical(
    intervals: list[tuple[datetime, datetime, datetime]],
) -> list[tuple[datetime, list[tuple[datetime, datetime]]]]:
    grouped: dict[datetime, list[tuple[datetime, datetime]]] = {}
    for label, left, right in intervals:
        grouped.setdefault(label, []).append((left, right))
    result = []
    for label, pieces in sorted(grouped.items()):
        merged = []
        for left, right in sorted(set(pieces)):
            if merged and left <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(right, merged[-1][1]))
            else:
                merged.append((left, right))
        result.append((label, merged))
    return result


def geometry(
    session: Session, timestamp: datetime | Sequence[datetime], timeframes: tuple[str, ...], before: int, after: int
) -> pl.DataFrame:
    inputs = (
        [timestamp]
        if isinstance(timestamp, datetime)
        else list(timestamp)
        if isinstance(timestamp, Sequence) and not isinstance(timestamp, (str, bytes))
        else None
    )
    if inputs is None or any(
        not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None for value in inputs
    ):
        raise ValueError("window timestamp must be an aware datetime or sequence of aware datetimes")
    inputs = [value.astimezone(UTC) for value in inputs]
    if not inputs:
        return pl.DataFrame(schema=_GEOMETRY_SCHEMA)
    preset = isinstance(session.schedule, CalendarSchedule) or session.schedule is _crypto_schedule
    lower = datetime(1900, 1, 1, tzinfo=UTC) if preset else datetime.min.replace(tzinfo=UTC)
    upper = datetime(2101, 1, 1, tzinfo=UTC) if preset else datetime.max.replace(tzinfo=UTC)
    if any(not lower <= value < upper for value in inputs):
        raise ScheduleRangeError("window input is outside the supported schedule range")
    span = 2
    pending = {(value, tf) for value in inputs for tf in timeframes}
    results = {}
    schedule_cache = {}
    geometry_cache = {}
    while True:
        requests = {}
        for value, tf in pending:
            left, right = _query_bounds(
                session,
                max(_shift(value, -span), lower),
                min(_shift(value, span), upper - timedelta(microseconds=1)),
                tf,
            )
            requests[value, tf] = (max(left, lower), min(right, upper))
        fetches = []
        for left, right in sorted(set(requests.values())):
            if fetches and left <= fetches[-1][1]:
                fetches[-1] = (fetches[-1][0], max(fetches[-1][1], right))
            else:
                fetches.append((left, right))
        schedules = []
        for left, right in fetches:
            fetch_left, fetch_right = max(_shift(left, -2), lower), min(_shift(right, 2), upper)
            if (fetch_left, fetch_right) in schedule_cache:
                schedules.append(schedule_cache[fetch_left, fetch_right])
                continue
            fetched = segments(session, fetch_left, fetch_right)
            if not fetched.is_empty():
                full_left = min(row[0] for row in fetched.select("session_open").iter_rows())
                full_right = max(row[0] for row in fetched.select("session_close").iter_rows())
                if full_left < max(_shift(left, -2), lower) or full_right > min(_shift(right, 2), upper):
                    fetched = segments(session, max(full_left, lower), min(full_right, upper))
            schedule_cache[fetch_left, fetch_right] = fetched
            schedules.append(fetched)
        schedule = pl.concat(schedules).unique().sort("session_open", "segment_open")
        # Build each overlapping timeframe/range once, even for many input minutes.
        groups = {}
        for tf in timeframes:
            merged = []
            for left, right in sorted({bounds for (value, target), bounds in requests.items() if target == tf}):
                if merged and left <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], right))
                else:
                    merged.append((left, right))
            groups[tf] = merged
        cache = {}
        for tf, ranges in groups.items():
            for left, right in ranges:
                if (tf, left, right) in geometry_cache:
                    cache[tf, left, right] = geometry_cache[tf, left, right]
                    continue
                subset = schedule.filter((pl.col("session_close") > left) & (pl.col("session_open") < right))
                intervals = windows(session, left, right, (tf,), schedule=subset)[tf]
                candidates = _canonical(intervals)
                physical = sorted((a, b, i) for i, (_, pieces) in enumerate(candidates) for a, b in pieces)
                cache[tf, left, right] = candidates, [a for a, _, _ in physical], physical
                geometry_cache[tf, left, right] = cache[tf, left, right]
        unresolved = set()
        for (value, tf), (request_left, request_right) in requests.items():
            left, right = next((a, b) for a, b in groups[tf] if a <= request_left and b >= request_right)
            candidates, starts, physical = cache[tf, left, right]
            index = bisect_right(starts, value) - 1
            containing = physical[index][2] if index >= 0 and value < physical[index][1] else None
            if containing is not None:
                prior = candidates[max(0, containing - before) : containing]
                following = candidates[containing + 1 : containing + 1 + after]
                chosen = [(0, candidates[containing])]
            else:
                prior = (
                    [candidate for candidate in candidates if candidate[1][-1][1] <= value][-before:] if before else []
                )
                following = [candidate for candidate in candidates if candidate[1][0][0] > value][:after]
                chosen = []
            if len(prior) != before or len(following) != after:
                unresolved.add((value, tf))
            chosen += [(i - len(prior), candidate) for i, candidate in enumerate(prior)]
            chosen += [(i + 1, candidate) for i, candidate in enumerate(following)]
            # A long custom session may belong to a period preceding the search
            # range. Expand until that label's entire civil period was fetched,
            # including earlier sessions contributing to the same canonical bar.
            if tf in ("1w", "1m", "1q", "1y") and any(not left <= candidate[0] < right for _, candidate in chosen):
                unresolved.add((value, tf))
            results[value, tf] = sorted(chosen, key=lambda pair: pair[1][0])
        if not unresolved:
            rows = []
            for value in inputs:
                for tf in timeframes:
                    for position, (label, pieces) in results[value, tf]:
                        rows.append(
                            (value, tf, position, label, pieces[-1][1], [{"start": a, "end": b} for a, b in pieces])
                        )
            return pl.DataFrame(rows, schema=_GEOMETRY_SCHEMA, orient="row")
        if all(left == lower and right == upper for left, right in requests.values()):
            raise ScheduleRangeError("requested neighbors exceed the supported schedule range")
        pending = unresolved
        span *= 2
