"""Shared boundary lookup, including custom schedules and DST walls."""

from datetime import timedelta

import polars as pl
import pytest
from test_sessions import bars, utc

from thestrat import Session, TheStrat, session_for


@pytest.mark.parametrize("anchor", [540, 570])
@pytest.mark.parametrize(
    "tf", ["1min", "5min", "15min", "30min", "1h", "4h", "6h", "12h", "1d", "1w", "1m", "1q", "1y"]
)
def test_lookup_and_aggregate_labels_agree(anchor, tf):
    regular = session_for("equity_us")
    stamp = utc(2026, 6, 8, 15, 22)
    s = TheStrat(session=Session(regular.timezone, anchor, regular.schedule), timeframes=[tf], bars=bars([stamp]))
    label = s.aggregate()[0, "timestamp"]
    assert s.window_starts(stamp)[0, "timestamp"] == label
    assert s.window_starts(label, direction="forward")[0, "timestamp"] == label


def test_direction_count_selection_inclusive():
    s = TheStrat(session=session_for("equity_us"))
    result = s.window_starts(utc(2026, 6, 8, 17, 30), timeframes=["4h", "1d"], count=2, inclusive=False)
    assert result.filter(pl.col("timeframe") == "4h")["timestamp"].to_list() == [
        utc(2026, 6, 5, 17, 30),
        utc(2026, 6, 8, 13, 30),
    ]
    forward = s.window_starts(utc(2026, 6, 8, 15, 22), timeframes=["4h"], count=2, direction="forward")
    assert forward["timestamp"].to_list() == [utc(2026, 6, 8, 17, 30), utc(2026, 6, 9, 13, 30)]


def custom_schedule(opened, closed, breaks=()):
    def schedule(*, start, end):
        intervals = [(opened, closed)] if not breaks else [(opened, breaks[0]), (breaks[1], closed)]
        rows = [(opened, closed, left, right) for left, right in intervals if left < end and right > start]
        return pl.DataFrame(
            rows,
            schema={
                n: pl.Datetime("us", "UTC") for n in ["session_open", "session_close", "segment_open", "segment_close"]
            },
            orient="row",
        )

    return schedule


def test_break_splits_eligibility_without_duplicate_label():
    opened, closed = utc(2026, 6, 8, 0), utc(2026, 6, 8, 8)
    session = Session(
        "UTC", 0, custom_schedule(opened, closed, (opened + timedelta(hours=1), opened + timedelta(hours=2)))
    )
    s = TheStrat(
        session=session,
        timeframes=["4h"],
        bars=bars([opened, opened + timedelta(hours=1), opened + timedelta(hours=2), opened + timedelta(hours=4)]),
    )
    assert s.aggregate()["volume"].to_list() == [2.0, 1.0]
    assert s.window_starts(opened, timeframes=["4h"], count=2, direction="forward")["timestamp"].to_list() == [
        opened,
        opened + timedelta(hours=4),
    ]


@pytest.mark.parametrize(
    "opened,closed,stamps,labels",
    [
        (
            utc(2025, 11, 2, 4),
            utc(2025, 11, 2, 9),
            [utc(2025, 11, 2, 5, 30), utc(2025, 11, 2, 6, 30)],
            [utc(2025, 11, 2, 5)],
        ),
        (
            utc(2025, 3, 9, 5),
            utc(2025, 3, 9, 9),
            [utc(2025, 3, 9, 6, 30), utc(2025, 3, 9, 7, 30)],
            [utc(2025, 3, 9, 6), utc(2025, 3, 9, 7)],
        ),
    ],
)
def test_dst_repeated_and_nonexistent_boundaries(opened, closed, stamps, labels):
    s = TheStrat(
        session=Session("America/New_York", 0, custom_schedule(opened, closed)), timeframes=["1h"], bars=bars(stamps)
    )
    assert s.aggregate()["timestamp"].to_list() == labels


@pytest.mark.parametrize(
    "kwargs",
    [{"count": 0}, {"count": True}, {"direction": "sideways"}, {"timeframes": []}, {"timeframes": ["1h", "1h"]}],
)
def test_lookup_invalid(kwargs):
    with pytest.raises(ValueError):
        TheStrat(session=session_for("crypto")).window_starts(utc(2026, 1, 1), **kwargs)


def test_malformed_schedule_is_rejected_atomically():
    s = TheStrat(session=Session("UTC", 0, lambda **kwargs: pl.DataFrame()), bars=bars([utc(2026, 1, 1)]))
    before = s.bars
    with pytest.raises(ValueError, match="schedule"):
        s.aggregate()
    assert s.bars is before


@pytest.mark.parametrize("date,offset", [((2025, 3, 10), 4), ((2025, 11, 3), 5)])
def test_equity_full_session_has_seven_independent_expected_hour_starts(date, offset):
    year, month, day = date
    opened = utc(year, month, day, 9 + offset, 30)
    expected = [utc(year, month, day, hour + offset, 30) for hour in range(9, 16)]
    stamps = [opened + timedelta(minutes=i) for i in range(390)]
    s = TheStrat(session=session_for("equity_us"), timeframes=["1h"], bars=bars(stamps))
    assert s.aggregate()["timestamp"].to_list() == expected
    assert s.window_starts(opened, count=7, direction="forward")["timestamp"].to_list() == expected


def test_four_hour_nonexistent_boundary_shifts_forward_one_hour():
    # An explicitly supplied overnight schedule spans the spring DST transition.
    opened, closed = utc(2025, 3, 8, 23), utc(2025, 3, 9, 21)
    session = Session("America/New_York", 1080, custom_schedule(opened, closed))
    s = TheStrat(session=session, timeframes=["4h"], bars=bars([utc(2025, 3, 9, 7, 30), utc(2025, 3, 9, 9, 30)]))
    assert s.aggregate()["timestamp"].to_list() == [utc(2025, 3, 9, 7)]
    assert s.window_starts(utc(2025, 3, 9, 7, 30))["timestamp"].to_list() == [utc(2025, 3, 9, 7)]


def test_fall_back_subhour_windows_merge_corresponding_repeated_wall_minutes():
    opened, closed = utc(2025, 11, 2, 4), utc(2025, 11, 2, 9)
    s = TheStrat(
        session=Session("America/New_York", 0, custom_schedule(opened, closed)),
        timeframes=["5min"],
        bars=bars([utc(2025, 11, 2, 5, 30), utc(2025, 11, 2, 6, 30)]),
    )
    result = s.aggregate()
    assert result["timestamp"].to_list() == [utc(2025, 11, 2, 5, 30)]
    assert result["volume"].to_list() == [2.0]
    assert s.window_starts(utc(2025, 11, 2, 6, 30))["timestamp"].to_list() == [utc(2025, 11, 2, 5, 55)]


@pytest.mark.parametrize(
    "stamp,direction",
    [
        (utc(2100, 6, 1), "backward"),
        (utc(1900, 6, 1), "forward"),
        (utc(1900, 1, 1), "forward"),
        (utc(2100, 12, 31, 23), "backward"),
    ],
)
def test_lookup_available_start_near_supported_calendar_range(stamp, direction):
    s = TheStrat(session=session_for("crypto"), timeframes=["1h"])
    assert s.window_starts(stamp, direction=direction)["timestamp"].to_list() == [stamp]


@pytest.mark.parametrize("stamp", [utc(1900, 1, 1), utc(2100, 12, 31, 23, 59)])
def test_first_and_last_supported_minutes_aggregate_and_lookup(stamp):
    s = TheStrat(session=session_for("crypto"), timeframes=["1min"], bars=bars([stamp]))
    assert s.aggregate()["timestamp"].to_list() == [stamp]
    assert s.window_starts(stamp)["timestamp"].to_list() == [stamp]


@pytest.mark.parametrize("stamp", [utc(1899, 12, 31, 23, 59), utc(2101, 1, 1)])
def test_lookup_beyond_preset_range_raises(stamp):
    from thestrat import ScheduleRangeError

    with pytest.raises(ScheduleRangeError):
        TheStrat(session=session_for("crypto"), timeframes=["1h"]).window_starts(stamp)


def test_one_minute_preserves_physical_source_identity_across_fall_back():
    opened, closed = utc(2025, 11, 2, 4), utc(2025, 11, 2, 9)
    stamps = [utc(2025, 11, 2, 5, 30), utc(2025, 11, 2, 6, 30)]
    s = TheStrat(
        session=Session("America/New_York", 0, custom_schedule(opened, closed)), timeframes=["1min"], bars=bars(stamps)
    )
    assert s.aggregate()["timestamp"].to_list() == stamps
    for stamp in stamps:
        assert s.window_starts(stamp)["timestamp"].to_list() == [stamp]
        assert s.window_starts(stamp, direction="forward")["timestamp"].to_list() == [stamp]
