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
    assert s.window(stamp)[0, "window_start"] == label


def test_membership_and_neighbor_selection():
    s = TheStrat(session=session_for("equity_us"))
    result = s.window(utc(2026, 6, 8, 17, 30), timeframes=["4h", "1d"], before=2)
    assert result.filter(pl.col("timeframe") == "4h")["window_start"].to_list() == [
        utc(2026, 6, 5, 17, 30),
        utc(2026, 6, 8, 13, 30),
        utc(2026, 6, 8, 17, 30),
    ]
    assert result.filter(pl.col("timeframe") == "4h")["position"].to_list() == [-2, -1, 0]
    forward = s.window(utc(2026, 6, 8, 15, 22), timeframes=["4h"], after=2)
    assert forward["window_start"].to_list() == [
        utc(2026, 6, 8, 13, 30),
        utc(2026, 6, 8, 17, 30),
        utc(2026, 6, 9, 13, 30),
    ]


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
    assert s.window(opened, timeframes=["4h"], after=1)["window_start"].to_list() == [
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
    [{"before": -1}, {"before": True}, {"after": 1.5}, {"timeframes": []}, {"timeframes": ["1h", "1h"]}],
)
def test_lookup_invalid(kwargs):
    with pytest.raises(ValueError):
        TheStrat(session=session_for("crypto")).window(utc(2026, 1, 1), **kwargs)


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
    assert s.window(opened, after=6)["window_start"].to_list() == expected


def test_four_hour_nonexistent_boundary_shifts_forward_one_hour():
    # An explicitly supplied overnight schedule spans the spring DST transition.
    opened, closed = utc(2025, 3, 8, 23), utc(2025, 3, 9, 21)
    session = Session("America/New_York", 1080, custom_schedule(opened, closed))
    s = TheStrat(session=session, timeframes=["4h"], bars=bars([utc(2025, 3, 9, 7, 30), utc(2025, 3, 9, 9, 30)]))
    assert s.aggregate()["timestamp"].to_list() == [utc(2025, 3, 9, 7)]
    assert s.window(utc(2025, 3, 9, 7, 30))["window_start"].to_list() == [utc(2025, 3, 9, 7)]


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
    assert s.window(utc(2025, 11, 2, 6, 30))["window_start"].to_list() == [utc(2025, 11, 2, 5, 30)]


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
    assert s.window(stamp)["window_start"].to_list() == [stamp]


@pytest.mark.parametrize("stamp", [utc(1900, 1, 1), utc(2100, 12, 31, 23, 59)])
def test_first_and_last_supported_minutes_aggregate_and_lookup(stamp):
    s = TheStrat(session=session_for("crypto"), timeframes=["1min"], bars=bars([stamp]))
    assert s.aggregate()["timestamp"].to_list() == [stamp]
    assert s.window(stamp)["window_start"].to_list() == [stamp]


@pytest.mark.parametrize("stamp", [utc(1899, 12, 31, 23, 59), utc(2101, 1, 1)])
def test_lookup_beyond_preset_range_raises(stamp):
    from thestrat import ScheduleRangeError

    with pytest.raises(ScheduleRangeError):
        TheStrat(session=session_for("crypto"), timeframes=["1h"]).window(stamp)


def test_one_minute_preserves_physical_source_identity_across_fall_back():
    opened, closed = utc(2025, 11, 2, 4), utc(2025, 11, 2, 9)
    stamps = [utc(2025, 11, 2, 5, 30), utc(2025, 11, 2, 6, 30)]
    s = TheStrat(
        session=Session("America/New_York", 0, custom_schedule(opened, closed)), timeframes=["1min"], bars=bars(stamps)
    )
    assert s.aggregate()["timestamp"].to_list() == stamps
    for stamp in stamps:
        assert s.window(stamp)["window_start"].to_list() == [stamp]


def test_year_geometry_and_neighbors_have_full_period_segments():
    s = TheStrat(session=session_for("equity_us"), timeframes=["1y"])
    result = s.window(utc(2026, 6, 8, 15), before=1, after=1)
    assert result["position"].to_list() == [-1, 0, 1]
    assert result["window_start"].to_list() == [
        utc(2025, 1, 1, 14, 30),
        utc(2026, 1, 1, 14, 30),
        utc(2027, 1, 1, 14, 30),
    ]
    current = result.filter(pl.col("position") == 0).row(0, named=True)
    assert current["segments"][0] == {"start": utc(2026, 1, 2, 14, 30), "end": utc(2026, 1, 2, 21)}
    assert current["segments"][-1] == {"start": utc(2026, 12, 31, 14, 30), "end": utc(2026, 12, 31, 21)}


def test_early_close_geometry_and_weekend_no_containing_window():
    s = TheStrat(session=session_for("equity_us"), timeframes=["4h"])
    current = s.window(utc(2026, 11, 27, 17, 59)).row(0, named=True)
    assert current["position"] == 0
    assert current["window_start"] == utc(2026, 11, 27, 14, 30)
    assert current["window_end"] == utc(2026, 11, 27, 18)
    assert current["segments"] == [{"start": utc(2026, 11, 27, 14, 30), "end": utc(2026, 11, 27, 18)}]
    assert s.window(utc(2026, 11, 28, 12)).is_empty()
    neighbors = s.window(utc(2026, 11, 28, 12), before=1, after=1)
    assert neighbors["position"].to_list() == [-1, 1]
    assert neighbors["window_start"].to_list() == [utc(2026, 11, 27, 14, 30), utc(2026, 11, 30, 14, 30)]


def test_break_membership_uses_segments_and_excludes_spanning_window_from_neighbors():
    opened, closed = utc(2026, 6, 8), utc(2026, 6, 8, 12)
    s = TheStrat(
        session=Session(
            "UTC", 0, custom_schedule(opened, closed, (opened + timedelta(hours=5), opened + timedelta(hours=6)))
        ),
        timeframes=["4h"],
    )
    current = s.window(opened + timedelta(hours=6, minutes=30)).row(0, named=True)
    assert current["window_start"] == opened + timedelta(hours=4)
    assert current["segments"] == [
        {"start": opened + timedelta(hours=4), "end": opened + timedelta(hours=5)},
        {"start": opened + timedelta(hours=6), "end": opened + timedelta(hours=8)},
    ]
    result = s.window(opened + timedelta(hours=5, minutes=30), before=1, after=1)
    assert result["position"].to_list() == [-1, 1]
    assert result["window_start"].to_list() == [opened, opened + timedelta(hours=8)]


def test_batch_preserves_input_order_duplicates_and_shared_geometry(monkeypatch):
    import thestrat.windows as module

    actual = module.windows
    generated = []

    def counted(*args, **kwargs):
        result = actual(*args, **kwargs)
        generated.append({key: len(value) for key, value in result.items()})
        return result

    monkeypatch.setattr(module, "windows", counted)
    stamps = [utc(2026, 6, 8, 15, 1), utc(2026, 6, 8, 15), utc(2026, 6, 8, 15, 1)]
    result = TheStrat(session=session_for("equity_us")).window(stamps)
    assert result.height == 39
    assert result["input_timestamp"].to_list() == [stamp for stamp in stamps for _ in range(13)]
    assert len(generated) == 13
    assert sum(part.get("1min", 0) for part in generated) < 3000


def test_second_fold_membership_and_neighbor_positions_use_canonical_labels():
    opened, closed = utc(2025, 11, 2, 4), utc(2025, 11, 2, 9)
    s = TheStrat(session=Session("America/New_York", 0, custom_schedule(opened, closed)), timeframes=["5min"])
    result = s.window(utc(2025, 11, 2, 6, 32), before=1, after=1)
    assert result["position"].to_list() == [-1, 0, 1]
    assert result["window_start"].to_list() == [
        utc(2025, 11, 2, 5, 25),
        utc(2025, 11, 2, 5, 30),
        utc(2025, 11, 2, 5, 35),
    ]
    assert result[1, "segments"].to_list() == [
        {"start": utc(2025, 11, 2, 5, 30), "end": utc(2025, 11, 2, 5, 35)},
        {"start": utc(2025, 11, 2, 6, 30), "end": utc(2025, 11, 2, 6, 35)},
    ]


@pytest.mark.parametrize(
    "tf,first,last",
    [
        ("1w", utc(2026, 6, 8, 13, 30), utc(2026, 6, 12, 20)),
        ("1m", utc(2026, 6, 1, 13, 30), utc(2026, 6, 30, 20)),
        ("1q", utc(2026, 4, 1, 13, 30), utc(2026, 6, 30, 20)),
    ],
)
def test_calendar_geometry_extends_to_full_period(tf, first, last):
    row = TheStrat(session=session_for("equity_us"), timeframes=[tf]).window(utc(2026, 6, 8, 15)).row(0, named=True)
    assert row["segments"][0]["start"] == first
    assert row["segments"][-1]["end"] == last
    assert row["window_end"] == last


def test_futures_week_keeps_sunday_opened_session_in_existing_civil_week():
    row = TheStrat(session=session_for("futures_cme"), timeframes=["1w"]).window(utc(2026, 6, 9, 1)).row(0, named=True)
    assert row["window_start"] == utc(2026, 6, 8, 22)
    assert row["segments"][0]["start"] == utc(2026, 6, 8, 22)
    assert row["segments"][-1]["end"] == utc(2026, 6, 15, 21)


@pytest.mark.parametrize("tf", ["1h", "1y"])
def test_finite_custom_schedule_missing_neighbor_raises_clear_range_error(tf):
    from thestrat import ScheduleRangeError

    opened, closed = utc(2026, 6, 8), utc(2026, 6, 8, 1)
    s = TheStrat(session=Session("UTC", 0, custom_schedule(opened, closed)), timeframes=[tf])
    with pytest.raises(ScheduleRangeError, match="neighbors"):
        s.window(opened, after=1)


@pytest.mark.parametrize("stamp", [None, "2026-06-08", [utc(2026, 6, 8), None]])
def test_window_rejects_invalid_timestamp_input(stamp):
    with pytest.raises(ValueError, match="aware datetime"):
        TheStrat(session=session_for("crypto")).window(stamp)


def test_window_empty_batch_schema_and_hard_cut():
    s = TheStrat(session=session_for("crypto"))
    assert not hasattr(s, "window_starts")
    result = s.window([])
    assert result.is_empty()
    assert result.columns == ["input_timestamp", "timeframe", "position", "window_start", "window_end", "segments"]
    assert result.schema["segments"] == pl.List(
        pl.Struct({"start": pl.Datetime("us", "UTC"), "end": pl.Datetime("us", "UTC")})
    )


def test_geometry_queries_do_not_change_working_state():
    s = TheStrat(session=session_for("equity_us"), bars=bars([utc(2026, 6, 8, 15)]))
    before = s.bars
    s.window(utc(2026, 6, 8, 15))
    assert s.bars is before


@pytest.mark.parametrize("tf", ["1d", "1w", "1m", "1q", "1y"])
def test_multiday_session_membership_matches_aggregation_and_batch(tf):
    opened, closed = utc(2026, 6, 1), utc(2027, 2, 5)
    stamp = utc(2027, 2, 4, 12)
    s = TheStrat(session=Session("UTC", 0, custom_schedule(opened, closed)), timeframes=[tf], bars=bars([stamp]))
    aggregate_label = s.aggregate()[0, "timestamp"]
    single = s.window(stamp)
    assert single.height == 1
    row = single.row(0, named=True)
    assert row["position"] == 0
    assert row["window_start"] == aggregate_label
    assert row["window_end"] == closed
    assert row["segments"] == [{"start": opened, "end": closed}]
    batch = s.window([opened, stamp]).filter(pl.col("input_timestamp") == stamp)
    assert batch.equals(single)


def test_calendar_membership_loads_other_sessions_from_containing_label_period():
    first_open, first_close = utc(2026, 6, 1), utc(2026, 6, 2)
    long_open, long_close = utc(2026, 6, 4), utc(2026, 8, 5)

    def schedule(*, start, end):
        rows = [(a, b, a, b) for a, b in [(first_open, first_close), (long_open, long_close)] if a < end and b > start]
        return pl.DataFrame(
            rows,
            schema={
                name: pl.Datetime("us", "UTC")
                for name in ["session_open", "session_close", "segment_open", "segment_close"]
            },
            orient="row",
        )

    s = TheStrat(session=Session("UTC", 0, schedule), timeframes=["1m"])
    row = s.window(utc(2026, 8, 4, 12)).row(0, named=True)
    assert row["window_start"] == utc(2026, 6, 1)
    assert row["segments"] == [{"start": first_open, "end": first_close}, {"start": long_open, "end": long_close}]
