"""Supported mixed-source calculation uses explicit coverage, never trade density."""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from thestrat import Session, SourceCoverage, TheStrat, session_for
from thestrat.frames import KEY, SCHEMA
from thestrat.sessions import SCHEDULE_SCHEMA


def declared(rows, intervals):
    return SourceCoverage(
        pl.DataFrame(
            intervals,
            schema={
                "symbol": pl.String,
                "timeframe": pl.String,
                "start": pl.Datetime("us", "UTC"),
                "end": pl.Datetime("us", "UTC"),
            },
            orient="row",
        ),
        rows.select(KEY),
    )


def worked():
    start = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)
    rows = pl.DataFrame(
        [
            ("A", start, "5min", 10.0, 30.0, 1.0, 20.0, 203.0),
            ("A", start + timedelta(minutes=5), "1min", 20.0, 25.0, 5.0, 22.0, 10.0),
            ("A", start + timedelta(minutes=6), "1min", 22.0, 26.0, 6.0, 23.0, 20.0),
        ],
        schema=SCHEMA,
        orient="row",
    )
    coverage = declared(
        rows,
        [
            ("A", "5min", start, start + timedelta(minutes=5)),
            ("A", "1min", start + timedelta(minutes=5), start + timedelta(minutes=7)),
        ],
    )
    return start, rows, coverage


def test_closed_summary_plus_minutes_snapshot_retractions_and_source_only_state():
    start, rows, coverage = worked()
    strat = TheStrat(session=session_for("equity_us"), timeframes=["1h", "5min"], bars=rows)
    output = strat.aggregate(coverage=coverage, through=start + timedelta(minutes=6))
    assert output["volume"].to_list() == [233.0, 30.0]
    assert output["open"].to_list() == [10.0, 20.0]
    assert output["close"].to_list() == [23.0, 23.0]
    assert strat.bars.height == 3
    update = rows.tail(1).with_columns(
        pl.lit(24.0).alias("high"),
        pl.lit(21.0).alias("low"),
        pl.lit(22.0).alias("close"),
        pl.lit(296.0).alias("volume"),
    )
    revised = strat.aggregate(update, coverage=coverage, through=start + timedelta(minutes=6))
    assert revised["volume"].to_list() == [509.0, 306.0]
    assert revised["high"].to_list() == [30.0, 25.0]
    assert revised["low"].to_list() == [1.0, 5.0]
    assert_frame_equal(strat.aggregate(update, coverage=coverage, through=start + timedelta(minutes=6)), revised)
    assert coverage.expected_keys.height == 3


def test_declared_empty_minute_and_entire_current_window_are_not_fabricated():
    start, rows, coverage = worked()
    rows = rows.head(2)
    coverage = SourceCoverage(coverage.intervals, rows.select(KEY))
    strat = TheStrat(session=session_for("equity_us"), timeframes=["1min", "5min", "1h"], bars=rows)
    result = strat.aggregate(coverage=coverage, through=start + timedelta(minutes=6))
    assert result["timeframe"].to_list() == ["5min", "1h"]
    assert result["volume"].to_list() == [10.0, 213.0]
    empty = rows.clear()
    result = TheStrat(session=session_for("equity_us"), timeframes=["1h"], bars=empty).aggregate(
        coverage=SourceCoverage(coverage.intervals, empty.select(KEY)), through=start + timedelta(minutes=6)
    )
    assert result.is_empty()
    assert result.schema == SCHEMA


def test_missing_extra_and_new_source_keys_are_explicit_and_atomic():
    start, rows, coverage = worked()
    through = start + timedelta(minutes=6)
    strat = TheStrat(session=session_for("equity_us"), timeframes=["1h"], bars=rows.head(2))
    before = strat.bars.clone()
    with pytest.raises(ValueError, match="missing expected"):
        strat.aggregate(coverage=coverage, through=through)
    assert_frame_equal(strat.bars, before)
    strat.bars = rows
    missing_key = SourceCoverage(coverage.intervals, rows.head(2).select(KEY))
    with pytest.raises(ValueError, match="not explicitly listed"):
        strat.aggregate(coverage=missing_key, through=through)
    strat.bars = rows.head(2)
    assert strat.aggregate(rows.tail(1), coverage=coverage, through=through)["volume"].item() == 233.0


@pytest.mark.parametrize("kind", ["duplicate_key", "duplicate_interval", "overlap_interval", "cut_window", "future"])
def test_invalid_coverage_does_not_mutate(kind):
    start, rows, coverage = worked()
    intervals, keys = coverage.intervals, coverage.expected_keys
    if kind == "duplicate_key":
        keys = pl.concat([keys, keys.head(1)])
    elif kind == "duplicate_interval":
        intervals = pl.concat([intervals, intervals.head(1)])
    elif kind == "overlap_interval":
        intervals = pl.concat([intervals, intervals.tail(1).with_columns(pl.col("start") + timedelta(minutes=1))])
    elif kind == "cut_window":
        intervals = intervals.with_columns(
            pl.when(pl.col("timeframe") == "5min")
            .then(pl.col("start") + timedelta(minutes=1))
            .otherwise(pl.col("start"))
            .alias("start")
        )
    else:
        intervals = intervals.with_columns(pl.col("end") + timedelta(hours=1))
    strat = TheStrat(session=session_for("equity_us"), timeframes=["1h"], bars=rows)
    with pytest.raises(ValueError):
        strat.aggregate(coverage=SourceCoverage(intervals, keys), through=start + timedelta(minutes=6))
    assert_frame_equal(strat.bars, rows)


def test_snapshot_cannot_leave_overlapping_summary_stale():
    start, rows, coverage = worked()
    old_minute = rows.tail(1).with_columns(pl.lit(start).alias("timestamp"))
    rows = pl.concat([rows, old_minute])
    coverage = declared(
        rows,
        [
            ("A", "5min", start, start + timedelta(minutes=5)),
            ("A", "1min", start, start + timedelta(minutes=7)),
        ],
    )
    strat = TheStrat(session=session_for("equity_us"), timeframes=["1h"], bars=rows)
    with pytest.raises(ValueError, match="overlaps a retained summary"):
        strat.aggregate(
            old_minute.with_columns(pl.lit(999.0).alias("volume")),
            coverage=coverage,
            through=start + timedelta(minutes=6),
        )
    assert_frame_equal(strat.bars, rows)


def test_final_future_summary_rejected_even_when_label_precedes_through():
    start, rows, _ = worked()
    hour = rows.head(1).with_columns(pl.lit("1h").alias("timeframe"))
    coverage = declared(hour, [("A", "1h", start, start + timedelta(minutes=7))])
    with pytest.raises(ValueError, match="cuts a source window"):
        TheStrat(session=session_for("equity_us"), timeframes=["1h"], bars=hour).aggregate(
            coverage=coverage, through=start + timedelta(minutes=6)
        )


@pytest.mark.parametrize(
    "preset,anchor,start",
    [
        ("equity_us", 570, datetime(2026, 3, 9, 13, 30, tzinfo=UTC)),
        ("equity_us", 540, datetime(2026, 3, 9, 13, 30, tzinfo=UTC)),
        ("crypto", 0, datetime(2026, 11, 1, 0, 0, tzinfo=UTC)),
        ("futures_cme", 1080, datetime(2026, 3, 8, 22, 0, tzinfo=UTC)),
    ],
)
def test_session_alignment_and_overnight_parity(preset, anchor, start):
    session = Session(session_for(preset).timezone, anchor, session_for(preset).schedule)
    timestamps = [start + timedelta(minutes=i) for i in range(67)]
    source = pl.DataFrame(
        [
            (None, t, "1min", float(i), float(i + 2), float(i - 1), float(i + 1), float(i + 1))
            for i, t in enumerate(timestamps)
        ],
        schema=SCHEMA,
        orient="row",
    )
    prefix = (
        TheStrat(session=session, timeframes=["5min"], bars=source.head(65))
        .aggregate()
        .filter(pl.col("timeframe") == "5min")
    )
    mixed = pl.concat([prefix, source.tail(2)])
    coverage = declared(
        mixed,
        [
            (None, "5min", start, start + timedelta(minutes=65)),
            (None, "1min", start + timedelta(minutes=65), start + timedelta(minutes=67)),
        ],
    )
    timeframes = ["1min", "5min", "15min", "30min", "1h", "4h", "6h", "12h", "1d"]
    # Session-open-to-watermark completeness is sufficient for these daily scopes.
    expected_strat = TheStrat(session=session, timeframes=timeframes, bars=source)
    keys = expected_strat.window(timestamps[-1]).select("timeframe", pl.col("window_start").alias("timestamp"))
    expected = expected_strat.aggregate().join(keys, on=["timeframe", "timestamp"], how="semi")
    actual = TheStrat(session=session, timeframes=timeframes, bars=mixed).aggregate(
        coverage=coverage, through=timestamps[-1]
    )
    assert_frame_equal(actual, expected)


def test_custom_break_coverage_and_early_close():
    start = datetime(2026, 4, 7, 9, tzinfo=UTC)
    close = start + timedelta(hours=3)

    def schedule(*, start: datetime, end: datetime):
        del start, end
        return pl.DataFrame(
            [(base, close, base, base + timedelta(hours=1)), (base, close, base + timedelta(hours=2), close)],
            schema=SCHEDULE_SCHEMA,
            orient="row",
        )

    base = start
    session = Session("UTC", 540, schedule)
    rows = pl.DataFrame(
        [
            ("A", start, "1h", 10.0, 12.0, 9.0, 11.0, 20.0),
            ("A", start + timedelta(hours=2), "1min", 11.0, 13.0, 10.0, 12.0, 30.0),
        ],
        schema=SCHEMA,
        orient="row",
    )
    coverage = declared(
        rows,
        [
            ("A", "1h", start, start + timedelta(hours=1)),
            ("A", "1min", start + timedelta(hours=2), start + timedelta(hours=2, minutes=1)),
        ],
    )
    output = TheStrat(session=session, timeframes=["1d"], bars=rows).aggregate(
        coverage=coverage, through=start + timedelta(hours=2)
    )
    assert output["volume"].item() == 50.0


def test_closed_multisegment_daily_bar_contributes_volume_once():
    opened = datetime(2026, 4, 7, 9, tzinfo=UTC)
    next_open = opened + timedelta(days=1)
    close = opened + timedelta(hours=3)
    rows_schedule = [
        (opened, close, opened, opened + timedelta(hours=1)),
        (opened, close, opened + timedelta(hours=2), close),
        (next_open, next_open + timedelta(hours=3), next_open, next_open + timedelta(hours=3)),
    ]

    def schedule(*, start, end):
        return pl.DataFrame(
            [row for row in rows_schedule if row[2] < end and row[3] > start], schema=SCHEDULE_SCHEMA, orient="row"
        )

    rows = pl.DataFrame(
        [
            ("A", opened, "1d", 10.0, 15.0, 8.0, 14.0, 200.0),
            ("A", next_open, "1min", 14.0, 16.0, 12.0, 15.0, 30.0),
        ],
        schema=SCHEMA,
        orient="row",
    )
    coverage = declared(rows, [("A", "1d", opened, close), ("A", "1min", next_open, next_open + timedelta(minutes=1))])
    result = TheStrat(session=Session("UTC", 540, schedule), timeframes=["1w"], bars=rows).aggregate(
        coverage=coverage, through=next_open
    )
    assert result["volume"].item() == 230.0
    assert result.select("open", "high", "low", "close").row(0) == (10.0, 16.0, 8.0, 15.0)


def test_holiday_quarter_label_and_cross_month_week_use_disjoint_covers():
    quarter = datetime(2026, 1, 1, 14, 30, tzinfo=UTC)  # Holiday label, not an eligible minute.
    monday = datetime(2026, 3, 30, 13, 30, tzinfo=UTC)
    tuesday = monday + timedelta(days=1)
    through = datetime(2026, 4, 1, 13, 30, tzinfo=UTC)
    rows = pl.DataFrame(
        [
            ("A", quarter, "1q", 10.0, 20.0, 5.0, 18.0, 100.0),
            ("A", monday, "1d", 15.0, 19.0, 14.0, 18.0, 30.0),
            ("A", tuesday, "1d", 18.0, 20.0, 17.0, 18.0, 20.0),
            ("A", through, "1min", 18.0, 21.0, 17.0, 20.0, 10.0),
        ],
        schema=SCHEMA,
        orient="row",
    )
    coverage = declared(
        rows,
        [
            ("A", "1q", quarter, through),
            ("A", "1d", monday, through),
            ("A", "1min", through, through + timedelta(minutes=1)),
        ],
    )
    result = TheStrat(session=session_for("equity_us"), timeframes=["1w", "1m", "1y"], bars=rows).aggregate(
        coverage=coverage, through=through
    )
    assert result["volume"].to_list() == [60.0, 10.0, 110.0]
    assert result["open"].to_list() == [15.0, 18.0, 10.0]
    assert result["close"].to_list() == [20.0, 20.0, 20.0]


def test_equity_early_close_declares_actual_end():
    opened = datetime(2026, 11, 27, 14, 30, tzinfo=UTC)
    closed = datetime(2026, 11, 27, 18, 0, tzinfo=UTC)
    through = datetime(2026, 11, 30, 14, 30, tzinfo=UTC)
    rows = pl.DataFrame(
        [
            ("A", opened, "1d", 10.0, 15.0, 8.0, 14.0, 200.0),
            ("A", through, "1min", 14.0, 16.0, 12.0, 15.0, 30.0),
        ],
        schema=SCHEMA,
        orient="row",
    )
    # Earlier November is explicitly complete and empty in this synthetic fixture.
    coverage = declared(
        rows,
        [
            ("A", "1d", datetime(2026, 11, 1, tzinfo=UTC), closed),
            ("A", "1min", through, through + timedelta(minutes=1)),
        ],
    )
    result = TheStrat(session=session_for("equity_us"), timeframes=["1m"], bars=rows).aggregate(
        coverage=coverage, through=through
    )
    assert result["volume"].item() == 230.0


def test_futures_closed_month_uses_session_open_civil_period():
    session = session_for("futures_cme")
    march = datetime(2026, 3, 1, 23, tzinfo=UTC)  # 18:00 EST anchor.
    april = datetime(2026, 4, 1, 22, tzinfo=UTC)  # 18:00 EDT anchor.
    # March's final session opens March31 and closes April1. April starts at its next open.
    rows = pl.DataFrame(
        [
            (None, march, "1m", 10.0, 20.0, 5.0, 18.0, 100.0),
            (None, april, "1min", 18.0, 21.0, 17.0, 20.0, 10.0),
        ],
        schema=SCHEMA,
        orient="row",
    )
    coverage = declared(
        rows,
        [(None, "1m", datetime(2026, 1, 1, tzinfo=UTC), april), (None, "1min", april, april + timedelta(minutes=1))],
    )
    result = TheStrat(session=session, timeframes=["1y"], bars=rows).aggregate(coverage=coverage, through=april)
    assert result["volume"].item() == 110.0


def test_two_symbols_and_null_series_remain_independent():
    start, source, coverage = worked()
    rows = pl.concat(
        [
            source,
            source.with_columns(pl.lit(None, dtype=pl.String).alias("symbol"), (pl.col("volume") * 2).alias("volume")),
        ]
    )
    intervals = pl.concat(
        [coverage.intervals, coverage.intervals.with_columns(pl.lit(None, dtype=pl.String).alias("symbol"))]
    )
    result = TheStrat(session=session_for("equity_us"), timeframes=["1h"], bars=rows).aggregate(
        coverage=SourceCoverage(intervals, rows.select(KEY)), through=start + timedelta(minutes=6)
    )
    assert dict(result.select("symbol", "volume").iter_rows()) == {None: 466.0, "A": 233.0}


def test_complete_cover_can_use_shorter_tiles_when_longest_dead_ends():
    start = datetime(2026, 1, 5, tzinfo=UTC)
    through = start + timedelta(hours=8)
    rows = pl.DataFrame(
        [
            ("A", start, "6h", 10.0, 12.0, 9.0, 11.0, 60.0),
            ("A", start, "4h", 10.0, 12.0, 9.0, 11.0, 40.0),
            ("A", start + timedelta(hours=4), "4h", 10.0, 12.0, 9.0, 11.0, 40.0),
            ("A", through, "1min", 10.0, 12.0, 9.0, 11.0, 1.0),
        ],
        schema=SCHEMA,
        orient="row",
    )
    coverage = declared(
        rows,
        [
            ("A", "6h", start, start + timedelta(hours=6)),
            ("A", "4h", start, through),
            ("A", "1min", through, through + timedelta(minutes=1)),
        ],
    )
    result = TheStrat(session=session_for("crypto"), timeframes=["12h"], bars=rows).aggregate(
        coverage=coverage, through=through
    )
    assert result["volume"].item() == 81.0


def test_cme_early_close_drops_break_pieces_after_actual_close():
    session = session_for("futures_cme")
    start = datetime(2026, 11, 26, tzinfo=UTC)
    end = datetime(2026, 11, 28, tzinfo=UTC)
    schedule = session.schedule(start=start, end=end)
    assert schedule.height > 0
    for row in schedule.iter_rows(named=True):
        assert row["session_open"] <= row["segment_open"] < row["segment_close"] <= row["session_close"]


@pytest.mark.parametrize("grain,expected", [("1h", 181.0), ("5min", None)])
def test_fall_back_merged_hours_and_explicit_interleaved_subhour_limit(grain, expected):
    opened = datetime(2026, 11, 1, 4, tzinfo=UTC)
    repeated = opened + timedelta(hours=1)
    through = opened + timedelta(hours=3)
    closed = opened + timedelta(hours=5)

    def schedule(*, start, end):
        rows = [(opened, closed, opened, closed)] if opened < end and closed > start else []
        return pl.DataFrame(rows, schema=SCHEDULE_SCHEMA, orient="row")

    session = Session("America/New_York", 0, schedule)
    if grain == "1h":
        source = [
            ("A", opened, "1h", 10.0, 12.0, 9.0, 11.0, 60.0),
            ("A", repeated, "1h", 11.0, 14.0, 10.0, 13.0, 120.0),
        ]
    else:
        source = [
            ("A", opened, "1h", 10.0, 12.0, 9.0, 11.0, 60.0),
            ("A", repeated, "5min", 11.0, 14.0, 10.0, 13.0, 10.0),
        ]
    source.append(("A", through, "1min", 13.0, 15.0, 12.0, 14.0, 1.0))
    rows = pl.DataFrame(source, schema=SCHEMA, orient="row")
    declarations = [
        ("A", "1h", opened, repeated),
        ("A", grain, repeated, through),
        ("A", "1min", through, through + timedelta(minutes=1)),
    ]
    # Adjacent same-grain ranges are explicitly allowed.
    coverage = declared(rows, declarations)
    strat = TheStrat(session=session, timeframes=["4h"], bars=rows)
    if expected is None:
        with pytest.raises(ValueError, match="non-interleaved"):
            strat.aggregate(coverage=coverage, through=through)
        assert_frame_equal(strat.bars, rows)
    else:
        assert strat.aggregate(coverage=coverage, through=through)["volume"].item() == expected


@pytest.mark.parametrize("later_only", [False, True])
def test_partial_overnight_daily_coverage_rejects_future_summary_and_double_count(later_only):
    opened = datetime(2026, 1, 5, 22, tzinfo=UTC)
    closed = opened + timedelta(hours=6)
    segments = pl.DataFrame(
        [(opened, closed, opened, opened + timedelta(hours=1)), (opened, closed, opened + timedelta(hours=3), closed)],
        schema=SCHEDULE_SCHEMA,
        orient="row",
    )

    def schedule(*, start, end):
        return segments.filter((pl.col("segment_open") < end) & (pl.col("segment_close") > start))

    rows = pl.DataFrame([("A", opened, "1d", 10.0, 12.0, 9.0, 11.0, 400.0)], schema=SCHEMA, orient="row")
    intervals = [("A", "1d", opened, opened + timedelta(hours=1))]
    through = opened + timedelta(minutes=59)
    if later_only:
        rows = pl.concat(
            [rows, pl.DataFrame([("A", opened, "1min", 10.0, 12.0, 9.0, 11.0, 100.0)], schema=SCHEMA, orient="row")]
        )
        intervals = [
            ("A", "1d", opened + timedelta(hours=3), closed),
            ("A", "1min", opened, opened + timedelta(hours=1)),
        ]
        through = closed - timedelta(minutes=1)
    owner = TheStrat(session=Session("UTC", 1320, schedule), timeframes=["1d"], bars=rows)
    before = owner.bars.clone()
    with pytest.raises(ValueError):
        owner.aggregate(coverage=declared(rows, intervals), through=through)
    assert_frame_equal(owner.bars, before)
