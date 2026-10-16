"""Independent arithmetic and full-source comparisons for the isolated experiment."""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from experiments.mixed_fixture import archive_for, source_year
from experiments.mixed_timeframes import CoverageError, Geometry, MixedCurrent, prepare_seed, replace
from thestrat import TheStrat, session_for
from thestrat.frames import KEY, SCHEMA


def make_engine(source, through):
    geometry = Geometry()
    return MixedCurrent(
        prepare_seed(
            archive_for(source), through, geometry, complete=True, coverage_start=datetime(2026, 1, 1, tzinfo=UTC)
        ),
        geometry,
    )


def current_oracle(source, through):
    strat = TheStrat(session=session_for("equity_us"), bars=source)
    bars = strat.aggregate()
    keys = strat.window(through).select("timeframe", pl.col("window_start").alias("timestamp"))
    return bars.join(keys, on=["timeframe", "timestamp"], how="semi").sort(KEY)


@pytest.fixture(scope="module")
def year():
    return source_year()


def test_manual_snapshot_retraction_open_close_volume_and_sparse_coverage():
    t = datetime(2026, 4, 7, 13, 30, tzinfo=UTC)
    # No trade at minute 1 is valid; completeness is explicitly declared.
    source = pl.DataFrame(
        [
            ("S0", t, "1min", 10.0, 30.0, 1.0, 20.0, 203.0),
            ("S0", t + timedelta(minutes=2), "1min", 20.0, 25.0, 5.0, 22.0, 10.0),
        ],
        schema=SCHEMA,
        orient="row",
    )
    engine = make_engine(source, t + timedelta(minutes=2))
    corrected = source.head(1).with_columns(
        pl.lit(12.0).alias("open"),
        pl.lit(23.0).alias("high"),
        pl.lit(8.0).alias("low"),
        pl.lit(21.0).alias("close"),
        pl.lit(296.0).alias("volume"),
    )
    output = engine.calculate(corrected)
    for row in output.filter(pl.col("timeframe") != "1min").iter_rows(named=True):
        assert [row[x] for x in ["open", "high", "low", "close", "volume"]] == [12.0, 25.0, 5.0, 22.0, 306.0]
    assert_frame_equal(engine.calculate(corrected), output)
    assert engine.bars.height == 2


def test_missing_child_rejected_and_complete_declaration_required():
    t = datetime(2026, 4, 7, 14, 30, tzinfo=UTC)
    source = pl.DataFrame(
        [
            ("S0", t - timedelta(hours=1), "1min", 10.0, 12.0, 9.0, 11.0, 20.0),
            ("S0", t, "1min", 11.0, 13.0, 10.0, 12.0, 30.0),
        ],
        schema=SCHEMA,
        orient="row",
    )
    engine = make_engine(source, t)
    saved = engine.bars
    engine.bars = saved.filter(pl.col("timeframe") != "1h")
    with pytest.raises(CoverageError, match="missing"):
        engine.calculate()
    engine.bars = saved
    assert engine.calculate().filter(pl.col("timeframe") == "1d")["volume"].item() == 50.0
    with pytest.raises(CoverageError, match="declare"):
        prepare_seed(
            archive_for(source), t, Geometry(), complete=False, coverage_start=datetime(2026, 1, 1, tzinfo=UTC)
        )


@pytest.mark.parametrize(
    "through",
    [
        datetime(2026, 4, 1, 15, 20, tzinfo=UTC),  # week starts in prior month/quarter
        datetime(2026, 11, 27, 17, 59, tzinfo=UTC),  # early close after DST
        datetime(2026, 12, 30, 20, 25, tzinfo=UTC),
    ],
)
def test_all_thirteen_match_full_minutes_and_new_minute(year, through):
    source = year.filter(pl.col("timestamp") <= through)
    engine = make_engine(source, through)
    assert_frame_equal(engine.calculate(), current_oracle(source, through))
    new_time = through + timedelta(minutes=1)
    # Early close is exclusive, so no eligible next minute in that case.
    update = year.filter(pl.col("timestamp") == new_time)
    if update.height:
        assert_frame_equal(
            engine.calculate(update, through=new_time), current_oracle(pl.concat([source, update]), new_time)
        )


def test_older_correction_requires_partition_and_rebuilds_current_ancestors(year):
    through = datetime(2026, 4, 1, 16, 20, tzinfo=UTC)
    source = year.filter(pl.col("timestamp") <= through)
    engine = make_engine(source, through)
    old = datetime(2026, 4, 1, 13, 30, tzinfo=UTC)
    update = source.filter(pl.col("timestamp") == old).with_columns(
        pl.lit(999.0).alias("high"), pl.lit(296.0).alias("volume")
    )
    before = engine.bars.clone()
    with pytest.raises(CoverageError, match="outside retained"):
        engine.calculate(update)
    assert_frame_equal(before, engine.bars)
    corrected = replace(source, update)
    restored = corrected.filter((pl.col("timestamp") >= old) & (pl.col("timestamp") < old + timedelta(hours=1)))
    engine.restore("1h", old, restored, complete=True)
    assert_frame_equal(engine.calculate(), current_oracle(corrected, through))
    assert engine.bars.filter((pl.col("timeframe") == "1min") & (pl.col("timestamp") <= old)).is_empty()


def test_hour_day_month_year_rollover_requires_bounded_hydration(year):
    for before, after in [
        (datetime(2026, 4, 1, 14, 29, tzinfo=UTC), datetime(2026, 4, 1, 14, 30, tzinfo=UTC)),
        (datetime(2026, 3, 31, 19, 59, tzinfo=UTC), datetime(2026, 4, 1, 13, 30, tzinfo=UTC)),
    ]:
        oldsource = year.filter(pl.col("timestamp") <= before)
        engine = make_engine(oldsource, before)
        with pytest.raises(CoverageError, match="rollover"):
            engine.calculate(year.filter(pl.col("timestamp") == after), through=after)
        # Caller hydrates only closed HTF rows and the new hour's minute sources.
        oldarchive = archive_for(oldsource)
        archive = pl.concat([oldarchive, year.filter(pl.col("timestamp") == after)])
        geometry = Geometry()
        seed = prepare_seed(archive, after, geometry, complete=True, coverage_start=datetime(2026, 1, 1, tzinfo=UTC))
        assert seed.rows.filter(pl.col("timeframe") == "1min").height == 1
        newengine = MixedCurrent(seed, geometry)
        expectedsource = pl.concat([oldsource, year.filter(pl.col("timestamp") == after)])
        assert_frame_equal(newengine.calculate(), current_oracle(expectedsource, after))


def test_restore_rejects_overlapping_retained_ancestor():
    march = datetime(2026, 3, 31, 13, 30, tzinfo=UTC)
    april = datetime(2026, 4, 1, 13, 30, tzinfo=UTC)
    source = pl.DataFrame(
        [
            ("S0", march, "1min", 10.0, 12.0, 9.0, 11.0, 20.0),
            ("S0", april, "1min", 11.0, 13.0, 10.0, 12.0, 30.0),
        ],
        schema=SCHEMA,
        orient="row",
    )
    engine = make_engine(source, april)
    before = engine.bars.clone()
    corrected = source.head(1).with_columns(pl.lit(40.0).alias("volume"))
    with pytest.raises(CoverageError, match="overlapping retained ancestor"):
        engine.restore("1d", march, corrected, complete=True)
    assert_frame_equal(engine.bars, before)


def test_restore_cannot_erase_missing_child_manifest():
    t = datetime(2026, 4, 7, 13, 30, tzinfo=UTC)
    source = pl.DataFrame(
        [("S0", t + timedelta(hours=i), "1min", 10.0, 12.0, 9.0, 11.0, float((i + 1) * 10)) for i in range(3)],
        schema=SCHEMA,
        orient="row",
    )
    engine = make_engine(source, t + timedelta(hours=2))
    engine.bars = engine.bars.filter(pl.col("timestamp") != t)
    with pytest.raises(CoverageError, match="missing"):
        engine.restore("1h", t + timedelta(hours=1), source.slice(1, 1), complete=True)
    with pytest.raises(CoverageError, match="missing"):
        engine.calculate()


def test_january_week_requires_prior_year_coverage_and_year_rollover():
    december = datetime(2025, 12, 31, 20, 59, tzinfo=UTC)
    january = datetime(2026, 1, 2, 14, 30, tzinfo=UTC)
    source = pl.DataFrame(
        [
            ("S0", december, "1min", 10.0, 12.0, 9.0, 11.0, 20.0),
            ("S0", january, "1min", 11.0, 13.0, 10.0, 12.0, 30.0),
        ],
        schema=SCHEMA,
        orient="row",
    )
    archive = archive_for(source)
    with pytest.raises(CoverageError, match="previous-year"):
        prepare_seed(archive, january, Geometry(), complete=True, coverage_start=datetime(2026, 1, 1, tzinfo=UTC))
    geometry = Geometry()
    seed = prepare_seed(archive, january, geometry, complete=True, coverage_start=datetime(2025, 12, 29, tzinfo=UTC))
    actual = MixedCurrent(seed, geometry).calculate()
    assert actual.filter(pl.col("timeframe") == "1w")["volume"].item() == 50.0
    assert actual.filter(pl.col("timeframe") == "1y")["volume"].item() == 30.0
    assert_frame_equal(actual, current_oracle(source, january))
    oldgeometry = Geometry()
    oldseed = prepare_seed(
        archive_for(source.head(1)),
        december,
        oldgeometry,
        complete=True,
        coverage_start=datetime(2025, 1, 1, tzinfo=UTC),
    )
    with pytest.raises(CoverageError, match="rollover"):
        MixedCurrent(oldseed, oldgeometry).calculate(source.tail(1), through=january)


def test_no_trade_in_current_minute_or_subhour_does_not_fabricate_rows():
    t = datetime(2026, 4, 7, 14, 30, tzinfo=UTC)
    source = pl.DataFrame([("S0", t, "1min", 10.0, 12.0, 9.0, 11.0, 20.0)], schema=SCHEMA, orient="row")
    through = t + timedelta(minutes=15)
    engine = make_engine(source, through)
    output = engine.calculate()
    assert output.filter(pl.col("timeframe").is_in(["1min", "5min", "15min"])).is_empty()
    assert output.filter(pl.col("timeframe") == "1h")["volume"].item() == 20.0
    assert_frame_equal(output, current_oracle(source, through))


def test_restore_rejects_overlapping_retained_descendant():
    march = datetime(2026, 3, 31, 13, 30, tzinfo=UTC)
    april = datetime(2026, 4, 1, 13, 30, tzinfo=UTC)
    quarter = datetime(2026, 1, 1, 14, 30, tzinfo=UTC)
    source = pl.DataFrame(
        [
            ("S0", march, "1min", 10.0, 12.0, 9.0, 11.0, 20.0),
            ("S0", april, "1min", 11.0, 13.0, 10.0, 12.0, 30.0),
        ],
        schema=SCHEMA,
        orient="row",
    )
    engine = make_engine(source, april)
    before = engine.bars.clone()
    corrected = source.head(1).with_columns(pl.lit(40.0).alias("volume"))
    with pytest.raises(CoverageError, match="overlapping retained descendant"):
        engine.restore("1q", quarter, corrected, complete=True)
    assert_frame_equal(engine.bars, before)
    assert_frame_equal(engine.calculate(), current_oracle(source, april))
