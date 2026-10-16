"""Visible state, authoritative snapshots, validation, and composed classification."""

from datetime import timedelta
from typing import cast

import polars as pl
import pytest
from test_sessions import bars, utc

from thestrat import DEFAULT_TIMEFRAMES, BarDict, InsufficientSourceDataError, TheStrat, session_for


def instance(frame=None, targets=("1min", "5min", "1h", "1d")):
    return TheStrat(session=session_for("crypto"), timeframes=targets, bars=frame)


def test_assignment_no_work_and_explicit_classification():
    source = bars([utc(2026, 1, 1)])
    s = instance(source)
    assert s.bars is source
    result = s.aggregate()
    assert "scenario0" not in result.columns
    working = s.bars
    classified = s.classify(result)
    assert s.bars is working
    assert classified.height == result.height
    s.bars = source.filter(pl.lit(False))
    assert s.aggregate().is_empty()


def test_snapshot_replaces_retracted_extrema_volume_and_endpoints():
    stamp = utc(2026, 1, 1, 11, 22)
    source = bars([stamp, stamp + timedelta(minutes=1)]).with_columns(
        pl.Series("open", [10.0, 11.0]),
        pl.Series("close", [11.0, 10.0]),
        pl.Series("high", [20.0, 15.0]),
        pl.Series("low", [1.0, 5.0]),
        pl.Series("volume", [203.0, 10.0]),
    )
    s = instance(source)
    s.aggregate()
    revised = bars([stamp]).with_columns(
        pl.lit(9.0).alias("open"),
        pl.lit(13.0).alias("high"),
        pl.lit(6.0).alias("low"),
        pl.lit(12.0).alias("close"),
        pl.lit(296.0).alias("volume"),
    )
    output = s.aggregate(revised)
    hour = output.filter(pl.col("timeframe") == "1h").row(0, named=True)
    assert [hour[n] for n in ("open", "high", "low", "close", "volume")] == [9.0, 15.0, 5.0, 10.0, 306.0]
    assert s.aggregate(revised).equals(output)
    # New batch from the final visible source set gives exactly the same output.
    assert instance(s.bars.filter(pl.col("timeframe") == "1min")).aggregate().equals(output)


def test_source_retained_when_not_selected_null_symbol_replacement():
    source = bars([utc(2026, 1, 1, 0, 1)])
    s = instance(source, targets=["5min"])
    assert s.aggregate()["timeframe"].to_list() == ["5min"]
    assert s.bars["timeframe"].to_list() == ["1min", "5min"]
    update = source.with_columns(pl.lit(5.0).alias("volume"))
    assert s.aggregate(update)["volume"].to_list() == [5.0]


def test_no_hidden_source_after_pruning_summary_insufficiency():
    s = instance(bars([utc(2026, 1, 1)]), targets=["5min"])
    s.aggregate()
    s.bars = s.bars.filter(pl.col("timeframe") != "1min")
    before = s.bars
    # Summary still supports classification, but cannot be guessed into a correction.
    assert s.classify(s.bars)[0, "color0"] == "green"
    with pytest.raises(InsufficientSourceDataError, match="underlying"):
        s.aggregate(bars([utc(2026, 1, 1)]))
    assert s.bars is before


def test_sparse_sources_valid_no_synthetic_grid_and_unaffected_seed():
    s = instance(bars([utc(2026, 1, 1, 0, 1), utc(2026, 1, 1, 0, 4)]), targets=["5min"])
    assert s.aggregate()[0, "volume"] == 2.0
    assert s.bars.height == 3
    # Retained opaque windows outside the submitted update remain summaries.
    seed = s.bars.filter(pl.col("timeframe") == "5min")
    other = bars([utc(2026, 1, 2)])
    s.bars = seed
    output = s.aggregate(other)
    assert output["volume"].to_list() == [2.0, 1.0]


def test_delta_unknown_and_validation_failure_atomic():
    s = instance(bars([utc(2026, 1, 1)]))
    original = s.bars
    with pytest.raises(NotImplementedError):
        s.aggregate(mode="delta")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        s.aggregate(mode="wat")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="duplicate"):
        s.aggregate(pl.concat([original, original]))
    assert s.bars is original


@pytest.mark.parametrize(
    "column,value,match",
    [("high", 9.0, "range"), ("volume", -1.0, "volume"), ("close", float("inf"), "finite"), ("open", None, "null")],
)
def test_invalid_values(column, value, match):
    frame = bars([utc(2026, 1, 1)]).with_columns(pl.lit(value, dtype=pl.Float64).alias(column))
    with pytest.raises(ValueError, match=match):
        instance(frame).aggregate()


def test_source_identity_and_update_timeframe():
    with pytest.raises(ValueError, match="exact start"):
        instance(bars([utc(2026, 1, 1) + timedelta(seconds=1)])).aggregate()
    s = instance()
    with pytest.raises(ValueError, match="only 1min"):
        s.aggregate(bars([utc(2026, 1, 1)]).with_columns(pl.lit("1h").alias("timeframe")))


def test_empty_and_all_default_targets():
    s = TheStrat(session=session_for("crypto"))
    assert s.aggregate().is_empty()
    assert s.classify(s.bars, history=4).schema["scenario4"] == pl.String
    assert s.classify(s.bars).schema["in_force"] == pl.Boolean
    assert len(DEFAULT_TIMEFRAMES) == 13
    assert s.aggregate(bars([utc(2026, 1, 1)]))["timeframe"].to_list() == list(DEFAULT_TIMEFRAMES)


def test_groups_history_metadata_selective_recomputation():
    stamps = [utc(2026, 1, 1) + timedelta(minutes=i) for i in range(6)]
    source = bars(stamps).with_columns(
        pl.Series("open", [10.0, 11.0, 10.0, 10.0, 11.0, 10.0]),
        pl.Series("close", [11.0, 10.0, 10.0, 11.0, 10.0, 10.0]),
        pl.Series("high", [12.0, 13.0, 12.0, 14.0, 13.0, 15.0]),
        pl.Series("low", [8.0, 9.0, 8.0, 7.0, 8.0, 6.0]),
        pl.lit("keep").alias("metadata"),
    )
    groups = pl.concat(
        [
            source.with_columns(pl.lit(symbol, dtype=pl.String).alias("symbol"), pl.lit(tf).alias("timeframe"))
            for symbol in [None, "A"]
            for tf in ["1min", "1h"]
        ]
    )
    s = instance()
    output = s.classify(groups.reverse(), fields=["scenario", "color"], history=4)
    assert output.height == 24
    for group in output.partition_by("symbol", "timeframe"):
        assert group[0, "scenario0"] is None
        assert group[-1, "color4"] == "red"
        assert group[-1, "scenario4"] == "2U"
        assert group["metadata"].to_list() == ["keep"] * 6
    assert "shape" not in output.columns and "in_force" not in output.columns
    stale = source.with_columns(pl.lit("wrong").alias("color0"), pl.lit("caller").alias("shape"))
    assert s.classify(stale, fields=["color"], history=0)[0, "color0"] == "green"
    assert s.classify(stale, fields=["color"], history=0)[0, "shape"] == "caller"
    assert s.classify(source, fields=[]).columns == [*source.columns, "symbol", "timeframe"]


@pytest.mark.parametrize(
    "kwargs",
    [{"history": True}, {"history": -1}, {"history": 1.5}, {"fields": ["unknown"]}, {"fields": ["color", "color"]}],
)
def test_classification_invalid_selection(kwargs):
    with pytest.raises(ValueError):
        instance().classify(bars([utc(2026, 1, 1)]), **kwargs)


def test_scalar_kinds_parity_and_no_history():
    frame = bars([utc(2026, 1, 1), utc(2026, 1, 1, 0, 1)]).with_columns(
        pl.Series("open", [10.0, 10.0]),
        pl.Series("high", [11.0, 12.0]),
        pl.Series("low", [9.0, 8.0]),
        pl.Series("close", [10.0, 11.5]),
    )
    s = instance()
    vector = s.classify(frame, history=0)
    previous = vector.row(0, named=True)
    current = vector.row(1, named=True)
    scalar = s.classify.scalar(kind="bar", bar=cast(BarDict, current), prior=cast(BarDict, previous))
    for field in ["scenario0", "color0", "shape", "in_force"]:
        assert scalar[field] == current[field]
    assert set(scalar) == set(current)
    assert s.classify.scalar(kind="color", open_price=10.0, close_price=11.5) == current["color0"]
    assert (
        s.classify.scalar(kind="scenario", high=12.0, low=8.0, reference_high=11.0, reference_low=9.0)
        == current["scenario0"]
    )
    assert s.classify.scalar(kind="shape", open_price=10.0, high=12.0, low=8.0, close_price=11.5) == current["shape"]
    assert (
        s.classify.scalar(
            kind="in_force",
            open_price=10.0,
            high=12.0,
            low=8.0,
            close_price=11.5,
            reference_high=11.0,
            reference_low=9.0,
        )
        is True
    )
    with pytest.raises(TypeError):
        s.classify.scalar(kind="scenario", high=12.0, low=8.0)  # type: ignore[call-overload]
    with pytest.raises(TypeError):
        s.classify.scalar(kind="color", open_price=10.0, close_price=11.0, history=1)  # type: ignore[call-overload]
    with pytest.raises(ValueError):
        s.classify.scalar(kind="shape", open_price=100.0, high=12.0, low=8.0, close_price=10.0)


def test_ten_thousand_rows_grouped_batch_and_prefix_only():
    start = utc(2026, 1, 1)
    source = bars([start + timedelta(minutes=i) for i in range(10_000)]).with_columns(pl.lit("A").alias("symbol"))
    s = instance(source, targets=["5min", "1d"])
    result = s.aggregate()
    assert result.filter(pl.col("timeframe") == "5min").height == 2000
    assert result.filter(pl.col("timeframe") == "5min")["volume"].sum() == 10_000.0
    prefix = instance(source.head(3), targets=["5min"]).aggregate()
    assert prefix[0, "volume"] == 3.0
    assert result.filter(pl.col("timeframe") == "5min")[0, "volume"] == 5.0


def test_hard_cut_no_old_exports_or_modules():
    import importlib.util

    import thestrat

    for name in ["TimeframeAggregator", "ClassifiedBar", "bucket_key_expr", "classify_bar", "classify_bars_df"]:
        assert not hasattr(thestrat, name)
    for module in ["aggregator", "aggregation", "base", "factory", "schemas", "signals", "indicators", "precision"]:
        assert importlib.util.find_spec(f"thestrat.{module}") is None


def test_scalar_whole_bar_uses_no_dataframe(monkeypatch):
    s = instance()
    current = {
        "symbol": None,
        "timestamp": utc(2026, 1, 1),
        "timeframe": "1d",
        "open": 10.0,
        "high": 12.0,
        "low": 8.0,
        "close": 11.0,
        "volume": 1.0,
    }

    def forbidden(*args, **kwargs):
        raise AssertionError("scalar path must not construct a DataFrame")

    monkeypatch.setattr(pl, "DataFrame", forbidden)
    assert s.classify.scalar(kind="bar", bar=cast(BarDict, current))["color0"] == "green"


@pytest.mark.parametrize("operation", ["aggregate", "update", "classify"])
def test_nanosecond_source_timestamp_rejected_before_precision_loss(operation):
    minute = utc(2026, 1, 1)
    minute_ns = int(minute.timestamp()) * 1_000_000_000
    source = bars([minute]).with_columns(pl.Series("timestamp", [minute_ns + 1]).cast(pl.Datetime("ns", "UTC")))
    s = instance(source if operation == "aggregate" else None)
    before = s.bars
    with pytest.raises(ValueError, match="exact start"):
        if operation == "update":
            s.aggregate(source)
        elif operation == "classify":
            s.classify(source)
        else:
            s.aggregate()
    assert s.bars is before
