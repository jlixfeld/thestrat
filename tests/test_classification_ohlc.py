"""OHLC-only classification does not require or synthesize volume."""

from datetime import UTC, datetime, timedelta
from typing import cast

import polars as pl
import pytest

from thestrat import BarDict, TheStrat, session_for


def ohlc():
    start = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)
    return pl.DataFrame(
        {
            "symbol": ["A", "A"],
            "timeframe": ["1min", "1min"],
            "timestamp": [start, start + timedelta(minutes=1)],
            "open": [10.0, 11.0],
            "high": [12.0, 14.0],
            "low": [9.0, 10.0],
            "close": [11.0, 13.0],
            "note": ["prior", "current"],
        }
    )


def test_ohlc_frame_classifies_without_fabricating_volume():
    strat = TheStrat(session=session_for("equity_us"))
    result = strat.classify(ohlc())
    assert "volume" not in result.columns
    assert result["note"].to_list() == ["prior", "current"]
    assert result["scenario"].to_list() == [None, "2U"]
    assert result["color"].to_list() == ["green", "green"]
    assert result["in_force"].to_list() == [None, True]
    assert strat.bars.is_empty()


@pytest.mark.parametrize("values", [[None, None], ["unavailable", "feed-specific"], [-1.0, float("nan")]])
def test_classification_preserves_unused_volume_metadata(values):
    frame = ohlc().with_columns(pl.Series("volume", values))
    result = TheStrat(session=session_for("equity_us")).classify(frame)
    assert result["volume"].equals(frame["volume"])
    assert result["scenario"].to_list() == [None, "2U"]


def test_scalar_whole_bar_matches_ohlc_frame_without_volume():
    strat = TheStrat(session=session_for("equity_us"))
    source = ohlc()
    scalar = strat.classify.scalar(
        kind="bar", bar=cast(BarDict, source.row(1, named=True)), prior=cast(BarDict, source.row(0, named=True))
    )
    vector = strat.classify(source).row(1, named=True)
    assert "volume" not in scalar
    for field in ("scenario", "color", "shape", "in_force"):
        assert scalar[field] == vector[field]


def test_scalar_whole_bar_preserves_unused_supplied_volume():
    strat = TheStrat(session=session_for("equity_us"))
    row = ohlc().row(0, named=True) | {"volume": "unavailable"}
    result = strat.classify.scalar(kind="bar", bar=cast(BarDict, row))
    assert "volume" in result
    assert result["volume"] == "unavailable"


@pytest.mark.parametrize("column,value", [("high", 8.0), ("open", None), ("close", float("inf"))])
def test_ohlc_classification_still_validates_required_prices(column, value):
    frame = ohlc().with_columns(pl.lit(value, dtype=pl.Float64).alias(column))
    with pytest.raises(ValueError):
        TheStrat(session=session_for("equity_us")).classify(frame)


def test_aggregation_still_requires_numeric_volume():
    for frame in (ohlc(), ohlc().with_columns(pl.lit("unavailable").alias("volume"))):
        with pytest.raises(ValueError, match="volume"):
            TheStrat(session=session_for("equity_us"), bars=frame).aggregate()
