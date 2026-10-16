"""Deterministic external source fixture. Never retained by MixedCurrent."""

from datetime import UTC, datetime

import polars as pl

from thestrat import TheStrat, session_for
from thestrat.frames import SCHEMA


def source_year() -> pl.DataFrame:
    schedule = session_for("equity_us").schedule(
        start=datetime(2026, 1, 1, tzinfo=UTC), end=datetime(2027, 1, 1, tzinfo=UTC)
    )
    timestamps = schedule.select(
        pl.datetime_ranges("segment_open", "segment_close", interval="1m", closed="left").alias("timestamp")
    ).explode("timestamp")
    return (
        timestamps.with_row_index("i")
        .with_columns(
            pl.lit("S0").alias("symbol"),
            pl.lit("1min").alias("timeframe"),
            (100.0 + (pl.col("i") % 1000) / 4).alias("open"),
            (104.0 + (pl.col("i") % 1000) / 4).alias("high"),
            (96.0 + (pl.col("i") % 1000) / 4).alias("low"),
            (101.0 + (pl.col("i") % 1000) / 4).alias("close"),
            (100 + pl.col("i") % 101).cast(pl.Float64).alias("volume"),
        )
        .select(list(SCHEMA))
    )


def archive_for(minutes: pl.DataFrame) -> pl.DataFrame:
    strat = TheStrat(session=session_for("equity_us"), timeframes=["1h", "1d", "1m", "1q"], bars=minutes)
    strat.aggregate()
    return strat.bars
