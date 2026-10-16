"""Validation shared by explicit consuming operations."""

from collections.abc import Sequence
from typing import cast

import polars as pl

from thestrat.types import DEFAULT_TIMEFRAMES

SCHEMA = {
    "symbol": pl.String,
    "timestamp": pl.Datetime("us", "UTC"),
    "timeframe": pl.String,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
}
KEY = ["symbol", "timeframe", "timestamp"]
PRICES = ["open", "high", "low", "close", "volume"]


def targets(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, str):
        raise ValueError("timeframes must be a nonempty sequence")
    result = tuple(values)
    if not result or len(set(result)) != len(result) or any(v not in DEFAULT_TIMEFRAMES for v in result):
        raise ValueError("timeframes must be nonempty, unique and supported")
    return result


def normalize(
    frame: pl.DataFrame, *, metadata: bool = False, updates: bool = False, classification: bool = False
) -> pl.DataFrame:
    if not isinstance(frame, pl.DataFrame):
        raise ValueError("bars must be a Polars DataFrame")
    schema = {k: v for k, v in SCHEMA.items() if not classification or k != "volume"}
    prices = [name for name in PRICES if not classification or name != "volume"]
    missing = set(schema) - {"symbol", "timeframe"} - set(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    if "symbol" not in frame.columns:
        frame = frame.with_columns(pl.lit(None, dtype=pl.String).alias("symbol"))
    if "timeframe" not in frame.columns:
        frame = frame.with_columns(pl.lit("1min").alias("timeframe"))
    if frame.schema["symbol"] not in (pl.String, pl.Null):
        raise ValueError("symbol must be String")
    if frame.schema["timeframe"] != pl.String:
        raise ValueError("timeframe must be String")
    if not isinstance(frame.schema["timestamp"], pl.Datetime):
        raise ValueError("timestamp must be Datetime")
    for name in prices:
        if not frame.schema[name].is_numeric():
            raise ValueError(f"{name} must be numeric")
    timestamp = pl.col("timestamp")
    if cast(pl.Datetime, frame.schema["timestamp"]).time_zone is None:
        timestamp = timestamp.dt.replace_time_zone("UTC")
    else:
        timestamp = timestamp.dt.convert_time_zone("UTC")
    # Validate at the supplied precision before normalizing storage to microseconds.
    if frame.filter(
        (pl.col("timeframe") == "1min") & ((timestamp.dt.second() != 0) | (timestamp.dt.nanosecond() != 0))
    ).height:
        raise ValueError("source timestamp must be the exact start of its minute")
    frame = frame.with_columns(
        timestamp.cast(SCHEMA["timestamp"]),
        pl.col("symbol").cast(pl.String),
        *(pl.col(n).cast(pl.Float64) for n in prices),
    )
    if frame.select(pl.any_horizontal(*(pl.col(n).is_null() for n in schema if n != "symbol")).any()).item():
        raise ValueError("required bar values must not be null")
    if frame.select(pl.any_horizontal(*(~pl.col(n).is_finite() for n in prices)).any()).item():
        raise ValueError("OHLCV values must be finite")
    invalid = (pl.col("low") > pl.min_horizontal("open", "close")) | (
        pl.col("high") < pl.max_horizontal("open", "close")
    )
    if not classification:
        invalid = invalid | (pl.col("volume") < 0)
    if frame.select(invalid.any()).item():
        raise ValueError("invalid OHLC range or negative volume")
    if frame.filter(~pl.col("timeframe").is_in([str(t) for t in DEFAULT_TIMEFRAMES])).height:
        raise ValueError("unsupported timeframe")
    if updates and frame.filter(pl.col("timeframe") != "1min").height:
        raise ValueError("updates must contain only 1min source bars")
    if frame.select(pl.struct(KEY).is_duplicated().any()).item():
        raise ValueError("duplicate bar key (symbol, timeframe, timestamp)")
    return frame if metadata else frame.select(list(SCHEMA))
