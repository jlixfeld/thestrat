"""Caller declarations of complete source intervals and expected bar keys."""

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class SourceCoverage:
    """Complete point-in-time source coverage, including intervals without trades.

    ``intervals`` has symbol, timeframe, start, end (exclusive) columns.
    ``expected_keys`` has symbol, timeframe, timestamp columns and describes the
    final input after snapshots. Neither frame is inferred from retained bars.
    """

    intervals: pl.DataFrame
    expected_keys: pl.DataFrame
