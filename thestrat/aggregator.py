"""Polars-based timeframe aggregation for OHLCV bars.

Hour-and-above timeframes (1h, 4h, 6h, 12h, 1d, 1w, 1m, 1q, 1y) can be
aligned to a `Session` (timezone + anchor offset) via the `session`
parameter, so that buckets line up with the relevant trading day's open
instead of UTC midnight.
"""

from __future__ import annotations

import polars as pl

from thestrat.sessions import SESSIONS, InstrumentType, Session

# Legacy: 30-minute offset for equity hour-based buckets aligned to 9:30 ET.
# `equity_offset=True` is now equivalent to `session=SESSIONS[EQUITY_US]`,
# kept for backward compat.
EQUITY_OFFSET_MINUTES = 30

_HOUR_BASED = {"1h", "4h", "6h", "12h"}
_DAILY_PLUS = {"1d", "1w", "1m", "1q", "1y"}
_TRUNC_UNIT = {
    "1min": "1m",
    "5min": "5m",
    "15min": "15m",
    "30min": "30m",
    "1h": "1h",
    "4h": "4h",
    "6h": "6h",
    "12h": "12h",
    "1d": "1d",
    "1w": "1w",
    "1m": "1mo",  # calendar month, not 1-minute
    "1q": "1q",
    "1y": "1y",
}


class TimeframeAggregator:
    """Aggregates raw OHLCV bars into higher timeframes using Polars."""

    def aggregate(
        self,
        bars: pl.DataFrame,
        timeframe: str,
        equity_offset: bool = False,
        *,
        session: Session | None = None,
    ) -> pl.DataFrame:
        """Aggregate bars to the given timeframe.

        Args:
            bars: DataFrame with columns timestamp, open, high, low, close, volume.
                Timestamp may be naive (assumed UTC) or timezone-aware.
            timeframe: Target timeframe ("5min", "1h", "1d", ...).
            equity_offset: Legacy. If True and `session` is None, applies a
                30-minute offset to hour-based buckets only (9:30 ET equity
                open). Equivalent to passing `session=SESSIONS[EQUITY_US]`
                but only affects hour-based, not daily+. Prefer `session`.
            session: Session preset (timezone + anchor minutes). When given,
                all hour-and-above timeframes are aligned to this anchor.
                Sub-hour timeframes (1min, 5min, 15min, 30min) are unaffected.

        Returns:
            Aggregated DataFrame with the same columns. Timestamps in the
            output are in the same form as the input — naive bars yield
            naive output, aware bars yield aware output (in their original
            timezone).
        """
        if bars.is_empty():
            return bars

        if timeframe not in _TRUNC_UNIT:
            raise ValueError(f"Unsupported aggregation timeframe: {timeframe}")

        bars = bars.sort("timestamp")
        ts_dtype = bars.schema["timestamp"]
        input_was_naive = isinstance(ts_dtype, pl.Datetime) and ts_dtype.time_zone is None

        # Mark naive timestamps as UTC so timezone math is well-defined.
        if input_was_naive:
            bars = bars.with_columns(pl.col("timestamp").dt.replace_time_zone("UTC"))

        group_expr = self._group_expression(timeframe, equity_offset, session)

        result = (
            bars.with_columns(group_expr.alias("_group"))
            .group_by("_group")
            .agg(
                pl.col("timestamp").first().alias("timestamp"),
                pl.col("open").first().alias("open"),
                pl.col("high").max().alias("high"),
                pl.col("low").min().alias("low"),
                pl.col("close").last().alias("close"),
                pl.col("volume").sum().alias("volume"),
            )
            .drop("_group")
            .sort("timestamp")
        )

        # Restore naive timestamps if input was naive.
        if input_was_naive:
            result = result.with_columns(pl.col("timestamp").dt.replace_time_zone(None))

        return result

    def aggregate_for_instrument(
        self,
        bars: pl.DataFrame,
        timeframe: str,
        instrument_type: InstrumentType,
    ) -> pl.DataFrame:
        """Convenience: look up the canonical Session for an instrument type
        and aggregate."""
        return self.aggregate(bars, timeframe, session=SESSIONS[instrument_type])

    def _group_expression(
        self,
        timeframe: str,
        equity_offset: bool,
        session: Session | None,
    ) -> pl.Expr:
        """Return a Polars expression that groups timestamps by timeframe."""
        ts = pl.col("timestamp")
        unit = _TRUNC_UNIT[timeframe]

        # Sub-hour: simple truncation, never session-aligned.
        if timeframe in ("1min", "5min", "15min", "30min"):
            return ts.dt.truncate(unit)

        # Session-aware path (preferred): aligns hour+ buckets to session anchor.
        if session is not None and (timeframe in _HOUR_BASED or timeframe in _DAILY_PLUS):
            offset = pl.duration(minutes=session.anchor_minutes)
            ts_local = ts.dt.convert_time_zone(session.timezone)
            return ((ts_local - offset).dt.truncate(unit) + offset).dt.convert_time_zone("UTC")

        # Legacy `equity_offset` path: 30-min shift for hour-based only.
        if timeframe in _HOUR_BASED:
            if equity_offset:
                offset = pl.duration(minutes=EQUITY_OFFSET_MINUTES)
                return (ts - offset).dt.truncate(unit) + offset
            return ts.dt.truncate(unit)

        # Daily+ without session: UTC-anchored truncation (legacy behavior).
        return ts.dt.truncate(unit)
