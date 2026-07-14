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


def bucket_key_expr(timeframe: str, session: Session | None) -> pl.Expr:
    """Return the Polars expression that maps each timestamp to its bucket key.

    The "bucket key" is the value thestrat groups on to aggregate 1-minute bars
    into a higher timeframe — internally the aggregator calls it `_group`. This
    is the supported public form of that grouping logic, so consumers can reuse
    it instead of re-implementing the session-anchored bucketing.

    Args:
        timeframe: Target timeframe ("1min", "5min", "15min", "30min", "1h",
            "4h", "6h", "12h", "1d", "1w", "1m", "1q", "1y").
        session: Session preset (timezone + anchor minutes). When given, all
            hour-and-above timeframes are aligned to this anchor (e.g.
            `SESSIONS[EQUITY_US]` anchors hourly buckets to 09:30 ET and daily+
            to the 09:30 ET session start). When None, hour-and-above buckets
            fall back to plain UTC-anchored truncation. Sub-hour timeframes
            (1min, 5min, 15min, 30min) are always plain truncation regardless
            of `session`.

    Returns:
        A `pl.Expr` over a `timestamp` column producing the bucket key.
        If a `session` is provided, the `timestamp` column must be
        timezone-aware — the expression calls `dt.convert_time_zone`, which
        raises on a naive column. (`TimeframeAggregator.aggregate` coerces
        naive timestamps to UTC before applying it; consumers using this
        expression directly must do the same.)

    DST note:
        Session-aware bucketing is wall-clock ("calendar") bucketing in the
        session timezone. Anchor arithmetic is performed on naive local
        timestamps so a DST transition inside a bucket does not shift its
        boundary (subtracting a physical duration from a tz-aware column
        would misanchor buckets that span a transition — e.g. the CME Globex
        Sunday session on a US spring-forward date). Consequences: on a
        fall-back date the repeated local hour merges into one wall-clock
        bucket, and a bucket key that would land inside a spring-forward gap
        (a non-existent local time) is shifted forward one hour to the first
        instant that exists — the true start of that bucket.

    Raises:
        ValueError: if `timeframe` is not a supported aggregation timeframe.
    """
    if timeframe not in _TRUNC_UNIT:
        raise ValueError(f"Unsupported aggregation timeframe: {timeframe}")

    ts = pl.col("timestamp")
    unit = _TRUNC_UNIT[timeframe]

    # Sub-hour: simple truncation, never session-aligned.
    if timeframe in ("1min", "5min", "15min", "30min"):
        return ts.dt.truncate(unit)

    # Session-aware path (preferred): aligns hour+ buckets to the session anchor.
    if session is not None:
        offset = pl.duration(minutes=session.anchor_minutes)
        # Wall-clock arithmetic: strip the timezone so the anchor offset and
        # truncation operate on local calendar time. Physical-duration
        # arithmetic on a tz-aware column shifts by exact physical time and
        # misanchors any bucket spanning a DST transition (see DST note).
        ts_naive = ts.dt.convert_time_zone(session.timezone).dt.replace_time_zone(None)
        key_naive = (ts_naive - offset).dt.truncate(unit) + offset
        # Re-attach the timezone. `ambiguous="earliest"`: a key in a repeated
        # (fall-back) hour is the first occurrence — the bucket's true start.
        # `non_existent="null"` + fill: a key inside a spring-forward gap is
        # shifted forward one hour to the first local instant that exists.
        key = key_naive.dt.replace_time_zone(session.timezone, ambiguous="earliest", non_existent="null")
        gap_shifted = (key_naive + pl.duration(hours=1)).dt.replace_time_zone(
            session.timezone, ambiguous="earliest", non_existent="null"
        )
        return key.fill_null(gap_shifted).dt.convert_time_zone("UTC")

    # No session: UTC-anchored truncation (legacy behavior).
    return ts.dt.truncate(unit)


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
        """Return a Polars expression that groups timestamps by timeframe.

        Delegates to the public `bucket_key_expr` for the session-aware and
        no-session paths. The only branch retained here is the legacy
        `equity_offset` shortcut (hour-based only, no session) preserved for
        backward compatibility of `aggregate(..., equity_offset=True)`.
        """
        # Legacy `equity_offset` path: 30-min shift for hour-based only, and
        # only when no session is supplied (a session takes precedence).
        if session is None and equity_offset and timeframe in _HOUR_BASED:
            ts = pl.col("timestamp")
            unit = _TRUNC_UNIT[timeframe]
            offset = pl.duration(minutes=EQUITY_OFFSET_MINUTES)
            return (ts - offset).dt.truncate(unit) + offset

        return bucket_key_expr(timeframe, session)
