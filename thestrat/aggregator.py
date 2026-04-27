"""Polars-based timeframe aggregation for OHLCV bars."""

import polars as pl

# Offset for equity hour-based aggregations: US equity markets open at 9:30 ET,
# so hour boundaries (10:30, 11:30, ...) align with the 30-minute offset.
EQUITY_OFFSET_MINUTES = 30


class TimeframeAggregator:
    """Aggregates raw OHLCV bars into higher timeframes using Polars."""

    def aggregate(self, bars: pl.DataFrame, timeframe: str, equity_offset: bool = False) -> pl.DataFrame:
        """Aggregate bars to the given timeframe.

        Args:
            bars: DataFrame with columns: timestamp, open, high, low, close, volume.
            timeframe: Target timeframe.
            equity_offset: If True, offset hour-based timeframes by 30 minutes
                          (for equities opening at :30). Only affects 1h, 4h, 6h, 12h.
        """
        if bars.is_empty():
            return bars

        bars = bars.sort("timestamp")
        group_expr = self._group_expression(timeframe, equity_offset)

        return (
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

    def _group_expression(self, timeframe: str, equity_offset: bool) -> pl.Expr:
        """Return a Polars expression that groups timestamps by timeframe."""
        ts = pl.col("timestamp")

        # Intraday — simple truncation (no offset needed for sub-hour)
        if timeframe == "1min":
            return ts.dt.truncate("1m")
        elif timeframe == "5min":
            return ts.dt.truncate("5m")
        elif timeframe == "15min":
            return ts.dt.truncate("15m")
        elif timeframe == "30min":
            return ts.dt.truncate("30m")

        # Hour-based — may need equity offset
        if timeframe in ("1h", "4h", "6h", "12h"):
            duration = {"1h": "1h", "4h": "4h", "6h": "6h", "12h": "12h"}[timeframe]
            if equity_offset:
                offset = pl.duration(minutes=EQUITY_OFFSET_MINUTES)
                return (ts - offset).dt.truncate(duration) + offset
            return ts.dt.truncate(duration)

        # Daily+ — calendar-based
        if timeframe == "1d":
            return ts.dt.truncate("1d")
        elif timeframe == "1w":
            return ts.dt.truncate("1w")
        elif timeframe == "1m":
            return ts.dt.truncate("1mo")
        elif timeframe == "1q":
            return ts.dt.truncate("1q")
        elif timeframe == "1y":
            return ts.dt.truncate("1y")

        raise ValueError(f"Unsupported aggregation timeframe: {timeframe}")
