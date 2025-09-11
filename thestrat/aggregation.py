"""
OHLC timeframe aggregation with precise time boundary control.

This module provides vectorized OHLC aggregation across different timeframes with
support for asset class-specific timezone handling and boundary alignment.
"""

from pandas import DataFrame as PandasDataFrame
from polars import (
    DataFrame as PolarsDataFrame,
)
from polars import (
    Datetime,
    col,
    first,
    last,
    lit,
    max,
    min,
    sum,
)

from .base import Component
from .schemas import AssetClassConfig


class Aggregation(Component):
    """
    Convert OHLC data between timeframes with precise time boundary control.

    Features:
        - **Time Boundary Control**: hour_boundary flag for hourly+ aggregation alignment
        - **Session Control**: session_start parameter for sub-hourly alignment
        - **Asset Class Support**: Crypto, equities, FX, futures with timezone handling
        - **Vectorized Processing**: Pure Polars operations for maximum performance
    """

    def __init__(
        self,
        target_timeframes: list[str],
        asset_class: str = "equities",
        hour_boundary: bool | None = None,
        session_start: str | None = None,
        timezone: str | None = None,
    ):
        """
        Initialize aggregation component.

        Args:
            target_timeframes: Target timeframes list (e.g., ["5m", "1H", "1D"])
            asset_class: Asset class type (crypto, equities, fx, futures)
            hour_boundary: Align to hour boundaries (auto-determined if None)
            session_start: Session start time (asset class default if None)
            timezone: Timezone (asset class default -> system -> UTC if None)
        """
        super().__init__()

        # Validate that target_timeframes is a list
        if not isinstance(target_timeframes, list):
            raise TypeError("target_timeframes must be a list")

        if not target_timeframes:
            raise ValueError("target_timeframes cannot be empty")

        self.target_timeframes = list(target_timeframes)

        self.asset_class = asset_class

        # Get asset class defaults from registry
        asset_config = AssetClassConfig.REGISTRY.get(asset_class, AssetClassConfig.REGISTRY["equities"])

        # Set timezone with priority: specified > system > asset_class default
        self.timezone = self._resolve_timezone(timezone, asset_class, asset_config)

        # Set session start with asset class default
        self.session_start = session_start or asset_config.session_start

        # Set hour_boundary with priority: specified > asset_class default
        self.hour_boundary = hour_boundary if hour_boundary is not None else asset_config.hour_boundary

        # Validate all timeframes
        for tf in self.target_timeframes:
            self._validate_timeframe(tf)

    def process(self, data: PolarsDataFrame | PandasDataFrame) -> PolarsDataFrame:
        """
        Convert OHLC data to target timeframes with boundary alignment.

        Args:
            data: Input DataFrame with OHLC data

        Returns:
            Aggregated OHLC DataFrame with timeframe column
        """
        if not self.validate_input(data):
            raise ValueError("Input data validation failed")

        # Convert to Polars if needed
        df = self._convert_to_polars(data)

        # Normalize timezone
        df = self.normalize_timezone(df)

        # Process each timeframe and collect results
        results = []
        for timeframe in self.target_timeframes:
            # Process this specific timeframe
            timeframe_result = self._process_single_timeframe(df, timeframe)
            results.append(timeframe_result)

        # Concatenate all results
        final_df = results[0]
        for result in results[1:]:
            final_df = final_df.vstack(result)

        # Sort by symbol (if present), timeframe, then timestamp
        sort_cols = []
        if "symbol" in final_df.columns:
            sort_cols.append("symbol")
        sort_cols.extend(["timeframe", "timestamp"])

        return final_df.sort(sort_cols)

    def _process_single_timeframe(self, data: PolarsDataFrame, timeframe: str) -> PolarsDataFrame:
        """Process data for a single timeframe using Polars native functionality."""
        from .schemas import TimeframeConfig

        polars_format = TimeframeConfig.get_polars_format(timeframe)
        if not polars_format:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        # Apply timezone normalization
        df = self.normalize_timezone(data)

        # Apply boundary alignment and aggregation using Polars native functionality
        result = self._aggregate_with_boundaries(df, timeframe, polars_format)

        return result

    def _aggregate_with_boundaries(self, data: PolarsDataFrame, timeframe: str, polars_format: str) -> PolarsDataFrame:
        """Aggregate data using Polars native functionality with proper boundary alignment."""
        df = data.clone()

        # Ensure data is sorted by timestamp (required for group_by_dynamic)
        sort_cols = []
        if "symbol" in df.columns:
            sort_cols.append("symbol")
        sort_cols.append("timestamp")
        df = df.sort(sort_cols)

        # Determine if we should use hour boundaries or session-based alignment
        should_use_hour_boundary = (
            self.hour_boundary if self.hour_boundary is not None else self._should_use_hour_boundary(timeframe)
        )

        # For timeframes >= 1h, when hour_boundary=False, always use session alignment
        if not should_use_hour_boundary and self._is_hourly_or_higher(timeframe):
            should_use_hour_boundary = False  # Explicitly use session alignment

        # Build OHLC aggregation expressions
        agg_expressions = [
            first("open").alias("open"),
            max("high").alias("high"),
            min("low").alias("low"),
            last("close").alias("close"),
        ]

        # Add volume if present
        if "volume" in df.columns:
            agg_expressions.append(sum("volume").alias("volume"))

        # Note: symbol column is preserved by group_by when using group_by_dynamic

        # Apply boundary-aware aggregation
        if should_use_hour_boundary:
            # Use hour boundary alignment - Polars handles this natively
            if "symbol" in df.columns:
                result = df.group_by_dynamic(
                    "timestamp", every=polars_format, period=polars_format, group_by="symbol", closed="left"
                ).agg(agg_expressions)
            else:
                result = df.group_by_dynamic("timestamp", every=polars_format, period=polars_format, closed="left").agg(
                    agg_expressions
                )
        else:
            # Use session-based alignment with offset
            session_hour, session_minute = map(int, self.session_start.split(":"))
            offset_minutes = session_hour * 60 + session_minute
            offset = f"{offset_minutes}m"

            if "symbol" in df.columns:
                result = df.group_by_dynamic(
                    "timestamp",
                    every=polars_format,
                    period=polars_format,
                    offset=offset,
                    group_by="symbol",
                    closed="left",
                ).agg(agg_expressions)
            else:
                result = df.group_by_dynamic(
                    "timestamp", every=polars_format, period=polars_format, offset=offset, closed="left"
                ).agg(agg_expressions)

        # Add timeframe column and sort
        result = result.with_columns([lit(timeframe).alias("timeframe")])

        # Sort by symbol (if present) and timestamp
        sort_cols = []
        if "symbol" in result.columns:
            sort_cols.append("symbol")
        sort_cols.append("timestamp")

        return result.sort(sort_cols)

    def normalize_timezone(self, data: PolarsDataFrame) -> PolarsDataFrame:
        """Convert naive timestamps to timezone-aware using timezone resolution priority."""
        df = data.clone()

        # Check if timestamp is already timezone-aware
        if df.schema["timestamp"] == Datetime("us", None):  # Naive timestamp
            # Convert to timezone-aware
            df = df.with_columns([col("timestamp").dt.replace_time_zone(self.timezone).alias("timestamp")])

        return df

    def validate_input(self, data: PolarsDataFrame | PandasDataFrame) -> bool:
        """Validate input data format."""
        df = self._convert_to_polars(data)

        # Check required columns
        required_cols = ["timestamp", "open", "high", "low", "close"]
        if not all(col in df.columns for col in required_cols):
            return False

        # Check for minimum data points
        if len(df) < 2:
            return False

        return True

    def _resolve_timezone(self, timezone: str | None, asset_class: str, asset_config) -> str:
        """Resolve timezone with priority: specified > asset_class default > system."""

        # UTC asset classes always use UTC
        if asset_class in ["crypto", "fx"]:
            return "UTC"

        # For non-UTC asset classes, use specified timezone if provided
        if timezone:
            return timezone

        # Use asset class default for non-UTC asset classes (equities)
        return asset_config.timezone

    def _should_use_hour_boundary(self, timeframe: str) -> bool:
        """Determine if timeframe should use hour boundary alignment."""
        # Check supported timeframe strings first
        hourly_or_higher_supported = {"1h", "4h", "1d", "1w", "1m", "1q", "1y"}
        if timeframe in hourly_or_higher_supported:
            return True

        # For polars-style timeframes, parse and check unit (case-insensitive)
        import re

        match = re.match(r"^(\d+)([mMhHdDwWyY]|[mM][oO]?)$", timeframe, re.IGNORECASE)
        if match:
            unit = match.group(2).lower()  # Normalize to lowercase
            # Use hour boundary for hourly and higher timeframes
            if unit in ["h", "d", "w", "mo", "y"]:
                return True

        return False

    def _is_hourly_or_higher(self, timeframe: str) -> bool:
        """Check if timeframe is hourly or higher (1h, 4h, 6h, 12h, 1d, 1w, etc)."""
        import re

        # Check for hourly timeframes (1h, 4h, 6h, 12h, etc)
        if re.match(r"^(\d+)([hH])$", timeframe):
            return True

        # Check for daily and higher timeframes
        if timeframe in ["1d", "1w", "1m", "1q", "1y"]:
            return True

        # Check polars-style patterns for daily and higher
        if re.match(r"^(\d+)([dDwWmMyY]|[mM][oO])$", timeframe, re.IGNORECASE):
            return True

        return False

    def _validate_timeframe(self, timeframe: str) -> None:
        """Validate that the timeframe is supported."""
        if not isinstance(timeframe, str):
            raise TypeError(f"Timeframe must be a string, got {type(timeframe)}")

        if not timeframe:
            raise ValueError("Timeframe cannot be empty")

        from .schemas import TimeframeConfig

        if not TimeframeConfig.validate_timeframe(timeframe):
            raise ValueError(
                f"Invalid timeframe '{timeframe}'. Supported timeframes: {sorted(TimeframeConfig.TIMEFRAME_TO_POLARS.keys())}"
            )
