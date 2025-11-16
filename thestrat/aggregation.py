"""
OHLC timeframe aggregation with precise time boundary control.

This module provides vectorized OHLC aggregation across different timeframes with
support for asset class-specific timezone handling and boundary alignment.
"""

import logging

from pandas import DataFrame as PandasDataFrame
from polars import (
    DataFrame as PolarsDataFrame,
)
from polars import (
    Int64,
    col,
    first,
    last,
    lit,
    max,
    min,
    sum,
)

from .base import Component
from .schemas import AggregationConfig


class Aggregation(Component):
    """
    Convert OHLC data between timeframes with precise time boundary control.

    Features:
        - **Time Boundary Control**: hour_boundary flag for hourly+ aggregation alignment
        - **Session Control**: session_start parameter for sub-hourly alignment
        - **Asset Class Support**: Crypto, equities, FX, futures with timezone handling
        - **Vectorized Processing**: Pure Polars operations for maximum performance
    """

    # Type hints for commonly accessed attributes
    # Note: These fields are guaranteed to be non-None after config validation
    timezone: str  # Resolved from config.timezone after asset class defaults applied
    asset_class: str  # Always set from config.asset_class
    target_timeframes: list[str]  # Always set from config.target_timeframes
    session_start: str  # Resolved from config.session_start after asset class defaults applied
    hour_boundary: bool  # Resolved from config.hour_boundary after asset class defaults applied

    def __init__(self, config: AggregationConfig):
        """
        Initialize aggregation component with validated configuration.

        Args:
            config: Validated AggregationConfig containing all configuration settings
        """
        super().__init__()

        # Store the validated Pydantic config
        self.config = config

        # Extract commonly used values for convenience
        self.target_timeframes = config.target_timeframes.copy()
        self.asset_class = config.asset_class

        # These fields are guaranteed to be non-None after model validation
        assert config.timezone is not None
        assert config.session_start is not None
        assert config.hour_boundary is not None

        self.timezone = config.timezone
        self.session_start = config.session_start
        self.hour_boundary = config.hour_boundary

        # Validate all timeframes
        from .schemas import TimeframeConfig

        for tf in self.target_timeframes:
            if not TimeframeConfig.validate_timeframe(tf):
                raise ValueError(
                    f"Invalid timeframe '{tf}'. Supported timeframes: {sorted(TimeframeConfig.TIMEFRAME_METADATA.keys())}"
                )

    def process(self, data: PolarsDataFrame | PandasDataFrame) -> PolarsDataFrame:
        """
        Convert OHLC data to target timeframes with boundary alignment.

        Args:
            data: Input DataFrame with OHLC data including MANDATORY timeframe column

        Returns:
            Aggregated OHLC DataFrame with consistent column ordering
        """
        if not self.validate_input(data):
            raise ValueError("Input data validation failed")

        df = self._convert_to_polars(data)

        # Validate and log timezone information
        self.validate_input_timezones(df)

        # Convert timestamps to target timezone
        df = self.normalize_timezone(df)

        # Process the timeframe data (timeframe column is mandatory)
        return self._process_timeframes(df)

    def _process_timeframes(self, data: PolarsDataFrame) -> PolarsDataFrame:
        """
        Process timeframe source data.
        Intelligently selects optimal source timeframe for each target.
        """
        from .schemas import TimeframeConfig

        # Get available source timeframes
        available_timeframes = data["timeframe"].unique().to_list()

        results = []
        for target_tf in self.target_timeframes:
            # Get optimal source for this target
            source_tf = TimeframeConfig.get_optimal_source_timeframe(target_tf, available_timeframes)

            if source_tf == target_tf:
                # Pass-through: target already exists in source
                target_data = data.filter(col("timeframe") == target_tf)
                target_data = self._standardize_column_order(target_data)
                results.append(target_data)
            elif source_tf:
                # Aggregate from optimal source
                source_data = data.filter(col("timeframe") == source_tf).drop("timeframe")
                aggregated = self._process_single_timeframe(source_data, target_tf)
                aggregated = self._standardize_column_order(aggregated)
                results.append(aggregated)
            else:
                # No valid source available for this target - log warning but continue
                logging.warning(
                    f"No valid source timeframe available for target '{target_tf}'. Available: {available_timeframes}"
                )

        if not results:
            raise ValueError("No valid aggregations could be performed")

        # Combine all results
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
            # Use session-based alignment with offset for all periods
            # This preserves calendar boundaries while maintaining session start times
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

        # Add timeframe column
        result = result.with_columns([lit(timeframe).alias("timeframe")])

        # Cast volume to integer to prevent floating-point precision issues
        if "volume" in result.columns:
            result = result.with_columns([col("volume").cast(Int64).alias("volume")])

        # Sort by symbol (if present) and timestamp
        sort_cols = []
        if "symbol" in result.columns:
            sort_cols.append("symbol")
        sort_cols.append("timestamp")

        return result.sort(sort_cols)

    def validate_input_timezones(self, data: PolarsDataFrame) -> None:
        """
        Validate and log timezone information for input data.

        Performs O(1) timezone detection using Polars dtype metadata and logs
        helpful warnings/guidance for users about timezone handling.

        Args:
            data: Input DataFrame with timestamp column

        Note:
            This method uses Polars schema metadata (O(1)) rather than scanning
            rows, as Polars enforces homogeneous column types - all timestamps
            in a column share the same timezone (or all are naive).
        """
        ts_dtype = data.schema["timestamp"]

        # Extract timezone from dtype metadata (None if naive)
        current_tz = ts_dtype.time_zone if hasattr(ts_dtype, "time_zone") else None

        if current_tz is None:
            # Naive timestamps - warn user and provide conversion example
            first_row_preview = data[0] if len(data) > 0 else "No data"
            logging.warning(
                f"Naive timestamps detected - assuming data is already in {self.timezone}\n"
                f"First row: {first_row_preview}\n"
                f"If your data is UTC from database, convert before feeding to Factory:\n"
                f"  df = df.with_columns(\n"
                f"      pl.col('timestamp')\n"
                f"        .dt.replace_time_zone('UTC')\n"
                f"        .dt.convert_time_zone('{self.timezone}')\n"
                f"  )"
            )
        elif current_tz != self.timezone:
            # Timezone-aware but different from target - will be converted
            first_row_preview = data[0] if len(data) > 0 else "No data"
            logging.info(f"Converting timestamps from {current_tz} to {self.timezone}\nFirst row: {first_row_preview}")
        # else: Already in target timezone, no logging needed

    def normalize_timezone(self, data: PolarsDataFrame) -> PolarsDataFrame:
        """
        Convert timestamps to target timezone.

        Handles both naive and timezone-aware timestamps:
        - Naive: Assumes already in target timezone, adds timezone awareness
        - Aware (different): Converts from current timezone to target timezone
        - Aware (same): No conversion needed

        Args:
            data: Input DataFrame with timestamp column

        Returns:
            DataFrame with timezone-aware timestamps in target timezone
        """
        df = data.clone()

        ts_dtype = df.schema["timestamp"]

        # Extract timezone from dtype metadata (None if naive)
        current_tz = ts_dtype.time_zone if hasattr(ts_dtype, "time_zone") else None

        if current_tz is None:
            # Naive timestamps - assume already in target timezone, add awareness
            df = df.with_columns([col("timestamp").dt.replace_time_zone(self.timezone)])
        elif current_tz != self.timezone:
            # Timezone-aware but different - CONVERT to target timezone
            df = df.with_columns([col("timestamp").dt.convert_time_zone(self.timezone)])
        # else: Already in target timezone, no conversion needed

        return df

    def validate_input(self, data: PolarsDataFrame | PandasDataFrame) -> bool:
        """Validate input data format."""
        df = self._convert_to_polars(data)

        from .schemas import IndicatorSchema, TimeframeConfig

        # Check all required columns (including mandatory timeframe)
        required_cols = IndicatorSchema.get_required_input_columns()
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            print(f"ERROR: Missing required columns: {missing_cols}")
            return False

        # Validate timeframe values
        unique_timeframes = df["timeframe"].unique().to_list()
        for tf in unique_timeframes:
            if not TimeframeConfig.validate_timeframe(tf):
                print(f"ERROR: Invalid timeframe '{tf}'")
                return False

        # Check for minimum data points
        if len(df) < 2:
            print(f"ERROR: Insufficient data points ({len(df)} < 2)")
            return False

        return True

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

    def _standardize_column_order(self, df: PolarsDataFrame) -> PolarsDataFrame:
        """Apply standard column ordering from schema."""
        from .schemas import IndicatorSchema

        standard_order = IndicatorSchema.get_standard_column_order()
        existing_cols = [col for col in standard_order if col in df.columns]
        remaining_cols = [col for col in df.columns if col not in existing_cols]

        return df.select(existing_cols + remaining_cols)
