"""
Vectorized Strat technical indicators implementation.

This module provides comprehensive Strat pattern analysis with high-performance
vectorized calculations using Polars operations.
"""

from typing import Any

from pandas import DataFrame as PandasDataFrame
from polars import DataFrame as PolarsDataFrame
from polars import Int32, col, concat_str, lit, when

from .base import Component
from .signals import SIGNALS


class Indicators(Component):
    """
    Vectorized implementation of all Strat technical indicators.

    Provides comprehensive technical analysis including swing point detection, market structure
    analysis, pattern recognition, and gap detection optimized for TheStrat methodology.
    """

    def __init__(self, timeframe_configs: list[dict[str, Any]]):
        """
        Initialize indicators component with per-timeframe configurations.

        Args:
            timeframe_configs: List of configuration dictionaries for different timeframes.
        """
        super().__init__()

        # Validate that timeframe_configs is provided
        if not timeframe_configs:
            raise ValueError(
                "timeframe_configs is required. Use [{'timeframes': ['all'], ...}] to apply same config to all data."
            )

        # Store timeframe-specific configurations
        self.timeframe_configs = timeframe_configs

        # No hardcoded defaults - all values come from configs

    def _get_config_for_timeframe(self, timeframe: str) -> dict[str, Any]:
        """
        Get configuration for a specific timeframe.

        Args:
            timeframe: The timeframe to get configuration for

        Returns:
            Configuration dictionary for the timeframe
        """
        # First check for "all" timeframe config
        for tf_config in self.timeframe_configs:
            if "all" in tf_config.get("timeframes", []):
                config = tf_config.copy()
                config.pop("timeframes", None)
                return config

        # Then check for specific timeframe config
        for tf_config in self.timeframe_configs:
            if timeframe in tf_config.get("timeframes", []):
                config = tf_config.copy()
                config.pop("timeframes", None)
                return config

        # No config found - this should not happen with proper validation
        raise ValueError(f"No configuration found for timeframe '{timeframe}'. Check your timeframe_configs.")

    def process(self, data: PolarsDataFrame | PandasDataFrame) -> PolarsDataFrame:
        """
        Calculate all Strat indicators for OHLC data.

        Performs comprehensive technical analysis including swing point detection, market structure
        analysis, pattern recognition, and gap detection. Calculations run sequentially due to
        data dependencies between indicators.

        Args:
            data: Input DataFrame with OHLC data

        Returns:
            DataFrame with all Strat indicators added
        """
        self.validate_input(data)

        # Convert to Polars if needed
        df = self._convert_to_polars(data)

        # Check if data has timeframe column for per-timeframe processing
        if "timeframe" in df.columns:
            # Check if "all" timeframe is configured
            has_all_config = any("all" in tf_config.get("timeframes", []) for tf_config in self.timeframe_configs)

            if has_all_config:
                # Process all data with the "all" configuration
                all_config = self._get_config_for_timeframe("any_timeframe_will_get_all_config")
                df = self._process_single_timeframe(df, all_config)
            else:
                # Process each timeframe group with its specific configuration
                timeframe_groups = df.partition_by("timeframe", as_dict=True)
                processed_groups = []

                for timeframe_key, timeframe_data in timeframe_groups.items():
                    # Extract timeframe string from tuple key
                    timeframe = timeframe_key[0] if isinstance(timeframe_key, tuple) else timeframe_key

                    # Get config for this timeframe
                    tf_config = self._get_config_for_timeframe(timeframe)

                    # Process this timeframe with its specific config
                    processed_data = self._process_single_timeframe(timeframe_data, tf_config)
                    processed_groups.append(processed_data)

                # Combine all processed groups
                df = processed_groups[0]
                for group in processed_groups[1:]:
                    df = df.vstack(group)

                # Sort by original order (symbol, timeframe, timestamp)
                sort_cols = []
                if "symbol" in df.columns:
                    sort_cols.append("symbol")
                sort_cols.extend(["timeframe", "timestamp"])
                df = df.sort(sort_cols)
        else:
            # No timeframe column - must have "all" configuration
            has_all_config = any("all" in tf_config.get("timeframes", []) for tf_config in self.timeframe_configs)
            if not has_all_config:
                raise ValueError("Data without timeframe column requires an 'all' timeframe configuration.")

            all_config = self._get_config_for_timeframe("any_timeframe_will_get_all_config")
            df = self._process_single_timeframe(df, all_config)

        return df

    def _process_single_timeframe(self, data: PolarsDataFrame, config: dict[str, Any]) -> PolarsDataFrame:
        """
        Process indicators for a single timeframe using specific configuration.

        Optimized implementation that combines independent calculations to minimize
        DataFrame passes and maximize vectorization performance.

        Args:
            data: DataFrame for a single timeframe
            config: Configuration to use for this timeframe

        Returns:
            DataFrame with indicators calculated using the specified config
        """
        # Get configuration values with defaults from Pydantic models
        from .schemas import GapDetectionConfig, SwingPointsConfig

        swing_config = config.get("swing_points", {})
        swing_defaults = SwingPointsConfig()
        swing_window = swing_config.get("window", swing_defaults.window)
        swing_threshold = swing_config.get("threshold", swing_defaults.threshold)

        gap_config = config.get("gap_detection", {})
        gap_defaults = GapDetectionConfig()
        gap_threshold = gap_config.get("threshold", gap_defaults.threshold)

        # Vectorized indicator calculation with dependency grouping
        df = data.clone()

        # Group 1: Swing points (no dependencies)
        df = self._calculate_swing_points(df, swing_window, swing_threshold)

        # Group 2: Market structure (depends on swing points)
        df = self._calculate_market_structure(df)

        # Group 3: Combined independent calculations (price analysis, ATH/ATL, gap analysis)
        # These have no dependencies and can be calculated in parallel
        df = self._calculate_independent_indicators(df, gap_threshold)

        # Group 4: Strat patterns (depends on continuity which is calculated first)
        df = self._calculate_strat_patterns(df, gap_threshold)

        return df

    def _calculate_independent_indicators(self, data: PolarsDataFrame, gap_threshold: float = None) -> PolarsDataFrame:
        """
        Calculate all independent indicators in a single vectorized operation.

        Combines price analysis, ATH/ATL, and gap analysis calculations into
        a single DataFrame pass for optimal performance.

        Args:
            data: Input DataFrame with OHLC data
            gap_threshold: Gap detection threshold (extracted from config if None)

        Returns:
            DataFrame with independent indicators added
        """
        # Extract gap threshold from instance if not provided
        if gap_threshold is None:
            from .schemas import GapDetectionConfig

            config = self._get_config_for_timeframe("all")
            gap_config = config.get("gap_detection", {})
            gap_defaults = GapDetectionConfig()
            gap_threshold = gap_config.get("threshold", gap_defaults.threshold)

        df = data.clone()

        # Combine all independent calculations in a single with_columns operation
        df = df.with_columns(
            [
                # Price analysis indicators
                (((col("high") - col("close")) / (col("high") - col("low"))) * 100).alias("percent_close_from_high"),
                (((col("close") - col("low")) / (col("high") - col("low"))) * 100).alias("percent_close_from_low"),
                # ATH/ATL indicators
                col("high").cum_max().alias("ath"),
                col("low").cum_min().alias("atl"),
                # Gap analysis indicators
                (col("open") > col("high").shift(1)).fill_null(False).alias("gap_up"),
                (col("open") < col("low").shift(1)).fill_null(False).alias("gap_down"),
            ]
        )

        # Add derived indicators that depend on the above calculations
        df = df.with_columns(
            [
                # ATH/ATL new markers
                (col("high") == col("ath")).fill_null(False).alias("new_ath"),
                (col("low") == col("atl")).fill_null(False).alias("new_atl"),
                # Combined gapper indicator
                (col("gap_up") | col("gap_down")).fill_null(False).alias("gapper"),
            ]
        )

        return df

    def _calculate_swing_points(
        self, data: PolarsDataFrame, swing_window: int = None, swing_threshold: float = None
    ) -> PolarsDataFrame:
        """
        Detect swing highs and lows using vectorized operations.

        Uses rolling window analysis to identify local extremes in price action.
        Swing points are detected when price moves beyond the percentage threshold
        (e.g., 5% = 5.0) compared to the previous rolling extreme.
        """
        # Extract config from instance if not provided (for direct method calls)
        if swing_window is None or swing_threshold is None:
            from .schemas import SwingPointsConfig

            # Get config for "all" timeframes (fallback for direct method calls)
            config = self._get_config_for_timeframe("all")
            swing_config = config.get("swing_points", {})
            swing_defaults = SwingPointsConfig()

            if swing_window is None:
                swing_window = swing_config.get("window", swing_defaults.window)
            if swing_threshold is None:
                swing_threshold = swing_config.get("threshold", swing_defaults.threshold)

        df = data.clone()

        # Calculate rolling min/max for swing detection
        df = df.with_columns(
            [
                # Rolling highs and lows for swing detection
                col("high").rolling_max(window_size=swing_window, center=True).alias("roll_high_max"),
                col("high").rolling_max(window_size=swing_window, center=False).shift(1).alias("roll_high_max_prev"),
                col("low").rolling_min(window_size=swing_window, center=True).alias("roll_low_min"),
                col("low").rolling_min(window_size=swing_window, center=False).shift(1).alias("roll_low_min_prev"),
            ]
        )

        # Identify swing highs and lows using percentage threshold
        df = df.with_columns(
            [
                # Boolean indicators for new swing points (fill nulls with False)
                (
                    (col("high") == col("roll_high_max"))
                    & (col("high") > col("roll_high_max_prev") * (1 + swing_threshold / 100))
                )
                .fill_null(False)
                .alias("new_swing_high"),
                (
                    (col("low") == col("roll_low_min"))
                    & (col("low") < col("roll_low_min_prev") * (1 - swing_threshold / 100))
                )
                .fill_null(False)
                .alias("new_swing_low"),
            ]
        )

        # Create forward-filled price columns for swing levels
        df = df.with_columns(
            [
                # swing_high: shows current high when new_swing_high is True, forward-fills
                when(col("new_swing_high"))
                .then(col("high"))
                .otherwise(None)
                .fill_null(strategy="forward")
                .alias("swing_high"),
                # swing_low: shows current low when new_swing_low is True, forward-fills
                when(col("new_swing_low"))
                .then(col("low"))
                .otherwise(None)
                .fill_null(strategy="forward")
                .alias("swing_low"),
            ]
        )

        # Create boolean indicators for new pivot points (same as swing points)
        df = df.with_columns(
            [
                col("new_swing_high").alias("new_pivot_high"),
                col("new_swing_low").alias("new_pivot_low"),
            ]
        )

        # Create forward-filled price columns for pivot levels
        df = df.with_columns(
            [
                # pivot_high: shows current high when new_pivot_high is True, forward-fills
                when(col("new_pivot_high"))
                .then(col("high"))
                .otherwise(None)
                .fill_null(strategy="forward")
                .alias("pivot_high"),
                # pivot_low: shows current low when new_pivot_low is True, forward-fills
                when(col("new_pivot_low"))
                .then(col("low"))
                .otherwise(None)
                .fill_null(strategy="forward")
                .alias("pivot_low"),
            ]
        )

        # Clean up temporary columns
        df = df.drop(["roll_high_max", "roll_high_max_prev", "roll_low_min", "roll_low_min_prev"])

        return df

    def _calculate_market_structure(self, data: PolarsDataFrame) -> PolarsDataFrame:
        """
        Classify market structure patterns: HH, HL, LH, LL.

        Analyzes sequence of swing points to determine market structure.
        """
        df = data.clone()

        # Get previous swing high and low values for comparison
        df = df.with_columns(
            [
                col("pivot_high").shift(1).alias("prev_swing_high"),
                col("pivot_low").shift(1).alias("prev_swing_low"),
            ]
        )

        # Create boolean indicators for new market structure
        df = df.with_columns(
            [
                # Boolean: True when current swing creates a new higher high (fill nulls with False)
                ((col("new_swing_high")) & (col("pivot_high") > col("prev_swing_high")))
                .fill_null(False)
                .alias("new_higher_high"),
                # Boolean: True when current swing creates a new lower high (fill nulls with False)
                ((col("new_swing_high")) & (col("pivot_high") < col("prev_swing_high")))
                .fill_null(False)
                .alias("new_lower_high"),
                # Boolean: True when current swing creates a new higher low (fill nulls with False)
                ((col("new_swing_low")) & (col("pivot_low") > col("prev_swing_low")))
                .fill_null(False)
                .alias("new_higher_low"),
                # Boolean: True when current swing creates a new lower low (fill nulls with False)
                ((col("new_swing_low")) & (col("pivot_low") < col("prev_swing_low")))
                .fill_null(False)
                .alias("new_lower_low"),
            ]
        )

        # Create forward-filled price columns for market structure levels
        df = df.with_columns(
            [
                # higher_high: shows current high when new_higher_high is True, forward-fills
                when(col("new_higher_high"))
                .then(col("high"))
                .otherwise(None)
                .fill_null(strategy="forward")
                .alias("higher_high"),
                # lower_high: shows current high when new_lower_high is True, forward-fills
                when(col("new_lower_high"))
                .then(col("high"))
                .otherwise(None)
                .fill_null(strategy="forward")
                .alias("lower_high"),
                # higher_low: shows current low when new_higher_low is True, forward-fills
                when(col("new_higher_low"))
                .then(col("low"))
                .otherwise(None)
                .fill_null(strategy="forward")
                .alias("higher_low"),
                # lower_low: shows current low when new_lower_low is True, forward-fills
                when(col("new_lower_low"))
                .then(col("low"))
                .otherwise(None)
                .fill_null(strategy="forward")
                .alias("lower_low"),
            ]
        )

        # Clean up temporary columns
        df = df.drop(["prev_swing_high", "prev_swing_low"])

        return df

    def _calculate_strat_patterns(self, data: PolarsDataFrame, gap_threshold: float = None) -> PolarsDataFrame:
        """
        Calculate Strat-specific patterns: continuity, in_force, scenario, signal,
        hammer, shooter, kicker, f23, pmg, motherbar_problems.
        """
        # Extract gap threshold from instance if not provided
        if gap_threshold is None:
            from .schemas import GapDetectionConfig

            config = self._get_config_for_timeframe("all")
            gap_config = config.get("gap_detection", {})
            gap_defaults = GapDetectionConfig()
            gap_threshold = gap_config.get("threshold", gap_defaults.threshold)

        df = data.clone()

        # Calculate basic Strat patterns - step 1: continuity first
        df = df.with_columns(
            [
                # Continuity: Single bar open/close relationship (corrected from setup_processor.py)
                when(col("close") > col("open"))
                .then(1)
                .when(col("close") < col("open"))
                .then(0)
                .otherwise(-1)
                .alias("continuity")
            ]
        )

        # Calculate basic Strat patterns - step 2: patterns that depend on continuity
        df = df.with_columns(
            [
                # In Force: Breakout of previous bar's range (corrected from setup_processor.py)
                # Note: F23 conditions will be added after F23 pattern is calculated
                (
                    ((col("continuity") == 1) & (col("close") > col("high").shift(1)))
                    | ((col("continuity") == 0) & (col("close") < col("low").shift(1)))
                )
                .fill_null(False)
                .alias("in_force_base"),
                # Scenario classification using strings ("1", "2U", "2D", "3")
                # Corrected to match setup_processor.py logic with proper priority order
                when((col("high") > col("high").shift(1)) & (col("low") >= col("low").shift(1)))
                .then(lit("2U"))
                .when((col("low") < col("low").shift(1)) & (col("high") <= col("high").shift(1)))
                .then(lit("2D"))
                .when((col("high") > col("high").shift(1)) & (col("low") < col("low").shift(1)))
                .then(lit("3"))
                .when((col("high") <= col("high").shift(1)) & (col("low") >= col("low").shift(1)))
                .then(lit("1"))
                .otherwise(None)
                .alias("scenario"),
                # Hammer: Rob Smith's original approach - long lower shadow, small body near top
                (
                    ((col("high") - col("low")) > 3 * (col("open") - col("close")).abs())  # Total range > 3x body size
                    & (((col("close") - col("low")) / (0.001 + col("high") - col("low"))) > 0.6)  # Close in upper 40%
                    & (((col("open") - col("low")) / (0.001 + col("high") - col("low"))) > 0.6)  # Open in upper 40%
                )
                .fill_null(False)
                .alias("hammer"),
                # Shooter: Rob Smith's original approach - long upper shadow, small body near bottom
                (
                    ((col("high") - col("low")) > 3 * (col("open") - col("close")).abs())  # Total range > 3x body size
                    & (((col("high") - col("close")) / (0.001 + col("high") - col("low"))) > 0.6)  # Close in lower 40%
                    & (((col("high") - col("open")) / (0.001 + col("high") - col("low"))) > 0.6)  # Open in lower 40%
                )
                .fill_null(False)
                .alias("shooter"),
            ]
        )

        # Add advanced Strat patterns
        df = self._calculate_advanced_patterns(df, gap_threshold)

        # Add signal pattern detection
        df = self._calculate_signals(df)

        return df

    def _calculate_signals(self, data: PolarsDataFrame) -> PolarsDataFrame:
        """
        Detect multi-bar signal patterns and create rich signal objects.

        Signals are collections of sequential scenarios that match specific
        patterns defined in the SIGNALS dictionary. Creates SignalMetadata
        objects with full trading context.
        """
        df = data.clone()

        # Create rolling windows of scenarios for pattern matching
        df = df.with_columns(
            [
                col("scenario").shift(2).alias("scenario_2"),
                col("scenario").shift(1).alias("scenario_1"),
                col("scenario").alias("scenario_0"),
            ]
        )

        # Build pattern strings for different bar counts
        # 2-bar patterns: scenario_1-scenario_0
        df = df.with_columns(
            [
                when(col("scenario_1").is_not_null() & col("scenario_0").is_not_null())
                .then(concat_str([col("scenario_1"), col("scenario_0")], separator="-"))
                .otherwise(None)
                .alias("pattern_2bar")
            ]
        )

        # 3-bar patterns: scenario_2-scenario_1-scenario_0
        df = df.with_columns(
            [
                when(
                    col("scenario_2").is_not_null() & col("scenario_1").is_not_null() & col("scenario_0").is_not_null()
                )
                .then(concat_str([col("scenario_2"), col("scenario_1"), col("scenario_0")], separator="-"))
                .otherwise(None)
                .alias("pattern_3bar")
            ]
        )

        # Initialize signal columns
        signal_values = []
        type_values = []
        bias_values = []
        signal_objects = []
        signal_json = []

        # Get scenarios as Python lists for pattern matching
        pattern_2bar = df["pattern_2bar"].to_list()
        pattern_3bar = df["pattern_3bar"].to_list()

        # Process each row to find matching signals and create signal objects
        for i in range(len(df)):
            signal_found = None
            signal_type = None
            signal_bias = None
            signal_obj = None

            # Check 3-bar patterns first (more specific)
            if pattern_3bar[i] and pattern_3bar[i] in SIGNALS:
                signal_found = pattern_3bar[i]
                signal_type = SIGNALS[pattern_3bar[i]]["category"]
                signal_bias = SIGNALS[pattern_3bar[i]]["bias"]

                # Create signal object
                signal_obj = self._create_signal_object(df, i, signal_found, signal_type, signal_bias)

            # Check 2-bar patterns
            elif pattern_2bar[i] and pattern_2bar[i] in SIGNALS:
                signal_found = pattern_2bar[i]
                signal_type = SIGNALS[pattern_2bar[i]]["category"]
                signal_bias = SIGNALS[pattern_2bar[i]]["bias"]

                # Create signal object
                signal_obj = self._create_signal_object(df, i, signal_found, signal_type, signal_bias)

            signal_values.append(signal_found)
            type_values.append(signal_type)
            bias_values.append(signal_bias)
            signal_objects.append(signal_obj)
            signal_json.append(signal_obj.to_json() if signal_obj else None)

        # Add signal columns using Polars Series
        from polars import Series

        df = df.with_columns(
            [
                Series("signal", signal_values),
                Series("type", type_values),
                Series("bias", bias_values),
                Series("signal_json", signal_json),
            ]
        )

        # Store signal objects as a separate attribute for programmatic access
        # (Polars doesn't handle complex objects well in DataFrames)
        df = df.with_row_index("__temp_idx")
        signal_obj_map = {i: signal_objects[i] for i in range(len(signal_objects))}

        # Add a method to retrieve signal objects by index
        def get_signal_object(row_index: int):
            return signal_obj_map.get(row_index)

        # Store the mapping as metadata (this is a workaround for Polars limitations)
        df.attrs = getattr(df, "attrs", {})
        df.attrs["signal_objects"] = signal_obj_map
        df.attrs["get_signal_object"] = get_signal_object

        # Remove temporary index
        df = df.drop("__temp_idx")

        # Clean up temporary columns
        df = df.drop(["scenario_2", "scenario_1", "scenario_0", "pattern_2bar", "pattern_3bar"])

        return df

    def _create_signal_object(self, df, index, pattern, category, bias):
        """Factory method to create signal objects."""
        from datetime import datetime

        from .signals import SignalBias, SignalCategory, SignalMetadata

        # Skip if bias is None (some patterns need special handling)
        if bias is None:
            return None

        # Get pattern configuration
        config = SIGNALS[pattern]
        bar_count = config["bar_count"]

        # Determine bar indices
        entry_bar_index = index
        trigger_bar_index = index - config["trigger_bar_offset"]

        # Ensure we have enough data for the trigger bar
        if trigger_bar_index < 0:
            return None

        # Get trigger bar data for entry/stop prices
        trigger_row = df.row(trigger_bar_index)

        # Find high, low columns (assuming standard OHLC structure)
        high_col_idx = df.columns.index("high")
        low_col_idx = df.columns.index("low")
        timestamp_col_idx = df.columns.index("timestamp")

        trigger_high = float(trigger_row[high_col_idx])
        trigger_low = float(trigger_row[low_col_idx])

        # Calculate entry and stop based on bias
        if bias == "long":
            entry_price = trigger_high
            stop_price = trigger_low
        else:  # short
            entry_price = trigger_low
            stop_price = trigger_high

        # Calculate target for reversals
        target_price = None
        target_bar_index = None

        if category == "reversal":
            target_bar_offset = config.get("target_bar_offset")
            if target_bar_offset is not None:
                target_bar_index = index - target_bar_offset

                if target_bar_index >= 0:
                    target_row = df.row(target_bar_index)
                    target_high = float(target_row[high_col_idx])
                    target_low = float(target_row[low_col_idx])

                    if bias == "long":
                        target_price = target_high
                    else:
                        target_price = target_low

        # Get current row data
        current_row = df.row(index)
        timestamp = current_row[timestamp_col_idx]

        # Handle timestamp conversion
        if hasattr(timestamp, "to_pydatetime"):
            timestamp = timestamp.to_pydatetime()
        elif not isinstance(timestamp, datetime):
            timestamp = datetime.now()

        # Get symbol if available
        symbol = None
        if "symbol" in df.columns:
            symbol_col_idx = df.columns.index("symbol")
            symbol = current_row[symbol_col_idx]

        # Create signal metadata object
        return SignalMetadata(
            pattern=pattern,
            category=SignalCategory[category.upper()],
            bias=SignalBias[bias.upper()],
            bar_count=bar_count,
            entry_bar_index=entry_bar_index,
            trigger_bar_index=trigger_bar_index,
            target_bar_index=target_bar_index,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            timestamp=timestamp,
            symbol=symbol,
            timeframe=getattr(self, "timeframe", None),
        )

    def _calculate_advanced_patterns(self, data: PolarsDataFrame, gap_threshold: float = None) -> PolarsDataFrame:
        """Calculate advanced Strat patterns: kicker, f23, pmg, motherbar_problems."""
        # Extract gap threshold from instance if not provided
        if gap_threshold is None:
            from .schemas import GapDetectionConfig

            config = self._get_config_for_timeframe("all")
            gap_config = config.get("gap_detection", {})
            gap_defaults = GapDetectionConfig()
            gap_threshold = gap_config.get("threshold", gap_defaults.threshold)

        df = data.clone()

        # Advanced gapper: Gap detection using percentage-based thresholds for all asset classes
        df = df.with_columns(
            [
                # Gap up = 1: Opening above previous high by gap_threshold percentage
                when(col("open") > (col("high").shift(1) * (1 + gap_threshold)))
                .then(1)
                # Gap down = 0: Opening below previous low by gap_threshold percentage
                .when(col("open") < (col("low").shift(1) * (1 - gap_threshold)))
                .then(0)
                .otherwise(None)
                .alias("advanced_gapper")
            ]
        )

        # Kicker: Continuity reversal with gap (corrected from setup_processor.py)
        df = df.with_columns(
            [
                # Bullish kicker: continuity1=0 & advanced_gapper=1 & continuity=1 (bearish to bullish with gap up)
                when((col("continuity").shift(1) == 0) & (col("advanced_gapper") == 1) & (col("continuity") == 1))
                .then(1)  # Bullish kicker = 1
                # Bearish kicker: continuity1=1 & advanced_gapper=0 & continuity=0 (bullish to bearish with gap down)
                .when((col("continuity").shift(1) == 1) & (col("advanced_gapper") == 0) & (col("continuity") == 0))
                .then(0)  # Bearish kicker = 0
                .otherwise(None)  # No kicker = None (converted to null)
                .alias("kicker")
            ]
        )

        # F23 Pattern: Failed 2 goes 3 (corrected from setup_processor.py)
        # First calculate temporary trigger price (midpoint of previous bar)
        df = df.with_columns(
            [(col("high").shift(1) - ((col("high").shift(1) - col("low").shift(1)) / 2)).alias("temp_f23_trigger")]
        )

        df = df.with_columns(
            [
                # F23D: 2U scenario but close below midpoint trigger
                when(
                    (col("high") > col("high").shift(1))
                    & (col("low") >= col("low").shift(1))
                    & (col("close") < col("temp_f23_trigger"))
                )
                .then(lit("F23D"))
                # F23U: 2D scenario but close above midpoint trigger
                .when(
                    (col("low") < col("low").shift(1))
                    & (col("high") <= col("high").shift(1))
                    & (col("close") > col("temp_f23_trigger"))
                )
                .then(lit("F23U"))
                .otherwise(None)
                .alias("f23x")
            ]
        )

        # Combined F23 pattern (boolean)
        df = df.with_columns(
            [((col("f23x") == lit("F23U")) | (col("f23x") == lit("F23D"))).fill_null(False).alias("f23")]
        )

        # f23_trigger: Only show trigger price when f23 pattern is True
        df = df.with_columns([when(col("f23")).then(col("temp_f23_trigger")).otherwise(None).alias("f23_trigger")])

        # Clean up temporary column
        df = df.drop(["temp_f23_trigger"])

        # PMG (Pivot Machine Gun): Vectorized cumulative tracking (10-50x faster than loops)
        # Calculate higher highs and lower lows first (using temp columns to avoid collision with market structure)
        df = df.with_columns(
            [
                (col("low") < col("low").shift(1)).cast(Int32).alias("pmg_temp_lower_low"),
                (col("high") > col("high").shift(1)).cast(Int32).alias("pmg_temp_higher_high"),
            ]
        )

        # Vectorized PMG calculation preserving exact temporal logic
        df = df.with_columns(
            [
                # Create directional signals: +1 for HH, -1 for LL, 0 for neither
                when(col("pmg_temp_higher_high") == 1)
                .then(lit(1))
                .when(col("pmg_temp_lower_low") == 1)
                .then(lit(-1))
                .otherwise(lit(0))
                .alias("pmg_direction"),
            ]
        )

        # Use cumulative logic to track PMG state changes
        # This preserves the exact temporal behavior of the original loop
        df = df.with_columns(
            [
                # Create reset groups - new group when direction changes or becomes 0
                ((col("pmg_direction") != col("pmg_direction").shift(1, fill_value=0)) | (col("pmg_direction") == 0))
                .cum_sum()
                .alias("pmg_group"),
            ]
        )

        # Calculate PMG values by group, maintaining cumulative count within each directional run
        df = df.with_columns(
            [
                when(col("pmg_direction") == 0)
                .then(lit(0))
                .otherwise(
                    col("pmg_direction")
                    * (col("pmg_direction") != 0).cum_sum().over("pmg_group")
                    * when(col("pmg_direction") != 0).then(1).otherwise(0)
                )
                .cast(Int32)
                .alias("pmg")
            ]
        )

        # Clean up temporary PMG columns
        df = df.drop(["pmg_temp_lower_low", "pmg_temp_higher_high", "pmg_direction", "pmg_group"])

        # Motherbar Problems: Vectorized tracking of when current bar has not broken out of mother bar range
        # A mother bar is the bar immediately before an inside bar (scenario 1)
        # Mother bar problems = True when current bar hasn't broken the mother bar's high or low

        # Step 1: Add row index and identify mother bars (bars immediately before inside bars)
        df = df.with_row_index("__row_idx")
        df = df.with_columns(
            [
                (col("scenario").shift(-1) == "1").alias("is_mother_bar"),
                # Get mother bar high/low values for when this bar becomes a mother bar
                when(col("scenario").shift(-1) == "1").then(col("high")).otherwise(None).alias("mother_high_candidate"),
                when(col("scenario").shift(-1) == "1").then(col("low")).otherwise(None).alias("mother_low_candidate"),
            ]
        )

        # Step 2: Create breakout detection and groups
        # We need to do this in steps to properly handle the temporal logic

        # First, forward-fill mother candidates to get current mother range
        df = df.with_columns(
            [
                col("mother_high_candidate").forward_fill().alias("temp_mother_high"),
                col("mother_low_candidate").forward_fill().alias("temp_mother_low"),
            ]
        )

        # Detect breakouts based on current mother range
        df = df.with_columns(
            [
                # Breakout occurs when current bar exceeds mother bar range AND mother range exists
                (
                    col("temp_mother_high").is_not_null()
                    & col("temp_mother_low").is_not_null()
                    & ((col("high") > col("temp_mother_high")) | (col("low") < col("temp_mother_low")))
                )
                .fill_null(False)
                .alias("is_breakout")
            ]
        )

        # Step 3: Create groups that reset after breakouts or establish new mother bars
        df = df.with_columns(
            [
                # New group starts when: new mother bar established OR breakout occurs
                (col("is_mother_bar") | col("is_breakout")).cum_sum().alias("mother_group")
            ]
        )

        # Step 4: Within each group, forward-fill mother values but stop after breakouts
        # Create a flag to indicate if we're past a breakout in the current group
        df = df.with_columns(
            [
                # Mark positions after breakout within each group
                col("is_breakout").cum_sum().over("mother_group").alias("breakouts_in_group")
            ]
        )

        df = df.with_columns(
            [
                # Only use mother values if no breakout has occurred in current group
                # AND we're not at the row that's establishing a new mother bar
                when((col("breakouts_in_group") == 0) & ~col("is_mother_bar"))
                .then(col("mother_high_candidate").forward_fill().over("mother_group"))
                .otherwise(None)
                .alias("active_mother_high"),
                when((col("breakouts_in_group") == 0) & ~col("is_mother_bar"))
                .then(col("mother_low_candidate").forward_fill().over("mother_group"))
                .otherwise(None)
                .alias("active_mother_low"),
            ]
        )

        # Step 6: Calculate motherbar problems
        df = df.with_columns(
            [
                # Motherbar problems = True when:
                # 1. Current bar is inside bar (scenario == "1"), OR
                # 2. Current bar is within active mother bar range (hasn't broken out)
                # BUT first bar is always False (no previous mother bar possible)
                when(col("__row_idx") == 0)  # First row
                .then(False)
                .otherwise(
                    (col("scenario") == "1")
                    | (
                        col("active_mother_high").is_not_null()
                        & col("active_mother_low").is_not_null()
                        & (col("high") <= col("active_mother_high"))
                        & (col("low") >= col("active_mother_low"))
                        & ~col("is_breakout")
                    )
                )
                .fill_null(False)
                .alias("motherbar_problems")
            ]
        )

        # Clean up temporary columns
        df = df.drop(
            [
                "__row_idx",
                "is_mother_bar",
                "mother_high_candidate",
                "mother_low_candidate",
                "temp_mother_high",
                "temp_mother_low",
                "is_breakout",
                "mother_group",
                "breakouts_in_group",
                "active_mother_high",
                "active_mother_low",
            ]
        )

        # Update in_force to include F23 conditions (final calculation after all patterns)
        # Handle case where in_force_base might not exist (if this method is called multiple times)
        if "in_force_base" in df.columns:
            df = df.with_columns(
                [
                    (
                        col("in_force_base")
                        | ((col("f23x") == "F23U") & (col("close") > col("f23_trigger")) & (col("continuity") == 1))
                        | ((col("f23x") == "F23D") & (col("close") < col("f23_trigger")) & (col("continuity") == 0))
                    )
                    .fill_null(False)
                    .alias("in_force")
                ]
            )
            # Clean up temporary columns
            df = df.drop(["in_force_base"])
        else:
            # in_force_base already processed, update existing in_force column
            df = df.with_columns(
                [
                    (
                        col("in_force")
                        | ((col("f23x") == "F23U") & (col("close") > col("f23_trigger")) & (col("continuity") == 1))
                        | ((col("f23x") == "F23D") & (col("close") < col("f23_trigger")) & (col("continuity") == 0))
                    )
                    .fill_null(False)
                    .alias("in_force")
                ]
            )

        return df

    def validate_input(self, data: PolarsDataFrame | PandasDataFrame) -> None:
        """
        Validate input data format.

        Raises:
            ValueError: If data format is invalid, with specific error message
        """
        df = self._convert_to_polars(data)

        # Check required columns
        required_cols = ["timestamp", "open", "high", "low", "close"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Get minimum swing window from any configuration
        min_swing_window = 5  # default
        for tf_config in self.timeframe_configs:
            swing_config = tf_config.get("swing_points", {})
            window = swing_config.get("window", 5)
            min_swing_window = min(min_swing_window, window)

        # Check for minimum data points for swing analysis
        if len(df) < min_swing_window * 2:
            raise ValueError(
                f"Insufficient data: {len(df)} rows provided, need at least {min_swing_window * 2} rows "
                f"for swing analysis with window={min_swing_window}"
            )

        # Verify price data integrity
        validations = df.select(
            [
                (col("high") >= col("low")).all().alias("high_gte_low"),
                (col("high") >= col("open")).all().alias("high_gte_open"),
                (col("high") >= col("close")).all().alias("high_gte_close"),
                (col("low") <= col("open")).all().alias("low_lte_open"),
                (col("low") <= col("close")).all().alias("low_lte_close"),
            ]
        )

        validation_results = validations.to_numpy().flatten()
        validation_names = ["high >= low", "high >= open", "high >= close", "low <= open", "low <= close"]

        failed_validations = [
            name for name, result in zip(validation_names, validation_results, strict=False) if not result
        ]
        if failed_validations:
            raise ValueError(f"Invalid price data: failed validations: {failed_validations}")
