"""
Vectorized Strat technical indicators implementation.

This module provides comprehensive Strat pattern analysis with high-performance
vectorized calculations using Polars operations.
"""

from pandas import DataFrame as PandasDataFrame
from polars import DataFrame as PolarsDataFrame
from polars import Float64, Int32, String, col, concat_str, lit, when

from .base import Component
from .schemas import IndicatorsConfig, TimeframeItemConfig
from .signals import SIGNALS


class Indicators(Component):
    """
    Vectorized implementation of all Strat technical indicators.

    Provides comprehensive technical analysis including swing point detection, market structure
    analysis, pattern recognition, and gap detection optimized for TheStrat methodology.
    """

    # Type hints for commonly accessed attributes
    config: IndicatorsConfig

    def __init__(self, config: IndicatorsConfig):
        """
        Initialize indicators component with validated configuration.

        Args:
            config: Validated IndicatorsConfig containing per-timeframe configurations
        """
        super().__init__()

        # Store the validated Pydantic config
        self.config = config

    def _get_config_for_timeframe(self, timeframe: str) -> TimeframeItemConfig:
        """
        Get configuration for a specific timeframe.

        Args:
            timeframe: The timeframe to get configuration for

        Returns:
            TimeframeItemConfig for the timeframe
        """
        # First check for "all" timeframe config
        for tf_config in self.config.timeframe_configs:
            if "all" in tf_config.timeframes:
                return tf_config

        # Then check for specific timeframe config
        for tf_config in self.config.timeframe_configs:
            if timeframe in tf_config.timeframes:
                return tf_config

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

        # Ensure optional schema columns are present
        if "symbol" not in df.columns:
            df = df.with_columns(lit(None, dtype=String).alias("symbol"))

        # Check if data has timeframe column for per-timeframe processing
        if "timeframe" in df.columns:
            # Check if "all" timeframe is configured
            has_all_config = any("all" in tf_config.timeframes for tf_config in self.config.timeframe_configs)

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
            has_all_config = any("all" in tf_config.timeframes for tf_config in self.config.timeframe_configs)
            if not has_all_config:
                raise ValueError("Data without timeframe column requires an 'all' timeframe configuration.")

            all_config = self._get_config_for_timeframe("any_timeframe_will_get_all_config")
            df = self._process_single_timeframe(df, all_config)

        return df

    def _process_single_timeframe(self, data: PolarsDataFrame, config: TimeframeItemConfig) -> PolarsDataFrame:
        """
        Process indicators for a single timeframe using specific configuration.

        Optimized implementation that combines independent calculations to minimize
        DataFrame passes and maximize vectorization performance.

        Args:
            data: DataFrame for a single timeframe
            config: TimeframeItemConfig to use for this timeframe

        Returns:
            DataFrame with indicators calculated using the specified config
        """
        # Vectorized indicator calculation with dependency grouping
        df = data.clone()

        # Group 1: Market structure (no dependencies)
        df = self._calculate_market_structure(df, config)

        # Group 3: Combined independent calculations (price analysis, ATH/ATL, gap analysis)
        # These have no dependencies and can be calculated in parallel
        df = self._calculate_independent_indicators(df, config)

        # Group 4: Strat patterns (depends on continuity which is calculated first)
        df = self._calculate_strat_patterns(df, config)

        # Round all numeric indicator outputs to 5 decimal places for consistent storage
        df = self._round_numeric_outputs(df)

        return df

    def _calculate_independent_indicators(self, data: PolarsDataFrame, config: TimeframeItemConfig) -> PolarsDataFrame:
        """
        Calculate all independent indicators in a single vectorized operation.

        Combines price analysis, ATH/ATL, and gap analysis calculations into
        a single DataFrame pass for optimal performance.

        Args:
            data: Input DataFrame with OHLC data
            config: TimeframeItemConfig containing configuration

        Returns:
            DataFrame with independent indicators added
        """
        # Get gap detection configuration from the passed config
        gap_detection_config = config.gap_detection
        if gap_detection_config is None:
            from .schemas import GapDetectionConfig

            gap_detection_config = GapDetectionConfig()  # Use defaults

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
            ]
        )

        # Add derived indicators that depend on the above calculations
        df = df.with_columns(
            [
                # ATH/ATL new markers
                (col("high") == col("ath")).fill_null(False).alias("new_ath"),
                (col("low") == col("atl")).fill_null(False).alias("new_atl"),
            ]
        )

        return df

    def _calculate_market_structure(self, data: PolarsDataFrame, config: TimeframeItemConfig) -> PolarsDataFrame:
        """
        Detect market structure patterns: HH, HL, LH, LL using vectorized operations.

        Combines swing point detection with immediate market structure classification.
        Uses rolling windows to identify local highs/lows, then classifies them based on
        comparison with previous structure points.
        """
        # Get swing points configuration from the passed config
        swing_points_config = config.swing_points
        if swing_points_config is None:
            from .schemas import SwingPointsConfig

            swing_points_config = SwingPointsConfig()  # Use defaults

        swing_window = swing_points_config.window
        swing_threshold = swing_points_config.threshold

        df = data.clone()

        # Performance safeguard: Handle datasets too small for analysis
        min_required_rows = 2 * swing_window + 1
        if len(df) < min_required_rows:
            return df.with_columns(
                [
                    lit(None, dtype=Float64).alias("higher_high"),
                    lit(None, dtype=Float64).alias("lower_high"),
                    lit(None, dtype=Float64).alias("higher_low"),
                    lit(None, dtype=Float64).alias("lower_low"),
                ]
            )

        # Add row index for boundary checks
        df = df.with_row_index("__row_idx")

        # Detect local highs and lows using rolling windows
        df = df.with_columns(
            [
                col("high").rolling_max(window_size=2 * swing_window + 1, center=True).alias("window_high_max"),
                col("low").rolling_min(window_size=2 * swing_window + 1, center=True).alias("window_low_min"),
            ]
        )

        # Identify potential structural points (local extremes)
        df = df.with_columns(
            [
                (
                    (col("__row_idx") >= swing_window)
                    & (col("__row_idx") < (len(df) - swing_window))
                    & (col("high") == col("window_high_max"))
                ).alias("is_local_high"),
                (
                    (col("__row_idx") >= swing_window)
                    & (col("__row_idx") < (len(df) - swing_window))
                    & (col("low") == col("window_low_min"))
                ).alias("is_local_low"),
            ]
        )

        # Apply threshold filtering if specified
        if swing_threshold > 0:
            df = df.with_columns(
                [
                    when(col("is_local_high")).then(col("high")).otherwise(None).alias("high_candidates"),
                    when(col("is_local_low")).then(col("low")).otherwise(None).alias("low_candidates"),
                ]
            )

            df = df.with_columns(
                [
                    col("high_candidates").fill_null(strategy="forward").alias("ref_high"),
                    col("low_candidates").fill_null(strategy="forward").alias("ref_low"),
                ]
            )

            df = df.with_columns(
                [
                    when(col("is_local_high") & col("ref_high").shift(1).is_not_null())
                    .then(((col("high") - col("ref_high").shift(1)).abs() / col("ref_high").shift(1) * 100))
                    .otherwise(100.0)
                    .alias("high_pct_change"),
                    when(col("is_local_low") & col("ref_low").shift(1).is_not_null())
                    .then(((col("low") - col("ref_low").shift(1)).abs() / col("ref_low").shift(1) * 100))
                    .otherwise(100.0)
                    .alias("low_pct_change"),
                ]
            )

            df = df.with_columns(
                [
                    (col("is_local_high") & (col("high_pct_change") >= swing_threshold)).alias("valid_high"),
                    (col("is_local_low") & (col("low_pct_change") >= swing_threshold)).alias("valid_low"),
                ]
            )
        else:
            df = df.with_columns(
                [
                    col("is_local_high").alias("valid_high"),
                    col("is_local_low").alias("valid_low"),
                ]
            )

        # Extract valid structural high/low values
        df = df.with_columns(
            [
                when(col("valid_high")).then(col("high")).otherwise(None).alias("structural_high_values"),
                when(col("valid_low")).then(col("low")).otherwise(None).alias("structural_low_values"),
            ]
        )

        # Get previous structural values for comparison
        df = df.with_columns(
            [
                col("structural_high_values").fill_null(strategy="forward").shift(1).alias("prev_structural_high"),
                col("structural_low_values").fill_null(strategy="forward").shift(1).alias("prev_structural_low"),
            ]
        )

        # Classify market structure and create forward-filled columns
        df = df.with_columns(
            [
                # Higher High: current structural high > previous structural high
                when(
                    col("valid_high")
                    & col("prev_structural_high").is_not_null()
                    & (col("high") > col("prev_structural_high"))
                )
                .then(col("high"))
                .otherwise(None)
                .fill_null(strategy="forward")
                .alias("higher_high"),
                # Lower High: current structural high < previous structural high
                when(
                    col("valid_high")
                    & col("prev_structural_high").is_not_null()
                    & (col("high") < col("prev_structural_high"))
                )
                .then(col("high"))
                .otherwise(None)
                .fill_null(strategy="forward")
                .alias("lower_high"),
                # Higher Low: current structural low > previous structural low
                when(
                    col("valid_low")
                    & col("prev_structural_low").is_not_null()
                    & (col("low") > col("prev_structural_low"))
                )
                .then(col("low"))
                .otherwise(None)
                .fill_null(strategy="forward")
                .alias("higher_low"),
                # Lower Low: current structural low < previous structural low
                when(
                    col("valid_low")
                    & col("prev_structural_low").is_not_null()
                    & (col("low") < col("prev_structural_low"))
                )
                .then(col("low"))
                .otherwise(None)
                .fill_null(strategy="forward")
                .alias("lower_low"),
            ]
        )

        # Clean up temporary columns
        temp_columns = [
            "__row_idx",
            "window_high_max",
            "window_low_min",
            "is_local_high",
            "is_local_low",
            "valid_high",
            "valid_low",
            "structural_high_values",
            "structural_low_values",
            "prev_structural_high",
            "prev_structural_low",
        ]

        if swing_threshold > 0:
            temp_columns.extend(
                ["high_candidates", "low_candidates", "ref_high", "ref_low", "high_pct_change", "low_pct_change"]
            )

        existing_temp_columns = [col for col in temp_columns if col in df.columns]
        if existing_temp_columns:
            df = df.drop(existing_temp_columns)

        return df

    def _calculate_strat_patterns(self, data: PolarsDataFrame, config: TimeframeItemConfig) -> PolarsDataFrame:
        """
        Calculate Strat-specific patterns: continuity, in_force, scenario, signal,
        hammer, shooter, kicker, f23, pmg, motherbar_problems.
        """
        # Get gap detection configuration from the passed config
        gap_detection_config = config.gap_detection
        if gap_detection_config is None:
            from .schemas import GapDetectionConfig

            gap_detection_config = GapDetectionConfig()  # Use defaults

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
        df = self._calculate_advanced_patterns(df, config)

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

        # Fully vectorized signal detection using cascaded when expressions
        # Build comprehensive pattern matching in a single operation
        #
        # **Cascaded Logic Explanation:**
        # This approach builds nested conditional expressions that process all rows simultaneously,
        # avoiding any loops while maintaining proper pattern priority (3-bar > 2-bar).
        #
        # **Priority System:**
        # 1. 3-bar patterns have higher priority and are checked first
        # 2. 2-bar patterns are only applied if no 3-bar pattern matched (signal_expr.is_null())
        # 3. This ensures that more specific patterns (3-bar) take precedence over general ones (2-bar)
        #
        # **Implementation Strategy:**
        # - Start with null expressions for all signal attributes
        # - For each pattern in SIGNALS dictionary, build cascaded when() expressions
        # - Each when() either sets the value (if pattern matches) or preserves existing value
        # - Final expressions are applied to all rows in a single vectorized operation

        # Initialize expressions with null values
        signal_expr = lit(None)
        type_expr = lit(None)
        bias_expr = lit(None)

        # Phase 1: Build cascaded expressions for 3-bar patterns (higher priority)
        # These patterns take precedence and will overwrite any existing matches
        for pattern, config in SIGNALS.items():
            if config["bar_count"] == 3:
                signal_expr = when(col("pattern_3bar") == pattern).then(lit(pattern)).otherwise(signal_expr)
                type_expr = when(col("pattern_3bar") == pattern).then(lit(config["category"])).otherwise(type_expr)
                bias_expr = when(col("pattern_3bar") == pattern).then(lit(config["bias"])).otherwise(bias_expr)

        # Phase 2: Add 2-bar patterns (lower priority, only if no 3-bar pattern matched)
        # Check signal_expr.is_null() to ensure 3-bar patterns maintain priority
        for pattern, config in SIGNALS.items():
            if config["bar_count"] == 2:
                signal_expr = (
                    when((col("pattern_2bar") == pattern) & signal_expr.is_null())
                    .then(lit(pattern))
                    .otherwise(signal_expr)
                )
                type_expr = (
                    when((col("pattern_2bar") == pattern) & type_expr.is_null())
                    .then(lit(config["category"]))
                    .otherwise(type_expr)
                )
                bias_expr = (
                    when((col("pattern_2bar") == pattern) & bias_expr.is_null())
                    .then(lit(config["bias"]))
                    .otherwise(bias_expr)
                )

        # Apply all signal detection in a single operation
        df = df.with_columns(
            [
                signal_expr.alias("signal"),
                type_expr.alias("type"),
                bias_expr.alias("bias"),
                # Add placeholder columns for multi-target support
                # These are populated on-demand via get_signal_objects()
                lit(None, dtype=String).alias("target_prices"),
                lit(None, dtype=Int32).alias("target_count"),
            ]
        )

        # Note: Signal objects are not created in this vectorized implementation
        # for performance reasons. If signal objects are needed, they can be created
        # on-demand using the get_signal_objects method.
        # Target detection is also deferred to get_signal_objects() to maintain
        # vectorized performance for the main indicator pipeline.

        # Clean up temporary columns
        df = df.drop(["scenario_2", "scenario_1", "scenario_0", "pattern_2bar", "pattern_3bar"])

        return df

    def get_signal_objects(self, result_df: PolarsDataFrame) -> list:
        """
        Create SignalMetadata objects on-demand from processed indicator results.

        This method processes rows with signals and creates full SignalMetadata objects
        with prices and trading context. Only used when signal objects are needed.

        Args:
            result_df: DataFrame with processed indicators including signal columns

        Returns:
            List of SignalMetadata objects for rows with detected signals
        """
        signal_objects = []

        # Add row index to track original positions
        df_with_index = result_df.with_row_index("__original_idx")

        # Filter to rows with signals
        signal_rows = df_with_index.filter(col("signal").is_not_null())

        if len(signal_rows) == 0:
            return signal_objects

        # Convert to pandas for row-by-row processing
        signal_data = signal_rows.to_pandas()

        for _, row in signal_data.iterrows():
            original_index = int(row["__original_idx"])

            # Get target_config for this row's timeframe
            target_config = None
            if "timeframe" in row and row["timeframe"] is not None:
                tf_config = self._get_config_for_timeframe(row["timeframe"])
                target_config = tf_config.target_config if tf_config else None

            signal_obj = self._create_signal_object(
                df_with_index, original_index, row["signal"], row["type"], row["bias"], target_config
            )
            if signal_obj:
                signal_objects.append(signal_obj)

        return signal_objects

    def _detect_targets_for_signal(
        self, df: PolarsDataFrame, signal_index: int, bias: str, target_config
    ) -> list[float]:
        """
        Detect multiple target levels for a signal using vectorized operations.

        Detects all relevant local highs (long) or lows (short) between signal bar
        and configured upper bound, applies merge logic, and returns targets in
        reverse chronological order.

        Args:
            df: DataFrame with market structure columns
            signal_index: Index of signal bar
            bias: Signal bias ("long" or "short")
            target_config: TargetConfig with detection parameters

        Returns:
            List of target prices in reverse chronological order (most recent first)
        """
        if target_config is None:
            return []

        # Import here to avoid circular dependency
        from .schemas import TargetConfig

        # Handle None config
        if target_config is None:
            target_config = TargetConfig()

        upper_bound = target_config.upper_bound
        merge_threshold_pct = target_config.merge_threshold_pct
        max_targets = target_config.max_targets

        # Determine target type from upper_bound
        # higher_high/lower_high -> targets are highs (long signals)
        # higher_low/lower_low -> targets are lows (short signals)
        target_is_high = upper_bound in ["higher_high", "lower_high"]

        # Get the column name for targets and upper bound
        if bias == "long":
            target_col = "high"
            structure_col = upper_bound  # e.g., "higher_high"
        else:  # short
            target_col = "low"
            structure_col = upper_bound  # e.g., "lower_low"

        # Get data up to signal bar (not including signal bar itself)
        historical_df = df.slice(0, signal_index)

        if len(historical_df) == 0:
            return []

        # Detect local highs/lows using rolling windows (similar to market structure detection)
        # Use same window approach as in _calculate_market_structure
        window = 1  # Use minimal window for target detection
        min_required = 2 * window + 1

        if len(historical_df) < min_required:
            return []

        # Add row index
        historical_df = historical_df.with_row_index("__row_idx")

        # Detect local extremes
        if target_is_high:
            historical_df = historical_df.with_columns(
                [col("high").rolling_max(window_size=2 * window + 1, center=True).alias("window_max")]
            )
            historical_df = historical_df.with_columns(
                [
                    (
                        (col("__row_idx") >= window)
                        & (col("__row_idx") < (len(historical_df) - window))
                        & (col("high") == col("window_max"))
                    ).alias("is_local_extreme")
                ]
            )
        else:
            historical_df = historical_df.with_columns(
                [col("low").rolling_min(window_size=2 * window + 1, center=True).alias("window_min")]
            )
            historical_df = historical_df.with_columns(
                [
                    (
                        (col("__row_idx") >= window)
                        & (col("__row_idx") < (len(historical_df) - window))
                        & (col("low") == col("window_min"))
                    ).alias("is_local_extreme")
                ]
            )

        # Filter to local extremes only
        local_extremes = historical_df.filter(col("is_local_extreme"))

        if len(local_extremes) == 0:
            return []

        # Extract target prices
        target_prices = local_extremes.select(target_col).to_series().to_list()

        # Filter for ascending/descending progression
        filtered_targets = []
        for price in reversed(target_prices):  # Scan backwards from most recent
            if not filtered_targets:
                filtered_targets.append(price)
            else:
                if bias == "long":
                    # For long: each target should be higher than previous
                    if price > filtered_targets[-1]:
                        filtered_targets.append(price)
                else:  # short
                    # For short: each target should be lower than previous
                    if price < filtered_targets[-1]:
                        filtered_targets.append(price)

        if not filtered_targets:
            return []

        # Check for upper bound in structure column
        # Stop at first occurrence of upper bound
        upper_bound_value = None
        if structure_col in df.columns:
            # Get upper bound value at signal bar
            signal_row = df.slice(signal_index, 1)
            bound_value = signal_row.select(structure_col).item()
            if bound_value is not None:
                upper_bound_value = bound_value

        # Trim targets to upper bound
        if upper_bound_value is not None:
            final_targets = []
            for price in filtered_targets:
                if bias == "long":
                    if price <= upper_bound_value:
                        final_targets.append(price)
                    else:
                        # Stop at and include first target beyond upper bound
                        final_targets.append(price)
                        break
                else:  # short
                    if price >= upper_bound_value:
                        final_targets.append(price)
                    else:
                        # Stop at and include first target beyond upper bound
                        final_targets.append(price)
                        break
            filtered_targets = final_targets

        # Apply merge logic
        if merge_threshold_pct > 0 and len(filtered_targets) > 1:
            merged = []
            i = 0
            while i < len(filtered_targets):
                current = filtered_targets[i]
                # Look ahead for targets within merge threshold
                group = [current]
                j = i + 1
                while j < len(filtered_targets):
                    next_price = filtered_targets[j]
                    pct_diff = abs(next_price - current) / current
                    if pct_diff <= merge_threshold_pct:
                        group.append(next_price)
                        j += 1
                    else:
                        break

                # Pick representative from group
                if bias == "long":
                    merged.append(max(group))  # Pick higher for long
                else:
                    merged.append(min(group))  # Pick lower for short

                i = j
            filtered_targets = merged

        # Apply max_targets limit
        if max_targets is not None and len(filtered_targets) > max_targets:
            filtered_targets = filtered_targets[:max_targets]

        return filtered_targets

    def _create_signal_object(self, df, index, pattern, category, bias, target_config=None):
        """Factory method to create signal objects with multi-target support."""
        from datetime import datetime

        from .signals import SignalBias, SignalCategory, SignalMetadata, TargetLevel

        # Skip if bias is None (some patterns need special handling)
        if bias is None:
            return None

        # Get pattern configuration
        config = SIGNALS[pattern]
        bar_count = config["bar_count"]

        # Get trigger bar (previous bar) for entry/stop prices
        # In TheStrat, entry and stop come from the trigger bar (typically bar before signal)
        # For 2-bar patterns: trigger is 1 bar back, for 3-bar: trigger is 1 bar back
        trigger_index = index - 1  # Standard: trigger bar is previous bar

        if trigger_index < 0:
            return None

        trigger_row = df.row(trigger_index)
        current_row = df.row(index)

        # Find column indices
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

        # Detect multiple targets for reversals
        target_levels = []
        if category == "reversal":
            target_prices = self._detect_targets_for_signal(df, index, bias, target_config)
            target_levels = [TargetLevel(price=price) for price in target_prices]

        # Get timestamp
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

        # Get timeframe if available
        timeframe = None
        if "timeframe" in df.columns:
            timeframe_col_idx = df.columns.index("timeframe")
            timeframe = current_row[timeframe_col_idx]

        # Create signal metadata object
        return SignalMetadata(
            pattern=pattern,
            category=SignalCategory[category.upper()],
            bias=SignalBias[bias.upper()],
            bar_count=bar_count,
            entry_price=entry_price,
            stop_price=stop_price,
            target_prices=target_levels,
            timestamp=timestamp,
            symbol=symbol,
            timeframe=timeframe,
        )

    def _calculate_advanced_patterns(self, data: PolarsDataFrame, config: TimeframeItemConfig) -> PolarsDataFrame:
        """Calculate advanced Strat patterns: kicker, f23, pmg, motherbar_problems."""
        # Get gap detection configuration from the passed config
        gap_detection_config = config.gap_detection
        if gap_detection_config is None:
            from .schemas import GapDetectionConfig

            gap_detection_config = GapDetectionConfig()  # Use defaults

        gap_threshold = gap_detection_config.threshold

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
                .alias("gapper")
            ]
        )

        # Kicker: Continuity reversal with gap (corrected from setup_processor.py)
        df = df.with_columns(
            [
                # Bullish kicker: continuity1=0 & gapper=1 & continuity=1 (bearish to bullish with gap up)
                when((col("continuity").shift(1) == 0) & (col("gapper") == 1) & (col("continuity") == 1))
                .then(1)  # Bullish kicker = 1
                # Bearish kicker: continuity1=1 & gapper=0 & continuity=0 (bullish to bearish with gap down)
                .when((col("continuity").shift(1) == 1) & (col("gapper") == 0) & (col("continuity") == 0))
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

    def _round_numeric_outputs(self, data: PolarsDataFrame) -> PolarsDataFrame:
        """
        Round all Float64 indicator outputs to 5 decimal places for consistent database storage.

        Dynamically detects Float64 columns from the schema and rounds them to prevent
        excessive precision in float calculations from cluttering database storage.

        Args:
            data: DataFrame with calculated indicators

        Returns:
            DataFrame with Float64 columns rounded to 5 decimal places
        """
        from polars import Float64

        from .schemas import IndicatorSchema

        df = data.clone()

        # Get all Float64 columns from the schema dynamically
        schema_types = IndicatorSchema.get_polars_dtypes()
        float64_columns = [
            column_name for column_name, expected_dtype in schema_types.items() if expected_dtype == Float64
        ]

        # Create rounding expressions for Float64 columns that exist in the DataFrame
        rounding_expressions = []
        for column_name in float64_columns:
            if column_name in df.columns:
                # Verify the column is actually numeric (handles edge cases)
                column_dtype = df[column_name].dtype
                if str(column_dtype) in ["Float64", "Int64", "Float32", "Int32"]:
                    rounding_expressions.append(col(column_name).round(5).alias(column_name))

        # Apply rounding if there are any Float64 columns to round
        if rounding_expressions:
            df = df.with_columns(rounding_expressions)

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

        # Note: Minimum data point validation removed as _calculate_swing_points
        # now handles small datasets gracefully with proper safeguards

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
