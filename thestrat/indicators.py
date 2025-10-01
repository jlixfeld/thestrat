"""
Vectorized Strat technical indicators implementation.

This module provides comprehensive Strat pattern analysis with high-performance
vectorized calculations using Polars operations.
"""

from typing import TYPE_CHECKING

from pandas import DataFrame as PandasDataFrame
from polars import DataFrame as PolarsDataFrame
from polars import Float64, Int32, String, col, concat_str, lit, when

from .base import Component
from .schemas import IndicatorsConfig, TimeframeItemConfig
from .signals import SIGNALS

if TYPE_CHECKING:
    from .signals import SignalMetadata


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
        df = self._calculate_signals(df, config)

        return df

    def _calculate_signals(self, data: PolarsDataFrame, config: "TimeframeItemConfig") -> PolarsDataFrame:
        """
        Detect multi-bar signal patterns and populate target prices.

        Signals are collections of sequential scenarios that match specific
        patterns defined in the SIGNALS dictionary. Targets are calculated
        eagerly for database integration.
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
        for pattern, signal_config in SIGNALS.items():
            if signal_config["bar_count"] == 3:
                signal_expr = when(col("pattern_3bar") == pattern).then(lit(pattern)).otherwise(signal_expr)
                type_expr = (
                    when(col("pattern_3bar") == pattern).then(lit(signal_config["category"])).otherwise(type_expr)
                )
                bias_expr = when(col("pattern_3bar") == pattern).then(lit(signal_config["bias"])).otherwise(bias_expr)

        # Phase 2: Add 2-bar patterns (lower priority, only if no 3-bar pattern matched)
        # Check signal_expr.is_null() to ensure 3-bar patterns maintain priority
        for pattern, signal_config in SIGNALS.items():
            if signal_config["bar_count"] == 2:
                signal_expr = (
                    when((col("pattern_2bar") == pattern) & signal_expr.is_null())
                    .then(lit(pattern))
                    .otherwise(signal_expr)
                )
                type_expr = (
                    when((col("pattern_2bar") == pattern) & type_expr.is_null())
                    .then(lit(signal_config["category"]))
                    .otherwise(type_expr)
                )
                bias_expr = (
                    when((col("pattern_2bar") == pattern) & bias_expr.is_null())
                    .then(lit(signal_config["bias"]))
                    .otherwise(bias_expr)
                )

        # Import List type for target_prices column
        from polars import List as PolarsListType

        # Apply all signal detection in a single operation
        df = df.with_columns(
            [
                signal_expr.alias("signal"),
                type_expr.alias("type"),
                bias_expr.alias("bias"),
                # Initialize target columns with proper types
                lit(None, dtype=PolarsListType(Float64)).alias("target_prices"),
                lit(None, dtype=Int32).alias("target_count"),
            ]
        )

        # Clean up temporary columns
        df = df.drop(["scenario_2", "scenario_1", "scenario_0", "pattern_2bar", "pattern_3bar"])

        # Populate targets eagerly for database integration
        df = self._populate_targets_in_dataframe(df, config)

        return df

    def _populate_targets_in_dataframe(self, df: PolarsDataFrame, config: "TimeframeItemConfig") -> PolarsDataFrame:
        """
        Populate target_prices and target_count for all signal rows.

        Args:
            df: DataFrame with signal column populated
            config: TimeframeItemConfig with target_config

        Returns:
            DataFrame with target_prices (List[Float64]) and target_count populated
        """
        import polars as pl

        # Handle both TimeframeItemConfig objects and dict configs (for backward compatibility in tests)
        if isinstance(config, dict):
            target_config = config.get("target_config")
        else:
            target_config = config.target_config if config else None

        if target_config is None:
            return df

        df_with_idx = df.with_row_index("__row_idx")
        signal_rows = df_with_idx.filter(col("signal").is_not_null())

        if len(signal_rows) == 0:
            return df

        updates = []
        for row in signal_rows.iter_rows(named=True):
            targets = self._detect_targets_for_signal(df_with_idx, row["__row_idx"], row["bias"], target_config)
            updates.append(
                {
                    "row_idx": row["__row_idx"],
                    "target_prices": targets if targets else None,
                    "target_count": len(targets) if targets else None,
                }
            )

        if updates:
            updates_df = pl.DataFrame(updates)
            df_with_idx = (
                df_with_idx.join(updates_df, left_on="__row_idx", right_on="row_idx", how="left")
                .with_columns(
                    [
                        col("target_prices_right").alias("target_prices"),
                        col("target_count_right").alias("target_count"),
                    ]
                )
                .drop(["target_prices_right", "target_count_right"])
            )

        return df_with_idx.drop("__row_idx")

    @classmethod
    def get_signal_object(cls, df: PolarsDataFrame) -> "SignalMetadata":
        """
        Create SignalMetadata object from single-row DataFrame.

        Decoupled from pipeline - can be called standalone with database query results.
        Expects DataFrame with exactly 1 row containing signal data with pre-calculated targets.

        Args:
            df: DataFrame with exactly 1 row containing signal data

        Returns:
            SignalMetadata object with targets parsed from DataFrame

        Raises:
            ValueError: If df doesn't have exactly 1 row

        Example:
            # Query database for specific signal
            signal_df = db.query(\"\"\"
                SELECT * FROM signals
                WHERE symbol='AAPL' AND timestamp='2024-01-15 10:30:00'
            \"\"\")

            # Create signal object (no pipeline needed)
            signal = Indicators.get_signal_object(signal_df)
        """
        if df.shape[0] != 1:
            raise ValueError(
                f"get_signal_object() expects DataFrame with exactly 1 row, got {df.shape[0]} rows. "
                "Query database to filter to specific signal first."
            )

        from .signals import SIGNALS, SignalBias, SignalCategory, SignalMetadata, TargetLevel

        # Extract row data
        row = df.row(0, named=True)

        # Parse target_prices from native list (not JSON)
        target_prices = []
        if row.get("target_prices") is not None:
            # target_prices is List[Float64] from database
            target_list = row["target_prices"]
            target_prices = [TargetLevel(price=price) for price in target_list]

        # Get signal configuration
        pattern = row["signal"]
        config = SIGNALS.get(pattern, {})

        # Determine entry/stop from trigger bar (current bar)
        if row["bias"] == "long":
            entry_price = float(row["high"])
            stop_price = float(row["low"])
        else:  # short
            entry_price = float(row["low"])
            stop_price = float(row["high"])

        # Create SignalMetadata
        signal = SignalMetadata(
            pattern=pattern,
            category=SignalCategory(row["type"]),
            bias=SignalBias(row["bias"]),
            bar_count=config.get("bar_count", 2),
            entry_price=entry_price,
            stop_price=stop_price,
            target_prices=target_prices,
            timestamp=row["timestamp"],
            symbol=row.get("symbol"),
            timeframe=row.get("timeframe"),
        )

        return signal

    def _detect_targets_for_signal(
        self, df: PolarsDataFrame, signal_index: int, bias: str, target_config
    ) -> list[float]:
        """
        Detect multiple target levels for a signal using vectorized operations.

        Detects all relevant local highs (long) or lows (short) between signal bar
        and configured bound, applies merge logic, and returns targets in
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

        # Select appropriate bound based on signal bias
        # Long signals use upper_bound (higher_high or lower_high)
        # Short signals use lower_bound (higher_low or lower_low)
        if bias == "long":
            structure_bound = target_config.upper_bound
            target_col = "high"
        else:  # short
            structure_bound = target_config.lower_bound
            target_col = "low"

        merge_threshold_pct = target_config.merge_threshold_pct
        max_targets = target_config.max_targets

        # Determine target column from structure bound
        # higher_high/lower_high -> targets are highs (long signals)
        # higher_low/lower_low -> targets are lows (short signals)
        structure_col = structure_bound

        # Get data up to signal bar (not including signal bar itself)
        historical_df = df.slice(0, signal_index)

        if len(historical_df) == 0:
            return []

        # Add row index for downstream sorting
        if "__row_idx" not in historical_df.columns:
            historical_df = historical_df.with_row_index("__row_idx")

        # Extract all historical prices (no pre-filtering for local/progressive extremes)
        # The descending/ascending ladder filter below will select the right targets
        all_prices = historical_df.select(target_col).to_series().to_list()

        if not all_prices:
            return []

        # Apply descending/ascending ladder filter with trigger bar validation
        # The bar immediately before signal (i=0) is the trigger bar
        # Targets must be beyond the trigger bar's price level
        # For short: build descending ladder (each target must be < previous accepted target)
        # For long: build ascending ladder (each target must be > previous accepted target)
        filtered_targets = []
        trigger_price = None
        for i, price in enumerate(reversed(all_prices)):  # Scan backwards from most recent
            if i == 0:
                # Save trigger bar's price for filtering (bar immediately before signal)
                trigger_price = price
                continue
            if not filtered_targets:
                # First target must be beyond trigger bar's price
                if bias == "long":
                    # Long: target must be above trigger bar's high
                    if price > trigger_price:
                        filtered_targets.append(price)
                else:  # short
                    # Short: target must be below trigger bar's low
                    if price < trigger_price:
                        filtered_targets.append(price)
            else:
                if bias == "long":
                    # For long: each target should be higher than last accepted target
                    if price > filtered_targets[-1]:
                        filtered_targets.append(price)
                else:  # short
                    # For short: each target should be lower than last accepted target
                    if price < filtered_targets[-1]:
                        filtered_targets.append(price)

        if not filtered_targets:
            return []

        # Targets are in reverse chronological order (newest to oldest)
        # This is the correct order for display and merging

        # Check for structure bound in structure column
        # Find most recent bar where structure column has a value (indicating swing point)
        bound_price = None
        if structure_col in historical_df.columns:
            # Filter to bars where structure column is not null (forward-filled values)
            # Structure columns contain float prices, not booleans
            bound_bars = historical_df.filter(col(structure_col).is_not_null())

            if len(bound_bars) > 0:
                # Sort by row index descending to get most recent occurrence first
                bound_bars = bound_bars.sort("__row_idx", descending=True)
                first_bound_row = bound_bars.row(0, named=True)

                # Extract the price from the structure column (not target column)
                # For long signals: structure_col is higher_high/lower_high containing the bound price level
                # For short signals: structure_col is higher_low/lower_low containing the bound price level
                bound_price = float(first_bound_row[structure_col])

        # Trim targets to structure bound
        if bound_price is not None:
            final_targets = []
            for price in filtered_targets:
                if bias == "long":
                    if price <= bound_price:
                        final_targets.append(price)
                    else:
                        # Stop at and include first target beyond bound
                        final_targets.append(price)
                        break
                else:  # short
                    if price >= bound_price:
                        final_targets.append(price)
                    else:
                        # Stop at and include first target beyond bound
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
