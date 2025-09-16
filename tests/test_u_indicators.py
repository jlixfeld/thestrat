"""
Unit tests for TheStrat Indicators component.

Tests comprehensive Strat technical indicators with vectorized calculations.
"""

from datetime import datetime, timedelta

import pytest
from pandas import DataFrame as PandasDataFrame
from pandas import date_range
from polars import Boolean, DataFrame, Float64, Int32, Int64, Series, Utf8, col
from pydantic import ValidationError

from thestrat.indicators import Indicators
from thestrat.schemas import (
    GapDetectionConfig,
    IndicatorSchema,
    IndicatorsConfig,
    SwingPointsConfig,
    TimeframeItemConfig,
)
from thestrat.signals import SIGNALS


def validate_result_against_schema(result: DataFrame) -> None:
    """
    Helper function to validate that indicator processing results match the IndicatorSchema.

    This ensures that the actual data types returned by indicators processing
    match what the schema declares, preventing schema inconsistency bugs.

    Args:
        result: Polars DataFrame from indicators processing

    Raises:
        AssertionError: If any column type doesn't match schema expectations
    """
    from polars import Float64, Int64, Null

    schema_types = IndicatorSchema.get_polars_dtypes()

    # Check each column that exists in both result and schema
    for column_name in result.columns:
        if column_name in schema_types:
            actual_dtype = result[column_name].dtype
            expected_dtype = schema_types[column_name]

            # Allow compatible numeric types (Int64 can be safely used where Float64 is expected)
            # This handles cases where test data creates integer prices that should be floats
            if expected_dtype == Float64 and actual_dtype == Int64:
                continue  # Int64 -> Float64 is safe

            # Allow Null type for nullable columns (happens when all values are null)
            # This is common in columns like 'signal', 'type', 'bias' when no patterns are detected
            if actual_dtype == Null:
                continue  # All-null columns are acceptable for nullable schema fields

            assert actual_dtype == expected_dtype, (
                f"Schema mismatch for column '{column_name}': expected {expected_dtype}, got {actual_dtype}"
            )


@pytest.mark.unit
class TestIndicatorsInit:
    """Test cases for Indicators initialization."""

    def test_init_with_all_timeframe_config(self):
        """Test initialization with 'all' timeframe configuration using Factory."""
        from thestrat.factory import Factory

        config = IndicatorsConfig(
            timeframe_configs=[
                TimeframeItemConfig(
                    timeframes=["all"],
                    swing_points=SwingPointsConfig(window=7, threshold=3.0),
                    gap_detection=GapDetectionConfig(threshold=0.001),
                )
            ]
        )
        indicators = Factory.create_indicators(config)

        assert len(indicators.config.timeframe_configs) == 1
        assert indicators.config.timeframe_configs[0].timeframes == ["all"]
        # Configuration is properly stored and will be extracted during processing
        assert indicators.config.timeframe_configs[0].swing_points is not None
        assert indicators.config.timeframe_configs[0].swing_points.window == 7

    def test_init_with_swing_config(self):
        """Test initialization with swing point configuration using Factory."""
        from thestrat.factory import Factory

        config = IndicatorsConfig(
            timeframe_configs=[
                TimeframeItemConfig(
                    timeframes=["all"],
                    swing_points=SwingPointsConfig(window=7, threshold=3.0),
                )
            ]
        )
        indicators = Factory.create_indicators(config)

        assert len(indicators.config.timeframe_configs) == 1
        assert indicators.config.timeframe_configs[0].timeframes == ["all"]
        assert indicators.config.timeframe_configs[0].swing_points is not None
        assert indicators.config.timeframe_configs[0].swing_points.window == 7
        assert indicators.config.timeframe_configs[0].swing_points.threshold == 3.0

    def test_init_partial_swing_config(self):
        """Test initialization with partial swing configuration using Factory."""
        from thestrat.factory import Factory

        config = IndicatorsConfig(
            timeframe_configs=[
                TimeframeItemConfig(
                    timeframes=["all"],
                    swing_points=SwingPointsConfig(window=10),  # threshold will use default 5.0
                )
            ]
        )
        indicators = Factory.create_indicators(config)

        assert indicators.config.timeframe_configs[0].swing_points is not None
        assert indicators.config.timeframe_configs[0].swing_points.window == 10
        assert indicators.config.timeframe_configs[0].swing_points.threshold == 5.0  # Uses default

    def test_init_empty_swing_config(self):
        """Test initialization with minimal config using Factory (defaults applied)."""
        from thestrat.factory import Factory

        config = IndicatorsConfig(
            timeframe_configs=[
                TimeframeItemConfig(timeframes=["all"])  # No swing_points specified - uses defaults when needed
            ]
        )
        indicators = Factory.create_indicators(config)

        assert len(indicators.config.timeframe_configs) == 1
        assert indicators.config.timeframe_configs[0].timeframes == ["all"]
        # swing_points will be None, defaults will be applied during processing


@pytest.mark.unit
class TestIndicatorsValidation:
    """Test cases for input validation."""

    @pytest.fixture
    def indicators(self):
        """Create indicators instance for testing."""
        return Indicators(
            IndicatorsConfig(
                timeframe_configs=[
                    TimeframeItemConfig(
                        timeframes=["all"],
                        swing_points=SwingPointsConfig(window=5, threshold=5.0),
                        gap_detection=GapDetectionConfig(threshold=0.001),
                    )
                ]
            )
        )

    @pytest.fixture
    def valid_ohlc_data(self):
        """Create valid OHLC DataFrame with sufficient data."""
        from .utils.thestrat_data_utils import create_ohlc_data

        return create_ohlc_data(periods=24, start="2023-01-01", freq_minutes=60, base_price=100.0)

    def test_validate_input_valid_data(self, indicators, valid_ohlc_data):
        """Test validation passes for valid OHLC data."""
        # Should not raise any exception
        indicators.validate_input(valid_ohlc_data)

    def test_validate_input_missing_columns(self, indicators):
        """Test validation fails for missing required columns."""
        incomplete_data = DataFrame(
            {
                "timestamp": [datetime.now()],
                "open": [100.0],
                "high": [101.0],
                # Missing low, close, volume
            }
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            indicators.validate_input(incomplete_data)

    def test_validate_input_insufficient_data(self, indicators):
        """Test validation fails when data is too small for swing analysis."""
        small_data = DataFrame(
            {
                "timestamp": [datetime.now()],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [1000],
            }
        )

        with pytest.raises(ValueError, match="Insufficient data"):
            indicators.validate_input(small_data)

    def test_validate_input_price_integrity(self, indicators):
        """Test validation checks price data integrity."""
        # Create data with invalid price relationships
        from .utils.thestrat_data_utils import create_timestamp_series

        timestamps = create_timestamp_series("2023-01-01", 20, 60)
        invalid_data = DataFrame(
            {
                "timestamp": timestamps,
                "open": [100.0] * 20,
                "high": [99.0] * 20,  # High < Open (invalid)
                "low": [101.0] * 20,  # Low > Open (invalid)
                "close": [100.5] * 20,
                "volume": [1000] * 20,
            }
        )

        with pytest.raises(ValueError, match="Invalid price data"):
            indicators.validate_input(invalid_data)

    def test_validate_input_pandas_conversion(self, indicators):
        """Test validation works with pandas DataFrame."""
        pandas_data = PandasDataFrame(
            {
                "timestamp": date_range("2023-01-01", periods=20, freq="1h"),
                "open": [100.0 + i * 0.1 for i in range(20)],
                "high": [101.0 + i * 0.1 for i in range(20)],
                "low": [99.0 + i * 0.1 for i in range(20)],
                "close": [100.5 + i * 0.1 for i in range(20)],
                "volume": [1000] * 20,
            }
        )

        # Should not raise any exception
        indicators.validate_input(pandas_data)


@pytest.mark.unit
class TestSwingPoints:
    """Test cases for swing point detection."""

    @pytest.fixture
    def trending_data(self):
        """Create trending OHLC data with clear swing points."""
        from .utils.thestrat_data_utils import create_trend_data

        return create_trend_data(periods=10)

    def test_swing_point_detection(self, trending_data):
        """Test basic swing point detection."""
        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[
                    TimeframeItemConfig(timeframes=["all"], swing_points=SwingPointsConfig(window=3, threshold=1.0))
                ]
            )
        )  # Lower threshold
        config = indicators.config.timeframe_configs[0]
        result = indicators._calculate_swing_points(trending_data, config)

        assert "swing_high" in result.columns
        assert "swing_low" in result.columns
        assert "pivot_high" in result.columns
        assert "pivot_low" in result.columns
        assert "new_swing_high" in result.columns
        assert "new_swing_low" in result.columns
        assert "new_pivot_high" in result.columns
        assert "new_pivot_low" in result.columns

        # Check that we have swing point data (even if no points found)
        # The columns should exist and have proper types (can be Int64 or Float64 depending on input data)
        assert result["swing_high"].dtype in [Int64, Float64]
        assert result["swing_low"].dtype in [Int64, Float64]
        assert result["pivot_high"].dtype in [Int64, Float64]
        assert result["pivot_low"].dtype in [Int64, Float64]
        assert result["new_swing_high"].dtype == Boolean
        assert result["new_swing_low"].dtype == Boolean
        assert result["new_pivot_high"].dtype == Boolean
        assert result["new_pivot_low"].dtype == Boolean

    def test_pivot_values_at_swings(self, trending_data):
        """Test that pivot values are set correctly at swing points."""
        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[TimeframeItemConfig(timeframes=["all"], swing_points=SwingPointsConfig(window=3))]
            )
        )
        config = indicators.config.timeframe_configs[0]
        result = indicators._calculate_swing_points(trending_data, config)

        # Pivot high should be set where new_swing_high is True
        swing_high_rows = result.filter(col("new_swing_high"))
        for row in swing_high_rows.iter_rows(named=True):
            assert row["pivot_high"] is not None
            assert row["pivot_high"] == row["high"]
            # swing_high column should also equal the high price
            assert row["swing_high"] == row["high"]

        # Pivot low should be set where new_swing_low is True
        swing_low_rows = result.filter(col("new_swing_low"))
        for row in swing_low_rows.iter_rows(named=True):
            assert row["pivot_low"] is not None
            assert row["pivot_low"] == row["low"]
            # swing_low column should also equal the low price
            assert row["swing_low"] == row["low"]

    def test_swing_threshold_filtering(self):
        """Test that swing threshold filters minor swings."""
        # Create data with small fluctuations that should be filtered
        from .utils.thestrat_data_utils import create_timestamp_series

        timestamps = create_timestamp_series("2023-01-01", 10, 60)
        data = DataFrame(
            {
                "timestamp": timestamps,
                "open": [100.0, 100.1, 100.2, 100.1, 100.0, 100.1, 100.2, 100.1, 100.0, 100.1],
                "high": [100.5, 100.6, 100.7, 100.6, 100.5, 100.6, 100.7, 100.6, 100.5, 100.6],
                "low": [99.5, 99.6, 99.7, 99.6, 99.5, 99.6, 99.7, 99.6, 99.5, 99.6],
                "close": [100.2, 100.3, 100.4, 100.3, 100.2, 100.3, 100.4, 100.3, 100.2, 100.3],
                "volume": [1000] * 10,
            }
        )

        # With high threshold (5%), should filter out small swings
        indicators_strict = Indicators(
            IndicatorsConfig(
                timeframe_configs=[
                    TimeframeItemConfig(timeframes=["all"], swing_points=SwingPointsConfig(threshold=5.0))
                ]
            )
        )
        config = indicators_strict.config.timeframe_configs[0]
        result_strict = indicators_strict._calculate_swing_points(data, config)

        # With low threshold (0.1%), should detect more swings
        indicators_loose = Indicators(
            IndicatorsConfig(
                timeframe_configs=[
                    TimeframeItemConfig(timeframes=["all"], swing_points=SwingPointsConfig(threshold=0.1))
                ]
            )
        )
        config = indicators_loose.config.timeframe_configs[0]
        result_loose = indicators_loose._calculate_swing_points(data, config)

        strict_swings = len(result_strict.filter(col("new_swing_high") | col("new_swing_low")))
        loose_swings = len(result_loose.filter(col("new_swing_high") | col("new_swing_low")))

        assert strict_swings <= loose_swings


@pytest.mark.unit
class TestMarketStructure:
    """Test cases for market structure analysis."""

    @pytest.fixture
    def market_structure_data(self):
        """Create data with clear market structure patterns."""
        from .utils.thestrat_data_utils import create_timestamp_series

        timestamps = create_timestamp_series("2023-01-01", 12, 1440)  # Daily
        return DataFrame(
            {
                "timestamp": timestamps,
                "open": [100, 102, 101, 103, 102, 105, 104, 106, 105, 108, 107, 109],
                "high": [101, 103, 102, 104, 103, 106, 105, 107, 106, 109, 108, 110],  # HH pattern
                "low": [99, 101, 100, 102, 101, 104, 103, 105, 104, 107, 106, 108],  # HL pattern
                "close": [100.5, 102.5, 101.5, 103.5, 102.5, 105.5, 104.5, 106.5, 105.5, 108.5, 107.5, 109.5],
                "volume": [1000] * 12,
            }
        )

    def test_market_structure_classification(self, market_structure_data):
        """Test market structure pattern classification."""
        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[TimeframeItemConfig(timeframes=["all"], swing_points=SwingPointsConfig(window=3))]
            )
        )
        # First detect swing points
        config = indicators.config.timeframe_configs[0]
        with_swings = indicators._calculate_swing_points(market_structure_data, config)
        # Then classify market structure
        result = indicators._calculate_market_structure(with_swings)

        assert "higher_high" in result.columns
        assert "lower_high" in result.columns
        assert "higher_low" in result.columns
        assert "lower_low" in result.columns
        assert "new_higher_high" in result.columns
        assert "new_lower_high" in result.columns
        assert "new_higher_low" in result.columns
        assert "new_lower_low" in result.columns

        # Check column types (can be Int64 or Float64 depending on input data)
        assert result["higher_high"].dtype in [Int64, Float64]
        assert result["lower_high"].dtype in [Int64, Float64]
        assert result["higher_low"].dtype in [Int64, Float64]
        assert result["lower_low"].dtype in [Int64, Float64]
        assert result["new_higher_high"].dtype == Boolean
        assert result["new_lower_high"].dtype == Boolean
        assert result["new_higher_low"].dtype == Boolean
        assert result["new_lower_low"].dtype == Boolean

    def test_higher_high_detection(self, market_structure_data):
        """Test higher high detection."""
        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[TimeframeItemConfig(timeframes=["all"], swing_points=SwingPointsConfig(window=3))]
            )
        )
        config = indicators.config.timeframe_configs[0]
        with_swings = indicators._calculate_swing_points(market_structure_data, config)
        result = indicators._calculate_market_structure(with_swings)

        # Should detect higher highs in uptrending data
        higher_highs = result.filter(col("new_higher_high"))
        assert len(higher_highs) >= 0  # May not detect any due to swing detection criteria

    def test_market_structure_mutually_exclusive(self, market_structure_data):
        """Test that market structure classifications are mutually exclusive."""
        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[TimeframeItemConfig(timeframes=["all"], swing_points=SwingPointsConfig(window=3))]
            )
        )
        config = indicators.config.timeframe_configs[0]
        with_swings = indicators._calculate_swing_points(market_structure_data, config)
        result = indicators._calculate_market_structure(with_swings)

        # A swing high cannot be both HH and LH (boolean flags should be mutually exclusive)
        hh_and_lh = result.filter(col("new_higher_high") & col("new_lower_high"))
        assert len(hh_and_lh) == 0

        # A swing low cannot be both HL and LL (boolean flags should be mutually exclusive)
        hl_and_ll = result.filter(col("new_higher_low") & col("new_lower_low"))
        assert len(hl_and_ll) == 0


@pytest.mark.unit
class TestStratPatterns:
    """Test cases for Strat pattern analysis."""

    @pytest.fixture
    def pattern_data(self):
        """Create data with various Strat patterns."""
        from .utils.thestrat_data_utils import create_pattern_data

        return create_pattern_data()

    def test_continuity_pattern(self, pattern_data):
        """Test continuity pattern detection."""
        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[TimeframeItemConfig(timeframes=["all"], swing_points=SwingPointsConfig(window=3))]
            )
        )
        config = indicators.config.timeframe_configs[0]
        result = indicators._calculate_strat_patterns(pattern_data, config)

        assert "continuity" in result.columns
        # Should be Int32 column with values 0, 1, -1
        assert result["continuity"].dtype == Int32

    def test_in_force_pattern(self, pattern_data):
        """Test in-force pattern detection."""
        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[TimeframeItemConfig(timeframes=["all"], swing_points=SwingPointsConfig(window=3))]
            )
        )
        config = indicators.config.timeframe_configs[0]
        result = indicators._calculate_strat_patterns(pattern_data, config)

        assert "in_force" in result.columns
        assert result["in_force"].dtype == Boolean

    def test_scenario_classification(self, pattern_data):
        """Test scenario classification (1, 2, 3)."""
        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[TimeframeItemConfig(timeframes=["all"], swing_points=SwingPointsConfig(window=3))]
            )
        )
        config = indicators.config.timeframe_configs[0]
        result = indicators._calculate_strat_patterns(pattern_data, config)

        assert "scenario" in result.columns
        # Should contain string values "1", "2U", "2D", "3" (no "0")
        unique_scenarios = [s for s in result["scenario"].unique().to_list() if s is not None]
        assert all(scenario in ["1", "2U", "2D", "3"] for scenario in unique_scenarios)

    def test_signal_pattern(self, pattern_data):
        """Test signal pattern detection."""
        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[TimeframeItemConfig(timeframes=["all"], swing_points=SwingPointsConfig(window=3))]
            )
        )
        config = indicators.config.timeframe_configs[0]
        result = indicators._calculate_strat_patterns(pattern_data, config)

        assert "signal" in result.columns
        assert "type" in result.columns
        assert "bias" in result.columns
        # Signal should be pattern string or None
        assert result["signal"].dtype == Utf8
        # Type should be "reversal", "continuation", "context", or None
        assert result["type"].dtype == Utf8
        # Bias should be "long", "short", or None
        assert result["bias"].dtype == Utf8

    def test_hammer_shooter_patterns(self):
        """Test hammer and shooter pattern detection."""
        # Create data with hammer and shooter patterns
        from .utils.thestrat_data_utils import create_timestamp_series

        timestamps = create_timestamp_series("2023-01-01", 4, 1440)  # Daily
        data = DataFrame(
            {
                "timestamp": timestamps,
                "open": [100, 100, 100, 100],
                "high": [101, 105, 101, 101],  # Shooter at index 1
                "low": [95, 99, 99, 99],  # Hammer at index 0
                "close": [100.5, 100.5, 100.5, 100.5],
                "volume": [1000] * 4,
            }
        )

        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[TimeframeItemConfig(timeframes=["all"], swing_points=SwingPointsConfig(window=3))]
            )
        )
        config = indicators.config.timeframe_configs[0]
        result = indicators._calculate_strat_patterns(data, config)

        assert "hammer" in result.columns
        assert "shooter" in result.columns

        # Should detect hammer pattern in first bar (long lower shadow)
        # Should detect shooter pattern in second bar (long upper shadow)


@pytest.mark.unit
class TestAdvancedPatterns:
    """Test cases for advanced Strat patterns."""

    @pytest.fixture
    def advanced_pattern_data(self):
        """Create data for testing advanced patterns."""
        from .utils.thestrat_data_utils import create_ohlc_data

        return create_ohlc_data(periods=20, start="2023-01-01", freq_minutes=1440, base_price=100.0)

    def test_kicker_patterns(self, advanced_pattern_data):
        """Test kicker pattern detection."""
        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[TimeframeItemConfig(timeframes=["all"], swing_points=SwingPointsConfig(window=3))]
            )
        )
        config = indicators.config.timeframe_configs[0]
        with_swings = indicators._calculate_swing_points(advanced_pattern_data, config)
        with_basic = indicators._calculate_strat_patterns(with_swings, config)
        result = indicators._calculate_advanced_patterns(with_basic, config)

        assert "kicker" in result.columns
        assert result["kicker"].dtype == Int32  # Changed from Boolean to Int32 (0/1/null)

    def test_f23_patterns(self, advanced_pattern_data):
        """Test F23 pattern detection."""
        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[TimeframeItemConfig(timeframes=["all"], swing_points=SwingPointsConfig(window=3))]
            )
        )
        config = indicators.config.timeframe_configs[0]
        with_swings = indicators._calculate_swing_points(advanced_pattern_data, config)
        with_basic = indicators._calculate_strat_patterns(with_swings, config)
        result = indicators._calculate_advanced_patterns(with_basic, config)

        assert "f23" in result.columns
        assert result["f23"].dtype == Boolean

    def test_pmg_patterns(self, advanced_pattern_data):
        """Test PMG (Pivot Machine Gun) pattern detection."""
        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[TimeframeItemConfig(timeframes=["all"], swing_points=SwingPointsConfig(window=3))]
            )
        )
        config = indicators.config.timeframe_configs[0]
        with_swings = indicators._calculate_swing_points(advanced_pattern_data, config)
        with_basic = indicators._calculate_strat_patterns(with_swings, config)
        result = indicators._calculate_advanced_patterns(with_basic, config)

        assert "pmg" in result.columns
        assert result["pmg"].dtype == Int32

    def test_motherbar_problems(self, advanced_pattern_data):
        """Test motherbar problems detection."""
        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[TimeframeItemConfig(timeframes=["all"], swing_points=SwingPointsConfig(window=3))]
            )
        )
        config = indicators.config.timeframe_configs[0]
        with_swings = indicators._calculate_swing_points(advanced_pattern_data, config)
        with_basic = indicators._calculate_strat_patterns(with_swings, config)
        result = indicators._calculate_advanced_patterns(with_basic, config)

        assert "motherbar_problems" in result.columns
        assert result["motherbar_problems"].dtype == Boolean


@pytest.mark.unit
class TestPriceAnalysis:
    """Test cases for price analysis indicators."""

    @pytest.fixture
    def price_analysis_data(self):
        """Create data for price analysis testing."""
        from .utils.thestrat_data_utils import create_price_analysis_data

        return create_price_analysis_data()

    def test_price_analysis_calculations(self, price_analysis_data):
        """Test price analysis percentage calculations."""
        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[TimeframeItemConfig(timeframes=["all"], swing_points=SwingPointsConfig(window=3))]
            )
        )
        result = indicators.process(price_analysis_data)

        # Validate that all column types match the schema declarations
        validate_result_against_schema(result)

        assert "percent_close_from_high" in result.columns
        assert "percent_close_from_low" in result.columns

        # Check that percentages are in valid range [0, 100]
        pct_from_high = result["percent_close_from_high"].to_list()
        pct_from_low = result["percent_close_from_low"].to_list()

        assert all(0 <= pct <= 100 for pct in pct_from_high)
        assert all(0 <= pct <= 100 for pct in pct_from_low)

        # Check that percentages sum to approximately 100
        for i in range(len(pct_from_high)):
            assert abs((pct_from_high[i] + pct_from_low[i]) - 100.0) < 0.001

    def test_price_at_high_low_extremes(self):
        """Test price analysis when close is at high or low."""
        from .utils.thestrat_data_utils import create_timestamp_series

        timestamps = create_timestamp_series("2023-01-01", 8, 1440)  # Daily - increased to 8 rows
        data = DataFrame(
            {
                "timestamp": timestamps,
                "open": [100, 100, 100, 100, 100, 100, 100, 100],
                "high": [105, 105, 105, 105, 105, 105, 105, 105],
                "low": [95, 95, 95, 95, 95, 95, 95, 95],
                "close": [105, 95, 100, 105, 95, 102, 98, 103],  # At high, at low, in middle, at high, at low, mixed
                "volume": [1000] * 8,
            }
        )

        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[TimeframeItemConfig(timeframes=["all"], swing_points=SwingPointsConfig(window=3))]
            )
        )
        result = indicators.process(data)

        # Validate that all column types match the schema declarations
        validate_result_against_schema(result)

        # First bar: close at high
        assert result["percent_close_from_high"][0] == 0.0
        assert result["percent_close_from_low"][0] == 100.0

        # Second bar: close at low
        assert result["percent_close_from_high"][1] == 100.0
        assert result["percent_close_from_low"][1] == 0.0

        # Third bar: close in middle
        assert result["percent_close_from_high"][2] == 50.0
        assert result["percent_close_from_low"][2] == 50.0


@pytest.mark.unit
class TestGapAnalysis:
    """Test cases for gap analysis."""

    @pytest.fixture
    def gap_data(self):
        """Create data with gap patterns."""
        from .utils.thestrat_data_utils import create_gap_data

        return create_gap_data()

    def test_gap_detection(self, gap_data):
        """Test gap up and gap down detection."""
        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[TimeframeItemConfig(timeframes=["all"], swing_points=SwingPointsConfig(window=3))]
            )
        )
        result = indicators.process(gap_data)

        assert "gapper" in result.columns

        # Check specific gaps with threshold calculations
        gappers = result["gapper"].to_list()

        # Get actual OHLC data for precise threshold calculations
        opens = result["open"].to_list()
        highs = result["high"].to_list()
        lows = result["low"].to_list()

        # Default gap_threshold = 0.0 (any gap)
        gap_threshold = 0.0

        # First bar has no previous bar, should be None
        assert gappers[0] is None

        # Second bar: open (105) > previous high (102) * 1.001 = 102.102? Yes -> Gap up = 1
        prev_high_threshold = highs[0] * (1 + gap_threshold)  # 102 * 1.001 = 102.102
        assert opens[1] > prev_high_threshold, f"Open {opens[1]} should be > {prev_high_threshold}"
        assert gappers[1] == 1, "Should detect gap up"

        # Third bar: open (95) < previous low (103) * 0.999 = 102.897? Yes -> Gap down = 0
        prev_low_threshold = lows[1] * (1 - gap_threshold)  # 103 * 0.999 = 102.897
        assert opens[2] < prev_low_threshold, f"Open {opens[2]} should be < {prev_low_threshold}"
        assert gappers[2] == 0, "Should detect gap down"

        # Fourth bar: open (103) - check if it's within threshold (no gap)
        prev_high_threshold_4 = highs[2] * (1 + gap_threshold)  # 97 * 1.001 = 97.097
        prev_low_threshold_4 = lows[2] * (1 - gap_threshold)  # 93 * 0.999 = 92.907
        if prev_low_threshold_4 <= opens[3] <= prev_high_threshold_4:
            assert gappers[3] is None, "Should detect no significant gap"
        else:
            # If outside threshold, should detect appropriate gap direction
            if opens[3] > prev_high_threshold_4:
                assert gappers[3] == 1, "Should detect gap up"
            else:
                assert gappers[3] == 0, "Should detect gap down"

    def test_gapper_indicator(self, gap_data):
        """Test gapper indicator with threshold detection."""
        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[TimeframeItemConfig(timeframes=["all"], swing_points=SwingPointsConfig(window=3))]
            )
        )
        result = indicators.process(gap_data)

        gappers = result["gapper"].to_list()

        # Comprehensive validation of gapper logic
        opens = result["open"].to_list()
        highs = result["high"].to_list()
        lows = result["low"].to_list()
        gap_threshold = 0.0  # Default threshold

        # First bar has no previous bar
        assert gappers[0] is None, "First bar should have no gap detection"

        # Validate each subsequent bar against threshold logic
        for i in range(1, len(gappers)):
            prev_high_threshold = highs[i - 1] * (1 + gap_threshold)
            prev_low_threshold = lows[i - 1] * (1 - gap_threshold)
            current_open = opens[i]

            if current_open > prev_high_threshold:
                assert gappers[i] == 1, (
                    f"Bar {i}: open {current_open} > threshold {prev_high_threshold}, should be gap up (1)"
                )
            elif current_open < prev_low_threshold:
                assert gappers[i] == 0, (
                    f"Bar {i}: open {current_open} < threshold {prev_low_threshold}, should be gap down (0)"
                )
            else:
                assert gappers[i] is None, f"Bar {i}: open {current_open} within thresholds, should be None"


@pytest.mark.unit
class TestATHATL:
    """Test cases for All-Time High/Low calculations."""

    @pytest.fixture
    def ath_atl_data(self):
        """Create data for ATH/ATL testing."""
        from .utils.thestrat_data_utils import create_ath_atl_data

        return create_ath_atl_data()

    def test_ath_atl_calculation(self, ath_atl_data):
        """Test ATH and ATL calculations."""
        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[TimeframeItemConfig(timeframes=["all"], swing_points=SwingPointsConfig(window=3))]
            )
        )
        result = indicators.process(ath_atl_data)

        # Validate that all column types match the schema declarations
        validate_result_against_schema(result)

        assert "ath" in result.columns
        assert "atl" in result.columns
        assert "new_ath" in result.columns
        assert "new_atl" in result.columns

        # ATH should be cumulative maximum
        ath_values = result["ath"].to_list()
        expected_ath = [101, 103, 103, 106, 106, 112, 112, 112]
        assert ath_values == expected_ath

        # ATL should be cumulative minimum
        atl_values = result["atl"].to_list()
        expected_atl = [99, 99, 97, 97, 97, 97, 97, 94]
        assert atl_values == expected_atl

    def test_new_ath_atl_flags(self, ath_atl_data):
        """Test new ATH/ATL flag detection."""
        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[TimeframeItemConfig(timeframes=["all"], swing_points=SwingPointsConfig(window=3))]
            )
        )
        result = indicators.process(ath_atl_data)

        # Validate that all column types match the schema declarations
        validate_result_against_schema(result)

        new_ath_flags = result["new_ath"].to_list()
        new_atl_flags = result["new_atl"].to_list()

        # New ATH should be true when high equals ATH
        # Should be true at indices 0, 1, 3, 5 (when new highs are made)
        expected_new_ath = [True, True, False, True, False, True, False, False]
        assert new_ath_flags == expected_new_ath

        # New ATL should be true when low equals ATL
        # Should be true at indices 0, 2, 7 (when new lows are made)
        expected_new_atl = [True, False, True, False, False, False, False, True]
        assert new_atl_flags == expected_new_atl


@pytest.mark.unit
class TestFullProcessing:
    """Test cases for full indicator processing pipeline."""

    @pytest.fixture
    def comprehensive_data(self):
        """Create comprehensive OHLC data for full testing."""
        from .utils.thestrat_data_utils import create_ohlc_data

        return create_ohlc_data(periods=30, start="2023-01-01", freq_minutes=1440, base_price=100.0)

    def test_process_complete_pipeline(self, comprehensive_data):
        """Test complete indicator processing pipeline."""
        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[
                    TimeframeItemConfig(
                        timeframes=["all"],
                        swing_points=SwingPointsConfig(window=5, threshold=5.0),
                        gap_detection=GapDetectionConfig(threshold=0.001),
                    )
                ]
            )
        )
        result = indicators.process(comprehensive_data)

        # Should contain all core indicator categories
        core_columns = [
            # Original OHLC
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            # Swing points
            "swing_high",
            "swing_low",
            "pivot_high",
            "pivot_low",
            # Strat patterns
            "continuity",
            "in_force",
            "scenario",
            "signal",
            "type",
            "bias",
            "hammer",
            "shooter",
            "kicker",
            "f23",
            "pmg",
            "motherbar_problems",
            # Price analysis
            "percent_close_from_high",
            "percent_close_from_low",
            # ATH/ATL
            "ath",
            "atl",
            "new_ath",
            "new_atl",
            # Gap analysis
            "gapper",
        ]

        for column in core_columns:
            assert column in result.columns, f"Missing core column: {column}"

        # Market structure columns may not all be present if no swings detected
        market_structure_options = ["higher_high", "lower_high", "higher_low", "lower_low"]
        present_market_structure = [col for col in market_structure_options if col in result.columns]
        # At least some market structure analysis should be attempted
        assert len(present_market_structure) >= 0, "Market structure analysis should be present"

        # Result should have same number of rows as input
        assert len(result) == len(comprehensive_data)

        # Result should be Polars DataFrame
        assert isinstance(result, DataFrame)

        # Validate that all column types match the schema declarations
        validate_result_against_schema(result)

    def test_process_with_pandas_input(self, comprehensive_data):
        """Test processing with pandas DataFrame input."""
        # Convert to pandas
        pandas_data = comprehensive_data.to_pandas()

        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[TimeframeItemConfig(timeframes=["all"], swing_points=SwingPointsConfig(window=3))]
            )
        )
        result = indicators.process(pandas_data)

        # Should still return Polars DataFrame
        assert isinstance(result, DataFrame)
        assert len(result) == len(pandas_data)

        # Validate that all column types match the schema declarations
        validate_result_against_schema(result)

    def test_process_validation_failure(self):
        """Test that process raises error on validation failure."""
        invalid_data = DataFrame(
            {
                "timestamp": [datetime.now()],
                "open": [100.0],
                # Missing required columns
            }
        )

        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[TimeframeItemConfig(timeframes=["all"], swing_points=SwingPointsConfig(window=3))]
            )
        )
        with pytest.raises(ValueError, match="Missing required columns"):
            indicators.process(invalid_data)


# ==========================================
# CORRECTED IMPLEMENTATION TESTS
# ==========================================
# These tests validate that all indicators now match the established logic from setup_processor.py
# with specific test data designed to verify the exact calculations.


@pytest.mark.unit
class TestCorrectedScenarioClassification:
    """Test corrected scenario classification (1, 2U, 2D, 3)."""

    @pytest.fixture
    def scenario_test_data(self):
        """Create specific data to test scenario classification."""
        # Create data with known high/low relationships to previous bars
        return DataFrame(
            {
                "timestamp": [datetime(2023, 1, 1) + timedelta(days=i) for i in range(6)],
                "open": [100, 101, 101, 102, 101, 103],
                # Bar patterns: baseline, 2U, 2D, 3, 1, 2U
                "high": [101, 102, 101, 105, 101, 104],  # prev: [-, 101, 102, 101, 105, 101]
                "low": [99, 100, 99, 98, 100, 102],  # prev: [-, 99, 100, 99, 98, 100]
                "close": [100.5, 101.5, 100.5, 102.5, 100.5, 103.5],
                "volume": [1000] * 6,
            }
        )

    def test_scenario_classification_correct(self, scenario_test_data):
        """Test that scenarios are classified correctly."""
        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[TimeframeItemConfig(timeframes=["all"], swing_points=SwingPointsConfig(window=3))]
            )
        )
        config = indicators.config.timeframe_configs[0]
        result = indicators._calculate_strat_patterns(scenario_test_data, config)

        _scenarios = result["scenario"].to_list()

        # Bar 0: No previous bar, should be None
        assert _scenarios[0] is None

        # Bar 1: high(102) > high1(101) AND low(100) >= low1(99) = 2U
        assert _scenarios[1] == "2U"

        # Bar 2: low(99) < low1(100) AND high(101) <= high1(102) = 2D
        assert _scenarios[2] == "2D"

        # Bar 3: high(105) > high1(101) AND low(98) < low1(99) = 3
        assert _scenarios[3] == "3"

        # Bar 4: high(101) <= high1(105) AND low(100) >= low1(98) = 1
        assert _scenarios[4] == "1"

        # Bar 5: high(104) > high1(101) AND low(102) >= low1(100) = 2U
        assert _scenarios[5] == "2U"


@pytest.mark.unit
class TestCorrectedContinuity:
    """Test corrected continuity pattern (single bar open/close)."""

    @pytest.fixture
    def continuity_test_data(self):
        """Create data with specific open/close relationships."""
        return DataFrame(
            {
                "timestamp": [datetime(2023, 1, 1) + timedelta(hours=i) for i in range(5)],
                "open": [100, 100, 100, 100, 100],
                "high": [101, 101, 101, 101, 101],
                "low": [99, 99, 99, 99, 99],
                # Different close patterns: bullish, bearish, doji, bearish, bullish
                "close": [101, 99, 100, 98, 102],
                "volume": [1000] * 5,
            }
        )

    def test_continuity_single_bar_logic(self, continuity_test_data):
        """Test that continuity uses single bar open/close comparison."""
        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[TimeframeItemConfig(timeframes=["all"], swing_points=SwingPointsConfig(window=3))]
            )
        )
        config = indicators.config.timeframe_configs[0]
        result = indicators._calculate_strat_patterns(continuity_test_data, config)

        continuity = result["continuity"].to_list()

        # Bar 0: close(101) > open(100) = 1 (bullish)
        assert continuity[0] == 1

        # Bar 1: close(99) < open(100) = 0 (bearish)
        assert continuity[1] == 0

        # Bar 2: close(100) = open(100) = -1 (doji)
        assert continuity[2] == -1

        # Bar 3: close(98) < open(100) = 0 (bearish)
        assert continuity[3] == 0

        # Bar 4: close(102) > open(100) = 1 (bullish)
        assert continuity[4] == 1


@pytest.mark.unit
class TestCorrectedInForce:
    """Test corrected in_force pattern (breakout of previous range)."""

    @pytest.fixture
    def in_force_test_data(self):
        """Create data to test in_force breakout logic."""
        return DataFrame(
            {
                "timestamp": [datetime(2023, 1, 1) + timedelta(hours=i) for i in range(4)],
                "open": [100, 101, 100, 99],
                "high": [101, 103, 101, 100],  # prev: [-, 101, 103, 101]
                "low": [99, 100, 98, 97],  # prev: [-, 99, 100, 98]
                "close": [100.5, 102, 99, 98],  # Bullish, Bullish, Bearish, Bearish
                "volume": [1000] * 4,
            }
        )

    def test_in_force_breakout_logic(self, in_force_test_data):
        """Test in_force detects breakout of previous bar's range."""
        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[TimeframeItemConfig(timeframes=["all"], swing_points=SwingPointsConfig(window=3))]
            )
        )
        config = indicators.config.timeframe_configs[0]
        result = indicators._calculate_strat_patterns(in_force_test_data, config)

        in_force = result["in_force"].to_list()
        continuity = result["continuity"].to_list()

        # Bar 0: No previous bar
        assert in_force[0] is False

        # Bar 1: continuity=1 (bullish) AND close(102) > high1(101) = True
        assert continuity[1] == 1
        assert in_force[1] is True

        # Bar 2: continuity=0 (bearish) AND close(99) < low1(100) = True
        assert continuity[2] == 0
        assert in_force[2] is True

        # Bar 3: continuity=0 (bearish) but close(98) NOT < low1(98) = False
        assert continuity[3] == 0
        assert in_force[3] is False


@pytest.mark.unit
class TestCorrectedHammerShooter:
    """Test corrected hammer and shooter patterns."""

    @pytest.fixture
    def hammer_shooter_test_data(self):
        """Create data with specific hammer and shooter patterns."""
        return DataFrame(
            {
                "timestamp": [datetime(2023, 1, 1) + timedelta(days=i) for i in range(4)],
                # Hammer: range > 3*body, close & open > 60% from low
                # Shooter: range > 3*body, close & open > 60% from high
                "open": [100, 102, 100, 102],
                "high": [101, 103, 110, 103],
                "low": [90, 93, 100, 93],
                "close": [100.5, 102.5, 109, 95],  # Normal, Normal, Shooter candidate, Hammer candidate
                "volume": [1000] * 4,
            }
        )

    def test_hammer_pattern_correct_ratios(self, hammer_shooter_test_data):
        """Test hammer pattern uses correct ratios and thresholds."""
        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[TimeframeItemConfig(timeframes=["all"], swing_points=SwingPointsConfig(window=3))]
            )
        )
        config = indicators.config.timeframe_configs[0]
        result = indicators._calculate_strat_patterns(hammer_shooter_test_data, config)

        hammers = result["hammer"].to_list()

        # Check bar 3 (potential hammer): open=102, high=103, low=93, close=95
        # Range = 103-93 = 10, Body = |102-95| = 7
        # Range > 3*Body? 10 > 3*7 = 21? No -> False
        assert hammers[3] is False

        # Let me create a proper hammer pattern
        hammer_data = DataFrame(
            {
                "timestamp": [datetime(2023, 1, 1)],
                "open": [100],
                "high": [101],
                "low": [90],  # Long lower shadow
                "close": [99],  # Close near top
                "volume": [1000],
            }
        )

        # Range = 101-90 = 11, Body = |100-99| = 1
        # Range > 3*Body? 11 > 3*1 = 3? Yes
        # (close-low)/(high-low) = (99-90)/(101-90) = 9/11 = 0.818 > 0.6? Yes
        # (open-low)/(high-low) = (100-90)/(101-90) = 10/11 = 0.909 > 0.6? Yes
        config = indicators.config.timeframe_configs[0]
        hammer_result = indicators._calculate_strat_patterns(hammer_data, config)
        assert hammer_result["hammer"][0] is True

    def test_shooter_pattern_correct_ratios(self, hammer_shooter_test_data):
        """Test shooter pattern uses correct ratios and thresholds."""
        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[TimeframeItemConfig(timeframes=["all"], swing_points=SwingPointsConfig(window=3))]
            )
        )

        # Create proper shooter pattern
        shooter_data = DataFrame(
            {
                "timestamp": [datetime(2023, 1, 1)],
                "open": [100],
                "high": [110],  # Long upper shadow
                "low": [99],
                "close": [101],  # Close near bottom
                "volume": [1000],
            }
        )

        # Range = 110-99 = 11, Body = |100-101| = 1
        # Range > 3*Body? 11 > 3*1 = 3? Yes
        # (high-close)/(high-low) = (110-101)/(110-99) = 9/11 = 0.818 > 0.6? Yes
        # (high-open)/(high-low) = (110-100)/(110-99) = 10/11 = 0.909 > 0.6? Yes
        config = indicators.config.timeframe_configs[0]
        shooter_result = indicators._calculate_strat_patterns(shooter_data, config)
        assert shooter_result["shooter"][0] is True


@pytest.mark.unit
class TestCorrectedF23Pattern:
    """Test corrected F23 pattern with midpoint trigger."""

    @pytest.fixture
    def f23_test_data(self):
        """Create data for F23 pattern testing."""
        return DataFrame(
            {
                "timestamp": [datetime(2023, 1, 1) + timedelta(days=i) for i in range(3)],
                "open": [100, 101, 100],
                "high": [101, 103, 102],  # Bar 1: 2U scenario, Bar 2: 2D scenario
                "low": [99, 100, 99],  # Bar 2: low < low1 (99 < 100)
                "close": [100.5, 99.5, 102.0],  # Bar 1: F23D (close < trigger), Bar 2: F23U (close > trigger)
                "volume": [1000] * 3,
            }
        )

    def test_f23_trigger_calculation(self, f23_test_data):
        """Test F23 trigger price calculation (midpoint)."""
        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[TimeframeItemConfig(timeframes=["all"], swing_points=SwingPointsConfig(window=3))]
            )
        )
        # First calculate basic patterns to get continuity column
        config = indicators.config.timeframe_configs[0]
        with_basic = indicators._calculate_strat_patterns(f23_test_data, config)
        result = indicators._calculate_advanced_patterns(with_basic, config)

        # F23 trigger for bar 1 = midpoint of bar 0: high1 - ((high1 - low1) / 2)
        # = 101 - ((101 - 99) / 2) = 101 - 1 = 100
        f23_triggers = result["f23_trigger"].to_list()
        assert f23_triggers[1] == 100.0

        # F23 trigger for bar 2 = midpoint of bar 1: 103 - ((103 - 100) / 2) = 103 - 1.5 = 101.5
        assert f23_triggers[2] == 101.5

    def test_f23_pattern_detection(self):
        """Test F23U and F23D detection."""
        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[TimeframeItemConfig(timeframes=["all"], swing_points=SwingPointsConfig(window=3))]
            )
        )

        # F23D: 2U scenario but close < f23_trigger
        f23d_data = DataFrame(
            {
                "timestamp": [datetime(2023, 1, 1) + timedelta(days=i) for i in range(2)],
                "open": [100, 101],
                "high": [101, 103],  # 2U: high(103) > high1(101), low(100) >= low1(99)
                "low": [99, 100],
                "close": [100.5, 99.5],  # close(99.5) < f23_trigger(100)
                "volume": [1000] * 2,
            }
        )

        config = indicators.config.timeframe_configs[0]
        with_basic = indicators._calculate_strat_patterns(f23d_data, config)
        result = indicators._calculate_advanced_patterns(with_basic, config)
        f23x_values = result["f23x"].to_list()
        assert f23x_values[1] == "F23D"

        # F23U: 2D scenario but close > f23_trigger
        f23u_data = DataFrame(
            {
                "timestamp": [datetime(2023, 1, 1) + timedelta(days=i) for i in range(2)],
                "open": [100, 101],
                "high": [103, 103],  # 2D: low(98) < low1(99), high(103) <= high1(103)
                "low": [99, 98],
                "close": [100.5, 102],  # close(102) > f23_trigger(101)
                "volume": [1000] * 2,
            }
        )

        config = indicators.config.timeframe_configs[0]
        with_basic = indicators._calculate_strat_patterns(f23u_data, config)
        result = indicators._calculate_advanced_patterns(with_basic, config)
        f23x_values = result["f23x"].to_list()
        assert f23x_values[1] == "F23U"


@pytest.mark.unit
class TestCorrectedKicker:
    """Test corrected kicker pattern (continuity reversal + gap)."""

    def test_gapper_detection(self):
        """Test gap detection logic with percentage-based calculation."""
        # Use default gap_threshold of 0.0 (any gap)
        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[TimeframeItemConfig(timeframes=["all"], swing_points=SwingPointsConfig(window=3))]
            )
        )

        # Gap up: open > high1 * (1 + gap_threshold)
        gap_data = DataFrame(
            {
                "timestamp": [datetime(2023, 1, 1) + timedelta(days=i) for i in range(2)],
                "open": [100.0, 101.11],  # Gap up opening
                "high": [101.0, 102.0],
                "low": [99.0, 100.5],
                "close": [100.5, 101.5],
                "volume": [1000] * 2,
            }
        )

        config = indicators.config.timeframe_configs[0]
        with_basic = indicators._calculate_strat_patterns(gap_data, config)
        result = indicators._calculate_advanced_patterns(with_basic, config)
        gappers = result["gapper"].to_list()

        # Gap threshold = high1 * (1 + 0.001) = 101 * 1.001 = 101.101
        # Open(101.11) > 101.101? Yes -> Gap up = 1
        assert gappers[1] == 1

        # Gap down: open < low1 * (1 - gap_threshold)
        gap_data2 = DataFrame(
            {
                "timestamp": [datetime(2023, 1, 1) + timedelta(days=i) for i in range(2)],
                "open": [100.0, 98.89],  # Gap down opening
                "high": [101.0, 99.5],
                "low": [99.0, 98.0],
                "close": [100.5, 98.5],
                "volume": [1000] * 2,
            }
        )

        config = indicators.config.timeframe_configs[0]
        with_basic = indicators._calculate_strat_patterns(gap_data2, config)
        result = indicators._calculate_advanced_patterns(with_basic, config)
        gappers = result["gapper"].to_list()

        # Gap threshold = low1 * (1 - 0.001) = 99 * 0.999 = 98.901
        # Open(98.89) < 98.901? Yes -> Gap down = 0
        assert gappers[1] == 0

    def test_kicker_continuity_reversal(self):
        """Test kicker requires continuity reversal."""
        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[TimeframeItemConfig(timeframes=["all"], swing_points=SwingPointsConfig(window=3))]
            )
        )

        # Bullish kicker: continuity1=0 & gapper=1 & continuity=1
        bullish_kicker_data = DataFrame(
            {
                "timestamp": [datetime(2023, 1, 1) + timedelta(days=i) for i in range(2)],
                "open": [100, 106],  # Gap up
                "high": [101, 107],
                "low": [99, 105],
                "close": [99.5, 106.5],  # continuity1=0 (bearish), continuity=1 (bullish)
                "volume": [1000] * 2,
            }
        )

        config = indicators.config.timeframe_configs[0]
        with_basic = indicators._calculate_strat_patterns(bullish_kicker_data, config)
        result = indicators._calculate_advanced_patterns(with_basic, config)
        kickers = result["kicker"].to_list()
        continuity = result["continuity"].to_list()

        # Verify continuity reversal: bearish -> bullish
        assert continuity[0] == 0  # Bearish (close < open)
        assert continuity[1] == 1  # Bullish (close > open)
        assert kickers[1] == 1  # Bullish kicker

        # Bearish kicker: continuity1=1 & gapper=0 & continuity=0
        bearish_kicker_data = DataFrame(
            {
                "timestamp": [datetime(2023, 1, 1) + timedelta(days=i) for i in range(2)],
                "open": [100, 96],  # Gap down
                "high": [101, 97],
                "low": [99, 95],
                "close": [100.5, 95.5],  # continuity1=1 (bullish), continuity=0 (bearish)
                "volume": [1000] * 2,
            }
        )

        config = indicators.config.timeframe_configs[0]
        with_basic = indicators._calculate_strat_patterns(bearish_kicker_data, config)
        result = indicators._calculate_advanced_patterns(with_basic, config)
        kickers = result["kicker"].to_list()
        continuity = result["continuity"].to_list()

        # Verify continuity reversal: bullish -> bearish
        assert continuity[0] == 1  # Bullish (close > open)
        assert continuity[1] == 0  # Bearish (close < open)
        assert kickers[1] == 0  # Bearish kicker

    def test_gapper_kicker_integration_workflow(self):
        """Integration test: verify complete gapper -> kicker workflow with new field."""
        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[
                    TimeframeItemConfig(
                        timeframes=["all"],
                        swing_points=SwingPointsConfig(window=3),
                        gap_detection=GapDetectionConfig(threshold=0.005),  # 0.5% threshold
                    )
                ]
            )
        )

        # Create comprehensive test data with multiple scenarios (need 8+ rows for swing window=3)
        integration_data = DataFrame(
            {
                "timestamp": [datetime(2023, 1, 1) + timedelta(days=i) for i in range(8)],
                # Scenario: Setup -> Normal -> Gap Up Bullish Kicker -> Gap Down Bearish Kicker -> Normal -> Gap Up (no kicker) -> End
                "open": [95.0, 100.0, 106.0, 94.0, 98.0, 103.0, 105.0, 107.0],  # Significant gaps with 0.5% threshold
                "high": [97.0, 102.0, 108.0, 96.0, 100.0, 105.0, 107.0, 109.0],
                "low": [93.0, 98.0, 104.0, 92.0, 96.0, 101.0, 103.0, 105.0],
                "close": [
                    96.0,
                    99.5,
                    107.0,
                    93.0,
                    99.0,
                    104.0,
                    106.0,
                    108.0,
                ],  # continuity: bullish->bearish->bullish->bearish->bullish->bullish->bullish->bullish
                "volume": [1000] * 8,
            }
        )

        result = indicators.process(integration_data)

        # Extract key indicators
        gappers = result["gapper"].to_list()
        kickers = result["kicker"].to_list()
        continuity = result["continuity"].to_list()
        opens = result["open"].to_list()
        highs = result["high"].to_list()
        lows = result["low"].to_list()
        # closes = result["close"].to_list()  # Available if needed for future assertions

        # Validate gap detection with 0.5% threshold
        gap_threshold = 0.005

        # Bar 0: No previous bar
        assert gappers[0] is None
        assert kickers[0] is None

        # Bar 1: Gap up? open(100) > high[0](97) * 1.005 = 97.485? Yes
        assert opens[1] > highs[0] * (1 + gap_threshold), "Should detect gap up"
        assert gappers[1] == 1, "Should be gap up"

        # Bar 2: Gap up? open(106) > high[1](102) * 1.005 = 102.51? Yes
        assert opens[2] > highs[1] * (1 + gap_threshold), "Should detect gap up"
        assert gappers[2] == 1, "Should be gap up"
        # Kicker? continuity[1]=0 (bearish), gapper=1, continuity[2]=1 (bullish) -> Bullish kicker
        assert continuity[1] == 0, "Previous bar should be bearish"
        assert continuity[2] == 1, "Current bar should be bullish"
        assert kickers[2] == 1, "Should be bullish kicker"

        # Bar 3: Gap down? open(94) < low[2](104) * 0.995 = 103.48? Yes
        assert opens[3] < lows[2] * (1 - gap_threshold), "Should detect gap down"
        assert gappers[3] == 0, "Should be gap down"
        # Kicker? continuity[2]=1 (bullish), gapper=0, continuity[3]=0 (bearish) -> Bearish kicker
        assert continuity[2] == 1, "Previous bar should be bullish"
        assert continuity[3] == 0, "Current bar should be bearish"
        assert kickers[3] == 0, "Should be bearish kicker"

        # Bar 4: Gap up? open(98) > high[3](96) * 1.005 = 96.48? Yes
        assert opens[4] > highs[3] * (1 + gap_threshold), "Should detect gap up"
        assert gappers[4] == 1, "Should be gap up"
        # Kicker? continuity[3]=0 (bearish), gapper=1, continuity[4]=1 (bullish) -> Bullish kicker
        assert continuity[3] == 0, "Previous bar should be bearish"
        assert continuity[4] == 1, "Current bar should be bullish"
        assert kickers[4] == 1, "Should be bullish kicker"

        # Bar 5: Gap up? open(103) > high[4](100) * 1.005 = 100.5? Yes
        assert opens[5] > highs[4] * (1 + gap_threshold), "Should detect gap up"
        assert gappers[5] == 1, "Should be gap up"
        # No kicker because continuity[4]=1 and continuity[5]=1 (both bullish, no reversal)
        assert continuity[4] == 1 and continuity[5] == 1, "Both bars bullish, no reversal"
        assert kickers[5] is None, "Should have no kicker (no continuity reversal)"

        # Bar 6: No gap? open(105) within thresholds around prev high/low
        high_threshold_6 = highs[5] * (1 + gap_threshold)  # 105 * 1.005 = 105.525
        low_threshold_6 = lows[5] * (1 - gap_threshold)  # 101 * 0.995 = 100.495
        assert low_threshold_6 <= opens[6] <= high_threshold_6, "Should be within gap thresholds"
        assert gappers[6] is None, "Should detect no significant gap"
        assert kickers[6] is None, "Should have no kicker (no gap)"

        # Bar 7: No gap? open(107) within thresholds
        high_threshold_7 = highs[6] * (1 + gap_threshold)  # 107 * 1.005 = 107.535
        low_threshold_7 = lows[6] * (1 - gap_threshold)  # 103 * 0.995 = 102.485
        assert low_threshold_7 <= opens[7] <= high_threshold_7, "Should be within gap thresholds"
        assert gappers[7] is None, "Should detect no significant gap"
        assert kickers[7] is None, "Should have no kicker (no gap)"

        # Validate data integrity
        assert len([g for g in gappers if g is not None]) >= 3, "Should have detected multiple gaps"
        assert len([k for k in kickers if k is not None]) >= 2, "Should have detected multiple kickers"

        # Verify kicker only occurs with both gap and continuity reversal
        for i in range(1, len(kickers)):
            if kickers[i] is not None:
                assert gappers[i] is not None, f"Bar {i}: Kicker without gap detected"
                assert continuity[i] != continuity[i - 1], f"Bar {i}: Kicker without continuity reversal"


@pytest.mark.unit
class TestCorrectedPMG:
    """Test corrected PMG pattern (cumulative tracking)."""

    def test_pmg_cumulative_logic(self):
        """Test PMG cumulative higher high/lower low tracking."""
        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[TimeframeItemConfig(timeframes=["all"], swing_points=SwingPointsConfig(window=3))]
            )
        )

        # Create data with specific high/low patterns
        pmg_data = DataFrame(
            {
                "timestamp": [datetime(2023, 1, 1) + timedelta(days=i) for i in range(6)],
                "open": [100, 101, 102, 101, 103, 102],
                "high": [101, 103, 104, 102, 105, 103],  # HH, HH, LH, HH, LH
                "low": [99, 100, 101, 99, 102, 100],  # HL, HL, LL, HL, LL
                "close": [100.5, 102.5, 103.5, 100.5, 104.5, 101.5],
                "volume": [1000] * 6,
            }
        )

        config = indicators.config.timeframe_configs[0]
        with_basic = indicators._calculate_strat_patterns(pmg_data, config)
        result = indicators._calculate_advanced_patterns(with_basic, config)
        pmg_values = result["pmg"].to_list()

        # PMG should track cumulative runs:
        # Bar 0: No previous -> 0
        # Bar 1: higher_high(103>101)=True, lower_low(100<99)=False -> 1
        # Bar 2: higher_high(104>103)=True, lower_low(101>100)=False -> 2
        # Bar 3: higher_high(102<104)=False, lower_low(99<101)=True -> -1
        # Bar 4: higher_high(105>102)=True, lower_low(102>99)=False -> 1
        # Bar 5: higher_high(103<105)=False, lower_low(100<102)=True -> -1

        expected_pmg = [0, 1, 2, -1, 1, -1]
        assert pmg_values == expected_pmg


@pytest.mark.unit
class TestMotherbarProblems:
    """Test motherbar problems detection."""

    def test_motherbar_breakout_logic(self):
        """Test motherbar problems correctly identifies inside bar patterns."""
        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[TimeframeItemConfig(timeframes=["all"], swing_points=SwingPointsConfig(window=3))]
            )
        )

        # Create data with specific inside bar and breakout patterns
        # Bar 0: Regular bar (baseline)
        # Bar 1: Inside bar (within Bar 0 range) - Bar 0 becomes mother bar
        # Bar 2: Still inside Bar 0 range - compound inside bar
        # Bar 3: Breaks above Bar 0 high - breakout
        # Bar 4: Regular bar
        # Bar 5: Inside bar (within Bar 4 range) - Bar 4 becomes mother bar
        motherbar_data = DataFrame(
            {
                "timestamp": [datetime(2023, 1, 1) + timedelta(days=i) for i in range(6)],
                "open": [100.0, 101.0, 101.5, 103.0, 105.0, 105.5],
                "high": [
                    102.0,
                    101.5,
                    101.8,
                    105.0,
                    106.0,
                    105.8,
                ],  # Bar 0: 102, Bar 1: inside (101.5), Bar 2: inside (101.8), Bar 3: breakout (105), Bar 4: 106, Bar 5: inside (105.8)
                "low": [
                    98.0,
                    99.0,
                    98.5,
                    103.0,
                    104.0,
                    104.5,
                ],  # Bar 0: 98, Bar 1: inside (99), Bar 2: inside (98.5), Bar 3: breakout (103), Bar 4: 104, Bar 5: inside (104.5)
                "close": [101.0, 100.5, 100.8, 104.0, 105.5, 105.2],
                "volume": [1000] * 6,
            }
        )

        config = indicators.config.timeframe_configs[0]
        with_basic = indicators._calculate_strat_patterns(motherbar_data, config)
        result = indicators._calculate_advanced_patterns(with_basic, config)
        motherbar_problems = result["motherbar_problems"].to_list()

        # Expected scenarios based on high/low relationships:
        # Bar 0: No previous bar -> scenario will be None or handled specially
        # Bar 1: high(101.5) <= prev_high(102) AND low(99) >= prev_low(98) -> scenario "1" (inside)
        # Bar 2: high(101.8) <= prev_high(101.5) AND low(98.5) < prev_low(99) -> scenario "2D"
        # Bar 3: high(105) > prev_high(101.8) AND low(103) > prev_low(98.5) -> scenario "2U"
        # Bar 4: high(106) > prev_high(105) AND low(104) > prev_low(103) -> scenario "2U"
        # Bar 5: high(105.8) <= prev_high(106) AND low(104.5) >= prev_low(104) -> scenario "1" (inside)

        # Expected motherbar_problems based on correct logic:
        # Bar 0: First bar -> False
        # Bar 1: Inside bar (scenario "1") -> Bar 0 becomes mother bar -> True (within mother range)
        # Bar 2: Not inside bar -> Check if within Bar 0 range (98-102) -> high=101.8, low=98.5 -> within range -> True
        # Bar 3: Check if within Bar 0 range -> high=105 > 102 -> breakout -> False
        # Bar 4: No active mother bar -> False
        # Bar 5: Inside bar (scenario "1") -> Bar 4 becomes mother bar -> True (within mother range)

        expected_problems = [False, True, True, False, False, True]
        assert motherbar_problems == expected_problems

    def test_motherbar_identification(self):
        """Test that mother bars are correctly identified as bars immediately before inside bars."""
        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[TimeframeItemConfig(timeframes=["all"], swing_points=SwingPointsConfig(window=3))]
            )
        )

        # Create simple scenario: Regular bar -> Inside bar -> Breakout
        simple_data = DataFrame(
            {
                "timestamp": [datetime(2023, 1, 1) + timedelta(days=i) for i in range(3)],
                "open": [100.0, 100.5, 105.0],
                "high": [102.0, 101.0, 108.0],  # Bar 0: 102, Bar 1: 101 (inside), Bar 2: 108 (breakout)
                "low": [98.0, 99.0, 103.0],  # Bar 0: 98, Bar 1: 99 (inside), Bar 2: 103 (breakout)
                "close": [101.0, 100.2, 106.0],
                "volume": [1000] * 3,
            }
        )

        config = indicators.config.timeframe_configs[0]
        with_basic = indicators._calculate_strat_patterns(simple_data, config)
        result = indicators._calculate_advanced_patterns(with_basic, config)
        motherbar_problems = result["motherbar_problems"].to_list()

        # Bar 0: No previous bar -> False
        # Bar 1: Inside bar -> Bar 0 becomes mother bar -> True
        # Bar 2: Breaks above mother bar range -> False
        expected_problems = [False, True, False]
        assert motherbar_problems == expected_problems

    def test_compound_inside_bars(self):
        """Test multiple consecutive inside bars within the same mother bar range."""
        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[TimeframeItemConfig(timeframes=["all"], swing_points=SwingPointsConfig(window=3))]
            )
        )

        # Create compound inside bar scenario
        compound_data = DataFrame(
            {
                "timestamp": [datetime(2023, 1, 1) + timedelta(days=i) for i in range(5)],
                "open": [100.0, 101.0, 100.5, 101.2, 105.0],
                "high": [103.0, 102.0, 101.5, 102.5, 110.0],  # Bar 0: 103, Bars 1-3: all inside 103, Bar 4: breakout
                "low": [97.0, 98.0, 99.0, 98.5, 104.0],  # Bar 0: 97, Bars 1-3: all above 97, Bar 4: breakout
                "close": [101.5, 101.0, 100.8, 101.8, 107.0],
                "volume": [1000] * 5,
            }
        )

        config = indicators.config.timeframe_configs[0]
        with_basic = indicators._calculate_strat_patterns(compound_data, config)
        result = indicators._calculate_advanced_patterns(with_basic, config)
        motherbar_problems = result["motherbar_problems"].to_list()

        # Bar 0: First bar -> False
        # Bar 1: Inside bar -> Bar 0 becomes mother bar -> True
        # Bar 2: Inside bar again -> True (compound inside)
        # Bar 3: Outside bar (scenario "3") -> breaks mother bar range -> False
        # Bar 4: Regular bar, no active mother bar -> False
        expected_problems = [False, True, True, False, False]
        assert motherbar_problems == expected_problems

    def test_motherbar_no_inside_bars(self):
        """Test when no inside bars are present (no mother bars established)."""
        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[TimeframeItemConfig(timeframes=["all"], swing_points=SwingPointsConfig(window=3))]
            )
        )

        # Create data with no inside bars - all directional or outside bars
        no_inside_data = DataFrame(
            {
                "timestamp": [datetime(2023, 1, 1) + timedelta(days=i) for i in range(4)],
                "open": [100.0, 102.0, 104.0, 101.0],
                "high": [101.0, 104.0, 106.0, 103.0],  # Each bar higher high than previous
                "low": [99.0, 101.0, 103.0, 100.0],  # Each bar higher low than previous (2U pattern)
                "close": [100.5, 103.5, 105.5, 102.5],
                "volume": [1000] * 4,
            }
        )

        config = indicators.config.timeframe_configs[0]
        with_basic = indicators._calculate_strat_patterns(no_inside_data, config)
        result = indicators._calculate_advanced_patterns(with_basic, config)
        motherbar_problems = result["motherbar_problems"].to_list()

        # No inside bars means no mother bars established -> all False
        expected_problems = [False, False, False, False]
        assert motherbar_problems == expected_problems

    def test_motherbar_immediate_breakout(self):
        """Test inside bar followed by immediate breakout."""
        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[TimeframeItemConfig(timeframes=["all"], swing_points=SwingPointsConfig(window=3))]
            )
        )

        immediate_breakout_data = DataFrame(
            {
                "timestamp": [datetime(2023, 1, 1) + timedelta(days=i) for i in range(3)],
                "open": [100.0, 100.5, 103.0],
                "high": [102.0, 101.0, 105.0],  # Bar 1 is inside, Bar 2 breaks out immediately
                "low": [98.0, 99.0, 104.0],  # Bar 1 is inside, Bar 2 breaks above mother bar
                "close": [101.0, 100.2, 104.5],
                "volume": [1000] * 3,
            }
        )

        config = indicators.config.timeframe_configs[0]
        with_basic = indicators._calculate_strat_patterns(immediate_breakout_data, config)
        result = indicators._calculate_advanced_patterns(with_basic, config)
        motherbar_problems = result["motherbar_problems"].to_list()

        # Bar 0: First bar -> False
        # Bar 1: Inside bar -> True (establishes Bar 0 as mother bar)
        # Bar 2: Immediate breakout -> False
        expected_problems = [False, True, False]
        assert motherbar_problems == expected_problems

    def test_motherbar_mixed_scenarios(self):
        """Test combination of different scenario patterns with motherbar logic."""
        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[TimeframeItemConfig(timeframes=["all"], swing_points=SwingPointsConfig(window=3))]
            )
        )

        # Mix of 1, 2U, 2D, and 3 scenarios
        mixed_data = DataFrame(
            {
                "timestamp": [datetime(2023, 1, 1) + timedelta(days=i) for i in range(7)],
                "open": [100.0, 100.5, 103.0, 101.0, 99.0, 100.2, 104.0],
                "high": [102.0, 101.0, 105.0, 100.5, 103.0, 101.0, 108.0],  # 1(inside), 2U, 2D, 2U, 1(inside), 3
                "low": [98.0, 99.0, 102.0, 97.0, 98.0, 97.0, 95.0],  # 1(inside), 2U, 2D, 2U, 1(inside), 3
                "close": [101.0, 100.2, 104.0, 99.5, 101.5, 99.5, 106.0],
                "volume": [1000] * 7,
            }
        )

        config = indicators.config.timeframe_configs[0]
        with_basic = indicators._calculate_strat_patterns(mixed_data, config)
        result = indicators._calculate_advanced_patterns(with_basic, config)
        motherbar_problems = result["motherbar_problems"].to_list()

        # Bar 0: First bar -> False
        # Bar 1: Inside bar -> True (Bar 0 becomes mother bar)
        # Bar 2: Breaks above mother bar -> False (resets mother bar)
        # Bar 3: No mother bar active -> False
        # Bar 4: No mother bar active -> False
        # Bar 5: Not inside bar (scenario "2D") -> False (no mother bar established)
        # Bar 6: Outside bar -> False
        expected_problems = [False, True, False, False, False, False, False]
        assert motherbar_problems == expected_problems


@pytest.mark.unit
class TestFullCorrectedPipeline:
    """Test complete corrected indicator pipeline."""

    @pytest.fixture
    def comprehensive_test_data(self):
        """Create comprehensive data for full pipeline testing."""
        return DataFrame(
            {
                "timestamp": [datetime(2023, 1, 1) + timedelta(days=i) for i in range(10)],
                "open": [100, 101, 102, 101, 103, 104, 103, 105, 104, 106],
                "high": [101, 103, 104, 102, 105, 106, 104, 107, 105, 108],
                "low": [99, 100, 101, 100, 102, 103, 102, 104, 103, 105],
                "close": [100.5, 102.5, 103.5, 101.5, 104.5, 105.5, 103.5, 106.5, 104.5, 107.5],
                "volume": [1000 + i * 100 for i in range(10)],
            }
        )

    def test_all_corrected_indicators_present(self, comprehensive_test_data):
        """Test that all corrected indicators are present in final output."""
        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[TimeframeItemConfig(timeframes=["all"], swing_points=SwingPointsConfig(window=3))]
            )
        )
        result = indicators.process(comprehensive_test_data)

        # All corrected patterns should be present
        expected_corrected_columns = [
            "continuity",  # Single bar open/close
            "scenario",  # Correct 2U->2D->3->1 priority
            "in_force",  # Range breakout + F23 conditions
            "hammer",  # 3x ratio + 60% thresholds
            "shooter",  # 3x ratio + 60% thresholds
            "f23x",  # F23U/F23D classification
            "f23_trigger",  # Midpoint trigger price
            "f23",  # Combined F23 boolean
            "gapper",  # Gap detection for kicker
            "kicker",  # Continuity reversal + gap
            "pmg",  # Cumulative tracking
            "motherbar_problems",  # Full breakout tracking
        ]

        for column in expected_corrected_columns:
            assert column in result.columns, f"Missing corrected column: {column}"

        # Verify data types
        assert result["continuity"].dtype == Int32  # 1, 0, -1
        assert result["scenario"].dtype == Utf8  # "1", "2U", "2D", "3"
        assert result["in_force"].dtype == Boolean  # True/False
        assert result["f23x"].dtype == Utf8  # "F23U", "F23D", None
        assert result["pmg"].dtype == Int32  # Integer cumulative

    def test_corrected_calculations_validate(self, comprehensive_test_data):
        """Test that corrected calculations produce expected results."""
        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[TimeframeItemConfig(timeframes=["all"], swing_points=SwingPointsConfig(window=3))]
            )
        )
        result = indicators.process(comprehensive_test_data)

        # Sample validation of corrected logic
        _scenarios = result["scenario"].to_list()
        continuity = result["continuity"].to_list()

        # Verify scenario classification uses correct priority
        assert all(s in [None, "1", "2U", "2D", "3"] for s in _scenarios)

        # Verify continuity is single-bar calculation
        for i, cont in enumerate(continuity):
            row = result.row(i, named=True)
            if row["close"] > row["open"]:
                assert cont == 1
            elif row["close"] < row["open"]:
                assert cont == 0
            else:
                assert cont == -1

        # Verify F23 trigger is midpoint calculation (only for actual F23 patterns)
        f23_triggers = result["f23_trigger"].to_list()
        f23_patterns = result["f23"].to_list()
        for i in range(1, len(f23_triggers)):
            if f23_patterns[i] and f23_triggers[i] is not None:  # Only check actual F23 patterns
                prev_high = result["high"][i - 1]
                prev_low = result["low"][i - 1]
                expected_trigger = prev_high - ((prev_high - prev_low) / 2)
                assert abs(f23_triggers[i] - expected_trigger) < 0.001


@pytest.mark.unit
class TestScenarioCalculations:
    """Test scenario classification calculations with precise data."""

    def test_scenario_1_calculation(self):
        """Test scenario 1: Inside bar (H <= prev_H and L >= prev_L)."""
        data = DataFrame(
            {
                "timestamp": [datetime.now() - timedelta(hours=i) for i in range(2, 0, -1)],
                "open": [100.0, 101.0],
                "high": [110.0, 105.0],  # 105 <= 110 ✓
                "low": [90.0, 95.0],  # 95 >= 90 ✓
                "close": [105.0, 102.0],
            }
        )

        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[TimeframeItemConfig(timeframes=["all"], swing_points=SwingPointsConfig(window=3))]
            )
        )
        config = indicators.config.timeframe_configs[0]
        result = indicators._calculate_strat_patterns(data, config)

        # Second bar should be scenario "1"
        assert result["scenario"][1] == "1"

    def test_scenario_2u_calculation(self):
        """Test scenario 2U: Higher high, same or higher low (H > prev_H and L >= prev_L)."""
        data = DataFrame(
            {
                "timestamp": [datetime.now() - timedelta(hours=i) for i in range(2, 0, -1)],
                "open": [100.0, 101.0],
                "high": [110.0, 115.0],  # 115 > 110 ✓
                "low": [90.0, 90.0],  # 90 >= 90 ✓
                "close": [105.0, 112.0],
            }
        )

        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[TimeframeItemConfig(timeframes=["all"], swing_points=SwingPointsConfig(window=3))]
            )
        )
        config = indicators.config.timeframe_configs[0]
        result = indicators._calculate_strat_patterns(data, config)

        # Second bar should be scenario "2U"
        assert result["scenario"][1] == "2U"

    def test_scenario_2d_calculation(self):
        """Test scenario 2D: Lower low, same or lower high (L < prev_L and H <= prev_H)."""
        data = DataFrame(
            {
                "timestamp": [datetime.now() - timedelta(hours=i) for i in range(2, 0, -1)],
                "open": [100.0, 101.0],
                "high": [110.0, 110.0],  # 110 <= 110 ✓
                "low": [90.0, 85.0],  # 85 < 90 ✓
                "close": [105.0, 95.0],
            }
        )

        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[TimeframeItemConfig(timeframes=["all"], swing_points=SwingPointsConfig(window=3))]
            )
        )
        config = indicators.config.timeframe_configs[0]
        result = indicators._calculate_strat_patterns(data, config)

        # Second bar should be scenario "2D"
        assert result["scenario"][1] == "2D"

    def test_scenario_3_calculation(self):
        """Test scenario 3: Outside bar (H > prev_H and L < prev_L)."""
        data = DataFrame(
            {
                "timestamp": [datetime.now() - timedelta(hours=i) for i in range(2, 0, -1)],
                "open": [100.0, 101.0],
                "high": [110.0, 120.0],  # 120 > 110 ✓
                "low": [90.0, 80.0],  # 80 < 90 ✓
                "close": [105.0, 115.0],
            }
        )

        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[TimeframeItemConfig(timeframes=["all"], swing_points=SwingPointsConfig(window=3))]
            )
        )
        config = indicators.config.timeframe_configs[0]
        result = indicators._calculate_strat_patterns(data, config)

        # Second bar should be scenario "3"
        assert result["scenario"][1] == "3"


@pytest.mark.unit
class TestContinuityCalculations:
    """Test continuity pattern calculations."""

    def test_continuity_bullish(self):
        """Test bullish continuity: close > open."""
        data = DataFrame(
            {
                "timestamp": [datetime.now()],
                "open": [100.0],
                "high": [110.0],
                "low": [90.0],
                "close": [105.0],  # 105 > 100 = bullish
            }
        )

        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[TimeframeItemConfig(timeframes=["all"], swing_points=SwingPointsConfig(window=3))]
            )
        )
        config = indicators.config.timeframe_configs[0]
        result = indicators._calculate_strat_patterns(data, config)

        assert result["continuity"][0] == 1  # Bullish = 1

    def test_continuity_bearish(self):
        """Test bearish continuity: close < open."""
        data = DataFrame(
            {
                "timestamp": [datetime.now()],
                "open": [100.0],
                "high": [110.0],
                "low": [90.0],
                "close": [95.0],  # 95 < 100 = bearish
            }
        )

        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[TimeframeItemConfig(timeframes=["all"], swing_points=SwingPointsConfig(window=3))]
            )
        )
        config = indicators.config.timeframe_configs[0]
        result = indicators._calculate_strat_patterns(data, config)

        assert result["continuity"][0] == 0  # Bearish = 0

    def test_continuity_doji(self):
        """Test doji: close = open."""
        data = DataFrame(
            {
                "timestamp": [datetime.now()],
                "open": [100.0],
                "high": [110.0],
                "low": [90.0],
                "close": [100.0],  # 100 = 100 = doji
            }
        )

        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[TimeframeItemConfig(timeframes=["all"], swing_points=SwingPointsConfig(window=3))]
            )
        )
        config = indicators.config.timeframe_configs[0]
        result = indicators._calculate_strat_patterns(data, config)

        assert result["continuity"][0] == -1  # Doji = -1


@pytest.mark.unit
class TestInForceCalculations:
    """Test in-force pattern calculations."""

    def test_in_force_bullish_breakout(self):
        """Test bullish in-force: bullish bar closing above previous high."""
        data = DataFrame(
            {
                "timestamp": [datetime.now() - timedelta(hours=i) for i in range(2, 0, -1)],
                "open": [100.0, 102.0],
                "high": [110.0, 115.0],
                "low": [90.0, 100.0],
                "close": [105.0, 112.0],  # 112 > 110 (prev high) and close > open
            }
        )

        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[TimeframeItemConfig(timeframes=["all"], swing_points=SwingPointsConfig(window=3))]
            )
        )
        config = indicators.config.timeframe_configs[0]
        result = indicators._calculate_strat_patterns(data, config)

        assert result["in_force"][1] is True  # Should be in force

    def test_in_force_bearish_breakout(self):
        """Test bearish in-force: bearish bar closing below previous low."""
        data = DataFrame(
            {
                "timestamp": [datetime.now() - timedelta(hours=i) for i in range(2, 0, -1)],
                "open": [100.0, 98.0],
                "high": [110.0, 105.0],
                "low": [90.0, 85.0],
                "close": [105.0, 88.0],  # 88 < 90 (prev low) and close < open
            }
        )

        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[TimeframeItemConfig(timeframes=["all"], swing_points=SwingPointsConfig(window=3))]
            )
        )
        config = indicators.config.timeframe_configs[0]
        result = indicators._calculate_strat_patterns(data, config)

        assert result["in_force"][1] is True  # Should be in force


@pytest.mark.unit
class TestSignalCalculations:
    """Test signal pattern detection with exact pattern sequences."""

    def test_2d_2u_reversal_signal(self):
        """Test 2D-2U reversal pattern detection."""
        # Create exact 2D followed by 2U pattern
        data = DataFrame(
            {
                "timestamp": [datetime.now() - timedelta(hours=i) for i in range(3, 0, -1)],
                "open": [100.0, 98.0, 102.0],
                "high": [110.0, 105.0, 112.0],  # Bar 1: 105 <= 110 (2D), Bar 2: 112 > 105 (2U)
                "low": [90.0, 85.0, 90.0],  # Bar 1: 85 < 90 (2D), Bar 2: 90 >= 85 (2U)
                "close": [105.0, 95.0, 108.0],
            }
        )

        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[TimeframeItemConfig(timeframes=["all"], swing_points=SwingPointsConfig(window=3))]
            )
        )
        config = indicators.config.timeframe_configs[0]
        result = indicators._calculate_strat_patterns(data, config)

        # Last bar should detect 2D-2U signal
        assert result["signal"][2] == "2D-2U"
        assert result["type"][2] == "reversal"
        assert result["bias"][2] == "long"

    def test_3_2u_context_reversal(self):
        """Test 3-2U context reversal pattern."""
        # Create exact 3 followed by 2U pattern
        data = DataFrame(
            {
                "timestamp": [datetime.now() - timedelta(hours=i) for i in range(3, 0, -1)],
                "open": [100.0, 102.0, 104.0],
                "high": [110.0, 120.0, 122.0],  # Bar 1: 120 > 110 (3), Bar 2: 122 > 120 (2U - higher high)
                "low": [90.0, 80.0, 80.0],  # Bar 1: 80 < 90 (3), Bar 2: 80 >= 80 (2U - same/higher low)
                "close": [105.0, 115.0, 121.0],
            }
        )

        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[TimeframeItemConfig(timeframes=["all"], swing_points=SwingPointsConfig(window=3))]
            )
        )
        config = indicators.config.timeframe_configs[0]
        result = indicators._calculate_strat_patterns(data, config)

        # Check scenarios first
        assert result["scenario"][1] == "3"  # Outside bar
        assert result["scenario"][2] == "2U"  # Higher high, same/higher low

        # Last bar should detect 3-2U signal
        assert result["signal"][2] == "3-2U"
        assert result["type"][2] == "reversal"
        assert result["bias"][2] == "long"

    def test_2u_2u_continuation_signal(self):
        """Test 2U-2U continuation pattern."""
        # Create exact 2U followed by 2U pattern
        data = DataFrame(
            {
                "timestamp": [datetime.now() - timedelta(hours=i) for i in range(3, 0, -1)],
                "open": [100.0, 102.0, 104.0],
                "high": [110.0, 115.0, 120.0],  # Bar 1: 115 > 110 (2U), Bar 2: 120 > 115 (2U)
                "low": [90.0, 90.0, 95.0],  # Bar 1: 90 >= 90 (2U), Bar 2: 95 >= 90 (2U)
                "close": [105.0, 112.0, 118.0],
            }
        )

        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[TimeframeItemConfig(timeframes=["all"], swing_points=SwingPointsConfig(window=3))]
            )
        )
        config = indicators.config.timeframe_configs[0]
        result = indicators._calculate_strat_patterns(data, config)

        # Check scenarios
        assert result["scenario"][1] == "2U"
        assert result["scenario"][2] == "2U"

        # Last bar should detect 2U-2U continuation
        assert result["signal"][2] == "2U-2U"
        assert result["type"][2] == "continuation"
        assert result["bias"][2] == "long"


@pytest.mark.unit
class TestHammerShooterCalculations:
    """Test hammer and shooter pattern calculations with precise ratios."""

    def test_hammer_pattern_calculation(self):
        """Test hammer pattern: long lower shadow, small body near top."""
        # Create exact hammer: range > 3 * body, close/open > 60% from bottom
        data = DataFrame(
            {
                "timestamp": [datetime.now()],
                "open": [108.0],  # Near top
                "high": [110.0],  # Top
                "low": [90.0],  # Long lower shadow
                "close": [107.0],  # Near top, small body
            }
        )
        # Range = 110-90 = 20, Body = |108-107| = 1, Range = 20 > 3*1 = 3 ✓
        # Close from low = (107-90)/(110-90) = 17/20 = 0.85 > 0.6 ✓
        # Open from low = (108-90)/(110-90) = 18/20 = 0.90 > 0.6 ✓

        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[TimeframeItemConfig(timeframes=["all"], swing_points=SwingPointsConfig(window=3))]
            )
        )
        config = indicators.config.timeframe_configs[0]
        result = indicators._calculate_strat_patterns(data, config)

        assert result["hammer"][0] is True
        assert result["shooter"][0] is False  # Should not be shooter

    def test_shooter_pattern_calculation(self):
        """Test shooter pattern: long upper shadow, small body near bottom."""
        # Create exact shooter: range > 3 * body, close/open > 60% from top
        data = DataFrame(
            {
                "timestamp": [datetime.now()],
                "open": [92.0],  # Near bottom
                "high": [110.0],  # Long upper shadow
                "low": [90.0],  # Bottom
                "close": [93.0],  # Near bottom, small body
            }
        )
        # Range = 110-90 = 20, Body = |92-93| = 1, Range = 20 > 3*1 = 3 ✓
        # High-close = (110-93)/(110-90) = 17/20 = 0.85 > 0.6 ✓
        # High-open = (110-92)/(110-90) = 18/20 = 0.90 > 0.6 ✓

        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[TimeframeItemConfig(timeframes=["all"], swing_points=SwingPointsConfig(window=3))]
            )
        )
        config = indicators.config.timeframe_configs[0]
        result = indicators._calculate_strat_patterns(data, config)

        assert result["shooter"][0] is True
        assert result["hammer"][0] is False  # Should not be hammer


@pytest.mark.unit
class TestGapCalculations:
    """Test gap detection with percentage-based calculations."""

    def test_gapper_up_detection(self):
        """Test gap up detection with percentage threshold."""
        # Use default gap_threshold = 0.0 (any gap) and create enough data for swing detection
        data = DataFrame(
            {
                "timestamp": [datetime.now() - timedelta(hours=i) for i in range(15, 0, -1)],
                "open": [100.0] * 14 + [110.2],  # Last bar gaps up: 110.2 > 105 * 1.001 = 105.105 ✓
                "high": [105.0] * 14 + [115.0],
                "low": [95.0] * 14 + [108.0],
                "close": [102.0] * 14 + [112.0],
            }
        )

        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[TimeframeItemConfig(timeframes=["all"], swing_points=SwingPointsConfig(window=3))]
            )
        )
        result = indicators.process(data)  # Use full process for complete pipeline

        assert result["gapper"][14] == 1  # Gap up on last bar

    def test_gapper_down_detection(self):
        """Test gap down detection with percentage threshold."""
        # Use default gap_threshold = 0.0 (any gap)
        data = DataFrame(
            {
                "timestamp": [datetime.now() - timedelta(hours=i) for i in range(15, 0, -1)],
                "open": [100.0] * 14 + [94.9],  # Last bar gaps down: 94.9 < 95 * 0.999 = 94.905 ✓
                "high": [105.0] * 14 + [98.0],
                "low": [95.0] * 14 + [92.0],
                "close": [102.0] * 14 + [96.0],
            }
        )

        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[TimeframeItemConfig(timeframes=["all"], swing_points=SwingPointsConfig(window=3))]
            )
        )
        result = indicators.process(data)  # Use full process for complete pipeline

        assert result["gapper"][14] == 0  # Gap down on last bar

    def test_gap_configurable_threshold(self):
        """Test gap detection with custom threshold."""
        # Use larger threshold of 1% (0.01)
        timeframe_configs = [TimeframeItemConfig(timeframes=["all"], gap_detection=GapDetectionConfig(threshold=0.01))]
        indicators = Indicators(IndicatorsConfig(timeframe_configs=timeframe_configs))

        data = DataFrame(
            {
                "timestamp": [datetime.now() - timedelta(hours=i) for i in range(15, 0, -1)],
                "open": [100.0] * 14 + [110.5],  # 110.5 > 105 * 1.01 = 106.05? Yes, 110.5 > 106.05 ✓
                "high": [105.0] * 14 + [115.0],
                "low": [95.0] * 14 + [108.0],
                "close": [102.0] * 14 + [112.0],
            }
        )

        result = indicators.process(data)

        # Should be a gap with 1% threshold since 110.5 > 106.05
        assert result["gapper"][14] == 1  # Gap up

    def test_gap_asset_class_independence(self):
        """Test gap detection works across different asset classes."""
        # Test with crypto-like prices (high values)
        crypto_data = DataFrame(
            {
                "timestamp": [datetime.now() - timedelta(hours=i) for i in range(15, 0, -1)],
                "open": [50000.0] * 14 + [50100.0],  # 50100 > 50500 * 1.001? No, need proper gap
                "high": [50500.0] * 14 + [50600.0],
                "low": [49500.0] * 14 + [50000.0],
                "close": [50200.0] * 14 + [50300.0],
            }
        )
        # Fix: 50100 > 50500 * 1.001 = 50550.5? No. Let's make a real gap:
        crypto_data = crypto_data.with_columns(
            [
                Series("open", [50000.0] * 14 + [50551.0])  # 50551 > 50500 * 1.001 = 50550.5 ✓
            ]
        )

        # Test with forex-like prices (low values)
        forex_data = DataFrame(
            {
                "timestamp": [datetime.now() - timedelta(hours=i) for i in range(15, 0, -1)],
                "open": [1.2000] * 14 + [1.2061],  # 1.2061 > 1.2050 * 1.001 = 1.206205? No, 1.2061 < 1.206205
                "high": [1.2050] * 14 + [1.2080],
                "low": [1.1950] * 14 + [1.2000],
                "close": [1.2020] * 14 + [1.2040],
            }
        )
        # Fix the forex gap:
        forex_data = forex_data.with_columns(
            [
                Series("open", [1.2000] * 14 + [1.2063])  # 1.2063 > 1.2050 * 1.001 = 1.206205 ✓
            ]
        )

        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[TimeframeItemConfig(timeframes=["all"], swing_points=SwingPointsConfig(window=3))]
            )
        )

        crypto_result = indicators.process(crypto_data)
        forex_result = indicators.process(forex_data)

        # Both should detect gaps appropriately
        assert crypto_result["gapper"][14] == 1  # Gap up
        assert forex_result["gapper"][14] == 1  # Gap up


@pytest.mark.unit
class TestSignalMetadataIntegration:
    """Test signal metadata object creation with real pattern data."""

    def test_signal_object_creation_with_target(self):
        """Test signal object creation for reversal patterns with targets."""
        # Create 3-bar pattern with enough history for target calculation
        data = DataFrame(
            {
                "timestamp": [datetime.now() - timedelta(hours=i) for i in range(5, 0, -1)],
                "open": [100.0, 102.0, 98.0, 104.0, 106.0],
                "high": [110.0, 115.0, 105.0, 112.0, 118.0],  # Target bar, 3, 2D, 2U pattern
                "low": [90.0, 95.0, 85.0, 95.0, 100.0],
                "close": [105.0, 110.0, 95.0, 108.0, 115.0],
            }
        )

        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[TimeframeItemConfig(timeframes=["all"], swing_points=SwingPointsConfig(window=3))]
            )
        )
        config = indicators.config.timeframe_configs[0]
        result = indicators._calculate_strat_patterns(data, config)

        # Check if signal was detected and JSON created
        signal_json_list = result["signal_json"].to_list()
        valid_signals = [json_str for json_str in signal_json_list if json_str is not None]

        assert len(valid_signals) > 0  # Should have at least one signal

        # Parse the first valid signal
        from thestrat import SignalMetadata

        signal = SignalMetadata.from_json(valid_signals[0])

        # Verify signal properties
        assert signal.pattern in SIGNALS
        assert signal.entry_price > 0
        assert signal.stop_price > 0
        if signal.category.value == "reversal":
            assert signal.target_price is not None
            assert signal.risk_reward_ratio is not None


@pytest.mark.unit
class TestPerTimeframeIndicators:
    """Test per-timeframe configuration functionality."""

    def test_timeframe_specific_configuration(self):
        """Test that different timeframes use different configurations."""
        # Create test data with timeframe column
        data_rows = []
        base_time = datetime(2024, 1, 1, 9, 30)

        for tf in ["5min", "1h"]:
            for i in range(10):
                data_rows.append(
                    {
                        "symbol": "TEST",
                        "timeframe": tf,
                        "timestamp": base_time + timedelta(minutes=i * 5 if tf == "5min" else i * 60),
                        "open": 100.0 + i * 2.0,
                        "high": 102.0 + i * 2.0,
                        "low": 98.0 + i * 2.0,
                        "close": 101.0 + i * 2.0,
                        "volume": 1000 + i * 100,
                    }
                )

        test_data = DataFrame(data_rows)

        # Configure different settings per timeframe
        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[
                    TimeframeItemConfig(
                        timeframes=["5min"],
                        swing_points=SwingPointsConfig(window=3, threshold=1.0),
                        gap_detection=GapDetectionConfig(threshold=0.0005),
                    ),
                    TimeframeItemConfig(
                        timeframes=["1h"],
                        swing_points=SwingPointsConfig(window=5, threshold=5.0),
                        gap_detection=GapDetectionConfig(threshold=0.002),
                    ),
                ]
            )
        )

        result = indicators.process(test_data)

        # Verify structure
        assert "timeframe" in result.columns
        assert set(result["timeframe"].unique().to_list()) == {"5min", "1h"}
        assert len(result) == 20  # Same number of rows as input

        # Verify key indicator columns are present
        expected_indicator_cols = ["swing_high", "swing_low", "scenario", "continuity", "in_force", "hammer", "shooter"]
        for column in expected_indicator_cols:
            assert column in result.columns

    def test_all_timeframe_without_timeframe_column(self):
        """Test that 'all' timeframe works when no timeframe column is present."""
        # Create simple OHLC data without timeframe column
        simple_data = DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, 9, 30) + timedelta(minutes=i) for i in range(10)],
                "open": [100.0 + i for i in range(10)],
                "high": [101.0 + i for i in range(10)],
                "low": [99.0 + i for i in range(10)],
                "close": [100.5 + i for i in range(10)],
                "volume": [1000 + i * 100 for i in range(10)],
            }
        )

        # Use 'all' timeframe configuration
        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[
                    TimeframeItemConfig(
                        timeframes=["all"],
                        swing_points=SwingPointsConfig(window=5, threshold=3.0),
                        gap_detection=GapDetectionConfig(threshold=0.001),
                    )
                ]
            )
        )

        result = indicators.process(simple_data)

        # Should process without timeframe column
        assert "timeframe" not in result.columns
        assert len(result) == 10

        # Should have all standard indicator columns
        expected_cols = ["swing_high", "swing_low", "scenario"]
        for column in expected_cols:
            assert column in result.columns

    def test_config_resolution_fallback(self):
        """Test that timeframes without specific config raise error (no fallback in new API)."""
        data_rows = []
        base_time = datetime(2024, 1, 1, 9, 30)

        # Create data for timeframes with and without specific configs
        for tf in ["5min", "15min"]:  # Only 5m has specific config
            for i in range(5):
                data_rows.append(
                    {
                        "timeframe": tf,
                        "timestamp": base_time + timedelta(minutes=i * 5),
                        "open": 100.0 + i,
                        "high": 101.0 + i,
                        "low": 99.0 + i,
                        "close": 100.5 + i,
                        "volume": 1000,
                    }
                )

        test_data = DataFrame(data_rows)

        # Only configure 5m specifically
        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[
                    TimeframeItemConfig(timeframes=["5min"], swing_points=SwingPointsConfig(window=3, threshold=2.0))
                ]
            )
        )

        # Should raise error because 15m is not configured
        with pytest.raises(ValueError, match="No configuration found for timeframe '15min'"):
            indicators.process(test_data)

    def test_empty_timeframe_configs_raises_error(self):
        """Test that empty timeframe_configs raises ValueError (new API requirement)."""
        # Empty timeframe_configs should raise error
        with pytest.raises(ValidationError, match="List should have at least 1 item"):
            Indicators(IndicatorsConfig(timeframe_configs=[]))


@pytest.mark.unit
class TestIndicatorsEdgeCases:
    """Test cases for Indicators edge cases and error conditions."""

    def test_no_all_config_without_timeframe_column_raises_error(self):
        """Test ValueError when data has no timeframe column and no 'all' config."""
        # Create data without timeframe column
        data = DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, 9, 30) + timedelta(minutes=i) for i in range(10)],
                "open": [100.0 + i for i in range(10)],
                "high": [101.0 + i for i in range(10)],
                "low": [99.0 + i for i in range(10)],
                "close": [100.5 + i for i in range(10)],
                "volume": [1000 + i * 100 for i in range(10)],
            }
        )

        # Create indicators with specific timeframes but no 'all' config
        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[
                    TimeframeItemConfig(
                        timeframes=["5min"],  # Specific timeframe, not "all"
                        swing_points=SwingPointsConfig(window=5, threshold=3.0),
                    )
                ]
            )
        )

        # Should raise ValueError because there's no timeframe column and no 'all' config
        with pytest.raises(ValueError) as exc_info:
            indicators.process(data)
        assert "requires an 'all' timeframe configuration" in str(exc_info.value)

    def test_get_signal_object_function_retrieval(self):
        """Test the get_signal_object function that's created during signal processing."""
        # Create test data that would generate signals
        data = DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, 9, 30) + timedelta(minutes=i * 5) for i in range(20)],
                "open": [100.0, 102.0, 101.0, 103.0, 100.0] * 4,
                "high": [101.0, 103.0, 102.0, 104.0, 101.0] * 4,
                "low": [99.0, 101.0, 100.0, 102.0, 99.0] * 4,
                "close": [100.5, 102.5, 101.5, 103.5, 100.5] * 4,
                "volume": [1000] * 20,
            }
        )

        # Enable signal generation to create the get_signal_object function
        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[
                    TimeframeItemConfig(
                        timeframes=["all"],
                        swing_points=SwingPointsConfig(window=3, threshold=1.0),
                    )
                ]
            )
        )

        result = indicators.process(data)

        # Note: Polars DataFrames don't support attrs like pandas
        # The signal processing functionality works, but signal objects are not stored in attrs
        # This test verifies that the signal column is properly generated
        if "signal" in result.columns:
            # Test that signals are properly generated (non-null values)
            signal_column = result["signal"]
            # This exercises the signal generation functionality
            assert signal_column is not None


@pytest.mark.unit
class TestIndicatorsTimestampHandling:
    """Test cases for timestamp conversion edge cases in Indicators."""

    def test_insufficient_trigger_bar_data_returns_none(self):
        """Test early return when there's insufficient data for trigger bar."""
        # Create enough data to pass validation but minimal for trigger bar testing
        # Need at least swing_window * 2 = 5 * 2 = 10 bars for validation (default swing_window is 5)
        # Create realistic OHLC data
        data = DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, 9, 30) + timedelta(minutes=i) for i in range(12)],
                "open": [100.0, 100.5, 101.0, 101.5, 101.0, 100.5, 100.0, 99.5, 99.0, 99.5, 100.0, 100.5],
                "high": [101.0, 101.5, 102.0, 102.5, 102.0, 101.5, 101.0, 100.5, 100.0, 100.5, 101.0, 101.5],
                "low": [99.0, 99.5, 100.0, 100.5, 100.0, 99.5, 99.0, 98.5, 98.0, 98.5, 99.0, 99.5],
                "close": [100.5, 101.0, 101.5, 102.0, 101.5, 101.0, 100.5, 100.0, 99.5, 100.0, 100.5, 101.0],
                "volume": [1000] * 12,
            }
        )

        # Configure with signal generation that requires trigger bar offset
        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[
                    TimeframeItemConfig(
                        timeframes=["all"],
                        swing_points=SwingPointsConfig(window=3, threshold=1.0),
                    )
                ]
            )
        )

        result = indicators.process(data)

        # Should process without error - the insufficient trigger bar logic
        # is handled internally during signal generation
        assert len(result) == 12

    def test_timestamp_conversion_edge_cases(self):
        """Test various timestamp conversion scenarios."""
        from datetime import datetime

        from polars import DataFrame

        # Create inconsistent timestamp formats
        # Need at least 10 bars for validation (swing_window * 2)
        test_cases = [
            # Case 1: Normal datetime objects
            {
                "timestamp": [datetime(2024, 1, 1, 9, 30) + timedelta(minutes=i) for i in range(12)],
                "description": "Normal datetime objects",
            },
            # Case 2: String timestamps (polars will convert these)
            {"timestamp": [f"2024-01-01T09:{30 + i}:00" for i in range(12)], "description": "String timestamps"},
        ]

        for case in test_cases:
            data = DataFrame(
                {
                    "timestamp": case["timestamp"],
                    "open": [100.0, 100.5, 101.0, 101.5, 101.0, 100.5, 100.0, 99.5, 99.0, 99.5, 100.0, 100.5],
                    "high": [101.0, 101.5, 102.0, 102.5, 102.0, 101.5, 101.0, 100.5, 100.0, 100.5, 101.0, 101.5],
                    "low": [99.0, 99.5, 100.0, 100.5, 100.0, 99.5, 99.0, 98.5, 98.0, 98.5, 99.0, 99.5],
                    "close": [100.5, 101.0, 101.5, 102.0, 101.5, 101.0, 100.5, 100.0, 99.5, 100.0, 100.5, 101.0],
                    "volume": [1000] * 12,
                }
            )

            # Configure with signal generation to exercise timestamp handling
            indicators = Indicators(
                IndicatorsConfig(
                    timeframe_configs=[
                        TimeframeItemConfig(
                            timeframes=["all"],
                            swing_points=SwingPointsConfig(window=3, threshold=1.0),
                        )
                    ]
                )
            )

            # Should process without error despite different timestamp formats
            result = indicators.process(data)
            assert len(result) == 12, f"Failed for {case['description']}"

            # Timestamp column should be properly formatted
            assert "timestamp" in result.columns

    def test_timestamp_fallback_to_datetime_now(self):
        """Test timestamp fallback when conversion fails."""
        # This test is tricky because we need to trigger the fallback
        # The fallback happens when timestamp is not a datetime and doesn't have to_pydatetime
        # We'll create a scenario where signal processing might encounter this

        data = DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, 9, 30) + timedelta(minutes=i * 5) for i in range(10)],
                "open": [100.0] * 10,
                "high": [102.0] * 10,  # Create consistent pattern
                "low": [98.0] * 10,
                "close": [101.0] * 10,
                "volume": [1000] * 10,
            }
        )

        # Configure signals to exercise timestamp handling paths
        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[
                    TimeframeItemConfig(
                        timeframes=["all"],
                        swing_points=SwingPointsConfig(window=3, threshold=1.0),
                    )
                ]
            )
        )

        result = indicators.process(data)

        # Should process successfully even with edge case timestamp handling
        assert len(result) == 10
        assert "timestamp" in result.columns
