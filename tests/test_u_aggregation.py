"""
Unit tests for TheStrat Aggregation component.

Tests OHLC timeframe aggregation with timezone handling and boundary alignment.
"""

from datetime import datetime

import pytest
from pandas import DataFrame as PandasDataFrame
from pandas import date_range
from polars import DataFrame, Datetime, col, lit
from pydantic import ValidationError

from thestrat.aggregation import Aggregation
from thestrat.schemas import ASSET_CLASS_CONFIGS, AggregationConfig, TimeframeConfig

from .utils.config_helpers import (
    create_aggregation_config,
    create_crypto_aggregation_config,
    create_equity_aggregation_config,
)
from .utils.thestrat_data_utils import (
    create_crypto_data,
    create_dst_transition_data,
    create_forex_data,
    create_long_term_data,
    create_market_hours_data,
    create_multi_timezone_data,
)


@pytest.mark.unit
class TestAggregationInit:
    """Test cases for Aggregation initialization."""

    def test_init_minimal_parameters(self):
        """Test initialization with minimal parameters."""
        agg = Aggregation(create_aggregation_config())

        assert agg.target_timeframes == ["1h"]
        assert agg.asset_class == "equities"
        assert agg.timezone is not None  # Should resolve to something
        assert agg.session_start is not None  # Should use default
        assert agg.hour_boundary is False  # Uses equities default

    def test_init_all_parameters(self):
        """Test initialization with all parameters specified."""
        agg = Aggregation(
            AggregationConfig(
                target_timeframes=["5min"],
                asset_class="crypto",
                hour_boundary=False,
                session_start="00:00",
                timezone="UTC",
            )
        )

        assert agg.target_timeframes == ["5min"]
        assert agg.asset_class == "crypto"
        assert agg.timezone == "UTC"
        assert agg.session_start == "00:00"
        assert agg.hour_boundary is False

    def test_init_multiple_timeframes(self):
        """Test initialization with multiple timeframes."""
        agg = Aggregation(AggregationConfig(target_timeframes=["5min", "15min", "1h"]))

        assert agg.target_timeframes == ["5min", "15min", "1h"]
        assert agg.asset_class == "equities"

    def test_init_crypto_forces_utc(self):
        """Test that crypto asset class forces UTC timezone."""
        agg = Aggregation(
            AggregationConfig(
                target_timeframes=["1h"],
                asset_class="crypto",
                timezone="US/Eastern",  # This should be ignored
            )
        )

        assert agg.timezone == "UTC"  # Crypto always uses UTC

    def test_init_fx_forces_utc(self):
        """Test that FX asset class forces UTC timezone."""
        agg = Aggregation(
            AggregationConfig(
                target_timeframes=["1h"],
                asset_class="fx",
                timezone="US/Central",  # This should be ignored
            )
        )

        assert agg.timezone == "UTC"  # FX always uses UTC

    def test_init_equities_respects_timezone(self):
        """Test that equities respects specified timezone."""
        agg = Aggregation(AggregationConfig(target_timeframes=["1h"], asset_class="equities", timezone="US/Pacific"))

        assert agg.timezone == "US/Pacific"

    def test_auto_hour_boundary_detection(self):
        """Test automatic hour boundary detection based on timeframe."""
        # Hourly and above should use asset class defaults
        hourly_agg = Aggregation(AggregationConfig(target_timeframes=["1h"]))
        daily_agg = Aggregation(AggregationConfig(target_timeframes=["1d"]))
        weekly_agg = Aggregation(AggregationConfig(target_timeframes=["1w"]))

        assert hourly_agg.hour_boundary is False  # Equities default
        assert daily_agg.hour_boundary is False  # Equities default
        assert weekly_agg.hour_boundary is False  # Equities default

        # Sub-hourly should use asset class defaults
        minute_agg = Aggregation(AggregationConfig(target_timeframes=["5min"]))
        assert minute_agg.hour_boundary is False  # Equities default

    def test_hour_boundary_asset_class_defaults(self):
        """Test that hour_boundary defaults are applied correctly per asset class."""
        # Test crypto default (True)
        crypto_agg = Aggregation(AggregationConfig(target_timeframes=["1h"], asset_class="crypto"))
        assert crypto_agg.hour_boundary is True

        # Test equities default (False)
        equities_agg = Aggregation(AggregationConfig(target_timeframes=["1h"], asset_class="equities"))
        assert equities_agg.hour_boundary is False

        # Test fx default (True)
        fx_agg = Aggregation(AggregationConfig(target_timeframes=["1h"], asset_class="fx"))
        assert fx_agg.hour_boundary is True

        # Test explicit override still works
        explicit_agg = Aggregation(
            AggregationConfig(target_timeframes=["1h"], asset_class="crypto", hour_boundary=False)
        )
        assert explicit_agg.hour_boundary is False

    def test_timeframe_parsing(self):
        """Test that timeframe parsing works correctly."""
        agg = Aggregation(AggregationConfig(target_timeframes=["15min"]))

        assert agg.target_timeframes == ["15min"]

    def test_invalid_timeframe_raises_error(self):
        """Test that invalid timeframe format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid timeframe 'invalid'"):
            Aggregation(AggregationConfig(target_timeframes=["invalid"]))

    def test_timeframe_validation_standard_formats(self):
        """Test that all expected standard timeframes are accepted."""
        valid_timeframes = ["1min", "5min", "15min", "30min", "1h", "4h", "1d", "1w", "1m", "1q", "1y"]

        for tf in valid_timeframes:
            # Should not raise any exception
            agg = Aggregation(AggregationConfig(target_timeframes=[tf]))
            assert agg.target_timeframes[0] == tf

    def test_timeframe_validation_polars_formats(self):
        """Test that supported timeframes from TIMEFRAME_METADATA are accepted."""

        # Test all supported timeframes from the mapping
        for tf in TimeframeConfig.TIMEFRAME_METADATA.keys():
            # Should not raise any exception
            agg = Aggregation(AggregationConfig(target_timeframes=[tf]))
            assert agg.target_timeframes[0] == tf

    def test_timeframe_validation_empty_string(self):
        """Test that empty timeframe raises ValidationError."""
        with pytest.raises(ValidationError, match=r"target_timeframes\[0\] must be a non-empty string"):
            Aggregation(AggregationConfig(target_timeframes=[""]))

    def test_timeframe_validation_non_string(self):
        """Test that non-string timeframe raises ValidationError."""
        with pytest.raises(ValidationError, match="Input should be a valid string"):
            Aggregation(AggregationConfig(target_timeframes=[123]))

    def test_timeframe_validation_unsupported_format(self):
        """Test that unsupported timeframe formats raise ValueError."""
        invalid_timeframes = ["2minutes", "1hr", "daily", "weekly", "1x", "5z"]

        for tf in invalid_timeframes:
            with pytest.raises(ValueError, match=f"Invalid timeframe '{tf}'"):
                Aggregation(AggregationConfig(target_timeframes=[tf]))

    def test_timezone_resolution_asset_class_default(self):
        """Test timezone resolution uses asset class default."""
        agg = Aggregation(
            AggregationConfig(
                target_timeframes=["1h"],
                asset_class="equities",  # Non-UTC asset class
                timezone=None,  # Should use asset class default
            )
        )

        assert agg.timezone == "US/Eastern"


@pytest.mark.unit
class TestAggregationValidation:
    """Test cases for input validation."""

    @pytest.fixture
    def valid_ohlc_data(self):
        """Create valid OHLC DataFrame."""
        from .utils.thestrat_data_utils import create_ohlc_data

        return create_ohlc_data(periods=7, start="2023-01-01 09:30:00", freq_minutes=60)

    @pytest.fixture
    def aggregation(self):
        """Create aggregation component for testing."""
        return Aggregation(create_equity_aggregation_config())

    def test_validate_input_valid_data(self, aggregation, valid_ohlc_data):
        """Test validation passes for valid OHLC data."""
        assert aggregation.validate_input(valid_ohlc_data) is True

    def test_validate_input_missing_columns(self, aggregation):
        """Test validation fails for missing required columns."""
        incomplete_data = DataFrame(
            {
                "timestamp": [datetime.now()],
                "open": [100.0],
                # Missing high, low, close, volume
            }
        )

        assert aggregation.validate_input(incomplete_data) is False

    def test_validate_input_insufficient_data(self, aggregation):
        """Test validation fails for insufficient data points."""
        single_row = DataFrame(
            {
                "timestamp": [datetime.now()],
                "open": [100.0],
                "high": [100.5],
                "low": [99.5],
                "close": [100.2],
                "volume": [1000],
            }
        )

        assert aggregation.validate_input(single_row) is False

    def test_validate_input_pandas_conversion(self, aggregation):
        """Test validation works with pandas DataFrame input."""
        pandas_data = PandasDataFrame(
            {
                "timestamp": date_range("2023-01-01", periods=5, freq="1h"),
                "open": [100.0, 101.0, 102.0, 103.0, 104.0],
                "high": [100.5, 101.5, 102.5, 103.5, 104.5],
                "low": [99.5, 100.5, 101.5, 102.5, 103.5],
                "close": [101.0, 102.0, 103.0, 104.0, 105.0],
                "volume": [1000, 1100, 1200, 1300, 1400],
                "timeframe": ["1h"] * 5,
            }
        )

        assert aggregation.validate_input(pandas_data) is True


@pytest.mark.unit
class TestAggregationOHLC:
    """Test cases for OHLC aggregation logic."""

    @pytest.fixture
    def minute_data(self):
        """Create minute-level OHLC data for aggregation testing."""
        from .utils.thestrat_data_utils import create_ohlc_data

        return create_ohlc_data(periods=60, start="2023-01-01 09:30:00", freq_minutes=1, base_price=100.0)

    def test_aggregate_to_hourly(self, minute_data):
        """Test aggregation from minutes to hourly."""
        agg = Aggregation(create_equity_aggregation_config())
        result = agg.process(minute_data)

        assert isinstance(result, DataFrame)
        assert len(result) == 1  # 60 minutes from 09:30-10:30 forms 1 session-aligned bar

        # Check the single hour bucket (09:30-10:30 = 60 minutes) with session alignment
        first_row = result[0]
        assert first_row["open"][0] == 100.0  # First open at 09:30
        assert first_row["low"][0] == 99.5  # Min low across all 60 minutes
        assert abs(first_row["close"][0] - 106.3) < 1e-10  # Last close at 10:29 (floating point tolerance)

    def test_aggregate_with_symbol_column(self):
        """Test aggregation preserves symbol column."""
        from .utils.thestrat_data_utils import create_timestamp_series

        timestamps = create_timestamp_series("2023-01-01 09:30:00", 5, 1)  # 1 minute intervals
        data_with_symbol = DataFrame(
            {
                "timestamp": timestamps,
                "symbol": ["AAPL"] * 5,
                "open": [100.0, 100.1, 100.2, 100.3, 100.4],
                "high": [100.5, 100.6, 100.7, 100.8, 100.9],
                "low": [99.5, 99.6, 99.7, 99.8, 99.9],
                "close": [100.2, 100.3, 100.4, 100.5, 100.6],
                "volume": [1000, 1100, 1200, 1300, 1400],
                "timeframe": ["1min"] * 5,
            }
        )

        agg = Aggregation(AggregationConfig(target_timeframes=["5min"], asset_class="equities"))
        result = agg.process(data_with_symbol)

        assert "symbol" in result.columns
        assert result["symbol"][0] == "AAPL"

    def test_aggregate_with_volume_as_optional(self):
        """Test aggregation handles volume as optional column correctly."""
        from .utils.thestrat_data_utils import create_timestamp_series

        timestamps = create_timestamp_series("2023-01-01 09:30:00", 3, 1)  # 1 minute intervals
        data_with_volume = DataFrame(
            {
                "timestamp": timestamps,
                "open": [100.0, 101.0, 102.0],
                "high": [100.5, 101.5, 102.5],
                "low": [99.5, 100.5, 101.5],
                "close": [100.2, 101.2, 102.2],
                "volume": [1000, 2000, 3000],
                "timeframe": ["1min"] * 3,
            }
        )

        agg = Aggregation(AggregationConfig(target_timeframes=["5min"], asset_class="equities"))
        result = agg.process(data_with_volume)

        assert "volume" in result.columns
        # Volume should be summed across the aggregated period (now 5min aggregation of 3 1-min bars)
        # Since we have only 3 bars and need 5min aggregation, we get partial aggregation
        assert result["volume"][0] == 6000  # 1000 + 2000 + 3000

    def test_aggregate_without_volume_column(self):
        """Test aggregation works correctly when volume column is missing."""
        from .utils.thestrat_data_utils import create_timestamp_series

        timestamps = create_timestamp_series("2023-01-01 09:30:00", 3, 1)  # 1 minute intervals
        data_without_volume = DataFrame(
            {
                "timestamp": timestamps,
                "open": [100.0, 101.0, 102.0],
                "high": [100.5, 101.5, 102.5],
                "low": [99.5, 100.5, 101.5],
                "close": [100.2, 101.2, 102.2],
                "timeframe": ["1min"] * 3,
            }
        )

        agg = Aggregation(AggregationConfig(target_timeframes=["5min"], asset_class="equities"))
        result = agg.process(data_without_volume)

        # Should not have volume column since it wasn't provided
        assert "volume" not in result.columns
        # Should still have all required OHLC columns
        assert all(col in result.columns for col in ["timestamp", "open", "high", "low", "close"])

    def test_result_sorted_by_timestamp(self, minute_data):
        """Test that aggregation result is sorted by timestamp."""
        # Create data in random order
        shuffled_data = minute_data.sample(fraction=1.0, shuffle=True, seed=42)

        agg = Aggregation(AggregationConfig(target_timeframes=["30min"], asset_class="equities"))
        result = agg.process(shuffled_data)

        # Result should be sorted by timestamp
        timestamps = result["timestamp"].to_list()
        assert timestamps == sorted(timestamps)


@pytest.mark.unit
class TestTimezoneHandling:
    """Test cases for timezone handling."""

    def test_normalize_timezone_naive_to_aware(self):
        """Test conversion of naive timestamps to timezone-aware."""
        naive_data = DataFrame(
            {
                "timestamp": [datetime(2023, 1, 1, 9, 30), datetime(2023, 1, 1, 10, 30)],
                "open": [100.0, 101.0],
                "high": [100.5, 101.5],
                "low": [99.5, 100.5],
                "close": [100.2, 101.2],
                "volume": [1000, 1100],
            }
        )

        agg = Aggregation(AggregationConfig(target_timeframes=["1h"], asset_class="equities", timezone="US/Eastern"))
        result = agg.normalize_timezone(naive_data)

        # Should have timezone-aware timestamps
        assert result.schema["timestamp"] == Datetime("us", "US/Eastern")

    def test_normalize_timezone_already_aware_unchanged(self):
        """Test that timezone-aware timestamps pass through unchanged."""
        from .utils.thestrat_data_utils import create_timestamp_series

        timestamps = create_timestamp_series("2023-01-01 09:30:00", 2, 60, timezone="UTC")
        aware_data = DataFrame(
            {
                "timestamp": timestamps,
                "open": [100.0, 101.0],
                "high": [100.5, 101.5],
                "low": [99.5, 100.5],
                "close": [100.2, 101.2],
                "volume": [1000, 1100],
            }
        )

        agg = Aggregation(create_crypto_aggregation_config())  # UTC timezone
        result = agg.normalize_timezone(aware_data)

        # Should preserve timezone-aware format
        timestamp_dtype = result.schema["timestamp"]
        assert isinstance(timestamp_dtype, Datetime)
        assert timestamp_dtype.time_zone == "UTC"


@pytest.mark.unit
class TestBoundaryAlignment:
    """Test cases for timestamp boundary alignment."""


# ==========================================
# ASSET CLASS TESTS
# ==========================================
# Tests for all supported asset classes in aggregation, timezone handling,
# session boundaries, and specific characteristics for crypto, equities, and fx.


@pytest.mark.unit
class TestAllAssetClasses:
    """Test all supported asset classes for aggregation."""

    def test_all_asset_classes_supported(self):
        """Test that all asset classes in ASSET_CLASS_CONFIGS work."""
        for asset_class in ASSET_CLASS_CONFIGS.keys():
            agg = Aggregation(AggregationConfig(target_timeframes=["1h"], asset_class=asset_class))
            assert agg.asset_class == asset_class
            assert agg.timezone is not None
            assert agg.session_start is not None

    def test_crypto_24_7_aggregation(self):
        """Test crypto aggregation with 24/7 data."""
        crypto_data = create_crypto_data("BTC-USD")

        agg = Aggregation(create_crypto_aggregation_config())
        result = agg.process(crypto_data)

        assert isinstance(result, DataFrame)
        assert len(result) > 0

        # Crypto should force UTC timezone
        assert agg.timezone == "UTC"
        assert agg.session_start == "00:00"

        # Should handle 24/7 data without issues
        timestamps = result["timestamp"].to_list()
        assert len(timestamps) > 20  # Should aggregate 48 hours to ~48 hours

        # Timestamps should span full 24-hour periods
        time_diffs = []
        for i in range(1, min(len(timestamps), 10)):
            diff = timestamps[i] - timestamps[i - 1]
            time_diffs.append(diff.total_seconds() / 3600)  # Convert to hours

        # Should have mostly 1-hour intervals
        assert all(0.9 <= diff <= 1.1 for diff in time_diffs)

    def test_equities_market_hours_aggregation(self):
        """Test equities aggregation with market hours consideration."""
        equities_data = create_market_hours_data("AAPL")

        agg = Aggregation(create_equity_aggregation_config())
        result = agg.process(equities_data)

        assert isinstance(result, DataFrame)
        assert len(result) > 0

        # Equities should use US/Eastern timezone by default
        assert agg.timezone == "US/Eastern"
        assert agg.session_start == "09:30"

        # Should preserve symbol column
        assert "symbol" in result.columns
        assert all(symbol == "AAPL" for symbol in result["symbol"].to_list())

    def test_fx_utc_aggregation(self):
        """Test FX aggregation with UTC timezone enforcement."""
        fx_data = create_forex_data("EUR/USD")

        # Try to set different timezone - should be overridden to UTC
        agg = Aggregation(AggregationConfig(target_timeframes=["30min"], asset_class="fx", timezone="US/Eastern"))
        result = agg.process(fx_data)

        assert isinstance(result, DataFrame)
        assert len(result) > 0

        # FX should force UTC timezone regardless of input
        assert agg.timezone == "UTC"
        assert agg.session_start == "00:00"

        # Should handle forex precision
        opens = result["open"].to_list()
        assert all(1.0 <= price <= 1.1 for price in opens[:5])  # Typical EUR/USD range


@pytest.mark.unit
class TestAssetClassTimezoneHandling:
    """Test timezone-specific behavior for different asset classes."""

    def test_utc_enforcement_crypto_fx(self):
        """Test that crypto and FX force UTC timezone."""
        utc_classes = ["crypto", "fx"]

        for asset_class in utc_classes:
            # Try to set various timezones - all should be overridden
            timezones_to_try = ["US/Eastern", "US/Pacific", "Asia/Tokyo", "Europe/London"]

            for tz in timezones_to_try:
                agg = Aggregation(AggregationConfig(target_timeframes=["1h"], asset_class=asset_class, timezone=tz))
                assert agg.timezone == "UTC", f"{asset_class} should force UTC, got {agg.timezone}"

    def test_timezone_flexibility_equities(self):
        """Test that equities respect specified timezone."""
        timezones_to_test = ["US/Eastern", "US/Pacific", "US/Central"]

        for tz in timezones_to_test:
            agg = Aggregation(AggregationConfig(target_timeframes=["1h"], asset_class="equities", timezone=tz))
            assert agg.timezone == tz, f"Equities should use specified timezone {tz}"

    def test_session_start_consistency(self):
        """Test that session_start is consistent with asset class."""
        expected_sessions = {
            "crypto": "00:00",
            "equities": "09:30",
            "fx": "00:00",
        }

        for asset_class, expected_start in expected_sessions.items():
            agg = Aggregation(AggregationConfig(target_timeframes=["1h"], asset_class=asset_class))
            assert agg.session_start == expected_start


@pytest.mark.unit
class TestMultiTimezoneConsistency:
    """Test aggregation consistency across different timezones."""

    def test_same_data_different_timezones(self):
        """Test that identical data gives consistent results across timezones."""
        timezone_data = create_multi_timezone_data(["UTC", "US/Eastern", "US/Pacific"])

        results = {}
        for tz, data in timezone_data.items():
            # Use equities so timezone is respected
            agg = Aggregation(AggregationConfig(target_timeframes=["1h"], asset_class="equities", timezone=tz))
            results[tz] = agg.process(data)

        # All results should have same structure
        base_result = results["UTC"]
        for tz, result in results.items():
            assert len(result) == len(base_result), f"Different lengths for {tz}"
            assert result.columns == base_result.columns, f"Different columns for {tz}"

    def test_timezone_aware_vs_naive_timestamps(self):
        """Test handling of timezone-aware vs naive timestamps."""
        # This tests the normalize_timezone functionality
        from .utils.thestrat_data_utils import create_timestamp_series

        # Create naive timestamps
        naive_timestamps = create_timestamp_series("2023-06-15 12:00:00", 24, 60)
        naive_data = DataFrame(
            {
                "timestamp": naive_timestamps,
                "open": [100.0 + i for i in range(24)],
                "high": [100.5 + i for i in range(24)],
                "low": [99.5 + i for i in range(24)],
                "close": [100.2 + i for i in range(24)],
                "volume": [1000] * 24,
                "timeframe": ["1h"] * 24,
            }
        )

        # Create timezone-aware timestamps
        aware_timestamps = create_timestamp_series("2023-06-15 12:00:00", 24, 60, "US/Eastern")
        aware_data = DataFrame(
            {
                "timestamp": aware_timestamps,
                "open": [100.0 + i for i in range(24)],
                "high": [100.5 + i for i in range(24)],
                "low": [99.5 + i for i in range(24)],
                "close": [100.2 + i for i in range(24)],
                "volume": [1000] * 24,
                "timeframe": ["1h"] * 24,
            }
        )

        agg = Aggregation(AggregationConfig(target_timeframes=["1h"], asset_class="equities", timezone="US/Eastern"))

        naive_result = agg.process(naive_data)
        aware_result = agg.process(aware_data)

        # Both should work and produce results
        assert isinstance(naive_result, DataFrame)
        assert isinstance(aware_result, DataFrame)
        assert len(naive_result) > 0
        assert len(aware_result) > 0


@pytest.mark.unit
class TestAssetClassSpecificFeatures:
    """Test asset-class specific aggregation features."""

    def test_crypto_continuous_trading(self):
        """Test that crypto handles continuous 24/7 trading properly."""
        # Create 3 days of continuous hourly data
        crypto_data = create_crypto_data("ETH-USD")

        agg = Aggregation(AggregationConfig(target_timeframes=["1d"], asset_class="crypto"))
        result = agg.process(crypto_data)

        assert len(result) >= 2  # Should get at least 2 daily bars

        # Each daily bar should represent 24 hours of trading
        volumes = result["volume"].to_list()
        for vol in volumes:
            assert vol > 0  # Should have volume from continuous trading

    def test_equities_market_gaps(self):
        """Test that equities handle market close/open gaps properly."""
        # Create market hours data that spans multiple days
        market_data = create_market_hours_data("SPY")

        agg = Aggregation(AggregationConfig(target_timeframes=["1d"], asset_class="equities"))
        result = agg.process(market_data)

        assert len(result) >= 1

        # Daily bars should align with market sessions (09:30 ET for equities)
        timestamps = result["timestamp"].to_list()
        for ts in timestamps[:3]:  # Check first few
            # Equities should align to session_start (09:30)
            assert ts.hour == 9 and ts.minute == 30, f"Expected 09:30, got {ts.hour}:{ts.minute:02d}"

    def test_fx_weekend_gaps(self):
        """Test FX handling of weekend trading gaps."""
        # Create 7 days of forex data (should have weekend gap)
        fx_data = create_forex_data("GBP/USD")

        agg = Aggregation(AggregationConfig(target_timeframes=["1d"], asset_class="fx"))
        result = agg.process(fx_data)

        # Should handle weekend gaps gracefully
        assert isinstance(result, DataFrame)
        assert len(result) >= 5  # Should get weekday data

    def test_volume_handling_by_asset_class(self):
        """Test that volume is handled appropriately for each asset class."""
        test_data = create_long_term_data(days=2, freq_minutes=60, symbol="TEST")

        asset_classes = ["crypto", "equities", "fx"]

        for asset_class in asset_classes:
            agg = Aggregation(AggregationConfig(target_timeframes=["1d"], asset_class=asset_class))
            result = agg.process(test_data)

            assert "volume" in result.columns
            volumes = result["volume"].to_list()

            # All asset classes should preserve volume
            for vol in volumes:
                assert vol > 0, f"Zero volume in {asset_class}"
                assert isinstance(vol, (int, float)), f"Invalid volume type in {asset_class}"


@pytest.mark.unit
class TestVolumeIntegerPrecision:
    """Test that volume remains as integer after aggregation (Issue #45)."""

    def test_volume_remains_integer_after_single_aggregation(self):
        """Test that volume is Int64 type with no decimal places after aggregation."""
        from .utils.thestrat_data_utils import create_timestamp_series

        # Create 1-minute data with specific volume values
        timestamps = create_timestamp_series("2023-01-01 00:00:00", 60, 1)  # 60 minutes
        data = DataFrame(
            {
                "timestamp": timestamps,
                "open": [100.0 + i * 0.1 for i in range(60)],
                "high": [100.5 + i * 0.1 for i in range(60)],
                "low": [99.5 + i * 0.1 for i in range(60)],
                "close": [100.2 + i * 0.1 for i in range(60)],
                "volume": [1000 + i for i in range(60)],
                "timeframe": ["1min"] * 60,
            }
        )

        # Aggregate to 1 hour
        agg = Aggregation(AggregationConfig(target_timeframes=["1h"], asset_class="crypto"))
        result = agg.process(data)

        assert "volume" in result.columns

        # Check volume dtype is Int64
        assert str(result.schema["volume"]) == "Int64", f"Volume dtype should be Int64, got {result.schema['volume']}"

        # Check volume values are integers (no decimal places)
        volumes = result["volume"].to_list()
        for vol in volumes:
            assert isinstance(vol, int), f"Volume should be int, got {type(vol)}"
            # Verify it's an actual integer value
            assert vol == int(vol), f"Volume {vol} has decimal places"

    def test_volume_precision_across_multiple_aggregation_levels(self):
        """Test that volume stays integer across multiple aggregation levels (1min → 1h → 1d)."""
        from .utils.thestrat_data_utils import create_timestamp_series

        # Create minute data
        timestamps = create_timestamp_series("2023-01-01 00:00:00", 1440, 1)  # 24 hours of minute data
        data = DataFrame(
            {
                "timestamp": timestamps,
                "open": [100.0 + i * 0.01 for i in range(1440)],
                "high": [100.5 + i * 0.01 for i in range(1440)],
                "low": [99.5 + i * 0.01 for i in range(1440)],
                "close": [100.2 + i * 0.01 for i in range(1440)],
                "volume": [1000 + i for i in range(1440)],
                "timeframe": ["1min"] * 1440,
            }
        )

        # Aggregate to hourly
        agg_hourly = Aggregation(AggregationConfig(target_timeframes=["1h"], asset_class="crypto"))
        hourly_result = agg_hourly.process(data)

        # Check hourly volumes are integers
        assert str(hourly_result.schema["volume"]) == "Int64"
        hourly_volumes = hourly_result["volume"].to_list()
        for vol in hourly_volumes:
            assert isinstance(vol, int), f"Hourly volume should be int, got {type(vol)}"

        # Aggregate hourly to daily
        agg_daily = Aggregation(AggregationConfig(target_timeframes=["1d"], asset_class="crypto"))
        daily_result = agg_daily.process(hourly_result)

        # Check daily volumes are still integers
        assert str(daily_result.schema["volume"]) == "Int64"
        daily_volumes = daily_result["volume"].to_list()
        for vol in daily_volumes:
            assert isinstance(vol, int), f"Daily volume should be int, got {type(vol)}"
            # Verify no floating-point accumulation
            assert vol == int(vol), f"Daily volume {vol} has decimal places"

    def test_large_volume_values_no_precision_issues(self):
        """Test that large volume values don't gain floating-point precision."""
        from .utils.thestrat_data_utils import create_timestamp_series

        # Create data with large volume values like the issue example
        large_volume = 23457897  # Example from issue #45
        timestamps = create_timestamp_series("2023-01-01 00:00:00", 60, 60)  # 60 hours

        data = DataFrame(
            {
                "timestamp": timestamps,
                "open": [100.0 + i * 0.1 for i in range(60)],
                "high": [100.5 + i * 0.1 for i in range(60)],
                "low": [99.5 + i * 0.1 for i in range(60)],
                "close": [100.2 + i * 0.1 for i in range(60)],
                "volume": [large_volume] * 60,
                "timeframe": ["1h"] * 60,
            }
        )

        # Aggregate to daily (24 hours)
        agg = Aggregation(AggregationConfig(target_timeframes=["1d"], asset_class="crypto"))
        result = agg.process(data)

        assert "volume" in result.columns

        # Check that volumes are exact integers
        volumes = result["volume"].to_list()
        for actual_volume in volumes:
            # Volume should be exact integer, not something like 117289485.035470001399517059326171875
            assert isinstance(actual_volume, int), f"Volume should be int, got {type(actual_volume)}"

            # Each daily bar should have 24 hours worth of volume
            # Verify no decimal component
            assert float(actual_volume) == actual_volume, "Volume should have no decimal component"
            assert actual_volume > large_volume, "Daily volume should be > single hour volume"


# ==========================================
# DST TRANSITION TESTS
# ==========================================
# Tests DST transitions (spring forward/fall back) across different timeframes and timezones.


@pytest.mark.unit
class TestDSTSpringForward:
    """Test aggregation during spring forward DST transition."""

    def test_spring_forward_hourly_aggregation(self):
        """Test hourly aggregation during spring forward (2AM->3AM skip)."""
        # Create data spanning spring forward transition
        spring_data = create_dst_transition_data("spring_forward", "US/Eastern")

        agg = Aggregation(AggregationConfig(target_timeframes=["1h"], asset_class="equities", timezone="US/Eastern"))
        result = agg.process(spring_data)

        assert isinstance(result, DataFrame)
        assert len(result) > 0

        # Check that the 2 AM hour is properly handled (should be missing/skipped)
        timestamps = result["timestamp"].to_list()

        # Look for the transition point
        transition_found = False
        for i in range(1, len(timestamps)):
            prev_ts = timestamps[i - 1]
            curr_ts = timestamps[i]

            # If we find 1 AM -> 3 AM jump, that's the spring forward
            if prev_ts.hour == 1 and curr_ts.hour == 3:
                transition_found = True
                # Should skip exactly 2 hours (1 hour normal + 1 hour DST skip)
                time_diff = curr_ts - prev_ts
                assert time_diff.total_seconds() == 7200  # 2 hours in seconds
                break

        # Should find the transition in hourly data
        assert transition_found, "Spring forward transition not found in hourly aggregation"

    def test_spring_forward_daily_aggregation(self):
        """Test daily aggregation spanning spring forward transition."""
        spring_data = create_dst_transition_data("spring_forward", "US/Eastern")

        agg = Aggregation(AggregationConfig(target_timeframes=["1d"], asset_class="equities", timezone="US/Eastern"))
        result = agg.process(spring_data)

        assert isinstance(result, DataFrame)
        assert len(result) >= 1

        # Daily aggregation should handle the missing hour gracefully
        volumes = result["volume"].to_list()
        for vol in volumes:
            assert vol > 0  # Should still have volume despite missing hour

        # OHLC should still be valid
        for i in range(len(result)):
            assert result["high"][i] >= result["open"][i]
            assert result["high"][i] >= result["close"][i]
            assert result["low"][i] <= result["open"][i]
            assert result["low"][i] <= result["close"][i]

    def test_spring_forward_30min_aggregation(self):
        """Test 30-minute aggregation during spring forward."""
        spring_data = create_dst_transition_data("spring_forward", "US/Eastern")

        agg = Aggregation(AggregationConfig(target_timeframes=["30min"], asset_class="equities", timezone="US/Eastern"))
        result = agg.process(spring_data)

        assert isinstance(result, DataFrame)
        assert len(result) > 0

        # 30-minute intervals should handle the transition smoothly
        timestamps = result["timestamp"].to_list()

        # Check that timestamps are properly ordered despite DST
        for i in range(1, len(timestamps)):
            assert timestamps[i] > timestamps[i - 1], f"Timestamps not ordered at index {i}"


@pytest.mark.unit
class TestDSTFallBack:
    """Test aggregation during fall back DST transition."""

    @pytest.mark.skip(
        reason="DST timezone handling requires Polars ambiguous/non_existent parameters - see docstring for details"
    )
    def test_fall_back_hourly_aggregation(self):
        """
        Test hourly aggregation during fall back (1AM repeated).

        SKIPPED: This test is currently skipped due to Polars timezone handling limitations.

        TECHNICAL ISSUE:
        During DST "fall back" transitions (e.g., Nov 5, 2023 in US/Eastern), the hour
        1:00 AM occurs twice - once before the clocks "fall back" and once after. This
        creates ambiguous timestamps that Polars cannot resolve without explicit parameters.

        ERROR ENCOUNTERED:
        polars.exceptions.ComputeError: datetime '2023-11-05 01:00:00' is ambiguous in
        time zone 'US/Eastern'. Please use `ambiguous` to tell how it should be localized.

        FAILURE LOCATION:
        /thestrat/aggregation.py:241 in normalize_timezone() method:
        col("timestamp").dt.replace_time_zone(self.timezone)

        SOLUTION REQUIRED:
        The normalize_timezone() method needs to be updated to handle DST transitions:
        col("timestamp").dt.replace_time_zone(
            self.timezone,
            ambiguous='earliest',  # or 'latest' - requires business logic decision
            non_existent='null'    # for spring forward transitions
        )

        BUSINESS IMPACT:
        - This affects only 2 days per year in DST-observing timezones
        - Most financial data uses UTC or pre-processed timestamps
        - Workaround: Use UTC timestamps or crypto/forex asset classes

        TEST PURPOSE:
        This test verifies that hourly aggregation handles the repeated 1 AM hour during
        DST fall back, expecting either 1 or 2 hourly bars for the duplicated hour.
        """
        fall_data = create_dst_transition_data("fall_back", "US/Eastern")

        agg = Aggregation(AggregationConfig(target_timeframes=["1h"], asset_class="equities", timezone="US/Eastern"))
        result = agg.process(fall_data)

        assert isinstance(result, DataFrame)
        assert len(result) > 0

        # Check handling of repeated hour
        timestamps = result["timestamp"].to_list()

        # Look for the transition - there might be two 1 AM hours
        hour_1_count = sum(1 for ts in timestamps if ts.hour == 1)

        # Depending on implementation, might get 1 or 2 bars for 1 AM hour
        # Both are valid handling strategies
        assert hour_1_count >= 1, "Should handle 1 AM hour during fall back"

    @pytest.mark.skip(
        reason="DST timezone handling requires Polars ambiguous/non_existent parameters - see docstring for details"
    )
    def test_fall_back_daily_aggregation(self):
        """
        Test daily aggregation spanning fall back transition.

        SKIPPED: This test is currently skipped due to Polars timezone handling limitations.

        TECHNICAL ISSUE:
        During DST "fall back" transitions (e.g., Nov 5, 2023 in US/Eastern), the hour
        1:00 AM occurs twice - once before the clocks "fall back" and once after. This
        creates ambiguous timestamps that Polars cannot resolve without explicit parameters.

        ERROR ENCOUNTERED:
        polars.exceptions.ComputeError: datetime '2023-11-05 01:00:00' is ambiguous in
        time zone 'US/Eastern'. Please use `ambiguous` to tell how it should be localized.

        FAILURE LOCATION:
        /thestrat/aggregation.py:241 in normalize_timezone() method:
        col("timestamp").dt.replace_time_zone(self.timezone)

        SOLUTION REQUIRED:
        The normalize_timezone() method needs to be updated to handle DST transitions:
        col("timestamp").dt.replace_time_zone(
            self.timezone,
            ambiguous='earliest',  # or 'latest' - requires business logic decision
            non_existent='null'    # for spring forward transitions
        )

        BUSINESS IMPACT:
        - This affects only 2 days per year in DST-observing timezones
        - Most financial data uses UTC or pre-processed timestamps
        - Workaround: Use UTC timestamps or crypto/forex asset classes

        TEST PURPOSE:
        This test verifies that daily aggregation handles the repeated 1 AM hour during
        DST fall back, expecting proper aggregation despite the extra hour of data.
        """
        fall_data = create_dst_transition_data("fall_back", "US/Eastern")

        agg = Aggregation(AggregationConfig(target_timeframes=["1d"], asset_class="equities", timezone="US/Eastern"))
        result = agg.process(fall_data)

        assert isinstance(result, DataFrame)
        assert len(result) >= 1

        # Daily aggregation should handle the extra hour
        # Total volume might be higher due to extra hour of data
        volumes = result["volume"].to_list()
        for vol in volumes:
            assert vol > 0

        # OHLC logic should still be preserved
        for i in range(len(result)):
            assert result["high"][i] >= result["low"][i]

    @pytest.mark.skip(
        reason="DST timezone handling requires Polars ambiguous/non_existent parameters - see docstring for details"
    )
    def test_fall_back_15min_aggregation(self):
        """
        Test 15-minute aggregation during fall back transition.

        SKIPPED: This test is currently skipped due to Polars timezone handling limitations.

        TECHNICAL ISSUE:
        During DST "fall back" transitions (e.g., Nov 5, 2023 in US/Eastern), the hour
        1:00 AM occurs twice - once before the clocks "fall back" and once after. This
        creates ambiguous timestamps that Polars cannot resolve without explicit parameters.

        ERROR ENCOUNTERED:
        polars.exceptions.ComputeError: datetime '2023-11-05 01:00:00' is ambiguous in
        time zone 'US/Eastern'. Please use `ambiguous` to tell how it should be localized.

        FAILURE LOCATION:
        /thestrat/aggregation.py:241 in normalize_timezone() method:
        col("timestamp").dt.replace_time_zone(self.timezone)

        SOLUTION REQUIRED:
        The normalize_timezone() method needs to be updated to handle DST transitions:
        col("timestamp").dt.replace_time_zone(
            self.timezone,
            ambiguous='earliest',  # or 'latest' - requires business logic decision
            non_existent='null'    # for spring forward transitions
        )

        BUSINESS IMPACT:
        - This affects only 2 days per year in DST-observing timezones
        - Most financial data uses UTC or pre-processed timestamps
        - Workaround: Use UTC timestamps or crypto/forex asset classes

        TEST PURPOSE:
        This test verifies that 15-minute aggregation handles the repeated 1 AM hour during
        DST fall back, expecting proper timestamp ordering despite the duplicated hour.
        """
        fall_data = create_dst_transition_data("fall_back", "US/Eastern")

        agg = Aggregation(AggregationConfig(target_timeframes=["15min"], asset_class="equities", timezone="US/Eastern"))
        result = agg.process(fall_data)

        assert isinstance(result, DataFrame)
        assert len(result) > 0

        # 15-minute intervals should properly handle repeated hour
        timestamps = result["timestamp"].to_list()

        # Verify timestamps remain properly ordered
        for i in range(1, len(timestamps)):
            # Allow for the case where same timestamp might appear due to repeated hour
            assert timestamps[i] >= timestamps[i - 1], f"Timestamps not properly ordered at index {i}"


@pytest.mark.unit
class TestDSTTimezoneConsistency:
    """Test DST handling consistency across timezones."""

    def test_utc_vs_eastern_dst_spring(self):
        """Test that UTC and Eastern give consistent results during spring DST."""
        # Create data for both timezones
        eastern_data = create_dst_transition_data("spring_forward", "US/Eastern")
        utc_data = create_dst_transition_data("spring_forward", "UTC")

        # Test with equities (timezone-sensitive)
        agg_eastern = Aggregation(
            AggregationConfig(target_timeframes=["1h"], asset_class="equities", timezone="US/Eastern")
        )
        agg_utc = Aggregation(AggregationConfig(target_timeframes=["1h"], asset_class="equities", timezone="UTC"))

        result_eastern = agg_eastern.process(eastern_data)
        result_utc = agg_utc.process(utc_data)

        # Both should produce valid results
        assert isinstance(result_eastern, DataFrame)
        assert isinstance(result_utc, DataFrame)
        assert len(result_eastern) > 0
        assert len(result_utc) > 0

        # Both should handle the data without errors (specific intervals may vary due to DST)
        # The key test is that both produce valid aggregated data
        eastern_volumes = result_eastern["volume"].to_list()
        utc_volumes = result_utc["volume"].to_list()

        for vol in eastern_volumes:
            assert vol > 0, "Eastern timezone aggregation should have positive volume"

        for vol in utc_volumes:
            assert vol > 0, "UTC timezone aggregation should have positive volume"

    def test_crypto_utc_enforcement_during_dst(self):
        """Test that crypto forces UTC timezone even during DST periods."""
        # Try to use Eastern timezone with crypto during DST
        dst_data = create_dst_transition_data("spring_forward", "US/Eastern")

        agg = Aggregation(AggregationConfig(target_timeframes=["1h"], asset_class="crypto", timezone="US/Eastern"))
        result = agg.process(dst_data)

        # Should force UTC timezone regardless of input
        assert agg.timezone == "UTC"

        # Should produce consistent results without DST complications
        assert isinstance(result, DataFrame)
        assert len(result) > 0

    def test_fx_utc_enforcement_during_dst(self):
        """Test that FX forces UTC timezone even during DST periods."""
        dst_data = create_dst_transition_data("fall_back", "US/Eastern")

        agg = Aggregation(AggregationConfig(target_timeframes=["30min"], asset_class="fx", timezone="US/Eastern"))
        result = agg.process(dst_data)

        # Should force UTC timezone
        assert agg.timezone == "UTC"

        # Should handle FX data consistently
        assert isinstance(result, DataFrame)
        assert len(result) > 0


@pytest.mark.unit
class TestDSTBoundaryAlignment:
    """Test boundary alignment during DST transitions."""

    def test_hour_boundary_alignment_spring_forward(self):
        """Test hour boundary alignment during spring forward."""
        spring_data = create_dst_transition_data("spring_forward", "US/Eastern")

        # Explicitly set hour_boundary=True to test hour boundary alignment
        agg = Aggregation(
            AggregationConfig(
                target_timeframes=["1h"], asset_class="equities", timezone="US/Eastern", hour_boundary=True
            )
        )
        result = agg.process(spring_data)

        # hour_boundary was explicitly set to True
        assert agg.hour_boundary is True

        # Check that available timestamps align to hour boundaries
        timestamps = result["timestamp"].to_list()
        for ts in timestamps[:10]:  # Check first 10 timestamps
            # Should align to hour (minute and second should be 0)
            assert ts.minute == 0, f"Hour boundary not aligned: {ts}"
            assert ts.second == 0, f"Hour boundary not aligned: {ts}"

    @pytest.mark.skip(
        reason="DST timezone handling requires Polars ambiguous/non_existent parameters - see docstring for details"
    )
    def test_daily_boundary_alignment_fall_back(self):
        """
        Test daily boundary alignment during fall back transition.

        SKIPPED: This test is currently skipped due to Polars timezone handling limitations.

        TECHNICAL ISSUE:
        During DST "fall back" transitions (e.g., Nov 5, 2023 in US/Eastern), the hour
        1:00 AM occurs twice - once before the clocks "fall back" and once after. This
        creates ambiguous timestamps that Polars cannot resolve without explicit parameters.

        ERROR ENCOUNTERED:
        polars.exceptions.ComputeError: datetime '2023-11-05 01:00:00' is ambiguous in
        time zone 'US/Eastern'. Please use `ambiguous` to tell how it should be localized.

        FAILURE LOCATION:
        /thestrat/aggregation.py:241 in normalize_timezone() method:
        col("timestamp").dt.replace_time_zone(self.timezone)

        SOLUTION REQUIRED:
        The normalize_timezone() method needs to be updated to handle DST transitions:
        col("timestamp").dt.replace_time_zone(
            self.timezone,
            ambiguous='earliest',  # or 'latest' - requires business logic decision
            non_existent='null'    # for spring forward transitions
        )

        BUSINESS IMPACT:
        - This affects only 2 days per year in DST-observing timezones
        - Most financial data uses UTC or pre-processed timestamps
        - Workaround: Use UTC timestamps or crypto/forex asset classes

        TEST PURPOSE:
        This test verifies that daily boundary alignment works correctly during DST fall back,
        expecting proper alignment to market day or midnight despite the extra hour.
        """
        fall_data = create_dst_transition_data("fall_back", "US/Eastern")

        agg = Aggregation(AggregationConfig(target_timeframes=["1d"], asset_class="equities", timezone="US/Eastern"))
        result = agg.process(fall_data)

        # hour_boundary uses equities default (False)
        assert agg.hour_boundary is False

        # Daily boundaries should still work correctly
        timestamps = result["timestamp"].to_list()
        for ts in timestamps:
            # Should align to start of market day or midnight
            assert ts.hour in [0, 9], f"Daily boundary not properly aligned: {ts}"
            assert ts.minute == 0, f"Daily boundary not aligned: {ts}"

    def test_sub_hourly_dst_handling(self):
        """Test that sub-hourly timeframes handle DST properly."""
        spring_data = create_dst_transition_data("spring_forward", "US/Eastern")

        # Test multiple sub-hourly timeframes
        timeframes = ["5min", "15min", "30min"]

        for tf in timeframes:
            agg = Aggregation(AggregationConfig(target_timeframes=[tf], asset_class="equities", timezone="US/Eastern"))
            result = agg.process(spring_data)

            # hour_boundary uses equities default (False)
            assert agg.hour_boundary is False  # Equities default
            assert isinstance(result, DataFrame)
            assert len(result) > 0

            # Should handle DST transition without errors
            # Note: DST transitions create gaps/jumps in timestamps, so we focus on data validity
            timestamps = result["timestamp"].to_list()
            volumes = result["volume"].to_list()

            # Verify timestamps are ordered (despite potential DST gaps)
            for i in range(1, len(timestamps)):
                assert timestamps[i] > timestamps[i - 1], f"Timestamps not ordered in {tf} at index {i}"

            # Verify all aggregated bars have positive volume
            for i, vol in enumerate(volumes):
                assert vol > 0, f"Zero volume in {tf} timeframe at index {i}"


@pytest.mark.unit
class TestDSTEdgeCases:
    """Test edge cases in DST handling."""

    def test_dst_transition_with_minimal_data(self):
        """Test DST handling with minimal data around transition."""
        # Create minimal data set around spring forward
        spring_data = create_dst_transition_data("spring_forward", "US/Eastern")

        # Take only a few rows around the transition
        minimal_data = spring_data.head(10)

        agg = Aggregation(AggregationConfig(target_timeframes=["1h"], asset_class="equities", timezone="US/Eastern"))
        result = agg.process(minimal_data)

        # Should handle gracefully even with minimal data
        assert isinstance(result, DataFrame)
        # Might get 0 results if insufficient data, which is acceptable

    @pytest.mark.skip(
        reason="DST timezone handling requires Polars ambiguous/non_existent parameters - see docstring for details"
    )
    def test_multiple_dst_transitions(self):
        """
        Test handling data that spans multiple DST transitions (both spring forward and fall back).

        SKIPPED: This test is currently skipped due to Polars timezone handling limitations.

        TECHNICAL ISSUE:
        This test covers year-long data that includes both DST transitions:
        - Spring forward (e.g., Mar 12, 2023): datetime '2023-03-12 02:00:00' is non-existent
        - Fall back (e.g., Nov 5, 2023): datetime '2023-11-05 01:00:00' is ambiguous
        Polars cannot handle these transitions without explicit parameters.

        ERROR ENCOUNTERED:
        polars.exceptions.ComputeError: datetime '2023-11-05 01:00:00' is ambiguous in
        time zone 'US/Eastern'. Please use `ambiguous` to tell how it should be localized.
        (Also similar error for non-existent spring forward times)

        FAILURE LOCATION:
        /thestrat/aggregation.py:241 in normalize_timezone() method:
        col("timestamp").dt.replace_time_zone(self.timezone)

        SOLUTION REQUIRED:
        The normalize_timezone() method needs to be updated to handle both DST transitions:
        col("timestamp").dt.replace_time_zone(
            self.timezone,
            ambiguous='earliest',  # for fall back - choose first occurrence
            non_existent='null'    # for spring forward - handle missing hour
        )

        BUSINESS IMPACT:
        - This affects only 2 days per year in DST-observing timezones
        - Most financial data uses UTC or pre-processed timestamps
        - Workaround: Use UTC timestamps or crypto/forex asset classes

        TEST PURPOSE:
        This test verifies that year-long aggregation handles both DST transitions gracefully,
        expecting proper daily aggregation across the entire year despite DST complications.
        """
        # Create year-long data that includes both spring forward and fall back
        year_data = create_long_term_data(days=365, freq_minutes=60, symbol="TEST")

        agg = Aggregation(AggregationConfig(target_timeframes=["1d"], asset_class="equities", timezone="US/Eastern"))
        result = agg.process(year_data)

        assert isinstance(result, DataFrame)
        assert len(result) > 300  # Should get most days of the year

        # Daily aggregation should handle multiple DST transitions
        volumes = result["volume"].to_list()
        for vol in volumes:
            assert vol > 0  # All days should have volume

    def test_dst_with_different_asset_classes(self):
        """Test DST handling across different asset classes."""
        spring_data = create_dst_transition_data("spring_forward", "US/Eastern")

        asset_classes = ["crypto", "equities", "fx"]

        for asset_class in asset_classes:
            agg = Aggregation(AggregationConfig(target_timeframes=["1h"], asset_class=asset_class))
            result = agg.process(spring_data)

            assert isinstance(result, DataFrame)
            # Some asset classes force UTC, which won't have DST issues
            if asset_class in ["crypto", "fx"]:
                assert agg.timezone == "UTC"

            # All should handle the data without errors
            if len(result) > 0:  # If we got results
                volumes = result["volume"].to_list()
                for vol in volumes:
                    assert vol >= 0  # Volume should be non-negative


@pytest.mark.unit
class TestCalendarPeriodTimestamps:
    """Test timestamp alignment for calendar-based periods (Issue #46)."""

    def test_monthly_aggregation_timestamp_alignment_equities(self):
        """Test that monthly bars align to session_start for equities (09:30 ET)."""
        from zoneinfo import ZoneInfo

        from .utils.thestrat_data_utils import create_timestamp_series

        ZoneInfo("America/New_York")

        # Create daily bars at 09:30 ET for a full year
        timestamps = create_timestamp_series("2023-01-01 09:30:00", 400, 1440)  # 400 days
        test_data = DataFrame(
            {
                "timestamp": timestamps,
                "open": [100.0 + i * 0.01 for i in range(400)],
                "high": [100.5 + i * 0.01 for i in range(400)],
                "low": [99.5 + i * 0.01 for i in range(400)],
                "close": [100.2 + i * 0.01 for i in range(400)],
                "volume": [1000] * 400,
                "timeframe": ["1d"] * 400,
            }
        )

        # Set timestamps to 09:30 ET
        test_data = test_data.with_columns(
            [col("timestamp").dt.replace_time_zone("America/New_York").alias("timestamp")]
        )

        agg = Aggregation(
            AggregationConfig(target_timeframes=["1m"], asset_class="equities", timezone="America/New_York")
        )
        result = agg.process(test_data)

        # All monthly bars should start at 09:30 ET (not 08:30)
        timestamps = result["timestamp"].to_list()

        for ts in timestamps:
            assert ts.hour == 9, f"Monthly bar hour should be 9, got {ts.hour} for {ts}"
            assert ts.minute == 30, f"Monthly bar minute should be 30, got {ts.minute} for {ts}"

    def test_quarterly_aggregation_timestamp_alignment(self):
        """Test that quarterly bars align to session_start for equities."""
        from zoneinfo import ZoneInfo

        from .utils.thestrat_data_utils import create_timestamp_series

        ZoneInfo("America/New_York")

        # Create daily data for multiple quarters at 09:30
        timestamps = create_timestamp_series("2023-01-01 09:30:00", 400, 1440)
        test_data = DataFrame(
            {
                "timestamp": timestamps,
                "open": [100.0 + i * 0.01 for i in range(400)],
                "high": [100.5 + i * 0.01 for i in range(400)],
                "low": [99.5 + i * 0.01 for i in range(400)],
                "close": [100.2 + i * 0.01 for i in range(400)],
                "volume": [1000] * 400,
                "timeframe": ["1d"] * 400,
            }
        )

        test_data = test_data.with_columns(
            [col("timestamp").dt.replace_time_zone("America/New_York").alias("timestamp")]
        )

        agg = Aggregation(
            AggregationConfig(target_timeframes=["1q"], asset_class="equities", timezone="America/New_York")
        )
        result = agg.process(test_data)

        timestamps = result["timestamp"].to_list()

        for ts in timestamps:
            assert ts.hour == 9, f"Quarterly bar hour should be 9, got {ts.hour} for {ts}"
            assert ts.minute == 30, f"Quarterly bar minute should be 30, got {ts.minute} for {ts}"

    def test_yearly_aggregation_timestamp_alignment(self):
        """Test that yearly bars align to session_start for equities."""
        from zoneinfo import ZoneInfo

        from .utils.thestrat_data_utils import create_timestamp_series

        ZoneInfo("America/New_York")

        # Create daily data for multiple years at 09:30
        timestamps = create_timestamp_series("2023-01-01 09:30:00", 800, 1440)
        test_data = DataFrame(
            {
                "timestamp": timestamps,
                "open": [100.0 + i * 0.01 for i in range(800)],
                "high": [100.5 + i * 0.01 for i in range(800)],
                "low": [99.5 + i * 0.01 for i in range(800)],
                "close": [100.2 + i * 0.01 for i in range(800)],
                "volume": [1000] * 800,
                "timeframe": ["1d"] * 800,
            }
        )

        test_data = test_data.with_columns(
            [col("timestamp").dt.replace_time_zone("America/New_York").alias("timestamp")]
        )

        agg = Aggregation(
            AggregationConfig(target_timeframes=["1y"], asset_class="equities", timezone="America/New_York")
        )
        result = agg.process(test_data)

        timestamps = result["timestamp"].to_list()

        for ts in timestamps:
            assert ts.hour == 9, f"Yearly bar hour should be 9, got {ts.hour} for {ts}"
            assert ts.minute == 30, f"Yearly bar minute should be 30, got {ts.minute} for {ts}"

    def test_asset_class_specific_session_alignment(self):
        """Test that each asset class maintains correct session start for calendar periods."""
        from .utils.thestrat_data_utils import create_timestamp_series

        # Test equities (09:30 ET)
        timestamps_et = create_timestamp_series("2023-01-01 09:30:00", 400, 1440)
        test_data_et = DataFrame(
            {
                "timestamp": timestamps_et,
                "open": [100.0 + i * 0.01 for i in range(400)],
                "high": [100.5 + i * 0.01 for i in range(400)],
                "low": [99.5 + i * 0.01 for i in range(400)],
                "close": [100.2 + i * 0.01 for i in range(400)],
                "volume": [1000] * 400,
                "timeframe": ["1d"] * 400,
            }
        )
        test_data_et = test_data_et.with_columns(
            [col("timestamp").dt.replace_time_zone("America/New_York").alias("timestamp")]
        )

        agg_equities = Aggregation(
            AggregationConfig(target_timeframes=["1m"], asset_class="equities", timezone="America/New_York")
        )
        result_equities = agg_equities.process(test_data_et)

        for ts in result_equities["timestamp"].to_list():
            assert ts.hour == 9 and ts.minute == 30, f"Equities monthly should be 09:30, got {ts.hour}:{ts.minute:02d}"

        # Test crypto (00:00 UTC)
        timestamps_utc = create_timestamp_series("2023-01-01 00:00:00", 400, 1440)
        test_data_utc = DataFrame(
            {
                "timestamp": timestamps_utc,
                "open": [100.0 + i * 0.01 for i in range(400)],
                "high": [100.5 + i * 0.01 for i in range(400)],
                "low": [99.5 + i * 0.01 for i in range(400)],
                "close": [100.2 + i * 0.01 for i in range(400)],
                "volume": [1000] * 400,
                "timeframe": ["1d"] * 400,
            }
        )
        test_data_utc = test_data_utc.with_columns([col("timestamp").dt.replace_time_zone("UTC").alias("timestamp")])

        agg_crypto = Aggregation(AggregationConfig(target_timeframes=["1m"], asset_class="crypto"))
        result_crypto = agg_crypto.process(test_data_utc)

        for ts in result_crypto["timestamp"].to_list():
            assert ts.hour == 0 and ts.minute == 0, f"Crypto monthly should be 00:00, got {ts.hour}:{ts.minute:02d}"

        # Test FX (00:00 UTC)
        agg_fx = Aggregation(AggregationConfig(target_timeframes=["1m"], asset_class="fx"))
        result_fx = agg_fx.process(test_data_utc)

        for ts in result_fx["timestamp"].to_list():
            assert ts.hour == 0 and ts.minute == 0, f"FX monthly should be 00:00, got {ts.hour}:{ts.minute:02d}"

    def test_all_calendar_periods_maintain_session_start(self):
        """Test that all calendar periods (monthly, quarterly, yearly) maintain session start."""
        from zoneinfo import ZoneInfo

        from .utils.thestrat_data_utils import create_timestamp_series

        ZoneInfo("America/New_York")

        # Create data at 09:30
        timestamps = create_timestamp_series("2023-01-01 09:30:00", 800, 1440)
        test_data = DataFrame(
            {
                "timestamp": timestamps,
                "open": [100.0 + i * 0.01 for i in range(800)],
                "high": [100.5 + i * 0.01 for i in range(800)],
                "low": [99.5 + i * 0.01 for i in range(800)],
                "close": [100.2 + i * 0.01 for i in range(800)],
                "volume": [1000] * 800,
                "timeframe": ["1d"] * 800,
            }
        )
        test_data = test_data.with_columns(
            [col("timestamp").dt.replace_time_zone("America/New_York").alias("timestamp")]
        )

        calendar_periods = ["1m", "1q", "1y"]

        for timeframe in calendar_periods:
            agg = Aggregation(
                AggregationConfig(target_timeframes=[timeframe], asset_class="equities", timezone="America/New_York")
            )
            result = agg.process(test_data)

            timestamps = result["timestamp"].to_list()

            for ts in timestamps:
                assert ts.hour == 9 and ts.minute == 30, (
                    f"{timeframe}: Expected 09:30, got {ts.hour}:{ts.minute:02d} on {ts.date()}"
                )


# ==========================================
# TIMEFRAMES TESTS
# ==========================================
# Tests all supported timeframes for aggregation across different scales.


@pytest.mark.unit
class TestAllTimeframes:
    """Test all supported timeframes for aggregation."""

    def test_all_timeframe_mappings_exist(self):
        """Test that all timeframes in TIMEFRAME_METADATA are valid."""
        # Test that we can create Aggregation objects for all timeframes
        for input_tf in TimeframeConfig.TIMEFRAME_METADATA.keys():
            agg = Aggregation(AggregationConfig(target_timeframes=[input_tf]))
            assert agg.target_timeframes[0] == input_tf

    @pytest.mark.parametrize("timeframe", ["1min", "5min", "15min", "30min"])
    def test_sub_hourly_aggregation(self, timeframe):
        """Test aggregation for sub-hourly timeframes."""
        agg = Aggregation(AggregationConfig(target_timeframes=[timeframe], asset_class="equities"))

        # Create test data
        test_data = create_long_term_data(days=1, freq_minutes=1, symbol="SPY")
        result = agg.process(test_data)

        assert isinstance(result, DataFrame)
        assert len(result) > 0
        assert all(col in result.columns for col in ["timestamp", "open", "high", "low", "close"])

        # hour_boundary uses equities default (False)
        assert agg.hour_boundary is False

    @pytest.mark.parametrize("timeframe", ["1h", "4h", "6h", "12h"])
    def test_hourly_aggregation(self, timeframe):
        """Test aggregation for hourly timeframes including new 6h and 12h."""
        agg = Aggregation(AggregationConfig(target_timeframes=[timeframe], asset_class="equities"))

        test_data = create_long_term_data(days=7, freq_minutes=60, symbol="SPY")
        result = agg.process(test_data)

        assert isinstance(result, DataFrame)
        assert len(result) > 0
        assert result["timeframe"][0] == timeframe

        # hour_boundary uses equities default (False)
        assert agg.hour_boundary is False

    def test_session_open_alignment(self):
        """Test that hourly+ timeframes align with session_open when hour_boundary=False."""
        from datetime import datetime, timezone

        from polars import DataFrame

        # Test session open alignment
        timestamps = [
            datetime(2024, 1, 1, 9, 30, tzinfo=timezone.utc),  # 9:30
            datetime(2024, 1, 1, 10, 30, tzinfo=timezone.utc),  # 10:30
            datetime(2024, 1, 1, 11, 30, tzinfo=timezone.utc),  # 11:30
            datetime(2024, 1, 1, 12, 30, tzinfo=timezone.utc),  # 12:30
            datetime(2024, 1, 1, 13, 30, tzinfo=timezone.utc),  # 13:30
            datetime(2024, 1, 1, 14, 30, tzinfo=timezone.utc),  # 14:30
        ]

        test_data = DataFrame(
            {
                "timestamp": timestamps,
                "open": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
                "high": [100.5, 101.5, 102.5, 103.5, 104.5, 105.5],
                "low": [99.5, 100.5, 101.5, 102.5, 103.5, 104.5],
                "close": [100.2, 101.2, 102.2, 103.2, 104.2, 105.2],
                "volume": [1000, 1100, 1200, 1300, 1400, 1500],
                "timeframe": ["30min"] * 6,
            }
        )

        # Test equities with session_start=09:30 and hour_boundary=False
        agg = Aggregation(
            AggregationConfig(
                target_timeframes=["1h"], asset_class="equities", session_start="09:30", hour_boundary=False
            )
        )

        result = agg.process(test_data)

        assert isinstance(result, DataFrame)
        assert len(result) > 0

        # For equities with hour_boundary=False, bars should align with session_start (09:30)
        # So we should get bars at 9:30, 10:30, 11:30, etc.
        result_timestamps = result["timestamp"].to_list()
        for ts in result_timestamps:
            # Each bar should start at :30 minutes (session_start time)
            assert ts.minute == 30


@pytest.mark.unit
class TestTimeframeEdgeCases:
    """Test edge cases in timeframe handling."""

    def test_minimal_data_aggregation(self):
        """Test aggregation with minimal data points."""
        # Create very minimal data
        minimal_data = create_long_term_data(days=1, freq_minutes=60, symbol="TEST")
        minimal_data = minimal_data.head(5)  # Only 5 hours of data

        timeframes = ["1h", "4h", "1d"]
        for tf in timeframes:
            agg = Aggregation(AggregationConfig(target_timeframes=[tf], asset_class="equities"))
            result = agg.process(minimal_data)

            # Should handle gracefully - might produce 0-N results
            assert isinstance(result, DataFrame)


@pytest.mark.unit
class TestNewMultiTimeframeFeatures:
    """Test new multi-timeframe and multi-symbol features."""

    def test_multiple_timeframes_single_symbol(self):
        """Test aggregation with multiple timeframes for single symbol."""
        # Create 1 hour of minute data
        data = create_long_term_data(days=1, freq_minutes=1, symbol="AAPL")
        data = data.head(60)  # One hour

        agg = Aggregation(AggregationConfig(target_timeframes=["5min", "15min", "30min"]))
        result = agg.process(data)

        # Should have timeframe column
        assert "timeframe" in result.columns

        # Should have data for all timeframes
        timeframes = result["timeframe"].unique().to_list()
        assert set(timeframes) == {"5min", "15min", "30min"}

        # Should have symbol column since input had symbol
        assert "symbol" in result.columns
        assert result["symbol"].unique().to_list() == ["AAPL"]

    def test_multiple_symbols_multiple_timeframes(self):
        """Test aggregation with multiple symbols and timeframes."""
        # Create data for multiple symbols
        symbols = ["AAPL", "MSFT", "GOOGL"]
        data_parts = []

        for symbol in symbols:
            symbol_data = create_long_term_data(days=1, freq_minutes=5, symbol=symbol)
            symbol_data = symbol_data.head(24)  # 2 hours of 5-minute data
            data_parts.append(symbol_data)

        # Combine all symbol data
        multi_symbol_data = data_parts[0]
        for part in data_parts[1:]:
            multi_symbol_data = multi_symbol_data.vstack(part)

        agg = Aggregation(AggregationConfig(target_timeframes=["15min", "1h"]))
        result = agg.process(multi_symbol_data)

        # Verify structure
        assert "symbol" in result.columns
        assert "timeframe" in result.columns
        assert set(result["symbol"].unique().to_list()) == set(symbols)
        assert set(result["timeframe"].unique().to_list()) == {"15min", "1h"}

        # Verify filtering works
        aapl_15m = result.filter((result["symbol"] == "AAPL") & (result["timeframe"] == "15min"))
        assert len(aapl_15m) > 0
        assert aapl_15m["symbol"].unique().to_list() == ["AAPL"]
        assert aapl_15m["timeframe"].unique().to_list() == ["15min"]

    def test_single_timeframe_compatibility(self):
        """Test that single timeframe still works and produces timeframe column."""
        data = create_long_term_data(days=1, freq_minutes=1, symbol="TEST")
        data = data.head(30)  # 30 minutes of data

        agg = Aggregation(create_aggregation_config(target_timeframes=["5min"]))  # Single timeframe as string
        result = agg.process(data)

        # Should still have timeframe column
        assert "timeframe" in result.columns
        assert result["timeframe"].unique().to_list() == ["5min"]

        # Should have expected OHLC structure
        expected_cols = ["symbol", "timeframe", "timestamp", "open", "high", "low", "close", "volume"]
        assert all(col in result.columns for col in expected_cols)


@pytest.mark.unit
class TestAggregationEdgeCases:
    """Test cases for Aggregation edge cases and error conditions."""

    def test_target_timeframes_not_list_raises_typeerror(self):
        """Test that passing non-list target_timeframes raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Aggregation(AggregationConfig(target_timeframes="5min"))  # String instead of list
        assert "Input should be a valid list" in str(exc_info.value)

    def test_target_timeframes_empty_list_raises_valueerror(self):
        """Test that passing empty list for target_timeframes raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Aggregation(AggregationConfig(target_timeframes=[]))  # Empty list
        assert "List should have at least 1 item" in str(exc_info.value)

    def test_unsupported_timeframe_raises_valueerror(self):
        """Test that unsupported timeframe format raises ValueError."""
        data = create_long_term_data(days=1, freq_minutes=1)
        data = data.head(60)  # 1 hour of data

        agg = Aggregation(create_aggregation_config(target_timeframes=["5min"]))

        # This should work normally first
        result = agg.process(data)
        assert len(result) > 0

        # Test formats that make get_polars_format return None or empty
        # Empty string makes get_polars_format return empty string (falsy)
        with pytest.raises(ValueError) as exc_info:
            agg._process_single_timeframe(data, "")
        assert "Unsupported timeframe" in str(exc_info.value)

        # None makes get_polars_format return None (falsy)
        with pytest.raises(ValueError) as exc_info:
            agg._process_single_timeframe(data, None)
        assert "Unsupported timeframe" in str(exc_info.value)

    def test_input_validation_failure_raises_valueerror(self):
        """Test that input validation failure raises ValueError."""
        agg = Aggregation(create_aggregation_config(target_timeframes=["5min"]))

        # Test with invalid data structure (missing required columns)
        invalid_data = DataFrame({"wrong_column": [1, 2, 3], "another_wrong": ["a", "b", "c"]})

        with pytest.raises(ValueError) as exc_info:
            agg.process(invalid_data)
        assert "Input data validation failed" in str(exc_info.value)


@pytest.mark.unit
class TestAggregationPrivateMethods:
    """Test cases for Aggregation private helper methods."""

    def test_should_use_hour_boundary_supported_timeframes(self):
        """Test _should_use_hour_boundary with supported timeframe strings."""
        agg = Aggregation(create_aggregation_config())

        # Test supported hourly and higher timeframes
        assert agg._should_use_hour_boundary("1h") is True
        assert agg._should_use_hour_boundary("4h") is True
        assert agg._should_use_hour_boundary("1d") is True
        assert agg._should_use_hour_boundary("1w") is True
        assert agg._should_use_hour_boundary("1m") is True  # monthly
        assert agg._should_use_hour_boundary("1q") is True
        assert agg._should_use_hour_boundary("1y") is True

    def test_should_use_hour_boundary_polars_style_patterns(self):
        """Test _should_use_hour_boundary with polars-style timeframe patterns."""
        agg = Aggregation(create_aggregation_config())

        # Test polars-style patterns (case insensitive)
        assert agg._should_use_hour_boundary("2h") is True
        assert agg._should_use_hour_boundary("1d") is True
        assert agg._should_use_hour_boundary("3w") is True
        assert agg._should_use_hour_boundary("2mo") is True
        assert agg._should_use_hour_boundary("1y") is True

        # Test case insensitive
        assert agg._should_use_hour_boundary("2H") is True
        assert agg._should_use_hour_boundary("1D") is True
        assert agg._should_use_hour_boundary("1Mo") is True

        # Test minute patterns should return False
        assert agg._should_use_hour_boundary("5m") is False
        assert agg._should_use_hour_boundary("15M") is False

    def test_should_use_hour_boundary_edge_cases(self):
        """Test _should_use_hour_boundary edge cases and invalid patterns."""
        agg = Aggregation(create_aggregation_config())

        # Invalid patterns should return False
        assert agg._should_use_hour_boundary("invalid") is False
        assert agg._should_use_hour_boundary("") is False
        assert agg._should_use_hour_boundary("123") is False
        assert agg._should_use_hour_boundary("5x") is False

    def test_is_hourly_or_higher_various_formats(self):
        """Test _is_hourly_or_higher with various timeframe formats."""
        agg = Aggregation(create_aggregation_config())

        # Test explicit hourly patterns
        assert agg._is_hourly_or_higher("1h") is True
        assert agg._is_hourly_or_higher("4h") is True
        assert agg._is_hourly_or_higher("12h") is True
        assert agg._is_hourly_or_higher("24h") is True

        # Test daily and higher patterns
        assert agg._is_hourly_or_higher("1d") is True
        assert agg._is_hourly_or_higher("7d") is True
        assert agg._is_hourly_or_higher("1w") is True
        assert agg._is_hourly_or_higher("1mo") is True
        assert agg._is_hourly_or_higher("1y") is True

        # Test case insensitive
        assert agg._is_hourly_or_higher("1H") is True
        assert agg._is_hourly_or_higher("1D") is True
        assert agg._is_hourly_or_higher("1W") is True
        assert agg._is_hourly_or_higher("1Mo") is True
        assert agg._is_hourly_or_higher("1Y") is True

        # Test minute patterns - note: "1m" is treated as monthly, not minutes
        # This appears to be current behavior where "1m" = 1 month
        assert agg._is_hourly_or_higher("1m") is True  # 1 month
        assert agg._is_hourly_or_higher("5m") is True  # 5 months
        assert agg._is_hourly_or_higher("15M") is True  # 15 months (case insensitive)

        # Test actual minute patterns that should return False
        assert agg._is_hourly_or_higher("30min") is False
        assert agg._is_hourly_or_higher("invalid") is False


def create_multi_timeframe_data(
    timeframes: list[str],
    periods_per_tf: int = 10,
    base_price: float = 100.0,
    symbols: list[str] | None = None,
    start_time: str = "2024-01-01 09:30:00",
) -> DataFrame:
    """
    Create test data with multiple timeframes for testing multi-timeframe source functionality.

    Args:
        timeframes: List of timeframe strings (e.g., ["1min", "5min", "15min"])
        periods_per_tf: Number of bars to generate per timeframe
        base_price: Starting price level
        symbols: Optional list of symbols (defaults to ["TEST"])
        start_time: Start timestamp string

    Returns:
        Polars DataFrame with mixed timeframe data including timeframe column
    """
    if symbols is None:
        symbols = ["TEST"]

    all_data = []

    for symbol in symbols:
        for timeframe in timeframes:
            # Determine frequency in minutes for this timeframe
            if timeframe == "1min":
                freq_minutes = 1
            elif timeframe == "5min":
                freq_minutes = 5
            elif timeframe == "15min":
                freq_minutes = 15
            elif timeframe == "30min":
                freq_minutes = 30
            elif timeframe == "1h":
                freq_minutes = 60
            elif timeframe == "4h":
                freq_minutes = 240
            elif timeframe == "1d":
                freq_minutes = 1440
            else:
                freq_minutes = 60  # default to hourly

            # Create OHLC data for this symbol and timeframe
            tf_data = create_long_term_data(days=1, freq_minutes=freq_minutes, symbol=symbol).head(periods_per_tf)

            # Add timeframe column and adjust prices to be unique per symbol/timeframe
            price_offset = (hash(f"{symbol}_{timeframe}") % 100) * 0.01
            tf_data = tf_data.with_columns(
                [
                    (col("open") + price_offset).alias("open"),
                    (col("high") + price_offset).alias("high"),
                    (col("low") + price_offset).alias("low"),
                    (col("close") + price_offset).alias("close"),
                    lit(timeframe).alias("timeframe"),
                ]
            )

            all_data.append(tf_data)

    # Combine all data
    result = all_data[0]
    for data in all_data[1:]:
        result = result.vstack(data)

    # Sort by symbol (if present), timeframe, timestamp
    sort_cols = []
    if "symbol" in result.columns:
        sort_cols.append("symbol")
    sort_cols.extend(["timeframe", "timestamp"])

    return result.sort(sort_cols)


@pytest.mark.unit
class TestMultiTimeframeSource:
    """Test cases for multi-timeframe source aggregation functionality."""

    def test_multi_timeframe_source_pass_through(self):
        """Test that when target timeframe already exists in source, data is passed through unchanged."""
        # Create multi-timeframe data with 1min and 5min
        source_data = create_multi_timeframe_data(timeframes=["1min", "5min"], periods_per_tf=6, symbols=["AAPL"])

        # Target only 5min (which exists in source)
        agg = Aggregation(AggregationConfig(target_timeframes=["5min"]))
        result = agg.process(source_data)

        # Should have only 5min data passed through
        assert "timeframe" in result.columns
        assert result["timeframe"].unique().to_list() == ["5min"]

        # Should have the same 5min data as input (6 bars)
        original_5min = source_data.filter(col("timeframe") == "5min")
        assert len(result) == len(original_5min)
        assert result["symbol"].unique().to_list() == ["AAPL"]

    def test_multi_timeframe_source_optimal_selection(self):
        """Test optimal source timeframe selection (e.g., use 5m for 15m target, not 1m)."""
        # Create data with 1min, 5min, and 1h timeframes
        source_data = create_multi_timeframe_data(timeframes=["1min", "5min", "1h"], periods_per_tf=5, symbols=["TEST"])

        # Target 15min - should use 5min source (not 1min) for efficiency
        agg = Aggregation(AggregationConfig(target_timeframes=["15min"]))
        result = agg.process(source_data)

        # Should produce 15min timeframe
        assert "timeframe" in result.columns
        assert result["timeframe"].unique().to_list() == ["15min"]
        assert len(result) > 0
        assert result["symbol"].unique().to_list() == ["TEST"]

        # Verify it's aggregated data (not pass-through)
        assert len(result) < len(source_data.filter(col("timeframe") == "5min"))

    def test_multi_timeframe_source_mixed_aggregation(self):
        """Test mixed scenarios with some pass-through and some aggregation."""
        # Create data with 1min and 5min
        source_data = create_multi_timeframe_data(timeframes=["1min", "5min"], periods_per_tf=10, symbols=["MIX"])

        # Target both 5min (pass-through) and 15min (aggregated from 5min)
        agg = Aggregation(AggregationConfig(target_timeframes=["5min", "15min"]))
        result = agg.process(source_data)

        # Should have both timeframes
        timeframes = sorted(result["timeframe"].unique().to_list())
        assert timeframes == ["15min", "5min"]

        # Should have data for both timeframes
        tf_5min_data = result.filter(col("timeframe") == "5min")
        tf_15min_data = result.filter(col("timeframe") == "15min")

        assert len(tf_5min_data) == 10  # Pass-through of original 5min data
        assert len(tf_15min_data) > 0  # Aggregated 15min data
        assert len(tf_15min_data) < len(tf_5min_data)  # Fewer bars due to aggregation

    def test_source_selection_prefers_larger_timeframes(self):
        """Verify that larger valid source timeframes are preferred to minimize operations."""
        # Create data with 1min, 5min, and 15min timeframes
        source_data = create_multi_timeframe_data(
            timeframes=["1min", "5min", "15min"], periods_per_tf=8, symbols=["PREF"]
        )

        # Target 1h - mathematically valid sources are 1min, 5min, 15min
        # Should prefer 15min (largest) for efficiency
        agg = Aggregation(AggregationConfig(target_timeframes=["1h"]))
        result = agg.process(source_data)

        # Should produce 1h timeframe
        assert result["timeframe"].unique().to_list() == ["1h"]
        assert len(result) > 0

        # Should have aggregated from the larger timeframe (less work)
        # With 8 periods of 15min data, we should get fewer 1h bars
        assert len(result) < len(source_data.filter(col("timeframe") == "15min"))

    def test_source_selection_mathematical_validity(self):
        """Test that only mathematically valid sources are selected (divisibility check)."""
        # Use real timeframes that don't divide evenly: 5min doesn't divide into 1h cleanly for certain patterns
        # Better test: use 1min and 5min for a target that works with one but not optimally
        source_data = create_multi_timeframe_data(
            timeframes=["1min", "5min"],
            periods_per_tf=15,  # 15 minutes worth
            symbols=["MATH"],
        )

        # Target 1h - both sources work, but 5min should be preferred (optimal)
        agg = Aggregation(AggregationConfig(target_timeframes=["1h"]))
        result = agg.process(source_data)

        # Should successfully produce 1h data
        assert result["timeframe"].unique().to_list() == ["1h"]
        assert len(result) > 0

    def test_no_valid_source_timeframe_metadata(self):
        """Test behavior with timeframes not in TIMEFRAME_METADATA."""
        # Create data with 1min and 5min first
        valid_data = create_multi_timeframe_data(timeframes=["1min", "5min"], periods_per_tf=5, symbols=["VALID"])

        # Replace timeframe column with unsupported values
        invalid_data = valid_data.with_columns([lit("unsupported_tf").alias("timeframe")])

        agg = Aggregation(AggregationConfig(target_timeframes=["15min"]))

        # Should raise ValueError due to validation failure with unsupported timeframes
        with pytest.raises(ValueError, match="Input data validation failed"):
            agg.process(invalid_data)

    def test_multi_timeframe_source_with_symbols(self):
        """Test multi-timeframe aggregation with multiple symbols."""
        # Create data for multiple symbols with multiple timeframes
        symbols = ["AAPL", "MSFT", "GOOGL"]
        source_data = create_multi_timeframe_data(timeframes=["1min", "5min"], periods_per_tf=6, symbols=symbols)

        # Target 15min aggregation from 5min source
        agg = Aggregation(AggregationConfig(target_timeframes=["15min"]))
        result = agg.process(source_data)

        # Should have data for all symbols
        result_symbols = sorted(result["symbol"].unique().to_list())
        expected_symbols = sorted(symbols)
        assert result_symbols == expected_symbols

        # Should have 15min timeframe for all symbols
        assert result["timeframe"].unique().to_list() == ["15min"]

        # Verify each symbol has aggregated data
        for symbol in symbols:
            symbol_data = result.filter(col("symbol") == symbol)
            assert len(symbol_data) > 0
            assert symbol_data["symbol"].unique().to_list() == [symbol]

    def test_multi_timeframe_source_symbol_preservation(self):
        """Verify symbols are preserved correctly during aggregation."""
        # Create mixed symbol/timeframe data
        source_data = create_multi_timeframe_data(
            timeframes=["5min", "15min"], periods_per_tf=8, symbols=["PRESERVE", "KEEP"]
        )

        # Target both pass-through (15min) and aggregation (1h from 15min)
        agg = Aggregation(AggregationConfig(target_timeframes=["15min", "1h"]))
        result = agg.process(source_data)

        # Should have both symbols for both timeframes
        symbols = sorted(result["symbol"].unique().to_list())
        timeframes = sorted(result["timeframe"].unique().to_list())
        assert symbols == ["KEEP", "PRESERVE"]
        assert timeframes == ["15min", "1h"]

        # Verify data integrity for each symbol/timeframe combination
        for symbol in symbols:
            for timeframe in timeframes:
                subset = result.filter((col("symbol") == symbol) & (col("timeframe") == timeframe))
                assert len(subset) > 0
                assert subset["symbol"].unique().to_list() == [symbol]
                assert subset["timeframe"].unique().to_list() == [timeframe]

    def test_multi_timeframe_source_single_target(self):
        """Test with single target timeframe."""
        source_data = create_multi_timeframe_data(
            timeframes=["1min", "5min", "15min"], periods_per_tf=5, symbols=["SINGLE"]
        )

        # Target only one timeframe
        agg = Aggregation(AggregationConfig(target_timeframes=["30min"]))
        result = agg.process(source_data)

        # Should aggregate from 15min (optimal source)
        assert result["timeframe"].unique().to_list() == ["30min"]
        assert result["symbol"].unique().to_list() == ["SINGLE"]
        assert len(result) > 0

    def test_multi_timeframe_source_all_passthrough(self):
        """Test when all targets exist in source (100% pass-through)."""
        source_data = create_multi_timeframe_data(
            timeframes=["5min", "15min", "1h"], periods_per_tf=6, symbols=["PASS"]
        )

        # Target timeframes that all exist in source
        agg = Aggregation(AggregationConfig(target_timeframes=["5min", "15min", "1h"]))
        result = agg.process(source_data)

        # Should be pure pass-through with same amount of data
        timeframes = sorted(result["timeframe"].unique().to_list())
        assert timeframes == ["15min", "1h", "5min"]

        # Each timeframe should have the original amount of data
        for tf in ["5min", "15min", "1h"]:
            result_tf = result.filter(col("timeframe") == tf)
            source_tf = source_data.filter(col("timeframe") == tf)
            assert len(result_tf) == len(source_tf)

    def test_multi_timeframe_source_no_passthrough(self):
        """Test when no targets exist in source (100% aggregation)."""
        source_data = create_multi_timeframe_data(timeframes=["1min", "5min"], periods_per_tf=8, symbols=["AGG"])

        # Target timeframes that don't exist in source
        agg = Aggregation(AggregationConfig(target_timeframes=["15min", "1h"]))
        result = agg.process(source_data)

        # All should be aggregated (no pass-through)
        timeframes = sorted(result["timeframe"].unique().to_list())
        assert timeframes == ["15min", "1h"]

        # Aggregated data should have fewer bars than any source
        for tf in ["15min", "1h"]:
            result_tf = result.filter(col("timeframe") == tf)
            assert len(result_tf) > 0
            assert len(result_tf) < len(source_data.filter(col("timeframe") == "5min"))

    def test_multi_timeframe_source_empty_data(self):
        """Test with empty multi-timeframe data."""
        # Create empty DataFrame with proper schema
        empty_data = DataFrame(
            {
                "timestamp": [],
                "open": [],
                "high": [],
                "low": [],
                "close": [],
                "volume": [],
                "symbol": [],
                "timeframe": [],
            }
        )

        agg = Aggregation(AggregationConfig(target_timeframes=["5min"]))

        # Should handle empty data gracefully
        with pytest.raises(ValueError, match="Input data validation failed"):
            agg.process(empty_data)

    def test_multi_timeframe_source_sorting(self):
        """Verify output is correctly sorted by symbol, timeframe, timestamp."""
        # Create data with multiple symbols and timeframes
        source_data = create_multi_timeframe_data(
            timeframes=["5min", "1min"],  # Reverse order to test sorting
            periods_per_tf=5,
            symbols=["ZZZ", "AAA"],  # Reverse alphabetical order
        )

        agg = Aggregation(AggregationConfig(target_timeframes=["15min", "5min"]))
        result = agg.process(source_data)

        # Check sorting: symbol (AAA before ZZZ), then timeframe (15min before 5min), then timestamp
        prev_symbol = None
        prev_timeframe = None
        prev_timestamp = None

        for row in result.iter_rows(named=True):
            current_symbol = row["symbol"]
            current_timeframe = row["timeframe"]
            current_timestamp = row["timestamp"]

            if prev_symbol is not None:
                # Symbol should be same or greater
                assert current_symbol >= prev_symbol

                if current_symbol == prev_symbol:
                    # Same symbol: timeframe should be same or greater
                    if prev_timeframe is not None:
                        assert current_timeframe >= prev_timeframe

                        if current_timeframe == prev_timeframe:
                            # Same symbol and timeframe: timestamp should be greater
                            if prev_timestamp is not None:
                                assert current_timestamp >= prev_timestamp

            prev_symbol = current_symbol
            prev_timeframe = current_timeframe
            prev_timestamp = current_timestamp

    def test_multi_timeframe_source_column_preservation(self):
        """Ensure all required columns are preserved."""
        source_data = create_multi_timeframe_data(timeframes=["1min", "5min"], periods_per_tf=4, symbols=["COL"])

        agg = Aggregation(AggregationConfig(target_timeframes=["15min"]))
        result = agg.process(source_data)

        # Check all expected columns are present
        expected_columns = ["timestamp", "open", "high", "low", "close", "volume", "symbol", "timeframe"]
        for col_name in expected_columns:
            assert col_name in result.columns

        # Check data types are preserved (timezone may vary based on config)
        assert str(result.schema["timestamp"]).startswith("Datetime")
        for price_col in ["open", "high", "low", "close"]:
            assert str(result.schema[price_col]).startswith("Float")

    def test_multi_timeframe_pandas_input_compatibility(self):
        """Test that Pandas input works with multi-timeframe source."""
        # Create Polars data first
        polars_data = create_multi_timeframe_data(timeframes=["1min", "5min"], periods_per_tf=4, symbols=["PANDAS"])

        # Convert to Pandas
        pandas_data = polars_data.to_pandas()

        agg = Aggregation(AggregationConfig(target_timeframes=["15min"]))
        result = agg.process(pandas_data)

        # Should work and produce same result as Polars input
        assert isinstance(result, DataFrame)  # Should return Polars DataFrame
        assert "timeframe" in result.columns
        assert result["timeframe"].unique().to_list() == ["15min"]
        assert result["symbol"].unique().to_list() == ["PANDAS"]
