"""Unit tests for precision utilities."""

import polars as pl
import pytest

from thestrat.precision import (
    PrecisionError,
    apply_precision,
    get_comparison_tolerance,
    get_field_decimal_places,
    get_field_precision_type,
)
from thestrat.schemas import IndicatorSchema


class TestGetFieldPrecisionType:
    """Test get_field_precision_type function."""

    def test_percentage_field(self):
        """Test percentage field returns 'percentage'."""
        assert get_field_precision_type("percent_close_from_high") == "percentage"
        assert get_field_precision_type("percent_close_from_low") == "percentage"

    def test_price_field(self):
        """Test price field returns 'price'."""
        assert get_field_precision_type("open") == "price"
        assert get_field_precision_type("high") == "price"
        assert get_field_precision_type("low") == "price"
        assert get_field_precision_type("close") == "price"
        assert get_field_precision_type("ath") == "price"
        assert get_field_precision_type("atl") == "price"
        assert get_field_precision_type("higher_high") == "price"
        assert get_field_precision_type("lower_high") == "price"
        assert get_field_precision_type("higher_low") == "price"
        assert get_field_precision_type("lower_low") == "price"
        assert get_field_precision_type("entry_price") == "price"
        assert get_field_precision_type("stop_price") == "price"
        assert get_field_precision_type("f23_trigger") == "price"

    def test_integer_field(self):
        """Test integer field returns 'integer'."""
        assert get_field_precision_type("target_count") == "integer"
        assert get_field_precision_type("continuity") == "integer"
        assert get_field_precision_type("gapper") == "integer"
        assert get_field_precision_type("kicker") == "integer"
        assert get_field_precision_type("pmg") == "integer"

    def test_nonexistent_field(self):
        """Test nonexistent field returns None."""
        assert get_field_precision_type("nonexistent_field") is None


class TestGetFieldDecimalPlaces:
    """Test get_field_decimal_places function."""

    def test_percentage_field_always_2_decimals(self):
        """Test percentage fields always return 2 decimals."""
        assert get_field_decimal_places("percent_close_from_high", security_precision=2) == 2
        assert get_field_decimal_places("percent_close_from_high", security_precision=5) == 2
        assert get_field_decimal_places("percent_close_from_high", security_precision=8) == 2

    def test_price_field_uses_security_precision(self):
        """Test price fields use security_precision parameter."""
        assert get_field_decimal_places("open", security_precision=2) == 2
        assert get_field_decimal_places("close", security_precision=5) == 5
        assert get_field_decimal_places("ath", security_precision=8) == 8

    def test_integer_field_returns_none(self):
        """Test integer fields return None (no rounding)."""
        assert get_field_decimal_places("target_count", security_precision=2) is None
        assert get_field_decimal_places("continuity", security_precision=5) is None
        assert get_field_decimal_places("pmg", security_precision=8) is None

    def test_nonexistent_field_raises_error(self):
        """Test nonexistent field raises PrecisionError."""
        with pytest.raises(PrecisionError, match="not found in IndicatorSchema"):
            get_field_decimal_places("nonexistent_field")

    def test_field_without_precision_type_raises_error(self):
        """Test field without precision_type raises PrecisionError."""
        # timestamp doesn't have precision_type
        with pytest.raises(PrecisionError, match="missing 'precision_type'"):
            get_field_decimal_places("timestamp")


class TestGetComparisonTolerance:
    """Test get_comparison_tolerance function."""

    def test_percentage_field_tolerance(self):
        """Test percentage field returns 0.01 tolerance."""
        assert get_comparison_tolerance("percent_close_from_high") == 0.01
        assert get_comparison_tolerance("percent_close_from_low") == 0.01

    def test_price_field_tolerance_varies_by_precision(self):
        """Test price field tolerance varies by security precision."""
        assert get_comparison_tolerance("open", security_precision=2) == 0.01
        assert get_comparison_tolerance("close", security_precision=5) == 0.00001
        assert get_comparison_tolerance("ath", security_precision=8) == 0.00000001

    def test_integer_field_returns_zero_tolerance(self):
        """Test integer fields return 0 (exact comparison)."""
        assert get_comparison_tolerance("target_count") == 0
        assert get_comparison_tolerance("continuity") == 0

    def test_unknown_field_returns_small_epsilon(self):
        """Test unknown field returns 1e-6 epsilon."""
        assert get_comparison_tolerance("nonexistent_field") == 1e-6


class TestApplyPrecision:
    """Test apply_precision function."""

    def test_rounds_percentage_fields_to_2_decimals(self):
        """Test percentage fields are rounded to 2 decimals."""
        df = pl.DataFrame(
            {
                "symbol": ["AAPL", "AAPL"],
                "percent_close_from_high": [45.123456, 67.987654],
                "percent_close_from_low": [54.876543, 32.012345],
            }
        )
        precision_map = {"AAPL": 2}
        result = apply_precision(df, precision_map)

        assert result["percent_close_from_high"].to_list() == [45.12, 67.99]
        assert result["percent_close_from_low"].to_list() == [54.88, 32.01]

    def test_rounds_price_fields_by_security_precision(self):
        """Test price fields are rounded by security precision."""
        df = pl.DataFrame(
            {
                "symbol": ["AAPL", "EURUSD", "BTC"],
                "open": [150.123456, 1.234567890, 42358.12345678],
                "close": [151.987654, 1.987654321, 42500.98765432],
            }
        )
        precision_map = {"AAPL": 2, "EURUSD": 5, "BTC": 8}
        result = apply_precision(df, precision_map)

        aapl = result.filter(pl.col("symbol") == "AAPL")
        eurusd = result.filter(pl.col("symbol") == "EURUSD")
        btc = result.filter(pl.col("symbol") == "BTC")

        assert aapl["open"][0] == 150.12
        assert aapl["close"][0] == 151.99

        assert eurusd["open"][0] == 1.23457
        assert eurusd["close"][0] == 1.98765

        assert btc["open"][0] == 42358.12345678
        assert btc["close"][0] == 42500.98765432

    def test_does_not_round_integer_fields(self):
        """Test integer fields are not rounded."""
        df = pl.DataFrame({"symbol": ["AAPL"], "target_count": [5], "continuity": [1], "pmg": [3]})
        precision_map = {"AAPL": 2}
        result = apply_precision(df, precision_map)

        assert result["target_count"][0] == 5
        assert result["continuity"][0] == 1
        assert result["pmg"][0] == 3

    def test_handles_list_columns(self):
        """Test list columns (target_prices) are rounded element-wise."""
        df = pl.DataFrame(
            {
                "symbol": ["AAPL"],
                "target_prices": [[150.123456, 151.987654, 153.555555]],
            }
        )
        precision_map = {"AAPL": 2}
        result = apply_precision(df, precision_map)

        # Extract the list and compare
        rounded_list = result["target_prices"].to_list()[0]
        assert rounded_list == [150.12, 151.99, 153.56]

    def test_handles_null_values(self):
        """Test null values are preserved."""
        df = pl.DataFrame(
            {
                "symbol": ["AAPL", "AAPL"],
                "entry_price": [150.123456, None],
                "stop_price": [None, 148.987654],
            }
        )
        precision_map = {"AAPL": 2}
        result = apply_precision(df, precision_map)

        assert result["entry_price"][0] == 150.12
        assert result["entry_price"][1] is None
        assert result["stop_price"][0] is None
        assert result["stop_price"][1] == 148.99

    def test_raises_error_for_missing_symbol_precision(self):
        """Test raises PrecisionError when symbol not in precision_map."""
        df = pl.DataFrame({"symbol": ["AAPL", "TSLA"], "open": [150.12, 250.34]})
        precision_map = {"AAPL": 2}  # Missing TSLA

        with pytest.raises(PrecisionError, match="Missing precision for symbols: \\['TSLA'\\]"):
            apply_precision(df, precision_map)

    def test_maintains_column_order(self):
        """Test column order is maintained."""
        df = pl.DataFrame(
            {
                "symbol": ["AAPL"],
                "timestamp": [1234567890],
                "open": [150.123],
                "close": [151.987],
                "target_count": [3],
            }
        )
        precision_map = {"AAPL": 2}
        result = apply_precision(df, precision_map)

        assert result.columns == ["symbol", "timestamp", "open", "close", "target_count"]

    def test_works_with_multiple_symbols(self):
        """Test works correctly with mixed symbols in same DataFrame."""
        df = pl.DataFrame(
            {
                "symbol": ["AAPL", "EURUSD", "AAPL", "EURUSD"],
                "open": [150.123456, 1.234567890, 151.987654, 1.987654321],
            }
        )
        precision_map = {"AAPL": 2, "EURUSD": 5}
        result = apply_precision(df, precision_map)

        aapl_rows = result.filter(pl.col("symbol") == "AAPL")
        eurusd_rows = result.filter(pl.col("symbol") == "EURUSD")

        assert aapl_rows["open"].to_list() == [150.12, 151.99]
        assert eurusd_rows["open"].to_list() == [1.23457, 1.98765]


class TestGetPrecisionMetadata:
    """Test IndicatorSchema.get_precision_metadata classmethod."""

    def test_returns_all_fields_with_precision_type(self):
        """Test returns metadata for all fields with precision_type."""
        metadata = IndicatorSchema.get_precision_metadata()

        # Should have precision metadata for all indicator fields
        assert "percent_close_from_high" in metadata
        assert "open" in metadata
        assert "target_count" in metadata

    def test_percentage_fields_have_correct_metadata(self):
        """Test percentage fields have precision_type='percentage' and decimal_places=2."""
        metadata = IndicatorSchema.get_precision_metadata()

        assert metadata["percent_close_from_high"] == {"precision_type": "percentage", "decimal_places": 2}
        assert metadata["percent_close_from_low"] == {"precision_type": "percentage", "decimal_places": 2}

    def test_price_fields_have_correct_metadata(self):
        """Test price fields have precision_type='price' and decimal_places=None."""
        metadata = IndicatorSchema.get_precision_metadata()

        assert metadata["open"] == {"precision_type": "price", "decimal_places": None}
        assert metadata["close"] == {"precision_type": "price", "decimal_places": None}
        assert metadata["ath"] == {"precision_type": "price", "decimal_places": None}

    def test_integer_fields_have_correct_metadata(self):
        """Test integer fields have precision_type='integer' and decimal_places=None."""
        metadata = IndicatorSchema.get_precision_metadata()

        assert metadata["target_count"] == {"precision_type": "integer", "decimal_places": None}
        assert metadata["continuity"] == {"precision_type": "integer", "decimal_places": None}

    def test_does_not_include_fields_without_precision_type(self):
        """Test does not include fields without precision_type (like timestamp)."""
        metadata = IndicatorSchema.get_precision_metadata()

        assert "timestamp" not in metadata
        assert "symbol" not in metadata
        assert "timeframe" not in metadata
