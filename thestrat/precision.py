"""Security-aware precision utilities for indicator field rounding."""

from typing import Dict

import polars as pl

from .schemas import IndicatorSchema


class PrecisionError(Exception):
    """Raised when precision cannot be determined."""

    pass


# Field classification from schema
PRECISION_TYPE_PERCENTAGE = "percentage"
PRECISION_TYPE_PRICE = "price"
PRECISION_TYPE_INTEGER = "integer"


def get_field_precision_type(field_name: str) -> str | None:
    """
    Get precision type for a field from IndicatorSchema metadata.

    Args:
        field_name: Name of the indicator field

    Returns:
        Precision type: 'percentage', 'price', 'integer', or None
    """
    metadata = IndicatorSchema.get_field_metadata(field_name)
    return metadata.get("precision_type")


def get_field_decimal_places(field_name: str, security_precision: int = 2) -> int | None:
    """
    Get decimal places for a field based on its precision type.

    Args:
        field_name: Name of the indicator field
        security_precision: Decimal places for this security (from IBKR minTick)

    Returns:
        Number of decimal places, or None for integer fields

    Raises:
        PrecisionError: If field not found or precision_type missing
    """
    metadata = IndicatorSchema.get_field_metadata(field_name)

    if not metadata:
        raise PrecisionError(f"Field '{field_name}' not found in IndicatorSchema")

    precision_type = metadata.get("precision_type")

    if precision_type is None:
        raise PrecisionError(f"Field '{field_name}' missing 'precision_type' in json_schema_extra")

    if precision_type == PRECISION_TYPE_PERCENTAGE:
        return 2  # Percentages always 2 decimals
    elif precision_type == PRECISION_TYPE_PRICE:
        return security_precision  # Use security's precision from IBKR
    elif precision_type == PRECISION_TYPE_INTEGER:
        return None  # No rounding for integers
    else:
        raise PrecisionError(f"Unknown precision_type '{precision_type}' for field '{field_name}'")


def apply_precision(
    df: pl.DataFrame, security_precision_map: Dict[str, int], symbol_column: str = "symbol"
) -> pl.DataFrame:
    """
    Apply security-aware precision rounding to indicator DataFrame.

    Args:
        df: DataFrame with indicator columns
        security_precision_map: Dict mapping symbol → decimal places (from IBKR minTick)
        symbol_column: Column name containing symbols (default: 'symbol')

    Returns:
        DataFrame with rounded values

    Raises:
        PrecisionError: If a symbol in df is not in security_precision_map

    Example:
        ```python
        # Precision from IBKR: {'AAPL': 2, 'EURUSD': 5, 'BTC': 8}
        rounded_df = apply_precision(indicators_df, precision_map)
        ```
    """
    # Validate all symbols have precision
    df_symbols = df[symbol_column].unique().to_list()
    missing_symbols = set(df_symbols) - set(security_precision_map.keys())

    if missing_symbols:
        raise PrecisionError(
            f"Missing precision for symbols: {sorted(missing_symbols)}. "
            f"All symbols must have precision fetched from IBKR before applying."
        )

    # Process each symbol group separately
    result_parts = []

    for symbol_tuple, group_df in df.group_by(symbol_column, maintain_order=True):
        symbol = symbol_tuple[0] if isinstance(symbol_tuple, tuple) else symbol_tuple
        security_precision = security_precision_map[symbol]

        # Round each field based on its precision type
        for field_name in group_df.columns:
            try:
                decimal_places = get_field_decimal_places(field_name, security_precision)

                if decimal_places is not None and field_name in group_df.columns:
                    # Handle list columns (target_prices)
                    if group_df.schema[field_name] == pl.List(pl.Float64):
                        group_df = group_df.with_columns(
                            pl.col(field_name).list.eval(pl.element().round(decimal_places))
                        )
                    else:
                        # Regular float columns
                        group_df = group_df.with_columns(pl.col(field_name).round(decimal_places))
            except PrecisionError:
                # Field not in schema or no precision metadata - skip
                continue

        result_parts.append(group_df)

    return pl.concat(result_parts)


def get_comparison_tolerance(field_name: str, security_precision: int = 2) -> float:
    """
    Get comparison tolerance for a field based on its precision.

    Args:
        field_name: Field to compare
        security_precision: Precision for the security (from IBKR minTick)

    Returns:
        Tolerance value (10^-decimal_places) or 0 for exact comparison

    Example:
        ```python
        # For percent_close_from_high (2 decimals): returns 0.01
        tolerance = get_comparison_tolerance('percent_close_from_high', 2)

        # For ath with security_precision=5: returns 0.00001
        tolerance = get_comparison_tolerance('ath', 5)

        # For target_count (integer): returns 0 (exact comparison)
        tolerance = get_comparison_tolerance('target_count', 2)
        ```
    """
    try:
        decimal_places = get_field_decimal_places(field_name, security_precision)

        if decimal_places is not None:
            return 10 ** (-decimal_places)
        else:
            # Integer fields - exact comparison
            return 0
    except PrecisionError:
        # Unknown field - use small epsilon
        return 1e-6
