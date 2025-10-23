"""
Signal validation utilities for CSV-based testing.

Provides assertion and validation functions for comparing actual indicators
output against expected CSV test data.
"""

from typing import List, Optional

import polars as pl
from polars import col


def assert_signal_detected(df: pl.DataFrame, expected_signal: str, min_occurrences: int = 1) -> None:
    """
    Assert that expected signal is detected in the indicators DataFrame.

    **Args:**
    - `df`: Indicators output DataFrame with signal column
    - `expected_signal`: Expected signal pattern name (e.g., "2D-2U")
    - `min_occurrences`: Minimum number of times signal should appear (default: 1)

    **Raises:**
    - `AssertionError`: If signal not detected, wrong signal detected, or insufficient occurrences

    **Example:**
    ```python
    from tests.utils.signal_validator import assert_signal_detected

    # Process indicators
    result = indicators.process(market_df)

    # Verify 2D-2U signal was detected at least once
    assert_signal_detected(result, "2D-2U")

    # Verify continuation signal appears multiple times
    assert_signal_detected(result, "2U-2U", min_occurrences=3)
    ```
    """
    signal_rows = df.filter(col("signal").is_not_null())

    if len(signal_rows) == 0:
        raise AssertionError(f"No signals detected, expected {expected_signal}")

    detected_signals = signal_rows["signal"].unique().to_list()

    if expected_signal not in detected_signals:
        raise AssertionError(
            f"Expected signal '{expected_signal}' not detected.\n"
            f"Detected signals: {detected_signals}\n"
            f"Total signal occurrences: {len(signal_rows)}"
        )

    # Check occurrence count
    signal_count = len(signal_rows.filter(col("signal") == expected_signal))
    if signal_count < min_occurrences:
        raise AssertionError(
            f"Expected signal '{expected_signal}' detected {signal_count} times, "
            f"but expected at least {min_occurrences} occurrences"
        )


def assert_indicators_match(
    actual_df: pl.DataFrame,
    expected_df: pl.DataFrame,
    check_columns: Optional[List[str]] = None,
    ignore_columns: Optional[List[str]] = None,
) -> None:
    """
    Assert that actual indicators output matches expected output from CSV.

    **Args:**
    - `actual_df`: Actual indicators output from processing
    - `expected_df`: Expected indicators output from CSV
    - `check_columns`: If provided, only validate these specific columns
    - `ignore_columns`: If provided, exclude these columns from validation

    **Raises:**
    - `AssertionError`: If outputs don't match

    **Example:**
    ```python
    from tests.utils.signal_validator import assert_indicators_match

    # Full DataFrame comparison
    assert_indicators_match(actual_df, expected_df)

    # Only check signal-related columns
    signal_cols = ["signal", "type", "bias", "entry_price", "stop_price"]
    assert_indicators_match(actual_df, expected_df, check_columns=signal_cols)

    # Ignore timestamp column (may differ due to processing time)
    assert_indicators_match(actual_df, expected_df, ignore_columns=["timestamp"])
    ```

    **Note:**
    This performs exact DataFrame equality check using Polars' `equals()` method.
    For looser comparisons (e.g., float tolerance), use column-specific validation.
    """
    if check_columns is not None:
        # Only check specified columns
        actual_subset = actual_df.select(check_columns)
        expected_subset = expected_df.select(check_columns)

        if not actual_subset.equals(expected_subset):
            # Provide detailed diff for debugging
            diff_rows = []
            for i in range(min(len(actual_subset), len(expected_subset))):
                actual_row = actual_subset.slice(i, 1)
                expected_row = expected_subset.slice(i, 1)
                if not actual_row.equals(expected_row):
                    diff_rows.append(i)

            raise AssertionError(
                f"Selected columns don't match expected.\n"
                f"Checked columns: {check_columns}\n"
                f"Rows with differences: {diff_rows[:5]}{'...' if len(diff_rows) > 5 else ''}"
            )

    elif ignore_columns is not None:
        # Check all columns except ignored ones
        all_cols = actual_df.columns
        check_cols = [c for c in all_cols if c not in ignore_columns]

        actual_subset = actual_df.select(check_cols)
        expected_subset = expected_df.select(check_cols)

        if not actual_subset.equals(expected_subset):
            raise AssertionError(
                f"DataFrames don't match (ignoring columns: {ignore_columns}).\nChecked columns: {check_cols}"
            )

    else:
        # Full DataFrame comparison
        if not actual_df.equals(expected_df):
            raise AssertionError(
                f"Indicators output doesn't match expected CSV.\n"
                f"Actual shape: {actual_df.shape}\n"
                f"Expected shape: {expected_df.shape}\n"
                f"Use check_columns or ignore_columns for partial validation."
            )


def get_signal_rows(df: pl.DataFrame, signal_name: str) -> pl.DataFrame:
    """
    Extract rows containing a specific signal from indicators DataFrame.

    **Args:**
    - `df`: Indicators DataFrame
    - `signal_name`: Signal pattern to filter for (e.g., "2D-2U")

    **Returns:**
    DataFrame containing only rows with the specified signal

    **Example:**
    ```python
    from tests.utils.signal_validator import get_signal_rows

    # Get all 2D-2U signal occurrences
    signal_rows = get_signal_rows(indicators_df, "2D-2U")
    assert len(signal_rows) > 0, "No 2D-2U signals found"

    # Get first occurrence
    first_signal = signal_rows.slice(0, 1)
    signal_obj = Indicators.get_signal_object(first_signal)
    ```
    """
    return df.filter(col("signal") == signal_name)


def assert_signal_properties(
    df: pl.DataFrame,
    signal_name: str,
    expected_type: str,
    expected_bias: str,
    expected_swing_point: Optional[str] = None,
) -> None:
    """
    Assert that signal has expected type, bias, and optional swing point location.

    **Args:**
    - `df`: Indicators DataFrame
    - `signal_name`: Signal pattern name
    - `expected_type`: Expected signal type ("reversal", "continuation", "breakout")
    - `expected_bias`: Expected bias ("long" or "short")
    - `expected_swing_point`: Optional swing point flag to check
                             ("signal_at_higher_high", "signal_at_lower_low", etc.)

    **Raises:**
    - `AssertionError`: If properties don't match or signal not found

    **Example:**
    ```python
    # Validate 2D-2U reversal properties
    assert_signal_properties(
        result_df,
        signal_name="2D-2U",
        expected_type="reversal",
        expected_bias="long",
        expected_swing_point="signal_at_lower_low"
    )

    # Validate continuation properties (no swing point requirement)
    assert_signal_properties(
        result_df,
        signal_name="2U-2U",
        expected_type="continuation",
        expected_bias="long"
    )
    ```
    """
    signal_rows = get_signal_rows(df, signal_name)

    if len(signal_rows) == 0:
        raise AssertionError(f"Signal '{signal_name}' not found in DataFrame")

    # Check first occurrence (all should have same properties)
    first_row = signal_rows.slice(0, 1)

    actual_type = first_row["type"][0]
    actual_bias = first_row["bias"][0]

    if actual_type != expected_type:
        raise AssertionError(f"Signal '{signal_name}' has type '{actual_type}', expected '{expected_type}'")

    if actual_bias != expected_bias:
        raise AssertionError(f"Signal '{signal_name}' has bias '{actual_bias}', expected '{expected_bias}'")

    # Check swing point flag if specified
    if expected_swing_point is not None:
        if expected_swing_point not in signal_rows.columns:
            raise ValueError(f"Column '{expected_swing_point}' not found in DataFrame")

        # Check if ANY occurrence has the expected swing point flag
        swing_point_values = signal_rows[expected_swing_point].to_list()
        if not any(swing_point_values):
            raise AssertionError(
                f"Signal '{signal_name}' should have at least one occurrence with {expected_swing_point}=True, "
                f"but all {len(swing_point_values)} occurrences have it as False"
            )


def assert_target_count(df: pl.DataFrame, signal_name: str, expected_count: int) -> None:
    """
    Assert that signal has expected number of target prices.

    **Args:**
    - `df`: Indicators DataFrame
    - `signal_name`: Signal pattern name
    - `expected_count`: Expected number of targets (0 for continuations)

    **Raises:**
    - `AssertionError`: If target count doesn't match

    **Example:**
    ```python
    # Validate reversal has targets
    assert_target_count(result_df, "2D-2U", expected_count=5)

    # Validate continuation has no targets
    assert_target_count(result_df, "2U-2U", expected_count=0)
    ```
    """
    signal_rows = get_signal_rows(df, signal_name)

    if len(signal_rows) == 0:
        raise AssertionError(f"Signal '{signal_name}' not found in DataFrame")

    first_row = signal_rows.slice(0, 1)
    actual_count = first_row["target_count"][0]

    if actual_count != expected_count:
        raise AssertionError(f"Signal '{signal_name}' has {actual_count} targets, expected {expected_count}")


def assert_entry_stop_valid(df: pl.DataFrame, signal_name: str) -> None:
    """
    Assert that entry and stop prices are valid for signal's bias.

    **Args:**
    - `df`: Indicators DataFrame
    - `signal_name`: Signal pattern name

    **Raises:**
    - `AssertionError`: If entry/stop relationship is invalid for bias

    **Example:**
    ```python
    # Will assert that for long signal: entry > stop
    # Will assert that for short signal: stop > entry
    assert_entry_stop_valid(result_df, "2D-2U")
    ```

    **Note:**
    This validates the basic risk management logic:
    - Long signals: Entry above stop (buy higher, stop lower)
    - Short signals: Stop above entry (sell lower, stop higher)
    """
    signal_rows = get_signal_rows(df, signal_name)

    if len(signal_rows) == 0:
        raise AssertionError(f"Signal '{signal_name}' not found in DataFrame")

    first_row = signal_rows.slice(0, 1)

    entry = first_row["entry_price"][0]
    stop = first_row["stop_price"][0]
    bias = first_row["bias"][0]

    if entry is None or stop is None:
        raise AssertionError(f"Signal '{signal_name}' has null entry or stop price")

    if bias == "long":
        if entry <= stop:
            raise AssertionError(
                f"Long signal '{signal_name}' has entry ({entry}) <= stop ({stop}), but entry should be > stop"
            )
    elif bias == "short":
        if stop <= entry:
            raise AssertionError(
                f"Short signal '{signal_name}' has stop ({stop}) <= entry ({entry}), but stop should be > entry"
            )
    else:
        raise ValueError(f"Unknown bias '{bias}' for signal '{signal_name}'")
