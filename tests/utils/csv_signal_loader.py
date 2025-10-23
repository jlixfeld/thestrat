"""
CSV-based signal test data loader utilities.

Provides functions to load pre-generated deterministic signal test data
from CSV files created by generate_all_signals.py.
"""

import glob
from pathlib import Path
from typing import Tuple

import polars as pl

# Base path to signal test data fixtures
SIGNAL_DATA_PATH = Path(__file__).parent.parent / "fixtures" / "signals"


def load_signal_test_data(signal_name: str) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Load market and indicators test data for a specific signal.

    **Args:**
    - `signal_name`: Signal pattern name (e.g., "2D-2U", "2U-2D")

    **Returns:**
    Tuple of (market_df, indicators_df):
    - `market_df`: Raw OHLC input data (7 columns: timestamp, symbol, open, high, low, close, volume)
    - `indicators_df`: Expected indicators output with signal detected (41 columns from IndicatorSchema)

    **Raises:**
    - `FileNotFoundError`: If CSV files don't exist for the signal

    **Example:**
    ```python
    from tests.utils.csv_signal_loader import load_signal_test_data

    # Load test data for 2D-2U pattern
    market_df, expected_df = load_signal_test_data("2D-2U")

    # Process market data through indicators
    actual_df = indicators.process(market_df)

    # Compare actual vs expected
    assert_indicators_match(actual_df, expected_df)
    ```

    **Note:**
    The `target_prices` column is automatically converted from CSV format (comma-separated string)
    back to List[Float64] for compatibility with the indicators pipeline.
    """
    # Normalize signal name to filename format
    # "2D-2U" -> "2d_2u"
    filename_signal = signal_name.lower().replace("-", "_")

    # Construct CSV paths in subdirectories
    market_csv = SIGNAL_DATA_PATH / "market" / f"signal_{filename_signal}_market.csv"
    indicators_csv = SIGNAL_DATA_PATH / "indicators" / f"signal_{filename_signal}_indicators.csv"

    # Verify files exist
    if not market_csv.exists():
        raise FileNotFoundError(
            f"Market CSV not found: {market_csv}\nRun 'python generate_all_signals.py' to generate test data."
        )
    if not indicators_csv.exists():
        raise FileNotFoundError(
            f"Indicators CSV not found: {indicators_csv}\nRun 'python generate_all_signals.py' to generate test data."
        )

    # Load CSVs
    market_df = pl.read_csv(market_csv, try_parse_dates=True)
    indicators_df = pl.read_csv(indicators_csv, try_parse_dates=True)

    # Convert target_prices from comma-separated string back to List[Float64]
    # CSV format: "92.0, 90.0, 85.0"  ->  List[92.0, 90.0, 85.0]
    # Empty string or null -> empty list []
    if "target_prices" in indicators_df.columns:
        # Cast to string first to ensure we can do string operations
        indicators_df = indicators_df.with_columns(pl.col("target_prices").cast(pl.Utf8))

        indicators_df = indicators_df.with_columns(
            pl.when(pl.col("target_prices").is_not_null() & (pl.col("target_prices").str.len_chars() > 0))
            .then(pl.col("target_prices").str.split(", ").list.eval(pl.element().cast(pl.Float64)))
            .otherwise(pl.lit([], dtype=pl.List(pl.Float64)))
            .alias("target_prices")
        )

    return market_df, indicators_df


def get_all_signal_names() -> list[str]:
    """
    Get list of all available signal test data.

    **Returns:**
    List of signal pattern names with available CSV test data, sorted alphabetically.

    **Example:**
    ```python
    from tests.utils.csv_signal_loader import get_all_signal_names

    signals = get_all_signal_names()
    assert "2D-2U" in signals
    assert "2U-2D" in signals
    assert len(signals) == 16  # All 16 TheStrat signals
    ```

    **Note:**
    This function scans for `signal_*_market.csv` files in `tests/fixtures/signals/market/`
    and extracts signal names from filenames.
    """
    # Find all market CSV files in market/ subdirectory
    pattern = str(SIGNAL_DATA_PATH / "market" / "signal_*_market.csv")
    market_files = glob.glob(pattern)

    # Extract signal names from filenames
    signals = []
    for filepath in market_files:
        # "signal_2d_2u_market.csv" -> "2d_2u" -> "2D-2U"
        filename = Path(filepath).stem  # Remove .csv extension
        signal_part = filename.replace("signal_", "").replace("_market", "")

        # Convert to standard format: "2d_2u" -> "2D-2U"
        signal_name = signal_part.upper().replace("_", "-")
        signals.append(signal_name)

    return sorted(signals)


def get_signal_chart_path(signal_name: str) -> Path:
    """
    Get path to the PNG chart for a specific signal.

    **Args:**
    - `signal_name`: Signal pattern name (e.g., "2D-2U")

    **Returns:**
    Path object pointing to the signal's PNG chart file.

    **Raises:**
    - `FileNotFoundError`: If chart file doesn't exist

    **Example:**
    ```python
    chart_path = get_signal_chart_path("2D-2U")
    # Use for visual debugging or documentation
    ```

    **Note:**
    Charts are generated by `generate_all_signals.py` and provide visual
    reference for understanding test scenarios.
    """
    filename_signal = signal_name.lower().replace("-", "_")
    chart_path = SIGNAL_DATA_PATH / "charts" / f"signal_{filename_signal}.png"

    if not chart_path.exists():
        raise FileNotFoundError(
            f"Chart not found: {chart_path}\nRun 'python generate_all_signals.py' to generate test data."
        )

    return chart_path


def verify_all_signals_available() -> bool:
    """
    Verify that all 16 expected signal CSVs are available.

    **Returns:**
    True if all 16 signals have market and indicators CSVs, False otherwise.

    **Example:**
    ```python
    # In test setup or CI validation
    assert verify_all_signals_available(), "Missing signal test data - run generate_all_signals.py"
    ```
    """
    expected_signals = [
        # Reversal Long
        "2D-2U",
        "1-2D-2U",
        "2D-1-2U",
        "3-1-2U",
        "3-2D-2U",
        # Reversal Short
        "2U-2D",
        "1-2U-2D",
        "2U-1-2D",
        "3-1-2D",
        "3-2U-2D",
        # Continuation
        "2U-2U",
        "2U-1-2U",
        "2D-2D",
        "2D-1-2D",
        # Context-Aware
        "3-2U",
        "3-2D",
    ]

    available_signals = get_all_signal_names()

    for signal in expected_signals:
        if signal not in available_signals:
            return False

    return len(available_signals) == 16
