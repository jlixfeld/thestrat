"""
Base component classes for TheStrat module.

This module provides the abstract base class and core functionality for all TheStrat components.
"""

from abc import ABC, abstractmethod

from pandas import DataFrame as PandasDataFrame
from polars import DataFrame as PolarsDataFrame
from polars import from_pandas

# Required columns for OHLC data
REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close"]

# Optional columns
OPTIONAL_COLUMNS = ["symbol", "volume"]

# Supported input formats
SUPPORTED_INPUT_TYPES = [
    "polars.DataFrame",  # Native format (preferred)
    "pandas.DataFrame",  # Auto-converted to Polars
]


class Component(ABC):
    """Base class for all TheStrat components."""

    def __init__(self):
        """Initialize base component with metadata tracking."""
        from datetime import datetime

        self._created_at = datetime.now()

    @abstractmethod
    def process(self, data: PolarsDataFrame | PandasDataFrame) -> PolarsDataFrame:
        """
        Process data and return Polars DataFrame results.

        Args:
            data: Input DataFrame (Polars or Pandas)

        Returns:
            Processed PolarsDataFrame with results
        """
        pass

    @abstractmethod
    def validate_input(self, data: PolarsDataFrame | PandasDataFrame) -> None:
        """
        Validate input data format.

        Args:
            data: Input DataFrame to validate

        Raises:
            ValueError: If data format is invalid, with specific error message
        """
        pass

    def _convert_to_polars(self, data: PolarsDataFrame | PandasDataFrame) -> PolarsDataFrame:
        """
        Convert input DataFrame to Polars if needed.

        Args:
            data: Input DataFrame (Polars or Pandas)

        Returns:
            PolarsDataFrame (converted if input was Pandas)
        """
        if isinstance(data, PandasDataFrame):
            return from_pandas(data)
        elif isinstance(data, PolarsDataFrame):
            return data
        else:
            raise TypeError(f"Unsupported data type: {type(data)}. Expected pandas.DataFrame or polars.DataFrame")
