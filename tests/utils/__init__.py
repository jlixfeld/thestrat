"""Test utilities package."""

from .config_helpers import (
    create_aggregation_config,
    create_crypto_aggregation_config,
    create_equity_aggregation_config,
    create_fx_aggregation_config,
    create_indicators_config,
)
from .csv_signal_loader import (
    get_all_signal_names,
    get_signal_chart_path,
    load_signal_test_data,
    verify_all_signals_available,
)
from .signal_validator import (
    assert_entry_stop_valid,
    assert_indicators_match,
    assert_signal_detected,
    assert_signal_properties,
    assert_target_count,
    get_signal_rows,
)

__all__ = [
    # Config helpers
    "create_aggregation_config",
    "create_crypto_aggregation_config",
    "create_equity_aggregation_config",
    "create_fx_aggregation_config",
    "create_indicators_config",
    # CSV signal loader
    "load_signal_test_data",
    "get_all_signal_names",
    "get_signal_chart_path",
    "verify_all_signals_available",
    # Signal validator
    "assert_signal_detected",
    "assert_indicators_match",
    "get_signal_rows",
    "assert_signal_properties",
    "assert_target_count",
    "assert_entry_stop_valid",
]
