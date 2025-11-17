# Precision Utilities

The precision utilities provide security-type-aware rounding for indicator fields to ensure consistent behavior between live trading and backtesting platforms.

## Overview

Different securities have different price precision requirements:
- **Equities** (AAPL, TSLA): 2 decimal places ($150.12)
- **Forex** (EURUSD): 5 decimal places (1.23457)
- **Crypto** (BTC): 8 decimal places (42358.12345678)

The precision module automatically handles these differences for all indicator fields.

## Quick Start

```python
from thestrat import apply_precision, IndicatorSchema
import polars as pl

# Your indicator DataFrame
df = pl.DataFrame({
    "symbol": ["AAPL", "EURUSD", "BTC"],
    "open": [150.123456, 1.234567890, 42358.12345678901],
    "close": [151.987654, 1.987654321, 42500.98765432109],
    "percent_close_from_high": [45.123456, 67.987654, 23.456789],
})

# Define precision per security (from IBKR minTick)
precision_map = {
    "AAPL": 2,      # $0.01 minimum tick
    "EURUSD": 5,    # 0.00001 minimum tick
    "BTC": 8,       # 0.00000001 minimum tick
}

# Apply precision rounding
rounded_df = apply_precision(df, precision_map)

# Results:
# AAPL:   open=150.12, close=151.99, percent=45.12
# EURUSD: open=1.23457, close=1.98765, percent=67.99
# BTC:    open=42358.12345678, close=42500.98765432, percent=23.46
```

## Field Types

All indicator fields have precision metadata defining how they should be rounded:

### Percentage Fields (Always 2 Decimals)

Percentage fields are always rounded to 2 decimal places, regardless of security:

```python
from thestrat import get_field_decimal_places

# Percentage fields always return 2
get_field_decimal_places("percent_close_from_high", security_precision=2)  # → 2
get_field_decimal_places("percent_close_from_high", security_precision=5)  # → 2
get_field_decimal_places("percent_close_from_high", security_precision=8)  # → 2
```

**Fields:**
- `percent_close_from_high`
- `percent_close_from_low`

### Price Fields (Security-Dependent)

Price fields use the security's precision from IBKR minTick:

```python
# Price fields use security_precision parameter
get_field_decimal_places("open", security_precision=2)  # → 2 (equities)
get_field_decimal_places("close", security_precision=5)  # → 5 (forex)
get_field_decimal_places("ath", security_precision=8)    # → 8 (crypto)
```

**Fields:**
- OHLC: `open`, `high`, `low`, `close`
- Price levels: `ath`, `atl`
- Market structure: `higher_high`, `lower_high`, `higher_low`, `lower_low`
- Signal prices: `entry_price`, `stop_price`, `target_prices`, `f23_trigger`

### Integer/Boolean Fields (No Rounding)

Integer and boolean fields are never rounded:

```python
# Integer fields return None (no rounding)
get_field_decimal_places("target_count", security_precision=2)  # → None
get_field_decimal_places("continuity", security_precision=5)    # → None
```

**Fields:**
- Integers: `target_count`, `continuity`, `gapper`, `kicker`, `pmg`
- Booleans: `new_ath`, `new_atl`, `in_force`, `hammer`, `shooter`, `f23`, etc.
- Strings: `signal`, `type`, `bias`, `scenario`, `f23x`

## API Reference

### apply_precision()

Apply security-aware precision rounding to an entire DataFrame:

```python
from thestrat import apply_precision

rounded_df = apply_precision(
    df,                      # DataFrame with indicator columns
    security_precision_map,  # Dict[str, int]: symbol → decimal places
    symbol_column="symbol"   # Column containing symbols (default: "symbol")
)
```

**Parameters:**
- `df`: Polars DataFrame with indicator columns
- `security_precision_map`: Dictionary mapping symbol to decimal places
- `symbol_column`: Name of column containing symbols (default: `"symbol"`)

**Returns:** DataFrame with all fields rounded according to their precision type

**Raises:** `PrecisionError` if any symbol in the DataFrame is missing from the precision map

**Features:**
- Handles list columns (`target_prices`) element-wise
- Preserves null values
- Maintains column order
- Processes multiple symbols in single DataFrame

### get_field_decimal_places()

Get decimal places for a specific field:

```python
from thestrat import get_field_decimal_places

decimal_places = get_field_decimal_places(
    field_name,              # Name of indicator field
    security_precision=2     # Decimal places for security (default: 2)
)
```

**Returns:**
- `int`: Number of decimal places (for percentage/price fields)
- `None`: No rounding needed (for integer/boolean/string fields)

**Raises:** `PrecisionError` if field not found or missing precision metadata

### get_field_precision_type()

Get the precision type for a field:

```python
from thestrat import get_field_precision_type

precision_type = get_field_precision_type("open")  # → "price"
precision_type = get_field_precision_type("percent_close_from_high")  # → "percentage"
precision_type = get_field_precision_type("target_count")  # → "integer"
```

**Returns:**
- `"percentage"`: Always 2 decimals
- `"price"`: Security-dependent decimals
- `"integer"`: No rounding
- `None`: Field not in schema

### get_comparison_tolerance()

Get comparison tolerance for floating-point comparisons:

```python
from thestrat import get_comparison_tolerance

# For assertions/tests: value1 ≈ value2 within tolerance
tolerance = get_comparison_tolerance(
    field_name,              # Field to compare
    security_precision=2     # Security precision (default: 2)
)
```

**Returns:**
- `10^(-decimal_places)` for percentage/price fields
- `0` for integer fields (exact comparison)
- `1e-6` for unknown fields (small epsilon)

**Example:**
```python
# Percentage field (2 decimals) → 0.01 tolerance
assert abs(value1 - value2) < get_comparison_tolerance("percent_close_from_high")

# Price field with 5 decimals → 0.00001 tolerance
assert abs(price1 - price2) < get_comparison_tolerance("close", security_precision=5)

# Integer field → 0 (exact comparison)
assert value1 == value2  # get_comparison_tolerance("target_count") == 0
```

### IndicatorSchema.get_precision_metadata()

Get precision metadata for all fields:

```python
from thestrat import IndicatorSchema

metadata = IndicatorSchema.get_precision_metadata()

# Returns: dict[str, dict[str, Any]]
# {
#     "percent_close_from_high": {"precision_type": "percentage", "decimal_places": 2},
#     "open": {"precision_type": "price", "decimal_places": None},
#     "target_count": {"precision_type": "integer", "decimal_places": None},
#     ...
# }
```

## Advanced Usage

### Multi-Symbol DataFrames

The `apply_precision()` function handles multiple symbols with different precisions:

```python
import polars as pl
from thestrat import apply_precision

# Mixed symbols in one DataFrame
df = pl.DataFrame({
    "symbol": ["AAPL", "EURUSD", "AAPL", "EURUSD"],
    "timestamp": [...],
    "open": [150.123456, 1.234567890, 151.987654, 1.987654321],
})

# Different precision per symbol
precision_map = {"AAPL": 2, "EURUSD": 5}

# Automatically groups by symbol and applies correct precision
result = apply_precision(df, precision_map)
```

### List Columns (target_prices)

List columns are automatically handled element-wise:

```python
df = pl.DataFrame({
    "symbol": ["AAPL"],
    "target_prices": [[150.123456, 151.987654, 153.555555]],
})

result = apply_precision(df, {"AAPL": 2})
# result["target_prices"][0] → [150.12, 151.99, 153.56]
```

### Database Integration

Round before storing to ensure consistent precision:

```python
from thestrat import apply_precision

def store_indicators(df, precision_map, connection):
    """Store indicator data with proper precision."""
    # Round all fields
    rounded_df = apply_precision(df, precision_map)
    
    # Write to database
    rounded_df.write_database("indicators", connection)
```

### Backtesting Consistency

Ensure backtests match live trading precision:

```python
from thestrat import Factory, apply_precision

# Process indicators
pipeline = Factory.create_all(config)
indicators = pipeline["indicators"].process(data)

# Apply same precision as live trading
# (precision_map fetched from IBKR contract details)
rounded = apply_precision(indicators, precision_map)

# Now backtest results match live trading behavior
backtest_results = run_backtest(rounded)
```

## Error Handling

### Missing Symbol Precision

```python
from thestrat import apply_precision, PrecisionError

df = pl.DataFrame({
    "symbol": ["AAPL", "TSLA"],
    "open": [150.12, 250.34],
})

precision_map = {"AAPL": 2}  # Missing TSLA

try:
    apply_precision(df, precision_map)
except PrecisionError as e:
    print(e)  # "Missing precision for symbols: ['TSLA']"
```

### Invalid Field Name

```python
from thestrat import get_field_decimal_places, PrecisionError

try:
    get_field_decimal_places("invalid_field")
except PrecisionError as e:
    print(e)  # "Field 'invalid_field' not found in IndicatorSchema"
```

## Best Practices

1. **Fetch precision from IBKR**: Use contract details to get the correct minTick for each security
2. **Apply before storage**: Round data before writing to database to ensure consistency
3. **Apply before backtesting**: Use same precision in backtests as live trading
4. **Cache precision map**: Build precision map once and reuse across processing runs
5. **Validate symbols**: Ensure all symbols have precision before processing

## Integration Example

Complete example integrating precision utilities with strattrader:

```python
from thestrat import Factory, apply_precision, IndicatorSchema
from thestrat.schemas import FactoryConfig, AggregationConfig, IndicatorsConfig
import polars as pl

# 1. Fetch precision from IBKR
def get_ibkr_precision(symbols):
    """Get precision from IBKR contract details."""
    precision_map = {}
    for symbol in symbols:
        contract = ib.reqContractDetails(symbol)
        min_tick = contract.minTick  # e.g., 0.01, 0.00001, 0.00000001
        
        # Convert minTick to decimal places
        decimal_places = len(str(min_tick).split('.')[-1])
        precision_map[symbol] = decimal_places
    
    return precision_map

# 2. Process indicators
config = FactoryConfig(
    aggregation=AggregationConfig(target_timeframes=["5min"], asset_class="equities"),
    indicators=IndicatorsConfig(timeframe_configs=[...])
)

pipeline = Factory.create_all(config)
aggregated = pipeline["aggregation"].process(raw_data)
indicators = pipeline["indicators"].process(aggregated)

# 3. Get precision for all symbols
symbols = indicators["symbol"].unique().to_list()
precision_map = get_ibkr_precision(symbols)

# 4. Apply precision
rounded_indicators = apply_precision(indicators, precision_map)

# 5. Store with consistent precision
rounded_indicators.write_database("indicators", connection)
```

## See Also

- [DataFrame Schema](dataframe-schema.md) - Complete schema documentation
- [Examples](examples.md) - Working code examples
- API Reference - Detailed function signatures
