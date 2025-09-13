# DataFrame Schema Usage

The `IndicatorSchema` class provides comprehensive DataFrame validation and schema information for TheStrat processing pipeline. This is essential for database integration and data validation workflows.

## Quick Start

```python
from thestrat import IndicatorSchema
from polars import DataFrame
from datetime import datetime

# Validate input DataFrame
data = {
    "timestamp": [datetime.now()],
    "open": [100.0], "high": [105.0], "low": [95.0], "close": [102.0],
    "symbol": ["AAPL"], "volume": [1000000.0], "timeframe": ["5min"]
}

df = DataFrame(data, schema=IndicatorSchema.get_polars_dtypes())
result = IndicatorSchema.validate_dataframe(df)

print(f"Valid: {result['valid']}")
print(f"Missing columns: {result['missing_required']}")
```

## Database Schema Generation

### SQL Table Creation

```python
# Get column types and descriptions
descriptions = IndicatorSchema.get_column_descriptions()
polars_types = IndicatorSchema.get_polars_dtypes()

# Map Polars types to SQL types
from polars import Datetime, Float64, String, Boolean, Int32

type_mapping = {
    Datetime: "TIMESTAMP",
    Float64: "DOUBLE PRECISION",
    String: "VARCHAR(50)",
    Boolean: "BOOLEAN",
    Int32: "INTEGER"
}

# Generate CREATE TABLE statement
def generate_sql_schema(table_name: str) -> str:
    lines = [f"CREATE TABLE {table_name} ("]

    for col, polars_type in polars_types.items():
        sql_type = type_mapping.get(polars_type, "TEXT")
        description = descriptions.get(col, "").replace("'", "''")
        lines.append(f"  {col} {sql_type}, -- {description}")

    lines.append("  PRIMARY KEY (timestamp, symbol, timeframe)")
    lines.append(");")
    return "\n".join(lines)

schema_sql = generate_sql_schema("thestrat_indicators")
```

### Column Categories

Organize columns by functionality for targeted database operations:

```python
categories = IndicatorSchema.get_column_categories()

# Create separate tables by category
for category, columns in categories.items():
    if category == "base_ohlc":
        # Core market data table
        create_base_table(columns)
    elif category == "signals":
        # Trading signals table with indexes
        create_signals_table(columns)
    elif category == "swing_points":
        # Technical analysis table
        create_analysis_table(columns)
```

## Input Validation

### Required Columns Check

```python
def validate_input_data(df) -> dict:
    """Validate DataFrame before processing."""
    result = IndicatorSchema.validate_dataframe(df)

    if not result['valid']:
        errors = []
        if result['missing_required']:
            errors.append(f"Missing: {result['missing_required']}")
        if result['type_issues']:
            errors.append(f"Type errors: {result['type_issues']}")

        raise ValueError(f"Invalid DataFrame: {'; '.join(errors)}")

    return result['converted_df'] if result['conversion_performed'] else df
```

### Auto-conversion from Pandas

```python
from pandas import DataFrame as PandasDataFrame

# Pandas DataFrame automatically converts
df_pandas = PandasDataFrame(data)
df_pandas['timestamp'] = pd.to_datetime(df_pandas['timestamp'])

result = IndicatorSchema.validate_dataframe(df_pandas)
# result['converted_df'] contains Polars DataFrame
```

## Column Documentation

### Get Field Information

```python
# Column descriptions for documentation
descriptions = IndicatorSchema.get_column_descriptions()
polars_types = IndicatorSchema.get_polars_dtypes()

# Generate documentation
for col in ["swing_high", "continuity", "signal"]:
    print(f"**{col}**: {descriptions[col]}")
    print(f"Type: `{polars_types[col].__name__}`\n")
```

### Category-based Operations

```python
categories = IndicatorSchema.get_column_categories()

# Process only price analysis columns
price_cols = categories['price_analysis']
df_prices = df.select(price_cols)

# Extract signal columns for trading system
signal_cols = categories['signals']
df_signals = df.select(signal_cols)
```

## Integration Patterns

### Database Insert with Validation

```python
def insert_thestrat_data(df, connection):
    """Insert validated DataFrame into database."""
    # Validate first
    validated_df = validate_input_data(df)

    # Get column info for proper insertion
    polars_types = IndicatorSchema.get_polars_dtypes()

    # Insert with proper type handling
    for row in validated_df.iter_rows(named=True):
        insert_row(connection, row, polars_types)

def insert_row(conn, row_data, type_info):
    """Insert single row with type conversion."""
    columns = list(row_data.keys())
    placeholders = ", ".join(["?" for _ in columns])

    # Convert values based on schema
    from polars import Datetime
    values = []
    for col, value in row_data.items():
        if col in type_info and type_info[col] == Datetime:
            values.append(value.isoformat() if value else None)
        else:
            values.append(value)

    query = f"INSERT INTO thestrat_indicators ({', '.join(columns)}) VALUES ({placeholders})"
    conn.execute(query, values)
```

### API Response Validation

```python
from polars import DataFrame

def validate_api_response(json_data: list) -> DataFrame:
    """Convert and validate API data."""
    df = DataFrame(json_data)

    # Validate structure
    result = IndicatorSchema.validate_dataframe(df)
    if not result['valid']:
        raise ValueError(f"API data invalid: {result}")

    return result.get('converted_df', df)
```

## Best Practices

- **Always validate** input data before processing
- **Use column categories** to organize database tables efficiently
- **Leverage auto-conversion** for Pandas compatibility
- **Check type_issues** for data quality problems
- **Use descriptions** for database comments and API documentation
