# Database Integration Guide

The `IndicatorSchema` class provides essential schema information for integrating TheStrat output with databases and validation systems. This guide shows how to use the schema to create database tables, validate data, and ensure consistent integration.

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
    elif category == "market_structure":
        # Market structure analysis table
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

### Advanced SQL Schema Generation

Generate complete SQL DDL with nullable constraints by examining the schema metadata:

```python
from thestrat.schemas import IndicatorSchema

# Get schema information
descriptions = IndicatorSchema.get_column_descriptions()
polars_types = IndicatorSchema.get_polars_dtypes()

def generate_sql_with_constraints(table_name: str) -> str:
    """Generate SQL schema with proper NULL/NOT NULL constraints."""
    lines = [f"CREATE TABLE {table_name} ("]

    # Map Polars types to SQL types
    type_mapping = {
        "Datetime": "TIMESTAMP",
        "Float64": "DOUBLE PRECISION",
        "String": "VARCHAR(255)",
        "Boolean": "BOOLEAN",
        "Int32": "INTEGER"
    }

    # Process each field using the new helper method
    for field_name in IndicatorSchema.model_fields.keys():
        # Get SQL type from Polars type
        polars_type = polars_types.get(field_name)
        polars_type_name = polars_type.__name__ if polars_type and hasattr(polars_type, '__name__') else "String"
        sql_type = type_mapping.get(polars_type_name, "TEXT")

        # Get nullable constraint using helper method
        metadata = IndicatorSchema.get_field_metadata(field_name)
        nullable = metadata.get('nullable', True)
        constraint = "" if nullable else " NOT NULL"

        # Add description as comment
        description = descriptions.get(field_name, "").replace("'", "''")
        lines.append(f"  {field_name} {sql_type}{constraint}, -- {description}")

    lines.append("  PRIMARY KEY (timestamp, symbol, timeframe)")
    lines.append(");")
    return "\n".join(lines)

schema_sql = generate_sql_with_constraints("thestrat_indicators")
print(schema_sql)
```

### Field Classification by Type

Organize columns by input/output type and nullable constraints:

```python
# Classify fields by their purpose using the helper method
input_fields = []
output_fields = []
nullable_fields = []
required_fields = []

for field_name in IndicatorSchema.model_fields.keys():
    metadata = IndicatorSchema.get_field_metadata(field_name)

    # Classify by input/output
    if metadata.get('input', False):
        input_fields.append(field_name)
    if metadata.get('output', False):
        output_fields.append(field_name)

    # Classify by nullable constraint
    if metadata.get('nullable', True):
        nullable_fields.append(field_name)
    else:
        required_fields.append(field_name)

print(f"Input fields ({len(input_fields)}): {input_fields}")
print(f"Output fields ({len(output_fields)}): {output_fields}")
print(f"Nullable fields ({len(nullable_fields)}): {nullable_fields}")
print(f"Required (non-null) fields ({len(required_fields)}): {required_fields}")
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
