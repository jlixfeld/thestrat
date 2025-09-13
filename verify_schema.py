#!/usr/bin/env python3
"""
Schema Verification Script for TheStrat Indicators

This script verifies that the IndicatorSchema matches the actual output columns
from the indicators processing pipeline. It creates sample data, processes it
through the full pipeline, and compares the output columns with schema fields.
"""

import sys
from datetime import datetime, timedelta
from typing import Dict, List, Set

import polars as pl

# Import the classes we need to test
from thestrat.indicators import Indicators
from thestrat.schemas import (
    GapDetectionConfig,
    IndicatorSchema,
    IndicatorsConfig,
    SwingPointsConfig,
    TimeframeItemConfig,
)


def create_sample_market_data(num_rows: int = 50) -> pl.DataFrame:
    """Create realistic sample market data for testing."""
    import random

    random.seed(42)  # For reproducible results

    timestamps = [datetime(2023, 1, 1) + timedelta(minutes=5 * i) for i in range(num_rows)]

    # Generate realistic OHLC data with some volatility
    base_price = 100.0
    data = []

    for i, timestamp in enumerate(timestamps):
        # Add some trend and volatility
        trend = i * 0.1
        volatility = random.uniform(-2, 2)

        open_price = base_price + trend + volatility

        # Generate high/low around open with some range
        range_size = random.uniform(0.5, 3.0)
        high = open_price + random.uniform(0, range_size)
        low = open_price - random.uniform(0, range_size)

        # Close somewhere in the range
        close = random.uniform(low, high)

        # Volume
        volume = random.uniform(1000, 5000)

        data.append(
            {
                "timestamp": timestamp,
                "open": round(open_price, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(close, 2),
                "volume": round(volume, 0),
                "symbol": "TEST",
            }
        )

    return pl.DataFrame(data)


def get_schema_fields() -> Dict[str, str]:
    """Extract all field names from IndicatorSchema."""
    schema_fields = {}

    for field_name, field_info in IndicatorSchema.model_fields.items():
        json_extra = getattr(field_info, "json_schema_extra", {}) or {}
        is_input = json_extra.get("input", False)
        is_output = json_extra.get("output", False)

        # Categorize fields
        if is_input and is_output:
            category = "input_output"
        elif is_input:
            category = "input_only"
        elif is_output:
            category = "output_only"
        else:
            category = "unknown"

        schema_fields[field_name] = category

    return schema_fields


def analyze_column_mismatch(actual_columns: Set[str], schema_fields: Dict[str, str]) -> Dict[str, List[str]]:
    """Analyze mismatches between actual columns and schema fields."""
    schema_column_names = set(schema_fields.keys())

    # Find differences
    missing_from_actual = schema_column_names - actual_columns
    extra_in_actual = actual_columns - schema_column_names
    matching_columns = actual_columns & schema_column_names

    # Categorize missing columns
    missing_input_only = []
    missing_output_only = []
    missing_input_output = []
    missing_unknown = []

    for col in missing_from_actual:
        category = schema_fields[col]
        if category == "input_only":
            missing_input_only.append(col)
        elif category == "output_only":
            missing_output_only.append(col)
        elif category == "input_output":
            missing_input_output.append(col)
        else:
            missing_unknown.append(col)

    return {
        "matching_columns": sorted(matching_columns),
        "missing_from_actual": sorted(missing_from_actual),
        "extra_in_actual": sorted(extra_in_actual),
        "missing_input_only": sorted(missing_input_only),
        "missing_output_only": sorted(missing_output_only),
        "missing_input_output": sorted(missing_input_output),
        "missing_unknown": sorted(missing_unknown),
    }


def main():
    """Main verification function."""
    print("=" * 80)
    print("TheStrat IndicatorSchema Verification Report")
    print("=" * 80)

    # Step 1: Create sample data
    print("\n1. Creating sample market data...")
    sample_data = create_sample_market_data(50)
    print(f"   Created {len(sample_data)} rows of sample OHLC data")
    print(f"   Input columns: {sorted(sample_data.columns)}")

    # Step 2: Process through indicators pipeline
    print("\n2. Processing through indicators pipeline...")
    try:
        indicators = Indicators(
            IndicatorsConfig(
                timeframe_configs=[
                    TimeframeItemConfig(
                        timeframes=["all"],
                        swing_points=SwingPointsConfig(window=5, threshold=5.0),
                        gap_detection=GapDetectionConfig(threshold=0.001),
                    )
                ]
            )
        )

        result_df = indicators.process(sample_data)
        actual_columns = set(result_df.columns)

        print("   Processing completed successfully")
        print(f"   Output DataFrame shape: {result_df.shape}")
        print(f"   Output columns ({len(actual_columns)}): {sorted(actual_columns)}")

    except Exception as e:
        print(f"   ERROR: Processing failed: {e}")
        sys.exit(1)

    # Step 3: Get schema fields
    print("\n3. Analyzing IndicatorSchema fields...")
    schema_fields = get_schema_fields()
    print(f"   Schema defines {len(schema_fields)} fields")

    # Categorize schema fields
    input_only_fields = [f for f, c in schema_fields.items() if c == "input_only"]
    output_only_fields = [f for f, c in schema_fields.items() if c == "output_only"]
    input_output_fields = [f for f, c in schema_fields.items() if c == "input_output"]
    unknown_fields = [f for f, c in schema_fields.items() if c == "unknown"]

    print(f"   - Input only fields ({len(input_only_fields)}): {sorted(input_only_fields)}")
    print(f"   - Output only fields ({len(output_only_fields)}): {sorted(output_only_fields)}")
    print(f"   - Input+Output fields ({len(input_output_fields)}): {sorted(input_output_fields)}")
    if unknown_fields:
        print(f"   - Unknown category ({len(unknown_fields)}): {sorted(unknown_fields)}")

    # Step 4: Compare and analyze mismatches
    print("\n4. Comparing actual output vs IndicatorSchema...")
    analysis = analyze_column_mismatch(actual_columns, schema_fields)

    print(f"\n   MATCHING COLUMNS ({len(analysis['matching_columns'])}):")
    for col in analysis["matching_columns"]:
        print(f"   ✓ {col} ({schema_fields[col]})")

    if analysis["extra_in_actual"]:
        print(f"\n   ❌ COLUMNS IN OUTPUT BUT MISSING FROM SCHEMA ({len(analysis['extra_in_actual'])}):")
        for col in analysis["extra_in_actual"]:
            print(f"   + {col}")

    if analysis["missing_from_actual"]:
        print(f"\n   ❌ COLUMNS IN SCHEMA BUT MISSING FROM OUTPUT ({len(analysis['missing_from_actual'])}):")

        if analysis["missing_input_only"]:
            print("\n      Input-only columns (expected to be missing):")
            for col in analysis["missing_input_only"]:
                print(f"      - {col}")

        if analysis["missing_output_only"]:
            print("\n      ⚠️  Output-only columns (PROBLEM - these should be in output):")
            for col in analysis["missing_output_only"]:
                print(f"      ! {col}")

        if analysis["missing_input_output"]:
            print("\n      ⚠️  Input+Output columns (PROBLEM - these should be in output):")
            for col in analysis["missing_input_output"]:
                print(f"      ! {col}")

        if analysis["missing_unknown"]:
            print("\n      Unknown category columns:")
            for col in analysis["missing_unknown"]:
                print(f"      ? {col}")

    # Step 5: Specific focus on the reported issue
    print("\n5. Specific Analysis - Mother Bar Columns:")
    motherbar_columns = ["is_mother_bar", "active_mother_high", "active_mother_low", "motherbar_problems"]

    for col in motherbar_columns:
        if col in actual_columns:
            print(f"   ✓ {col} - PRESENT in output")
        else:
            print(f"   ❌ {col} - MISSING from output")

        if col in schema_fields:
            print(f"     ✓ {col} - PRESENT in schema ({schema_fields[col]})")
        else:
            print(f"     ❌ {col} - MISSING from schema")

    # Step 6: Summary and recommendations
    print("\n6. SUMMARY:")
    total_issues = (
        len(analysis["extra_in_actual"]) + len(analysis["missing_output_only"]) + len(analysis["missing_input_output"])
    )

    if total_issues == 0:
        print("   ✅ SCHEMA IS CORRECTLY ALIGNED WITH OUTPUT")
    else:
        print(f"   ❌ FOUND {total_issues} SCHEMA ALIGNMENT ISSUES")

        print("\n   REQUIRED ACTIONS:")
        if analysis["missing_output_only"] or analysis["missing_input_output"]:
            print("   1. Remove columns from IndicatorSchema that are not in output:")
            for col in analysis["missing_output_only"] + analysis["missing_input_output"]:
                print(f"      - Remove: {col}")

        if analysis["extra_in_actual"]:
            print("   2. Add columns to IndicatorSchema that are in output:")
            for col in analysis["extra_in_actual"]:
                print(f"      - Add: {col}")

    print("\n" + "=" * 80)

    return total_issues == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
