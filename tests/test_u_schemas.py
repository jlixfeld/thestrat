"""
Test module for SchemaDocGenerator functionality.
This module tests the documentation generation features added to schemas.py.
"""

from thestrat.schemas import (
    AggregationConfig,
    FactoryConfig,
    IndicatorSchema,
    SchemaDocGenerator,
    SwingPointsConfig,
)


class TestSchemaDocGenerator:
    """Test the SchemaDocGenerator class."""

    def test_generate_field_docs_basic_model(self):
        """Test generate_field_docs with a basic model."""
        docs = SchemaDocGenerator.generate_field_docs(SwingPointsConfig)

        assert isinstance(docs, dict)
        assert "window" in docs
        assert "threshold" in docs

        # Check field structure
        window_doc = docs["window"]
        assert window_doc["name"] == "window"
        assert "int" in window_doc["type"]
        assert isinstance(window_doc["required"], bool)

    def test_generate_field_docs_with_constraints(self):
        """Test generate_field_docs extracts constraints properly."""
        docs = SchemaDocGenerator.generate_field_docs(SwingPointsConfig)

        # SwingPointsConfig has constraints on window (ge=3) and threshold (ge=0)
        window_doc = docs["window"]
        threshold_doc = docs["threshold"]

        # Should have constraint information if available
        assert "constraints" in window_doc or "ge" in str(window_doc)
        assert "constraints" in threshold_doc or "ge" in str(threshold_doc)

    def test_generate_markdown_table_basic(self):
        """Test generate_markdown_table creates proper markdown."""
        table = SchemaDocGenerator.generate_markdown_table(SwingPointsConfig)

        # Should be a string with markdown table format
        assert isinstance(table, str)
        assert "| Field" in table
        assert "| Type" in table
        assert "| Default" in table
        assert "| Description" in table
        assert "|---" in table or "|-" in table  # Table separator
        assert "window" in table
        assert "threshold" in table

    def test_generate_json_schema_basic(self):
        """Test generate_json_schema creates JSON schema."""
        schema = SchemaDocGenerator.generate_json_schema(SwingPointsConfig)

        # Should be a dictionary with JSON schema structure
        assert isinstance(schema, dict)
        assert "properties" in schema
        assert "window" in schema["properties"]
        assert "threshold" in schema["properties"]

    def test_get_type_string_basic_types(self):
        """Test _get_type_string with basic Python types."""
        # Test basic types
        assert SchemaDocGenerator._get_type_string(int) == "int"
        assert SchemaDocGenerator._get_type_string(float) == "float"
        assert SchemaDocGenerator._get_type_string(str) == "str"
        assert SchemaDocGenerator._get_type_string(bool) == "bool"

    def test_get_type_string_optional_types(self):
        """Test _get_type_string with Optional types."""
        from typing import Optional

        # Test Optional types
        result = SchemaDocGenerator._get_type_string(Optional[str])
        assert "str" in result and ("None" in result or "optional" in result.lower())

    def test_get_type_string_list_types(self):
        """Test _get_type_string with List types."""
        from typing import List

        # Test List types
        result = SchemaDocGenerator._get_type_string(List[str])
        assert "list" in result.lower() or "str" in result

    def test_format_default_values(self):
        """Test _format_default handles different default value types."""
        # Test various default types
        assert SchemaDocGenerator._format_default(None) == "`None`"
        assert SchemaDocGenerator._format_default(5) == "`5`"
        assert SchemaDocGenerator._format_default(5.0) == "`5.0`"
        assert SchemaDocGenerator._format_default("test") == '`"test"`'
        assert SchemaDocGenerator._format_default(True) == "`True`"
        assert SchemaDocGenerator._format_default(...) == "*required*"

    def test_extract_constraints_numeric(self):
        """Test _extract_constraints with numeric constraints."""
        from pydantic import Field

        # Create a field with constraints for testing
        field_info = Field(ge=0, le=100)
        constraints = SchemaDocGenerator._extract_constraints(field_info)

        # Should extract constraints
        assert isinstance(constraints, dict)

    def test_generate_all_model_docs_function(self):
        """Test the generate_all_model_docs function."""
        docs = SchemaDocGenerator.generate_all_model_docs()

        assert isinstance(docs, dict)
        assert len(docs) > 0
        # Should contain our configuration models
        model_names = [name for name in docs.keys()]
        assert any("Config" in name for name in model_names)

    def test_generate_complete_documentation_function(self):
        """Test the SchemaDocGenerator.generate_complete_documentation class method."""
        docs = SchemaDocGenerator.generate_complete_documentation()

        assert isinstance(docs, str)
        assert len(docs) > 500  # Should be comprehensive
        assert "Config" in docs  # Should contain our models


class TestSchemaDocHelpers:
    """Test SchemaDocGenerator helper methods and edge cases."""

    def test_get_type_string_union_types(self):
        """Test _get_type_string with Union types."""
        from typing import Union

        result = SchemaDocGenerator._get_type_string(Union[str, int])
        assert ("str" in result and "int" in result) or "|" in result

    def test_get_type_string_complex_types(self):
        """Test _get_type_string with complex nested types."""
        from typing import Dict, List

        # Test nested types
        result = SchemaDocGenerator._get_type_string(Dict[str, List[int]])
        assert "dict" in result.lower() or "str" in result

    def test_format_default_edge_cases(self):
        """Test _format_default with edge cases."""
        # Test edge cases
        assert SchemaDocGenerator._format_default([]) == "`[]`"
        assert SchemaDocGenerator._format_default({}) == "`{}`"
        assert SchemaDocGenerator._format_default(False) == "`False`"
        assert SchemaDocGenerator._format_default(0) == "`0`"

    def test_extract_constraints_string_patterns(self):
        """Test _extract_constraints with string pattern constraints."""
        from pydantic import Field

        # Create a field with string constraints
        field_info = Field(min_length=1, max_length=50, pattern=r"^[A-Z]")
        constraints = SchemaDocGenerator._extract_constraints(field_info)

        # Should extract string constraints
        assert isinstance(constraints, dict)

    def test_documentation_generation_with_metadata(self):
        """Test documentation generation includes model metadata."""
        docs = SchemaDocGenerator.generate_field_docs(AggregationConfig)

        # Should include all required fields
        assert "target_timeframes" in docs
        assert "asset_class" in docs

        # Check metadata is preserved
        for _field_name, field_doc in docs.items():
            assert "name" in field_doc
            assert "type" in field_doc
            assert "description" in field_doc

    def test_markdown_table_formatting(self):
        """Test markdown table has proper formatting."""
        table = SchemaDocGenerator.generate_markdown_table(AggregationConfig)

        # Check table structure
        lines = table.split("\n")
        header_found = any("Field" in line for line in lines)
        assert header_found

    def test_json_schema_structure(self):
        """Test JSON schema has proper structure."""
        schema = SchemaDocGenerator.generate_json_schema(FactoryConfig)

        assert isinstance(schema, dict)
        assert "properties" in schema

        # Should include nested model schemas
        properties = schema["properties"]
        assert len(properties) > 0


class TestIndicatorSchemaClassMethods:
    """Test IndicatorSchema class methods."""

    def test_get_column_descriptions(self):
        """Test get_column_descriptions returns proper descriptions."""
        descriptions = IndicatorSchema.get_column_descriptions()

        assert isinstance(descriptions, dict)
        assert len(descriptions) > 0

        # Check some known descriptions
        assert "timestamp" in descriptions
        assert "open" in descriptions
        assert "swing_high" in descriptions
        assert isinstance(descriptions["timestamp"], str)

    def test_get_polars_dtypes(self):
        """Test get_polars_dtypes returns Polars type mappings."""
        polars_types = IndicatorSchema.get_polars_dtypes()

        assert isinstance(polars_types, dict)
        assert len(polars_types) > 0

        # Check some expected mappings
        from polars import Boolean, Datetime, Float64, String

        assert "timestamp" in polars_types
        assert polars_types["timestamp"] == Datetime
        assert polars_types["open"] == Float64
        assert polars_types["gapper"] == Boolean
        assert polars_types["signal"] == String

    def test_validate_dataframe(self):
        """Test DataFrame column validation function."""
        from datetime import datetime

        import polars as pl

        # Create a simple DataFrame with required input columns
        data = {
            "timestamp": [datetime.now()],
            "open": [100.0],
            "high": [105.0],
            "low": [95.0],
            "close": [102.0],
            "symbol": ["AAPL"],
            "volume": [1000.0],
            "timeframe": ["5min"],
        }
        df = pl.DataFrame(
            data,
            schema={
                "timestamp": pl.Datetime,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "symbol": pl.String,
                "volume": pl.Float64,
                "timeframe": pl.String,
            },
        )

        result = IndicatorSchema.validate_dataframe(df)

        assert result["valid"] is True
        assert len(result["missing_required"]) == 0
        assert "timestamp" in result["required_fields"]
        assert result["df_type"] == "polars"

    def test_get_column_categories(self):
        """Test get_column_categories returns proper categorization."""
        categories = IndicatorSchema.get_column_categories()

        assert isinstance(categories, dict)
        assert len(categories) > 0

        # Check expected categories exist
        expected_categories = [
            "base_ohlc",
            "price_analysis",
            "gap_detection",
            "swing_points",
            "thestrat_patterns",
            "signals",
            "special_patterns",
            "mother_bar",
        ]

        for category in expected_categories:
            assert category in categories
            assert isinstance(categories[category], list)
            assert len(categories[category]) > 0

        # Check some expected column placements
        assert "timestamp" in categories["base_ohlc"]
        assert "swing_high" in categories["swing_points"]
        assert "continuity" in categories["thestrat_patterns"]
        assert "signal" in categories["signals"]
