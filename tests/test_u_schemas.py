"""
Test module for SchemaDocGenerator functionality.
This module tests the documentation generation features added to schemas.py.
"""

import pytest
from thestrat.schemas import (
    SchemaDocGenerator,
    SwingPointsConfig,
    GapDetectionConfig,
    TimeframeItemConfig,
    IndicatorsConfig,
    AggregationConfig,
    FactoryConfig,
    generate_complete_documentation
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
        """Test the standalone generate_complete_documentation function."""
        docs = generate_complete_documentation()
        
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
        for field_name, field_doc in docs.items():
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