"""Generate the API reference navigation for mkdocstrings."""

import sys
from pathlib import Path

import mkdocs_gen_files

# Add the parent directory to the path so we can import thestrat
sys.path.insert(0, str(Path(__file__).parent.parent))

# Define the source code structure
nav = mkdocs_gen_files.Nav()

# TheStrat module structure based on your codebase
modules = {
    "thestrat": "TheStrat package root",
    "thestrat.aggregation": "OHLCV timeframe aggregation",
    "thestrat.indicators": "TheStrat technical indicators",
    "thestrat.signals": "Signal processing and metadata",
    "thestrat.factory": "Component creation with factory pattern",
    "thestrat.schemas": "Pydantic configuration schemas",
    "thestrat.base": "Abstract base classes",
}

# Generate reference pages for each module
for module_name, description in modules.items():
    # Create the file path
    doc_path = Path("reference") / f"{module_name.replace('.', '/')}.md"

    # Add to navigation
    nav[module_name.split(".")] = f"{module_name.replace('.', '/')}.md"

    # Generate the content
    with mkdocs_gen_files.open(doc_path, "w") as f:
        # Module title and description
        module_display = module_name.split(".")[-1] if "." in module_name else module_name
        f.write(f"# {module_display.title()}\n\n")
        f.write(f"{description}\n\n")

        # mkdocstrings directive
        f.write(f"::: {module_name}\n")
        if module_name != "thestrat":
            f.write("    options:\n")
            f.write("      show_root_heading: false\n")
            f.write("      show_root_full_path: false\n")

# Create the main reference index
with mkdocs_gen_files.open("reference/index.md", "w") as f:
    f.write("# API Reference\n\n")
    f.write("Complete API documentation for all TheStrat components.\n\n")

    f.write("## Core Modules\n\n")
    f.write("| Module | Description |\n")
    f.write("|--------|-------------|\n")
    for module_name, description in modules.items():
        if module_name != "thestrat":
            module_link = f"[{module_name.split('.')[-1]}]({module_name.replace('.', '/')}.md)"
            f.write(f"| {module_link} | {description} |\n")

    f.write("\n## Quick Navigation\n\n")
    f.write("- **[Factory](thestrat/factory.md)** - Start here for component creation\n")
    f.write("- **[Schemas](thestrat/schemas.md)** - Configuration models and validation\n")
    f.write("- **[Aggregation](thestrat/aggregation.md)** - Timeframe data processing\n")
    f.write("- **[Indicators](thestrat/indicators.md)** - TheStrat analysis functions\n")
    f.write("- **[Base](thestrat/base.md)** - Abstract base classes\n")

# Write the navigation summary
with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())

# Generate schema documentation
try:
    from thestrat.schemas import generate_complete_documentation

    # Create a comprehensive configuration documentation page
    config_template = """# Configuration Reference

TheStrat uses Pydantic schemas for configuration validation, providing type safety, detailed error reporting, and comprehensive documentation.

## Overview

Configuration in TheStrat follows a hierarchical structure:

- **FactoryConfig** - Root configuration for complete pipeline setup
- **AggregationConfig** - OHLCV timeframe aggregation settings
- **IndicatorsConfig** - TheStrat indicator configurations
- **AssetClassConfig** - Market-specific settings (crypto, equities, FX)

## Quick Start

```python
from thestrat import Factory

# Minimal configuration
config = {
    "aggregation": {
        "target_timeframes": ["5m", "1h"],
        "asset_class": "equities"
    },
    "indicators": {
        "timeframe_configs": [{
            "timeframes": ["all"],
            "swing_points": {"window": 5, "threshold": 2.0}
        }]
    }
}

pipeline = Factory.create_all(config)
```

## Schema Documentation

"""

    schema_docs = generate_complete_documentation()
    full_config_docs = config_template + schema_docs

    with mkdocs_gen_files.open("user-guide/configuration.md", "w") as f:
        f.write(full_config_docs)

except ImportError:
    # Fallback if import fails during build
    with mkdocs_gen_files.open("user-guide/configuration.md", "w") as f:
        f.write("# Configuration Reference\n\nConfiguration documentation will be generated during build.\n")
