# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Overview

**TheStrat** is a Python module for financial data aggregation and technical analysis using #TheStrat methodology. It provides vectorized operations using Polars for high-performance multi-timeframe analysis with comprehensive TheStrat indicators for crypto, equities, FX, and other asset classes.

## Development Environment Setup

### Prerequisites

This project uses **uv** for dependency management and virtual environment handling. You must activate the virtual environment before using any project commands.

```bash
# Install dependencies and create virtual environment
uv sync --extra test --extra dev --extra docs

# Activate virtual environment (REQUIRED before any other commands)
source .venv/bin/activate
```

### Development Installation

```bash
# Clone and setup for development
git clone https://github.com/jlixfeld/thestrat.git
cd thestrat
uv sync --extra test --extra dev --extra docs
source .venv/bin/activate
```

## Common Commands

### Testing

```bash
# Run all tests
uv run pytest

# Run with coverage (required >95%)
uv run pytest --cov=thestrat --cov-report=html --cov-report=term-missing

# Run only unit tests
uv run pytest -m unit

# Run only integration tests
uv run pytest -m integration

# Run specific test file
uv run pytest tests/test_u_factory.py -v

# Run tests with detailed output
uv run pytest -v --tb=short

# Quick test (pre-commit style)
uv run pytest --tb=short -x -q
```

### Code Quality

```bash
# Format code (required before commits)
uv run ruff format .

# Check linting
uv run ruff check .

# Auto-fix linting issues
uv run ruff check --fix .

# Check formatting without fixing
uv run ruff format --check .
```

### Documentation

```bash
# Serve documentation locally
uv run mkdocs serve

# Build documentation
uv run mkdocs build

# Documentation is auto-generated from docstrings using mkdocstrings
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
uv run pre-commit install

# Run hooks manually on all files
uv run pre-commit run --all-files

# Run only on staged files
uv run pre-commit run
```

### Building and Packaging

```bash
# Build the package
uv build

# Install in development mode
uv sync --extra dev
```

## Architecture Overview

### Core Components

The module follows a **Factory Pattern** with three main components:

1. **Factory** (`factory.py`) - Creates and configures components with Pydantic validation
2. **Aggregation** (`aggregation.py`) - Multi-timeframe OHLCV data aggregation
3. **Indicators** (`indicators.py`) - TheStrat technical indicators implementation

### Data Flow

```
Raw Market Data (Polars/Pandas DataFrame)
    ↓
Factory.create_all(FactoryConfig)
    ↓
Aggregation.process(data) → Multi-timeframe OHLCV
    ↓
Indicators.process(aggregated) → TheStrat indicators added
    ↓
Final DataFrame with timeframe column + indicators
```

### Configuration System

The module uses **Pydantic v2** for comprehensive validation:

- `FactoryConfig` - Complete pipeline configuration
- `AggregationConfig` - Timeframe and asset class settings
- `IndicatorsConfig` - Per-timeframe indicator configurations
- `AssetClassConfig` - Asset-specific timezone and session settings
- `TimeframeConfig` - Supported timeframe metadata and validation

### Asset Class Support

Built-in configurations for:
- **Crypto**: 24/7 UTC markets (`"crypto"`)
- **Equities**: US Eastern timezone with session times (`"equities"`)
- **FX**: 24/5 UTC markets (`"fx"`)

## Project Structure

```
thestrat/
├── thestrat/                    # Core module
│   ├── __init__.py             # Public API exports
│   ├── factory.py              # Factory pattern for component creation
│   ├── base.py                 # Abstract base classes
│   ├── aggregation.py          # Multi-timeframe aggregation
│   ├── indicators.py           # TheStrat indicators implementation
│   ├── schemas.py              # Pydantic validation models
│   └── signals.py              # Signal definitions and metadata
├── tests/                      # Test suite
│   ├── test_u_*.py            # Unit tests
│   ├── test_i_*.py            # Integration tests
│   └── utils/                 # Test utilities and fixtures
├── docs/                      # Documentation source
└── site/                      # Generated documentation
```

### Key Configuration Files

- `pyproject.toml` - Project metadata, dependencies, tool configuration
- `.pre-commit-config.yaml` - Code quality hooks (ruff, pytest)
- `mkdocs.yml` - Documentation configuration
- `uv.lock` - Locked dependency versions

## Testing Strategy

### Test Categories

Tests are organized with pytest markers:

```python
@pytest.mark.unit        # Fast, isolated unit tests
@pytest.mark.integration # End-to-end workflow tests
```

### Test Naming Convention

Tests should be named `test_i_*.py` for integration tests and `test_u_*.py` for unit tests, where `*` is the filename of the codebase being tested.

**Examples:**
- `test_u_factory.py` - Unit tests for `factory.py`
- `test_u_indicators.py` - Unit tests for `indicators.py`
- `test_u_schemas.py` - Unit tests for `schemas.py`
- `test_i_thestrat.py` - Integration tests for the overall `thestrat` module

**Incorrect naming:**
- `test_u_processor_validator.py` (describes functionality, not the file being tested)
- `test_schema_validation.py` (missing u/i prefix and doesn't match source file)

### Coverage Requirements

- **Minimum coverage**: 95%
- Coverage reports generated in `htmlcov/`
- All public APIs must have comprehensive tests

### Test Data

The test suite includes utilities for creating realistic market data:

- `tests/utils/thestrat_data_utils.py` - Market data generators
- `tests/utils/config_helpers.py` - Configuration builders
- Fixtures for different asset classes (equities, crypto, forex)

### Running Specific Tests

```bash
# Test specific functionality
uv run pytest -k "aggregation" -v

# Test with pattern matching
uv run pytest tests/ -k "test_factory" --tb=short

# Test integration workflows only
uv run pytest -m integration --tb=short
```

## TheStrat Methodology Context

### Core Concepts

**TheStrat** is a price action trading methodology focused on:

- **Multi-timeframe Analysis**: Simultaneous analysis across timeframes
- **Market Structure**: Identifying swing points, continuity, and scenarios
- **Signal Generation**: Actionable trading signals based on price patterns
- **Gap Analysis**: Identifying and categorizing price gaps

### Indicator Implementation

Key indicators provided by this module:

- **Swing Points**: `swing_high`, `swing_low` with configurable windows
- **Market Structure**: `higher_high`, `lower_high`, `higher_low`, `lower_low`
- **Continuity Analysis**: `continuity`, `in_force` status tracking
- **Signal Detection**: `scenario`, `signal` classifications
- **Extremes**: `ath` (all-time high), `atl` (all-time low)
- **Gap Detection**: `gapper` identification

### Usage Patterns

```python
from thestrat import Factory
from thestrat.schemas import FactoryConfig, AggregationConfig, IndicatorsConfig

# Create multi-timeframe pipeline
config = FactoryConfig(
    aggregation=AggregationConfig(
        target_timeframes=["5min", "15min", "1h"],
        asset_class="equities"
    ),
    indicators=IndicatorsConfig(
        timeframe_configs=[
            TimeframeItemConfig(
                timeframes=["all"],
                swing_points=SwingPointsConfig(window=5, threshold=2.0)
            )
        ]
    )
)

pipeline = Factory.create_all(config)
```

## Development Workflow

### Before Starting Work

```bash
# Always activate virtual environment first
source .venv/bin/activate

# Pull latest changes
git pull origin main

# Update dependencies
uv sync --extra test --extra dev
```

### During Development

```bash
# Run tests frequently
uv run pytest -x --tb=short

# Check code quality
uv run ruff check . --fix
uv run ruff format .

# Verify changes don't break integration
uv run pytest -m integration
```

### Before Committing

```bash
# Run full test suite with coverage
uv run pytest --cov=thestrat --cov-report=term-missing

# Ensure all quality checks pass
uv run ruff check .
uv run ruff format --check .

# Pre-commit hooks will run automatically
git commit -m "Your commit message"
```

### Code Standards

- **Type Hints**: Required for all public APIs
- **Docstrings**: Google-style docstrings for all public methods
- **Line Length**: 120 characters maximum
- **Import Style**: Use double quotes, organized by ruff
- **DataFrame Imports**: Never import `pandas as pd` or `polars as pl`. Always import required classes directly:
  - ✅ `from polars import DataFrame, Series`
  - ✅ `from pandas import DataFrame as PandasDataFrame`
  - ❌ `import polars as pl`
  - ❌ `import pandas as pd`
- **Testing**: >95% coverage required for all new code

## Performance Considerations

### Polars Integration

- **Primary DataFrame Library**: Polars for high-performance operations
- **Pandas Compatibility**: Automatic conversion from pandas input
- **Memory Efficiency**: Lazy evaluation and columnar operations
- **Type Safety**: Strong typing with Polars schema validation

### Optimization Notes

- Use vectorized operations over loops
- Leverage Polars expressions for complex calculations
- Minimize DataFrame conversions between pandas/Polars
- Use appropriate timeframe aggregation to reduce data volume

## Troubleshooting

### Common Issues

1. **Virtual Environment**: Always activate before running commands
2. **Test Failures**: Check that test data fixtures match expected schemas
3. **Import Errors**: Ensure development installation with `uv sync --extra dev`
4. **Coverage Issues**: Use `--cov-report=html` to identify uncovered lines

### Debug Commands

```bash
# Verbose test output with full tracebacks
uv run pytest -vvs --tb=long

# Run single test with debugging
uv run pytest tests/test_u_factory.py::test_specific_function -vvs

# Check installed packages
uv pip list
```
