# Installation

This guide covers how to install the TheStrat module in different environments and scenarios.

## Prerequisites

Before installing TheStrat, ensure you have:

- **Python 3.10 or higher**
- **uv** package manager (recommended) or **pip**
- **Git** (for development installation)

### Installing uv (Recommended)

If you don't have `uv` installed, it's the fastest Python package installer:

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install uv
```

## Installation Options

### Option 1: Direct Installation (Recommended)

Install directly from the GitHub repository:

```bash title="Install TheStrat"
uv add git+https://github.com/jlixfeld/thestrat.git
```

### Option 2: Development Installation

For development work or to run tests:

```bash title="Development Setup"
# Clone the repository
git clone https://github.com/jlixfeld/thestrat.git
cd thestrat

# Install with development dependencies
uv sync --extra test --extra dev --extra docs
```

### Option 3: Using pip

If you prefer using pip:

```bash
pip install git+https://github.com/jlixfeld/thestrat.git
```

## Verify Installation

Test your installation by importing the module:

```python title="Verify Installation"
import thestrat
print(f"TheStrat version: {thestrat.__version__}")

# Test basic functionality with Pydantic models
from thestrat import Factory
from thestrat.schemas import (
    FactoryConfig, AggregationConfig, IndicatorsConfig,
    TimeframeItemConfig, SwingPointsConfig
)

config = FactoryConfig(
    aggregation=AggregationConfig(target_timeframes=["5m"]),
    indicators=IndicatorsConfig(
        timeframe_configs=[
            TimeframeItemConfig(
                timeframes=["all"],
                swing_points=SwingPointsConfig(window=5, threshold=2.0)
            )
        ]
    )
)

components = Factory.create_all(config)
print("Installation successful!")
```

## Development Setup

If you're planning to contribute or modify the code:

### 1. Clone and Install

```bash
git clone https://github.com/jlixfeld/thestrat.git
cd thestrat
uv sync --extra test --extra dev --extra docs
```

### 2. Verify Development Environment

```bash
# Run tests
uv run pytest

# Check code formatting
uv run ruff check .

# Format code
uv run ruff format .

# Build documentation
uv run mkdocs serve
```

### 3. Run Development Tests

Verify your development environment:

```bash
# Run all tests
uv run pytest

# Check code quality
uv run ruff check .
```

## Dependencies

TheStrat has the following dependencies:

### Core Dependencies
- **polars[timezone]** ≥1.0.0 - High-performance data processing
- **pandas** ≥1.5.0 - Data manipulation and analysis
- **numpy** ≥1.21.0 - Numerical computing
- **pytz** ≥2022.1 - Timezone handling

### Development Dependencies
- **pytest** ≥6.0 - Testing framework
- **ruff** ==0.11.13 - Linting and formatting
- **pytest-cov** ≥2.0 - Coverage reporting

### Documentation Dependencies
- **mkdocs-material** ≥9.4.0 - Documentation theme
- **mkdocstrings[python]** ≥0.24.0 - API documentation generation

## Troubleshooting

### Common Issues

**Import Error**: `ModuleNotFoundError: No module named 'thestrat'`
:   Ensure you've activated the correct Python environment and the module is installed.

**Version Conflicts**: Dependency resolution errors
:   Use `uv` which has better dependency resolution than pip:
    ```bash
    uv add git+https://github.com/jlixfeld/thestrat.git
    ```

**Permission Errors**: Cannot write to installation directory  
:   Use a virtual environment or user installation:
    ```bash
    # Create virtual environment
    uv venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    uv add git+https://github.com/jlixfeld/thestrat.git
    ```

**Test Failures**: Tests failing during development setup
:   Ensure you have the test dependencies:
    ```bash
    uv sync --extra test
    uv run pytest -v
    ```

### Getting Help

If you encounter issues not covered here:

1. Check that all prerequisites are installed
2. Verify your Python version: `python --version`
3. Try creating a fresh virtual environment
4. Contact the maintainer with error details

## Next Steps

Once installation is complete, proceed to the [Quick Start](quickstart.md) guide to begin using TheStrat in your applications.