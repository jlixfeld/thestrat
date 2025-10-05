# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TheStrat is a Python financial analysis library implementing Rob Smith's "The Strat" trading methodology. The library provides vectorized technical analysis using Polars for high-performance multi-timeframe aggregation and pattern recognition.

**Core Architecture:**
- **Factory Pattern**: All components created via `Factory` class with Pydantic validation
- **Component-Based**: All processors inherit from `Component` abstract base class
- **Schema-First**: Comprehensive Pydantic schemas drive validation and configuration
- **High Performance**: Polars DataFrames for vectorized operations, fully optimized
- **Type Safety**: Full type hints and Pydantic validation throughout

## Performance Optimizations

### Recent Improvements (2025)

**Fully Vectorized Market Structure Detection:**
- ✅ Eliminated all for loops from market structure calculations
- ✅ Uses centered rolling windows for accurate peak/valley detection
- ✅ Vectorized percentage threshold filtering
- ✅ Direct HH/LH/HL/LL classification without intermediate swing points
- ✅ Performance: ~35,000 rows/second for complete indicator analysis

**Vectorized Market Structure Classification:**
- ✅ Shift-based comparisons for HH/HL/LH/LL patterns
- ✅ Forward-fill operations for state tracking
- ✅ Mutually exclusive pattern detection

**Optimized Signal Processing:**
- ✅ Cascaded pattern matching in single operation
- ✅ On-demand signal object creation
- ✅ Lightweight JSON serialization

## Development Commands

### Environment Setup
```bash
# Install in development mode with all dependencies
uv sync --extra test --extra dev --extra docs
```

### Testing
```bash
# Run all tests
uv run pytest

# Run specific test categories
uv run pytest -m unit          # Unit tests only
uv run pytest -m integration   # Integration tests only

# Run single test file
uv run pytest tests/test_u_indicators.py

# Run with coverage
uv run pytest --cov=thestrat --cov-report=term-missing

# Quick test run (fail fast)
uv run pytest --tb=short -x -q
```

### Code Quality
```bash
# Format and lint (automatically run by pre-commit)
uv run ruff check .           # Lint
uv run ruff format .          # Format
uv run ruff check --fix .     # Fix issues

# Type checking
uv run pyright

# Run pre-commit hooks manually
pre-commit run --all-files
```

### Documentation
```bash
# Build docs locally
mkdocs serve

# Build static docs
mkdocs build
```

## Architecture Deep Dive

### Core Components

**Factory (`factory.py`)**
- Central component creation with validated configs
- `Factory.create_all(config)` returns complete pipeline
- `Factory.create_aggregation()` and `Factory.create_indicators()` for individual components
- All methods use Pydantic schema validation

**Base Component (`base.py`)**
- Abstract `Component` class with `process(data) -> PolarsDataFrame` method
- Supports both Polars and Pandas input (auto-converts Pandas to Polars)
- All processors inherit from this base class

**Aggregation (`aggregation.py`)**
- Multi-timeframe OHLCV aggregation (1m → 5m, 1H, 1D, etc.)
- Asset class-specific timezone and session handling
- Precise time boundary control (hour_boundary, session_start)
- Returns normalized data with `timeframe` column for indicators

**Indicators (`indicators.py`)**
- Complete Strat pattern analysis (market structure, reversals, continuations)
- **Fully vectorized market structure detection** using proper peak/valley analysis
- **Direct HH/LH/HL/LL classification** without redundant intermediate columns
- **Optimized signal processing** with cascaded pattern matching
- Per-timeframe configuration via `TimeframeItemConfig`
- Gap detection (kicker, f23x, gapper patterns)
- Signal generation with on-demand metadata objects

**Signals (`signals.py`)**
- Rich signal metadata objects with trading logic
- Pattern definitions in `SIGNALS` dictionary
- `SignalMetadata` dataclass with risk management data
- Categories: reversal, continuation, breakout

### Schema System (`schemas.py`)

**Configuration Hierarchy:**
```
FactoryConfig
├── AggregationConfig (timeframes, asset_class, timezone)
└── IndicatorsConfig
    └── TimeframeItemConfig[] (per-timeframe settings)
        ├── SwingPointsConfig
        └── GapDetectionConfig
```

**Data Schema:**
- `IndicatorSchema`: Complete output schema with 34 columns
- Database integration helpers: `get_polars_dtypes()`, `get_column_descriptions()`
- Field metadata for nullable constraints and categories

### Data Flow

```
Raw OHLCV → Aggregation → Multi-timeframe data → Indicators → Analysis with signals
          (asset class)   (normalized format)    (per-TF config)  (34 columns)
```

**Key Data Requirements:**
- Input: `timestamp`, `open`, `high`, `low`, `close`, `volume`, `symbol` columns
- Aggregation adds: `timeframe` column
- Indicators output: All 41 `IndicatorSchema` columns always present
- Signals: `signal`, `type`, `bias` columns (None when no patterns)

### Test Structure

**Test Categories:**
- `test_u_*.py`: Unit tests (fast, isolated)
- `test_i_*.py`: Integration tests (slower, end-to-end)

**Test Naming Convention:**
- `test_u_<module>.py`: Unit tests for each module
- Classes: `TestClassName`
- Methods: `test_specific_behavior`

**Key Test Utilities:**
- `tests/utils/thestrat_data_utils.py`: Common test data generation
- `tests/utils/config_helpers.py`: Configuration helpers
- Schema consistency tests ensure all 34 columns always present

## Development Patterns

### Component Creation
Always use Factory pattern with validated configs:
```python
from thestrat.schemas import FactoryConfig, AggregationConfig, IndicatorsConfig
config = FactoryConfig(aggregation=..., indicators=...)
components = Factory.create_all(config)
```

### Data Processing Pipeline
```python
# Standard pipeline
aggregated = components["aggregation"].process(raw_data)
analyzed = components["indicators"].process(aggregated)
```

### Schema Consistency
The output always contains exactly 38 columns defined in `IndicatorSchema`. Signal and pattern columns are always present (with None values when no patterns detected) to ensure database integration consistency.

### Configuration Validation
All configs use Pydantic v2 with strict validation. Asset class validation includes timezone compatibility checks and session time validation.

## Zero Technical Debt Policy

**Clean Cuts Always** - No backwards compatibility shims, no temporary code, no migration helpers.
- Remove old code completely when replacing with new implementation
- No compatibility layers or feature flags for gradual migration
- Database schema changes are applied directly, not phased
- Breaking changes are acceptable for clean architecture

## Important Development Notes

- **No Pandas**: Internal processing uses Polars exclusively for performance
- **No Manual Validation**: All validation handled by Pydantic schemas
- **Schema Consistency**: All 41 `IndicatorSchema` columns must always be present in output
- **Timeframe Column**: Added by aggregation, required by indicators for per-TF processing
- **Asset Classes**: crypto, equities, fx, futures, commodities with specific timezone rules
- **Private Repository**: Code is view-only, no license for copying or redistribution
