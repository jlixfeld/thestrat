# TheStrat Signal Test Fixtures

Deterministic test data for all 16 TheStrat signal patterns, providing guaranteed pattern detection and reproducible test results.

## Directory Structure

```
tests/fixtures/signals/
├── README.md                        # This file
├── generate_all_signals.py          # Fixture generation script
├── market/                          # Raw OHLC input data (16 CSVs)
│   ├── signal_2d_2u_market.csv
│   ├── signal_1_2d_2u_market.csv
│   └── ...
├── indicators/                      # Expected processing output (16 CSVs)
│   ├── signal_2d_2u_indicators.csv
│   ├── signal_1_2d_2u_indicators.csv
│   └── ...
└── charts/                          # Visual references (17 PNGs)
    ├── signal_2d_2u.png
    ├── signal_1_2d_2u.png
    ├── ...
    └── signal_comparison_grid.png   # Overview of all patterns
```

**Total:** 16 signals × 3 files = 48 files + 1 comparison grid

## Signal Patterns Included

### Reversal Patterns - Long (5)
- **2D-2U**: 2-bar reversal at Lower Low
- **1-2D-2U**: Inside bar, 2D down, 2U up at Lower Low
- **2D-1-2U**: 2D down, inside bar, 2U up at Lower Low
- **3-1-2U**: Outside bar, inside bar, 2U up at Lower Low
- **3-2D-2U**: Outside bar, 2D down, 2U up at Lower Low

### Reversal Patterns - Short (5)
- **2U-2D**: 2-bar reversal at Higher High
- **1-2U-2D**: Inside bar, 2U up, 2D down at Higher High
- **2U-1-2D**: 2U up, inside bar, 2D down at Higher High
- **3-1-2D**: Outside bar, inside bar, 2D down at Higher High
- **3-2U-2D**: Outside bar, 2U up, 2D down at Higher High

### Continuation Patterns (4)
- **2U-2U**: Directional up, continue up (long)
- **2U-1-2U**: Up, inside bar, continue up (long)
- **2D-2D**: Directional down, continue down (short)
- **2D-1-2D**: Down, inside bar, continue down (short)

### Context-Aware Reversals (2)
- **3-2U**: Outside bar then 2U with opposite continuity (long)
- **3-2D**: Outside bar then 2D with opposite continuity (short)

## Generating Fixtures

**From project root:**
```bash
python tests/fixtures/signals/generate_all_signals.py
```

**From this directory:**
```bash
python generate_all_signals.py
```

**What it does:**
1. Creates deterministic OHLC data for each signal pattern
2. Processes data through TheStrat indicators pipeline
3. Validates signal detection
4. Saves market CSVs, indicators CSVs, and annotated charts
5. Creates comparison grid showing all 16 patterns

**Output:** Byte-identical across runs (deterministic)

## Using in Tests

### Basic Usage

```python
from tests.utils.csv_signal_loader import load_signal_test_data
from tests.utils.signal_validator import assert_signal_detected, get_signal_rows

# Load test data (guaranteed to contain 2D-2U pattern)
market_df, expected_df = load_signal_test_data("2D-2U")

# Configure and process
config = IndicatorsConfig(...)
indicators = Indicators(config)
result = indicators.process(market_df)

# Validate signal detection
assert_signal_detected(result, "2D-2U")

# Get signal row
signal_row = get_signal_rows(result, "2D-2U").slice(0, 1)
signal_obj = Indicators.get_signal_object(signal_row)

# Validate properties
assert signal_obj.pattern == "2D-2U"
assert signal_obj.category.value == "reversal"
assert signal_obj.bias.value == "long"
```

### Available Utilities

**`tests/utils/csv_signal_loader.py`:**
- `load_signal_test_data(signal_name)` - Load market and indicators CSVs
- `get_all_signal_names()` - List all available signal fixtures
- `get_signal_chart_path(signal_name)` - Get path to visualization PNG
- `verify_all_signals_available()` - Verify all 16 signals have fixtures

**`tests/utils/signal_validator.py`:**
- `assert_signal_detected(df, signal_name)` - Assert pattern was detected
- `assert_indicators_match(actual, expected)` - Compare full output to CSV
- `get_signal_rows(df, signal_name)` - Filter rows for specific signal
- `assert_signal_properties()` - Validate type, bias, swing point location
- `assert_target_count()` - Validate number of target prices
- `assert_entry_stop_valid()` - Validate entry/stop relationship

## CSV File Format

### Market CSVs
Raw OHLC input data with 7 columns:
```csv
timestamp,symbol,open,high,low,close,volume
2024-01-01T09:30:00,TEST,95.0,100.0,90.0,93.0,1000.0
```

### Indicators CSVs
Full indicators output with 41 columns from `IndicatorSchema`:
- Basic OHLC: `timestamp`, `symbol`, `open`, `high`, `low`, `close`, `volume`
- Market structure: `higher_high`, `lower_high`, `higher_low`, `lower_low`
- Patterns: `continuity`, `scenario`, `in_force`
- Gaps: `kicker`, `f23x`, `f23`, `f23_trigger`, `pmg`
- Signals: `signal`, `type`, `bias`, `entry_price`, `stop_price`
- Targets: `target_prices` (comma-separated), `target_count`
- Flags: `signal_at_higher_high`, `signal_at_lower_high`, etc.

## Key Design Principles

### 1. Deterministic Generation
Every run of `generate_all_signals.py` produces **byte-identical** output:
- Fixed timestamps starting at `2024-01-01 09:30`
- Manually specified OHLC values (no randomness)
- Consistent pattern structure across regenerations

### 2. Guaranteed Pattern Detection
Each CSV contains the exact pattern it's named for:
- **5+ transition bars** between HH/LL and reversal patterns
- **Unique target prices** - within each CSV, no duplicate highs (for long signals) or duplicate lows (for short signals) to prevent duplicate targets in the target ladder
- **Verified detection** - generator validates pattern is present

### 3. Visual Documentation
PNG charts provide visual reference for debugging:
- Annotated candlestick charts with pattern highlights
- Legend showing pattern structure and bars
- Comparison grid for side-by-side pattern analysis

### 4. Regression Protection
Expected indicators CSVs lock in correct behavior:
- Can compare actual processing output to expected CSV
- Detect unintended changes in calculations
- Validate complete indicator schema (all 41 columns)

## Pattern Structure Details

### Reversal Patterns
All reversal patterns follow this structure:

**Long Reversals (at Lower Low):**
```
Bars 0-4:   Setup with initial swing high at 110
Bar 5:      Higher High at 120 (historical market structure)
Bars 6-10:  5 transition bars (below HH, above target LL)
Bars 11-13: Pattern-specific reversal bars
Bar 14:     Boundary bar
```

**Short Reversals (at Higher High):**
```
Bars 0-4:   Setup with initial swing low at 90
Bar 5:      Lower Low at 80 (historical market structure)
Bars 6-10:  5 transition bars (above LL, below target HH)
Bars 11-13: Pattern-specific reversal bars
Bar 14:     Boundary bar
```

### Continuation Patterns
Simpler structure without market structure requirements:
```
Bars 0-6:   Trending movement in one direction
Bar 7:      Continuation signal detected
```

## Migration History

**Original Approach (Before Migration):**
- Tests used inline DataFrame generation with generic uptrending data
- Patterns not guaranteed to be detected (conditional validation)
- Unreliable test results with warning messages

**Current Approach (CSV-Based):**
- Pre-generated deterministic CSV fixtures
- Patterns guaranteed to exist in test data
- 100% reliable test execution
- Visual debugging support via PNG charts

**Migration completed:** October 2025
- **16 signal tests** migrated to CSV-based approach
- **19/19 tests passing** (TestSignalMetadataIntegration)
- **Zero "pattern not detected" warnings**

## Related Documentation

- **Test module**: `tests/test_u_indicators.py` - See module docstring for CSV-based testing approach
- **Loader utilities**: `tests/utils/csv_signal_loader.py` - Documented with examples
- **Validator utilities**: `tests/utils/signal_validator.py` - Assertion helpers
- **Project guide**: `CLAUDE.md` - Testing standards and conventions

## Maintenance

### When to Regenerate
Run `generate_all_signals.py` when:
- Updating pattern structure or bar counts
- Fixing bugs in signal generation logic
- Changing indicator calculation behavior
- Adding new test scenarios

### Validation Checklist
After regenerating fixtures:
1. ✅ All 16 market CSVs created
2. ✅ All 16 indicators CSVs created
3. ✅ All 16 chart PNGs created
4. ✅ Comparison grid PNG created
5. ✅ Run tests: `pytest tests/test_u_indicators.py::TestSignalMetadataIntegration -v`
6. ✅ Verify all tests pass with no warnings

### Git Workflow
Test fixtures are version controlled:
- Commit CSVs to track expected behavior changes
- Commit PNGs for visual documentation (optional)
- Review diffs when indicators logic changes
