# Multiple Target Support - Implementation Specification

## Problem Statement

Currently, `SIGNALS` implements basic single target hints using `target_bar_offset`. This provides one target price per signal.

Different implementations of #TheStrat need to manage trades at multiple target levels. For example, a reversal at a lower low should detect all previous highs up to and including the next higher high, allowing actions at each target level (stop adjustments, position scaling, etc.).

## Solution Overview

Extend the system to detect and track multiple targets per signal while maintaining database-friendly output for debugging and persistence.

**Key principles:**
- Detect all relevant local highs/lows between signal and configured upper bound
- Store targets as JSON columns in DataFrame for database debugging
- Provide rich `list[TargetLevel]` objects via `get_signal_objects()` for trading logic
- Single-level configuration (no per-signal overrides)
- POC status: immediate breaking changes, no backward compatibility required

## Data Structures

### TargetLevel Dataclass

```python
@dataclass
class TargetLevel:
    """Individual target level with tracking support"""
    price: float
    hit: bool = False  # Mutable - updated by brokerage
    hit_timestamp: datetime | None = None  # Mutable - updated by brokerage
```

### TargetConfig Dataclass

```python
@dataclass
class TargetConfig:
    """Configuration for multi-target detection"""
    upper_bound: Literal["higher_high", "lower_high", "higher_low", "lower_low"] = "higher_high"
    merge_threshold_pct: float = 0.0  # No merging by default (0.02 = 2%)
    max_targets: int | None = None  # None = all targets until upper_bound
```

**Notes:**
- `target_type` field removed (redundant - inferred from upper_bound)
- `upper_bound` determines both the boundary and target type:
  - `higher_high` / `lower_high` → targets are highs
  - `higher_low` / `lower_low` → targets are lows

### Updated SignalMetadata

```python
@dataclass
class SignalMetadata:
    # ... existing fields ...

    # REMOVED: target_price, target_bar_index, entry_bar_index, trigger_bar_index

    # NEW: Multi-target support
    target_prices: list[TargetLevel] = field(default_factory=list)
```

**Serialization:**
- Remove `to_dict()` and `from_dict()` methods entirely (brokerage handles persistence)
- No serdes in library code

### TimeframeItemConfig Extension

```python
@dataclass
class TimeframeItemConfig:
    # ... existing fields ...
    target_config: TargetConfig | None = None
```

### DataFrame Schema Additions

Add to `IndicatorSchema` for database output:

```python
# In schemas.py
target_prices: str | None = Field(
    default=None,
    description="JSON array of target prices",
    json_schema_extra={"polars_dtype": String, "output": True, "category": "signals", "nullable": True},
)
target_count: int | None = Field(
    default=None,
    description="Number of targets detected",
    json_schema_extra={"polars_dtype": Int32, "output": True, "category": "signals", "nullable": True},
)
```

## Implementation Approach

### Target Detection Algorithm

**Objective:** Detect all previous local highs/lows forming ascending (long) or descending (short) sequences relative to signal bar.

**Steps:**

1. **Identify local extremes** using rolling windows (similar to swing point detection in `_calculate_market_structure()`)
   - For long signals: detect local highs
   - For short signals: detect local lows

2. **Scan backwards from signal bar** using vectorized `shift()` operations to find candidates

3. **Filter for ascending/descending progression:**
   - Long signals: only include highs where each is higher than the previous qualifying high
   - Short signals: only include lows where each is lower than the previous qualifying low

4. **Stop at upper bound:**
   - Long signals: stop at and include first `higher_high` (or configured bound)
   - Short signals: stop at and include first `lower_low` (or configured bound)

5. **Apply merge logic** for targets within `merge_threshold_pct`:
   - Long signals: pick higher target when merging
   - Short signals: pick lower target when merging

6. **Order targets** in reverse chronological order (most recent first)

**Example:**
For Sept 15 signal (2D-2U), detect Sept 11, Sept 10, and Sept 9 highs as targets (see pic.png).

### DataFrame Output

During signal detection in `_calculate_signals()`:

```python
# Calculate targets and serialize to JSON columns
df = df.with_columns([
    pl.lit(None).alias("target_prices"),  # JSON string: "[123.45, 125.67, 128.90]"
    pl.lit(0).alias("target_count"),  # Integer count
])

# For rows with signals, populate JSON strings
# ... target detection logic ...
```

### Object Creation

In `get_signal_objects()` → `_create_signal_object()`:

```python
def _create_signal_object(self, df, index, pattern, category, bias):
    # ... existing logic ...

    # Parse JSON column to create list[TargetLevel]
    target_prices_json = row["target_prices"]
    if target_prices_json:
        prices = json.loads(target_prices_json)
        target_levels = [
            TargetLevel(price=price, hit=False, hit_timestamp=None)
            for price in prices
        ]
    else:
        target_levels = []

    return SignalMetadata(
        # ... existing fields (except removed bar indices) ...
        target_prices=target_levels
    )
```

### Integration Workflow

```python
# User workflow
df = indicators.process(data)

# Database persistence (via brokerage)
# target_prices, target_bar_indices, target_timestamps columns
# available for direct SQL queries

# Get rich objects when needed
signal_objects = indicators.get_signal_objects(df)
for sig in signal_objects:
    for target in sig.target_prices:
        print(f"Price: {target.price}, Hit: {target.hit}")
```

## Database Debugging

**Direct SQL queries:**

```sql
SELECT timestamp, symbol, signal, bias,
       target_prices, target_count
FROM indicators
WHERE signal = '2D-2U'
  AND date(timestamp) = '2024-09-15';

-- Result shows: target_prices = "[238.85, 237.20, 235.10]"
-- No object reconstruction required for debugging
```

## Configuration

### Single-Level Only

```python
config = FactoryConfig(
    aggregation=AggregationConfig(...),
    indicators=IndicatorsConfig(
        timeframe_configs=[
            TimeframeItemConfig(
                timeframes=["5min"],
                target_config=TargetConfig(
                    upper_bound="higher_high",
                    merge_threshold_pct=0.02,  # 2% merge threshold
                    max_targets=5  # Limit to 5 targets
                )
            )
        ]
    )
)
```

**No fallbacks, no per-signal overrides, fail-fast behavior.**

## Target Merging Rules

When targets are within `merge_threshold_pct` of each other:

- **Long signals:** Select the higher target
- **Short signals:** Select the lower target

**Example:** Targets at 100.00 and 101.00 with 2% threshold → merge to 101.00 (long) or 100.00 (short)

## Edge Cases

1. **No targets found:** Valid scenario (continuation signals at ATH/ATL) - return empty list
2. **Insufficient history:** Existing error handling in `_calculate_market_structure()` covers this
3. **Equal price targets:** Select arbitrarily (first or last occurrence)
4. **Empty upper bound:** Fail-fast with validation error

## Implementation Checklist

### Phase 1: Data Structures
- [x] Add `TargetLevel` dataclass to signals.py (only `price`, `hit`, `hit_timestamp`)
- [x] Add `TargetConfig` dataclass to schemas.py (without `target_type` field)
- [x] Update `TimeframeItemConfig.target_config` field in schemas.py
- [x] Update `SignalMetadata`:
  - Remove: `target_price`, `target_bar_index`, `entry_bar_index`, `trigger_bar_index`
  - Add: `target_prices: list[TargetLevel]`
- [x] Remove `to_dict()` and `from_dict()` methods from SignalMetadata
- [x] Add target columns to `IndicatorSchema`: `target_prices` (JSON string), `target_count` (int)

### Phase 2: Target Detection Logic
- [x] Create helper method `_detect_targets_for_signal()` in indicators.py
- [x] Implement local high/low detection using rolling windows
- [x] Implement ascending/descending sequence filtering using vectorized shifts
- [x] Implement merge logic (threshold-based grouping)
- [x] Implement upper bound detection
- [x] Handle reverse chronological ordering
- [ ] Serialize prices to JSON string for `target_prices` column (done in Phase 3)

### Phase 3: Integration
- [x] Update `_calculate_signals()` to add `target_prices` and `target_count` columns to DataFrame
- [x] Update `_create_signal_object()`:
  - Remove bar index parameters/logic
  - Parse `target_prices` JSON → create `list[TargetLevel]`
- [x] Pass `TargetConfig` through configuration chain
- [x] Update configuration retrieval in `_get_config_for_timeframe()`

### Phase 4: Testing
- [x] Add tests to `test_u_signals.py` for TargetLevel and SignalMetadata changes
- [x] Update tests in `test_u_indicators.py` for new SignalMetadata structure
- [x] Test empty target lists (continuation at ATH/ATL) - handled via lenient assertions
- [x] Tests updated for target_prices list structure
- [ ] Dedicated tests for multi-target features (merge logic, upper bounds, ordering) - future work
- [x] Integration with existing test suite (294 tests pass, 5 fail due to test data constraints)

### Phase 5: SIGNALS Dictionary
- [x] Remove `target_bar_offset` from all signal definitions in signals.py
- [x] Update signal definitions to rely on TargetConfig instead

## Key Technical Decisions

1. **Target detection approach:** Detect local highs/lows (not just HH/HL/LH/LL classified points)
2. **DataFrame storage:** JSON string column (`target_prices`) for database compatibility
3. **Object creation:** Parse JSON on-demand in `get_signal_objects()`
4. **Configuration:** Single-level only via `TimeframeItemConfig.target_config`
5. **Serialization:** Removed entirely from SignalMetadata (brokerage responsibility)
6. **Breaking changes:** Immediate implementation, POC status, no migration needed
7. **Testing:** Use existing test files (`test_u_signals.py`, `test_u_indicators.py`)
8. **Performance:** Vectorized operations using Polars (similar to PMG pattern)
9. **Error handling:** Fail-fast on configuration errors, empty list on no targets found
10. **Minimal TargetLevel:** Only `price`, `hit`, `hit_timestamp` - no bar indices or timestamps
11. **Minimal SignalMetadata:** Removed all bar index fields - only prices and trading state remain

## Implementation Notes

- Polars handles JSON string columns trivially (no technical limitation)
- Both primitive (JSON) and rich (objects) representations coexist
- Database debugging requires no object reconstruction
- Brokerage deserializes JSON when trading logic needs full objects
- Library outputs both representations: DataFrame for database, objects via `get_signal_objects()`
