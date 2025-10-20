# Pattern Terminology and Visual Guide

This guide establishes clear terminology for TheStrat patterns and provides visual diagrams for all reversal and continuation signals.

## Core Terminology

### Bar Roles

**Setup Bar**

- The bar being broken (reversals) or continued (continuations)
- Provides the entry and stop price levels
- **Always the bar immediately before the trigger bar** (regardless of pattern type)
- In 2-bar patterns (e.g., 2D-2U): The first directional bar (2D)
- In 3-bar patterns (e.g., 2D-1-2U): The inside bar (1)

**Trigger Bar**

- The bar that completes and confirms the pattern
- Where the signal is detected in the DataFrame
- **Always the final bar of the pattern** (regardless of pattern type)
- In 2-bar patterns (e.g., 2D-2U): The second directional bar (2U)
- In 3-bar patterns (e.g., 2D-1-2U): The final directional bar (2U)

**Inside Bar**

- Bar with high ≤ previous bar's high AND low ≥ previous bar's low
- Creates compression before breakout/breakdown
- Appears in 3-bar patterns (labeled "1" in pattern names)

### Price Levels

**Entry Price**

- Breakout/breakdown level from setup bar
- **Long signals:** Setup bar high (breakout above)
- **Short signals:** Setup bar low (breakdown below)

**Stop Price**

- Invalidation level from setup bar
- **Long signals:** Setup bar low (invalidation if broken)
- **Short signals:** Setup bar high (invalidation if broken)

**Target Ladder**

- Series of target price levels extending to structural bounds
- Detected from historical bars before the trigger bar
- **Long signals:** Ascending ladder of highs (each target higher than previous) extending to `higher_high` or `lower_high` bound
- **Short signals:** Descending ladder of lows (each target lower than previous) extending to `lower_low` or `higher_low` bound
- First target must be beyond the trigger bar's price (above trigger high for long, below trigger low for short)
- Targets are arranged in reverse chronological order (most recent bar first)

## Visual Pattern Guide

### Inside Bar Reversals

![Inside Bar Reversals](../../diagrams/reversals-inside-bar.drawio)

**Patterns:** 2d-1-2u, 2u-1-2d, 3-1-2u, 3-1-2d

- Yellow candle = Inside bar (compression), also the setup bar
- Entry/Stop levels from setup bar (inside bar) high/low
- Targets extend from historical bars beyond trigger bar

### 2-Bar Reversals

![2-Bar Reversals](../../diagrams/reversals-2bar.drawio)

**Patterns:** 2d-2u, 2u-2d

- Two bars form the reversal pattern
- Setup bar provides entry/stop levels
- Targets extend from historical bars beyond trigger bar

### 1-Bar Rev-Strat

![1-Bar Rev-Strat](../../diagrams/reversals-1bar-revstrat.drawio)

**Patterns:** 1-3u, 1-3d

- Inside bar (setup) followed by large breakout bar (3 outside bar trigger)
- Entry/Stop from inside bar (setup bar)
- Targets extend from historical bars

### 2-Bar Rev-Strat

![2-Bar Rev-Strat](../../diagrams/reversals-2bar-revstrat.drawio)

**Patterns:** 1-2d-2u, 1-2u-2d, 1-3-2u, 1-3-2d

- Pattern starts with inside bar (1)
- Setup bar is the middle bar (2D/2U or 3 bar)
- Entry/Stop from setup bar, targets from historical bars

### 3-2 Context Reversals

![3-2 Reversals](../../diagrams/reversals-3-2.drawio)

**Patterns:** 3-2u, 3-2d

- Context-dependent patterns (require continuity analysis)
- 3-bar followed by 2-bar in opposite direction
- Reversal determined by previous trend context

### Inside Bar Continuations

![Inside Bar Continuations](../../diagrams/continuations-inside-bar.drawio)

**Patterns:** 2u-1-2u, 2d-1-2d, 3-1-2u, 3-1-2d

- Yellow candle = Inside bar
- Green line = Entry level (setup bar high/low)
- No targets stored for continuations (trend-following)

### Pattern Anatomy Example

![Pattern Anatomy](../../diagrams/pattern-anatomy-example.drawio)

**Detailed 2D-2U Example** showing:

- Setup Bar (2D) provides entry at high, stop at low
- Trigger Bar (2U) completes the pattern
- Target ladder detected from historical bars
- Targets extend to higher_high bound
- Actual price levels from real market data

## Price Level Rules

### Long Reversals (e.g., 2D-2U, 2D-1-2U)

- **Entry:** Setup bar high
- **Stop:** Setup bar low
- **Targets:** Ascending ladder of historical highs extending to `higher_high` or `lower_high` bound
  - First target must be above trigger bar high
  - Each subsequent target higher than previous
  - Maximum targets determined by `max_targets` config
- **Relationship:** `stop < entry` and `trigger_high < target_1 < target_2 < ... ≤ bound`

### Short Reversals (e.g., 2U-2D, 2U-1-2D)

- **Entry:** Setup bar low
- **Stop:** Setup bar high
- **Targets:** Descending ladder of historical lows extending to `lower_low` or `higher_low` bound
  - First target must be below trigger bar low
  - Each subsequent target lower than previous
  - Maximum targets determined by `max_targets` config
- **Relationship:** `stop > entry` and `trigger_low > target_1 > target_2 > ... ≥ bound`

### Continuations (All patterns)

- **Entry:** Setup bar high (long) or low (short)
- **Stop:** Setup bar low (long) or high (short)
- **Targets:** None (trend-following, no target ladder)

## All Pattern Names Reference

### Reversal Patterns - Long Bias

**2-Bar Patterns:**

- `2D-2U` - Down bar followed by up bar (simple reversal)

**3-Bar Patterns:**

- `1-2D-2U` - Inside, down, up (Rev Strat)
- `3-1-2U` - Context, inside, up
- `3-2D-2U` - Context, down, up
- `2D-1-2U` - Down, inside, up

**Context Patterns:**

- `3-2U` - Context-dependent reversal (requires downtrend)

### Reversal Patterns - Short Bias

**2-Bar Patterns:**

- `2U-2D` - Up bar followed by down bar (simple reversal)

**3-Bar Patterns:**

- `1-2U-2D` - Inside, up, down (Rev Strat)
- `3-1-2D` - Context, inside, down
- `3-2U-2D` - Context, up, down
- `2U-1-2D` - Up, inside, down

**Context Patterns:**

- `3-2D` - Context-dependent reversal (requires uptrend)

### Continuation Patterns - Long Bias

**2-Bar Patterns:**

- `2U-2U` - Up bar followed by up bar (continuation)

**3-Bar Patterns:**

- `2U-1-2U` - Up, inside, up

### Continuation Patterns - Short Bias

**2-Bar Patterns:**

- `2D-2D` - Down bar followed by down bar (continuation)

**3-Bar Patterns:**

- `2D-1-2D` - Down, inside, down

## Implementation Reference

### Code Locations

**Signal Detection:** `thestrat/indicators.py`

- `_detect_signals()`: Pattern matching
- `_detect_targets_for_signal()`: Target ladder detection

**Signal Objects:** `thestrat/signals.py`

- `SignalMetadata`: Complete signal data
- `TargetLevel`: Individual target tracking

**Configuration:** `thestrat/schemas.py`

- `TargetConfig`: Target detection settings
- `upper_bound`, `lower_bound`: Structural boundaries

### Database Schema

**DataFrame Columns:**

- `signal`: Pattern name (e.g., "2D-2U")
- `type`: "reversal" or "continuation"
- `bias`: "long" or "short"
- `entry_price`: From setup bar high/low
- `stop_price`: From setup bar low/high
- `target_prices`: List[Float64] - setup bar first, then ladder
- `target_count`: Number of targets

See [Signal Metadata Guide](signal-metadata.md) for complete field documentation.

## Common Questions

**Q: How are targets calculated for reversals?**

A: Targets are detected from historical bars before the trigger bar. The algorithm scans backwards through price history, building an ascending (long) or descending (short) ladder of highs/lows. The first target must be beyond the trigger bar's price level, and each subsequent target must continue the progression. Targets extend up to the configured structural bound (`higher_high`, `lower_high`, `higher_low`, or `lower_low`).

**Q: How are targets different from entry for continuations?**

A: Continuations have no targets - they're trend-following signals. Entry and stop come from the setup bar, but there's no target ladder since you're riding the trend.

**Q: What does the trigger bar do?**

A: The trigger bar completes and confirms the pattern. It's the bar where the signal is detected in the DataFrame. The setup bar provides the trading levels (entry/stop), while the trigger bar validates that the pattern is complete.

**Q: Why does setup bar matter more than trigger bar?**

A: The setup bar provides the actual trading levels (entry and stop). The trigger bar just confirms the pattern is complete. Think of setup as "what you're trading" and trigger as "when you trade it."

## Related Documentation

- [Signal Metadata Guide](signal-metadata.md) - Complete field documentation
- [DataFrame Schema](dataframe-schema.md) - Output column specifications
- [Asset Classes](asset-classes.md) - Timezone and session configuration
- [Examples](examples.md) - Real-world usage examples
