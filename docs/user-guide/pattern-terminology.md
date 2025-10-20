# Pattern Terminology and Visual Guide

This guide establishes clear terminology for TheStrat patterns and provides visual diagrams for all reversal and continuation signals.

## Core Terminology

### Bar Roles

**Setup Bar**

- The bar being broken (reversals) or continued (continuations)
- Provides the entry and stop price levels
- For 2-bar patterns: First bar
- For 3-bar patterns: First bar (before inside bar)

**Trigger Bar** (aka Signal Bar)

- The bar that completes and confirms the pattern
- Where the signal is detected in the DataFrame
- For 2-bar patterns: Second bar
- For 3-bar patterns: Third bar (after inside bar)

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

**First Target**

- The high/low of the setup bar (what's being broken)
- **Long signals:** Setup bar high
- **Short signals:** Setup bar low
- Always the first element in `target_prices` list for reversals

**Target Ladder**

- Series of additional targets extending to structural bounds
- **Long signals:** Ascending ladder of higher highs to `higher_high` or `lower_high` bound
- **Short signals:** Descending ladder of lower lows to `lower_low` or `higher_low` bound

### Bar Indexing Convention

**For 2-bar patterns (e.g., 2D-2U):**

- Bar 0: Setup bar (2D)
- Bar 1: Trigger/Signal bar (2U)

**For 3-bar patterns (e.g., 2D-1-2U):**

- Bar 0: Setup bar (2D)
- Bar 1: Inside bar (1)
- Bar 2: Trigger/Signal bar (2U)

## Visual Pattern Guide

### Inside Bar Reversals

![Inside Bar Reversals](../../diagrams/reversals-inside-bar.drawio)

**Patterns:** 2d-1-2u, 2u-1-2d, 3-1-2u, 3-1-2d

- Yellow candle = Inside bar (compression)
- Green line = First target (setup bar high/low)
- Black dashed line = Trigger level (pattern completion)

### 2-Bar Reversals

![2-Bar Reversals](../../diagrams/reversals-2bar.drawio)

**Patterns:** 2d-2u, 2u-2d, 3-2-2

- Green line = First target under/over setup bar
- Two bars form the reversal pattern
- Simple breakout/breakdown structure

### 1-Bar Rev-Strat

![1-Bar Rev-Strat](../../diagrams/reversals-1bar-revstrat.drawio)

**Patterns:** 1-3u, 1-3d

- Inside bar followed by large breakout bar (3-bar)
- Green line marks first target
- Explosive breakout from compression

### 2-Bar Rev-Strat

![2-Bar Rev-Strat](../../diagrams/reversals-2bar-revstrat.drawio)

**Patterns:** 1-2d-2u, 1-2u-2d, 1-3-2u, 1-3-2d

- Pattern starts with inside bar (1)
- Green line marks first target
- Combination of compression and directional bars

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

- Setup Bar (2D) with entry at high, stop at low
- Trigger Bar (2U) completing the pattern
- First target = setup bar high
- Target ladder extending to higher_high bound
- Actual price levels from real market data

## Price Level Rules

### Long Reversals (e.g., 2D-2U, 2D-1-2U)

- **Entry:** Setup bar high
- **Stop:** Setup bar low
- **First Target:** Setup bar high (same as entry)
- **Additional Targets:** Ascending ladder to `higher_high` or `lower_high` bound
- **Relationship:** `stop < entry ≤ first_target < target_2 < ... < bound`

### Short Reversals (e.g., 2U-2D, 2U-1-2D)

- **Entry:** Setup bar low
- **Stop:** Setup bar high
- **First Target:** Setup bar low (same as entry)
- **Additional Targets:** Descending ladder to `lower_low` or `higher_low` bound
- **Relationship:** `stop > entry ≥ first_target > target_2 > ... > bound`

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

**Q: Why is entry_price the same as first target for reversals?**

A: For reversals, the entry is at the setup bar's high (long) or low (short) - the level being broken. The first target is also this same level, representing what's being broken. Additional targets extend beyond this in a ladder.

**Q: How are targets different from entry for continuations?**

A: Continuations have no targets - they're trend-following signals. Entry and stop come from the setup bar, but there's no target ladder since you're riding the trend.

**Q: What's the difference between trigger bar and signal bar?**

A: They're the same bar - different terminology for the same concept. "Trigger bar" emphasizes it completes the pattern, "signal bar" emphasizes it's where the signal is detected in the DataFrame.

**Q: Why does setup bar matter more than trigger bar?**

A: The setup bar provides the actual trading levels (entry and stop). The trigger bar just confirms the pattern is complete. Think of setup as "what you're trading" and trigger as "when you trade it."

## Related Documentation

- [Signal Metadata Guide](signal-metadata.md) - Complete field documentation
- [DataFrame Schema](dataframe-schema.md) - Output column specifications
- [Asset Classes](asset-classes.md) - Timezone and session configuration
- [Examples](examples.md) - Real-world usage examples
