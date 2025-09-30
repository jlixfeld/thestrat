# Signal Metadata

**Complete guide to signal metadata objects, examples, and database integration**

TheStrat signals return rich metadata objects that provide comprehensive trading information including entry/stop/target levels, risk management data, and change tracking capabilities.

## Overview

When TheStrat indicators detect trading patterns, they generate signals with detailed metadata through the `SignalMetadata` class. This metadata transforms simple pattern strings into actionable trading objects with:

- **Price levels**: Entry and stop prices, with multiple target levels for reversals
- **Target tracking**: `TargetLevel` objects with hit status and timestamps
- **Risk management**: Risk/reward ratios and position sizing data
- **State tracking**: Signal lifecycle and execution status
- **Change history**: Audit trail for stop price adjustments
- **DataFrame integration**: JSON serialization for database storage

## Basic Signal Example

```python
from datetime import datetime
from thestrat.signals import SignalMetadata, SignalCategory, SignalBias, TargetLevel

# Create a reversal signal with multiple targets
signal = SignalMetadata(
    pattern="3-2U",
    category=SignalCategory.REVERSAL,
    bias=SignalBias.LONG,
    bar_count=2,
    entry_price=150.0,
    stop_price=148.0,
    target_prices=[
        TargetLevel(price=155.0),
        TargetLevel(price=158.0),
        TargetLevel(price=160.0)
    ],
    timestamp=datetime.now(),
    symbol="AAPL",
    timeframe="5min"
)

print(f"Signal: {signal.pattern}")
print(f"Entry: ${signal.entry_price}")
print(f"Stop: ${signal.stop_price}")
print(f"Targets: {[t.price for t in signal.target_prices]}")
print(f"Risk/Reward (first target): {signal.risk_reward_ratio:.2f}")
```

Output:
```
Signal: 3-2U
Entry: $150.0
Stop: $148.0
Targets: [155.0, 158.0, 160.0]
Risk/Reward (first target): 2.50
```

## Signal Categories and Examples

### Reversal Signals

Reversal signals have entry, stop, and multiple target prices:

```python
# Long reversal signal with multiple targets
long_reversal = SignalMetadata(
    pattern="2D-2U",
    category=SignalCategory.REVERSAL,
    bias=SignalBias.LONG,
    bar_count=2,
    entry_price=125.50,
    stop_price=123.75,
    target_prices=[
        TargetLevel(price=129.00),
        TargetLevel(price=131.50),
        TargetLevel(price=134.00)
    ],
    timestamp=datetime.now()
)

# Short reversal signal with multiple targets
short_reversal = SignalMetadata(
    pattern="2U-2D",
    category=SignalCategory.REVERSAL,
    bias=SignalBias.SHORT,
    bar_count=2,
    entry_price=98.25,
    stop_price=99.50,
    target_prices=[
        TargetLevel(price=95.00),
        TargetLevel(price=93.50),
        TargetLevel(price=92.00)
    ],
    timestamp=datetime.now()
)
```

### Continuation Signals

Continuation signals have no targets (trend-following):

```python
# Long continuation signal
continuation = SignalMetadata(
    pattern="2U-2U",
    category=SignalCategory.CONTINUATION,
    bias=SignalBias.LONG,
    bar_count=2,
    entry_price=87.50,
    stop_price=85.25,
    timestamp=datetime.now()
)

# Note: target_prices is empty for continuation signals
assert len(continuation.target_prices) == 0
assert continuation.reward_amount is None
```

## Risk Management Data

All signals automatically calculate risk metrics using the first target:

```python
signal = SignalMetadata(
    pattern="3-1-2U",
    category=SignalCategory.REVERSAL,
    bias=SignalBias.LONG,
    bar_count=3,
    entry_price=100.0,
    stop_price=97.0,
    target_prices=[
        TargetLevel(price=106.0),
        TargetLevel(price=109.0),
        TargetLevel(price=112.0)
    ],
    timestamp=datetime.now()
)

# Automatic risk calculations (based on first target)
print(f"Risk Amount: ${signal.risk_amount}")      # $3.00 (100 - 97)
print(f"Reward Amount: ${signal.reward_amount}")  # $6.00 (106 - 100, first target)
print(f"R/R Ratio: {signal.risk_reward_ratio}")   # 2.0 (6 / 3)

# Original stop preserved for tracking
print(f"Original Stop: ${signal.original_stop}")  # $97.0
```

## Stop Loss Management

### Adjusting Stop Loss

```python
# Trail stop loss (with change tracking)
signal.update_stop(98.0, "trailing_stop")

print(f"New Stop: ${signal.stop_price}")           # $98.0
print(f"Updated Risk: ${signal.risk_amount}")      # $2.0 (100 - 98)
print(f"New R/R: {signal.risk_reward_ratio}")      # 3.0 (6 / 2)

# Check change history
change = signal.change_history[0]
print(f"Change: {change.field_name} from ${change.from_value} to ${change.to_value}")
print(f"Reason: {change.reason}")
print(f"Time: {change.timestamp}")
```

### Smart Trailing

The `trail_stop()` method only allows favorable moves:

```python
# For long signals, only trails stop UP
result = signal.trail_stop(99.0)  # Move stop from 98 to 99
print(f"Trailed successfully: {result}")  # True

result = signal.trail_stop(96.0)  # Try to move stop DOWN
print(f"Trailed successfully: {result}")  # False (rejected)
print(f"Stop unchanged: ${signal.stop_price}")  # Still $99.0

# For short signals, only trails stop DOWN
short_signal.trail_stop(95.0)  # Move stop from 99.50 to 95.0 ✓
short_signal.trail_stop(101.0) # Try to move stop UP ✗ (rejected)
```

## Multiple Target Tracking

### Target Hit Management

Track when each target level is reached:

```python
# Check target status
for i, target in enumerate(signal.target_prices):
    print(f"Target {i+1}: ${target.price} - Hit: {target.hit}")

# Mark first target as hit (typically done by brokerage)
signal.target_prices[0].hit = True
signal.target_prices[0].hit_timestamp = datetime.now()

# Scale out or adjust stops based on targets hit
hit_count = sum(1 for t in signal.target_prices if t.hit)
print(f"Targets hit: {hit_count}/{len(signal.target_prices)}")

# Adjust stop to breakeven after first target
if signal.target_prices[0].hit:
    signal.trail_stop(signal.entry_price, "breakeven_after_target_1")
```

## Database Integration

### DataFrame Schema

TheStrat indicators output signal data as DataFrame columns for database storage:

```python
from thestrat import Factory, FactoryConfig, AggregationConfig, IndicatorsConfig

# Process data through TheStrat
config = FactoryConfig(
    aggregation=AggregationConfig(target_timeframes=["5min"], asset_class="equities"),
    indicators=IndicatorsConfig()
)

pipeline = Factory.create_all(config)
aggregated = pipeline["aggregation"].process(raw_data)
analyzed = pipeline["indicators"].process(aggregated)

# Signal data is available in DataFrame columns
signals_df = analyzed.filter(analyzed["signal"].is_not_null())

# Key columns for database storage:
# - signal: Pattern string (e.g., "2D-2U")
# - type: Signal category ("reversal" or "continuation")
# - bias: Direction ("long" or "short")
# - target_prices: JSON string of target price array (e.g., "[155.0, 158.0, 160.0]")
# - target_count: Integer count of targets

print(signals_df.select(["timestamp", "symbol", "signal", "type", "bias", "target_prices", "target_count"]))
```

### Database Storage Example

```python
import sqlite3
import polars as pl

# Create database table
conn = sqlite3.connect("trading.db")
cursor = conn.cursor()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS signals (
        timestamp TIMESTAMP,
        symbol TEXT,
        timeframe TEXT,
        signal TEXT,
        type TEXT,
        bias TEXT,
        target_prices TEXT,  -- JSON array
        target_count INTEGER,
        PRIMARY KEY (timestamp, symbol, timeframe)
    )
""")

# Insert signals from DataFrame
signals_df = analyzed.filter(pl.col("signal").is_not_null())

for row in signals_df.iter_rows(named=True):
    cursor.execute("""
        INSERT OR REPLACE INTO signals
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        row["timestamp"].isoformat(),
        row["symbol"],
        row["timeframe"],
        row["signal"],
        row["type"],
        row["bias"],
        row["target_prices"],  # Already JSON string
        row["target_count"]
    ))

conn.commit()
conn.close()
```

### Querying Signal Data

```python
# Query signals with SQL
cursor.execute("""
    SELECT timestamp, symbol, signal, bias, target_prices, target_count
    FROM signals
    WHERE signal = '2D-2U' AND date(timestamp) = '2024-01-15'
""")

for row in cursor.fetchall():
    timestamp, symbol, signal, bias, target_prices_json, target_count = row
    print(f"{timestamp}: {signal} on {symbol}")
    print(f"  Bias: {bias}")
    print(f"  Targets: {target_prices_json}")  # "[155.0, 158.0, 160.0]"
    print(f"  Count: {target_count}")
```

## Signal Status Management

```python
from thestrat.signals import SignalStatus

# Signal lifecycle
signal.status = SignalStatus.PENDING     # Initial state
signal.status = SignalStatus.ACTIVE      # Entry triggered
signal.status = SignalStatus.TARGET_HIT  # Target reached
signal.status = SignalStatus.STOPPED     # Stop loss hit
signal.status = SignalStatus.EXPIRED     # Signal expired
signal.status = SignalStatus.CANCELLED   # Manually cancelled

# Track execution times
signal.triggered_at = datetime.now()  # When entry was filled
signal.closed_at = datetime.now()     # When position was closed
signal.close_reason = "target_hit"    # Why position closed

# Performance tracking (update as position runs)
signal.entry_filled_price = 245.80    # Actual fill price
signal.exit_price = 251.90           # Actual exit price
signal.pnl = 6.10                    # Realized P&L
signal.max_favorable_excursion = 7.25 # Best unrealized gain
signal.max_adverse_excursion = -1.15  # Worst unrealized loss
```

## Complete Metadata Fields

The `SignalMetadata` object contains 25+ fields organized by category:

**Core Signal Data:**
- `pattern`, `category`, `bias`, `bar_count`

**Price Levels:**
- `entry_price`, `stop_price`
- `target_prices` (list of `TargetLevel` objects)
- `original_stop`

**Risk Management:**
- `risk_amount`, `reward_amount`, `risk_reward_ratio`

**State & Lifecycle:**
- `signal_id`, `status`, `timestamp`
- `triggered_at`, `closed_at`, `close_reason`

**Change Tracking:**
- `change_history` (list of `PriceChange` objects)

**Context:**
- `symbol`, `timeframe`

**Performance:**
- `entry_filled_price`, `exit_price`, `pnl`
- `max_favorable_excursion`, `max_adverse_excursion`

**Target Tracking** (via `TargetLevel` objects):
- `price`, `hit`, `hit_timestamp`

## Real-World Trading Example

Here's a complete example showing how to use `get_signal_objects()` to detect patterns and prepare for trade entry:

```python
from thestrat import Factory, FactoryConfig, AggregationConfig, IndicatorsConfig
from polars import col
import polars as pl

def monitor_signals_for_trading(raw_data):
    """
    Complete workflow: data processing → signal detection → trade preparation
    """
    # 1. Configure TheStrat components
    config = FactoryConfig(
        aggregation=AggregationConfig(
            timeframes=["5min", "1h"],
            asset_class="equities"
        ),
        indicators=IndicatorsConfig()
    )

    # 2. Create processing pipeline
    components = Factory.create_all(config)

    # 3. Process raw OHLCV data
    aggregated_data = components["aggregation"].process(raw_data)
    analyzed_data = components["indicators"].process(aggregated_data)

    # 4. Filter for current signals (last few bars only)
    current_signals = analyzed_data.filter(
        col("signal").is_not_null() &
        (col("timestamp") >= analyzed_data["timestamp"].max() - pl.duration(hours=2))
    )

    if len(current_signals) == 0:
        print("No signals detected in recent data")
        return []

    # 5. Get full signal objects with trading metadata
    signal_objects = components["indicators"].get_signal_objects(current_signals)

    # 6. Evaluate each signal for trade entry
    trade_candidates = []

    for signal in signal_objects:
        print(f"\n🎯 Signal Detected: {signal.pattern}")
        print(f"   Symbol: {signal.symbol}")
        print(f"   Timeframe: {signal.timeframe}")
        print(f"   Bias: {signal.bias.value.upper()}")
        print(f"   Category: {signal.category.value}")

        # Price levels for order placement
        print(f"\n💰 Trading Levels:")
        print(f"   Entry: ${signal.entry_price:.2f}")
        print(f"   Stop:  ${signal.stop_price:.2f}")

        if signal.target_prices:
            target_list = [f"${t.price:.2f}" for t in signal.target_prices]
            print(f"   Targets: {', '.join(target_list)}")
            print(f"   Risk/Reward (first target): {signal.risk_reward_ratio:.2f}:1")
        else:
            print(f"   Targets: None (continuation signal)")

        # Risk management
        risk_dollars = signal.risk_amount
        print(f"\n📊 Risk Management:")
        print(f"   Risk per share: ${risk_dollars:.2f}")

        # Position sizing (example: risk $100 per trade)
        max_risk = 100.0
        position_size = int(max_risk / risk_dollars)
        print(f"   Suggested position size: {position_size} shares")
        print(f"   Total capital at risk: ${position_size * risk_dollars:.2f}")

        # Entry criteria check
        entry_criteria = {
            "reasonable_risk_reward": signal.risk_reward_ratio is None or signal.risk_reward_ratio >= 1.5,
            "reasonable_risk_amount": risk_dollars <= 5.0,  # Max $5 risk per share
            "recent_signal": True,  # Already filtered above
            "clear_levels": abs(signal.entry_price - signal.stop_price) > 0.01
        }

        all_criteria_met = all(entry_criteria.values())

        print(f"\n✅ Entry Criteria:")
        for criterion, met in entry_criteria.items():
            status = "✓" if met else "✗"
            print(f"   {status} {criterion.replace('_', ' ').title()}")

        if all_criteria_met:
            trade_candidates.append({
                "signal": signal,
                "action": "BUY" if signal.bias.value == "long" else "SELL",
                "quantity": position_size,
                "entry_price": signal.entry_price,
                "stop_loss": signal.stop_price,
                "targets": [t.price for t in signal.target_prices],  # Multiple targets
                "risk_amount": position_size * risk_dollars
            })
            print(f"\n🚀 TRADE READY: {signal.pattern} on {signal.symbol}")
        else:
            print(f"\n⏸️  Criteria not met - monitoring only")

    return trade_candidates

def execute_trades(trade_candidates):
    """
    Execute trades using your broker API
    """
    for trade in trade_candidates:
        signal = trade["signal"]

        # Example order placement with multiple targets (adapt to your broker's API)
        order_params = {
            "symbol": signal.symbol,
            "side": trade["action"],
            "quantity": trade["quantity"],
            "type": "LIMIT",
            "price": trade["entry_price"],
            "stop_loss": trade["stop_loss"],
            "targets": trade["targets"]  # List of target prices
        }

        print(f"Placing order: {order_params}")
        # broker_api.place_bracket_order(**order_params)

        # Store signal data for tracking (use DataFrame columns)
        # database.store_active_signal(signal)

# Usage example
if __name__ == "__main__":
    # Your raw market data (timestamp, open, high, low, close, volume, symbol)
    raw_market_data = get_latest_market_data()  # Your data source

    # Monitor for signals and get trade-ready candidates
    candidates = monitor_signals_for_trading(raw_market_data)

    if candidates:
        print(f"\n🎯 Found {len(candidates)} trade candidates")

        # Execute trades (uncomment when ready)
        # execute_trades(candidates)
    else:
        print("\n⏳ No trade candidates found - continue monitoring")
```

### Key Benefits of This Approach

**Separation of Concerns:**
- Vectorized pattern detection handles the heavy lifting
- `get_signal_objects()` provides rich metadata only when signals are found
- No wasted computation on incomplete JSON during pattern detection

**Complete Trading Context:**
- Entry, stop, and multiple target prices calculated from actual market structure
- Risk/reward ratios and position sizing based on real price levels
- Target hit tracking for position scaling strategies

**Database Integration:**
- Store signal data as DataFrame columns for easy querying
- Update stop levels as trades evolve
- Maintain audit trail of all stop modifications
- Target prices stored as JSON for database compatibility

**Performance Optimized:**
- Fast vectorized detection identifies patterns quickly
- Signal objects created on-demand only for actionable signals
- Minimal memory overhead during bulk processing
