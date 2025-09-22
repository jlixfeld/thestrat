# Signal Metadata

**Complete guide to signal metadata objects, examples, and database integration**

TheStrat signals return rich metadata objects that provide comprehensive trading information including entry/stop/target levels, risk management data, and change tracking capabilities.

## Overview

When TheStrat indicators detect trading patterns, they generate signals with detailed metadata through the `SignalMetadata` class. This metadata transforms simple pattern strings into actionable trading objects with:

- **Price levels**: Entry, stop, and target prices
- **Risk management**: Risk/reward ratios and position sizing data
- **State tracking**: Signal lifecycle and execution status
- **Change history**: Audit trail for stop/target adjustments
- **Database integration**: Full serialization/deserialization support

## Basic Signal Example

```python
from datetime import datetime
from thestrat.signals import SignalMetadata, SignalCategory, SignalBias

# Create a reversal signal
signal = SignalMetadata(
    pattern="3-2U",
    category=SignalCategory.REVERSAL,
    bias=SignalBias.LONG,
    bar_count=2,
    entry_bar_index=100,
    trigger_bar_index=99,
    target_bar_index=98,
    entry_price=150.0,
    stop_price=148.0,
    target_price=155.0,
    timestamp=datetime.now(),
    symbol="AAPL",
    timeframe="5min"
)

print(f"Signal: {signal.pattern}")
print(f"Entry: ${signal.entry_price}")
print(f"Stop: ${signal.stop_price}")
print(f"Target: ${signal.target_price}")
print(f"Risk/Reward: {signal.risk_reward_ratio:.2f}")
```

Output:
```
Signal: 3-2U
Entry: $150.0
Stop: $148.0
Target: $155.0
Risk/Reward: 2.50
```

## Signal Categories and Examples

### Reversal Signals

Reversal signals have entry, stop, and target prices:

```python
# Long reversal signal
long_reversal = SignalMetadata(
    pattern="2D-2U",
    category=SignalCategory.REVERSAL,
    bias=SignalBias.LONG,
    bar_count=2,
    entry_bar_index=50,
    trigger_bar_index=49,
    target_bar_index=48,
    entry_price=125.50,
    stop_price=123.75,
    target_price=129.00,
    timestamp=datetime.now()
)

# Short reversal signal
short_reversal = SignalMetadata(
    pattern="2U-2D",
    category=SignalCategory.REVERSAL,
    bias=SignalBias.SHORT,
    bar_count=2,
    entry_bar_index=75,
    trigger_bar_index=74,
    target_bar_index=73,
    entry_price=98.25,
    stop_price=99.50,
    target_price=95.00,
    timestamp=datetime.now()
)
```

### Continuation Signals

Continuation signals have no target (trend-following):

```python
# Long continuation signal
continuation = SignalMetadata(
    pattern="2U-2U",
    category=SignalCategory.CONTINUATION,
    bias=SignalBias.LONG,
    bar_count=2,
    entry_bar_index=200,
    trigger_bar_index=199,
    entry_price=87.50,
    stop_price=85.25,
    timestamp=datetime.now()
)

# Note: target_price is None for continuation signals
assert continuation.target_price is None
assert continuation.reward_amount is None
```

## Risk Management Data

All signals automatically calculate risk metrics:

```python
signal = SignalMetadata(
    pattern="3-1-2U",
    category=SignalCategory.REVERSAL,
    bias=SignalBias.LONG,
    bar_count=3,
    entry_bar_index=100,
    trigger_bar_index=99,
    target_bar_index=97,
    entry_price=100.0,
    stop_price=97.0,
    target_price=106.0,
    timestamp=datetime.now()
)

# Automatic risk calculations
print(f"Risk Amount: ${signal.risk_amount}")      # $3.00 (100 - 97)
print(f"Reward Amount: ${signal.reward_amount}")  # $6.00 (106 - 100)
print(f"R/R Ratio: {signal.risk_reward_ratio}")   # 2.0 (6 / 3)

# Original values preserved
print(f"Original Stop: ${signal.original_stop}")     # $97.0
print(f"Original Target: ${signal.original_target}") # $106.0
```

## Updating Stop and Target Prices

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

### Adjusting Targets

```python
# Extend target for reversal signals
signal.update_target(108.0, "extended_target")

print(f"New Target: ${signal.target_price}")       # $108.0
print(f"New Reward: ${signal.reward_amount}")      # $8.0 (108 - 100)
print(f"New R/R: {signal.risk_reward_ratio}")      # 4.0 (8 / 2)

# Continuation signals cannot have targets
try:
    continuation_signal.update_target(90.0)
except ValueError as e:
    print(f"Error: {e}")  # "Continuation signals have no target"
```

## Database Integration

### Serializing for Database Storage

```python
# Convert signal to database-friendly dictionary
signal_dict = signal.to_dict()

# All fields are JSON-serializable
import json
json_data = json.dumps(signal_dict)

# Example database insert (using any database library)
cursor.execute("""
    INSERT INTO signals (
        signal_id, pattern, category, bias, entry_price, stop_price,
        target_price, timestamp, symbol, timeframe, signal_data
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""", (
    signal.signal_id,
    signal.pattern,
    signal.category.value,
    signal.bias.value,
    signal.entry_price,
    signal.stop_price,
    signal.target_price,
    signal.timestamp.isoformat(),
    signal.symbol,
    signal.timeframe,
    json_data  # Complete metadata as JSON
))
```

### Recreating Signals from Database

```python
# Retrieve from database
cursor.execute("SELECT signal_data FROM signals WHERE signal_id = ?", (signal_id,))
json_data = cursor.fetchone()[0]

# Method 1: From JSON string
restored_signal = SignalMetadata.from_json(json_data)

# Method 2: From dictionary
signal_dict = json.loads(json_data)
restored_signal = SignalMetadata.from_dict(signal_dict)

# All metadata is preserved
assert restored_signal.pattern == original_signal.pattern
assert restored_signal.entry_price == original_signal.entry_price
assert restored_signal.risk_reward_ratio == original_signal.risk_reward_ratio
assert len(restored_signal.change_history) == len(original_signal.change_history)

# Original values and calculations are restored
print(f"Original stop restored: ${restored_signal.original_stop}")
print(f"Risk metrics recalculated: {restored_signal.risk_reward_ratio}")
```

## Complete Database Workflow Example

```python
import json
import sqlite3
from datetime import datetime
from thestrat.signals import SignalMetadata, SignalCategory, SignalBias

# 1. Create and modify a signal
signal = SignalMetadata(
    pattern="1-2D-2U",
    category=SignalCategory.REVERSAL,
    bias=SignalBias.LONG,
    bar_count=3,
    entry_bar_index=500,
    trigger_bar_index=499,
    target_bar_index=496,  # Rev Strat uses 4th bar back
    entry_price=245.75,
    stop_price=243.50,
    target_price=250.00,
    timestamp=datetime.now(),
    symbol="SPY",
    timeframe="15min"
)

# Apply some updates
signal.update_stop(244.00, "partial_trail")
signal.update_target(252.00, "extended_target")

# 2. Store in database
conn = sqlite3.connect("trading.db")
cursor = conn.cursor()

# Create table
cursor.execute("""
    CREATE TABLE IF NOT EXISTS signals (
        signal_id TEXT PRIMARY KEY,
        pattern TEXT,
        symbol TEXT,
        timeframe TEXT,
        entry_price REAL,
        current_stop REAL,
        current_target REAL,
        status TEXT,
        created_at TEXT,
        signal_metadata TEXT
    )
""")

# Insert signal
cursor.execute("""
    INSERT INTO signals VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""", (
    signal.signal_id,
    signal.pattern,
    signal.symbol,
    signal.timeframe,
    signal.entry_price,
    signal.stop_price,
    signal.target_price,
    signal.status.value,
    signal.timestamp.isoformat(),
    signal.to_json()
))

conn.commit()

# 3. Retrieve and reconstruct
cursor.execute("SELECT signal_metadata FROM signals WHERE signal_id = ?",
               (signal.signal_id,))
json_data = cursor.fetchone()[0]

# Restore complete signal object
restored_signal = SignalMetadata.from_json(json_data)

# 4. Continue trading operations
print(f"Restored signal: {restored_signal.pattern}")
print(f"Current stop: ${restored_signal.stop_price}")
print(f"Changes made: {len(restored_signal.change_history)}")

# Apply more updates
if restored_signal.trail_stop(244.50, "continued_trail"):
    print("Stop trailed successfully")

    # Update database with new state
    cursor.execute("""
        UPDATE signals
        SET current_stop = ?, signal_metadata = ?
        WHERE signal_id = ?
    """, (
        restored_signal.stop_price,
        restored_signal.to_json(),
        restored_signal.signal_id
    ))
    conn.commit()

conn.close()
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

The `SignalMetadata` object contains 30+ fields organized by category:

**Core Signal Data:**
- `pattern`, `category`, `bias`, `bar_count`
- `entry_bar_index`, `trigger_bar_index`, `target_bar_index`

**Price Levels:**
- `entry_price`, `stop_price`, `target_price`
- `original_stop`, `original_target`

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

All fields support full serialization and database integration with type preservation.

## Trading Integration Example

Here's a complete example showing how to use `get_signal_objects()` to prepare for live trading when current signals are detected:

```python
from datetime import datetime
from thestrat import Factory
from thestrat.schemas import FactoryConfig, AggregationConfig, IndicatorsConfig, TimeframeItemConfig, SwingPointsConfig

def monitor_signals_for_trading(raw_data):
    """Monitor incoming market data for trading signals."""

    # Create complete pipeline
    config = FactoryConfig(
        aggregation=AggregationConfig(
            target_timeframes=["5min"],
            asset_class="equities"
        ),
        indicators=IndicatorsConfig(
            timeframe_configs=[
                TimeframeItemConfig(
                    timeframes=["all"],
                    swing_points=SwingPointsConfig(window=3, threshold=0.5)
                )
            ]
        )
    )

    components = Factory.create_all(config)

    # Process data through pipeline
    aggregated = components["aggregation"].process(raw_data)
    current_signals = components["indicators"].process(aggregated)

    # Get complete signal objects for trading decisions
    signal_objects = components["indicators"].get_signal_objects(current_signals)

    if signal_objects:
        print(f"Found {len(signal_objects)} active signals:")

        for signal in signal_objects:
            print(f"\n=== {signal.pattern} Signal ===")
            print(f"Symbol: {signal.symbol}")
            print(f"Category: {signal.category.value}")
            print(f"Bias: {signal.bias.value}")
            print(f"Entry Price: ${signal.entry_price:.2f}")
            print(f"Stop Loss: ${signal.stop_price:.2f}")

            if signal.target_price:
                print(f"Target Price: ${signal.target_price:.2f}")
                print(f"Risk/Reward Ratio: {signal.risk_reward_ratio:.2f}")
            else:
                print("Target: None (continuation signal)")

            print(f"Risk Amount: ${signal.risk_amount:.2f}")

            # Example trading logic
            prepare_trade_order(signal)
    else:
        print("No signals detected in current data")

def prepare_trade_order(signal):
    """Prepare actual trade order from signal metadata."""

    # Calculate position size based on risk
    account_risk_percent = 0.02  # Risk 2% of account per trade
    account_balance = 100000     # $100k account
    max_risk_dollars = account_balance * account_risk_percent

    # Position size = Max risk / Signal risk
    position_size = int(max_risk_dollars / signal.risk_amount)

    order_details = {
        "symbol": signal.symbol,
        "action": "BUY" if signal.bias.value == "long" else "SELL",
        "quantity": position_size,
        "order_type": "STOP_LIMIT",
        "entry_price": signal.entry_price,
        "stop_loss": signal.stop_price,
        "target_price": signal.target_price,
        "signal_id": signal.signal_id,
        "risk_amount": signal.risk_amount,
        "expected_reward": signal.reward_amount
    }

    print(f"  → Order prepared: {order_details['action']} {position_size} shares")
    print(f"  → Max risk: ${max_risk_dollars:.2f}")
    print(f"  → Signal ID: {signal.signal_id}")

    # Send to broker API or trading system
    # place_order(order_details)

    return order_details

# Example usage with live market data
if __name__ == "__main__":
    # Simulate incoming market data
    import polars as pl

    live_data = pl.DataFrame({
        "timestamp": [datetime.now()],
        "symbol": ["AAPL"],
        "open": [150.0],
        "high": [152.0],
        "low": [149.0],
        "close": [151.5],
        "volume": [1000000]
    })

    monitor_signals_for_trading(live_data)
```

This example demonstrates:

- **Real-time Processing**: Using the complete pipeline to analyze incoming market data
- **Signal Detection**: Finding active patterns in current market conditions
- **Rich Metadata**: Accessing complete trading information from signal objects
- **Risk Management**: Calculating position sizes based on signal risk
- **Order Preparation**: Converting signals into actionable trade orders
- **Trading Integration**: Ready-to-use structure for broker API integration

The key advantage is that `get_signal_objects()` provides complete trading metadata on-demand, eliminating the need for redundant JSON generation during vectorized processing while maintaining full trading functionality.
