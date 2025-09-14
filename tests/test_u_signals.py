"""
Comprehensive tests for signal metadata functionality.
"""

import json
from datetime import datetime

import pytest

from thestrat.signals import PriceChange, SignalBias, SignalCategory, SignalMetadata, SignalStatus


class TestSignalEnums:
    """Test signal enumeration classes."""

    def test_signal_category_enum(self):
        """Test SignalCategory enum values."""
        assert SignalCategory.REVERSAL.value == "reversal"
        assert SignalCategory.CONTINUATION.value == "continuation"
        assert SignalCategory.CONTEXT.value == "context"

    def test_signal_bias_enum(self):
        """Test SignalBias enum values."""
        assert SignalBias.LONG.value == "long"
        assert SignalBias.SHORT.value == "short"

    def test_signal_status_enum(self):
        """Test SignalStatus enum values."""
        assert SignalStatus.PENDING.value == "pending"
        assert SignalStatus.ACTIVE.value == "active"
        assert SignalStatus.STOPPED.value == "stopped"
        assert SignalStatus.TARGET_HIT.value == "target_hit"
        assert SignalStatus.EXPIRED.value == "expired"
        assert SignalStatus.CANCELLED.value == "cancelled"


class TestPriceChange:
    """Test PriceChange dataclass."""

    def test_price_change_creation(self):
        """Test creating a price change record."""
        timestamp = datetime.now()
        change = PriceChange(
            field_name="stop_price", from_value=100.0, to_value=105.0, timestamp=timestamp, reason="trailing_stop"
        )

        assert change.field_name == "stop_price"
        assert change.from_value == 100.0
        assert change.to_value == 105.0
        assert change.timestamp == timestamp
        assert change.reason == "trailing_stop"

    def test_price_change_optional_reason(self):
        """Test price change with no reason."""
        timestamp = datetime.now()
        change = PriceChange(field_name="target_price", from_value=150.0, to_value=160.0, timestamp=timestamp)

        assert change.reason is None


class TestSignalMetadata:
    """Test SignalMetadata class functionality."""

    def create_sample_signal(self):
        """Create a sample signal for testing."""
        return SignalMetadata(
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
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            symbol="AAPL",
            timeframe="5min",
        )

    def test_signal_creation(self):
        """Test basic signal creation."""
        signal = self.create_sample_signal()

        assert signal.pattern == "3-2U"
        assert signal.category == SignalCategory.REVERSAL
        assert signal.bias == SignalBias.LONG
        assert signal.bar_count == 2
        assert signal.entry_price == 150.0
        assert signal.stop_price == 148.0
        assert signal.target_price == 155.0
        assert signal.status == SignalStatus.PENDING
        assert len(signal.change_history) == 0

    def test_post_init_calculations(self):
        """Test automatic calculations after initialization."""
        signal = self.create_sample_signal()

        # Check original values are stored
        assert signal.original_stop == 148.0
        assert signal.original_target == 155.0

        # Check risk/reward calculations
        assert signal.risk_amount == 2.0  # 150 - 148
        assert signal.reward_amount == 5.0  # 155 - 150
        assert signal.risk_reward_ratio == 2.5  # 5 / 2

    def test_short_signal_calculations(self):
        """Test calculations for short signals."""
        signal = SignalMetadata(
            pattern="3-2D",
            category=SignalCategory.REVERSAL,
            bias=SignalBias.SHORT,
            bar_count=2,
            entry_bar_index=100,
            trigger_bar_index=99,
            entry_price=150.0,
            stop_price=152.0,
            target_price=145.0,
            timestamp=datetime.now(),
        )

        # Check risk/reward calculations for short
        assert signal.risk_amount == 2.0  # 152 - 150
        assert signal.reward_amount == 5.0  # 150 - 145
        assert signal.risk_reward_ratio == 2.5

    def test_continuation_signal_no_target(self):
        """Test continuation signals have no target."""
        signal = SignalMetadata(
            pattern="2U-2U",
            category=SignalCategory.CONTINUATION,
            bias=SignalBias.LONG,
            bar_count=2,
            entry_bar_index=100,
            trigger_bar_index=99,
            entry_price=150.0,
            stop_price=148.0,
            timestamp=datetime.now(),
        )

        assert signal.target_price is None
        assert signal.reward_amount is None
        assert signal.risk_reward_ratio is None

    def test_update_stop_price(self):
        """Test updating stop price with tracking."""
        signal = self.create_sample_signal()
        original_stop = signal.stop_price

        # Update stop price
        signal.update_stop(149.0, "trailing_stop")

        # Check stop was updated
        assert signal.stop_price == 149.0
        assert len(signal.change_history) == 1

        # Check change record
        change = signal.change_history[0]
        assert change.field_name == "stop_price"
        assert change.from_value == original_stop
        assert change.to_value == 149.0
        assert change.reason == "trailing_stop"

        # Check metrics recalculated
        assert signal.risk_amount == 1.0  # 150 - 149
        assert signal.risk_reward_ratio == 5.0  # 5 / 1

    def test_update_stop_same_value(self):
        """Test updating stop to same value does nothing."""
        signal = self.create_sample_signal()
        original_count = len(signal.change_history)

        signal.update_stop(148.0)  # Same as current

        assert len(signal.change_history) == original_count

    def test_update_target_price(self):
        """Test updating target price with tracking."""
        signal = self.create_sample_signal()

        # Update target price
        signal.update_target(160.0, "extended_target")

        # Check target was updated
        assert signal.target_price == 160.0
        assert len(signal.change_history) == 1

        # Check change record
        change = signal.change_history[0]
        assert change.field_name == "target_price"
        assert change.from_value == 155.0
        assert change.to_value == 160.0
        assert change.reason == "extended_target"

        # Check metrics recalculated
        assert signal.reward_amount == 10.0  # 160 - 150
        assert signal.risk_reward_ratio == 5.0  # 10 / 2

    def test_update_continuation_target_raises_error(self):
        """Test updating target on continuation signal raises error."""
        signal = SignalMetadata(
            pattern="2U-2U",
            category=SignalCategory.CONTINUATION,
            bias=SignalBias.LONG,
            bar_count=2,
            entry_bar_index=100,
            trigger_bar_index=99,
            entry_price=150.0,
            stop_price=148.0,
            timestamp=datetime.now(),
        )

        with pytest.raises(ValueError, match="Continuation signals have no target"):
            signal.update_target(155.0)

    def test_trail_stop_long_signal(self):
        """Test trailing stop for long signal."""
        signal = self.create_sample_signal()

        # Trail stop up (should work)
        result = signal.trail_stop(149.0)
        assert result is True
        assert signal.stop_price == 149.0

        # Try to trail stop down (should fail)
        result = signal.trail_stop(147.0)
        assert result is False
        assert signal.stop_price == 149.0  # Unchanged

    def test_trail_stop_short_signal(self):
        """Test trailing stop for short signal."""
        signal = SignalMetadata(
            pattern="3-2D",
            category=SignalCategory.REVERSAL,
            bias=SignalBias.SHORT,
            bar_count=2,
            entry_bar_index=100,
            trigger_bar_index=99,
            entry_price=150.0,
            stop_price=152.0,
            timestamp=datetime.now(),
        )

        # Trail stop down (should work)
        result = signal.trail_stop(151.0)
        assert result is True
        assert signal.stop_price == 151.0

        # Try to trail stop up (should fail)
        result = signal.trail_stop(153.0)
        assert result is False
        assert signal.stop_price == 151.0  # Unchanged

    def test_serialization_to_dict(self):
        """Test converting signal to dictionary."""
        signal = self.create_sample_signal()

        # Add a price change for testing
        signal.update_stop(149.0, "test_update")

        data = signal.to_dict()

        # Check basic fields
        assert data["pattern"] == "3-2U"
        assert data["category"] == "reversal"
        assert data["bias"] == "long"
        assert data["status"] == "pending"

        # Check timestamp conversion
        assert data["timestamp"] == "2024-01-15T10:30:00"

        # Check price fields are numeric (not strings)
        assert isinstance(data["entry_price"], float)
        assert data["entry_price"] == 150.0

        # Check change history
        assert len(data["change_history"]) == 1
        change = data["change_history"][0]
        assert change["field_name"] == "stop_price"
        assert change["from_value"] == 148.0
        assert change["to_value"] == 149.0
        assert change["reason"] == "test_update"
        assert "timestamp" in change

    def test_deserialization_from_dict(self):
        """Test reconstructing signal from dictionary."""
        original_signal = self.create_sample_signal()
        original_signal.update_stop(149.0, "test_update")

        # Serialize and deserialize
        data = original_signal.to_dict()
        restored_signal = SignalMetadata.from_dict(data)

        # Check all fields match
        assert restored_signal.pattern == original_signal.pattern
        assert restored_signal.category == original_signal.category
        assert restored_signal.bias == original_signal.bias
        assert restored_signal.entry_price == original_signal.entry_price
        assert restored_signal.stop_price == original_signal.stop_price
        assert restored_signal.target_price == original_signal.target_price
        assert restored_signal.timestamp == original_signal.timestamp

        # Check change history
        assert len(restored_signal.change_history) == 1
        change = restored_signal.change_history[0]
        assert change.field_name == "stop_price"
        assert change.from_value == 148.0
        assert change.to_value == 149.0
        assert change.reason == "test_update"

    def test_json_serialization(self):
        """Test JSON serialization and deserialization."""
        original_signal = self.create_sample_signal()

        # Serialize to JSON
        json_str = original_signal.to_json()
        assert isinstance(json_str, str)

        # Verify valid JSON
        data = json.loads(json_str)
        assert data["pattern"] == "3-2U"

        # Deserialize from JSON
        restored_signal = SignalMetadata.from_json(json_str)

        # Check key fields match
        assert restored_signal.pattern == original_signal.pattern
        assert restored_signal.entry_price == original_signal.entry_price
        assert restored_signal.timestamp == original_signal.timestamp

    def test_signal_with_minimal_data(self):
        """Test signal creation with minimal required data."""
        signal = SignalMetadata(
            pattern="2U-2U",
            category=SignalCategory.CONTINUATION,
            bias=SignalBias.LONG,
            bar_count=2,
            entry_bar_index=50,
            trigger_bar_index=49,
            entry_price=100.0,
            stop_price=95.0,
            timestamp=datetime.now(),
        )

        assert signal.symbol is None
        assert signal.timeframe is None
        assert signal.target_price is None
        assert signal.target_bar_index is None
        assert signal.status == SignalStatus.PENDING

    def test_unique_signal_ids(self):
        """Test that each signal gets a unique ID."""
        signal1 = self.create_sample_signal()
        signal2 = self.create_sample_signal()

        assert signal1.signal_id != signal2.signal_id
        assert len(signal1.signal_id) > 0
        assert len(signal2.signal_id) > 0


@pytest.mark.unit
class TestSignalEdgeCases:
    """Test cases for Signal edge cases and error conditions."""

    def create_sample_signal(self):
        """Helper to create a sample signal for testing."""
        return SignalMetadata(
            pattern="3-1D",
            category=SignalCategory.REVERSAL,
            bias=SignalBias.SHORT,
            bar_count=3,
            entry_bar_index=100,
            trigger_bar_index=99,
            entry_price=150.0,
            stop_price=148.0,
            target_price=155.0,
            timestamp=datetime(2024, 1, 15, 10, 30),
            symbol="TEST",
            timeframe="5min",
        )

    def test_update_target_same_price_noop(self):
        """Test that updating target to the same price does nothing (early return)."""
        signal = self.create_sample_signal()

        # Get initial state
        initial_target = signal.target_price
        initial_change_count = len(signal.change_history)

        # Update to the same target price - should be a no-op
        signal.update_target(initial_target, "same_price_test")

        # Should not have created a new change record
        assert len(signal.change_history) == initial_change_count
        assert signal.target_price == initial_target

    def test_update_target_continuation_raises_error(self):
        """Test that updating target on continuation signals raises ValueError."""
        # Create a continuation signal (no target)
        continuation_signal = SignalMetadata(
            pattern="2U-2U",
            category=SignalCategory.CONTINUATION,
            bias=SignalBias.LONG,
            bar_count=2,
            entry_bar_index=100,
            trigger_bar_index=99,
            entry_price=150.0,
            stop_price=148.0,
            timestamp=datetime(2024, 1, 15, 10, 30),
        )

        # Should raise ValueError when trying to update target
        with pytest.raises(ValueError, match="Continuation signals have no target"):
            continuation_signal.update_target(155.0, "invalid_update")

    def test_to_dict_with_triggered_and_closed_timestamps(self):
        """Test to_dict serialization with triggered_at and closed_at timestamps."""
        signal = self.create_sample_signal()

        # Set triggered_at and closed_at times directly
        triggered_time = datetime(2024, 1, 15, 10, 35)
        closed_time = datetime(2024, 1, 15, 10, 40)

        signal.triggered_at = triggered_time
        signal.closed_at = closed_time
        signal.status = SignalStatus.TARGET_HIT

        # Convert to dict
        signal_dict = signal.to_dict()

        # Check that triggered_at and closed_at are in ISO format
        assert "triggered_at" in signal_dict
        assert "closed_at" in signal_dict
        assert signal_dict["triggered_at"] == triggered_time.isoformat()
        assert signal_dict["closed_at"] == closed_time.isoformat()

        # Check status
        assert signal_dict["status"] == SignalStatus.TARGET_HIT.value

    def test_from_dict_with_triggered_and_closed_timestamps(self):
        """Test from_dict deserialization with triggered_at and closed_at timestamps."""
        # Create signal dict with timestamp fields
        signal_data = {
            "signal_id": "test_123",
            "pattern": "3-1D",
            "category": "reversal",
            "bias": "short",
            "bar_count": 3,
            "entry_bar_index": 100,
            "trigger_bar_index": 99,
            "entry_price": 150.0,
            "stop_price": 148.0,
            "target_price": 155.0,
            "timestamp": "2024-01-15T10:30:00",
            "triggered_at": "2024-01-15T10:35:00",
            "closed_at": "2024-01-15T10:40:00",
            "status": "target_hit",
            "symbol": "TEST",
            "timeframe": "5min",
            "change_history": [],
        }

        # Restore from dict
        restored_signal = SignalMetadata.from_dict(signal_data)

        # Check that timestamps were properly converted
        assert restored_signal.triggered_at == datetime(2024, 1, 15, 10, 35)
        assert restored_signal.closed_at == datetime(2024, 1, 15, 10, 40)
        assert restored_signal.status == SignalStatus.TARGET_HIT

    def test_from_dict_with_none_timestamps(self):
        """Test from_dict deserialization with None timestamp fields."""
        # Create signal dict without triggered_at and closed_at
        signal_data = {
            "signal_id": "test_456",
            "pattern": "2U-2U",
            "category": "continuation",
            "bias": "long",
            "bar_count": 2,
            "entry_bar_index": 50,
            "trigger_bar_index": 49,
            "entry_price": 100.0,
            "stop_price": 95.0,
            "timestamp": "2024-01-15T10:30:00",
            "status": "pending",
            "change_history": [],
        }

        # Restore from dict
        restored_signal = SignalMetadata.from_dict(signal_data)

        # Check that None timestamps are handled properly
        assert restored_signal.triggered_at is None
        assert restored_signal.closed_at is None
        assert restored_signal.status == SignalStatus.PENDING
