"""
Comprehensive tests for signal metadata functionality.
"""

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
        from thestrat.signals import TargetLevel

        return SignalMetadata(
            pattern="3-2U",
            category=SignalCategory.REVERSAL,
            bias=SignalBias.LONG,
            bar_count=2,
            entry_price=150.0,
            stop_price=148.0,
            target_prices=[TargetLevel(price=155.0), TargetLevel(price=158.0), TargetLevel(price=160.0)],
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
        assert len(signal.target_prices) == 3
        assert signal.target_prices[0].price == 155.0
        assert signal.target_prices[1].price == 158.0
        assert signal.target_prices[2].price == 160.0
        assert signal.status == SignalStatus.PENDING
        assert len(signal.change_history) == 0

    def test_post_init_calculations(self):
        """Test automatic calculations after initialization."""
        signal = self.create_sample_signal()

        # Check original values are stored
        assert signal.original_stop == 148.0

        # Check risk/reward calculations (uses first target)
        assert signal.risk_amount == 2.0  # 150 - 148
        assert signal.reward_amount == 5.0  # 155 - 150 (first target)
        assert signal.risk_reward_ratio == 2.5  # 5 / 2

    def test_short_signal_calculations(self):
        """Test calculations for short signals."""
        from thestrat.signals import TargetLevel

        signal = SignalMetadata(
            pattern="3-2D",
            category=SignalCategory.REVERSAL,
            bias=SignalBias.SHORT,
            bar_count=2,
            entry_price=150.0,
            stop_price=152.0,
            target_prices=[TargetLevel(price=145.0)],
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
            entry_price=150.0,
            stop_price=148.0,
            timestamp=datetime.now(),
        )

        assert len(signal.target_prices) == 0
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

    def test_target_level_tracking(self):
        """Test TargetLevel tracking for hit status."""

        # Create a signal with multiple targets
        signal = self.create_sample_signal()

        # Verify initial state
        assert len(signal.target_prices) == 3
        assert all(not target.hit for target in signal.target_prices)
        assert all(target.hit_timestamp is None for target in signal.target_prices)

        # Update first target as hit
        signal.target_prices[0].hit = True
        signal.target_prices[0].hit_timestamp = datetime.now()

        assert signal.target_prices[0].hit is True
        assert signal.target_prices[0].hit_timestamp is not None
        assert signal.target_prices[1].hit is False  # Other targets unaffected

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
        from thestrat.signals import TargetLevel

        signal = SignalMetadata(
            pattern="3-2D",
            category=SignalCategory.REVERSAL,
            bias=SignalBias.SHORT,
            bar_count=2,
            entry_price=150.0,
            stop_price=152.0,
            target_prices=[TargetLevel(price=145.0)],
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

    # Serialization methods removed as per spec - brokerage handles persistence

    def test_signal_with_minimal_data(self):
        """Test signal creation with minimal required data."""
        signal = SignalMetadata(
            pattern="2U-2U",
            category=SignalCategory.CONTINUATION,
            bias=SignalBias.LONG,
            bar_count=2,
            entry_price=100.0,
            stop_price=95.0,
            timestamp=datetime.now(),
        )

        assert signal.symbol is None
        assert signal.timeframe is None
        assert len(signal.target_prices) == 0
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
        from thestrat.signals import TargetLevel

        return SignalMetadata(
            pattern="3-1D",
            category=SignalCategory.REVERSAL,
            bias=SignalBias.SHORT,
            bar_count=3,
            entry_price=150.0,
            stop_price=148.0,
            target_prices=[TargetLevel(price=155.0)],
            timestamp=datetime(2024, 1, 15, 10, 30),
            symbol="TEST",
            timeframe="5min",
        )

    # Tests for removed update_target and serialization methods removed
    # Brokerage handles persistence as per spec
