"""
Signal metadata implementation for TheStrat trading system.

This module provides comprehensive signal metadata objects that transform simple
pattern strings into rich objects with trading logic, risk management, and change tracking.
"""

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

# Enhanced signal patterns with metadata for object creation
SIGNALS = {
    # Reversal Long Signals (3-bar patterns using 3rd bar for target)
    "1-2D-2U": {
        "category": "reversal",
        "bias": "long",
        "bar_count": 3,
        "entry_bar_offset": 0,  # Current bar
        "trigger_bar_offset": 1,  # Previous bar (for entry/stop)
        "target_bar_offset": 3,  # 4th bar back (for target) - Rev Strat needs 4th bar
        "description": "Rev Strat 3-bar pattern",
    },
    "3-1-2U": {
        "category": "reversal",
        "bias": "long",
        "bar_count": 3,
        "entry_bar_offset": 0,
        "trigger_bar_offset": 1,
        "target_bar_offset": 2,  # Uses 3rd bar in signal
        "description": "3-bar reversal pattern",
    },
    "3-2D-2U": {
        "category": "reversal",
        "bias": "long",
        "bar_count": 3,
        "entry_bar_offset": 0,
        "trigger_bar_offset": 1,
        "target_bar_offset": 2,  # Uses 3rd bar in signal
        "description": "3-bar reversal pattern",
    },
    "2D-1-2U": {
        "category": "reversal",
        "bias": "long",
        "bar_count": 3,
        "entry_bar_offset": 0,
        "trigger_bar_offset": 1,
        "target_bar_offset": 2,  # Uses 3rd bar in signal
        "description": "3-bar reversal pattern",
    },
    "2D-2U": {
        "category": "reversal",
        "bias": "long",
        "bar_count": 2,
        "entry_bar_offset": 0,
        "trigger_bar_offset": 1,
        "target_bar_offset": 2,  # Requires 3rd bar for target
        "description": "2-bar reversal pattern",
    },
    # Reversal Short Signals
    "1-2U-2D": {
        "category": "reversal",
        "bias": "short",
        "bar_count": 3,
        "entry_bar_offset": 0,
        "trigger_bar_offset": 1,
        "target_bar_offset": 3,  # 4th bar back (for target) - Rev Strat needs 4th bar
        "description": "Rev Strat 3-bar pattern",
    },
    "3-1-2D": {
        "category": "reversal",
        "bias": "short",
        "bar_count": 3,
        "entry_bar_offset": 0,
        "trigger_bar_offset": 1,
        "target_bar_offset": 2,  # Uses 3rd bar in signal
        "description": "3-bar reversal pattern",
    },
    "3-2U-2D": {
        "category": "reversal",
        "bias": "short",
        "bar_count": 3,
        "entry_bar_offset": 0,
        "trigger_bar_offset": 1,
        "target_bar_offset": 2,  # Uses 3rd bar in signal
        "description": "3-bar reversal pattern",
    },
    "2U-1-2D": {
        "category": "reversal",
        "bias": "short",
        "bar_count": 3,
        "entry_bar_offset": 0,
        "trigger_bar_offset": 1,
        "target_bar_offset": 2,  # Uses 3rd bar in signal
        "description": "3-bar reversal pattern",
    },
    "2U-2D": {
        "category": "reversal",
        "bias": "short",
        "bar_count": 2,
        "entry_bar_offset": 0,
        "trigger_bar_offset": 1,
        "target_bar_offset": 2,  # Requires 3rd bar for target
        "description": "2-bar reversal pattern",
    },
    # Continuation Signals (no target)
    "2U-2U": {
        "category": "continuation",
        "bias": "long",
        "bar_count": 2,
        "entry_bar_offset": 0,
        "trigger_bar_offset": 1,
        "target_bar_offset": None,  # No target for continuation
        "description": "2-bar continuation pattern",
    },
    "2U-1-2U": {
        "category": "continuation",
        "bias": "long",
        "bar_count": 3,
        "entry_bar_offset": 0,
        "trigger_bar_offset": 1,
        "target_bar_offset": None,  # No target for continuation
        "description": "3-bar continuation pattern",
    },
    "2D-2D": {
        "category": "continuation",
        "bias": "short",
        "bar_count": 2,
        "entry_bar_offset": 0,
        "trigger_bar_offset": 1,
        "target_bar_offset": None,  # No target for continuation
        "description": "2-bar continuation pattern",
    },
    "2D-1-2D": {
        "category": "continuation",
        "bias": "short",
        "bar_count": 3,
        "entry_bar_offset": 0,
        "trigger_bar_offset": 1,
        "target_bar_offset": None,  # No target for continuation
        "description": "3-bar continuation pattern",
    },
    # Context-Aware Signals (reversal based on continuity)
    "3-2U": {
        "category": "reversal",
        "bias": "long",
        "bar_count": 2,
        "entry_bar_offset": 0,
        "trigger_bar_offset": 1,
        "target_bar_offset": 2,  # Requires 3rd bar for target
        "description": "Context reversal - 3-2U with opposite continuity",
    },
    "3-2D": {
        "category": "reversal",
        "bias": "short",
        "bar_count": 2,
        "entry_bar_offset": 0,
        "trigger_bar_offset": 1,
        "target_bar_offset": 2,  # Requires 3rd bar for target
        "description": "Context reversal - 3-2D with opposite continuity",
    },
    # Special Rev Strat signals
    "1-3": {
        "category": "reversal",
        "bias": None,  # Determined by 1-bar type
        "bar_count": 2,
        "entry_bar_offset": 0,
        "trigger_bar_offset": 1,
        "target_bar_offset": 2,  # Requires 3rd bar for target
        "description": "1-bar Rev Strat signal",
    },
}


class SignalCategory(Enum):
    """Signal categorization types."""

    REVERSAL = "reversal"
    CONTINUATION = "continuation"
    CONTEXT = "context"


class SignalBias(Enum):
    """Signal directional bias."""

    LONG = "long"
    SHORT = "short"


class SignalStatus(Enum):
    """Signal lifecycle status."""

    PENDING = "pending"  # Signal detected but not triggered
    ACTIVE = "active"  # Entry triggered
    STOPPED = "stopped"  # Stop hit
    TARGET_HIT = "target_hit"  # Target reached
    EXPIRED = "expired"  # Signal no longer valid
    CANCELLED = "cancelled"  # Manually cancelled


@dataclass
class PriceChange:
    """Track changes to stop/target prices."""

    field_name: str  # "stop_price" or "target_price"
    from_value: float
    to_value: float
    timestamp: datetime
    reason: str | None = None  # e.g., "trailing_stop", "manual_adjustment"


@dataclass
class SignalMetadata:
    """Complete signal metadata with change tracking and serialization."""

    # Signal identification (required)
    pattern: str  # e.g., "3-2U", "2D-2U"
    category: SignalCategory
    bias: SignalBias
    bar_count: int

    # Bar references (indices in dataframe) (required)
    entry_bar_index: int  # Current bar (rightmost)
    trigger_bar_index: int  # Bar with entry/stop levels

    # Price levels (required)
    entry_price: float  # Trigger bar high (long) or low (short)
    stop_price: float  # Trigger bar low (long) or high (short)

    # Context (required)
    timestamp: datetime

    # Unique identifier (with default)
    signal_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Bar references (optional)
    target_bar_index: int | None = None  # For reversals

    # Price levels (optional)
    target_price: float | None = None  # For reversals only

    # Original values (for tracking changes)
    original_stop: float = field(init=False)
    original_target: float | None = field(init=False, default=None)

    # State management
    status: SignalStatus = SignalStatus.PENDING
    triggered_at: datetime | None = None
    closed_at: datetime | None = None
    close_reason: str | None = None

    # Change history
    change_history: list[PriceChange] = field(default_factory=list)

    # Context (optional)
    symbol: str | None = None
    timeframe: str | None = None

    # Risk metrics (calculated)
    risk_amount: float | None = None  # entry - stop
    reward_amount: float | None = None  # target - entry
    risk_reward_ratio: float | None = None

    # Performance tracking
    entry_filled_price: float | None = None  # Actual fill
    exit_price: float | None = None
    pnl: float | None = None
    max_favorable_excursion: float | None = None
    max_adverse_excursion: float | None = None

    def __post_init__(self):
        """Store original values and calculate metrics."""
        self.original_stop = self.stop_price
        self.original_target = self.target_price
        self._calculate_risk_metrics()

    def update_stop(self, new_stop: float, reason: str = None) -> None:
        """
        Update stop with change tracking.

        Args:
            new_stop: New stop loss price
            reason: Optional reason for the update (e.g., "trailing_stop", "manual_adjustment")
        """
        if new_stop == self.stop_price:
            return

        change = PriceChange(
            field_name="stop_price",
            from_value=self.stop_price,
            to_value=new_stop,
            timestamp=datetime.now(),
            reason=reason,
        )
        self.change_history.append(change)
        self.stop_price = new_stop
        self._calculate_risk_metrics()

    def update_target(self, new_target: float, reason: str = None) -> None:
        """
        Update target with change tracking.

        Args:
            new_target: New target price
            reason: Optional reason for the update (e.g., "manual_adjustment", "scale_out")

        Raises:
            ValueError: If called on continuation signals (they have no target)
        """
        if self.category == SignalCategory.CONTINUATION:
            raise ValueError("Continuation signals have no target")

        if new_target == self.target_price:
            return

        change = PriceChange(
            field_name="target_price",
            from_value=self.target_price,
            to_value=new_target,
            timestamp=datetime.now(),
            reason=reason,
        )
        self.change_history.append(change)
        self.target_price = new_target
        self._calculate_risk_metrics()

    def trail_stop(self, new_stop: float, reason: str = "trailing_stop") -> bool:
        """
        Trail stop loss, only allowing favorable moves.

        Args:
            new_stop: Proposed new stop price
            reason: Reason for trailing (default: "trailing_stop")

        Returns:
            True if stop was trailed, False if move was unfavorable and rejected
        """
        if self.bias == SignalBias.LONG:
            # For long trades, only trail up
            if new_stop > self.stop_price:
                self.update_stop(new_stop, reason)
                return True
        else:
            # For short trades, only trail down
            if new_stop < self.stop_price:
                self.update_stop(new_stop, reason)
                return True
        return False

    def _calculate_risk_metrics(self) -> None:
        """Calculate risk/reward metrics."""
        if self.bias == SignalBias.LONG:
            self.risk_amount = self.entry_price - self.stop_price
            if self.target_price:
                self.reward_amount = self.target_price - self.entry_price
        else:  # SHORT
            self.risk_amount = self.stop_price - self.entry_price
            if self.target_price:
                self.reward_amount = self.entry_price - self.target_price

        if self.risk_amount and self.reward_amount and self.risk_amount > 0:
            self.risk_reward_ratio = self.reward_amount / self.risk_amount

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation with all fields serializable to JSON
        """
        data = asdict(self)

        # Convert enums to strings
        data["category"] = self.category.value
        data["bias"] = self.bias.value
        data["status"] = self.status.value

        # Convert datetime to ISO format
        data["timestamp"] = self.timestamp.isoformat()

        if self.triggered_at:
            data["triggered_at"] = self.triggered_at.isoformat()
        if self.closed_at:
            data["closed_at"] = self.closed_at.isoformat()

        # Convert change history
        data["change_history"] = [
            {**asdict(change), "timestamp": change.timestamp.isoformat()} for change in self.change_history
        ]

        # Keep numeric fields as numbers for database compatibility
        # No float-to-string conversion needed - JSON natively supports numbers
        # This preserves proper types for database insertion

        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SignalMetadata":
        """
        Reconstruct SignalMetadata from dictionary.

        Args:
            data: Dictionary representation from `to_dict()`

        Returns:
            SignalMetadata instance with original values and metrics recalculated
        """
        # Make a copy to avoid modifying the original
        data = data.copy()

        # Convert strings back to enums
        data["category"] = SignalCategory(data["category"])
        data["bias"] = SignalBias(data["bias"])
        data["status"] = SignalStatus(data["status"])

        # Convert ISO strings to datetime
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])

        if data.get("triggered_at"):
            data["triggered_at"] = datetime.fromisoformat(data["triggered_at"])
        if data.get("closed_at"):
            data["closed_at"] = datetime.fromisoformat(data["closed_at"])

        # Reconstruct change history
        data["change_history"] = [
            PriceChange(**{**ch, "timestamp": datetime.fromisoformat(ch["timestamp"])})
            for ch in data.get("change_history", [])
        ]

        # Dynamically determine float fields from dataclass annotations
        import types
        import typing
        from dataclasses import fields

        float_fields = []
        for field_info in fields(cls):
            field_type = field_info.type
            # Handle Union types like float | None (both typing.Union and types.UnionType)
            origin = typing.get_origin(field_type)
            if origin is typing.Union or origin is types.UnionType:
                args = typing.get_args(field_type)
                # Check if float is in the union (e.g., float | None)
                if float in args:
                    float_fields.append(field_info.name)
            # Handle direct float type
            elif field_type is float:
                float_fields.append(field_info.name)

        for float_field in float_fields:
            if data.get(float_field) is not None:
                value = data[float_field]
                # Ensure numeric values are floats (not int)
                if isinstance(value, (int, float)):
                    data[float_field] = float(value)

        # Remove fields that are calculated in __post_init__ and shouldn't be passed to constructor
        calculated_fields = ["original_stop", "original_target", "risk_amount", "reward_amount", "risk_reward_ratio"]
        for calc_field in calculated_fields:
            data.pop(calc_field, None)

        # Create the object (original values and metrics will be recalculated in __post_init__)
        return cls(**data)

    def to_json(self) -> str:
        """
        Serialize to JSON string.

        Returns:
            JSON string representation of the signal metadata
        """
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "SignalMetadata":
        """
        Deserialize from JSON string.

        Args:
            json_str: JSON string from `to_json()`

        Returns:
            SignalMetadata instance reconstructed from JSON
        """
        return cls.from_dict(json.loads(json_str))
