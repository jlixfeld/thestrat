"""
Signal metadata implementation for TheStrat trading system.

This module provides comprehensive signal metadata objects that transform simple
pattern strings into rich objects with trading logic, risk management, and change tracking.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


@dataclass
class TargetLevel:
    """Individual target level with tracking support."""

    price: float
    hit: bool = False
    hit_timestamp: datetime | None = None
    id: str | None = None  # Broker's fill ID


# Signal patterns with metadata for object creation
SIGNALS = {
    # Reversal Long Signals
    "1-2D-2U": {
        "category": "reversal",
        "bias": "long",
        "bar_count": 3,
        "description": "Rev Strat 3-bar pattern",
    },
    "3-1-2U": {
        "category": "reversal",
        "bias": "long",
        "bar_count": 3,
        "description": "3-bar reversal pattern",
    },
    "3-2D-2U": {
        "category": "reversal",
        "bias": "long",
        "bar_count": 3,
        "description": "3-bar reversal pattern",
    },
    "2D-1-2U": {
        "category": "reversal",
        "bias": "long",
        "bar_count": 3,
        "description": "3-bar reversal pattern",
    },
    "2D-2U": {
        "category": "reversal",
        "bias": "long",
        "bar_count": 2,
        "description": "2-bar reversal pattern",
    },
    # Reversal Short Signals
    "1-2U-2D": {
        "category": "reversal",
        "bias": "short",
        "bar_count": 3,
        "description": "Rev Strat 3-bar pattern",
    },
    "3-1-2D": {
        "category": "reversal",
        "bias": "short",
        "bar_count": 3,
        "description": "3-bar reversal pattern",
    },
    "3-2U-2D": {
        "category": "reversal",
        "bias": "short",
        "bar_count": 3,
        "description": "3-bar reversal pattern",
    },
    "2U-1-2D": {
        "category": "reversal",
        "bias": "short",
        "bar_count": 3,
        "description": "3-bar reversal pattern",
    },
    "2U-2D": {
        "category": "reversal",
        "bias": "short",
        "bar_count": 2,
        "description": "2-bar reversal pattern",
    },
    # Continuation Signals
    "2U-2U": {
        "category": "continuation",
        "bias": "long",
        "bar_count": 2,
        "description": "2-bar continuation pattern",
    },
    "2U-1-2U": {
        "category": "continuation",
        "bias": "long",
        "bar_count": 3,
        "description": "3-bar continuation pattern",
    },
    "2D-2D": {
        "category": "continuation",
        "bias": "short",
        "bar_count": 2,
        "description": "2-bar continuation pattern",
    },
    "2D-1-2D": {
        "category": "continuation",
        "bias": "short",
        "bar_count": 3,
        "description": "3-bar continuation pattern",
    },
    # Context-Aware Signals (reversal based on continuity)
    "3-2U": {
        "category": "reversal",
        "bias": "long",
        "bar_count": 2,
        "description": "Context reversal - 3-2U with opposite continuity",
    },
    "3-2D": {
        "category": "reversal",
        "bias": "short",
        "bar_count": 2,
        "description": "Context reversal - 3-2D with opposite continuity",
    },
    # Special Rev Strat signals
    # "1-3": {  # too complex to implement properly. Can be inferred by looking for inside bar followed by a gapping directional 2 that changes continuity before the low (or high) of the inside bar
    #     "category": "reversal",
    #     "bias": None,  # Determined by 1-bar type
    #     "bar_count": 2,
    #     "entry_bar_offset": 0,
    #     "trigger_bar_offset": 1,
    #     "target_bar_offset": 2,  # Requires 3rd bar for target
    #     "description": "1-bar Rev Strat signal",
    # },
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

    # Price levels (required)
    entry_price: float  # Setup bar high (long) or low (short) - breakout level
    stop_price: float  # Setup bar low (long) or high (short) - invalidation level

    # Context (required)
    timestamp: datetime

    # Unique identifier (with default)
    signal_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Multi-target support (for reversals)
    target_prices: list[TargetLevel] = field(default_factory=list)

    # Gap detection fields
    signal_entry_gap: bool = False
    signal_path_gaps: int = 0

    # Original values (for tracking changes)
    original_stop: float = field(init=False)

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
    reward_amount: float | None = None  # target - entry (first target if multiple)
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
        """Calculate risk/reward metrics using first target if multiple targets exist."""
        if self.bias == SignalBias.LONG:
            self.risk_amount = self.entry_price - self.stop_price
            if self.target_prices:
                self.reward_amount = self.target_prices[0].price - self.entry_price
        else:  # SHORT
            self.risk_amount = self.stop_price - self.entry_price
            if self.target_prices:
                self.reward_amount = self.entry_price - self.target_prices[0].price

        if self.risk_amount and self.reward_amount and self.risk_amount > 0:
            self.risk_reward_ratio = self.reward_amount / self.risk_amount
