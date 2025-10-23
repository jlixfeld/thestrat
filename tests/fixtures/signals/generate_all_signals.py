"""
Comprehensive TheStrat Signal Test Data Generation.

Generates deterministic synthetic OHLC datasets, full indicator outputs, and
visualizations for all 16 TheStrat signals (reversals and continuations).

Every run produces byte-identical outputs for test case stability.

Usage:
    From project root: python tests/fixtures/signals/generate_all_signals.py
    From this directory: python generate_all_signals.py

Output Directory Structure (relative to this script):
./
├── market/              - signal_<name>_market.csv (16 files)
├── indicators/          - signal_<name>_indicators.csv (16 files)
└── charts/              - signal_<name>.png (16 files + 1 comparison grid)

Total: 16 signals × 3 files = 48 files + 1 comparison chart
"""

from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import polars as pl
from matplotlib.patches import Rectangle

from thestrat.factory import Factory
from thestrat.schemas import (
    AggregationConfig,
    FactoryConfig,
    IndicatorsConfig,
    SwingPointsConfig,
    TargetConfig,
    TimeframeItemConfig,
)

# Configure Polars for full display
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(2000)
pl.Config.set_fmt_str_lengths(1000)

# =============================================================================
# OUTPUT DIRECTORY CONFIGURATION
# =============================================================================

# Output directories for test fixtures (relative to this script's location)
OUTPUT_BASE = Path(__file__).parent
OUTPUT_MARKET = OUTPUT_BASE / "market"
OUTPUT_INDICATORS = OUTPUT_BASE / "indicators"
OUTPUT_CHARTS = OUTPUT_BASE / "charts"

# Ensure directories exist
OUTPUT_MARKET.mkdir(parents=True, exist_ok=True)
OUTPUT_INDICATORS.mkdir(parents=True, exist_ok=True)
OUTPUT_CHARTS.mkdir(parents=True, exist_ok=True)


# =============================================================================
# REUSABLE CONFIGURATION
# =============================================================================


def create_factory_config() -> FactoryConfig:
    """
    Create deterministic factory configuration.

    Same config used for all patterns to ensure consistency.

    Returns:
        FactoryConfig with swing points and target detection enabled
    """
    return FactoryConfig(
        aggregation=AggregationConfig(
            target_timeframes=["1min"],
            asset_class="equities",
        ),
        indicators=IndicatorsConfig(
            timeframe_configs=[
                TimeframeItemConfig(
                    timeframes=["all"],
                    swing_points=SwingPointsConfig(
                        window=1,  # 3-bar rolling window
                        threshold=0.0,  # No percentage threshold
                    ),
                    target_config=TargetConfig(),  # Enable target detection
                )
            ]
        ),
    )


# =============================================================================
# REUSABLE VISUALIZATION
# =============================================================================


def create_signal_chart(df: pl.DataFrame, signal_name: str, metadata: dict[str, Any], output_file: str) -> None:
    """
    Create annotated OHLC chart for any signal pattern.

    Reuses the chart design from test_2d2u_dataframe.py with centered overlays
    and left-justified OHLC text.

    Args:
        df: Polars DataFrame with OHLC data
        signal_name: Signal pattern name (e.g., "2D-2U")
        metadata: Dict with pattern info (legend text, key bar numbers, etc.)
        output_file: Output filename
    """
    # Convert to pandas for matplotlib
    pdf = df.to_pandas()

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(14, 8))

    # Bar styling
    bar_width = 0.6
    color_up = "#26a69a"  # Green for bullish
    color_down = "#ef5350"  # Red for bearish

    # Plot each OHLC bar
    for idx, row in pdf.iterrows():
        x_pos = idx

        # Determine color based on close vs open
        if row["close"] >= row["open"]:
            color = color_up
            body_bottom = row["open"]
            body_height = row["close"] - row["open"]
        else:
            color = color_down
            body_bottom = row["close"]
            body_height = row["open"] - row["close"]

        # Draw high-low line (wick)
        ax.plot([x_pos, x_pos], [row["low"], row["high"]], color=color, linewidth=1.5)

        # Draw open-close rectangle (body)
        rect = Rectangle(
            (x_pos - bar_width / 2, body_bottom),
            bar_width,
            body_height,
            facecolor=color,
            edgecolor=color,
            linewidth=1,
        )
        ax.add_patch(rect)

        # Add combined bar number and OHLC price labels above the bar
        # Left-justified text within centered box
        ohlc_text = f"Bar {idx}\no: {row['open']:.1f}\nh: {row['high']:.1f}\nl: {row['low']:.1f}\nc: {row['close']:.1f}"

        # Estimate box width for centering
        max_line_length = max(len(line) for line in ohlc_text.split("\n"))
        box_width_estimate = max_line_length * 0.08

        ax.text(
            x_pos - box_width_estimate / 2,  # Center the box over the bar
            row["high"] + 1,
            ohlc_text,
            ha="left",  # Left-justify text within box
            va="bottom",
            fontsize=6,
            color="darkblue",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="lightgray", linewidth=0.5),
        )

    # Customize plot
    ax.set_xlabel("Time / Bar Index", fontsize=12)
    ax.set_ylabel("Price", fontsize=12)
    ax.set_title(f"OHLC Bar Chart - {signal_name} Pattern", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")

    # Adjust y-axis limits to prevent label overlap
    y_min = pdf["low"].min()
    y_max = pdf["high"].max()
    y_range = y_max - y_min
    ax.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.35)

    # Set x-axis labels with timestamps
    x_labels = [f"{row['timestamp'].strftime('%H:%M')}" for _, row in pdf.iterrows()]
    ax.set_xticks(range(len(pdf)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right")

    # Add legend from metadata
    if "legend_text" in metadata:
        ax.text(
            0.02,
            0.98,
            metadata["legend_text"],
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    # Save chart
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# DETERMINISTIC HELPER FUNCTIONS
# =============================================================================


def validate_ohlc(bar: dict[str, float]) -> None:
    """
    Validate OHLC data meets requirements.

    Args:
        bar: Dict with open, high, low, close keys

    Raises:
        AssertionError: If OHLC constraints violated
    """
    assert bar["high"] >= bar["open"], f"High {bar['high']} < Open {bar['open']}"
    assert bar["high"] >= bar["close"], f"High {bar['high']} < Close {bar['close']}"
    assert bar["low"] <= bar["open"], f"Low {bar['low']} > Open {bar['open']}"
    assert bar["low"] <= bar["close"], f"Low {bar['low']} > Close {bar['close']}"


def create_inside_bar(prev_high: float, prev_low: float, continuity: str = "green") -> dict[str, float]:
    """
    Create deterministic inside bar (scenario "1").

    Bar is contained within previous range: high <= prev_high AND low >= prev_low.

    Args:
        prev_high: Previous bar's high
        prev_low: Previous bar's low
        continuity: "green" (bullish) or "red" (bearish)

    Returns:
        Dict with open, high, low, close (deterministic)
    """
    range_size = prev_high - prev_low

    if continuity == "green":
        open_price = prev_low + (range_size * 0.3)  # Open lower
        close_price = prev_low + (range_size * 0.7)  # Close higher
    else:  # red
        open_price = prev_low + (range_size * 0.7)  # Open higher
        close_price = prev_low + (range_size * 0.3)  # Close lower

    bar = {
        "open": open_price,
        "high": prev_high - 0.5,  # Slightly below prev high
        "low": prev_low + 0.5,  # Slightly above prev low
        "close": close_price,
    }

    validate_ohlc(bar)
    return bar


def create_2u_bar(
    prev_high: float, prev_low: float, break_amount: float = 2.0, continuity: str = "green"
) -> dict[str, float]:
    """
    Create deterministic 2U bar (scenario "2U").

    Bar breaks above: high > prev_high AND low >= prev_low.

    Args:
        prev_high: Previous bar's high
        prev_low: Previous bar's low
        break_amount: How much to break above prev_high (default 2.0)
        continuity: "green" (bullish) or "red" (bearish)

    Returns:
        Dict with open, high, low, close (deterministic)
    """
    new_high = prev_high + break_amount

    if continuity == "green":
        open_price = prev_low + 1.0
        close_price = new_high - 0.5  # Close near high
    else:  # red
        open_price = new_high - 0.5  # Open near high
        close_price = prev_low + 1.0  # Close near low

    bar = {
        "open": open_price,
        "high": new_high,
        "low": prev_low,  # Equal to or slightly above prev_low
        "close": close_price,
    }

    validate_ohlc(bar)
    return bar


def create_2d_bar(
    prev_high: float, prev_low: float, break_amount: float = 2.0, continuity: str = "red"
) -> dict[str, float]:
    """
    Create deterministic 2D bar (scenario "2D").

    Bar breaks below: low < prev_low AND high <= prev_high.

    Args:
        prev_high: Previous bar's high
        prev_low: Previous bar's low
        break_amount: How much to break below prev_low (default 2.0)
        continuity: "red" (bearish) or "green" (bullish)

    Returns:
        Dict with open, high, low, close (deterministic)
    """
    new_low = prev_low - break_amount

    if continuity == "red":
        open_price = prev_high - 1.0
        close_price = new_low + 0.5  # Close near low
    else:  # green
        open_price = new_low + 0.5  # Open near low
        close_price = prev_high - 1.0  # Close near high

    bar = {
        "open": open_price,
        "high": prev_high,  # Equal to or slightly below prev_high
        "low": new_low,
        "close": close_price,
    }

    validate_ohlc(bar)
    return bar


def create_3_bar(
    prev_high: float, prev_low: float, break_up: float = 2.0, break_down: float = 2.0, continuity: str = "green"
) -> dict[str, float]:
    """
    Create deterministic 3 bar (scenario "3").

    Bar breaks both: high > prev_high AND low < prev_low (outside bar).

    Args:
        prev_high: Previous bar's high
        prev_low: Previous bar's low
        break_up: How much to break above prev_high (default 2.0)
        break_down: How much to break below prev_low (default 2.0)
        continuity: "green" (bullish) or "red" (bearish)

    Returns:
        Dict with open, high, low, close (deterministic)
    """
    new_high = prev_high + break_up
    new_low = prev_low - break_down

    if continuity == "green":
        open_price = new_low + 1.0
        close_price = new_high - 1.0  # Close near high
    else:  # red
        open_price = new_high - 1.0
        close_price = new_low + 1.0  # Close near low

    bar = {
        "open": open_price,
        "high": new_high,
        "low": new_low,
        "close": close_price,
    }

    validate_ohlc(bar)
    return bar


def add_transition_bars_long(
    bars: list[dict[str, float]], hh_price: float, target_ll: float, num_bars: int = 5
) -> None:
    """
    Add transition bars for long reversals between HH and LL.

    Transition bars must:
    - Stay BELOW the HH (no new higher high)
    - Stay ABOVE the target LL (don't create LL yet)
    - Create a sideways-to-down pattern with UNIQUE prices
    - Include lower_high swing point

    Args:
        bars: List to append transition bars to (modified in place)
        hh_price: Higher High price to stay below
        target_ll: Lower Low price to stay above (until reversal)
        num_bars: Number of transition bars to add (default 5)

    Modifies bars list in place by appending transition bars.
    """
    # Create deterministic, unique transition bars
    # Each bar has unique high and low values
    # Highs descend from near HH, lows descend but stay above target LL

    # Define exact transition bars with unique prices
    transition_data = [
        # Bar 6: Just below HH
        {"high": hh_price - 2.0, "low": target_ll + 18.0},
        # Bar 7: Lower high swing point (important for target detection)
        {"high": hh_price - 5.0, "low": target_ll + 15.0},
        # Bar 8: Continue down
        {"high": hh_price - 8.0, "low": target_ll + 12.0},
        # Bar 9: Continue down
        {"high": hh_price - 11.0, "low": target_ll + 9.0},
        # Bar 10: Last transition bar
        {"high": hh_price - 14.0, "low": target_ll + 6.0},
    ]

    for i, data in enumerate(transition_data[:num_bars]):
        high = data["high"]
        low = data["low"]
        # Alternate between bullish and bearish closes
        if i % 2 == 0:
            open_price = low + 2.0
            close_price = high - 2.0  # Bullish
        else:
            open_price = high - 2.0
            close_price = low + 2.0  # Bearish

        bar = {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close_price,
        }
        validate_ohlc(bar)
        bars.append(bar)


def add_transition_bars_short(
    bars: list[dict[str, float]], ll_price: float, target_hh: float, num_bars: int = 5
) -> None:
    """
    Add transition bars for short reversals between LL and HH.

    Transition bars must:
    - Stay ABOVE the LL (no new lower low)
    - Stay BELOW the target HH (don't create HH yet)
    - Create a sideways-to-up pattern with UNIQUE prices
    - Include higher_low swing point

    Args:
        bars: List to append transition bars to (modified in place)
        ll_price: Lower Low price to stay above
        target_hh: Higher High price to stay below (until reversal)
        num_bars: Number of transition bars to add (default 5)

    Modifies bars list in place by appending transition bars.
    """
    # Create deterministic, unique transition bars
    # Each bar has unique high and low values
    # Lows ascend from near LL, highs ascend but stay below target HH

    # Define exact transition bars with unique prices
    transition_data = [
        # Bar 6: Just above LL
        {"low": ll_price + 2.0, "high": target_hh - 18.0},
        # Bar 7: Higher low swing point (important for target detection)
        {"low": ll_price + 5.0, "high": target_hh - 15.0},
        # Bar 8: Continue up
        {"low": ll_price + 8.0, "high": target_hh - 12.0},
        # Bar 9: Continue up
        {"low": ll_price + 11.0, "high": target_hh - 9.0},
        # Bar 10: Last transition bar
        {"low": ll_price + 14.0, "high": target_hh - 6.0},
    ]

    for i, data in enumerate(transition_data[:num_bars]):
        low = data["low"]
        high = data["high"]
        # Alternate between bullish and bearish closes
        if i % 2 == 0:
            open_price = low + 2.0
            close_price = high - 2.0  # Bullish
        else:
            open_price = high - 2.0
            close_price = low + 2.0  # Bearish

        bar = {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close_price,
        }
        validate_ohlc(bar)
        bars.append(bar)


def build_uptrend_to_ll(base_price: float = 100.0, target_ll: float = 80.0) -> list[dict[str, float]]:
    """
    Build deterministic uptrend structure with HH and 5 transition bars.

    Creates:
    - Bar 0: Baseline
    - Bar 1: First swing high
    - Bar 2: Inside bar
    - Bar 3: First swing low
    - Bar 4: Transition 2D
    - Bar 5: Higher High (HISTORICAL) ✓
    - Bars 6-10: 5 transition bars (stay below HH, above target LL) ✓

    All HIGH values are unique within this dataset to prevent duplicate targets
    in the target ladder for long signals.

    Used for long reversal patterns. Reversal pattern should be added after this.

    Args:
        base_price: Starting price level (default 100.0)
        target_ll: Target Lower Low price (default 80.0)

    Returns:
        List of 11 bar dicts (bars 0-10) - deterministic
    """
    bars = []
    hh_price = base_price + 20  # HH at 120 for base_price=100

    # Manually specify all 11 bars with unique prices
    # Bar 0: Baseline
    bars.append({"open": 95.0, "high": 100.0, "low": 90.0, "close": 93.0})

    # Bar 1: First swing high at 110
    bars.append({"open": 101.0, "high": 110.0, "low": 94.0, "close": 108.0})

    # Bar 2: Inside bar
    bars.append({"open": 105.0, "high": 107.0, "low": 96.0, "close": 99.0})

    # Bar 3: First swing low at 85
    bars.append({"open": 97.0, "high": 98.0, "low": 85.0, "close": 87.0})

    # Bar 4: Transition up
    bars.append({"open": 91.0, "high": 102.0, "low": 88.0, "close": 100.5})

    # Bar 5: Higher High (HISTORICAL) at 120
    bars.append({"open": 109.0, "high": hh_price, "low": 104.0, "close": 118.0})

    # Bar 6: Transition (below HH, above LL)
    bars.append({"open": 111.0, "high": 117.0, "low": 103.0, "close": 115.0})

    # Bar 7: Transition with swing high at 114
    bars.append({"open": 112.0, "high": 114.0, "low": 106.0, "close": 108.5})

    # Bar 8: Transition down
    bars.append({"open": 109.5, "high": 111.0, "low": 101.0, "close": 103.5})

    # Bar 9: Transition down
    bars.append({"open": 105.5, "high": 107.5, "low": 97.5, "close": 99.5})

    # Bar 10: Last transition (sets up for LL)
    bars.append({"open": 101.5, "high": 104.5, "low": 93.5, "close": 95.5})

    # Validate all bars
    for bar in bars:
        validate_ohlc(bar)

    return bars


def build_downtrend_to_hh(base_price: float = 100.0, target_hh: float = 120.0) -> list[dict[str, float]]:
    """
    Build deterministic downtrend structure with LL and 5 transition bars.

    Creates:
    - Bar 0: Baseline
    - Bar 1: First swing low
    - Bar 2: Inside bar
    - Bar 3: First swing high
    - Bar 4: Transition 2U
    - Bar 5: Lower Low (HISTORICAL) ✓
    - Bars 6-10: 5 transition bars (stay above LL, below target HH) ✓

    All LOW values are unique within this dataset to prevent duplicate targets
    in the target ladder for short signals.

    Used for short reversal patterns. Reversal pattern should be added after this.

    Args:
        base_price: Starting price level (default 100.0)
        target_hh: Target Higher High price (default 120.0)

    Returns:
        List of 11 bar dicts (bars 0-10) - deterministic
    """
    bars = []
    ll_price = base_price - 20  # LL at 80 for base_price=100

    # Manually specify all 11 bars with unique prices
    # Bar 0: Baseline
    bars.append({"open": 105.0, "high": 110.0, "low": 100.0, "close": 107.0})

    # Bar 1: First swing low at 90
    bars.append({"open": 99.0, "high": 108.0, "low": 90.0, "close": 92.0})

    # Bar 2: Inside bar
    bars.append({"open": 95.0, "high": 105.0, "low": 93.0, "close": 101.0})

    # Bar 3: First swing high at 115
    bars.append({"open": 103.0, "high": 115.0, "low": 102.0, "close": 113.0})

    # Bar 4: Transition down
    bars.append({"open": 109.0, "high": 112.0, "low": 96.0, "close": 98.0})

    # Bar 5: Lower Low (HISTORICAL) at 80
    bars.append({"open": 91.0, "high": 97.0, "low": ll_price, "close": 82.0})

    # Bar 6: Transition (above LL, below HH)
    bars.append({"open": 89.0, "high": 103.5, "low": 83.0, "close": 85.0})

    # Bar 7: Transition with swing low at 86
    bars.append({"open": 88.0, "high": 106.0, "low": 86.0, "close": 91.5})

    # Bar 8: Transition up
    bars.append({"open": 90.5, "high": 109.0, "low": 89.0, "close": 96.5})

    # Bar 9: Transition up
    bars.append({"open": 94.5, "high": 112.5, "low": 92.5, "close": 100.5})

    # Bar 10: Last transition (sets up for HH)
    bars.append({"open": 98.5, "high": 115.5, "low": 93.5, "close": 104.5})

    # Validate all bars
    for bar in bars:
        validate_ohlc(bar)

    return bars


# =============================================================================
# PATTERN GENERATORS - Reversal Long (5 patterns)
# =============================================================================


def generate_2d_2u_reversal() -> tuple[pl.DataFrame, str, dict[str, Any]]:
    """
    Generate 2D-2U long reversal at Lower Low.

    Ported from test_2d2u_dataframe.py - deterministic reference implementation.

    Pattern structure:
    - Bar 1: First swing high (110)
    - Bar 3: First swing low (85)
    - Bar 5: Higher High (120) - HISTORICAL
    - Bars 6-10: 5 transition bars (no new HH)
    - Bar 7: Local swing high (115) - creates lower_high
    - Bar 11: 2D setup bar - creates LL at 80
    - Bar 12: 2U trigger bar - completes pattern AT the LL

    Returns:
        Tuple of (DataFrame, description, metadata)
    """
    base_time = datetime(2024, 1, 1, 9, 30)

    # All HIGH values are unique within this CSV to prevent duplicate targets
    data = {
        "timestamp": [base_time + timedelta(minutes=i) for i in range(15)],
        "symbol": ["TEST"] * 15,
        "open": [
            95.0,
            100.0,
            102.0,
            96.0,
            92.0,  # 0-4: Setup to HH
            109.0,  # 5: HH bar
            107.0,
            111.0,
            104.0,
            101.0,
            98.0,  # 6-10: 5 transition bars
            91.0,
            94.0,  # 11-12: 2D-2U reversal
            97.0,
            100.5,  # 13-14: Boundary bars
        ],
        "high": [
            100.0,
            110.0,
            105.0,
            97.0,
            95.0,  # 0-4: Bar 1 = swing high at 110
            120.0,  # 5: HH = 120 (swing high)
            112.0,
            115.0,
            108.0,
            106.0,
            103.0,  # 6-10: Transition (bar 7 = swing high at 115)
            93.0,
            96.0,  # 11-12: Reversal
            99.0,
            101.0,  # 13-14: Boundary bars
        ],
        "low": [
            90.0,
            92.0,
            94.0,
            85.0,
            88.0,  # 0-4: Bar 3 = swing low at 85
            98.0,  # 5
            100.0,
            97.0,
            91.0,
            87.0,
            83.0,  # 6-10: Transition
            80.0,
            89.0,  # 11-12: Bar 11 = LL at 80 (swing low)
            95.0,
            99.0,  # 13-14: Boundary bars
        ],
        "close": [
            93.0,
            108.0,
            96.0,
            86.0,
            89.0,  # 0-4
            118.0,  # 5
            105.0,
            113.0,
            102.0,
            92.0,
            88.0,  # 6-10
            82.0,
            94.5,  # 11-12
            97.5,
            99.5,  # 13-14
        ],
        "volume": [1000.0] * 15,
    }

    df = pl.DataFrame(data)

    description = "2D-2U long reversal at Lower Low with historical Higher High"

    metadata = {
        "legend_text": (
            "Bar 1: First swing high (110)\n"
            "Bar 3: First swing low (85)\n"
            "Bar 5: Higher High (120) ✓\n"
            "Bars 6-10: 5 transition bars\n"
            "Bar 11: 2D setup @ Lower Low (80) ✓\n"
            "Bar 12: 2U trigger (2D-2U pattern) ✓"
        ),
        "signal_bar": 12,
        "setup_bar": 11,
        "hh_bar": 5,
        "ll_bar": 11,
    }

    return df, description, metadata


def generate_1_2d_2u_reversal() -> tuple[pl.DataFrame, str, dict[str, Any]]:
    """
    Generate 1-2D-2U long reversal at Lower Low.

    3-bar pattern: Inside bar, 2D down, 2U up reversal.

    Returns:
        Tuple of (DataFrame, description, metadata)
    """
    base_time = datetime(2024, 1, 1, 9, 30)
    # build_uptrend_to_ll includes HH + 5 transition bars (bars 0-10)
    bars = build_uptrend_to_ll(base_price=100.0, target_ll=80.0)

    # Manually add pattern-specific bars (all highs unique within CSV for target ladder)
    # Bar 11: Inside bar (within bar 10 range)
    bars.append({"open": 98.0, "high": 103.25, "low": 94.75, "close": 96.5})

    # Bar 12: 2D setup bar (creates LL at 80.0)
    bars.append({"open": 95.0, "high": 103.0, "low": 80.0, "close": 82.5})

    # Bar 13: 2U trigger bar
    bars.append({"open": 83.0, "high": 116.0, "low": 81.0, "close": 113.5})

    # Bar 14: Boundary bar
    bars.append({"open": 115.0, "high": 119.0, "low": 84.5, "close": 117.0})

    # Validate pattern-specific bars
    for bar in bars[11:]:
        validate_ohlc(bar)

    # Convert to DataFrame
    data = {
        "timestamp": [base_time + timedelta(minutes=i) for i in range(len(bars))],
        "symbol": ["TEST"] * len(bars),
        "open": [b["open"] for b in bars],
        "high": [b["high"] for b in bars],
        "low": [b["low"] for b in bars],
        "close": [b["close"] for b in bars],
        "volume": [1000.0] * len(bars),
    }

    df = pl.DataFrame(data)
    description = "1-2D-2U long reversal: inside bar, 2D down, 2U up at Lower Low"
    metadata = {
        "legend_text": "3-bar reversal:\nBar N: Inside bar\nBar N+1: 2D setup @ LL\nBar N+2: 2U trigger",
        "signal_bar": len(bars) - 2,
    }

    return df, description, metadata


def generate_2d_1_2u_reversal() -> tuple[pl.DataFrame, str, dict[str, Any]]:
    """
    Generate 2D-1-2U long reversal at Lower Low.

    3-bar pattern: 2D down, inside bar, 2U up reversal.

    Returns:
        Tuple of (DataFrame, description, metadata)
    """
    base_time = datetime(2024, 1, 1, 9, 30)
    # build_uptrend_to_ll includes HH + 5 transition bars (bars 0-10)
    bars = build_uptrend_to_ll(base_price=100.0, target_ll=80.0)

    # Manually add pattern-specific bars (all highs unique within CSV for target ladder)
    # Bar 11: 2D setup bar (creates LL at 80.0)
    bars.append({"open": 96.0, "high": 104.0, "low": 80.0, "close": 82.0})

    # Bar 12: Inside bar
    bars.append({"open": 85.0, "high": 103.5, "low": 81.5, "close": 100.0})

    # Bar 13: 2U trigger bar
    bars.append({"open": 101.5, "high": 116.5, "low": 82.5, "close": 114.5})

    # Bar 14: Boundary bar
    bars.append({"open": 113.0, "high": 119.5, "low": 86.5, "close": 117.5})

    # Validate pattern-specific bars
    for bar in bars[11:]:
        validate_ohlc(bar)

    # Convert to DataFrame
    data = {
        "timestamp": [base_time + timedelta(minutes=i) for i in range(len(bars))],
        "symbol": ["TEST"] * len(bars),
        "open": [b["open"] for b in bars],
        "high": [b["high"] for b in bars],
        "low": [b["low"] for b in bars],
        "close": [b["close"] for b in bars],
        "volume": [1000.0] * len(bars),
    }

    df = pl.DataFrame(data)
    description = "2D-1-2U long reversal: 2D down, inside bar, 2U up at Lower Low"
    metadata = {
        "legend_text": "3-bar reversal:\nBar N: 2D setup @ LL\nBar N+1: Inside bar\nBar N+2: 2U trigger",
        "signal_bar": len(bars) - 2,
    }

    return df, description, metadata


def generate_3_1_2u_reversal() -> tuple[pl.DataFrame, str, dict[str, Any]]:
    """
    Generate 3-1-2U long reversal at Lower Low.

    3-bar pattern: Outside bar (3), inside bar, 2U up reversal.

    Returns:
        Tuple of (DataFrame, description, metadata)
    """
    base_time = datetime(2024, 1, 1, 9, 30)
    # build_uptrend_to_ll includes HH + 5 transition bars (bars 0-10)
    bars = build_uptrend_to_ll(base_price=100.0, target_ll=80.0)

    # Manually add pattern-specific bars (all highs unique within CSV for target ladder)
    # Bar 11: 3 bar - outside bar (creates LL at 79.0)
    bars.append({"open": 92.0, "high": 105.25, "low": 79.0, "close": 81.75})

    # Bar 12: Inside bar
    bars.append({"open": 86.0, "high": 104.75, "low": 79.5, "close": 102.5})

    # Bar 13: 2U trigger bar
    bars.append({"open": 100.25, "high": 117.5, "low": 83.5, "close": 115.25})

    # Bar 14: Boundary bar
    bars.append({"open": 114.0, "high": 121.0, "low": 87.5, "close": 118.5})

    # Validate pattern-specific bars
    for bar in bars[11:]:
        validate_ohlc(bar)

    # Convert to DataFrame
    data = {
        "timestamp": [base_time + timedelta(minutes=i) for i in range(len(bars))],
        "symbol": ["TEST"] * len(bars),
        "open": [b["open"] for b in bars],
        "high": [b["high"] for b in bars],
        "low": [b["low"] for b in bars],
        "close": [b["close"] for b in bars],
        "volume": [1000.0] * len(bars),
    }

    df = pl.DataFrame(data)
    description = "3-1-2U long reversal: outside bar, inside bar, 2U up at Lower Low"
    metadata = {
        "legend_text": "3-bar reversal:\nBar N: 3 (outside) @ LL\nBar N+1: Inside bar\nBar N+2: 2U trigger",
        "signal_bar": len(bars) - 2,
    }

    return df, description, metadata


def generate_3_2d_2u_reversal() -> tuple[pl.DataFrame, str, dict[str, Any]]:
    """
    Generate 3-2D-2U long reversal at Lower Low.

    3-bar pattern: Outside bar (3), 2D down, 2U up reversal.

    Returns:
        Tuple of (DataFrame, description, metadata)
    """
    base_time = datetime(2024, 1, 1, 9, 30)
    # build_uptrend_to_ll includes HH + 5 transition bars (bars 0-10)
    bars = build_uptrend_to_ll(base_price=100.0, target_ll=80.0)

    # Manually add pattern-specific bars (all highs unique within CSV for target ladder)
    # Bar 11: 3 bar - outside bar (creates LL at 78.0)
    bars.append({"open": 94.0, "high": 106.5, "low": 78.0, "close": 79.75})

    # Bar 12: 2D bar
    bars.append({"open": 81.0, "high": 105.75, "low": 76.0, "close": 77.5})

    # Bar 13: 2U trigger bar
    bars.append({"open": 78.5, "high": 122.0, "low": 77.0, "close": 121.25})

    # Bar 14: Boundary bar
    bars.append({"open": 120.5, "high": 123.0, "low": 91.5, "close": 122.5})

    # Validate pattern-specific bars
    for bar in bars[11:]:
        validate_ohlc(bar)

    # Convert to DataFrame
    data = {
        "timestamp": [base_time + timedelta(minutes=i) for i in range(len(bars))],
        "symbol": ["TEST"] * len(bars),
        "open": [b["open"] for b in bars],
        "high": [b["high"] for b in bars],
        "low": [b["low"] for b in bars],
        "close": [b["close"] for b in bars],
        "volume": [1000.0] * len(bars),
    }

    df = pl.DataFrame(data)
    description = "3-2D-2U long reversal: outside bar, 2D down, 2U up at Lower Low"
    metadata = {
        "legend_text": "3-bar reversal:\nBar N: 3 (outside) @ LL\nBar N+1: 2D\nBar N+2: 2U trigger",
        "signal_bar": len(bars) - 2,
    }

    return df, description, metadata


# =============================================================================
# PATTERN GENERATORS - Reversal Short (5 patterns)
# =============================================================================


def generate_2u_2d_reversal() -> tuple[pl.DataFrame, str, dict[str, Any]]:
    """
    Generate 2U-2D short reversal at Higher High.

    2-bar pattern: 2U up, then 2D down reversal (mirror of 2D-2U).

    Returns:
        Tuple of (DataFrame, description, metadata)
    """
    base_time = datetime(2024, 1, 1, 9, 30)
    # build_downtrend_to_hh includes LL + 5 transition bars (bars 0-10)
    bars = build_downtrend_to_hh(base_price=100.0, target_hh=120.0)

    # Manually add pattern-specific bars (all lows unique within CSV for target ladder)
    # Bar 11: 2U setup bar (creates HH at 120.0)
    bars.append({"open": 106.5, "high": 120.0, "low": 94.0, "close": 118.0})

    # Bar 12: 2D trigger bar
    bars.append({"open": 117.0, "high": 119.0, "low": 84.0, "close": 87.0})

    # Bar 13: Boundary bar
    bars.append({"open": 95.5, "high": 116.0, "low": 88.5, "close": 111.0})

    # Bar 14: Boundary bar
    bars.append({"open": 107.5, "high": 114.0, "low": 91.0, "close": 109.5})

    # Validate pattern-specific bars
    for bar in bars[11:]:
        validate_ohlc(bar)

    # Convert to DataFrame
    data = {
        "timestamp": [base_time + timedelta(minutes=i) for i in range(len(bars))],
        "symbol": ["TEST"] * len(bars),
        "open": [b["open"] for b in bars],
        "high": [b["high"] for b in bars],
        "low": [b["low"] for b in bars],
        "close": [b["close"] for b in bars],
        "volume": [1000.0] * len(bars),
    }

    df = pl.DataFrame(data)
    description = "2U-2D short reversal at Higher High with historical Lower Low"
    metadata = {
        "legend_text": "Bar 5: Lower Low (80) ✓\nBars 6-10: 5 transition bars\nBar 11: 2U setup @ HH (120) ✓\nBar 12: 2D trigger",
        "signal_bar": 12,
        "setup_bar": 11,
    }

    return df, description, metadata


def generate_1_2u_2d_reversal() -> tuple[pl.DataFrame, str, dict[str, Any]]:
    """
    Generate 1-2U-2D short reversal at Higher High.

    3-bar pattern: Inside bar, 2U up, 2D down reversal.

    Returns:
        Tuple of (DataFrame, description, metadata)
    """
    base_time = datetime(2024, 1, 1, 9, 30)
    # build_downtrend_to_hh includes LL + 5 transition bars (bars 0-10)
    bars = build_downtrend_to_hh(base_price=100.0, target_hh=120.0)

    # Manually add pattern-specific bars (all lows unique within CSV for target ladder)
    # Bar 11: Inside bar
    bars.append({"open": 99.5, "high": 114.5, "low": 94.5, "close": 111.5})

    # Bar 12: 2U setup bar (creates HH at 121.0)
    bars.append({"open": 112.0, "high": 121.0, "low": 95.5, "close": 119.5})

    # Bar 13: 2D trigger bar
    bars.append({"open": 118.5, "high": 120.5, "low": 87.5, "close": 88.0})

    # Bar 14: Boundary bar
    bars.append({"open": 94.25, "high": 117.5, "low": 84.5, "close": 89.5})

    # Validate pattern-specific bars
    for bar in bars[11:]:
        validate_ohlc(bar)

    data = {
        "timestamp": [base_time + timedelta(minutes=i) for i in range(len(bars))],
        "symbol": ["TEST"] * len(bars),
        "open": [b["open"] for b in bars],
        "high": [b["high"] for b in bars],
        "low": [b["low"] for b in bars],
        "close": [b["close"] for b in bars],
        "volume": [1000.0] * len(bars),
    }

    df = pl.DataFrame(data)
    description = "1-2U-2D short reversal: inside bar, 2U up, 2D down at Higher High"
    metadata = {
        "legend_text": "3-bar reversal:\nBar N: Inside bar\nBar N+1: 2U setup @ HH\nBar N+2: 2D trigger",
        "signal_bar": len(bars) - 2,
    }

    return df, description, metadata


def generate_2u_1_2d_reversal() -> tuple[pl.DataFrame, str, dict[str, Any]]:
    """
    Generate 2U-1-2D short reversal at Higher High.

    3-bar pattern: 2U up, inside bar, 2D down reversal.

    Returns:
        Tuple of (DataFrame, description, metadata)
    """
    base_time = datetime(2024, 1, 1, 9, 30)
    # build_downtrend_to_hh includes LL + 5 transition bars (bars 0-10)
    bars = build_downtrend_to_hh(base_price=100.0, target_hh=120.0)

    # Manually add pattern-specific bars (all lows unique within CSV for target ladder)
    # Bar 11: 2U setup bar (creates HH at 122.0)
    bars.append({"open": 110.5, "high": 122.0, "low": 97.5, "close": 120.75})

    # Bar 12: Inside bar
    bars.append({"open": 118.25, "high": 121.5, "low": 98.5, "close": 99.25})

    # Bar 13: 2D trigger bar
    bars.append({"open": 101.25, "high": 121.25, "low": 85.5, "close": 86.75})

    # Bar 14: Boundary bar
    bars.append({"open": 92.75, "high": 118.75, "low": 81.5, "close": 83.25})

    # Validate pattern-specific bars
    for bar in bars[11:]:
        validate_ohlc(bar)

    data = {
        "timestamp": [base_time + timedelta(minutes=i) for i in range(len(bars))],
        "symbol": ["TEST"] * len(bars),
        "open": [b["open"] for b in bars],
        "high": [b["high"] for b in bars],
        "low": [b["low"] for b in bars],
        "close": [b["close"] for b in bars],
        "volume": [1000.0] * len(bars),
    }

    df = pl.DataFrame(data)
    description = "2U-1-2D short reversal: 2U up, inside bar, 2D down at Higher High"
    metadata = {
        "legend_text": "3-bar reversal:\nBar N: 2U setup @ HH\nBar N+1: Inside bar\nBar N+2: 2D trigger",
        "signal_bar": len(bars) - 2,
    }

    return df, description, metadata


def generate_3_1_2d_reversal() -> tuple[pl.DataFrame, str, dict[str, Any]]:
    """
    Generate 3-1-2D short reversal at Higher High.

    3-bar pattern: Outside bar (3), inside bar, 2D down reversal.

    Returns:
        Tuple of (DataFrame, description, metadata)
    """
    base_time = datetime(2024, 1, 1, 9, 30)
    # build_downtrend_to_hh includes LL + 5 transition bars (bars 0-10)
    bars = build_downtrend_to_hh(base_price=100.0, target_hh=120.0)

    # Manually add pattern-specific bars (all lows unique within CSV for target ladder)
    # Bar 11: 3 bar - outside bar (creates HH at 123.0)
    bars.append({"open": 104.0, "high": 123.0, "low": 91.75, "close": 121.75})

    # Bar 12: Inside bar
    bars.append({"open": 119.75, "high": 122.5, "low": 99.75, "close": 101.75})

    # Bar 13: 2D trigger bar
    bars.append({"open": 103.25, "high": 122.25, "low": 82.5, "close": 83.75})

    # Bar 14: Boundary bar
    bars.append({"open": 88.75, "high": 119.25, "low": 78.5, "close": 79.25})

    # Validate pattern-specific bars
    for bar in bars[11:]:
        validate_ohlc(bar)

    data = {
        "timestamp": [base_time + timedelta(minutes=i) for i in range(len(bars))],
        "symbol": ["TEST"] * len(bars),
        "open": [b["open"] for b in bars],
        "high": [b["high"] for b in bars],
        "low": [b["low"] for b in bars],
        "close": [b["close"] for b in bars],
        "volume": [1000.0] * len(bars),
    }

    df = pl.DataFrame(data)
    description = "3-1-2D short reversal: outside bar, inside bar, 2D down at Higher High"
    metadata = {
        "legend_text": "3-bar reversal:\nBar N: 3 (outside) @ HH\nBar N+1: Inside bar\nBar N+2: 2D trigger",
        "signal_bar": len(bars) - 2,
    }

    return df, description, metadata


def generate_3_2u_2d_reversal() -> tuple[pl.DataFrame, str, dict[str, Any]]:
    """
    Generate 3-2U-2D short reversal at Higher High.

    3-bar pattern: Outside bar (3), 2U up, 2D down reversal.

    Returns:
        Tuple of (DataFrame, description, metadata)
    """
    base_time = datetime(2024, 1, 1, 9, 30)
    # build_downtrend_to_hh includes LL + 5 transition bars (bars 0-10)
    bars = build_downtrend_to_hh(base_price=100.0, target_hh=120.0)

    # Manually add pattern-specific bars (all lows unique within CSV for target ladder)
    # Bar 11: 3 bar - outside bar (creates HH at 124.0)
    bars.append({"open": 107.25, "high": 124.0, "low": 92.25, "close": 123.25})

    # Bar 12: 2U bar
    bars.append({"open": 121.0, "high": 125.0, "low": 101.25, "close": 124.25})

    # Bar 13: 2D trigger bar
    bars.append({"open": 122.75, "high": 124.75, "low": 77.5, "close": 78.75})

    # Bar 14: Boundary bar
    bars.append({"open": 85.25, "high": 118.25, "low": 75.5, "close": 76.75})

    # Validate pattern-specific bars
    for bar in bars[11:]:
        validate_ohlc(bar)

    data = {
        "timestamp": [base_time + timedelta(minutes=i) for i in range(len(bars))],
        "symbol": ["TEST"] * len(bars),
        "open": [b["open"] for b in bars],
        "high": [b["high"] for b in bars],
        "low": [b["low"] for b in bars],
        "close": [b["close"] for b in bars],
        "volume": [1000.0] * len(bars),
    }

    df = pl.DataFrame(data)
    description = "3-2U-2D short reversal: outside bar, 2U up, 2D down at Higher High"
    metadata = {
        "legend_text": "3-bar reversal:\nBar N: 3 (outside) @ HH\nBar N+1: 2U\nBar N+2: 2D trigger",
        "signal_bar": len(bars) - 2,
    }

    return df, description, metadata


# =============================================================================
# PATTERN GENERATORS - Continuation (4 patterns)
# =============================================================================


def generate_2u_2u_continuation() -> tuple[pl.DataFrame, str, dict[str, Any]]:
    """
    Generate 2U-2U long continuation pattern.

    2-bar pattern: 2U up, then another 2U up (continuing uptrend).

    Returns:
        Tuple of (DataFrame, description, metadata)
    """
    base_time = datetime(2024, 1, 1, 9, 30)

    # Build uptrend
    bars = []
    bars.append({"open": 95.0, "high": 100.0, "low": 90.0, "close": 98.0})
    bars.append({"open": 100.0, "high": 105.0, "low": 98.0, "close": 104.0})
    bars.append({"open": 105.0, "high": 110.0, "low": 104.0, "close": 109.0})
    bars.append({"open": 109.0, "high": 113.0, "low": 108.0, "close": 112.0})
    bars.append({"open": 112.0, "high": 116.0, "low": 111.0, "close": 115.0})

    # 2U-2U continuation
    prev_bar = bars[-1]
    bars.append(create_2u_bar(prev_bar["high"], prev_bar["low"], break_amount=3.0, continuity="green"))
    prev_bar = bars[-1]
    bars.append(create_2u_bar(prev_bar["high"], prev_bar["low"], break_amount=3.0, continuity="green"))

    # Boundary bars
    prev_bar = bars[-1]
    bars.append(create_2u_bar(prev_bar["high"], prev_bar["low"], break_amount=2.0, continuity="green"))

    data = {
        "timestamp": [base_time + timedelta(minutes=i) for i in range(len(bars))],
        "symbol": ["TEST"] * len(bars),
        "open": [b["open"] for b in bars],
        "high": [b["high"] for b in bars],
        "low": [b["low"] for b in bars],
        "close": [b["close"] for b in bars],
        "volume": [1000.0] * len(bars),
    }

    df = pl.DataFrame(data)
    description = "2U-2U long continuation: directional up, then continue up"
    metadata = {
        "legend_text": "2-bar continuation:\nBar N: 2U\nBar N+1: 2U (continues uptrend)",
        "signal_bar": len(bars) - 2,
    }

    return df, description, metadata


def generate_2u_1_2u_continuation() -> tuple[pl.DataFrame, str, dict[str, Any]]:
    """
    Generate 2U-1-2U long continuation pattern.

    3-bar pattern: 2U up, inside bar, 2U up (continuing uptrend).

    Returns:
        Tuple of (DataFrame, description, metadata)
    """
    base_time = datetime(2024, 1, 1, 9, 30)

    # Build uptrend
    bars = []
    bars.append({"open": 95.0, "high": 100.0, "low": 90.0, "close": 98.0})
    bars.append({"open": 100.0, "high": 105.0, "low": 98.0, "close": 104.0})
    bars.append({"open": 105.0, "high": 110.0, "low": 104.0, "close": 109.0})
    bars.append({"open": 109.0, "high": 113.0, "low": 108.0, "close": 112.0})
    bars.append({"open": 112.0, "high": 116.0, "low": 111.0, "close": 115.0})

    # 2U-1-2U continuation
    prev_bar = bars[-1]
    bars.append(create_2u_bar(prev_bar["high"], prev_bar["low"], break_amount=3.0, continuity="green"))
    prev_bar = bars[-1]
    bars.append(create_inside_bar(prev_bar["high"], prev_bar["low"], continuity="green"))
    prev_bar = bars[-1]
    bars.append(create_2u_bar(prev_bar["high"], prev_bar["low"], break_amount=3.0, continuity="green"))

    # Boundary bars
    prev_bar = bars[-1]
    bars.append(create_2u_bar(prev_bar["high"], prev_bar["low"], break_amount=2.0, continuity="green"))

    data = {
        "timestamp": [base_time + timedelta(minutes=i) for i in range(len(bars))],
        "symbol": ["TEST"] * len(bars),
        "open": [b["open"] for b in bars],
        "high": [b["high"] for b in bars],
        "low": [b["low"] for b in bars],
        "close": [b["close"] for b in bars],
        "volume": [1000.0] * len(bars),
    }

    df = pl.DataFrame(data)
    description = "2U-1-2U long continuation: up, inside bar, continue up"
    metadata = {
        "legend_text": "3-bar continuation:\nBar N: 2U\nBar N+1: Inside bar\nBar N+2: 2U (continues)",
        "signal_bar": len(bars) - 2,
    }

    return df, description, metadata


def generate_2d_2d_continuation() -> tuple[pl.DataFrame, str, dict[str, Any]]:
    """
    Generate 2D-2D short continuation pattern.

    2-bar pattern: 2D down, then another 2D down (continuing downtrend).

    Returns:
        Tuple of (DataFrame, description, metadata)
    """
    base_time = datetime(2024, 1, 1, 9, 30)

    # Build downtrend
    bars = []
    bars.append({"open": 105.0, "high": 110.0, "low": 100.0, "close": 102.0})
    bars.append({"open": 100.0, "high": 102.0, "low": 95.0, "close": 96.0})
    bars.append({"open": 95.0, "high": 96.0, "low": 90.0, "close": 91.0})
    bars.append({"open": 91.0, "high": 92.0, "low": 87.0, "close": 88.0})
    bars.append({"open": 88.0, "high": 89.0, "low": 84.0, "close": 85.0})

    # 2D-2D continuation
    prev_bar = bars[-1]
    bars.append(create_2d_bar(prev_bar["high"], prev_bar["low"], break_amount=3.0, continuity="red"))
    prev_bar = bars[-1]
    bars.append(create_2d_bar(prev_bar["high"], prev_bar["low"], break_amount=3.0, continuity="red"))

    # Boundary bars
    prev_bar = bars[-1]
    bars.append(create_2d_bar(prev_bar["high"], prev_bar["low"], break_amount=2.0, continuity="red"))

    data = {
        "timestamp": [base_time + timedelta(minutes=i) for i in range(len(bars))],
        "symbol": ["TEST"] * len(bars),
        "open": [b["open"] for b in bars],
        "high": [b["high"] for b in bars],
        "low": [b["low"] for b in bars],
        "close": [b["close"] for b in bars],
        "volume": [1000.0] * len(bars),
    }

    df = pl.DataFrame(data)
    description = "2D-2D short continuation: directional down, then continue down"
    metadata = {
        "legend_text": "2-bar continuation:\nBar N: 2D\nBar N+1: 2D (continues downtrend)",
        "signal_bar": len(bars) - 2,
    }

    return df, description, metadata


def generate_2d_1_2d_continuation() -> tuple[pl.DataFrame, str, dict[str, Any]]:
    """
    Generate 2D-1-2D short continuation pattern.

    3-bar pattern: 2D down, inside bar, 2D down (continuing downtrend).

    Returns:
        Tuple of (DataFrame, description, metadata)
    """
    base_time = datetime(2024, 1, 1, 9, 30)

    # Build downtrend
    bars = []
    bars.append({"open": 105.0, "high": 110.0, "low": 100.0, "close": 102.0})
    bars.append({"open": 100.0, "high": 102.0, "low": 95.0, "close": 96.0})
    bars.append({"open": 95.0, "high": 96.0, "low": 90.0, "close": 91.0})
    bars.append({"open": 91.0, "high": 92.0, "low": 87.0, "close": 88.0})
    bars.append({"open": 88.0, "high": 89.0, "low": 84.0, "close": 85.0})

    # 2D-1-2D continuation
    prev_bar = bars[-1]
    bars.append(create_2d_bar(prev_bar["high"], prev_bar["low"], break_amount=3.0, continuity="red"))
    prev_bar = bars[-1]
    bars.append(create_inside_bar(prev_bar["high"], prev_bar["low"], continuity="red"))
    prev_bar = bars[-1]
    bars.append(create_2d_bar(prev_bar["high"], prev_bar["low"], break_amount=3.0, continuity="red"))

    # Boundary bars
    prev_bar = bars[-1]
    bars.append(create_2d_bar(prev_bar["high"], prev_bar["low"], break_amount=2.0, continuity="red"))

    data = {
        "timestamp": [base_time + timedelta(minutes=i) for i in range(len(bars))],
        "symbol": ["TEST"] * len(bars),
        "open": [b["open"] for b in bars],
        "high": [b["high"] for b in bars],
        "low": [b["low"] for b in bars],
        "close": [b["close"] for b in bars],
        "volume": [1000.0] * len(bars),
    }

    df = pl.DataFrame(data)
    description = "2D-1-2D short continuation: down, inside bar, continue down"
    metadata = {
        "legend_text": "3-bar continuation:\nBar N: 2D\nBar N+1: Inside bar\nBar N+2: 2D (continues)",
        "signal_bar": len(bars) - 2,
    }

    return df, description, metadata


# =============================================================================
# PATTERN GENERATORS - Context-Aware (2 patterns)
# =============================================================================


def generate_3_2u_context_reversal() -> tuple[pl.DataFrame, str, dict[str, Any]]:
    """
    Generate 3-2U context-aware reversal (long bias).

    2-bar pattern: Outside bar (3) followed by 2U with opposite continuity.
    Context-dependent reversal based on prior trend.

    Returns:
        Tuple of (DataFrame, description, metadata)
    """
    base_time = datetime(2024, 1, 1, 9, 30)

    # Build downtrend to create context (all highs unique within CSV for target ladder)
    bars = []
    bars.append({"open": 105.0, "high": 110.0, "low": 100.0, "close": 102.0})
    bars.append({"open": 100.0, "high": 102.0, "low": 95.0, "close": 97.0})
    bars.append({"open": 96.0, "high": 98.0, "low": 90.0, "close": 91.0})
    bars.append({"open": 91.0, "high": 92.0, "low": 87.0, "close": 88.0})
    bars.append({"open": 88.0, "high": 89.0, "low": 84.0, "close": 85.0})

    # 3-2U context reversal - outside bar breaks down and up
    bars.append({"open": 86.0, "high": 93.0, "low": 81.0, "close": 82.0})

    # 2U trigger bar with opposite (green) continuity
    bars.append({"open": 83.0, "high": 106.0, "low": 82.5, "close": 104.0})

    # Boundary bar
    bars.append({"open": 105.0, "high": 108.0, "low": 103.0, "close": 107.0})

    data = {
        "timestamp": [base_time + timedelta(minutes=i) for i in range(len(bars))],
        "symbol": ["TEST"] * len(bars),
        "open": [b["open"] for b in bars],
        "high": [b["high"] for b in bars],
        "low": [b["low"] for b in bars],
        "close": [b["close"] for b in bars],
        "volume": [1000.0] * len(bars),
    }

    df = pl.DataFrame(data)
    description = "3-2U context-aware reversal: outside bar then 2U with opposite continuity"
    metadata = {
        "legend_text": "2-bar context reversal:\nBar N: 3 (outside)\nBar N+1: 2U (opposite continuity)",
        "signal_bar": len(bars) - 2,
    }

    return df, description, metadata


def generate_3_2d_context_reversal() -> tuple[pl.DataFrame, str, dict[str, Any]]:
    """
    Generate 3-2D context-aware reversal (short bias).

    2-bar pattern: Outside bar (3) followed by 2D with opposite continuity.
    Context-dependent reversal based on prior trend.

    Returns:
        Tuple of (DataFrame, description, metadata)
    """
    base_time = datetime(2024, 1, 1, 9, 30)

    # Build uptrend to create context (all lows unique within CSV for target ladder)
    bars = []
    bars.append({"open": 95.0, "high": 100.0, "low": 90.0, "close": 98.0})
    bars.append({"open": 100.0, "high": 105.0, "low": 98.0, "close": 103.0})
    bars.append({"open": 102.0, "high": 110.0, "low": 101.0, "close": 109.0})
    bars.append({"open": 109.0, "high": 113.0, "low": 108.0, "close": 112.0})
    bars.append({"open": 112.0, "high": 116.0, "low": 111.0, "close": 115.0})

    # 3-2D context reversal - outside bar breaks up and down
    bars.append({"open": 114.0, "high": 119.0, "low": 107.0, "close": 118.0})

    # 2D trigger bar with opposite (red) continuity
    bars.append({"open": 117.0, "high": 117.5, "low": 94.0, "close": 96.0})

    # Boundary bar
    bars.append({"open": 95.0, "high": 97.0, "low": 92.0, "close": 93.0})

    data = {
        "timestamp": [base_time + timedelta(minutes=i) for i in range(len(bars))],
        "symbol": ["TEST"] * len(bars),
        "open": [b["open"] for b in bars],
        "high": [b["high"] for b in bars],
        "low": [b["low"] for b in bars],
        "close": [b["close"] for b in bars],
        "volume": [1000.0] * len(bars),
    }

    df = pl.DataFrame(data)
    description = "3-2D context-aware reversal: outside bar then 2D with opposite continuity"
    metadata = {
        "legend_text": "2-bar context reversal:\nBar N: 3 (outside)\nBar N+1: 2D (opposite continuity)",
        "signal_bar": len(bars) - 2,
    }

    return df, description, metadata


# =============================================================================
# COMPARISON GRID VISUALIZATION
# =============================================================================


def create_comparison_grid(results: dict[str, dict[str, Any]]) -> None:
    """
    Create a 4×4 comparison grid showing all 16 signal patterns.

    Args:
        results: Dictionary of signal results from main() containing
                raw_df, metadata, and file paths for each signal
    """
    # Set up 4×4 grid
    fig, axes = plt.subplots(4, 4, figsize=(24, 18))
    fig.suptitle("TheStrat Comprehensive Signal Comparison", fontsize=20, fontweight="bold")

    # Flatten axes for easier iteration
    axes_flat = axes.flatten()

    # Define signal order (matching pattern_generators OrderedDict)
    signal_order = [
        "2D-2U",
        "1-2D-2U",
        "2D-1-2U",
        "3-1-2U",
        "3-2D-2U",
        "2U-2D",
        "1-2U-2D",
        "2U-1-2D",
        "3-1-2D",
        "3-2U-2D",
        "2U-2U",
        "2U-1-2U",
        "2D-2D",
        "2D-1-2D",
        "3-2U",
        "3-2D",
    ]

    # Plot each signal in the grid
    for idx, signal_name in enumerate(signal_order):
        ax = axes_flat[idx]
        result = results[signal_name]
        raw_df = result["raw_df"]
        metadata = result["metadata"]

        # Extract OHLC data
        timestamps = raw_df["timestamp"].to_list()
        opens = raw_df["open"].to_list()
        highs = raw_df["high"].to_list()
        lows = raw_df["low"].to_list()
        closes = raw_df["close"].to_list()

        # Plot candlesticks (simplified for grid view)
        for i, (_ts, o, h, low_price, c) in enumerate(zip(timestamps, opens, highs, lows, closes, strict=False)):
            # Determine color
            color = "green" if c >= o else "red"

            # Draw high-low line (wick)
            ax.plot([i, i], [low_price, h], color="black", linewidth=0.5, zorder=1)

            # Draw open-close body
            body_height = abs(c - o)
            body_bottom = min(o, c)
            body = Rectangle(
                (i - 0.3, body_bottom),
                0.6,
                body_height if body_height > 0 else 0.1,
                facecolor=color,
                edgecolor="black",
                linewidth=0.5,
                alpha=0.7,
                zorder=2,
            )
            ax.add_patch(body)

        # Styling
        ax.set_xlim(-0.5, len(timestamps) - 0.5)
        y_min = min(lows) * 0.98
        y_max = max(highs) * 1.02
        ax.set_ylim(y_min, y_max)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

        # Title with signal info
        category = metadata.get("category", "unknown")
        bias = metadata.get("bias", "unknown")
        title_color = "darkgreen" if bias == "long" else "darkred" if bias == "short" else "black"

        ax.set_title(
            f"{signal_name}\n{category} · {bias}",
            fontsize=9,
            fontweight="bold",
            color=title_color,
            pad=5,
        )

        # Add subtle border
        for spine in ax.spines.values():
            spine.set_edgecolor("gray")
            spine.set_linewidth(1)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    comparison_grid_path = OUTPUT_CHARTS / "signal_comparison_grid.png"
    plt.savefig(str(comparison_grid_path), dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main():
    """
    Main execution - generates all 16 signal patterns deterministically.

    For each pattern:
    1. Generate raw OHLC market data (deterministic)
    2. Process through indicators pipeline
    3. Verify signal detection
    4. Save market data CSV
    5. Save indicators CSV
    6. Create and save chart
    """
    print("=" * 100)
    print("THESTRAT COMPREHENSIVE SIGNAL GENERATION")
    print("=" * 100)
    print("Generating deterministic test data for all 16 signals...")
    print()

    # Pattern registry - OrderedDict for reproducible iteration
    pattern_generators = OrderedDict(
        [
            # Reversal Long (5 patterns)
            ("2D-2U", generate_2d_2u_reversal),
            ("1-2D-2U", generate_1_2d_2u_reversal),
            ("2D-1-2U", generate_2d_1_2u_reversal),
            ("3-1-2U", generate_3_1_2u_reversal),
            ("3-2D-2U", generate_3_2d_2u_reversal),
            # Reversal Short (5 patterns)
            ("2U-2D", generate_2u_2d_reversal),
            ("1-2U-2D", generate_1_2u_2d_reversal),
            ("2U-1-2D", generate_2u_1_2d_reversal),
            ("3-1-2D", generate_3_1_2d_reversal),
            ("3-2U-2D", generate_3_2u_2d_reversal),
            # Continuation Long (2 patterns)
            ("2U-2U", generate_2u_2u_continuation),
            ("2U-1-2U", generate_2u_1_2u_continuation),
            # Continuation Short (2 patterns)
            ("2D-2D", generate_2d_2d_continuation),
            ("2D-1-2D", generate_2d_1_2d_continuation),
            # Context-Aware (2 patterns)
            ("3-2U", generate_3_2u_context_reversal),
            ("3-2D", generate_3_2d_context_reversal),
        ]
    )

    # Create fixed configuration
    config = create_factory_config()
    factory = Factory.create_all(config)

    results = OrderedDict()

    # Generate and process each pattern
    for signal_name, generator_func in pattern_generators.items():
        print(f"{'=' * 100}")
        print(f"Generating {signal_name}")
        print("=" * 100)

        # Generate deterministic raw OHLC data
        raw_df, description, metadata = generator_func()
        print(f"  Description: {description}")
        print(f"  Bars: {len(raw_df)}")

        # Process through indicators pipeline
        indicators_df = factory["indicators"].process(raw_df)

        # Verify signal was detected
        signal_rows = indicators_df.filter(pl.col("signal") == signal_name)
        if len(signal_rows) == 0:
            raise AssertionError(f"Signal {signal_name} not detected in indicators output!")

        print(f"  ✓ Signal detected at bar(s): {signal_rows.select('timestamp').to_series().to_list()}")

        # Generate file paths in subdirectories
        base_filename = f"signal_{signal_name.lower().replace('-', '_')}"
        market_csv = OUTPUT_MARKET / f"{base_filename}_market.csv"
        indicators_csv = OUTPUT_INDICATORS / f"{base_filename}_indicators.csv"
        chart_png = OUTPUT_CHARTS / f"{base_filename}.png"

        # Save market data CSV
        raw_df.write_csv(str(market_csv))
        print(f"  ✓ Market CSV: {market_csv}")

        # Save indicators CSV (convert list columns to string for CSV compatibility)
        # Check if target_prices is actually a list type (not null) before attempting list operations
        if indicators_df.schema["target_prices"] == pl.List(pl.Float64):
            indicators_csv_df = indicators_df.with_columns(
                pl.when(pl.col("target_prices").is_not_null())
                .then(pl.col("target_prices").list.eval(pl.element().cast(pl.String)).list.join(", "))
                .otherwise(pl.lit(None))
                .alias("target_prices")
            )
        else:
            # Column is entirely null, no conversion needed
            indicators_csv_df = indicators_df
        indicators_csv_df.write_csv(str(indicators_csv))
        print(f"  ✓ Indicators CSV: {indicators_csv}")

        # Create chart
        create_signal_chart(raw_df, signal_name, metadata, str(chart_png))
        print(f"  ✓ Chart: {chart_png}")

        # Store results
        results[signal_name] = {
            "raw_df": raw_df,
            "indicators_df": indicators_df,
            "metadata": metadata,
            "market_csv": market_csv,
            "indicators_csv": indicators_csv,
            "chart_png": chart_png,
        }

        print()

    # Create comparison grid
    print()
    print("=" * 100)
    print("Creating comparison grid...")
    print("=" * 100)
    create_comparison_grid(results)
    print(f"✓ Comparison grid: {OUTPUT_CHARTS / 'signal_comparison_grid.png'}")

    # Summary
    print()
    print("=" * 100)
    print("GENERATION COMPLETE")
    print("=" * 100)
    print(f"Total signals generated: {len(results)}")
    print(f"Total files created: {len(results) * 3 + 1}")  # 3 files per signal + comparison grid
    print()
    print("Output files:")
    for signal_name, result in results.items():
        print(f"  {signal_name}:")
        print(f"    - {result['market_csv']}")
        print(f"    - {result['indicators_csv']}")
        print(f"    - {result['chart_png']}")
    print()
    print("Additional files:")
    print(f"  - {OUTPUT_CHARTS / 'signal_comparison_grid.png'}")
    print()
    print("✓ All outputs are deterministic - identical across runs")
    print("✓ Use these CSVs for regression testing")


if __name__ == "__main__":
    main()
