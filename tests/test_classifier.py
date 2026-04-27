"""Tests for pure Strat classification functions."""

from datetime import UTC, datetime

import polars as pl
import pytest

from thestrat.classifier import (
    classify_bar,
    classify_bars_df,
    classify_color,
    classify_scenario,
)
from thestrat.types import Color, Scenario

# ---------------------------------------------------------------------------
# classify_color
# ---------------------------------------------------------------------------


def test_color_green():
    assert classify_color(100.0, 105.0) == Color.GREEN


def test_color_red():
    assert classify_color(105.0, 100.0) == Color.RED


def test_color_neutral():
    assert classify_color(100.0, 100.0) == Color.NEUTRAL


# ---------------------------------------------------------------------------
# classify_scenario
# ---------------------------------------------------------------------------


def test_scenario_inside_bar():
    result = classify_scenario(curr_high=104.0, curr_low=101.0, prev_high=105.0, prev_low=100.0)
    assert result == Scenario.ONE


def test_scenario_two_up():
    result = classify_scenario(curr_high=106.0, curr_low=101.0, prev_high=105.0, prev_low=100.0)
    assert result == Scenario.TWO_UP


def test_scenario_two_down():
    result = classify_scenario(curr_high=104.0, curr_low=99.0, prev_high=105.0, prev_low=100.0)
    assert result == Scenario.TWO_DOWN


def test_scenario_outside_bar():
    result = classify_scenario(curr_high=106.0, curr_low=99.0, prev_high=105.0, prev_low=100.0)
    assert result == Scenario.THREE


def test_scenario_equal_high_is_not_takeout():
    result = classify_scenario(curr_high=105.0, curr_low=101.0, prev_high=105.0, prev_low=100.0)
    assert result == Scenario.ONE


def test_scenario_equal_low_is_not_takeout():
    result = classify_scenario(curr_high=104.0, curr_low=100.0, prev_high=105.0, prev_low=100.0)
    assert result == Scenario.ONE


def test_scenario_equal_both_is_inside():
    result = classify_scenario(curr_high=105.0, curr_low=100.0, prev_high=105.0, prev_low=100.0)
    assert result == Scenario.ONE


def test_scenario_gap_up_inside():
    result = classify_scenario(curr_high=104.0, curr_low=102.0, prev_high=105.0, prev_low=100.0)
    assert result == Scenario.ONE


def test_scenario_gap_above_prior_high():
    result = classify_scenario(curr_high=110.0, curr_low=106.0, prev_high=105.0, prev_low=100.0)
    assert result == Scenario.TWO_UP


# ---------------------------------------------------------------------------
# classify_bar
# ---------------------------------------------------------------------------


def _bar(open_: float, high: float, low: float, close: float) -> dict:
    return {
        "symbol": "SPY",
        "timestamp": datetime(2026, 4, 7, tzinfo=UTC),
        "timeframe": "1d",
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": 1000.0,
    }


def test_classify_bar_first_bar():
    result = classify_bar(_bar(100.0, 105.0, 99.0, 104.0), prior=None)
    assert result["scenario0"] is None
    assert result["color0"] == Color.GREEN
    assert result["in_force"] is None


def test_classify_bar_2u_green():
    prior = _bar(100.0, 105.0, 99.0, 104.0)
    curr = _bar(104.0, 107.0, 100.0, 106.0)
    result = classify_bar(curr, prior=prior)
    assert result["scenario0"] == Scenario.TWO_UP
    assert result["color0"] == Color.GREEN
    assert result["in_force"] is True


def test_classify_bar_2u_red():
    prior = _bar(100.0, 105.0, 99.0, 104.0)
    curr = _bar(107.0, 108.0, 100.0, 103.0)
    result = classify_bar(curr, prior=prior)
    assert result["scenario0"] == Scenario.TWO_UP
    assert result["color0"] == Color.RED
    assert result["in_force"] is False


def test_classify_bar_2d_green():
    prior = _bar(100.0, 105.0, 99.0, 104.0)
    curr = _bar(97.0, 104.0, 96.0, 102.0)
    result = classify_bar(curr, prior=prior)
    assert result["scenario0"] == Scenario.TWO_DOWN
    assert result["color0"] == Color.GREEN
    assert result["in_force"] is False


def test_classify_bar_2d_red():
    prior = _bar(100.0, 105.0, 99.0, 104.0)
    curr = _bar(102.0, 104.0, 97.0, 98.0)
    result = classify_bar(curr, prior=prior)
    assert result["scenario0"] == Scenario.TWO_DOWN
    assert result["color0"] == Color.RED
    assert result["in_force"] is True


def test_classify_bar_scenario_1():
    prior = _bar(100.0, 105.0, 99.0, 104.0)
    curr = _bar(101.0, 103.0, 100.0, 100.5)
    result = classify_bar(curr, prior=prior)
    assert result["scenario0"] == Scenario.ONE
    assert result["in_force"] is False


def test_classify_bar_scenario_3():
    prior = _bar(100.0, 105.0, 99.0, 104.0)
    curr = _bar(98.0, 107.0, 97.0, 101.0)
    result = classify_bar(curr, prior=prior)
    assert result["scenario0"] == Scenario.THREE
    assert result["in_force"] is False


def test_classify_bar_neutral():
    prior = _bar(100.0, 105.0, 99.0, 104.0)
    curr = _bar(103.0, 106.0, 100.0, 103.0)
    result = classify_bar(curr, prior=prior)
    assert result["scenario0"] == Scenario.TWO_UP
    assert result["color0"] == Color.NEUTRAL


# ---------------------------------------------------------------------------
# classify_bars_df (vectorized)
# ---------------------------------------------------------------------------


def test_classify_bars_df_scenarios():
    """Vectorized classification produces correct scenarios."""
    df = pl.DataFrame(
        {
            "timestamp": [
                datetime(2026, 4, 1, tzinfo=UTC),
                datetime(2026, 4, 2, tzinfo=UTC),
                datetime(2026, 4, 3, tzinfo=UTC),
                datetime(2026, 4, 4, tzinfo=UTC),
                datetime(2026, 4, 5, tzinfo=UTC),
            ],
            "open": [100.0, 104.0, 107.0, 98.0, 101.0],
            "high": [105.0, 108.0, 106.0, 109.0, 103.0],
            "low": [99.0, 103.0, 104.0, 97.0, 100.0],
            "close": [104.0, 107.0, 105.0, 101.0, 100.5],
            "volume": [1000.0] * 5,
        }
    )
    result = classify_bars_df(df)

    # First bar: no prior -> scenario is null
    assert result[0, "scenario0"] is None
    # Bar 2: high 108 > 105, low 103 > 99 -> 2U
    assert result[1, "scenario0"] == "2U"
    # Bar 3: high 106 < 108, low 104 > 103 -> 1
    assert result[2, "scenario0"] == "1"
    # Bar 4: high 109 > 106, low 97 < 104 -> 3
    assert result[3, "scenario0"] == "3"
    # Bar 5: high 103 < 109, low 100 > 97 -> 1
    assert result[4, "scenario0"] == "1"


def test_classify_bars_df_color():
    """Vectorized color classification."""
    df = pl.DataFrame(
        {
            "timestamp": [datetime(2026, 4, i, tzinfo=UTC) for i in range(1, 4)],
            "open": [100.0, 105.0, 100.0],
            "high": [106.0, 106.0, 106.0],
            "low": [99.0, 99.0, 99.0],
            "close": [105.0, 100.0, 100.0],  # green, red, neutral
            "volume": [1000.0] * 3,
        }
    )
    result = classify_bars_df(df)
    assert result[0, "color0"] == "green"
    assert result[1, "color0"] == "red"
    assert result[2, "color0"] == "neutral"


def test_classify_bars_df_equal_high_not_takeout():
    """Matching prior high exactly is NOT taking out (strict inequality)."""
    df = pl.DataFrame(
        {
            "timestamp": [datetime(2026, 4, 1, tzinfo=UTC), datetime(2026, 4, 2, tzinfo=UTC)],
            "open": [100.0, 101.0],
            "high": [105.0, 105.0],  # equal high
            "low": [99.0, 100.0],
            "close": [104.0, 103.0],
            "volume": [1000.0] * 2,
        }
    )
    result = classify_bars_df(df)
    assert result[1, "scenario0"] == "1"  # inside bar, not 2U


def test_classify_bars_df_missing_columns():
    """classify_bars_df raises ValueError when required columns are missing."""
    df = pl.DataFrame(
        {
            "timestamp": [datetime(2026, 4, 1, tzinfo=UTC)],
            "open": [100.0],
            "high": [105.0],
        }
    )
    with pytest.raises(ValueError, match="missing"):
        classify_bars_df(df)


# ---------------------------------------------------------------------------
# classify_bars_df — history shifts
# ---------------------------------------------------------------------------


def test_classify_bars_df_history_shift():
    """Verify scenario1/color1 etc. match prior bar's values."""
    df = pl.DataFrame(
        {
            "timestamp": [
                datetime(2026, 4, 1, tzinfo=UTC),
                datetime(2026, 4, 2, tzinfo=UTC),
                datetime(2026, 4, 3, tzinfo=UTC),
                datetime(2026, 4, 4, tzinfo=UTC),
                datetime(2026, 4, 5, tzinfo=UTC),
            ],
            "open": [100.0, 104.0, 107.0, 98.0, 101.0],
            "high": [105.0, 108.0, 106.0, 109.0, 103.0],
            "low": [99.0, 103.0, 104.0, 97.0, 100.0],
            "close": [104.0, 107.0, 105.0, 101.0, 100.5],
            "volume": [1000.0] * 5,
        }
    )
    result = classify_bars_df(df)

    # Bar 0: no history
    assert result[0, "scenario1"] is None
    assert result[0, "scenario2"] is None
    assert result[0, "scenario3"] is None
    assert result[0, "color1"] is None

    # Bar 1: scenario1 = bar0's scenario0 (None), color1 = bar0's color0 (green)
    assert result[1, "scenario1"] is None
    assert result[1, "color1"] == "green"

    # Bar 2: scenario1 = bar1's scenario0 (2U), color1 = bar1's color0 (green)
    assert result[2, "scenario1"] == "2U"
    assert result[2, "color1"] == "green"

    # Bar 4: full history
    assert result[4, "scenario1"] == "3"
    assert result[4, "scenario2"] == "1"
    assert result[4, "scenario3"] == "2U"
    # bar3: open=98, close=101 -> 101 > 98 -> green
    # bar2: open=107, close=105 -> 105 < 107 -> red
    # bar1: open=104, close=107 -> 107 > 104 -> green
    assert result[4, "color1"] == "green"  # bar3 is green (101 > 98)
    assert result[4, "color2"] == "red"  # bar2 is red (105 < 107)
    assert result[4, "color3"] == "green"  # bar1 is green (107 > 104)


# ---------------------------------------------------------------------------
# classify_bars_df — shape
# ---------------------------------------------------------------------------


def _shape_df(open_: float, close: float, low: float = 100.0, high: float = 110.0):
    """Build a 2-bar DataFrame for shape testing (first bar is a dummy)."""
    return pl.DataFrame(
        {
            "timestamp": [datetime(2026, 4, 1, tzinfo=UTC), datetime(2026, 4, 2, tzinfo=UTC)],
            "open": [100.0, open_],
            "high": [110.0, high],
            "low": [100.0, low],
            "close": [105.0, close],
            "volume": [1000.0, 1000.0],
        }
    )


def test_classify_bars_df_shape_hammer():
    """Body in top 33% of range -> hammer (green bar)."""
    result = classify_bars_df(_shape_df(open_=108.0, close=109.0))
    assert result[1, "shape"] == "hammer"


def test_classify_bars_df_shape_hammer_red():
    """Red bar with body in top 33% -> still hammer."""
    result = classify_bars_df(_shape_df(open_=109.0, close=108.0))
    assert result[1, "shape"] == "hammer"


def test_classify_bars_df_shape_shooter():
    """Body in bottom 33% of range -> shooter (red bar)."""
    result = classify_bars_df(_shape_df(open_=102.0, close=101.0))
    assert result[1, "shape"] == "shooter"


def test_classify_bars_df_shape_shooter_green():
    """Green bar with body in bottom 33% -> still shooter."""
    result = classify_bars_df(_shape_df(open_=101.0, close=102.0))
    assert result[1, "shape"] == "shooter"


def test_classify_bars_df_shape_null():
    """Body in middle of range -> None."""
    result = classify_bars_df(_shape_df(open_=104.0, close=106.0))
    assert result[1, "shape"] is None


def test_classify_bars_df_shape_zero_range():
    """Zero-range bar (high == low) -> None."""
    result = classify_bars_df(_shape_df(open_=100.0, close=100.0, low=100.0, high=100.0))
    assert result[1, "shape"] is None


# ---------------------------------------------------------------------------
# classify_bars_df — in_force
# ---------------------------------------------------------------------------


def _in_force_df(
    prior_open: float,
    prior_high: float,
    prior_low: float,
    prior_close: float,
    curr_open: float,
    curr_high: float,
    curr_low: float,
    curr_close: float,
):
    """Build a 2-bar DataFrame for in_force testing."""
    return pl.DataFrame(
        {
            "timestamp": [datetime(2026, 4, 1, tzinfo=UTC), datetime(2026, 4, 2, tzinfo=UTC)],
            "open": [prior_open, curr_open],
            "high": [prior_high, curr_high],
            "low": [prior_low, curr_low],
            "close": [prior_close, curr_close],
            "volume": [1000.0, 1000.0],
        }
    )


def test_classify_bars_df_in_force_2u():
    """2U with close > prior high -> in_force True."""
    # Prior: high=105, Current: high=108 (>105 -> 2U), close=107 > 105
    df = _in_force_df(100, 105, 99, 104, 104, 108, 100, 107)
    result = classify_bars_df(df)
    assert result[1, "scenario0"] == "2U"
    assert result[1, "in_force"] is True


def test_classify_bars_df_in_force_2u_not_past():
    """2U with close < prior high -> in_force False."""
    # Prior: high=105, Current: high=108 (>105 -> 2U), close=103 < 105
    df = _in_force_df(100, 105, 99, 104, 104, 108, 100, 103)
    result = classify_bars_df(df)
    assert result[1, "scenario0"] == "2U"
    assert result[1, "in_force"] is False


def test_classify_bars_df_in_force_2d():
    """2D with close < prior low -> in_force True."""
    # Prior: low=99, Current: low=97 (<99 -> 2D), close=98 < 99
    df = _in_force_df(100, 105, 99, 104, 102, 104, 97, 98)
    result = classify_bars_df(df)
    assert result[1, "scenario0"] == "2D"
    assert result[1, "in_force"] is True


def test_classify_bars_df_in_force_inside():
    """Scenario 1 (inside) -> in_force False."""
    # Prior: h=105, l=99, Current: h=104, l=100 -> inside
    df = _in_force_df(100, 105, 99, 104, 101, 104, 100, 103)
    result = classify_bars_df(df)
    assert result[1, "scenario0"] == "1"
    assert result[1, "in_force"] is False


def test_classify_bars_df_in_force_first_bar():
    """First bar -> in_force None."""
    df = pl.DataFrame(
        {
            "timestamp": [datetime(2026, 4, 1, tzinfo=UTC)],
            "open": [100.0],
            "high": [105.0],
            "low": [99.0],
            "close": [104.0],
            "volume": [1000.0],
        }
    )
    result = classify_bars_df(df)
    assert result[0, "in_force"] is None


def test_classify_bars_df_in_force_3_green():
    """Scenario 3 green, close > prior high -> in_force True."""
    # Prior: h=105, l=99. Current: h=108(>105), l=97(<99) -> 3, green (close=106>open=100)
    # close 106 > prior high 105 -> True
    df = _in_force_df(100, 105, 99, 104, 100, 108, 97, 106)
    result = classify_bars_df(df)
    assert result[1, "scenario0"] == "3"
    assert result[1, "color0"] == "green"
    assert result[1, "in_force"] is True


def test_classify_bars_df_in_force_3_red():
    """Scenario 3 red, close < prior low -> in_force True."""
    # Prior: h=105, l=99. Current: h=108(>105), l=97(<99) -> 3, red (close=98<open=104)
    # close 98 < prior low 99 -> True
    df = _in_force_df(100, 105, 99, 104, 104, 108, 97, 98)
    result = classify_bars_df(df)
    assert result[1, "scenario0"] == "3"
    assert result[1, "color0"] == "red"
    assert result[1, "in_force"] is True


def test_classify_bars_df_in_force_3_neutral():
    """Scenario 3 neutral (open == close) -> in_force False."""
    # Prior: h=105, l=99. Current: h=108(>105), l=97(<99) -> 3, neutral (close==open==102)
    # close 102 < prior high 105 AND close 102 > prior low 99 -> neither branch, False
    df = _in_force_df(100, 105, 99, 104, 102, 108, 97, 102)
    result = classify_bars_df(df)
    assert result[1, "scenario0"] == "3"
    assert result[1, "color0"] == "neutral"
    assert result[1, "in_force"] is False


def test_classify_bars_df_in_force_2d_not_past():
    """2D with close > prior low -> in_force False."""
    # Prior: low=99, Current: low=97 (<99 -> 2D), close=100 > 99
    df = _in_force_df(100, 105, 99, 104, 102, 104, 97, 100)
    result = classify_bars_df(df)
    assert result[1, "scenario0"] == "2D"
    assert result[1, "in_force"] is False


# ---------------------------------------------------------------------------
# classify_bars_multi_symbol
# ---------------------------------------------------------------------------


def test_classify_bars_multi_symbol_history_isolation():
    """History shifts are per-symbol — bars from one symbol don't bleed into another."""
    from thestrat.classifier import classify_bars_multi_symbol

    df = pl.DataFrame(
        {
            "symbol": ["SPY", "SPY", "AAPL", "AAPL"],
            "timestamp": [
                datetime(2026, 4, 1, tzinfo=UTC),
                datetime(2026, 4, 2, tzinfo=UTC),
                datetime(2026, 4, 1, tzinfo=UTC),
                datetime(2026, 4, 2, tzinfo=UTC),
            ],
            "open": [100.0, 104.0, 200.0, 195.0],
            "high": [105.0, 108.0, 210.0, 205.0],
            "low": [99.0, 103.0, 198.0, 190.0],
            "close": [104.0, 107.0, 208.0, 192.0],
            "volume": [1000.0] * 4,
        }
    ).sort("symbol", "timestamp")
    result = classify_bars_multi_symbol(df)

    # AAPL bar 2 (index depends on sort): scenario1 should be AAPL bar 1's scenario, not SPY's
    aapl = result.filter(pl.col("symbol") == "AAPL").sort("timestamp")
    spy = result.filter(pl.col("symbol") == "SPY").sort("timestamp")

    # SPY bar 1: color0 should be green (104>100), scenario0 should be None (first bar)
    assert spy[0, "color0"] == "green"
    assert spy[0, "scenario0"] is None
    # SPY bar 2: color1 should be SPY bar 1's color ("green"), not AAPL's
    assert spy[1, "color1"] == "green"

    # AAPL bar 1: color0 should be green (208>200), scenario0 should be None (first bar)
    assert aapl[0, "color0"] == "green"
    assert aapl[0, "scenario0"] is None
    # AAPL bar 2: color1 should be AAPL bar 1's color ("green"), not SPY bar 2's
    assert aapl[1, "color1"] == "green"
    # AAPL bar 2: scenario1 should be None (AAPL bar 1's scenario0 is None)
    assert aapl[1, "scenario1"] is None
