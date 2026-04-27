"""Pure classification functions for Strat scenario analysis."""

import polars as pl

from thestrat.types import ClassifiedBar, Color, Scenario

SHAPE_BODY_ZONE = 0.33


def classify_bars_df(df: pl.DataFrame) -> pl.DataFrame:
    """Add scenario0-3, color0-3, shape, in_force columns using vectorized Polars ops.

    Expects a DataFrame sorted by timestamp for a SINGLE symbol+timeframe
    with at minimum open, high, low, close columns.
    """
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"classify_bars_df requires columns {required}, missing: {missing}")
    took_high = pl.col("high") > pl.col("high").shift(1)
    took_low = pl.col("low") < pl.col("low").shift(1)

    # Split into 3 with_columns passes: each pass depends on columns from the prior.
    df = df.with_columns(
        pl.when(pl.col("close") > pl.col("open"))
        .then(pl.lit("green"))
        .when(pl.col("close") < pl.col("open"))
        .then(pl.lit("red"))
        .otherwise(pl.lit("neutral"))
        .alias("color0"),
        pl.when(pl.col("high").shift(1).is_null())
        .then(pl.lit(None))
        .when(took_high & took_low)
        .then(pl.lit("3"))
        .when(took_high)
        .then(pl.lit("2U"))
        .when(took_low)
        .then(pl.lit("2D"))
        .otherwise(pl.lit("1"))
        .alias("scenario0"),
    )

    df = df.with_columns(
        pl.col("color0").shift(1).alias("color1"),
        pl.col("color0").shift(2).alias("color2"),
        pl.col("color0").shift(3).alias("color3"),
        pl.col("scenario0").shift(1).alias("scenario1"),
        pl.col("scenario0").shift(2).alias("scenario2"),
        pl.col("scenario0").shift(3).alias("scenario3"),
    )

    prior_high = pl.col("high").shift(1)
    prior_low = pl.col("low").shift(1)
    range_ = pl.col("high") - pl.col("low")
    body_top = pl.max_horizontal("open", "close")
    body_bottom = pl.min_horizontal("open", "close")

    df = df.with_columns(
        pl.when(range_ == 0)
        .then(pl.lit(None))
        .when(body_bottom >= pl.col("low") + range_ * (1 - SHAPE_BODY_ZONE))
        .then(pl.lit("hammer"))
        .when(body_top <= pl.col("low") + range_ * SHAPE_BODY_ZONE)
        .then(pl.lit("shooter"))
        .otherwise(pl.lit(None))
        .alias("shape"),
        pl.when(prior_high.is_null())
        .then(pl.lit(None).cast(pl.Boolean))
        .when(pl.col("scenario0") == pl.lit("1"))
        .then(pl.lit(False))
        .when(
            (pl.col("scenario0") == pl.lit("2U"))
            | ((pl.col("scenario0") == pl.lit("3")) & (pl.col("color0") == pl.lit("green")))
        )
        .then(pl.col("close") > prior_high)
        .when(
            (pl.col("scenario0") == pl.lit("2D"))
            | ((pl.col("scenario0") == pl.lit("3")) & (pl.col("color0") == pl.lit("red")))
        )
        .then(pl.col("close") < prior_low)
        .otherwise(pl.lit(False))
        .alias("in_force"),
    )

    return df


def classify_bars_multi_symbol(df: pl.DataFrame) -> pl.DataFrame:
    """Vectorized classification across multiple symbols.

    Uses `over("symbol")` for per-symbol shift operations. Input must have
    a "symbol" column. DataFrame must be sorted by (symbol, timestamp).
    """
    required = {"symbol", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"classify_bars_multi_symbol requires columns {required}, missing: {missing}")

    took_high = pl.col("high") > pl.col("high").shift(1).over("symbol")
    took_low = pl.col("low") < pl.col("low").shift(1).over("symbol")

    df = df.with_columns(
        pl.when(pl.col("close") > pl.col("open"))
        .then(pl.lit("green"))
        .when(pl.col("close") < pl.col("open"))
        .then(pl.lit("red"))
        .otherwise(pl.lit("neutral"))
        .alias("color0"),
        pl.when(pl.col("high").shift(1).over("symbol").is_null())
        .then(pl.lit(None))
        .when(took_high & took_low)
        .then(pl.lit("3"))
        .when(took_high)
        .then(pl.lit("2U"))
        .when(took_low)
        .then(pl.lit("2D"))
        .otherwise(pl.lit("1"))
        .alias("scenario0"),
    )

    df = df.with_columns(
        pl.col("color0").shift(1).over("symbol").alias("color1"),
        pl.col("color0").shift(2).over("symbol").alias("color2"),
        pl.col("color0").shift(3).over("symbol").alias("color3"),
        pl.col("scenario0").shift(1).over("symbol").alias("scenario1"),
        pl.col("scenario0").shift(2).over("symbol").alias("scenario2"),
        pl.col("scenario0").shift(3).over("symbol").alias("scenario3"),
    )

    prior_high = pl.col("high").shift(1).over("symbol")
    prior_low = pl.col("low").shift(1).over("symbol")
    range_ = pl.col("high") - pl.col("low")
    body_top = pl.max_horizontal("open", "close")
    body_bottom = pl.min_horizontal("open", "close")

    df = df.with_columns(
        pl.when(range_ == 0)
        .then(pl.lit(None))
        .when(body_bottom >= pl.col("low") + range_ * (1 - SHAPE_BODY_ZONE))
        .then(pl.lit("hammer"))
        .when(body_top <= pl.col("low") + range_ * SHAPE_BODY_ZONE)
        .then(pl.lit("shooter"))
        .otherwise(pl.lit(None))
        .alias("shape"),
        pl.when(prior_high.is_null())
        .then(pl.lit(None).cast(pl.Boolean))
        .when(pl.col("scenario0") == pl.lit("1"))
        .then(pl.lit(False))
        .when(
            (pl.col("scenario0") == pl.lit("2U"))
            | ((pl.col("scenario0") == pl.lit("3")) & (pl.col("color0") == pl.lit("green")))
        )
        .then(pl.col("close") > prior_high)
        .when(
            (pl.col("scenario0") == pl.lit("2D"))
            | ((pl.col("scenario0") == pl.lit("3")) & (pl.col("color0") == pl.lit("red")))
        )
        .then(pl.col("close") < prior_low)
        .otherwise(pl.lit(False))
        .alias("in_force"),
    )

    return df


def classify_color(open_price: float, close_price: float) -> Color:
    """Determine bar color from open and close prices."""
    if close_price > open_price:
        return Color.GREEN
    elif close_price < open_price:
        return Color.RED
    return Color.NEUTRAL


def classify_scenario(curr_high: float, curr_low: float, prev_high: float, prev_low: float) -> Scenario:
    """Classify the Strat scenario by comparing current bar range to prior bar range.

    Uses strict inequality — matching is NOT taking out.
    """
    took_high = curr_high > prev_high
    took_low = curr_low < prev_low

    if took_high and took_low:
        return Scenario.THREE
    elif took_high:
        return Scenario.TWO_UP
    elif took_low:
        return Scenario.TWO_DOWN
    return Scenario.ONE


def classify_bar(bar: dict, prior: dict | None) -> ClassifiedBar:
    """Classify a single bar given the prior bar (or None for first bar)."""
    color = classify_color(bar["open"], bar["close"])

    if prior is None:
        scenario = None
        in_force = None
    else:
        scenario = classify_scenario(bar["high"], bar["low"], prior["high"], prior["low"])
        if scenario == Scenario.ONE:
            in_force = False
        elif scenario == Scenario.TWO_UP or (scenario == Scenario.THREE and color == Color.GREEN):
            in_force = bar["close"] > prior["high"]
        elif scenario == Scenario.TWO_DOWN or (scenario == Scenario.THREE and color == Color.RED):
            in_force = bar["close"] < prior["low"]
        else:
            in_force = False

    range_ = bar["high"] - bar["low"]
    if range_ == 0:
        shape = None
    else:
        body_bottom = min(bar["open"], bar["close"])
        body_top = max(bar["open"], bar["close"])
        if body_bottom >= bar["low"] + range_ * (1 - SHAPE_BODY_ZONE):
            shape = "hammer"
        elif body_top <= bar["low"] + range_ * SHAPE_BODY_ZONE:
            shape = "shooter"
        else:
            shape = None

    return ClassifiedBar(
        symbol=bar["symbol"],
        timestamp=bar["timestamp"],
        timeframe=bar["timeframe"],
        open=bar["open"],
        high=bar["high"],
        low=bar["low"],
        close=bar["close"],
        volume=bar["volume"],
        scenario3=None,
        scenario2=None,
        scenario1=None,
        scenario0=scenario,
        color3=None,
        color2=None,
        color1=None,
        color0=color,
        shape=shape,
        in_force=in_force,
    )
