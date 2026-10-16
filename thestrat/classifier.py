"""Callable classification components, with no retained state."""

from collections.abc import Sequence
from datetime import UTC, datetime
from math import isfinite
from numbers import Real
from typing import Literal, overload

import polars as pl

from thestrat.frames import normalize
from thestrat.types import DEFAULT_TIMEFRAMES, BarDict, Color, CurrentBar, Scenario, Shape

Field = Literal["scenario", "color", "shape", "in_force"]
_FIELDS = ("scenario", "color", "shape", "in_force")
_ZONE = 0.33


def _numbers(values: dict) -> None:
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
            raise ValueError(f"{key} must be finite numeric")
    if "high" in values and values["high"] < values["low"]:
        raise ValueError("high must be >= low")
    if "reference_high" in values and values["reference_high"] < values["reference_low"]:
        raise ValueError("reference_high must be >= reference_low")
    if "high" in values and "open_price" in values:
        if (
            not values["low"]
            <= min(values["open_price"], values["close_price"])
            <= max(values["open_price"], values["close_price"])
            <= values["high"]
        ):
            raise ValueError("invalid OHLC range")


def _color(o: float, c: float) -> Color:
    return Color.GREEN if c > o else Color.RED if c < o else Color.NEUTRAL


def _scenario(h: float, low: float, rh: float, rl: float) -> Scenario:
    return (
        Scenario.THREE
        if h > rh and low < rl
        else Scenario.TWO_UP
        if h > rh
        else Scenario.TWO_DOWN
        if low < rl
        else Scenario.ONE
    )


def _shape(o: float, h: float, low: float, c: float) -> Shape | None:
    r = h - low
    if r == 0:
        return None
    if min(o, c) >= low + r * (1 - _ZONE):
        return Shape.HAMMER
    if max(o, c) <= low + r * _ZONE:
        return Shape.SHOOTER
    return None


def _force(o: float, h: float, low: float, c: float, rh: float, rl: float) -> bool:
    scenario, color = _scenario(h, low, rh, rl), _color(o, c)
    if scenario == Scenario.TWO_UP or (scenario == Scenario.THREE and color == Color.GREEN):
        return c > rh
    if scenario == Scenario.TWO_DOWN or (scenario == Scenario.THREE and color == Color.RED):
        return c < rl
    return False


def _bar_values(bar: dict) -> dict:
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    if not isinstance(bar, dict) or required - bar.keys():
        raise ValueError("bar missing required values")
    timestamp = bar["timestamp"]
    if not isinstance(timestamp, datetime):
        raise ValueError("timestamp must be datetime")
    timestamp = timestamp.replace(tzinfo=UTC) if timestamp.tzinfo is None else timestamp.astimezone(UTC)
    symbol, timeframe = bar.get("symbol"), bar.get("timeframe", "1min")
    if symbol is not None and not isinstance(symbol, str):
        raise ValueError("symbol must be String or null")
    if timeframe not in DEFAULT_TIMEFRAMES:
        raise ValueError("unsupported timeframe")
    if timeframe == "1min" and (timestamp.second or timestamp.microsecond):
        raise ValueError("source timestamp must be the exact start of its minute")
    _numbers(
        {
            "open_price": bar["open"],
            "high": bar["high"],
            "low": bar["low"],
            "close_price": bar["close"],
            "volume": bar["volume"],
        }
    )
    if bar["volume"] < 0:
        raise ValueError("volume must be nonnegative")
    return {
        "symbol": symbol,
        "timestamp": timestamp,
        "timeframe": timeframe,
        **{name: float(bar[name]) for name in ("open", "high", "low", "close", "volume")},
    }


class ScalarClassifier:
    @overload
    def __call__(self, *, kind: Literal["color"], open_price: float, close_price: float) -> Color: ...

    @overload
    def __call__(
        self, *, kind: Literal["scenario"], high: float, low: float, reference_high: float, reference_low: float
    ) -> Scenario: ...

    @overload
    def __call__(
        self, *, kind: Literal["shape"], open_price: float, high: float, low: float, close_price: float
    ) -> Shape | None: ...

    @overload
    def __call__(
        self,
        *,
        kind: Literal["in_force"],
        open_price: float,
        high: float,
        low: float,
        close_price: float,
        reference_high: float,
        reference_low: float,
    ) -> bool: ...

    @overload
    def __call__(self, *, kind: Literal["bar"], bar: BarDict, prior: BarDict | None = None) -> CurrentBar: ...

    def __call__(self, *, kind: str, **kwargs):
        if kind == "bar":
            if set(kwargs) - {"bar", "prior"} or "bar" not in kwargs:
                raise TypeError("bar requires bar and optional prior")
            bar = _bar_values(kwargs["bar"])
            prior = kwargs.get("prior")
            if prior is not None:
                prior = _bar_values(prior)
            o, h, low, c = (bar[n] for n in ("open", "high", "low", "close"))
            return {
                **{k: bar[k] for k in ("symbol", "timestamp", "timeframe", "open", "high", "low", "close", "volume")},
                "color0": _color(o, c),
                "shape": _shape(o, h, low, c),
                "scenario0": None if prior is None else _scenario(h, low, prior["high"], prior["low"]),
                "in_force": None if prior is None else _force(o, h, low, c, prior["high"], prior["low"]),
            }
        inputs = {
            "color": ("open_price", "close_price"),
            "scenario": ("high", "low", "reference_high", "reference_low"),
            "shape": ("open_price", "high", "low", "close_price"),
            "in_force": ("open_price", "high", "low", "close_price", "reference_high", "reference_low"),
        }
        if kind not in inputs:
            raise ValueError(f"unknown scalar kind: {kind}")
        if set(kwargs) != set(inputs[kind]):
            raise TypeError(f"{kind} requires exactly {inputs[kind]}")
        _numbers(kwargs)
        fn = {"color": _color, "scenario": _scenario, "shape": _shape, "in_force": _force}[kind]
        return fn(*(kwargs[k] for k in inputs[kind]))


class Classifier:
    def __init__(self) -> None:
        self.scalar = ScalarClassifier()

    def __call__(self, bars: pl.DataFrame, *, fields: Sequence[Field] | None = None, history: int = 3) -> pl.DataFrame:
        selected = _FIELDS if fields is None else tuple(fields)
        if isinstance(fields, str) or len(set(selected)) != len(selected) or any(f not in _FIELDS for f in selected):
            raise ValueError("fields must contain unique supported field names")
        if isinstance(history, bool) or not isinstance(history, int) or history < 0:
            raise ValueError("history must be a nonnegative integer")
        df = normalize(bars, metadata=True).sort("symbol", "timeframe", "timestamp", nulls_last=False)
        group = ["symbol", "timeframe"]
        ph = pl.col("high").shift(1).over(group)
        previous_low = pl.col("low").shift(1).over(group)
        color = (
            pl.when(pl.col("close") > pl.col("open"))
            .then(pl.lit("green"))
            .when(pl.col("close") < pl.col("open"))
            .then(pl.lit("red"))
            .otherwise(pl.lit("neutral"))
        )
        scenario = (
            pl.when(ph.is_null())
            .then(pl.lit(None, dtype=pl.String))
            .when((pl.col("high") > ph) & (pl.col("low") < previous_low))
            .then(pl.lit("3"))
            .when(pl.col("high") > ph)
            .then(pl.lit("2U"))
            .when(pl.col("low") < previous_low)
            .then(pl.lit("2D"))
            .otherwise(pl.lit("1"))
        )
        expressions = []
        if "color" in selected:
            expressions.extend(
                color.shift(i).over(group).alias(f"color{i}") if i else color.alias("color0")
                for i in range(history + 1)
            )
        if "scenario" in selected:
            # Window expressions cannot be nested; derive current scenario in a temporary frame first.
            temp = df.select(scenario.alias("__scenario"), *group)
            for i in range(history + 1):
                expressions.append(
                    temp.select(pl.col("__scenario").shift(i).over(group)).to_series().alias(f"scenario{i}")
                )
        if "shape" in selected:
            r = pl.col("high") - pl.col("low")
            expressions.append(
                pl.when(r == 0)
                .then(pl.lit(None, dtype=pl.String))
                .when(pl.min_horizontal("open", "close") >= pl.col("low") + r * (1 - _ZONE))
                .then(pl.lit("hammer"))
                .when(pl.max_horizontal("open", "close") <= pl.col("low") + r * _ZONE)
                .then(pl.lit("shooter"))
                .otherwise(pl.lit(None, dtype=pl.String))
                .alias("shape")
            )
        if "in_force" in selected:
            expressions.append(
                pl.when(ph.is_null())
                .then(pl.lit(None, dtype=pl.Boolean))
                .when((scenario == "2U") | ((scenario == "3") & (color == "green")))
                .then(pl.col("close") > ph)
                .when((scenario == "2D") | ((scenario == "3") & (color == "red")))
                .then(pl.col("close") < previous_low)
                .otherwise(pl.lit(False))
                .alias("in_force")
            )
        return df.with_columns(expressions)
