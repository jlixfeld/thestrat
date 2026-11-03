"""A loaded higher-timeframe row is extended by new minutes; the result equals one aggregate over all minutes."""

import random
from datetime import datetime, timedelta, timezone
from math import isclose

import polars as pl
import pytest

from thestrat import DEFAULT_TIMEFRAMES, TheStrat, session_for

START = datetime(2025, 1, 1, tzinfo=timezone.utc)
OHLCV = ["open", "high", "low", "close", "volume"]
KEY = ["symbol", "timeframe", "timestamp"]


def utc(*args: int) -> datetime:
    return datetime(*args, tzinfo=timezone.utc)


def bar(stamp: datetime, open_, high, low, close, volume) -> dict:
    return dict(symbol="SPY", timestamp=stamp, open=open_, high=high, low=low, close=close, volume=volume)


def frame(rows: list[dict]) -> pl.DataFrame:
    schema = {"symbol": pl.String, "timestamp": pl.Datetime("us", "UTC")} | dict.fromkeys(OHLCV, pl.Float64)
    return pl.DataFrame(rows, schema=schema)


def instance(preset: str, rows: pl.DataFrame | None = None, timeframes=DEFAULT_TIMEFRAMES) -> TheStrat:
    return TheStrat(session=session_for(preset), timeframes=timeframes, bars=rows)


def random_minutes(rng: random.Random) -> list[dict]:
    """Sparse random-walk minutes over 420 days, with whole days and weeks missing."""
    rows, price, day = [], 100.0, 0
    while day < 420:
        if rng.random() < 0.08:
            day += rng.randint(2, 40)
            continue
        for offset in sorted(rng.sample(range(13 * 60, 22 * 60), rng.randint(2, 5))):
            close = round(price + rng.gauss(0, 1), 2)
            high = round(max(price, close) + abs(rng.gauss(0, 0.6)), 2)
            low = round(min(price, close) - abs(rng.gauss(0, 0.6)), 2)
            stamp = START + timedelta(days=day, minutes=offset)
            rows.append(bar(stamp, round(price, 2), high, low, close, round(rng.uniform(0, 5000), 3)))
            price = close
        day += 1
    return rows


def keep_last_minute(strat: TheStrat) -> None:
    minutes = strat.bars.filter(pl.col("timeframe") == "1min")
    newest = minutes.group_by("symbol").agg(pl.col("timestamp").max().alias("newest"))
    kept = minutes.join(newest, on="symbol", nulls_equal=True).filter(pl.col("timestamp") == pl.col("newest"))
    strat.bars = pl.concat([kept.drop("newest"), strat.bars.filter(pl.col("timeframe") != "1min")])


def assert_same_higher_bars(actual: pl.DataFrame, expected: pl.DataFrame) -> None:
    actual, expected = (f.filter(pl.col("timeframe") != "1min").sort(KEY, nulls_last=False) for f in (actual, expected))
    assert actual.select(KEY).equals(expected.select(KEY))
    for name in OHLCV[:4]:
        assert actual[name].to_list() == expected[name].to_list(), name
    pairs = zip(actual["volume"], expected["volume"], strict=True)
    assert all(isclose(a, b, rel_tol=1e-9, abs_tol=1e-6) for a, b in pairs)


@pytest.mark.parametrize("preset", ["crypto", "equity_us"])
@pytest.mark.parametrize("scenario", ["append", "overlap", "replace"])
def test_split_then_extend_equals_one_aggregate(preset, scenario):
    """Aggregate the first part, prune, then send the rest. Replace keeps every minute, so it can retract."""
    rng = random.Random(7)
    rows = random_minutes(rng)
    cuts = sorted(rng.sample(range(1, len(rows)), 2))
    parts = [rows[a:b] for a, b in zip([0, *cuts], [*cuts, len(rows)], strict=True)]
    final = {r["timestamp"]: r for r in rows}
    strat, sent = instance(preset), []
    for part in parts:
        batch = {r["timestamp"]: r for r in part}
        if scenario == "overlap" and sent:
            batch = {sent[-1]["timestamp"]: sent[-1]} | batch
        if scenario == "replace" and sent:
            old = sent[-1]
            batch = {old["timestamp"]: old | {"high": old["high"] + 1, "close": old["close"] + 0.5}} | batch
            final[old["timestamp"]] = batch[old["timestamp"]]
        strat.aggregate(frame(list(batch.values())))
        sent = list(batch.values())
        if scenario != "replace":
            keep_last_minute(strat)
    expected = instance(preset, frame(sorted(final.values(), key=lambda r: r["timestamp"])))
    expected.aggregate()
    assert_same_higher_bars(strat.bars, expected.bars)


def loaded_month() -> TheStrat:
    last = bar(utc(2026, 1, 20, 12, 0), 104.0, 105.0, 103.5, 105.0, 10.0)
    month = bar(utc(2026, 1, 1), 100.0, 110.0, 90.0, 105.0, 1000.0)
    bars = frame([last]).with_columns(pl.lit("1min").alias("timeframe"))
    month = frame([month]).with_columns(pl.lit("1m").alias("timeframe"))
    return instance("crypto", pl.concat([bars, month.select(bars.columns)]), timeframes=["1min", "1m"])


def month(strat: TheStrat) -> list[float]:
    row = strat.bars.filter(pl.col("timeframe") == "1m").row(0, named=True)
    return [row[n] for n in OHLCV]


def test_loaded_month_keeps_its_open_and_low_when_new_minutes_arrive():
    strat = loaded_month()
    strat.aggregate(frame([bar(utc(2026, 1, 20, 12, i), 101.0, 102.0, 100.5, 101.5, 5.0) for i in range(1, 6)]))
    assert month(strat) == [100.0, 110.0, 90.0, 101.5, 1025.0]


def test_new_snapshot_of_the_forming_minute_replaces_its_volume_and_close():
    strat = loaded_month()
    strat.aggregate(frame([bar(utc(2026, 1, 20, 12, 0), 104.0, 120.0, 80.0, 106.0, 30.0)]))
    assert month(strat) == [100.0, 120.0, 80.0, 106.0, 1020.0]


def test_missed_days_and_rollover_build_new_windows_and_extend_the_week():
    monday = bar(utc(2026, 1, 5, 10), 50.0, 52.0, 49.0, 51.0, 7.0)
    thursday = bar(utc(2026, 1, 8, 10), 51.0, 60.0, 50.5, 58.0, 3.0)
    next_month = bar(utc(2026, 2, 2, 10), 58.0, 59.0, 57.0, 57.5, 2.0)
    strat = instance("crypto", frame([monday]))
    strat.aggregate()
    keep_last_minute(strat)
    strat.aggregate(frame([thursday]))
    keep_last_minute(strat)
    strat.aggregate(frame([next_month]))
    expected = instance("crypto", frame([monday, thursday, next_month]))
    expected.aggregate()
    assert_same_higher_bars(strat.bars, expected.bars)
    week = strat.bars.filter((pl.col("timeframe") == "1w") & (pl.col("timestamp") == utc(2026, 1, 5)))
    assert [week.row(0, named=True)[n] for n in OHLCV] == [50.0, 60.0, 49.0, 58.0, 10.0]
