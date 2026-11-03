"""Extending higher-timeframe rows the caller already loaded with new source minutes."""

from math import isclose
from typing import Any

import polars as pl

from thestrat.frames import KEY, PRICES


def tag_prior(sources: pl.DataFrame, incoming: pl.DataFrame) -> pl.DataFrame:
    """Final source set; each row carries the prior snapshot of its key in ``__old_*`` columns."""
    kept = sources.join(incoming.select(KEY), on=KEY, how="anti", nulls_equal=True)
    final = pl.concat([kept, incoming]) if incoming.height else kept
    prior = sources.select("symbol", "timestamp", *(pl.col(n).alias(f"__old_{n}") for n in PRICES))
    return final.join(prior, on=["symbol", "timestamp"], how="left", nulls_equal=True)


def window_stats(mapped: pl.DataFrame, *, tracked: bool) -> pl.DataFrame:
    """Per window: aggregate of the final minutes (``new_*``) and, when tracked, of their prior snapshots (``old_*``)."""
    final = [
        pl.col("open").first().alias("new_open"),
        pl.col("high").max().alias("new_high"),
        pl.col("low").min().alias("new_low"),
        pl.col("close").last().alias("new_close"),
        pl.col("volume").sum().alias("new_volume"),
    ]
    grouped = mapped.sort("timestamp").group_by("symbol", "__window")
    if not tracked:
        return grouped.agg(*final)
    unchanged = pl.all_horizontal(*(pl.col(n) == pl.col(f"__old_{n}") for n in PRICES)).fill_null(False)
    return grouped.agg(
        *final,
        pl.col("__old_open").drop_nulls().first().alias("old_open"),
        pl.col("__old_high").max().alias("old_high"),
        pl.col("__old_low").min().alias("old_low"),
        pl.col("__old_close").drop_nulls().last().alias("old_close"),
        pl.col("__old_volume").sum().alias("old_volume"),
        pl.col("__old_open").count().alias("retained"),
        (~unchanged).any().alias("touched"),
    )


def extend_window(window: dict[str, Any]) -> dict[str, float]:
    """OHLCV for a window with a loaded row (``held_*``) that a submitted minute changed.

    Retained minutes that reproduce the loaded row are the whole window and are recalculated.
    Otherwise the new minutes are added to the loaded row.
    """
    held = {n: window[f"held_{n}"] for n in PRICES}
    if (
        window["retained"]
        and all(held[n] == window[f"old_{n}"] for n in ("open", "high", "low", "close"))
        and isclose(held["volume"], window["old_volume"], rel_tol=1e-9, abs_tol=1e-6)
    ):
        return {n: window[f"new_{n}"] for n in PRICES}
    return {
        "open": held["open"],
        "high": max(held["high"], window["new_high"]),
        "low": min(held["low"], window["new_low"]),
        "close": window["new_close"],
        "volume": held["volume"] + window["new_volume"] - window["old_volume"],
    }
