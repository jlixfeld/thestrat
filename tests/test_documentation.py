"""Every README step and displayed output is an executable contract."""

import re
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from thestrat import Color


def _render(value):
    if not isinstance(value, pl.DataFrame):
        return f"```text\n{value}\n```"

    def cell(item):
        if item is None:
            return "null"
        if isinstance(item, bool):
            return str(item).lower()
        if isinstance(item, datetime):
            return item.strftime("%Y-%m-%d %H:%M")
        return str(item)

    headers = [f"{name} (UTC)" if name == "timestamp" else name for name in value.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(cell(item) for item in row) + " |" for row in value.iter_rows())
    return "\n".join(lines)


def test_readme_steps_execute_and_every_output_matches():
    readme = (Path(__file__).parents[1] / "README.md").read_text()
    namespace = {}
    pattern = r"```python\n(?P<code>.*?)```|<!-- output: (?P<name>\w+) -->\s*(?P<output>.*?)\s*<!-- /output -->"
    checked = []
    pending_step = False
    for match in re.finditer(pattern, readme, re.S):
        if match["code"] is not None:
            assert not pending_step, "every Python step needs a visible output"
            exec(compile(match["code"], "README.md", "exec"), namespace)
            pending_step = True
        else:
            assert pending_step, "an output must follow its executable step"
            assert match["output"] == _render(namespace[match["name"]])
            checked.append(match["name"])
            pending_step = False
    assert not pending_step
    assert checked == [
        "input_preview",
        "aggregate_preview",
        "preview",
        "minute_history",
        "revised_preview",
        "updated_preview",
        "updated_label_preview",
        "color",
        "windows",
    ]

    # Independent worked expectations protect the examples from plausible but wrong output.
    assert namespace["input_preview"]["volume"].to_list() == [10.0, 20.0, 30.0, 40.0, 50.0]
    assert namespace["input_preview"]["timeframe"].to_list() == ["1min"] * 5
    assert namespace["aggregated"].height == 17
    aggregate = namespace["aggregate_preview"]
    assert aggregate.height == 13
    assert aggregate.filter(pl.col("timeframe") != "1min")["volume"].to_list() == [150.0] * 12
    latest = aggregate.filter(pl.col("timeframe") == "1min").row(0, named=True)
    assert [latest[name] for name in ["open", "high", "low", "close", "volume"]] == [103.0, 105.0, 102.0, 104.0, 50.0]
    history = namespace["minute_history"]
    assert history["scenario0"].to_list() == [None, "2U", "1", "2U", "2U"]
    assert history["color0"].to_list() == ["green", "green", "red", "green", "green"]
    assert history[-1, "scenario3"] == "2U"
    assert history[-1, "color3"] == "green"
    assert history[0, "scenario0"] is None and history[0, "color3"] is None
    assert namespace["revised_preview"][0, "volume"] == 70.0
    updated = namespace["updated_preview"].filter(pl.col("timeframe") == "4h").row(0, named=True)
    assert [updated[name] for name in ["open", "high", "low", "close", "volume"]] == [100.0, 106.0, 99.0, 105.0, 170.0]
    labels = namespace["updated_label_preview"]
    assert labels.filter(pl.col("timeframe") == "1min")[0, "in_force"] is True
    assert labels.filter(pl.col("timeframe") == "4h")[0, "in_force"] is None
    assert namespace["color"] == Color.GREEN
    assert namespace["windows"].rows() == [
        ("1h", datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc)),
        ("1h", datetime(2026, 1, 5, 15, 30, tzinfo=timezone.utc)),
        ("4h", datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc)),
        ("4h", datetime(2026, 1, 5, 18, 30, tzinfo=timezone.utc)),
    ]
