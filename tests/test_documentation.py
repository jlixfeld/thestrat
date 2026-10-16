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
        if isinstance(item, list):
            return "; ".join(f"[{cell(segment['start'])}, {cell(segment['end'])})" for segment in item)
        return str(item)

    headers = [
        f"{name} (UTC)" if name in {"timestamp", "input_timestamp", "window_start", "window_end"} else name
        for name in value.columns
    ]
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
        "recent_minutes",
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
    recent = namespace["recent_minutes"]
    assert recent.height == 3
    assert recent["scenario"].to_list() == ["1", "2U", "2U"]
    assert recent["color"].to_list() == ["red", "green", "green"]
    assert recent["in_force"].to_list() == [False, False, False]
    assert namespace["classified"].filter(pl.col("timeframe") == "1min")["scenario"].to_list() == [
        None,
        "2U",
        "1",
        "2U",
        "2U",
    ]
    assert namespace["revised_preview"][0, "volume"] == 70.0
    updated = namespace["updated_preview"].filter(pl.col("timeframe") == "4h").row(0, named=True)
    assert [updated[name] for name in ["open", "high", "low", "close", "volume"]] == [100.0, 106.0, 99.0, 105.0, 170.0]
    labels = namespace["updated_label_preview"]
    assert labels.filter(pl.col("timeframe") == "1min")[0, "in_force"] is True
    assert labels.filter(pl.col("timeframe") == "4h")[0, "in_force"] is None
    assert namespace["color"] == Color.GREEN
    windows = namespace["windows"]
    assert windows.height == 12
    assert windows["position"].to_list() == [-1, 0, 1] * 4
    assert (
        windows["input_timestamp"].to_list()
        == [datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc)] * 6 + [datetime(2026, 1, 5, 16, tzinfo=timezone.utc)] * 6
    )
    first_hour = windows.row(1, named=True)
    assert first_hour["window_start"] == datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc)
    assert first_hour["window_end"] == datetime(2026, 1, 5, 15, 30, tzinfo=timezone.utc)
    assert first_hour["segments"] == [{"start": first_hour["window_start"], "end": first_hour["window_end"]}]
    previous_four_hour = windows.row(3, named=True)
    assert previous_four_hour["window_start"] == datetime(2026, 1, 2, 18, 30, tzinfo=timezone.utc)
    assert previous_four_hour["window_end"] == datetime(2026, 1, 2, 21, tzinfo=timezone.utc)


def test_agent_documentation_index_is_discoverable_and_links_resolve():
    root = Path(__file__).parents[1].resolve()
    readme = (root / "README.md").read_text()
    assert re.search(r"\[Agent documentation\]\(llms\.txt\)", readme)
    index = (root / "llms.txt").read_text()
    destinations = re.findall(r"\[[^\]]+\]\(([^)]+)\)", index)
    assert {"docs/api.md", "docs/behavior.md", "README.md"} <= set(destinations)
    for destination in destinations:
        assert not Path(destination).is_absolute(), "index links must use the same checkout"
        resolved = (root / destination).resolve()
        assert resolved.is_relative_to(root), "index links must stay within the repository"
        assert resolved.is_file(), f"broken agent documentation link: {destination}"


def test_readme_navigation_anchors_resolve():
    readme = (Path(__file__).parents[1] / "README.md").read_text()
    # Fenced examples can contain comments; only prose headings create anchors.
    prose = re.sub(r"```.*?```", "", readme, flags=re.S)
    anchors = set()
    for heading in re.findall(r"^#{1,6} (.+)$", prose, re.M):
        anchor = re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")
        anchors.add(anchor)
    destinations = re.findall(r"\[[^\]]+\]\(#([^)]+)\)", prose)
    assert destinations, "README should provide section navigation"
    for destination in destinations:
        assert destination in anchors, f"broken README section link: #{destination}"
