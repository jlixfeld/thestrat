"""README examples and the displayed DataFrame rows are executable contracts."""

import re
from pathlib import Path

import polars as pl

from thestrat import Color


def test_readme_examples_and_displayed_rows():
    readme = (Path(__file__).parents[1] / "README.md").read_text()
    namespace = {}
    for block in re.findall(r"```python\n(.*?)```", readme, re.S):
        exec(compile(block, "README.md", "exec"), namespace)
    preview = namespace["preview"]
    rendered = []
    for row in preview.iter_rows(named=True):
        values = [
            row["timeframe"],
            row["timestamp"].strftime("%Y-%m-%d %H:%M"),
            *(str(row[n]) for n in ["open", "high", "low", "close", "volume"]),
            "null" if row["scenario0"] is None else row["scenario0"],
            row["color0"],
        ]
        rendered.append("| " + " | ".join(values) + " |")
    table = readme.split("<!-- dataframe-preview: generated from the executable example -->", 1)[1].split(
        "\n\nAggregation", 1
    )[0]
    assert table.strip().splitlines()[2:] == rendered
    assert preview.height == 13
    updated = namespace["updated"].filter(pl.col("timeframe") == "4h").row(0, named=True)
    assert [updated[n] for n in ["high", "close", "volume"]] == [106.0, 105.0, 170.0]
    assert namespace["color"] == Color.GREEN
