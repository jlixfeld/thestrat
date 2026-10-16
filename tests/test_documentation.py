"""Run the public examples and verify the results shown to readers."""

import importlib
import re
from pathlib import Path

import pytest

from thestrat import Timeframe

README = Path(__file__).resolve().parents[1] / "README.md"


def _examples():
    return re.findall(r"```python\n(.*?)```(?:\n\n```text\n(.*?)```)?", README.read_text(), re.DOTALL)


@pytest.mark.parametrize(
    "source, expected_output", _examples(), ids=["aggregation", "history", "sessions", "scalar-and-symbols"]
)
def test_readme_examples(source, expected_output, capsys):
    namespace = {}
    exec(compile(source, str(README), "exec"), namespace)
    actual = capsys.readouterr().out
    assert actual == expected_output

    if "summary" in namespace:
        expected_timeframes = [tf.value for tf in Timeframe if tf != Timeframe.FIVE_SEC]
        assert namespace["summary"]["timeframe"].to_list() == expected_timeframes
        agg, bars = namespace["agg"], namespace["bars"]
        for tf in expected_timeframes:
            rows = agg.aggregate(bars, tf).select("open", "high", "low", "close", "volume").rows()
            expected = (
                [(100, 103, 99, 102, 10), (102, 105, 101, 104, 20)] if tf == "1min" else [(100, 105, 99, 104, 30)]
            )
            assert rows == expected
    if "classified" in namespace:
        assert namespace["classified"]["color1"].to_list() == [None, "green", "green", "green"]
        assert namespace["classified"]["color3"].to_list() == [None, None, None, "green"]


@pytest.mark.parametrize("module", ["aggregation", "base", "factory", "indicators", "precision", "schemas", "signals"])
def test_documented_legacy_imports_fail(module):
    with pytest.raises(ImportError, match="rewritten in 1.0.0"):
        importlib.import_module(f"thestrat.{module}")
