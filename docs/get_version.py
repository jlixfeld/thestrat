#!/usr/bin/env python3
"""Get version from pyproject.toml for MkDocs."""

import sys
import tomllib
from pathlib import Path


def get_version():
    """Get version from pyproject.toml."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"

    try:
        with open(pyproject_path, "rb") as f:
            pyproject = tomllib.load(f)
        return pyproject["project"]["version"]
    except Exception as e:
        print(f"Error reading version: {e}", file=sys.stderr)
        return "unknown"


if __name__ == "__main__":
    print(get_version())
