"""MkDocs hooks to inject dynamic version."""

import tomllib
from pathlib import Path


def on_config(config, **kwargs):
    """Hook to inject version from pyproject.toml into config.extra."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"

    try:
        with open(pyproject_path, "rb") as f:
            pyproject = tomllib.load(f)
        version = pyproject["project"]["version"]

        # Inject version into config.extra
        if "extra" not in config:
            config["extra"] = {}
        config["extra"]["version"] = version

    except Exception as e:
        print(f"Warning: Could not read version from pyproject.toml: {e}")
        config["extra"]["version"] = "unknown"

    return config
