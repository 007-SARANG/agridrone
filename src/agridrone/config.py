"""Configuration loading utilities.

All hyperparameters and paths come from YAML files in ``configs/`` — this module
is the single entry point for reading them, so no code hardcodes paths or magic
numbers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Return the repository root (two levels up from this file's package).

    Layout: ``<root>/src/agridrone/config.py`` -> root is ``parents[2]``.
    """
    return Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a dictionary.

    Args:
        path: Path to the YAML file. May be absolute, or relative to the
            project root (so callers can pass ``"configs/data.yaml"``).

    Returns:
        Parsed configuration as a nested dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    p = Path(path)
    if not p.is_absolute():
        p = project_root() / p
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {p} did not parse to a mapping.")
    return data


def resolve_path(relative: str | Path) -> Path:
    """Resolve a config-relative path against the project root.

    Absolute paths are returned unchanged.
    """
    p = Path(relative)
    return p if p.is_absolute() else project_root() / p
