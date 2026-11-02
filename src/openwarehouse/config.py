"""Configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(*names: str) -> dict[str, Any]:
    """Merge `configs/<name>.yaml` files in order over `base.yaml`."""
    config: dict[str, Any] = {}
    for name in ("base", *names):
        with (CONFIG_DIR / f"{name}.yaml").open(encoding="utf-8") as handle:
            config = _merge(config, yaml.safe_load(handle) or {})
    return config
