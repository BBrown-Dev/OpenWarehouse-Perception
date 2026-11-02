"""Experiment tracking, so each notebook logs runs the same way."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def init_tracking(config: dict[str, Any]) -> None:
    """Point the tracking backend at the configured URI and experiment."""
    raise NotImplementedError


def log_run(
    name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    artifacts: list[Path] | None = None,
) -> str:
    """Log one run, including the git commit and config, and return its run ID."""
    raise NotImplementedError
