"""Evaluates a model across every corruption and severity."""

from __future__ import annotations

from typing import Any


def run_sweep(predict_fn: Any, dataset: Any, metric_fn: Any, config: dict[str, Any]) -> Any:
    """Return one results row per corruption and severity, plus the clean baseline."""
    raise NotImplementedError


def summarize_degradation(results: Any) -> Any:
    """Compute the metric drop from clean and find the worst-case conditions."""
    raise NotImplementedError
