"""Latency, throughput, memory, and size benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def benchmark_model(
    model_path: Path,
    sample_input: Any,
    config: dict[str, Any],
) -> dict[str, float]:
    """Measure p50 and p95 latency, throughput, peak memory, and file size."""
    raise NotImplementedError


def check_budgets(results: dict[str, float], config: dict[str, Any]) -> dict[str, bool]:
    """Compare benchmark results against the configured edge budgets."""
    raise NotImplementedError
