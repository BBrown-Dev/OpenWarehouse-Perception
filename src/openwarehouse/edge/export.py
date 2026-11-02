"""Model export for edge runtimes."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def merge_lora(model: Any) -> Any:
    """Merge adapters into the base weights before export."""
    raise NotImplementedError


def export_onnx(model: Any, output_path: Path, config: dict[str, Any]) -> Path:
    """Export to ONNX and check its outputs match the PyTorch model."""
    raise NotImplementedError
