"""Model quantization."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def quantize_dynamic(model_path: Path, output_path: Path) -> Path:
    """Apply dynamic quantization to an ONNX model."""
    raise NotImplementedError


def quantize_static(model_path: Path, output_path: Path, calibration_data: Any) -> Path:
    """Apply static quantization, calibrating on dev data only, never held-out data."""
    raise NotImplementedError
