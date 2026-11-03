"""Data contract passed between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Instance:
    """One detected object."""

    label: str
    score: float
    box_xyxy: tuple[float, float, float, float]
    mask: Any | None = None  # Boolean H x W array
    encoder_score: float | None = None  # Set by the SigLIP crop-scoring step
