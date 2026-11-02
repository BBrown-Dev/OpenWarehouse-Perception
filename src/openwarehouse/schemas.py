"""Data contracts passed between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class RiskLevel(IntEnum):
    CLEAR = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Instance:
    """One detected object."""

    label: str
    score: float
    box_xyxy: tuple[float, float, float, float]
    mask: Any | None = None  # Boolean H x W array
    encoder_score: float | None = None  # Set by the CLIP/SigLIP re-scoring step


@dataclass
class Hazard:
    """One triggered safety rule."""

    rule_id: str
    risk_level: RiskLevel
    description: str
    instances: list[Instance] = field(default_factory=list)


@dataclass
class SceneAssessment:
    """Pipeline output for one frame."""

    risk_level: RiskLevel
    instances: list[Instance] = field(default_factory=list)
    hazards: list[Hazard] = field(default_factory=list)
    is_uncertain: bool = False
    latency_ms: float | None = None
