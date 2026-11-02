"""Cheap, model-free steps for image enhancement and input quality checks."""

from __future__ import annotations

from typing import Any


def enhance_low_light(image: Any) -> Any:
    """Improve contrast in dark frames, for example with CLAHE."""
    raise NotImplementedError


def assess_image_quality(image: Any) -> dict[str, float]:
    """Score blur, brightness, and glare so bad frames can be flagged as uncertain."""
    raise NotImplementedError


def detect_floor_markings(image: Any) -> Any:
    """Segment painted aisle and pedestrian lanes by color and line detection."""
    raise NotImplementedError
