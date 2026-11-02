"""Open-vocabulary class and prompt handling."""

from __future__ import annotations

from typing import Any


def build_prompts(config: dict[str, Any]) -> dict[str, list[str]]:
    """Expand class synonyms through the prompt templates, keyed by class name."""
    raise NotImplementedError


def build_detector_caption(config: dict[str, Any]) -> str:
    """Build the period-separated text prompt Grounding DINO expects."""
    raise NotImplementedError


def critical_classes(config: dict[str, Any]) -> set[str]:
    """Return the classes marked `critical`."""
    raise NotImplementedError
