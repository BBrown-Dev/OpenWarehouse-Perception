"""Combines rule results and uncertainty into one scene risk level."""

from __future__ import annotations

from typing import Any

from openwarehouse.schemas import Instance, SceneAssessment


class SafetyAssessor:
    def __init__(self, config: dict[str, Any]) -> None:
        raise NotImplementedError

    def assess(self, instances: list[Instance], image_quality: dict[str, float]) -> SceneAssessment:
        """Use the highest triggered risk. Uncertain input escalates to `uncertain_risk_level`."""
        raise NotImplementedError
