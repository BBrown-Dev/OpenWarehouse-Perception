"""Deterministic hazard rules, kept separate from models so they can be audited."""

from __future__ import annotations

from typing import Any

from openwarehouse.schemas import Hazard, Instance


def check_person_proximity(instances: list[Instance], config: dict[str, Any]) -> list[Hazard]:
    """Flag people inside the forklift's configured safety zone."""
    raise NotImplementedError


def check_blocked_aisle(instances: list[Instance], config: dict[str, Any]) -> list[Hazard]:
    """Flag objects obstructing the travel path."""
    raise NotImplementedError


def check_floor_hazards(instances: list[Instance], config: dict[str, Any]) -> list[Hazard]:
    """Flag spills and fallen loads."""
    raise NotImplementedError
