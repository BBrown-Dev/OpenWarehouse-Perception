"""Warehouse-relevant image corruptions, registered by the names used in the config."""

from __future__ import annotations

from typing import Any


def apply_corruption(image: Any, name: str, severity: int) -> Any:
    """Apply one configured corruption at severity 1 to 5."""
    raise NotImplementedError
