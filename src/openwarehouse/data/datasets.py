"""Dataset loading and splitting."""

from __future__ import annotations

from typing import Any


class WarehouseDataset:
    """Map-style dataset yielding an image and its annotations."""

    def __init__(self, config: dict[str, Any], split: str, transform: Any | None = None) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> dict[str, Any]:
        raise NotImplementedError


def create_splits(config: dict[str, Any]) -> dict[str, list[str]]:
    """Split by `dataset.group_key` so frames from one sequence stay in one split."""
    raise NotImplementedError


def build_dataloaders(config: dict[str, Any]) -> dict[str, Any]:
    """Build a DataLoader for each split."""
    raise NotImplementedError
