"""LoRA adaptation of the vision-language encoder."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def apply_lora(model: Any, config: dict[str, Any]) -> Any:
    """Wrap the model with LoRA adapters and freeze the base weights."""
    raise NotImplementedError


def train_lora(model: Any, dataloaders: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Train the adapters with early stopping and return the training history."""
    raise NotImplementedError


def save_adapter(model: Any, path: Path) -> None:
    """Save only the adapter weights."""
    raise NotImplementedError


def load_adapter(model: Any, path: Path) -> Any:
    """Load saved adapter weights onto a base model."""
    raise NotImplementedError
