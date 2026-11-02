"""SigLIP presence check before detection, tested once on images no earlier step touched."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from openwarehouse.config import PROJECT_ROOT
from openwarehouse.data.leakage import find_exclusions, fingerprint_images, load_authors

GATE_DIR = PROJECT_ROOT / "artifacts" / "poc" / "gate"
GATE_SELECTION_PATH = GATE_DIR / "gate_selection.json"
FRESH_MANIFEST_PATH = GATE_DIR / "fresh_manifest.csv"


def recall_floor_cut(is_positive: np.ndarray, logits: np.ndarray, floor: float) -> float:
    """Return the highest logit cut that still keeps at least `floor` of the positives."""
    positives = np.sort(logits[is_positive])
    return float(positives[int(np.floor((1 - floor) * len(positives)))])


def image_rates(is_positive: np.ndarray, fired: np.ndarray) -> dict[str, float]:
    """Return image-level precision, recall, and false positive rate."""
    return {
        "precision": float((fired & is_positive).sum() / max(fired.sum(), 1)),
        "recall": float((fired & is_positive).sum() / max(is_positive.sum(), 1)),
        "fpr": float((fired & ~is_positive).sum() / max((~is_positive).sum(), 1)),
    }


def freeze_gate(cuts: dict[str, float], path: Path = GATE_SELECTION_PATH) -> dict[str, float]:
    """Save the per-class cuts once, before any fresh image is chosen or scored."""
    if path.exists():
        raise RuntimeError(f"{path} is already frozen. Delete it deliberately and say so.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cuts, indent=2), encoding="utf-8")
    return cuts


def require_gate(path: Path = GATE_SELECTION_PATH) -> dict[str, float]:
    if not path.exists():
        raise RuntimeError("Freeze the gate on dev data before scoring fresh images.")
    return json.loads(path.read_text(encoding="utf-8"))


def select_fresh(
    labels: pd.DataFrame, used: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    """Sample verified labels from unused images whose photographers no earlier image shares."""
    settings = config["gate"]
    held = labels[
        labels["split"].isin(["validation", "test"])
        & labels["class_name"].isin(settings["test_classes"])
        & ~labels["image_id"].isin(used["image_id"])
    ]
    images = held[["split", "image_id"]].drop_duplicates().reset_index(drop=True)
    authors = load_authors(images, config)
    used_images = used[["split", "image_id"]].drop_duplicates().reset_index(drop=True)
    seen = set(load_authors(used_images, config)["author"].dropna())
    clean = set(authors.loc[~authors["author"].isin(seen), "image_id"])
    held = held[held["image_id"].isin(clean)]
    chosen = []
    for (_, is_positive), group in held.groupby(["class_name", "is_positive"]):
        cap = settings["sample_cap"]["positive" if is_positive else "negative"]
        chosen.append(group.sample(min(cap, len(group)), random_state=settings["seed"]))
    picked = pd.concat(chosen)[["image_id", "split"]].drop_duplicates()
    return held.merge(picked, on=["image_id", "split"]).sort_values(
        ["split", "image_id", "class_name"], ignore_index=True
    )


def drop_linked_images(
    fresh: pd.DataFrame, used: pd.DataFrame, config: dict[str, Any], max_distance: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove downloaded fresh images that are copies of, or near, any earlier image."""
    images = pd.concat(
        [
            used[["split", "image_id"]].drop_duplicates().assign(role="dev"),
            fresh[["split", "image_id"]].drop_duplicates().assign(role="heldout"),
        ],
        ignore_index=True,
    )
    images, hashes = fingerprint_images(images, config)
    links = find_exclusions(load_authors(images, config), hashes, max_distance)
    return fresh[~fresh["image_id"].isin(links["image_id"])].reset_index(drop=True), links


def load_fresh_boxes(
    fresh: pd.DataFrame, config: dict[str, Any], box_labels: dict[str, set[str]]
) -> dict[str, dict[str, list[tuple[list[float], bool]]]]:
    """Return normalized boxes for fresh images as {class: {image: [(box, is_group_of)]}}."""
    root = PROJECT_ROOT / config["open_images"]["dataset_dir"]
    wanted = set(fresh["image_id"])
    metadata = root / "test" / "metadata" / "classes.csv"
    names = pd.read_csv(metadata, header=None, names=["mid", "name"])
    mid_to_name = dict(zip(names["mid"], names["name"]))
    parent = {
        label: name
        for name in config["gate"]["test_classes"]
        for label in box_labels[name]
    }
    boxes: dict[str, dict[str, list[tuple[list[float], bool]]]] = {
        name: {} for name in config["gate"]["test_classes"]
    }
    for split in fresh["split"].unique():
        path = root / split / "labels" / "detections.csv"
        columns = ["ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax", "IsGroupOf"]
        for chunk in pd.read_csv(path, usecols=columns, dtype={"ImageID": str}, chunksize=500_000):
            for row in chunk[chunk["ImageID"].isin(wanted)].itertuples(index=False):
                name = parent.get(mid_to_name.get(row.LabelName))
                if name:
                    box = [row.XMin, row.YMin, row.XMax, row.YMax]
                    boxes[name].setdefault(row.ImageID, []).append((box, bool(row.IsGroupOf)))
    return boxes
