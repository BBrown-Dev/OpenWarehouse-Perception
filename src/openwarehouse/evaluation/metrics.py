"""Metrics for the zero-shot experiments."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from sklearn.metrics import roc_auc_score


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return pairwise IoU for Nx4 and Mx4 xyxy arrays."""
    overlap = _intersection(a, b)
    return overlap / np.maximum(_area(a)[:, None] + _area(b)[None, :] - overlap, 1e-9)


def _area(boxes: np.ndarray) -> np.ndarray:
    return np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)


def _intersection(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    return np.prod(np.clip(bottom_right - top_left, 0, None), axis=-1)


def detection_ap(
    predictions: dict[str, list[tuple[float, list[float]]]],
    ground_truth: dict[str, list[tuple[list[float], bool]]],
    iou_threshold: float = 0.5,
) -> float:
    """Compute Open Images style AP for one class.

    `predictions` maps image ID to (score, box). `ground_truth` maps every evaluated image ID to
    (box, is_group_of), and verified-negative images map to an empty list. Each group-of box
    counts as one object: its first hit is a true positive and later hits are ignored.
    """
    total = sum(len(boxes) for boxes in ground_truth.values())
    if total == 0:
        return float("nan")
    flat = [
        (score, image_id, box)
        for image_id, entries in predictions.items()
        if image_id in ground_truth
        for score, box in entries
    ]
    flat.sort(key=lambda entry: -entry[0])
    matched: dict[str, set[int]] = {image_id: set() for image_id in ground_truth}
    outcomes: list[int] = []
    for _, image_id, box in flat:
        entries = ground_truth[image_id]
        outcome = _match(np.array([box], dtype=float), entries, matched[image_id], iou_threshold)
        if outcome is not None:
            outcomes.append(outcome)
    if not outcomes:
        return 0.0
    true_positives = np.cumsum(outcomes)
    recall = true_positives / total
    precision = true_positives / np.arange(1, len(outcomes) + 1)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    return float(np.sum(np.diff(np.concatenate([[0.0], recall])) * envelope))


def _match(
    box: np.ndarray,
    entries: list[tuple[list[float], bool]],
    matched: set[int],
    iou_threshold: float,
) -> int | None:
    """Return 1 for a true positive, 0 for a false positive, and None for an ignored prediction."""
    instance_ids = [i for i, entry in enumerate(entries) if not entry[1]]
    if instance_ids:
        boxes = np.array([entries[i][0] for i in instance_ids], dtype=float)
        ious = box_iou(box, boxes)[0]
        best = int(ious.argmax())
        if ious[best] >= iou_threshold and instance_ids[best] not in matched:
            matched.add(instance_ids[best])
            return 1
    # A box inside a group-of region is judged by how much of the prediction falls within it.
    for index, (group_box, is_group) in enumerate(entries):
        if is_group:
            group = np.array([group_box], dtype=float)
            overlap = _intersection(box, group)[0, 0] / max(_area(box)[0], 1e-9)
            if overlap >= iou_threshold:
                if index in matched:
                    return None
                matched.add(index)
                return 1
    return 0


def match_one_to_one(
    detections: list[list[float]], targets: list[list[float]]
) -> dict[int, tuple[int, float]]:
    """Match detections to target boxes one-to-one, greedily by IoU.

    Returns {target index: (detection index, IoU)} for every target that overlaps a detection.
    One detection covering two objects is credited to only one of them.
    """
    if not targets or not detections:
        return {}
    ious = box_iou(np.array(targets, dtype=float), np.array(detections, dtype=float))
    matches: dict[int, tuple[int, float]] = {}
    used: set[int] = set()
    for flat in np.argsort(-ious, axis=None, kind="stable"):
        target, detection = (int(v) for v in np.unravel_index(flat, ious.shape))
        if ious[target, detection] <= 0:
            break
        if target in matches or detection in used:
            continue
        matches[target] = (detection, float(ious[target, detection]))
        used.add(detection)
    return matches


def localization_iou(
    detections: list[list[float]], targets: list[list[float]]
) -> np.ndarray:
    """Return, for each target box, the IoU of its one-to-one matched detection (0 if none)."""
    matched = np.zeros(len(targets))
    for target, (_, iou) in match_one_to_one(detections, targets).items():
        matched[target] = iou
    return matched


def mask_iou(prediction: np.ndarray, target: np.ndarray) -> float:
    """Return IoU between two boolean masks of the same shape."""
    union = np.logical_or(prediction, target).sum()
    return float(np.logical_and(prediction, target).sum() / union) if union else 0.0


def auroc(y_true: Any, scores: Any) -> float:
    """Return AUROC, which does not depend on the sampled share of positives."""
    y_true = np.asarray(y_true, dtype=bool)
    if y_true.all() or not y_true.any():
        return float("nan")
    return float(roc_auc_score(y_true, np.asarray(scores)))


def bootstrap_ci(
    statistic: Callable[[np.ndarray], float],
    n_items: int,
    samples: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Return a percentile CI, resampling item indices so paired comparisons stay paired."""
    generator = np.random.default_rng(seed)
    values = [statistic(generator.integers(0, n_items, n_items)) for _ in range(samples)]
    low, high = np.nanquantile(values, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)
