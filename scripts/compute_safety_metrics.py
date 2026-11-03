"""Person miss rate at the frozen operating point, overall, per slice, and by apparent size.

A missed person is the costly error for a forklift, so this reports it directly instead of
through AP50. Box height as a share of image height stands in for distance from the camera.

Data, all scored with the frozen `bare` prompt and threshold, no model run:
- held-out: the main held-out set;
- warehouse-like: held-out images verified positive for Warehouse, Shelving, Aisle or Inventory;
- fresh: the gate test's unused images (cached detections in `artifacts/poc/gate/`).

Image-level intervals are Wilson intervals. Instance-level intervals resample images, since people
in one image are not independent. Group-of boxes are excluded at instance level.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Sibling scripts are imported directly, so make them importable from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import math

import numpy as np
import pandas as pd

import compute_metrics as cm
from compute_slices import SCENE_CLASSES
from openwarehouse.config import load_config
from openwarehouse.data.open_images import load_box_labels
from openwarehouse.evaluation.metrics import localization_iou
from openwarehouse.experiments import _load_ground_truth, require_selection
from openwarehouse.gate import FRESH_MANIFEST_PATH, GATE_DIR, load_fresh_boxes

RESULTS = Path("docs/results")
HEIGHT_EDGES = [0.0, 0.05, 0.10, 0.25, 0.50, 1.0001]
HEIGHT_LABELS = ["<5%", "5-10%", "10-25%", "25-50%", ">=50%"]


def wilson(misses: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = misses / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def instances_from(images: list[dict]) -> pd.DataFrame:
    """Flatten [{image, fired, targets, heights, detections}] into one row per person box."""
    rows = []
    for record in images:
        best = localization_iou(record["detections"], record["targets"])
        for iou, height in zip(best, record["heights"]):
            rows.append({"image": record["image"], "found": iou >= 0.5, "height": height})
    return pd.DataFrame(rows)


def heldout_records(config, selection):
    frame = cm.decisions(config, selection)
    held = frame[(frame["role"] == "heldout")]
    scene = set(held.loc[held["class_name"].isin(SCENE_CLASSES) & held["is_positive"], "image_id"])
    person = held[(held["class_name"] == "Person") & held["is_positive"]]
    ground_truth = _load_ground_truth()
    labels = load_box_labels(config)["Person"]
    threshold = selection["detector_threshold"]
    records = []
    for row in person.itertuples():
        entry = ground_truth.get(row.image_id.split("/", 1)[1], {})
        height = entry.get("size", [1, 1])[1]
        boxes = [b["box"] for b in entry.get("boxes", []) if b["label"] in labels and not b["is_group_of"]]
        records.append(
            {
                "image": row.image_id,
                "warehouse_like": row.image_id in scene,
                "fired": bool(row.detector_yes),
                "targets": boxes,
                "heights": [(b[3] - b[1]) / height for b in boxes],
                "detections": [d[1] for d in row.detections if d[0] >= threshold],
            }
        )
    return records


def fresh_records(config, threshold):
    fresh = pd.read_csv(FRESH_MANIFEST_PATH, dtype={"image_id": str})
    person = fresh[(fresh["class_name"] == "Person") & fresh["is_positive"]]
    scores = json.loads((GATE_DIR / "fresh_scores.json").read_text(encoding="utf-8"))
    boxes = load_fresh_boxes(person, config, load_box_labels(config))["Person"]
    records = []
    for image_id in person["image_id"]:
        found = scores[f"Person/{image_id}"]["detections"]
        targets = [b for b, group in boxes.get(image_id, []) if not group]
        records.append(
            {
                "image": image_id,
                "warehouse_like": False,
                "fired": max((d[0] for d in found), default=0.0) >= threshold,
                "targets": targets,  # normalized, like the cached fresh detections
                "heights": [b[3] - b[1] for b in targets],
                "detections": [d[1] for d in found if d[0] >= threshold],
            }
        )
    return records


def summarize(name, records):
    rows = []
    images = len(records)
    image_misses = sum(not r["fired"] for r in records)
    low, high = wilson(image_misses, images)
    rows.append({"dataset": name, "level": "image", "n": images, "misses": image_misses,
                 "miss_rate": image_misses / images if images else np.nan, "low": low, "high": high})
    table = instances_from([r for r in records if r["targets"]])
    if len(table):
        per_image = table.groupby("image").agg(n=("found", "size"), missed=("found", lambda f: int((~f).sum())))
        low, high = cm.image_bootstrap(per_image[["n", "missed"]], lambda s: s[1] / s[0])
        rows.append({"dataset": name, "level": "instance (IoU >= 0.5)", "n": len(table),
                     "misses": int((~table["found"]).sum()), "miss_rate": float((~table["found"]).mean()),
                     "low": low, "high": high})
    return rows, table


def by_height(name, table):
    table = table.assign(bucket=pd.cut(table["height"], HEIGHT_EDGES, labels=HEIGHT_LABELS, right=False))
    rows = []
    for bucket, part in table.groupby("bucket", observed=False):
        if not len(part):
            continue
        per_image = part.groupby("image").agg(n=("found", "size"), found=("found", "sum"))
        low, high = cm.image_bootstrap(per_image[["n", "found"]], lambda s: s[1] / s[0])
        rows.append({"dataset": name, "box_height_share": bucket, "instances": len(part),
                     "recall": float(part["found"].mean()), "low": low, "high": high})
    return rows


CONFIDENCE_LEVELS = (0.32, 0.5, 0.7)


def rate_rows(condition, query_set, verification, scores):
    """One row per confidence level: share of wrong queries scoring at or above it."""
    scores = np.asarray(scores, dtype=float)
    rows = []
    for level in CONFIDENCE_LEVELS:
        wrong = int((scores >= level).sum())
        low, high = wilson(wrong, len(scores))
        rows.append(
            {
                "condition": condition,
                "query_set": query_set,
                "absence_or_presence": verification,
                "n": len(scores),
                "confidence_at_least": level,
                "wrong_confident": wrong,
                "rate": wrong / len(scores) if len(scores) else np.nan,
                "low": low,
                "high": high,
                "max_score": float(scores.max()) if len(scores) else np.nan,
            }
        )
    return rows


def confident_false_positives(config, selection):
    """How often the detector is wrong with high confidence, by query condition (held-out)."""
    frame = cm.decisions(config, selection)
    held = frame[frame["role"] == "heldout"]
    rows = []
    negatives = held[~held["is_positive"]]
    rows += rate_rows("ordinary query, absent class", "all 12 classes pooled", "verified absent", negatives["detector_score"])
    for class_name in ("Person", "Forklift truck", "Truck", "Box"):
        part = negatives[negatives["class_name"] == class_name]
        rows += rate_rows("ordinary query, absent class", class_name, "verified absent", part["detector_score"])

    from openwarehouse.evaluation.hallucination import pope_questions

    labels = held[["image_id", "class_name", "is_positive"]]
    for strategy in ("random", "adversarial"):
        questions = pope_questions(labels, strategy, config["seed"])
        asked = questions[~questions["is_positive"]].merge(
            held.drop(columns="is_positive"), on=["image_id", "class_name"]
        )
        rows += rate_rows(f"POPE {strategy} query, absent class", "12 classes, 3 per image", "verified absent", asked["detector_score"])

    absent = pd.DataFrame(json.loads((Path("artifacts/poc/failure") / "absent_heldout.json").read_text(encoding="utf-8")))
    absent["score"] = absent["score"].astype(float)
    for prompt, verification in (("a forklift", "verified absent"), ("a lift truck", "verified absent (forklift synonym)")):
        rows += rate_rows("absent-object prompt with article", f'"{prompt}" on no-forklift images', verification, absent.loc[absent["prompt"] == prompt, "score"])

    modifier = pd.read_csv(RESULTS / "modifier_heldout.csv")
    for prompt in ("no person", "not a person", "no forklift", "no box", "a fallen person", "a fallen box"):
        part = modifier[modifier["prompt"] == prompt]
        rows += rate_rows("negated or state query, object present", f'"{prompt}"', "object verified present; a detection means the negation or state was ignored", part["score"])
    return pd.DataFrame(rows)


def main() -> None:
    config = load_config("open_images", "poc", "gate")
    selection = require_selection()
    confident = confident_false_positives(config, selection)
    confident.to_csv(RESULTS / "safety_confident_errors.csv", index=False)
    print(confident.round(3).to_string(index=False))
    held = heldout_records(config, selection)
    fresh = fresh_records(config, selection["detector_threshold"])
    # The detector keeps at most this many boxes per image, so larger crowds cannot be fully found.
    cap = config["inference"]["max_detections"]
    datasets = {
        "held-out": held,
        "held-out, warehouse-like": [r for r in held if r["warehouse_like"]],
        "fresh (gate images)": fresh,
        f"held-out, images with <= {cap} people": [r for r in held if len(r["targets"]) <= cap],
        f"fresh, images with <= {cap} people": [r for r in fresh if len(r["targets"]) <= cap],
    }
    miss_rows, height_rows = [], []
    for name, records in datasets.items():
        rows, table = summarize(name, records)
        miss_rows += rows
        if len(table):
            height_rows += by_height(name, table)
    pd.DataFrame(miss_rows).to_csv(RESULTS / "safety_person_miss.csv", index=False)
    pd.DataFrame(height_rows).to_csv(RESULTS / "safety_person_recall_by_height.csv", index=False)
    print(pd.DataFrame(miss_rows).round(3).to_string(index=False))
    print(pd.DataFrame(height_rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
