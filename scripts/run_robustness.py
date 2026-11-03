"""Person detection under warehouse-relevant corruptions, at the frozen dev choices.

Held-out images are used only for measurement: the condition, threshold and corruption set were
all fixed before this sweep (`configs/poc.yaml`, `selection.json`, `configs/robustness.yaml`).
Ground truth is unchanged under corruption, so a person hidden by occlusion still counts as a
miss, which is the safety-relevant reading.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from openwarehouse.config import load_config
from openwarehouse.data.open_images import load_box_labels
from openwarehouse.evaluation.metrics import detection_ap, localization_iou
from openwarehouse.experiments import (
    _load_ground_truth,
    build_prompts,
    image_path,
    load_image,
    load_manifest,
    require_selection,
)
from openwarehouse.robustness.runner import run_sweep, summarize_degradation
from openwarehouse.segmentation.grounded_sam import GroundedSAM
from openwarehouse.utils import resolve_device, set_seed

RESULTS = Path("docs/results")
FIGURES = Path("docs/figures")
PER_LABEL = 150


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cached-only", action="store_true", help="report only conditions already cached; no model run")
    args = parser.parse_args()
    config = load_config("open_images", "poc", "robustness")
    set_seed(config["seed"])
    selection = require_selection()
    threshold = selection["detector_threshold"]
    ground_truth = _load_ground_truth()
    box_labels = load_box_labels(config)["Person"]
    manifest = load_manifest(config)
    rows = manifest[(manifest["role"] == "heldout") & (manifest["class_name"] == "Person")]
    rows = rows.assign(
        boxes=[
            [(b["box"], b["is_group_of"]) for b in ground_truth.get(i, {}).get("boxes", []) if b["label"] in box_labels]
            for i in rows["image_id"]
        ]
    )
    positives = rows[rows["is_positive"] & (rows["boxes"].map(len) > 0)]
    negatives = rows[~rows["is_positive"]]
    sample = pd.concat(
        [positives.sample(min(PER_LABEL, len(positives)), random_state=config["seed"]),
         negatives.sample(min(PER_LABEL, len(negatives)), random_state=config["seed"])]
    ).reset_index(drop=True)
    dataset = [(f"{s}/{i}", load_image(image_path(config, s, i))) for s, i in zip(sample["split"], sample["image_id"])]
    truth = {f"{s}/{i}": (b if p else []) for s, i, b, p in zip(sample["split"], sample["image_id"], sample["boxes"], sample["is_positive"])}
    y = sample["is_positive"].to_numpy(bool)
    keys = [key for key, _ in dataset]

    model = None if args.cached_only else GroundedSAM(config, device=resolve_device(config["device"]))
    prompt = build_prompts(config)["Person"][selection["detector_condition"]]
    batch = config["inference"]["detector_batch_size"]

    def predict(images):
        """Return [[score, box], ...] per image, so detections can be cached as JSON."""
        found = []
        for start in range(0, len(images), batch):
            found += [
                [[d.score, list(d.box_xyxy)] for d in image_found]
                for image_found in model.detect_batch(images[start : start + batch], prompt)
            ]
        return found

    def metrics(found):
        predictions = {k: [(d[0], d[1]) for d in f] for k, f in zip(keys, found)}
        fired = np.array([max((d[0] for d in f), default=0.0) >= threshold for f in found])
        best = np.concatenate(
            [
                localization_iou([d[1] for d in f if d[0] >= threshold], [b for b, g in truth[k] if not g])
                for k, f in zip(keys, found)
            ]
        )
        return {
            "ap50": detection_ap(predictions, truth, config["evaluation"]["iou_threshold"]),
            "image_recall": float(fired[y].mean()),
            "image_fpr": float(fired[~y].mean()),
            "instance_recall_iou50": float((best >= 0.5).mean()),
        }

    results = summarize_degradation(
        run_sweep(predict, dataset, metrics, config, cache_dir=Path("artifacts/poc/robustness"), cached_only=args.cached_only)
    )
    results.insert(0, "images", len(dataset))
    results.to_csv(RESULTS / "robustness_person.csv", index=False)
    print(results.round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    clean = results[results["corruption"] == "clean"].iloc[0]
    for axis, metric, title in ((axes[0], "instance_recall_iou50", "Person instances found (IoU ≥ 0.5)"), (axes[1], "image_fpr", "False alarms on no-person images")):
        for name, group in results[results["corruption"] != "clean"].groupby("corruption"):
            axis.plot(group["severity"], group[metric], marker="o", linewidth=1.5, label=name.replace("_", " "))
        axis.axhline(clean[metric], color="0.5", linestyle="--", linewidth=1, label="clean")
        axis.set_title(title)
        axis.set_xlabel("Severity")
        axis.set_ylim(0, 1)
        axis.set_xticks([1, 2, 3, 4, 5])
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    fig.suptitle(f"Held-out Person detection under corruption ({len(dataset)} images, threshold {threshold})", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "robustness_person.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
