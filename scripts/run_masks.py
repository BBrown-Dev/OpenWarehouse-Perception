"""Evaluate SAM masks on held-out ground truth with one-to-one detection matching."""

from __future__ import annotations

from pathlib import Path

from openwarehouse.config import load_config
from openwarehouse.experiments import build_prompts, require_selection
from openwarehouse.failure_analysis import evaluate_masks, load_gt_segmentations
from openwarehouse.segmentation.grounded_sam import GroundedSAM
from openwarehouse.utils import resolve_device, set_seed

CLASSES = ["Box", "Truck", "Person"]


def main() -> None:
    config = load_config("open_images", "poc")
    set_seed(config["seed"])
    selection = require_selection()
    model = GroundedSAM(config, device=resolve_device(config["device"]))
    prompts = build_prompts(config)
    records = load_gt_segmentations(config, CLASSES)
    masks = evaluate_masks(model, config, {c: prompts[c][selection["detector_condition"]] for c in CLASSES}, records)
    masks.to_csv(Path("docs/results/masks_heldout.csv"), index=False)
    print(masks.groupby("class_name").agg(
        instances=("image_id", "size"), oracle=("oracle_box_mask_iou", "mean"),
        detected=("detected_mask_iou", "mean"), unmatched=("detected_box_iou", lambda v: (v < 0.5).mean()),
    ).round(3).to_string())


if __name__ == "__main__":
    main()
