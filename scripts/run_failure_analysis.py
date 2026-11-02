"""Find failure modes on dev data, then measure the same probes on held-out data."""

from __future__ import annotations

import json
import sys

from openwarehouse.config import load_config
from openwarehouse.experiments import (
    build_prompts,
    load_detector_predictions,
    require_selection,
)
from openwarehouse.failure_analysis import (
    BASE_PROMPTS,
    FAILURE_DIR,
    evaluate_masks,
    evaluate_rescoring,
    load_gt_segmentations,
    run_absent_probe,
    run_modifier_probe,
    summarize_absent_probe,
    summarize_modifier_probe,
)
from openwarehouse.models.vision_language import VisionLanguageEncoder
from openwarehouse.segmentation.grounded_sam import GroundedSAM
from openwarehouse.utils import resolve_device, set_seed

PROBE_IMAGES = 60


def main(role: str) -> None:
    config = load_config("open_images", "poc")
    set_seed(config["seed"])
    device = resolve_device(config["device"])
    selection = require_selection()
    threshold = selection["detector_threshold"]
    FAILURE_DIR.mkdir(parents=True, exist_ok=True)
    model = GroundedSAM(config, device=device)

    modifier = run_modifier_probe(model, config, role, PROBE_IMAGES, config["seed"])
    modifier.to_csv(FAILURE_DIR / f"modifier_{role}.csv", index=False)
    print(summarize_modifier_probe(modifier, threshold).round(3).to_string())

    absent = run_absent_probe(model, config, role, PROBE_IMAGES, config["seed"])
    absent.to_json(FAILURE_DIR / f"absent_{role}.json", orient="records")
    print(summarize_absent_probe(absent, threshold).round(3).to_string())

    prompts = build_prompts(config)
    encoder = VisionLanguageEncoder(config["models"]["siglip"], device=device)
    rescoring = evaluate_rescoring(
        model,
        encoder,
        config,
        load_detector_predictions(),
        role,
        selection["detector_condition"],
        prompts,
        selection["siglip_condition"],
    )
    rescoring.to_csv(FAILURE_DIR / f"rescoring_{role}.csv", index=False)
    print(rescoring.round(3).to_string())

    if role == "heldout":
        classes = ["Box", "Truck", "Person"]
        records = load_gt_segmentations(config, classes)
        class_prompts = {c: prompts[c][selection["detector_condition"]] for c in classes}
        masks = evaluate_masks(model, config, class_prompts, records)
        masks.to_csv(FAILURE_DIR / "masks_heldout.csv", index=False)
        summary = masks.groupby("class_name").agg(
            instances=("image_id", "size"),
            oracle_box_mask_iou=("oracle_box_mask_iou", "mean"),
            detected_mask_iou=("detected_mask_iou", "mean"),
            detected_box_iou=("detected_box_iou", "mean"),
        )
        print(summary.round(3).to_string())
    print(json.dumps({"role": role, "selection": selection, "bases": BASE_PROMPTS}))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dev")
