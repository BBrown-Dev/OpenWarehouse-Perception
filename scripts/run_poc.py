"""Run the zero-shot experiments in an order that keeps held-out data out of every choice."""

from __future__ import annotations

from openwarehouse.config import load_config
from openwarehouse.experiments import (
    EXCLUSIONS_PATH,
    OUTPUT_DIR,
    SELECTION_PATH,
    build_leakage_exclusions,
    export_ground_truth,
    freeze_selection,
    require_selection,
    run_detector,
    run_siglip,
    select_condition,
    select_detector_threshold,
    summarize_detector,
    summarize_siglip,
)
from openwarehouse.utils import resolve_device, set_seed


def main() -> None:
    config = load_config("open_images", "poc")
    set_seed(config["seed"])
    device = resolve_device(config["device"])
    print("device:", device)

    # 1. Remove held-out images linked to dev data, before any model output is read.
    if not EXCLUSIONS_PATH.exists():
        exclusions = build_leakage_exclusions(config)
        print(exclusions.drop(columns=["split", "image_id"]).sum().to_string())
        print("held-out images excluded:", len(exclusions))
    if not (OUTPUT_DIR / "ground_truth.json").exists():
        export_ground_truth(config)

    # 2. Run the models. This writes predictions but computes no metrics.
    scores = run_siglip(config, device)
    predictions = run_detector(config, device)

    # 3. Choose from dev results alone, and freeze those choices.
    siglip_dev = summarize_siglip(scores, config, "dev", with_ci=False)
    detector_dev = summarize_detector(predictions, config, "dev", with_ci=False)
    siglip_dev.to_csv(OUTPUT_DIR / "siglip_dev.csv", index=False)
    detector_dev.to_csv(OUTPUT_DIR / "detector_dev.csv", index=False)
    condition = select_condition(detector_dev, "image_auroc")
    threshold = select_detector_threshold(predictions, condition, config)
    if SELECTION_PATH.exists():
        print("already frozen:", require_selection())
    else:
        print("frozen:", freeze_selection(siglip_dev, detector_dev, threshold))

    # 4. Only now score the held-out set.
    summarize_siglip(scores, config, "heldout").to_csv(OUTPUT_DIR / "siglip_heldout.csv", index=False)
    summarize_detector(predictions, config, "heldout").to_csv(
        OUTPUT_DIR / "detector_heldout.csv", index=False
    )


if __name__ == "__main__":
    main()
