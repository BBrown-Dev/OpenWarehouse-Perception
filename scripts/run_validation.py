"""Validation added by the review, on dev data only, then corrected held-out re-scoring."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from openwarehouse.config import load_config
from openwarehouse.evaluation.metrics import auroc, box_iou
from openwarehouse.experiments import (
    OUTPUT_DIR,
    build_prompts,
    image_path,
    load_detector_predictions,
    load_image,
    load_manifest,
    require_selection,
    restrict_to_manifest,
    select_condition,
    select_detector_threshold,
    summarize_detector,
)
from openwarehouse.failure_analysis import (
    evaluate_masks,
    evaluate_rescoring,
    images_for,
    load_gt_segmentations,
)
from openwarehouse.models.vision_language import VisionLanguageEncoder
from openwarehouse.segmentation.grounded_sam import GroundedSAM
from openwarehouse.utils import resolve_device, set_seed

RESULTS = Path("docs/results")
# Nouns unrelated to industrial scenes, because absence was verified for forklifts only.
CONTROL_NOUNS = [
    "zebra", "penguin", "violin", "banana", "pineapple",
    "dolphin", "piano", "kangaroo", "snowman", "lighthouse",
]


def top(found):
    return (found[0].score, list(found[0].box_xyxy)) if found else (0.0, None)


def control_nouns(model, config, threshold):
    """Test whether the article effect holds beyond the one noun that revealed it."""
    images = images_for(config, "dev", "Forklift truck", False, 60, config["seed"])
    rows = []
    for split, image_id in tqdm(images.itertuples(index=False), total=len(images), desc="controls"):
        image = load_image(image_path(config, split, image_id))
        for noun in CONTROL_NOUNS:
            article = "an" if noun[0] in "aeiou" else "a"
            for wording, prompt in (("bare", noun), ("article", f"{article} {noun}")):
                rows.append(
                    {"noun": noun, "wording": wording, "score": top(model.detect(image, prompt))[0]}
                )
    frame = pd.DataFrame(rows)
    summary = (
        frame.assign(fires=frame["score"] >= threshold)
        .pivot_table(index="noun", columns="wording", values="fires", aggfunc="mean")
        .reset_index()
    )
    summary.to_csv(RESULTS / "validation_control_nouns.csv", index=False)
    print(summary.round(3).to_string())
    print(summary[["bare", "article"]].mean().round(3).to_dict())


def score_drift(reference, other, threshold):
    reference, other = np.array(reference), np.array(other)
    difference = np.abs(reference - other)
    return {
        "images": len(reference),
        "median_abs_diff": float(np.median(difference)),
        "p95_abs_diff": float(np.quantile(difference, 0.95)),
        "max_abs_diff": float(difference.max()),
        "decision_flips": int(((reference >= threshold) != (other >= threshold)).sum()),
    }


def detector_sample(config, limit):
    positive = images_for(config, "dev", "Forklift truck", True, limit, config["seed"])
    negative = images_for(config, "dev", "Forklift truck", False, limit, config["seed"])
    return pd.concat([positive, negative])


def precision_drift(config, device, threshold):
    """Compare mixed-precision scores, used for all results, with full 32-bit scores."""
    model = GroundedSAM(config, device=device)
    images = detector_sample(config, 50)
    mixed, full = [], []
    for split, image_id in tqdm(images.itertuples(index=False), total=len(images), desc="fp16/fp32"):
        image = load_image(image_path(config, split, image_id))
        model.use_autocast = True
        mixed.append(top(model.detect(image, "forklift"))[0])
        model.use_autocast = False
        full.append(top(model.detect(image, "forklift"))[0])
    del model
    torch.cuda.empty_cache()
    return score_drift(full, mixed, threshold) | {"comparison": "detector fp32 vs autocast fp16 (GPU)"}


def int8_detector(config, threshold):
    """Compare a dynamically quantized detector with its float version, on CPU."""
    model = GroundedSAM(config, device="cpu")
    images = detector_sample(config, 20)
    loaded = [load_image(image_path(config, s, i)) for s, i in images.itertuples(index=False)]
    start = time.perf_counter()
    baseline = [top(model.detect(image, "forklift")) for image in tqdm(loaded, desc="det fp32")]
    float_seconds = time.perf_counter() - start
    model.detector = torch.ao.quantization.quantize_dynamic(
        model.detector, {torch.nn.Linear}, dtype=torch.qint8
    )
    start = time.perf_counter()
    quantized = [top(model.detect(image, "forklift")) for image in tqdm(loaded, desc="det int8")]
    int8_seconds = time.perf_counter() - start
    overlaps = [
        float(box_iou(np.array([a[1]]), np.array([b[1]]))[0, 0])
        for a, b in zip(baseline, quantized)
        if a[1] and b[1]
    ]
    return score_drift([b[0] for b in baseline], [q[0] for q in quantized], threshold) | {
        "comparison": "detector fp32 vs dynamic int8 (CPU)",
        "median_top_box_iou": float(np.median(overlaps)) if overlaps else float("nan"),
        "fp32_seconds_per_image": float_seconds / len(loaded),
        "int8_seconds_per_image": int8_seconds / len(loaded),
    }


def int8_siglip(config):
    """Compare dynamically quantized SigLIP with its float version on every dev forklift image."""
    encoder = VisionLanguageEncoder(config["models"]["siglip"], device="cpu")
    manifest = load_manifest(config)
    rows = manifest[(manifest["role"] == "dev") & (manifest["class_name"] == "Forklift truck")]
    labels = rows["is_positive"].to_numpy(bool)
    images = [load_image(image_path(config, s, i)) for s, i in zip(rows["split"], rows["image_id"])]

    def logits():
        text = encoder.encode_text(["a photo of a forklift"])
        embedded = [encoder.encode_images(images[i : i + 16]) for i in range(0, len(images), 16)]
        return encoder.logits(torch.cat(embedded), text).detach().numpy()[:, 0]

    baseline = logits()
    scale = float(encoder.model.logit_scale.exp())
    bias = float(encoder.model.logit_bias)
    encoder.model = torch.ao.quantization.quantize_dynamic(
        encoder.model, {torch.nn.Linear}, dtype=torch.qint8
    )
    quantized = logits()
    difference = np.abs(baseline - quantized)
    return {
        "comparison": "siglip fp32 vs dynamic int8 (CPU)",
        "images": len(images),
        "logit_scale": scale,
        "logit_bias": bias,
        "median_abs_logit_diff": float(np.median(difference)),
        "max_abs_logit_diff": float(difference.max()),
        "auroc_fp32": auroc(labels, baseline),
        "auroc_int8": auroc(labels, quantized),
    }


def fresh_process_repeat(model):
    """Compare the saved example scores, from an earlier process, with this process."""
    records = json.loads((RESULTS / "reproductions.json").read_text(encoding="utf-8"))
    config = load_config("open_images", "poc")
    splits = load_manifest(config).drop_duplicates("image_id").set_index("image_id")["split"]
    differences = []
    for record in records:
        image = load_image(image_path(config, splits[record["image_id"]], record["image_id"]))
        found = model.detect(image, record["prompt"])
        saved = record["top_results"][0]
        differences.append(abs((found[0].score if found else 0.0) - (saved[0] if saved else 0.0)))
    return {"examples": len(records), "max_abs_diff": float(np.max(differences))}


def lookalike_example(model, config):
    """Draw the bare forklift prompt on the fire truck, the proper example of failure mode 3."""
    manifest = load_manifest(config).drop_duplicates("image_id").set_index("image_id")
    image_id = "046c751beec3d33a"
    image = load_image(image_path(config, manifest.loc[image_id, "split"], image_id))
    found = model.predict(image, "forklift")[0]
    plt.figure(figsize=(6, 4))
    plt.imshow(image)
    overlay = np.zeros((*found.mask.shape, 4))
    overlay[found.mask] = (0.9, 0.2, 0.1, 0.45)
    plt.imshow(overlay)
    plt.title(f'"forklift" on a fire truck: {found.score:.2f}')
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("docs/figures/lookalike_fire_truck.png", dpi=130)
    plt.close()
    return found.score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-checks", action="store_true",
                        help="skip repeat, control-noun and quantization checks already written")
    args = parser.parse_args()
    config = load_config("open_images", "poc")
    set_seed(config["seed"])
    device = resolve_device(config["device"])
    selection = require_selection()
    threshold = selection["detector_threshold"]
    predictions = load_detector_predictions()

    # The label fix must not change anything that was frozen.
    dev = summarize_detector(predictions, config, "dev", with_ci=False)
    dev.to_csv(OUTPUT_DIR / "detector_dev.csv", index=False)
    condition = select_condition(dev, "image_auroc")
    recomputed = select_detector_threshold(predictions, condition, config)
    print("recomputed selection:", condition, recomputed, "frozen:", selection)
    assert condition == selection["detector_condition"] and recomputed == threshold

    if not args.skip_checks:
        model = GroundedSAM(config, device=device)
        print("fresh process:", (repeat := fresh_process_repeat(model)))
        print("fire truck 'forklift' score:", lookalike_example(model, config))
        control_nouns(model, config, threshold)
        del model
        torch.cuda.empty_cache()

        drift = [precision_drift(config, device, threshold)]
        for name, measure in (("siglip", int8_siglip), ("detector", int8_detector)):
            try:
                drift.append(measure(config) if name == "siglip" else measure(config, threshold))
            except Exception as error:  # Quantization support varies, and a failure is itself a finding.
                drift.append({"comparison": f"{name} dynamic int8", "error": repr(error)[:300]})
            print(drift[-1])
        pd.DataFrame(drift).to_csv(RESULTS / "validation_quantization.csv", index=False)
        pd.DataFrame([repeat]).to_csv(RESULTS / "validation_fresh_process.csv", index=False)

    # Corrected re-scoring: box labels now include child classes.
    held = summarize_detector(predictions, config, "heldout")
    held.to_csv(OUTPUT_DIR / "detector_heldout.csv", index=False)
    print(held[held["condition"].isin(["bare", "article"])].dropna(subset=["ap50"]).round(3).to_string())

    model = GroundedSAM(config, device=device)
    encoder = VisionLanguageEncoder(config["models"]["siglip"], device=device)
    prompts = build_prompts(config)
    for role in ("dev", "heldout"):
        rescoring = evaluate_rescoring(
            model, encoder, config, predictions, role,
            selection["detector_condition"], prompts, selection["siglip_condition"],
        )
        rescoring.to_csv(RESULTS / f"rescoring_{role}.csv", index=False)
        print(rescoring.round(3).to_string())
    del encoder

    classes = ["Box", "Truck", "Person"]
    records = load_gt_segmentations(config, classes)
    masks = evaluate_masks(
        model, config, {c: prompts[c][selection["detector_condition"]] for c in classes}, records
    )
    masks.to_csv(RESULTS / "masks_heldout.csv", index=False)
    print(
        masks.groupby("class_name")
        .agg(
            instances=("image_id", "size"),
            images=("image_id", "nunique"),
            oracle=("oracle_box_mask_iou", "mean"),
            detected=("detected_mask_iou", "mean"),
            no_box=("detected_box_iou", lambda v: (v < 0.5).mean()),
        )
        .round(3)
        .to_string()
    )


if __name__ == "__main__":
    main()
