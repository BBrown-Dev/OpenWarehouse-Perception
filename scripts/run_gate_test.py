"""Freeze the SigLIP presence check on dev data, then test it once on fresh images."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from openwarehouse.config import load_config
from openwarehouse.data.open_images import _download_images, fetch_labels, load_box_labels
from openwarehouse.evaluation.metrics import bootstrap_ci, detection_ap
from openwarehouse.experiments import (
    OUTPUT_DIR,
    build_prompts,
    image_path,
    load_detector_predictions,
    load_image,
    load_manifest,
    require_selection,
    restrict_to_manifest,
)
from openwarehouse.gate import (
    FRESH_MANIFEST_PATH,
    GATE_DIR,
    GATE_SELECTION_PATH,
    drop_linked_images,
    freeze_gate,
    image_rates,
    load_fresh_boxes,
    recall_floor_cut,
    require_gate,
    select_fresh,
)
from openwarehouse.models.vision_language import VisionLanguageEncoder
from openwarehouse.segmentation.grounded_sam import GroundedSAM
from openwarehouse.utils import resolve_device, set_seed

RESULTS = Path("docs/results")


def cached_scores(config, selection, manifest):
    """Join the frozen SigLIP logit and top detector score for every cached image-class row."""
    scores = pd.read_csv(OUTPUT_DIR / "siglip_scores.csv", dtype={"image_id": str})
    scores = restrict_to_manifest(scores, manifest)
    scores = scores[scores["condition"] == selection["siglip_condition"]]
    detector = restrict_to_manifest(load_detector_predictions(), manifest)
    detector = detector[detector["condition"] == selection["detector_condition"]].copy()
    detector["max_score"] = [max((d[0] for d in ds), default=0.0) for ds in detector["detections"]]
    return scores[["image_id", "class_name", "role", "is_positive", "logit"]].merge(
        detector[["image_id", "class_name", "max_score"]], on=["image_id", "class_name"]
    )


def score_fresh(fresh, config, selection, device):
    """Run both models on fresh images with their frozen prompts, caching detections."""
    path = GATE_DIR / "fresh_scores.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    prompts = build_prompts(config)
    records = {}
    encoder = VisionLanguageEncoder(config["models"]["siglip"], device=device)
    for class_name, group in fresh.groupby("class_name"):
        text = encoder.encode_text([prompts[class_name][selection["siglip_condition"]]])
        rows = group.reset_index(drop=True)
        for start in tqdm(range(0, len(rows), 32), desc=f"siglip {class_name}"):
            batch = rows.iloc[start : start + 32]
            images = [load_image(image_path(config, s, i)) for s, i in zip(batch["split"], batch["image_id"])]
            logits = encoder.logits(encoder.encode_images(images), text).detach().cpu().numpy()[:, 0]
            for image_id, logit in zip(batch["image_id"], logits):
                records[f"{class_name}/{image_id}"] = {"logit": float(logit)}
    del encoder
    torch.cuda.empty_cache()

    model = GroundedSAM(config, device=device)
    batch_size = config["inference"]["detector_batch_size"]
    for class_name, group in fresh.groupby("class_name"):
        prompt = prompts[class_name][selection["detector_condition"]]
        rows = group.reset_index(drop=True)
        for start in tqdm(range(0, len(rows), batch_size), desc=f"detector {class_name}"):
            batch = rows.iloc[start : start + batch_size]
            images = [load_image(image_path(config, s, i)) for s, i in zip(batch["split"], batch["image_id"])]
            for image_id, image, found in zip(batch["image_id"], images, model.detect_batch(images, prompt)):
                width, height = image.size
                records[f"{class_name}/{image_id}"]["detections"] = [
                    [
                        round(d.score, 4),
                        [d.box_xyxy[0] / width, d.box_xyxy[1] / height,
                         d.box_xyxy[2] / width, d.box_xyxy[3] / height],
                    ]
                    for d in found
                ]
    path.write_text(json.dumps(records), encoding="utf-8")
    return records


def test_class(class_name, rows, records, boxes, cut, threshold, settings, iou):
    keys = [f"{class_name}/{i}" for i in rows["image_id"]]
    y = rows["is_positive"].to_numpy(bool)
    logits = np.array([records[k]["logit"] for k in keys])
    scores = np.array([max((d[0] for d in records[k]["detections"]), default=0.0) for k in keys])
    detector = scores >= threshold
    gated = detector & (logits >= cut)
    alpha = settings["alpha"] / len(settings["test_classes"])
    samples = settings["bootstrap_samples"]

    def fpr(fired, index):
        negatives = ~y[index]
        return fired[index][negatives].mean() if negatives.any() else np.nan

    def recall(fired, index):
        positives = y[index]
        return fired[index][positives].mean() if positives.any() else np.nan

    fpr_drop = bootstrap_ci(lambda i: fpr(detector, i) - fpr(gated, i), len(y), samples, alpha=alpha)
    recall_loss = bootstrap_ci(
        lambda i: recall(detector, i) - recall(gated, i), len(y), samples, alpha=alpha
    )
    base, with_gate = image_rates(y, detector), image_rates(y, gated)
    max_loss = settings["max_recall_loss"][class_name]
    point_loss = base["recall"] - with_gate["recall"]

    # Secondary outcome: AP50 when the check removes all of an image's detections.
    ids = rows["image_id"].tolist()
    passed = dict(zip(ids, logits >= cut))
    truth, plain, filtered = {}, {}, {}
    for image_id, positive, key in zip(ids, y, keys):
        image_boxes = boxes.get(image_id, [])
        if positive and not image_boxes:
            continue
        truth[image_id] = image_boxes if positive else []
        plain[image_id] = [(d[0], d[1]) for d in records[key]["detections"]]
        filtered[image_id] = plain[image_id] if passed[image_id] else []
    names = list(truth)

    def ap(predictions, index):
        pairs = [(f"{n}:{names[i]}", names[i]) for n, i in enumerate(index)]
        return detection_ap(
            {new: predictions[old] for new, old in pairs},
            {new: truth[old] for new, old in pairs},
            iou,
        )

    ap_delta = bootstrap_ci(lambda i: ap(filtered, i) - ap(plain, i), len(names), 300, alpha=alpha)
    return {
        "class_name": class_name,
        "n_pos": int(y.sum()),
        "n_neg": int((~y).sum()),
        "gate_logit_cut": cut,
        "detector_fpr": base["fpr"],
        "gated_fpr": with_gate["fpr"],
        "fpr_drop_low": fpr_drop[0],
        "fpr_drop_high": fpr_drop[1],
        "detector_recall": base["recall"],
        "gated_recall": with_gate["recall"],
        "recall_loss_low": recall_loss[0],
        "recall_loss_high": recall_loss[1],
        "detector_precision": base["precision"],
        "gated_precision": with_gate["precision"],
        "ap50_images": len(names),
        "detector_ap50": detection_ap(plain, truth, iou),
        "gated_ap50": detection_ap(filtered, truth, iou),
        "ap50_delta_low": ap_delta[0],
        "ap50_delta_high": ap_delta[1],
        "adopt": bool(fpr_drop[0] > 0 and point_loss <= max_loss),
    }


def main() -> None:
    config = load_config("open_images", "poc", "gate")
    settings = config["gate"]
    set_seed(settings["seed"])
    device = resolve_device(config["device"])
    selection = require_selection()
    threshold = selection["detector_threshold"]
    manifest = load_manifest(config)
    table = cached_scores(config, selection, manifest)

    # 1. Dev only: set each class's cut, and freeze before any fresh image is chosen.
    dev = table[table["role"] == "dev"]
    if GATE_SELECTION_PATH.exists():
        cuts = require_gate()
    else:
        cuts = freeze_gate(
            {
                name: recall_floor_cut(
                    group["is_positive"].to_numpy(bool),
                    group["logit"].to_numpy(),
                    settings["dev_recall_floor"],
                )
                for name, group in dev.groupby("class_name")
            }
        )
    print("cuts:", {k: round(v, 3) for k, v in cuts.items()})

    def compare(frame):
        rows = []
        for name, group in frame.groupby("class_name"):
            y = group["is_positive"].to_numpy(bool)
            detector = group["max_score"].to_numpy() >= threshold
            gated = detector & (group["logit"].to_numpy() >= cuts[name])
            rows.append(
                {
                    "class_name": name,
                    "n_pos": int(y.sum()),
                    "n_neg": int((~y).sum()),
                    "detector_recall": image_rates(y, detector)["recall"],
                    "gated_recall": image_rates(y, gated)["recall"],
                    "detector_fpr": image_rates(y, detector)["fpr"],
                    "gated_fpr": image_rates(y, gated)["fpr"],
                }
            )
        return pd.DataFrame(rows)

    dev_table = compare(dev)
    dev_table.to_csv(RESULTS / "gate_dev.csv", index=False)
    print(dev_table.round(3).to_string())

    # 2. Choose and download images no earlier step touched.
    if FRESH_MANIFEST_PATH.exists():
        fresh = pd.read_csv(FRESH_MANIFEST_PATH, dtype={"image_id": str})
    else:
        labels = fetch_labels(config)
        labels["image_id"] = labels["image_id"].astype(str)
        used = pd.read_csv(OUTPUT_DIR / "manifest.csv", dtype={"image_id": str})
        fresh = select_fresh(labels, used, config)
        for split, group in fresh.groupby("split"):
            data_dir = image_path(config, split, "x").parent
            _download_images(
                sorted(group["image_id"].unique()), split, data_dir,
                config["open_images"]["image_bucket"],
            )
        present = [
            image_path(config, s, i).exists() for s, i in zip(fresh["split"], fresh["image_id"])
        ]
        fresh = fresh[present].reset_index(drop=True)
        fresh, links = drop_linked_images(
            fresh, used, config, config["evaluation"]["near_duplicate_distance"]
        )
        print("fresh images removed as linked:", len(links))
        fresh.to_csv(FRESH_MANIFEST_PATH, index=False)
    print(fresh.groupby(["class_name", "is_positive"])["image_id"].nunique().to_string())
    shutil.copy(FRESH_MANIFEST_PATH, RESULTS / "gate_fresh_manifest.csv")

    # 3. The one confirmatory run.
    records = score_fresh(fresh, config, selection, device)
    boxes = load_fresh_boxes(fresh, config, load_box_labels(config))
    results = pd.DataFrame(
        [
            test_class(
                name, group.reset_index(drop=True), records, boxes[name], cuts[name], threshold,
                settings, config["evaluation"]["iou_threshold"],
            )
            for name, group in fresh.groupby("class_name")
        ]
    )
    results.to_csv(RESULTS / "gate_fresh_test.csv", index=False)
    print(results.round(3).T.to_string())

    # 4. Classes with no fresh negatives, including forklift: reused images, not confirmatory.
    reused = compare(table[table["role"] == "heldout"])
    reused.to_csv(RESULTS / "gate_reused_heldout_nonconfirmatory.csv", index=False)
    print(reused.round(3).to_string())
    shutil.copy(GATE_SELECTION_PATH, RESULTS / "gate_selection.json")


if __name__ == "__main__":
    main()
