"""Measures how the models fail, so each failure can be reproduced and explained."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from openwarehouse.data.open_images import load_box_labels
from openwarehouse.evaluation.metrics import box_iou, mask_iou
from openwarehouse.experiments import (
    OUTPUT_DIR,
    _load_ground_truth,
    image_path,
    load_image,
    load_manifest,
    require_selection,
)
from openwarehouse.schemas import Instance
from openwarehouse.segmentation.grounded_sam import GroundedSAM, rescore_instances

# A probe compares a plain prompt with variants that change or contradict its meaning.
MODIFIER_PROBES = {
    "Box": ["a fallen box", "a damaged box", "a crushed box", "no box", "not a box"],
    "Person": ["a fallen person", "a person lying on the floor", "no person", "not a person"],
    "Forklift truck": ["a damaged forklift", "no forklift", "not a forklift"],
}
BASE_PROMPTS = {"Box": "a box", "Person": "a person", "Forklift truck": "a forklift"}
ABSENT_PROMPTS = ["a forklift", "a lift truck", "a wooden pallet", "a fallen package", "a giraffe"]


def images_for(
    config: dict[str, Any], role: str, class_name: str, is_positive: bool, limit: int, seed: int
) -> pd.DataFrame:
    """Return a reproducible sample of images verified for a class."""
    if role == "heldout":
        require_selection()
    manifest = load_manifest(config)
    rows = manifest[
        (manifest["role"] == role)
        & (manifest["class_name"] == class_name)
        & (manifest["is_positive"] == is_positive)
    ]
    return rows.sample(min(limit, len(rows)), random_state=seed)[["split", "image_id"]]


def run_modifier_probe(
    model: GroundedSAM, config: dict[str, Any], role: str, limit: int, seed: int
) -> pd.DataFrame:
    """Test whether a word that changes a prompt's meaning also changes the top detection."""
    rows = []
    for class_name, variants in MODIFIER_PROBES.items():
        images = images_for(config, role, class_name, True, limit, seed)
        base_prompt = BASE_PROMPTS[class_name]
        for split, image_id in tqdm(
            images.itertuples(index=False), total=len(images), desc=f"modifier {class_name}"
        ):
            image = load_image(image_path(config, split, image_id))
            base = model.detect(image, base_prompt)
            for prompt in variants:
                found = model.detect(image, prompt)
                same_box = float("nan")
                if base and found:
                    same_box = float(
                        box_iou(
                            np.array([base[0].box_xyxy]), np.array([found[0].box_xyxy])
                        )[0, 0]
                    )
                rows.append(
                    {
                        "role": role,
                        "class_name": class_name,
                        "split": split,
                        "image_id": image_id,
                        "base_prompt": base_prompt,
                        "prompt": prompt,
                        "base_score": base[0].score if base else 0.0,
                        "score": found[0].score if found else 0.0,
                        "label": found[0].label if found else "",
                        "top_box_iou": same_box,
                    }
                )
    return pd.DataFrame(rows)


def run_absent_probe(
    model: GroundedSAM, config: dict[str, Any], role: str, limit: int, seed: int
) -> pd.DataFrame:
    """Score prompts on images humans verified do not contain a forklift."""
    images = images_for(config, role, "Forklift truck", False, limit, seed)
    rows = []
    for split, image_id in tqdm(images.itertuples(index=False), total=len(images), desc="absent"):
        image = load_image(image_path(config, split, image_id))
        for prompt in ABSENT_PROMPTS:
            found = model.detect(image, prompt)
            rows.append(
                {
                    "role": role,
                    "split": split,
                    "image_id": image_id,
                    "prompt": prompt,
                    "score": found[0].score if found else 0.0,
                    "label": found[0].label if found else "",
                    "box": list(found[0].box_xyxy) if found else None,
                }
            )
    return pd.DataFrame(rows)


def summarize_modifier_probe(probe: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Report how often a modified prompt still fires, on the same object, at the same score."""
    return (
        probe.assign(
            fires=probe["score"] >= threshold,
            same_object=probe["top_box_iou"] >= 0.9,
            score_change=probe["score"] - probe["base_score"],
        )
        .groupby(["role", "class_name", "prompt"])
        .agg(
            images=("image_id", "nunique"),
            fire_rate=("fires", "mean"),
            same_object_rate=("same_object", "mean"),
            mean_score=("score", "mean"),
            mean_score_change=("score_change", "mean"),
        )
        .reset_index()
    )


def summarize_absent_probe(probe: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Report the false detection rate for each prompt on verified-negative images."""
    return (
        probe.assign(fires=probe["score"] >= threshold)
        .groupby(["role", "prompt"])
        .agg(images=("image_id", "nunique"), fire_rate=("fires", "mean"), mean_score=("score", "mean"))
        .reset_index()
    )


@torch.inference_mode()
def siglip_occlusion_map(encoder: Any, image: Image.Image, prompt: str, grid: int = 8) -> np.ndarray:
    """Measure how much the logit drops when each image region is hidden.

    SigLIP pools an image into one vector, so it has no native text-to-region score. Occlusion
    measures which regions influence the match without assuming anything about internals.
    """
    width, height = image.size
    text = encoder.encode_text([prompt])
    baseline = float(encoder.logits(encoder.encode_images([image]), text)[0, 0])
    occluded = []
    for row in range(grid):
        for column in range(grid):
            pixels = np.array(image)
            top, bottom = row * height // grid, (row + 1) * height // grid
            left, right = column * width // grid, (column + 1) * width // grid
            pixels[top:bottom, left:right] = 127
            occluded.append(Image.fromarray(pixels))
    logits = []
    for start in range(0, len(occluded), 16):
        embeddings = encoder.encode_images(occluded[start : start + 16])
        logits.extend(encoder.logits(embeddings, text)[:, 0].tolist())
    return baseline - np.array(logits).reshape(grid, grid)


def siglip_prompt_logits(encoder: Any, image: Image.Image, prompts: list[str]) -> dict[str, float]:
    """Return SigLIP's logit for each prompt on one image."""
    logits = encoder.logits(encoder.encode_images([image]), encoder.encode_text(prompts))
    return {prompt: float(value) for prompt, value in zip(prompts, logits[0].tolist())}


def text_similarity(encoder: Any, prompts: list[str]) -> pd.DataFrame:
    """Return cosine similarity between prompt embeddings."""
    embeddings = encoder.encode_text(prompts)
    return pd.DataFrame((embeddings @ embeddings.T).cpu().numpy(), index=prompts, columns=prompts)


def reproduce(model: GroundedSAM, image: Image.Image, prompt: str, runs: int = 3) -> dict[str, Any]:
    """Run a prompt repeatedly in fresh calls and report whether the result is identical."""
    results = [model.detect(image, prompt) for _ in range(runs)]
    top = [(round(r[0].score, 4), r[0].label) if r else None for r in results]
    return {"prompt": prompt, "top_results": top, "identical": len(set(top)) == 1}


def load_gt_segmentations(config: dict[str, Any], classes: list[str]) -> list[dict[str, Any]]:
    """Return held-out ground-truth masks as full-image arrays, under their parent class."""
    import fiftyone as fo

    require_selection()
    manifest = load_manifest(config)
    heldout = manifest[manifest["role"] == "heldout"]
    allowed = set(heldout["image_id"])
    dataset = fo.load_dataset(config["open_images"]["fiftyone_dataset"])
    box_labels = load_box_labels(config)
    parent = {label: name for name in classes for label in box_labels[name]}
    records = []
    for sample in dataset.exists("segmentations").iter_samples():
        image_id = sample.filepath.replace("\\", "/").rsplit("/", 1)[-1][:-4]
        if image_id not in allowed:
            continue
        split = sample.filepath.replace("\\", "/").split("/")[-3]
        width, height = load_image(image_path(config, split, image_id)).size
        for detection in sample.segmentations.detections:
            if detection.label not in parent or detection.mask is None:
                continue
            x, y, w, h = detection.bounding_box
            left, top = int(round(x * width)), int(round(y * height))
            right, bottom = int(round((x + w) * width)), int(round((y + h) * height))
            target = np.zeros((height, width), dtype=bool)
            resized = Image.fromarray(detection.mask.astype(np.uint8) * 255).resize(
                (max(right - left, 1), max(bottom - top, 1))
            )
            patch = np.asarray(resized) > 127
            target[top : top + patch.shape[0], left : left + patch.shape[1]] = patch[
                : target[top:].shape[0], : target[:, left:].shape[1]
            ]
            records.append(
                {
                    "split": split,
                    "image_id": image_id,
                    "label": parent[detection.label],
                    "box": [left, top, right, bottom],
                    "mask": target,
                }
            )
    return records


def evaluate_masks(
    model: GroundedSAM,
    config: dict[str, Any],
    prompts: dict[str, str],
    records: list[dict[str, Any]],
) -> pd.DataFrame:
    """Compare masks from detector boxes with masks from ground-truth boxes, an upper bound."""
    rows = []
    threshold = require_selection()["detector_threshold"]
    for record in tqdm(records, desc="masks"):
        image = load_image(image_path(config, record["split"], record["image_id"]))
        oracle = Instance(record["label"], 1.0, tuple(float(v) for v in record["box"]))
        model.segment(image, [oracle])
        detections = [
            d for d in model.detect(image, prompts[record["label"]]) if d.score >= threshold
        ]
        detected_iou, box_overlap = 0.0, 0.0
        if detections:
            overlaps = box_iou(
                np.array([d.box_xyxy for d in detections]), np.array([record["box"]], dtype=float)
            )[:, 0]
            best = detections[int(overlaps.argmax())]
            box_overlap = float(overlaps.max())
            model.segment(image, [best])
            detected_iou = mask_iou(best.mask, record["mask"])
        rows.append(
            {
                "image_id": record["image_id"],
                "class_name": record["label"],
                "oracle_box_mask_iou": mask_iou(oracle.mask, record["mask"]),
                "detected_mask_iou": detected_iou,
                "detected_box_iou": box_overlap,
            }
        )
    return pd.DataFrame(rows)


def evaluate_rescoring(
    model: GroundedSAM,
    encoder: Any,
    config: dict[str, Any],
    predictions: pd.DataFrame,
    role: str,
    condition: str,
    prompts: dict[str, dict[str, str]],
    encoder_condition: str,
    top_k: int = 10,
) -> pd.DataFrame:
    """Compare AP50 from detector scores with AP50 after SigLIP checks each masked crop.

    Each model gets its own frozen prompt wording, since the best wording differs between them.
    """
    from openwarehouse.evaluation.metrics import detection_ap
    from openwarehouse.experiments import restrict_to_manifest

    if role == "heldout":
        require_selection()
    ground_truth = _load_ground_truth()
    box_labels = load_box_labels(config)
    rows = restrict_to_manifest(predictions, load_manifest(config))
    results = []
    for class_name in config["boxable_classes"]:
        group = rows[
            (rows["role"] == role)
            & (rows["class_name"] == class_name)
            & (rows["condition"] == condition)
        ]
        truth, detector, fused, siglip_only = {}, {}, {}, {}
        for record in tqdm(group.itertuples(), total=len(group), desc=f"rescore {class_name}"):
            boxes = [
                (b["box"], b["is_group_of"])
                for b in ground_truth.get(record.image_id, {}).get("boxes", [])
                if b["label"] in box_labels[class_name]
            ]
            if record.is_positive and not boxes:
                continue
            truth[record.image_id] = boxes if record.is_positive else []
            kept = [
                Instance(d[2], d[0], tuple(d[1]))
                for d in sorted(record.detections, key=lambda d: -d[0])[:top_k]
            ]
            image = load_image(image_path(config, record.split, record.image_id))
            kept = rescore_instances(
                image, model.segment(image, kept), encoder, prompts[class_name][encoder_condition]
            )
            detector[record.image_id] = [(d.score, list(d.box_xyxy)) for d in kept]
            fused[record.image_id] = [
                ((d.score * d.encoder_score) ** 0.5, list(d.box_xyxy)) for d in kept
            ]
            siglip_only[record.image_id] = [(d.encoder_score, list(d.box_xyxy)) for d in kept]
        iou = config["evaluation"]["iou_threshold"]
        results.append(
            {
                "role": role,
                "class_name": class_name,
                "images": len(truth),
                "detector_ap50": detection_ap(detector, truth, iou),
                "siglip_only_ap50": detection_ap(siglip_only, truth, iou),
                "fused_ap50": detection_ap(fused, truth, iou),
            }
        )
    return pd.DataFrame(results)


FAILURE_DIR = OUTPUT_DIR / "failure"
