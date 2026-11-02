"""Zero-shot prompt-variation experiments. Results are cached so notebooks can reload them."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from openwarehouse.config import CONFIG_DIR, PROJECT_ROOT
from openwarehouse.data.leakage import find_exclusions, fingerprint_images, load_authors
from openwarehouse.data.open_images import load_box_labels
from openwarehouse.evaluation.metrics import auroc, bootstrap_ci, detection_ap
from openwarehouse.models.vision_language import VisionLanguageEncoder
from openwarehouse.segmentation.grounded_sam import GroundedSAM

OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "poc"
EXCLUSIONS_PATH = OUTPUT_DIR / "leakage_exclusions.csv"
SELECTION_PATH = OUTPUT_DIR / "selection.json"
PREDICTIONS_PATH = OUTPUT_DIR / "detector_predictions.jsonl.gz"
PREDICTIONS_META_PATH = OUTPUT_DIR / "detector_predictions.meta.json"
SIGLIP_META_PATH = OUTPUT_DIR / "siglip_scores.meta.json"
BASELINE_CONDITION = "article"


def build_prompts(config: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Expand every class through every prompt condition."""
    return {
        class_name: {
            condition: template.format(**fields)
            for condition, template in config["prompt_conditions"].items()
        }
        for class_name, fields in config["prompt_classes"].items()
    }


def image_path(config: dict[str, Any], split: str, image_id: str) -> Path:
    root = PROJECT_ROOT / config["open_images"]["dataset_dir"]
    return root / split / "data" / f"{image_id}.jpg"


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def build_leakage_exclusions(config: dict[str, Any]) -> pd.DataFrame:
    """Record held-out images linked to dev data. Uses no model outputs or results."""
    manifest = pd.read_csv(OUTPUT_DIR / "manifest.csv", dtype={"image_id": str})
    images = manifest[["split", "image_id", "role"]].drop_duplicates().reset_index(drop=True)
    images, hashes = fingerprint_images(images, config)
    images = load_authors(images, config)
    exclusions = find_exclusions(images, hashes, config["evaluation"]["near_duplicate_distance"])
    exclusions.to_csv(EXCLUSIONS_PATH, index=False)
    return exclusions


def load_manifest(config: dict[str, Any]) -> pd.DataFrame:
    """Load the manifest, keeping downloaded images that are not linked across roles."""
    manifest = pd.read_csv(OUTPUT_DIR / "manifest.csv", dtype={"image_id": str})
    if not EXCLUSIONS_PATH.exists():
        raise RuntimeError("Run build_leakage_exclusions before loading the manifest.")
    excluded = set(pd.read_csv(EXCLUSIONS_PATH, dtype={"image_id": str})["image_id"])
    manifest = manifest[~manifest["image_id"].isin(excluded)].reset_index(drop=True)
    exists = {
        (split, image_id): image_path(config, split, image_id).exists()
        for split, image_id in manifest[["split", "image_id"]].drop_duplicates().itertuples(index=False)
    }
    keep = [exists[key] for key in zip(manifest["split"], manifest["image_id"])]
    return manifest[keep].reset_index(drop=True)


def export_ground_truth(config: dict[str, Any]) -> Path:
    """Copy boxes from the FiftyOne dataset into a JSON file the experiments can read quickly."""
    import fiftyone as fo

    dataset = fo.load_dataset(config["open_images"]["fiftyone_dataset"])
    records = {}
    for sample in dataset.iter_samples(progress=True):
        image_id = Path(sample.filepath).stem
        width_height = load_image(Path(sample.filepath)).size
        width, height = width_height
        boxes = []
        field = sample["detections"] if sample.has_field("detections") else None
        for detection in field.detections if field else []:
            x, y, w, h = detection.bounding_box
            boxes.append(
                {
                    "label": detection.label,
                    "box": [x * width, y * height, (x + w) * width, (y + h) * height],
                    "is_group_of": bool(detection["IsGroupOf"]),
                }
            )
        records[image_id] = {"size": [width, height], "boxes": boxes}
    path = OUTPUT_DIR / "ground_truth.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


def run_siglip(config: dict[str, Any], device: str) -> pd.DataFrame:
    """Score every verified image-label pair under every prompt condition, then cache it."""
    path = OUTPUT_DIR / "siglip_scores.csv"
    # Settings the cached logits depend on. A cache made before this check existed is adopted
    # with the current settings, which is what produced it at review time.
    _check_cache_settings(
        {
            "siglip": config["models"]["siglip"],
            "prompt_conditions": config["prompt_conditions"],
            "prompt_classes": config["prompt_classes"],
        },
        SIGLIP_META_PATH,
    )
    if path.exists():
        return pd.read_csv(path, dtype={"image_id": str})

    manifest = load_manifest(config)
    prompts = build_prompts(config)
    encoder = VisionLanguageEncoder(config["models"]["siglip"], device=device)
    images = manifest[["split", "image_id"]].drop_duplicates().reset_index(drop=True)
    batch_size = config["inference"]["siglip_batch_size"]
    embeddings = []
    for start in tqdm(range(0, len(images), batch_size), desc="siglip images"):
        batch = images.iloc[start : start + batch_size]
        loaded = [load_image(image_path(config, s, i)) for s, i in batch.itertuples(index=False)]
        embeddings.append(encoder.encode_images(loaded).cpu())
    image_embeddings = torch.cat(embeddings).to(device)
    index = {key: i for i, key in enumerate(images.itertuples(index=False, name=None))}

    rows = []
    for class_name, group in manifest.groupby("class_name"):
        if class_name not in prompts:
            continue
        conditions = list(prompts[class_name])
        text = encoder.encode_text([prompts[class_name][c] for c in conditions])
        ensemble = torch.nn.functional.normalize(text.mean(0, keepdim=True), dim=-1)
        text = torch.cat([text, ensemble])
        positions = [index[(s, i)] for s, i in zip(group["split"], group["image_id"])]
        logits = encoder.logits(image_embeddings[positions], text).detach().cpu().numpy()
        for j, condition in enumerate([*conditions, "ensemble"]):
            rows.append(group.assign(condition=condition, logit=logits[:, j]))
    scores = pd.concat(rows)[
        ["image_id", "split", "role", "class_name", "is_positive", "condition", "logit"]
    ]
    scores.to_csv(path, index=False)
    return scores


def _check_cache_settings(settings: dict[str, Any], path: Path) -> None:
    """Refuse to reuse or extend a cache that was made with different settings."""
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != settings:
            raise RuntimeError(f"Cache described by {path.name} used different settings.")
    else:
        path.write_text(json.dumps(settings, indent=2), encoding="utf-8")


def _check_prediction_settings(config: dict[str, Any], path: Path = PREDICTIONS_META_PATH) -> None:
    """Refuse to extend a prediction cache that was made with different model settings."""
    _check_cache_settings(
        {"detector": config["models"]["detector"], "inference": config["inference"]}, path
    )


def run_detector(config: dict[str, Any], device: str) -> pd.DataFrame:
    """Run Grounding DINO for every verified image-label pair and condition, resuming if stopped."""
    path = PREDICTIONS_PATH
    _check_prediction_settings(config)
    done: set[tuple[str, str, str]] = set()
    if path.exists():
        rows, is_complete = read_prediction_rows(path)
        done = {(row["image_id"], row["class_name"], row["condition"]) for row in rows}
        # Appending after a truncated gzip stream would make the file unreadable.
        if not is_complete:
            temporary = path.with_suffix(".tmp")
            with gzip.open(temporary, "wt", encoding="utf-8") as handle:
                handle.writelines(json.dumps(row) + "\n" for row in rows)
            os.replace(temporary, path)

    manifest = load_manifest(config)
    prompts = build_prompts(config)
    jobs = []
    for class_name, group in manifest.groupby("class_name"):
        if class_name in prompts:
            for condition, prompt in prompts[class_name].items():
                pending = [
                    (s, i)
                    for s, i in zip(group["split"], group["image_id"])
                    if (i, class_name, condition) not in done
                ]
                if pending:
                    jobs.append((class_name, condition, prompt, pending))
    total = sum(len(job[3]) for job in jobs)
    if total:
        model = GroundedSAM(config, device=device)
        batch_size = config["inference"]["detector_batch_size"]
        with gzip.open(path, "at", encoding="utf-8") as output, tqdm(total=total) as progress:
            for class_name, condition, prompt, pending in jobs:
                for start in range(0, len(pending), batch_size):
                    batch = pending[start : start + batch_size]
                    images = [load_image(image_path(config, s, i)) for s, i in batch]
                    results = model.detect_batch(images, prompt)
                    for (split, image_id), instances in zip(batch, results):
                        output.write(
                            json.dumps(
                                {
                                    "image_id": image_id,
                                    "split": split,
                                    "class_name": class_name,
                                    "condition": condition,
                                    "detections": [
                                        [round(d.score, 4), [round(v, 1) for v in d.box_xyxy], d.label]
                                        for d in instances
                                    ],
                                }
                            )
                            + "\n"
                        )
                    output.flush()
                    progress.update(len(batch))
    return load_detector_predictions()


def read_prediction_rows(path: Path) -> tuple[list[dict[str, Any]], bool]:
    """Read every intact row, and report whether the file ended cleanly."""
    rows: list[dict[str, Any]] = []
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                rows.append(json.loads(line))
    except (EOFError, zlib.error, gzip.BadGzipFile, json.JSONDecodeError):
        return rows, False
    return rows, True


def load_detector_predictions() -> pd.DataFrame:
    return pd.DataFrame(read_prediction_rows(PREDICTIONS_PATH)[0])


def restrict_to_manifest(rows: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Keep rows for manifest labels, with labels and roles taken only from the manifest.

    Fails if any condition lacks a result, so a stale cache cannot silently change what is scored.
    """
    keys = ["split", "image_id", "class_name"]
    expected = manifest[manifest["class_name"].isin(rows["class_name"].unique())]
    stripped = rows.drop(columns=["is_positive", "role"], errors="ignore")
    merged = stripped.merge(expected[[*keys, "is_positive", "role"]], on=keys)
    counts = merged.groupby("condition").size() if len(merged) else pd.Series(dtype=int)
    if len(counts) == 0 or (counts != len(expected)).any():
        raise ValueError(f"Results do not cover the manifest: {counts.to_dict()}")
    return merged


def config_hash(names: tuple[str, ...] = ("poc", "open_images")) -> str:
    """Hash the experiment definition, recording what was fixed when a selection was frozen."""
    digest = hashlib.sha256()
    for name in names:
        digest.update((CONFIG_DIR / f"{name}.yaml").read_bytes())
    return digest.hexdigest()


def freeze_selection(
    siglip_dev: pd.DataFrame,
    detector_dev: pd.DataFrame,
    threshold: float,
    path: Path = SELECTION_PATH,
) -> dict[str, Any]:
    """Save choices made from dev data. Held-out scoring is refused until this file exists.

    A selection can only be frozen once, so held-out results cannot prompt a quiet re-selection.
    """
    if path.exists():
        raise RuntimeError(f"{path} is already frozen. Delete it deliberately and say so.")
    selection = {
        "siglip_condition": select_condition(siglip_dev, "auroc"),
        "detector_condition": select_condition(detector_dev, "image_auroc"),
        "detector_threshold": threshold,
        "config_sha256": config_hash(),
    }
    path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    return selection


def require_selection(path: Path = SELECTION_PATH) -> dict[str, Any]:
    """Return the frozen selection, or fail if held-out scoring is attempted before it."""
    if not path.exists():
        raise RuntimeError("Freeze the dev selection before scoring held-out data.")
    selection = json.loads(path.read_text(encoding="utf-8"))
    if selection["config_sha256"] != config_hash():
        raise RuntimeError("The experiment config changed after the selection was frozen.")
    return selection


def _check_role(role: str) -> None:
    if role == "heldout":
        require_selection()


def select_detector_threshold(
    predictions: pd.DataFrame, condition: str, config: dict[str, Any]
) -> float:
    """Choose the detector score cutoff with the best image-level F1 on dev data."""
    rows = restrict_to_manifest(predictions, load_manifest(config))
    rows = rows[(rows["role"] == "dev") & (rows["condition"] == condition)]
    scores = np.array([max((d[0] for d in ds), default=0.0) for ds in rows["detections"]])
    labels = rows["is_positive"].to_numpy(dtype=bool)
    candidates = np.round(np.arange(0.10, 0.91, 0.01), 2)
    f1 = [
        2 * (hit := (scores >= t)[labels].sum()) / max(hit + (scores >= t).sum() + labels.sum() - hit, 1)
        for t in candidates
    ]
    return float(candidates[int(np.argmax(f1))])


def _load_ground_truth() -> dict[str, Any]:
    return json.loads((OUTPUT_DIR / "ground_truth.json").read_text(encoding="utf-8"))


def summarize_siglip(
    scores: pd.DataFrame, config: dict[str, Any], role: str, with_ci: bool = True
) -> pd.DataFrame:
    """Compute AUROC by class and condition, with a paired CI against the baseline condition."""
    _check_role(role)
    rows = []
    samples = config["evaluation"]["bootstrap_samples"]
    scores = restrict_to_manifest(scores, load_manifest(config))
    scores = scores[scores["role"] == role]
    for class_name, group in scores.groupby("class_name"):
        table = group.pivot_table(index=["split", "image_id"], columns="condition", values="logit")
        labels = group.drop_duplicates(["split", "image_id"]).set_index(["split", "image_id"])
        y = labels.loc[table.index, "is_positive"].to_numpy(dtype=bool)
        if y.sum() < 5 or (~y).sum() < 5:
            continue
        base = table[BASELINE_CONDITION].to_numpy()
        for condition in table.columns:
            values = table[condition].to_numpy()
            row = {
                "role": role,
                "class_name": class_name,
                "condition": condition,
                "n_pos": int(y.sum()),
                "n_neg": int((~y).sum()),
                "auroc": auroc(y, values),
            }
            if with_ci:
                row["ci_low"], row["ci_high"] = bootstrap_ci(
                    lambda i: auroc(y[i], values[i]), len(y), samples
                )
                row["delta_low"], row["delta_high"] = bootstrap_ci(
                    lambda i: auroc(y[i], values[i]) - auroc(y[i], base[i]), len(y), samples
                )
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_detector(
    predictions: pd.DataFrame, config: dict[str, Any], role: str, with_ci: bool = True
) -> pd.DataFrame:
    """Compute AP50 for boxable classes and max-score image-level AUROC for every class."""
    _check_role(role)
    ground_truth = _load_ground_truth()
    box_labels = load_box_labels(config)
    predictions = restrict_to_manifest(predictions, load_manifest(config))
    predictions = predictions[predictions["role"] == role].copy()
    predictions["max_score"] = [
        max((d[0] for d in ds), default=0.0) for ds in predictions["detections"]
    ]

    rows = []
    iou = config["evaluation"]["iou_threshold"]
    samples = config["evaluation"]["bootstrap_samples"]
    for (class_name, condition), group in predictions.groupby(["class_name", "condition"]):
        group = group.reset_index(drop=True)
        y = group["is_positive"].to_numpy(dtype=bool)
        if y.sum() < 5 or (~y).sum() < 5:
            continue
        values = group["max_score"].to_numpy()
        row = {
            "role": role,
            "class_name": class_name,
            "condition": condition,
            "n_pos": int(y.sum()),
            "n_neg": int((~y).sum()),
            "image_auroc": auroc(y, values),
        }
        if with_ci:
            row["auroc_low"], row["auroc_high"] = bootstrap_ci(
                lambda i: auroc(y[i], values[i]), len(y), samples
            )
        if class_name in config["boxable_classes"]:
            keys = group["split"] + "/" + group["image_id"]
            truth, preds = {}, {}
            for key, record, positive in zip(keys, group.itertuples(), y):
                gt_boxes = ground_truth.get(record.image_id, {}).get("boxes", [])
                boxes = [
                    (b["box"], b["is_group_of"])
                    for b in gt_boxes
                    if b["label"] in box_labels[class_name]
                ]
                # A verified positive with no boxes was never box-annotated for this class, so
                # scoring it would turn correct detections into false positives.
                if positive and not boxes:
                    continue
                truth[key] = boxes if positive else []
                preds[key] = [(d[0], d[1]) for d in record.detections]
            row["ap50_images"] = len(truth)
            row["ap50"] = detection_ap(preds, truth, iou)
            key_list = list(truth)

            def resampled_ap(index: np.ndarray) -> float:
                sampled = [(f"{n}:{key_list[i]}", key_list[i]) for n, i in enumerate(index)]
                return detection_ap(
                    {new: preds[old] for new, old in sampled},
                    {new: truth[old] for new, old in sampled},
                    iou,
                )

            if with_ci:
                row["ap50_low"], row["ap50_high"] = bootstrap_ci(
                    resampled_ap, len(key_list), min(samples, 300)
                )
        rows.append(row)
    return pd.DataFrame(rows)


def select_condition(summary: pd.DataFrame, metric: str) -> str:
    """Choose the condition with the best mean metric from a dev-only summary."""
    if set(summary["role"]) != {"dev"}:
        raise ValueError("Conditions may only be selected from dev results.")
    return str(summary.groupby("condition")[metric].mean().idxmax())
