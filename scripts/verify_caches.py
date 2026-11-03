"""Check that the cached model outputs still match the current code and pinned models.

Reruns both models on a seeded sample of the original batches and compares with the caches.
Batches are rebuilt exactly as the original run formed them (same images, same order), because
detector scores depend on batch composition (up to 0.05). Acceptance criteria are fixed here,
before the run:

- detector: top score within 0.001 of the cache for at least 99% of sampled image-prompt pairs,
  and no decision flips at the frozen threshold;
- SigLIP: logit within 0.01 of the cache for at least 99% of sampled image-prompt pairs.

A batch that fails while its neighbours pass most likely comes from a resumed original run whose
batch boundaries differed; failures are reported by batch so that can be seen.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from openwarehouse.config import load_config
from openwarehouse.experiments import (
    OUTPUT_DIR,
    build_prompts,
    image_path,
    load_detector_predictions,
    load_image,
    load_manifest,
    require_selection,
)
from openwarehouse.models.vision_language import VisionLanguageEncoder
from openwarehouse.segmentation.grounded_sam import GroundedSAM
from openwarehouse.utils import resolve_device, set_seed

RESULTS = Path("docs/results")
SHARE = 0.10
DETECTOR_TOLERANCE, SIGLIP_TOLERANCE, REQUIRED_SHARE = 0.001, 0.01, 0.99


def sample_batches(n_batches: int, rng: np.random.Generator) -> list[int]:
    count = max(1, int(round(SHARE * n_batches)))
    return sorted(rng.choice(n_batches, count, replace=False).tolist())


def verify_detector(config, device, rng):
    manifest = load_manifest(config)
    prompts = build_prompts(config)
    cached = load_detector_predictions()
    top = {
        (r.image_id, r.class_name, r.condition): max((d[0] for d in r.detections), default=0.0)
        for r in cached.itertuples()
    }
    threshold = require_selection()["detector_threshold"]
    batch_size = config["inference"]["detector_batch_size"]
    model = GroundedSAM(config, device=device)
    rows = []
    jobs = []
    for class_name, group in manifest.groupby("class_name"):
        if class_name not in prompts:
            continue
        images = list(zip(group["split"], group["image_id"]))
        for condition, prompt in prompts[class_name].items():
            batches = [images[i : i + batch_size] for i in range(0, len(images), batch_size)]
            for index in sample_batches(len(batches), rng):
                jobs.append((class_name, condition, prompt, index, batches[index]))
    for class_name, condition, prompt, index, batch in tqdm(jobs, desc="detector batches"):
        loaded = [load_image(image_path(config, s, i)) for s, i in batch]
        for (split, image_id), found in zip(batch, model.detect_batch(loaded, prompt)):
            new = max((d.score for d in found), default=0.0)
            old = top[(image_id, class_name, condition)]
            rows.append(
                {
                    "model": "detector",
                    "class_name": class_name,
                    "condition": condition,
                    "batch": index,
                    "image_id": image_id,
                    "cached": old,
                    "rerun": round(new, 4),
                    "abs_diff": abs(round(new, 4) - old),
                    "decision_flip": (old >= threshold) != (round(new, 4) >= threshold),
                }
            )
    del model
    torch.cuda.empty_cache()
    return rows


def verify_detector_file_order(config, device, rng, classes):
    """Rebuild batches from the cache file's own order (the order the original run used).

    The manifest-order reconstruction assumes the original batches contained only images in the
    current manifest. For classes scored before the leakage exclusions were applied, the original
    batches also held images excluded later, so only the file order reproduces them.
    """
    from openwarehouse.experiments import PREDICTIONS_PATH, read_prediction_rows

    rows_in_file, _ = read_prediction_rows(PREDICTIONS_PATH)
    prompts = build_prompts(config)
    threshold = require_selection()["detector_threshold"]
    batch_size = config["inference"]["detector_batch_size"]
    model = GroundedSAM(config, device=device)
    rows = []
    for class_name in classes:
        for condition, prompt in prompts[class_name].items():
            records = [r for r in rows_in_file if r["class_name"] == class_name and r["condition"] == condition]
            batches = [records[i : i + batch_size] for i in range(0, len(records), batch_size)]
            for index in sample_batches(len(batches), rng):
                batch = batches[index]
                loaded = [load_image(image_path(config, r["split"], r["image_id"])) for r in batch]
                for record, found in zip(batch, model.detect_batch(loaded, prompt)):
                    new = round(max((d.score for d in found), default=0.0), 4)
                    old = max((d[0] for d in record["detections"]), default=0.0)
                    rows.append({"model": "detector (file-order batches)", "class_name": class_name, "condition": condition,
                                 "batch": index, "image_id": record["image_id"], "cached": old, "rerun": new,
                                 "abs_diff": abs(new - old), "decision_flip": (old >= threshold) != (new >= threshold)})
    del model
    torch.cuda.empty_cache()
    return rows


def verify_siglip(config, device, rng):
    manifest = load_manifest(config)
    prompts = build_prompts(config)
    cached = pd.read_csv(OUTPUT_DIR / "siglip_scores.csv", dtype={"image_id": str})
    cached = cached.set_index(["image_id", "class_name", "condition"])["logit"]
    images = manifest[["split", "image_id"]].drop_duplicates().reset_index(drop=True)
    batch_size = config["inference"]["siglip_batch_size"]
    n_batches = (len(images) + batch_size - 1) // batch_size
    chosen = sample_batches(n_batches, rng)
    encoder = VisionLanguageEncoder(config["models"]["siglip"], device=device)
    embeddings = {}
    for index in tqdm(chosen, desc="siglip batches"):
        batch = images.iloc[index * batch_size : (index + 1) * batch_size]
        loaded = [load_image(image_path(config, s, i)) for s, i in batch.itertuples(index=False)]
        for image_id, vector in zip(batch["image_id"], encoder.encode_images(loaded)):
            embeddings[image_id] = (index, vector)
    rows = []
    sampled = manifest[manifest["image_id"].isin(embeddings) & manifest["class_name"].isin(prompts)]
    for class_name, group in sampled.groupby("class_name"):
        conditions = list(prompts[class_name])
        text = encoder.encode_text([prompts[class_name][c] for c in conditions])
        for image_id in group["image_id"].unique():
            index, vector = embeddings[image_id]
            logits = encoder.logits(vector[None], text).detach().cpu().numpy()[0]
            for condition, logit in zip(conditions, logits):
                old = float(cached[(image_id, class_name, condition)])
                rows.append(
                    {
                        "model": "siglip",
                        "class_name": class_name,
                        "condition": condition,
                        "batch": index,
                        "image_id": image_id,
                        "cached": old,
                        "rerun": float(logit),
                        "abs_diff": abs(float(logit) - old),
                        "decision_flip": False,
                    }
                )
    return rows


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file-order", nargs="+", metavar="CLASS",
                        help="only rerun these classes, with batches rebuilt from the cache file order")
    args = parser.parse_args()
    config = load_config("open_images", "poc")
    set_seed(config["seed"])
    device = resolve_device(config["device"])
    rng = np.random.default_rng(config["seed"])
    if args.file_order:
        frame = pd.DataFrame(verify_detector_file_order(config, device, rng, args.file_order))
        frame.to_csv(RESULTS / "verification_cache_file_order.csv", index=False)
        within = float((frame["abs_diff"] <= DETECTOR_TOLERANCE).mean())
        summary = {"classes": args.file_order, "pairs": len(frame), "share_within_tolerance": within,
                   "max_abs_diff": float(frame["abs_diff"].max()), "decision_flips": int(frame["decision_flip"].sum()),
                   "passed": bool(within >= REQUIRED_SHARE and frame["decision_flip"].sum() == 0)}
        (RESULTS / "verification_cache_file_order.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return
    frame = pd.DataFrame(verify_detector(config, device, rng) + verify_siglip(config, device, rng))
    frame.to_csv(RESULTS / "verification_cache_sample.csv", index=False)
    summary = {}
    for model, tolerance in (("detector", DETECTOR_TOLERANCE), ("siglip", SIGLIP_TOLERANCE)):
        part = frame[frame["model"] == model]
        within = float((part["abs_diff"] <= tolerance).mean())
        failing_batches = part[part["abs_diff"] > tolerance].groupby(["class_name", "condition", "batch"]).size()
        summary[model] = {
            "pairs": len(part),
            "batches": int(part.groupby(["class_name", "condition", "batch"]).ngroups) if model == "detector" else int(part["batch"].nunique()),
            "tolerance": tolerance,
            "share_within_tolerance": within,
            "max_abs_diff": float(part["abs_diff"].max()),
            "median_abs_diff": float(part["abs_diff"].median()),
            "decision_flips": int(part["decision_flip"].sum()),
            "batches_with_any_failure": int(len(failing_batches)),
            "passed": bool(within >= REQUIRED_SHARE and part["decision_flip"].sum() == 0),
        }
    (RESULTS / "verification_cache_sample.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
