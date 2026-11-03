"""Open-vocabulary warehouse inspection demo: text prompt in, boxes and masks out.

Examples:
    python scripts/demo.py --image path/to/photo.jpg --prompt "box" --prompt "damaged box"
    python scripts/demo.py --examples      # builds docs/figures/demo_examples.png

For each image and prompt it writes, under --out:
    <image>__<prompt>.json         boxes, detector scores, SigLIP crop scores, mask areas
    <image>__<prompt>.png          overlay of boxes and masks
    <image>__<prompt>__mask<k>.png one binary mask per detection

Grounding DINO proposes boxes for the prompt, SAM segments each box, and SigLIP scores each
masked crop. The default threshold is the one frozen on development data (0.32). An empty result
means nothing passed the threshold. It is not evidence that the object is absent, and the
detector also fires on many absent objects (docs/EVALUATION.md).
"""

from __future__ import annotations

import argparse
import json
import re
import textwrap
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from openwarehouse.config import load_config
from openwarehouse.experiments import image_path, load_image, load_manifest, require_selection
from openwarehouse.pipeline import PerceptionPipeline
from openwarehouse.utils import resolve_device

COLORS = [(0.90, 0.25, 0.20), (0.18, 0.43, 0.71), (0.20, 0.63, 0.35), (0.85, 0.55, 0.15), (0.55, 0.30, 0.70)]

# Fixed examples for the report: three ordinary detections and three failures, all with warehouse
# prompts. Images are drawn by seed from verified held-out labels (not chosen by their results),
# except the fire truck, which RESULTS.md already discusses.
EXAMPLES = [
    ("Box", True, "box", "Detection: box"),
    ("Shelf", True, "shelf", "Detection: shelf"),
    ("Person", True, "person", "Detection: person"),
    ("046c751beec3d33a", None, "forklift", "Confident false positive (lookalike): fire truck, no forklift"),
    ("Forklift truck", False, "forklift", "Confident false positive (lookalike): backhoe loader, verified no forklift; fires on 14 of 14 such images"),
    ("Box", True, "a damaged box", "State ignored: fires on 97% of 60 box images; condition unlabeled"),
]


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60]


def siglip_prompt(prompt: str) -> str:
    """SigLIP's frozen wording is 'a photo of <phrase>'; add an article if the prompt has none."""
    words = prompt.strip().split()
    phrase = prompt if words and words[0].lower() in {"a", "an", "the"} else f"a {prompt}"
    return f"a photo of {phrase}"


def run(pipeline, image, prompt, threshold):
    start = time.perf_counter()
    instances = pipeline(image, prompt, min_score=threshold, encoder_prompt=siglip_prompt(prompt))
    elapsed = (time.perf_counter() - start) * 1000
    return instances, elapsed


def draw(axis, image, instances, title):
    axis.imshow(image)
    for k, instance in enumerate(instances):
        color = COLORS[k % len(COLORS)]
        if instance.mask is not None:
            overlay = np.zeros((*instance.mask.shape, 4))
            overlay[instance.mask] = (*color, 0.40)
            axis.imshow(overlay)
        x0, y0, x1, y1 = instance.box_xyxy
        axis.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color=color, linewidth=2))
        axis.text(x0, y0, f"{instance.score:.2f}", color="white", fontsize=8, va="bottom",
                  bbox={"facecolor": color, "edgecolor": "none", "pad": 1.5})
    axis.set_title(textwrap.fill(title, 48), fontsize=9)
    axis.axis("off")


def save(out_dir: Path, name: str, image, prompt, instances, elapsed, threshold):
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{slug(name)}__{slug(prompt)}"
    records = []
    for k, instance in enumerate(instances):
        mask_file = None
        if instance.mask is not None:
            mask_file = f"{stem}__mask{k}.png"
            Image.fromarray(instance.mask.astype(np.uint8) * 255).save(out_dir / mask_file)
        records.append(
            {
                "label": instance.label,
                "detector_score": round(instance.score, 4),
                "siglip_crop_score": None if instance.encoder_score is None else round(instance.encoder_score, 6),
                "box_xyxy": [round(v, 1) for v in instance.box_xyxy],
                "mask_area_px": int(instance.mask.sum()) if instance.mask is not None else None,
                "mask_file": mask_file,
            }
        )
    result = {
        "image": name,
        "prompt": prompt,
        "threshold": threshold,
        "latency_ms": round(elapsed, 1),
        "detections": records,
        "note": "An empty list means nothing passed the threshold, not that the object is absent.",
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    fig, axis = plt.subplots(figsize=(7, 5))
    draw(axis, image, instances, f'"{prompt}": {len(instances)} detection(s) ≥ {threshold}')
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png", dpi=120)
    plt.close(fig)
    return result


def example_images(config):
    manifest = load_manifest(config).drop_duplicates(["image_id", "class_name"])
    held = manifest[manifest["role"] == "heldout"]
    chosen = []
    used = set()
    for key, positive, prompt, title in EXAMPLES:
        if positive is None:
            row = manifest[manifest["image_id"] == key].iloc[0]
        else:
            pool = held[(held["class_name"] == key) & (held["is_positive"] == positive) & ~held["image_id"].isin(used)]
            row = pool.sample(1, random_state=config["seed"]).iloc[0]
        used.add(row["image_id"])
        chosen.append((row["image_id"], load_image(image_path(config, row["split"], row["image_id"])), prompt, title))
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", action="append", default=[], help="image file; repeat for several")
    parser.add_argument("--prompt", action="append", default=[], help="text prompt; repeat for several")
    parser.add_argument("--threshold", type=float, default=None, help="detector score cut (default: frozen 0.32)")
    parser.add_argument("--out", type=Path, default=Path("artifacts/demo"))
    parser.add_argument("--examples", action="store_true", help="build the report's example figure")
    args = parser.parse_args()

    config = load_config("open_images", "poc")
    threshold = args.threshold if args.threshold is not None else require_selection()["detector_threshold"]
    device = resolve_device(config["device"])
    pipeline = PerceptionPipeline(config, device=device)
    print(f"device: {device}, threshold: {threshold}")

    if args.examples:
        chosen = example_images(config)
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        summary = []
        for axis, (image_id, image, prompt, title) in zip(axes.flat, chosen):
            instances, elapsed = run(pipeline, image, prompt, threshold)
            result = save(args.out, image_id, image, prompt, instances, elapsed, threshold)
            best = max((d["detector_score"] for d in result["detections"]), default=None)
            draw(axis, image, instances, f'{title}\n"{prompt}" · {len(instances)} detection(s), top score {best}')
            summary.append({"image_id": image_id, "prompt": prompt, "case": title, "detections": len(instances), "top_score": best})
        fig.suptitle(f"Grounding DINO + SAM on held-out images, threshold {threshold} (numbers are detector scores)", fontsize=11)
        fig.tight_layout()
        Path("docs/figures").mkdir(parents=True, exist_ok=True)
        fig.savefig("docs/figures/demo_examples.png", dpi=110)
        plt.close(fig)
        Path("docs/results/demo_examples.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        for row in summary:
            print(row)
        return

    if not args.image or not args.prompt:
        parser.error("give at least one --image and one --prompt, or use --examples")
    for path in args.image:
        image = load_image(Path(path))
        for prompt in args.prompt:
            instances, elapsed = run(pipeline, image, prompt, threshold)
            result = save(args.out, Path(path).stem, image, prompt, instances, elapsed, threshold)
            scores = [d["detector_score"] for d in result["detections"]]
            print(f'{Path(path).name} | "{prompt}": {len(scores)} detection(s) {scores} in {result["latency_ms"]} ms')


if __name__ == "__main__":
    main()
