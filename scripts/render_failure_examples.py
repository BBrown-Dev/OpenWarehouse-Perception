"""Reproduce selected failures in fresh single-image calls and draw them."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from openwarehouse.config import load_config
from openwarehouse.experiments import image_path, load_image, load_manifest, require_selection
from openwarehouse.failure_analysis import FAILURE_DIR, reproduce, siglip_prompt_logits
from openwarehouse.models.vision_language import VisionLanguageEncoder
from openwarehouse.segmentation.grounded_sam import GroundedSAM
from openwarehouse.utils import resolve_device, set_seed

FIGURES = Path("docs/figures")


def draw(ax, image, instance, title):
    ax.imshow(image)
    if instance is not None:
        if instance.mask is not None:
            overlay = np.zeros((*instance.mask.shape, 4))
            overlay[instance.mask] = (0.9, 0.2, 0.1, 0.45)
            ax.imshow(overlay)
        x0, y0, x1, y1 = instance.box_xyxy
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color="#d62728", lw=2))
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def panel(model, encoder, config, split, image_id, prompts, path, threshold):
    image = load_image(image_path(config, split, image_id))
    fig, axes = plt.subplots(1, len(prompts), figsize=(4.2 * len(prompts), 4))
    logits = siglip_prompt_logits(encoder, image, prompts)
    records = []
    for ax, prompt in zip(np.atleast_1d(axes), prompts):
        found = model.detect(image, prompt)
        top = found[0] if found and found[0].score >= threshold else None
        if top is not None:
            model.segment(image, [top])
        score = f"{top.score:.2f}" if top else "none"
        draw(ax, image, top, f'"{prompt}"\nDINO {score} | SigLIP logit {logits[prompt]:.2f}')
        records.append(
            reproduce(model, image, prompt)
            | {"image_id": image_id, "siglip_logit": logits[prompt],
               "box": list(found[0].box_xyxy) if found else None}
        )
    fig.tight_layout()
    if path is not None:  # every panel is reproduced, but only the ones the report shows are saved
        fig.savefig(path, dpi=130)
    plt.close(fig)
    return records


def main() -> None:
    config = load_config("open_images", "poc")
    set_seed(config["seed"])
    device = resolve_device(config["device"])
    threshold = require_selection()["detector_threshold"]
    model = GroundedSAM(config, device=device)
    encoder = VisionLanguageEncoder(config["models"]["siglip"], device=device)
    FIGURES.mkdir(parents=True, exist_ok=True)
    tokens = pd.read_csv(FAILURE_DIR / "token_attribution_dev.csv")
    manifest = load_manifest(config).drop_duplicates("image_id").set_index("image_id")
    records = []

    wide = tokens[tokens["class_name"] == "Forklift truck"].pivot(
        index="image_id", columns="prompt", values="top_score"
    )
    ranked = (wide["a giraffe"] - wide["giraffe"]).sort_values(ascending=False).index[:3]
    for rank, image_id in enumerate(ranked):
        records += panel(
            model, encoder, config, manifest.loc[image_id, "split"], image_id,
            ["giraffe", "a giraffe", "a photo of a forklift"],
            FIGURES / f"hallucination_{rank}.png" if rank < 2 else None, threshold,
        )

    probe = pd.read_csv(FAILURE_DIR / "modifier_dev.csv")
    same = probe[(probe["prompt"] == "no person") & (probe["top_box_iou"] >= 0.9)]
    for rank, row in enumerate(same.sort_values("score", ascending=False).head(3).itertuples()):
        records += panel(
            model, encoder, config, row.split, row.image_id,
            ["a person", "no person", "a fallen person"],
            FIGURES / f"negation_{rank}.png" if rank < 1 else None, threshold,
        )
    (FAILURE_DIR / "reproductions.json").write_text(json.dumps(records, indent=1))
    print(json.dumps(records, indent=0))


if __name__ == "__main__":
    main()
