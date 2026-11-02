"""Attribute detector scores to caption tokens and measure SigLIP's response to negation."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from openwarehouse.config import load_config
from openwarehouse.evaluation.metrics import auroc
from openwarehouse.experiments import image_path, load_image, load_manifest, require_selection
from openwarehouse.failure_analysis import FAILURE_DIR, images_for, text_similarity
from openwarehouse.models.vision_language import VisionLanguageEncoder
from openwarehouse.segmentation.grounded_sam import GroundedSAM, to_caption
from openwarehouse.utils import resolve_device, set_seed

IMAGES = 60
# Each entry is scored on images verified not to contain the class, so any detection is false.
TOKEN_PROMPTS = {
    "Forklift truck": [
        "forklift", "a forklift", "a photo of a forklift", "no forklift", "a giraffe", "giraffe",
    ],
    "Box": ["box", "a box", "a photo of a box", "no box", "a fallen box"],
}
SIGLIP_PROMPTS = {
    "Forklift truck": ["a forklift", "no forklift", "not a forklift", "a damaged forklift"],
    "Box": ["a box", "no box", "not a box", "a fallen box", "a damaged box"],
    "Person": ["a person", "no person", "not a person", "a fallen person"],
}
STRUCTURAL = {"[CLS]", "[SEP]", ".", "a", "of", "photo"}


@torch.inference_mode()
def attribute(model: GroundedSAM, image, prompt: str) -> dict:
    """Return the winning token of the top query, and that query's best object-word score."""
    inputs, outputs = model._run_detector([image], [to_caption(prompt)])
    probabilities = outputs.logits.sigmoid()[0]
    tokens = model.detector_processor.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    per_query = probabilities[:, : len(tokens)]
    top_query = int(per_query.max(dim=-1).values.argmax())
    winner = int(per_query[top_query].argmax())
    content = [i for i, token in enumerate(tokens) if token not in STRUCTURAL]
    return {
        "tokens": " ".join(tokens),
        "winning_token": tokens[winner],
        "top_score": float(per_query[top_query, winner]),
        "content_score": float(per_query[:, content].max()),
    }


def main(role: str) -> None:
    config = load_config("open_images", "poc")
    set_seed(config["seed"])
    device = resolve_device(config["device"])
    threshold = require_selection()["detector_threshold"]
    model = GroundedSAM(config, device=device)

    rows = []
    for class_name, prompts in TOKEN_PROMPTS.items():
        images = images_for(config, role, class_name, False, IMAGES, config["seed"])
        for split, image_id in tqdm(images.itertuples(index=False), total=len(images), desc=class_name):
            image = load_image(image_path(config, split, image_id))
            for prompt in prompts:
                rows.append(
                    {"role": role, "class_name": class_name, "image_id": image_id, "prompt": prompt}
                    | attribute(model, image, prompt)
                )
    tokens = pd.DataFrame(rows)
    tokens.to_csv(FAILURE_DIR / f"token_attribution_{role}.csv", index=False)
    summary = (
        tokens.assign(
            structural=tokens["winning_token"].isin(STRUCTURAL),
            fires=tokens["top_score"] >= threshold,
            content_fires=tokens["content_score"] >= threshold,
        )
        .groupby(["class_name", "prompt"])
        .agg(
            images=("image_id", "size"),
            structural_winner=("structural", "mean"),
            fire_rate=("fires", "mean"),
            content_fire_rate=("content_fires", "mean"),
            top_token=("winning_token", lambda s: s.value_counts().index[0]),
        )
    )
    summary.to_csv(FAILURE_DIR / f"token_summary_{role}.csv")
    print(summary.round(3).to_string())

    del model
    torch.cuda.empty_cache()
    encoder = VisionLanguageEncoder(config["models"]["siglip"], device=device)
    manifest = load_manifest(config)
    results = []
    for class_name, prompts in SIGLIP_PROMPTS.items():
        rows = manifest[(manifest["role"] == role) & (manifest["class_name"] == class_name)]
        labels = rows["is_positive"].to_numpy(dtype=bool)
        text = encoder.encode_text(prompts)
        logits = []
        paths = [image_path(config, s, i) for s, i in zip(rows["split"], rows["image_id"])]
        for start in tqdm(range(0, len(paths), 32), desc=f"siglip {class_name}"):
            embedded = encoder.encode_images([load_image(p) for p in paths[start : start + 32]])
            logits.append(encoder.logits(embedded, text).detach().cpu().numpy())
        logits = np.concatenate(logits)
        for index, prompt in enumerate(prompts):
            results.append(
                {
                    "role": role,
                    "class_name": class_name,
                    "prompt": prompt,
                    "n_pos": int(labels.sum()),
                    "n_neg": int((~labels).sum()),
                    "auroc": auroc(labels, logits[:, index]),
                    "pos_score_mean": float(torch.sigmoid(torch.tensor(logits[labels, index])).mean()),
                    "text_cosine_to_base": float(text_similarity(encoder, prompts).iloc[0, index]),
                    "correlation_with_base": float(np.corrcoef(logits[:, 0], logits[:, index])[0, 1]),
                }
            )
    siglip = pd.DataFrame(results)
    siglip.to_csv(FAILURE_DIR / f"siglip_negation_{role}.csv", index=False)
    print(siglip.round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dev")
