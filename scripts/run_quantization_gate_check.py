"""Measure SigLIP precision drift at the frozen gate cuts, on dev images only.

All SigLIP results were produced under fp16 autocast, and the presence check compares raw logits
with absolute cuts, so any precision change moves decisions directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from openwarehouse.config import load_config
from openwarehouse.evaluation.metrics import auroc
from openwarehouse.experiments import build_prompts, image_path, load_image, load_manifest
from openwarehouse.gate import GATE_SELECTION_PATH
from openwarehouse.models.vision_language import VisionLanguageEncoder, _as_embedding

RESULTS = Path("docs/results")
CLASSES = ["Forklift truck", "Truck", "Shelf"]


@torch.inference_mode()
def image_logits(encoder, images, text, autocast):
    out = []
    for start in range(0, len(images), 16):
        inputs = encoder.processor(images=images[start : start + 16], return_tensors="pt").to(encoder.device)
        with torch.autocast(encoder.device.split(":")[0], enabled=autocast):
            emb = _as_embedding(encoder.model.get_image_features(**inputs))
        emb = torch.nn.functional.normalize(emb.float(), dim=-1)
        out.append(encoder.logits(emb, text).float().cpu().numpy()[:, 0])
    return np.concatenate(out)


def compare(name, class_name, y, reference, other, cut):
    diff = np.abs(reference - other)
    return {
        "comparison": name, "class_name": class_name, "images": len(y),
        "median_abs_logit_diff": float(np.median(diff)), "p95_abs_logit_diff": float(np.quantile(diff, 0.95)),
        "max_abs_logit_diff": float(diff.max()), "gate_cut": cut,
        "gate_flips": int(((reference >= cut) != (other >= cut)).sum()),
        "auroc_reference": auroc(y, reference), "auroc_other": auroc(y, other),
    }


def main() -> None:
    config = load_config("open_images", "poc")
    cuts = json.loads(GATE_SELECTION_PATH.read_text(encoding="utf-8"))
    selection = json.loads(Path("docs/results/selection.json").read_text(encoding="utf-8"))
    prompts = build_prompts(config)
    manifest = load_manifest(config)
    data = {}
    for class_name in CLASSES:
        rows = manifest[(manifest["role"] == "dev") & (manifest["class_name"] == class_name)]
        data[class_name] = (
            rows["is_positive"].to_numpy(bool),
            [load_image(image_path(config, s, i)) for s, i in zip(rows["split"], rows["image_id"])],
            prompts[class_name][selection["siglip_condition"]],
        )
    results = []
    for device in ("cuda", "cpu"):
        encoder = VisionLanguageEncoder(config["models"]["siglip"], device=device)
        for class_name, (y, images, prompt) in data.items():
            text = encoder.encode_text([prompt])
            fp32 = image_logits(encoder, images, text, autocast=False)
            if device == "cuda":
                fp16 = image_logits(encoder, images, text, autocast=True)
                results.append(compare("fp32 vs fp16 autocast (GPU, as used for all results)", class_name, y, fp32, fp16, cuts[class_name]))
            else:
                quantized = VisionLanguageEncoder.__new__(VisionLanguageEncoder)
                quantized.device, quantized.processor = "cpu", encoder.processor
                quantized.model = torch.ao.quantization.quantize_dynamic(encoder.model, {torch.nn.Linear}, dtype=torch.qint8)
                int8 = image_logits(quantized, images, quantized.encode_text([prompt]), autocast=False)
                results.append(compare("fp32 vs dynamic int8 (CPU)", class_name, y, fp32, int8, cuts[class_name]))
            print(results[-1])
        del encoder
        torch.cuda.empty_cache()
    pd.DataFrame(results).to_csv(RESULTS / "validation_siglip_precision_gate.csv", index=False)


if __name__ == "__main__":
    main()
