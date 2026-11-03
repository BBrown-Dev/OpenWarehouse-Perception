"""Compare precision variants of SigLIP and Grounding DINO on size, accuracy, and latency.

Dev data only, as `docs/DESIGN_DECISIONS.md` requires for compression checks.

Accuracy is 12-class zero-shot top-1 on dev images verified positive for exactly one of the
12 classes. SigLIP picks the class prompt (frozen `photo` wording) with the highest logit. Grounding
DINO gets one caption listing all 12 bare nouns and picks the class whose tokens score highest over
all queries. Other classes are only as absent as Open Images verification makes them, so top-1 is a
consistent comparison between variants, not an absolute accuracy.

Latency is per image at batch size 1: preprocessing, forward pass, and scoring, excluding JPEG
decoding. GPU timings are synchronized. Text embeddings for SigLIP are computed once, as a
deployment would cache them.
"""

from __future__ import annotations

import copy
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoModelForZeroShotObjectDetection, AutoProcessor

from openwarehouse.config import load_config, model_revision
from openwarehouse.experiments import build_prompts, image_path, load_image, load_manifest
from openwarehouse.segmentation.grounded_sam import to_caption
from openwarehouse.utils import set_seed

RESULTS = Path("docs/results")
OUT = RESULTS / "benchmark.csv"
SCRATCH = Path(os.environ.get("BENCH_SCRATCH", "artifacts/benchmark"))
SAMPLE = 200
WARMUP = 5


def size_mb(model: torch.nn.Module) -> float:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    path = SCRATCH / "weights.pt"
    torch.save(model.state_dict(), path)
    size = path.stat().st_size / 1e6
    path.unlink()
    return size


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def benchmark_images(config, classes):
    manifest = load_manifest(config)
    dev = manifest[(manifest["role"] == "dev") & manifest["class_name"].isin(classes)]
    positives = dev[dev["is_positive"]].groupby(["split", "image_id"])["class_name"].agg(list)
    single = positives[positives.map(len) == 1]
    chosen = single.sample(min(SAMPLE, len(single)), random_state=config["seed"])
    rows = [(s, i, labels[0]) for (s, i), labels in chosen.items()]
    images = [load_image(image_path(config, s, i)) for s, i, _ in rows]
    return images, [label for _, _, label in rows]


class SigLIPRunner:
    name = "SigLIP base/16-224"

    def __init__(self, config, classes):
        model_id = config["models"]["siglip"]
        revision = model_revision(model_id)
        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        self.base = AutoModel.from_pretrained(model_id, revision=revision).eval()
        prompts = build_prompts(config)
        self.texts = [prompts[c]["photo"] for c in classes]

    def prepare(self, device, dtype, autocast, quantize):
        model = copy.deepcopy(self.base)
        if quantize:
            model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        model = model.to(device=device, dtype=dtype) if not quantize else model
        self.model, self.device, self.dtype, self.autocast = model, device, dtype, autocast
        with torch.inference_mode():
            inputs = self.processor(
                text=self.texts, padding="max_length", max_length=64, truncation=True, return_tensors="pt"
            ).to(device)
            with torch.autocast(device, enabled=autocast):
                text = model.get_text_features(**inputs)
            text = text if isinstance(text, torch.Tensor) else text.pooler_output
            self.text = torch.nn.functional.normalize(text.float(), dim=-1)
        return model

    @torch.inference_mode()
    def predict(self, image) -> int:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        pixels = inputs["pixel_values"].to(self.dtype) if not self.autocast else inputs["pixel_values"]
        with torch.autocast(self.device, enabled=self.autocast):
            embedding = self.model.get_image_features(pixel_values=pixels)
        embedding = embedding if isinstance(embedding, torch.Tensor) else embedding.pooler_output
        embedding = torch.nn.functional.normalize(embedding.float(), dim=-1)
        logits = embedding @ self.text.T * self.model.logit_scale.exp().float() + self.model.logit_bias.float()
        return int(logits.argmax())


class DetectorRunner:
    name = "Grounding DINO tiny"

    def __init__(self, config, classes):
        model_id = config["models"]["detector"]
        revision = model_revision(model_id)
        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        settings = config["inference"]
        self.processor.image_processor.size = {
            "shortest_edge": settings["detector_shortest_edge"],
            "longest_edge": settings["detector_longest_edge"],
        }
        self.base = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, revision=revision).eval()
        prompts = build_prompts(config)
        nouns = [prompts[c]["bare"] for c in classes]
        self.caption = to_caption(". ".join(nouns))
        encoded = self.processor.tokenizer(self.caption, return_offsets_mapping=True)
        self.spans, position = [], 0
        for noun in nouns:
            start = self.caption.index(noun.lower(), position)
            end, position = start + len(noun), start + len(noun)
            self.spans.append(
                [t for t, (a, b) in enumerate(encoded["offset_mapping"]) if b > a and a >= start and b <= end]
            )
        assert all(self.spans), "every class needs at least one token"

    def prepare(self, device, dtype, autocast, quantize):
        model = copy.deepcopy(self.base)
        if quantize:
            model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        model = model.to(device=device, dtype=dtype) if not quantize else model
        self.model, self.device, self.dtype, self.autocast = model, device, dtype, autocast
        return model

    @torch.inference_mode()
    def predict(self, image) -> int:
        inputs = self.processor(images=image, text=self.caption, return_tensors="pt").to(self.device)
        if not self.autocast:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        with torch.autocast(self.device, enabled=self.autocast):
            outputs = self.model(**inputs)
        probabilities = outputs.logits.float().sigmoid()[0]  # queries x tokens
        scores = [float(probabilities[:, span].max()) for span in self.spans]
        return int(np.argmax(scores))


VARIANTS = [
    # label, device, dtype, autocast, quantize, note
    ("fp32", "cuda", torch.float32, False, False, "GPU"),
    ("fp16 autocast", "cuda", torch.float32, True, False, "GPU; how all reported results ran; weights stay fp32"),
    ("fp16 weights", "cuda", torch.float16, False, False, "GPU; weights and activations fp16"),
    ("fp32", "cpu", torch.float32, False, False, "CPU"),
    ("int8 dynamic", "cpu", torch.float32, False, True, "CPU; linear layers int8, rest fp32"),
]


def main() -> None:
    config = load_config("open_images", "poc")
    set_seed(config["seed"])
    classes = list(config["prompt_classes"])
    images, labels = benchmark_images(config, classes)
    truth = np.array([classes.index(label) for label in labels])
    print(f"{len(images)} images:", pd.Series(labels).value_counts().to_dict())
    done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame()
    rows = done.to_dict("records")
    for runner_class in (SigLIPRunner, DetectorRunner):
        runner = runner_class(config, classes)
        for label, device, dtype, autocast, quantize, note in VARIANTS:
            key = (runner.name, label, device.upper())
            if any((r["model"], r["precision"], r["device"]) == key for r in rows):
                continue
            model = runner.prepare(device, dtype, autocast, quantize)
            for image in images[:WARMUP]:
                runner.predict(image)
            predictions, times = [], []
            for image in images:
                sync(device)
                start = time.perf_counter()
                predictions.append(runner.predict(image))
                sync(device)
                times.append((time.perf_counter() - start) * 1000)
            rows.append(
                {
                    "model": runner.name,
                    "precision": label,
                    "device": device.upper(),
                    "file_size_mb": size_mb(model),
                    "top1_accuracy_pct": 100 * float((np.array(predictions) == truth).mean()),
                    "latency_mean_ms": float(np.mean(times)),
                    "latency_std_ms": float(np.std(times, ddof=1)),
                    "agreement_with_fp32_gpu_pct": np.nan,
                    "images": len(images),
                    "note": note,
                    "predictions": json.dumps(predictions),
                }
            )
            print({k: v for k, v in rows[-1].items() if k != "predictions"})
            pd.DataFrame(rows).to_csv(OUT, index=False)
            del model, runner.model
            torch.cuda.empty_cache()

    frame = pd.DataFrame(rows)
    for name, group in frame.groupby("model"):
        reference = np.array(json.loads(group[(group["precision"] == "fp32") & (group["device"] == "CUDA")]["predictions"].iloc[0]))
        for index, row in group.iterrows():
            frame.loc[index, "agreement_with_fp32_gpu_pct"] = 100 * float((np.array(json.loads(row["predictions"])) == reference).mean())
    frame.to_csv(OUT, index=False)
    environment = {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cpu": platform.processor(),
        "cpu_threads": torch.get_num_threads(),
        "torch": torch.__version__,
        "images": len(images),
        "warmup_images": WARMUP,
        "megabyte": "10^6 bytes, size of the saved state_dict",
    }
    (RESULTS / "benchmark_environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    print(frame.drop(columns=["predictions", "note"]).round(2).to_string())


if __name__ == "__main__":
    main()
