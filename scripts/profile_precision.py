"""Explain the fp16 vs fp32 latency result by separating where the time goes.

For each model, precision mode (fp32, fp16 autocast, fp16 weights) and batch size, this measures:
- preprocessing time on the CPU (the processor), which precision cannot change;
- forward-pass wall time, and GPU time from CUDA events;
- kernel launches and cast operations per forward pass, from the PyTorch profiler.

Modes are interleaved round by round so GPU clock or thermal drift cannot favor one of them.
Dev images only.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.profiler import ProfilerActivity, profile
from transformers import AutoModel, AutoModelForZeroShotObjectDetection, AutoProcessor

from openwarehouse.config import load_config, model_revision
from openwarehouse.experiments import image_path, load_image, load_manifest

RESULTS = Path("docs/results")
ROUNDS, WARMUP = 30, 10
MODES = ("fp32", "fp16 autocast", "fp16 weights")


def dev_images(config, count):
    manifest = load_manifest(config)
    rows = manifest[manifest["role"] == "dev"].drop_duplicates("image_id").sample(count, random_state=config["seed"])
    return [load_image(image_path(config, s, i)) for s, i in zip(rows["split"], rows["image_id"])]


def build(config):
    siglip_id, detector_id = config["models"]["siglip"], config["models"]["detector"]
    siglip_processor = AutoProcessor.from_pretrained(siglip_id, revision=model_revision(siglip_id))
    siglip = AutoModel.from_pretrained(siglip_id, revision=model_revision(siglip_id)).eval()
    detector_processor = AutoProcessor.from_pretrained(detector_id, revision=model_revision(detector_id))
    detector_processor.image_processor.size = {
        "shortest_edge": config["inference"]["detector_shortest_edge"],
        "longest_edge": config["inference"]["detector_longest_edge"],
    }
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(detector_id, revision=model_revision(detector_id)).eval()

    def siglip_inputs(images):
        return {"pixel_values": siglip_processor(images=images, return_tensors="pt")["pixel_values"]}

    def detector_inputs(images):
        return dict(detector_processor(images=images, text=["person."] * len(images), padding=True, return_tensors="pt"))

    return [
        ("SigLIP base/16-224", siglip, siglip_inputs, lambda m, x: m.get_image_features(**x), (1, 32)),
        ("Grounding DINO tiny", detector, detector_inputs, lambda m, x: m(**x), (1, 4)),
    ]


def variants(base):
    fp32 = copy.deepcopy(base).cuda()
    half = copy.deepcopy(base).cuda().half()
    return {"fp32": (fp32, False, torch.float32), "fp16 autocast": (fp32, True, torch.float32), "fp16 weights": (half, False, torch.float16)}


def to_device(inputs, dtype):
    out = {}
    for key, value in inputs.items():
        value = value.cuda(non_blocking=True)
        out[key] = value.to(dtype) if value.is_floating_point() else value
    return out


@torch.inference_mode()
def forward(run, model, inputs, autocast):
    with torch.autocast("cuda", enabled=autocast):
        return run(model, inputs)


def kernel_counts(run, model, inputs, autocast):
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        forward(run, model, inputs, autocast)
        torch.cuda.synchronize()
    events = prof.key_averages()
    kernels = sum(e.count for e in events if e.device_type == torch.autograd.DeviceType.CUDA)
    casts = sum(e.count for e in events if e.key in ("aten::_to_copy", "aten::copy_"))
    gpu_us = sum(e.self_device_time_total for e in events if e.device_type == torch.autograd.DeviceType.CUDA)
    return kernels, casts, gpu_us / 1000


def main() -> None:
    config = load_config("open_images", "poc")
    torch.backends.cudnn.benchmark = False
    images = dev_images(config, 64)
    rows = []
    for name, base, make_inputs, run, batches in build(config):
        models = variants(base)
        for batch in batches:
            sample = images[:batch]
            start = time.perf_counter()
            for _ in range(5):
                cpu_inputs = make_inputs(sample)
            preprocess_ms = (time.perf_counter() - start) / 5 * 1000
            prepared = {mode: to_device(cpu_inputs, models[mode][2]) for mode in MODES}
            for mode in MODES:
                model, autocast, _ = models[mode]
                for _ in range(WARMUP):
                    forward(run, model, prepared[mode], autocast)
            torch.cuda.synchronize()
            wall = {mode: [] for mode in MODES}
            gpu = {mode: [] for mode in MODES}
            for _ in range(ROUNDS):
                for mode in MODES:  # interleaved
                    model, autocast, _ = models[mode]
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    begin.record()
                    forward(run, model, prepared[mode], autocast)
                    end.record()
                    torch.cuda.synchronize()
                    wall[mode].append((time.perf_counter() - t0) * 1000)
                    gpu[mode].append(begin.elapsed_time(end))
            for mode in MODES:
                model, autocast, _ = models[mode]
                kernels, casts, kernel_ms = kernel_counts(run, model, prepared[mode], autocast)
                rows.append(
                    {
                        "model": name,
                        "batch": batch,
                        "mode": mode,
                        "preprocess_ms_per_batch": preprocess_ms,
                        "forward_wall_ms": float(np.median(wall[mode])),
                        "forward_gpu_event_ms": float(np.median(gpu[mode])),
                        "kernel_busy_ms": kernel_ms,
                        "cuda_kernels": kernels,
                        "cast_ops": casts,
                        "forward_ms_per_image": float(np.median(wall[mode])) / batch,
                    }
                )
                print(rows[-1])
        del models
        torch.cuda.empty_cache()
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "profile_precision.csv", index=False)
    print(table.round(2).to_string(index=False))
    print("attention implementation:", base.config._attn_implementation if hasattr(base.config, "_attn_implementation") else "n/a")


if __name__ == "__main__":
    main()
