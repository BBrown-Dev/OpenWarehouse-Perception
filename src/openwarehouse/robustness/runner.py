"""Evaluates a model across every corruption and severity."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from openwarehouse.robustness.corruptions import apply_corruption


def run_sweep(
    predict_fn: Any,
    dataset: Any,
    metric_fn: Any,
    config: dict[str, Any],
    cache_dir: Path | None = None,
    cached_only: bool = False,
) -> Any:
    """Return one results row per corruption and severity, plus the clean baseline.

    `dataset` is a list of (key, image) pairs, `predict_fn` maps a list of images to a list of
    predictions, and `metric_fn` maps the predictions, in dataset order, to a dict of metrics.
    With `cache_dir`, each condition's predictions (which must be JSON-serializable) are saved
    and reused, so metrics can be recomputed without running the model again.
    """
    settings = config["robustness"]
    keys = [key for key, _ in dataset]
    images = [image for _, image in dataset]
    conditions = [("clean", 0)] + [
        (name, severity) for name in settings["corruptions"] for severity in settings["severities"]
    ]
    if cached_only:
        if cache_dir is None:
            raise ValueError("cached_only needs a cache_dir.")
        conditions = [c for c in conditions if (cache_dir / f"{c[0]}_{c[1]}.json").exists()]
    rows = []
    for name, severity in tqdm(conditions, desc="conditions"):
        path = cache_dir / f"{name}_{severity}.json" if cache_dir else None
        if path is not None and path.exists():
            cached = json.loads(path.read_text(encoding="utf-8"))
            if cached["keys"] != keys:
                raise RuntimeError(f"{path} was made for different images.")
            predictions = cached["predictions"]
        else:
            shifted = (
                images
                if name == "clean"
                else [apply_corruption(image, name, severity, key) for key, image in zip(keys, images)]
            )
            predictions = predict_fn(shifted)
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"keys": keys, "predictions": predictions}), encoding="utf-8")
        rows.append({"corruption": name, "severity": severity} | metric_fn(predictions))
    return pd.DataFrame(rows)


def summarize_degradation(results: Any) -> Any:
    """Compute the metric drop from clean and find the worst-case conditions."""
    clean = results[results["corruption"] == "clean"].iloc[0]
    metrics = [c for c in results.columns if c not in ("corruption", "severity")]
    drops = results.copy()
    for metric in metrics:
        drops[f"{metric}_change"] = results[metric] - clean[metric]
    return drops
