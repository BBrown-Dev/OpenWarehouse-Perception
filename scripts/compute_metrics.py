"""Compute IoU, CHAIR, POPE and ECE from cached outputs, at the frozen dev choices.

No model is run. CHAIR and POPE score the detector at its frozen `bare` condition and threshold.
SigLIP is not given a yes/no decision: its sigmoid output is badly calibrated, and comparing it with a
fixed probability is ruled out by the safety requirements, so it appears only in the per-class ECE.
Intervals are 95% bootstrap intervals over images.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from openwarehouse.config import load_config
from openwarehouse.evaluation.calibration import calibration_errors
from openwarehouse.evaluation.hallucination import chair, pope_questions, pope_scores
from openwarehouse.evaluation.metrics import localization_iou
from openwarehouse.experiments import (
    OUTPUT_DIR,
    _load_ground_truth,
    load_detector_predictions,
    load_manifest,
    require_selection,
    restrict_to_manifest,
)
from openwarehouse.data.open_images import load_box_labels

RESULTS = Path("docs/results")
FIGURES = Path("docs/figures")
SAMPLES = 2000


def decisions(config, selection):
    classes = list(config["prompt_classes"])
    manifest = load_manifest(config)
    manifest = manifest[manifest["class_name"].isin(classes)]
    detector = restrict_to_manifest(load_detector_predictions(), manifest)
    detector = detector[detector["condition"] == selection["detector_condition"]].copy()
    detector["detector_score"] = [max((d[0] for d in ds), default=0.0) for ds in detector["detections"]]
    siglip = pd.read_csv(OUTPUT_DIR / "siglip_scores.csv", dtype={"image_id": str})
    siglip = restrict_to_manifest(siglip, manifest)
    siglip = siglip[siglip["condition"] == selection["siglip_condition"]].copy()
    siglip["siglip_probability"] = 1 / (1 + np.exp(-siglip["logit"]))
    keys = ["split", "image_id", "class_name"]
    frame = detector[[*keys, "role", "is_positive", "detector_score", "detections"]].merge(
        siglip[[*keys, "siglip_probability"]], on=keys
    )
    frame["image_id"] = frame["split"] + "/" + frame["image_id"]
    frame["detector_yes"] = frame["detector_score"] >= selection["detector_threshold"]
    return frame


def image_bootstrap(per_image: pd.DataFrame, statistic, seed: int = 7):
    """Resample images; `statistic` maps summed per-image counts to a value."""
    values = per_image.to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = [statistic(values[rng.integers(0, len(values), len(values))].sum(0)) for _ in range(SAMPLES)]
    low, high = np.nanquantile(draws, [0.025, 0.975])
    return float(low), float(high)


def chair_table(frame):
    rows = []
    for role, group in frame.groupby("role"):
        for model in ("detector",):
            answers = group.assign(says_yes=group[f"{model}_yes"])
            row = {"role": role, "model": model} | chair(answers)
            counts = answers.assign(
                m=answers["says_yes"], h=answers["says_yes"] & ~answers["is_positive"]
            ).groupby("image_id")[["m", "h"]].sum()
            counts["any_h"] = counts["h"] > 0
            row["chair_i_low"], row["chair_i_high"] = image_bootstrap(
                counts[["m", "h"]], lambda s: s[1] / s[0] if s[0] else np.nan
            )
            row["chair_s_low"], row["chair_s_high"] = image_bootstrap(
                counts[["any_h"]].assign(n=1), lambda s: s[0] / s[1]
            )
            rows.append(row)
    return pd.DataFrame(rows)


def pope_table(frame, seed):
    rows = []
    for role, group in frame.groupby("role"):
        labels = group[["image_id", "class_name", "is_positive"]]
        for strategy in ("random", "popular", "adversarial"):
            questions = pope_questions(labels, strategy, seed)
            joined = questions.merge(group.drop(columns="is_positive"), on=["image_id", "class_name"])
            assert len(joined) == len(questions)
            for model in ("detector",):
                answers = joined.assign(says_yes=joined[f"{model}_yes"])
                row = {"role": role, "strategy": strategy, "model": model} | pope_scores(answers)
                y, yes = answers["is_positive"], answers["says_yes"]
                counts = answers.assign(
                    tp=yes & y, fp=yes & ~y, fn=~yes & y, tn=~yes & ~y
                ).groupby("image_id")[["tp", "fp", "fn", "tn"]].sum()
                row["f1_low"], row["f1_high"] = image_bootstrap(
                    counts, lambda s: 2 * s[0] / (2 * s[0] + s[1] + s[2]) if s[0] else 0.0
                )
                row["accuracy_low"], row["accuracy_high"] = image_bootstrap(
                    counts, lambda s: (s[0] + s[3]) / s.sum()
                )
                rows.append(row)
    return pd.DataFrame(rows)


def ece_table(frame):
    """Per-class calibration only. A pooled ECE mixes classes with different positive shares and
    can look well calibrated when no class is; it is deliberately not reported."""
    rows = []
    for role, group in frame.groupby("role"):
        for model, column in (("detector", "detector_score"), ("siglip", "siglip_probability")):
            for class_name, part in group.groupby("class_name"):
                rows.append(
                    {"role": role, "model": model, "class_name": class_name}
                    | calibration_errors(part["is_positive"], part[column])
                )
    return pd.DataFrame(rows)


def reliability_figure(ece):
    """Per-class held-out ECE for both models, sorted by the detector's error."""
    held = ece[ece["role"] == "heldout"].pivot(index="class_name", columns="model", values="ece")
    held = held.sort_values("detector")
    fig, axis = plt.subplots(figsize=(7, 5))
    rows = np.arange(len(held))
    axis.barh(rows - 0.2, held["detector"], height=0.4, color="#2f6db5", label="Grounding DINO max score")
    axis.barh(rows + 0.2, held["siglip"], height=0.4, color="#d08c2c", label="SigLIP probability")
    axis.set_yticks(rows, held.index)
    axis.set_xlabel("Expected calibration error (lower is better)")
    axis.set_xlim(0, 1)
    axis.legend(frameon=False, loc="lower right")
    axis.set_title(
        f"Held-out ECE per class (mean: detector {held['detector'].mean():.2f}, SigLIP {held['siglip'].mean():.2f})",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "calibration_per_class_heldout.png", dpi=150)
    plt.close(fig)


def iou_table(frame, config, threshold):
    ground_truth = _load_ground_truth()
    box_labels = load_box_labels(config)
    rows = []
    for (role, class_name), group in frame[frame["class_name"].isin(config["boxable_classes"])].groupby(
        ["role", "class_name"]
    ):
        best = []
        for record in group[group["is_positive"]].itertuples():
            image_id = record.image_id.split("/", 1)[1]
            targets = [
                b["box"]
                for b in ground_truth.get(image_id, {}).get("boxes", [])
                if b["label"] in box_labels[class_name] and not b["is_group_of"]
            ]
            kept = [d[1] for d in record.detections if d[0] >= threshold]
            best.extend(localization_iou(kept, targets).tolist())
        best = np.array(best)
        matched = best[best >= 0.5]
        rows.append(
            {
                "role": role,
                "class_name": class_name,
                "instances": len(best),
                "mean_matched_iou": float(best.mean()) if len(best) else np.nan,
                "recall_iou50": float((best >= 0.5).mean()) if len(best) else np.nan,
                "recall_iou75": float((best >= 0.75).mean()) if len(best) else np.nan,
                "mean_iou_when_matched": float(matched.mean()) if len(matched) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    config = load_config("open_images", "poc")
    selection = require_selection()
    frame = decisions(config, selection)

    iou = iou_table(frame, config, selection["detector_threshold"])
    iou.to_csv(RESULTS / "metrics_iou.csv", index=False)
    masks = pd.read_csv(RESULTS / "masks_heldout.csv")
    masks.groupby("class_name").agg(
        instances=("image_id", "size"),
        mask_iou_gt_box=("oracle_box_mask_iou", "mean"),
        mask_iou_detected_box=("detected_mask_iou", "mean"),
    ).reset_index().to_csv(RESULTS / "metrics_mask_iou.csv", index=False)

    chair_table(frame).to_csv(RESULTS / "metrics_chair.csv", index=False)
    pope_table(frame, config["seed"]).to_csv(RESULTS / "metrics_pope.csv", index=False)
    ece = ece_table(frame)
    ece.to_csv(RESULTS / "metrics_ece.csv", index=False)
    reliability_figure(ece)
    (RESULTS / "metrics_settings.json").write_text(
        json.dumps(
            {
                "detector_condition": selection["detector_condition"],
                "detector_threshold": selection["detector_threshold"],
                "siglip_condition": selection["siglip_condition"],
                "bootstrap_samples": SAMPLES,
                "ece_bins": 15,
                "pope_questions_per_image": 3,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for name in ("iou", "chair", "pope", "ece"):
        table = pd.read_csv(RESULTS / f"metrics_{name}.csv")
        if name == "ece":
            table = table.groupby(["role", "model"])[["ece", "brier"]].mean().reset_index()
            print("\n(mean of per-class values)")
        print(f"\n== {name}\n" + table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
