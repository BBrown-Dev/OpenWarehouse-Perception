"""Compute the remaining report numbers, copy results to docs/, and draw the figures."""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from openwarehouse.config import load_config
from openwarehouse.data.open_images import load_box_labels
from openwarehouse.evaluation.metrics import bootstrap_ci, detection_ap
from openwarehouse.experiments import (
    OUTPUT_DIR,
    _load_ground_truth,
    image_path,
    load_detector_predictions,
    load_image,
    load_manifest,
    require_selection,
    restrict_to_manifest,
)
from openwarehouse.failure_analysis import FAILURE_DIR, images_for, siglip_occlusion_map
from openwarehouse.models.vision_language import VisionLanguageEncoder
from openwarehouse.segmentation.grounded_sam import GroundedSAM
from openwarehouse.utils import resolve_device, set_seed

RESULTS = Path("docs/results")
FIGURES = Path("docs/figures")
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#898781", "#e1e0d9"
CONDITIONS = ["bare", "article", "synonym", "context", "descriptive", "photo"]
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": "#52514e",
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
    }
)


def per_image(rows, class_name, condition, ground_truth, box_labels):
    truth, preds = {}, {}
    group = rows[(rows["class_name"] == class_name) & (rows["condition"] == condition)]
    for record in group.itertuples():
        boxes = [
            (b["box"], b["is_group_of"])
            for b in ground_truth.get(record.image_id, {}).get("boxes", [])
            if b["label"] in box_labels[class_name]
        ]
        if record.is_positive and not boxes:
            continue
        key = f"{record.split}/{record.image_id}"
        truth[key] = boxes if record.is_positive else []
        preds[key] = [(d[0], d[1]) for d in record.detections]
    return truth, preds


def paired_ap_delta(rows, config, ground_truth):
    """Bootstrap AP50(bare) - AP50(article) over the same resampled images."""
    results = []
    iou = config["evaluation"]["iou_threshold"]
    box_labels = load_box_labels(config)
    for class_name in config["boxable_classes"]:
        truth, bare = per_image(rows, class_name, "bare", ground_truth, box_labels)
        _, article = per_image(rows, class_name, "article", ground_truth, box_labels)
        keys = list(truth)

        def ap(preds, index):
            names = [(f"{n}:{keys[i]}", keys[i]) for n, i in enumerate(index)]
            return detection_ap(
                {new: preds[old] for new, old in names},
                {new: truth[old] for new, old in names},
                iou,
            )

        low, high = bootstrap_ci(lambda i: ap(bare, i) - ap(article, i), len(keys), 300)
        results.append(
            {
                "class_name": class_name,
                "images": len(keys),
                "ap50_bare": detection_ap(bare, truth, iou),
                "ap50_article": detection_ap(article, truth, iou),
                "delta_low": low,
                "delta_high": high,
            }
        )
    return pd.DataFrame(results)


def operating_point(rows, condition, threshold):
    """Image-level precision, recall, and false positive rate at the frozen threshold."""
    rows = rows[rows["condition"] == condition].copy()
    rows["fires"] = [max((d[0] for d in ds), default=0.0) >= threshold for ds in rows["detections"]]
    results = []
    for class_name, group in rows.groupby("class_name"):
        positive = group["is_positive"].astype(bool)
        hits = int((group["fires"] & positive).sum())
        results.append(
            {
                "class_name": class_name,
                "n_pos": int(positive.sum()),
                "n_neg": int((~positive).sum()),
                "precision": hits / max(int(group["fires"].sum()), 1),
                "recall": hits / max(int(positive.sum()), 1),
                "false_positive_rate": float(group.loc[~positive, "fires"].mean()),
            }
        )
    return pd.DataFrame(results)


def batch_check(model, rows, config, condition, prompt, limit=100):
    """Compare cached batch-of-4 top scores with single-image scores."""
    group = rows[(rows["class_name"] == "Person") & (rows["condition"] == condition)]
    group = group.sample(min(limit, len(group)), random_state=config["seed"])
    cached, single = [], []
    for record in group.itertuples():
        found = model.detect(load_image(image_path(config, record.split, record.image_id)), prompt)
        cached.append(max((d[0] for d in record.detections), default=0.0))
        single.append(found[0].score if found else 0.0)
    difference = np.abs(np.array(cached) - np.array(single))
    return {
        "images": len(cached),
        "median_abs_diff": float(np.median(difference)),
        "p95_abs_diff": float(np.quantile(difference, 0.95)),
        "max_abs_diff": float(difference.max()),
        "pearson": float(np.corrcoef(cached, single)[0, 1]),
    }


def occlusion(encoder, config):
    images = images_for(config, "dev", "Forklift truck", True, 20, config["seed"])
    correlations, first = [], None
    for split, image_id in images.itertuples(index=False):
        image = load_image(image_path(config, split, image_id))
        plain = siglip_occlusion_map(encoder, image, "a forklift")
        negated = siglip_occlusion_map(encoder, image, "no forklift")
        correlations.append(float(np.corrcoef(plain.ravel(), negated.ravel())[0, 1]))
        first = first or (image, plain, negated, image_id)
    image, plain, negated, image_id = first
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    limit = max(np.abs(plain).max(), np.abs(negated).max())
    axes[0].imshow(image)
    axes[0].set_title(f"Dev image {image_id[:8]}")
    for ax, values, name in zip(axes[1:], (plain, negated), ("a forklift", "no forklift")):
        ax.imshow(image)
        shown = ax.imshow(
            values,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            alpha=0.6,
            extent=(0, image.width, image.height, 0),
        )
        ax.set_title(f'Logit drop when hidden: "{name}"')
    for ax in axes:
        ax.axis("off")
    fig.colorbar(shown, ax=axes[2], fraction=0.04, label="logit drop")
    fig.tight_layout()
    fig.savefig(FIGURES / "siglip_occlusion.png", dpi=130)
    plt.close(fig)
    return {"images": len(correlations), "median_map_correlation": float(np.median(correlations))}


def figure_prompts():
    frame = pd.read_csv(OUTPUT_DIR / "detector_heldout.csv").dropna(subset=["ap50"])
    classes = ["Ladder", "Person", "Truck", "Box", "Shelf"]
    fig, axes = plt.subplots(1, 5, figsize=(12, 3.2), sharex=True, sharey=True)
    for ax, class_name in zip(axes, classes):
        group = frame[frame["class_name"] == class_name].set_index("condition").loc[CONDITIONS]
        y = np.arange(len(CONDITIONS))[::-1]
        ax.errorbar(
            group["ap50"],
            y,
            xerr=[group["ap50"] - group["ap50_low"], group["ap50_high"] - group["ap50"]],
            fmt="o",
            color=BLUE,
            ecolor="#86b6ef",
            ms=6,
            lw=2,
        )
        ax.set_yticks(y, CONDITIONS)
        ax.set_title(class_name)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlim(0, 0.8)
    axes[2].set_xlabel("Held-out AP50 with 95% bootstrap CI")
    fig.suptitle("Same object, different wording", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / "prompt_sensitivity_ap50.png", dpi=150)
    plt.close(fig)


def figure_absent():
    dev = pd.read_csv(FAILURE_DIR / "token_summary_dev.csv")
    held = pd.read_csv(FAILURE_DIR / "token_summary_heldout.csv")
    prompts = ["giraffe", "a giraffe", "forklift", "a forklift", "a photo of a forklift", "no forklift"]
    values = [
        frame[frame["class_name"] == "Forklift truck"].set_index("prompt").loc[prompts, "fire_rate"]
        for frame in (dev, held)
    ]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = np.arange(len(prompts))[::-1]
    names = ("Dev (60 images)", "Held-out (14 images)")
    for offset, series, color, name in zip((0.19, -0.19), values, (BLUE, ORANGE), names):
        ax.barh(y + offset, series, height=0.34, color=color, label=name)
        for position, value in zip(y + offset, series):
            ax.text(value + 0.012, position, f"{value:.0%}", va="center", fontsize=8, color="#52514e")
    ax.set_yticks(y, [f'"{p}"' for p in prompts])
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Share of verified no-forklift images with a detection at the frozen threshold (0.32)")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    ax.set_title("Adding one word makes an absent object appear", loc="left")
    fig.tight_layout()
    fig.savefig(FIGURES / "absent_object_fire_rate.png", dpi=150)
    plt.close(fig)


def figure_negation():
    frame = pd.read_csv(FAILURE_DIR / "siglip_negation_heldout.csv")
    pairs = [
        ("Forklift truck", "a forklift", "no forklift"),
        ("Person", "a person", "no person"),
        ("Box", "a box", "no box"),
    ]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    y = np.arange(len(pairs))[::-1]
    series = ((0.19, 1, BLUE, 'Plain prompt ("a X")'), (-0.19, 2, ORANGE, 'Negated prompt ("no X")'))
    for offset, column, color, name in series:
        values = [
            frame[(frame["class_name"] == p[0]) & (frame["prompt"] == p[column])]["auroc"].iloc[0]
            for p in pairs
        ]
        ax.barh(y + offset, values, height=0.34, color=color, label=name)
        for position, value in zip(y + offset, values):
            ax.text(value + 0.008, position, f"{value:.2f}", va="center", fontsize=9, color="#52514e")
    ax.axvline(0.5, color=INK, lw=1)
    ax.text(0.505, 2.45, "chance", fontsize=8, color=INK)
    ax.set_yticks(y, [p[0] for p in pairs])
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Held-out AUROC for the presence of the object")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax.set_title("A working negation would fall below 0.5; none does", loc="left")
    fig.tight_layout()
    fig.savefig(FIGURES / "siglip_negation_auroc.png", dpi=150)
    plt.close(fig)


def main() -> None:
    config = load_config("open_images", "poc")
    set_seed(config["seed"])
    device = resolve_device(config["device"])
    selection = require_selection()
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    predictions = restrict_to_manifest(load_detector_predictions(), load_manifest(config))
    ground_truth = _load_ground_truth()
    held = predictions[predictions["role"] == "heldout"]
    dev = predictions[predictions["role"] == "dev"]

    delta = paired_ap_delta(held, config, ground_truth)
    delta.to_csv(RESULTS / "ap50_bare_vs_article_heldout.csv", index=False)
    print(delta.round(3).to_string())
    points = operating_point(held, selection["detector_condition"], selection["detector_threshold"])
    points.to_csv(RESULTS / "operating_point_heldout.csv", index=False)
    print(points.round(3).to_string())

    model = GroundedSAM(config, device=device)
    check = batch_check(model, dev, config, "bare", "person")
    print(check)
    del model
    encoder = VisionLanguageEncoder(config["models"]["siglip"], device=device)
    maps = occlusion(encoder, config)
    print(maps)
    pd.DataFrame([check | {"check": "batch_vs_single"}, maps | {"check": "occlusion"}]).to_csv(
        RESULTS / "checks.csv", index=False
    )

    # scripts/run_validation.py writes corrected versions of these straight to docs/results;
    # the copies in FAILURE_DIR predate the box-label fix and must not overwrite them.
    corrected = {"rescoring_dev.csv", "rescoring_heldout.csv", "masks_heldout.csv"}
    for path in list(FAILURE_DIR.glob("*.csv")) + [FAILURE_DIR / "reproductions.json"]:
        if path.name not in corrected:
            shutil.copy(path, RESULTS / path.name)
    for name in ("siglip_dev", "siglip_heldout", "detector_dev", "detector_heldout", "leakage_exclusions"):
        shutil.copy(OUTPUT_DIR / f"{name}.csv", RESULTS / f"{name}.csv")
    shutil.copy(OUTPUT_DIR / "selection.json", RESULTS / "selection.json")
    figure_prompts()
    figure_absent()
    figure_negation()


if __name__ == "__main__":
    main()
