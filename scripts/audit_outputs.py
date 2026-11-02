"""Audit saved outputs without running a model: label hierarchy, cohort shift, and claim checks."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from openwarehouse.config import PROJECT_ROOT, load_config
from openwarehouse.data.leakage import load_authors
from openwarehouse.evaluation.metrics import auroc, detection_ap
from openwarehouse.experiments import (
    OUTPUT_DIR,
    _load_ground_truth,
    load_detector_predictions,
    load_manifest,
    restrict_to_manifest,
)

RESULTS = Path("docs/results")


def descendants(config, class_names):
    """Map each class to itself plus its children in the official box label hierarchy."""
    root = PROJECT_ROOT / config["open_images"]["dataset_dir"]
    hierarchy = json.loads(next(root.rglob("hierarchy.json")).read_text(encoding="utf-8"))
    classes = pd.read_csv(next(root.rglob("classes.csv")), header=None, names=["mid", "name"])
    names = dict(zip(classes["mid"], classes["name"]))
    expanded = {name: {name} for name in class_names}

    def walk(node, ancestors):
        name = names.get(node.get("LabelName"), node.get("LabelName"))
        for ancestor in ancestors:
            if ancestor in expanded:
                expanded[ancestor].add(name)
        for child in node.get("Subcategory", []):
            walk(child, [*ancestors, name])

    walk(hierarchy, [])
    return expanded


def hierarchy_ap(config, predictions, truth_boxes, condition, expanded):
    rows = []
    iou = config["evaluation"]["iou_threshold"]
    for role in ("dev", "heldout"):
        for class_name in config["boxable_classes"]:
            group = predictions[
                (predictions["role"] == role)
                & (predictions["class_name"] == class_name)
                & (predictions["condition"] == condition)
            ]
            literal, full, preds = {}, {}, {}
            literal_only, full_all = {}, {}
            for record in group.itertuples():
                boxes = truth_boxes.get(record.image_id, {}).get("boxes", [])
                exact = [(b["box"], b["is_group_of"]) for b in boxes if b["label"] == class_name]
                wide = [
                    (b["box"], b["is_group_of"]) for b in boxes if b["label"] in expanded[class_name]
                ]
                preds[record.image_id] = [(d[0], d[1]) for d in record.detections]
                if record.is_positive:
                    if exact:
                        literal[record.image_id] = exact
                    if wide:
                        full[record.image_id] = wide
                else:
                    literal[record.image_id] = []
                    full[record.image_id] = []
                literal_only[record.image_id] = exact
                full_all[record.image_id] = wide
            rows.append(
                {
                    "role": role,
                    "class_name": class_name,
                    "condition": condition,
                    "child_labels": ", ".join(sorted(expanded[class_name] - {class_name})),
                    "images_literal": len(literal),
                    "images_expanded": len(full),
                    "boxes_literal": sum(len(v) for v in literal_only.values()),
                    "boxes_expanded": sum(len(v) for v in full_all.values()),
                    "ap50_literal": detection_ap(preds, literal, iou),
                    "ap50_expanded": detection_ap(preds, full, iou),
                }
            )
    return pd.DataFrame(rows)


def negative_conflicts(manifest, truth_boxes, expanded):
    """Count verified negatives that nonetheless have a box of the class or a child class."""
    rows = []
    for class_name, labels in expanded.items():
        negatives = manifest[(manifest["class_name"] == class_name) & (~manifest["is_positive"])]
        hits = 0
        for image_id in negatives["image_id"]:
            boxes = truth_boxes.get(image_id, {}).get("boxes", [])
            hits += any(b["label"] in labels for b in boxes)
        rows.append({"class_name": class_name, "negatives": len(negatives), "with_box": hits})
    return pd.DataFrame(rows)


def cluster_bootstrap(frame, value, samples=2000, seed=42):
    """Percentile CI for AUROC, resampling images or whole photographers."""
    generator = np.random.default_rng(seed)
    y, scores = frame["is_positive"].to_numpy(bool), frame[value].to_numpy()
    plain = [
        auroc(y[i], scores[i])
        for i in (generator.integers(0, len(y), len(y)) for _ in range(samples))
    ]
    groups = frame.groupby("author").indices
    members = list(groups.values())
    clustered = []
    for _ in range(samples):
        picked = generator.integers(0, len(members), len(members))
        index = np.concatenate([members[p] for p in picked])
        clustered.append(auroc(y[index], scores[index]))
    return (*np.nanquantile(plain, [0.025, 0.975]), *np.nanquantile(clustered, [0.025, 0.975]))


def main() -> None:
    config = load_config("open_images", "poc")
    manifest = load_manifest(config)
    raw = pd.read_csv(OUTPUT_DIR / "manifest.csv", dtype={"image_id": str})
    truth_boxes = _load_ground_truth()
    predictions = restrict_to_manifest(load_detector_predictions(), manifest)

    expanded = descendants(config, config["boxable_classes"])
    print({k: sorted(v - {k}) for k, v in expanded.items()})
    table = pd.concat(
        hierarchy_ap(config, predictions, truth_boxes, condition, expanded)
        for condition in ("bare", "article")
    )
    table.to_csv(RESULTS / "audit_ap_hierarchy.csv", index=False)
    print(table.drop(columns="child_labels").round(3).to_string())

    conflicts = negative_conflicts(manifest[manifest["role"] == "heldout"], truth_boxes, expanded)
    print(conflicts.to_string())

    duplicate = load_detector_predictions().duplicated(["image_id", "class_name", "condition"]).sum()
    print("duplicate prediction rows:", int(duplicate))

    # Did removing photographers change who is left?
    excluded = set(pd.read_csv(OUTPUT_DIR / "leakage_exclusions.csv", dtype=str)["image_id"])
    held = raw[raw["role"] == "heldout"]
    shift = (
        held.assign(kept=~held["image_id"].isin(excluded))
        .groupby(["class_name", "kept"])["is_positive"]
        .agg(["size", "mean"])
        .unstack("kept")
        .round(3)
    )
    shift.to_csv(RESULTS / "audit_exclusion_shift.csv")
    print(shift.to_string())

    # Authors: missing values cannot match, and repeated authors make images non-independent.
    images = raw[["split", "image_id", "role"]].drop_duplicates().reset_index(drop=True)
    authors = load_authors(images, config)
    print("missing author by role:", authors[authors["author"].isna()].groupby("role").size().to_dict())
    kept = authors[(authors["role"] == "heldout") & ~authors["image_id"].isin(excluded)]
    counts = Counter(kept["author"].dropna())
    print("held-out images", len(kept), "authors", len(counts), "top", counts.most_common(3)[0][1])

    # Do images cluster enough by photographer to widen the reported intervals?
    scores = pd.read_csv(OUTPUT_DIR / "siglip_scores.csv", dtype={"image_id": str})
    scores = restrict_to_manifest(scores, manifest)
    scores = scores[(scores["role"] == "heldout") & (scores["condition"] == "photo")]
    scores = scores.merge(kept[["image_id", "author"]], on="image_id")
    scores["author"] = scores["author"].fillna(scores["image_id"])
    rows = []
    for class_name, group in scores.groupby("class_name"):
        plain_low, plain_high, low, high = cluster_bootstrap(group.reset_index(drop=True), "logit")
        rows.append(
            {
                "class_name": class_name,
                "authors": group["author"].nunique(),
                "images": len(group),
                "auroc": auroc(group["is_positive"], group["logit"]),
                "image_low": plain_low,
                "image_high": plain_high,
                "author_low": low,
                "author_high": high,
            }
        )
    clusters = pd.DataFrame(rows)
    clusters.to_csv(RESULTS / "audit_cluster_bootstrap.csv", index=False)
    print(clusters.round(3).to_string())

    # Would the exclusions have changed the answer? SigLIP scored every image before they applied.
    all_scores = pd.read_csv(OUTPUT_DIR / "siglip_scores.csv", dtype={"image_id": str})
    all_scores = all_scores[(all_scores["role"] == "heldout") & (all_scores["condition"] == "photo")]
    all_scores = all_scores.drop(columns="is_positive").merge(
        raw[["image_id", "class_name", "is_positive"]], on=["image_id", "class_name"]
    )
    sensitivity = []
    for class_name, group in all_scores.groupby("class_name"):
        removed = group["image_id"].isin(excluded)
        sensitivity.append(
            {
                "class_name": class_name,
                "auroc_kept": auroc(group[~removed]["is_positive"], group[~removed]["logit"]),
                "auroc_excluded_only": auroc(group[removed]["is_positive"], group[removed]["logit"]),
                "auroc_all": auroc(group["is_positive"], group["logit"]),
            }
        )
    sensitivity = pd.DataFrame(sensitivity)
    sensitivity.to_csv(RESULTS / "audit_exclusion_sensitivity.csv", index=False)
    print(sensitivity.round(3).to_string())

    # Mitigations from the report, with every threshold chosen on dev. Exploratory.
    siglip = restrict_to_manifest(
        pd.read_csv(OUTPUT_DIR / "siglip_scores.csv", dtype={"image_id": str}), manifest
    )
    siglip = siglip[siglip["condition"] == "photo"][["image_id", "class_name", "logit"]]
    bare = predictions[predictions["condition"] == "bare"].copy()
    bare["max_score"] = [max((d[0] for d in ds), default=0.0) for ds in bare["detections"]]
    bare = bare.merge(siglip, on=["image_id", "class_name"])

    def best_f1(y, values):
        candidates = np.unique(np.quantile(values, np.linspace(0, 1, 101)))
        f1 = [2 * ((values >= t) & y).sum() / max((values >= t).sum() + y.sum(), 1) for t in candidates]
        return float(candidates[int(np.argmax(f1))])

    def rates(y, fired):
        return {
            "precision": (fired & y).sum() / max(fired.sum(), 1),
            "recall": (fired & y).sum() / max(y.sum(), 1),
            "fpr": (fired & ~y).sum() / max((~y).sum(), 1),
        }

    mitigations = []
    for class_name, group in bare.groupby("class_name"):
        dev = group[group["role"] == "dev"]
        test = group[group["role"] == "heldout"]
        y_dev, y_test = dev["is_positive"].to_numpy(bool), test["is_positive"].to_numpy(bool)
        detector_cut = best_f1(y_dev, dev["max_score"].to_numpy())
        gate_cut = best_f1(y_dev, dev["logit"].to_numpy())
        detector = test["max_score"].to_numpy() >= detector_cut
        gate = test["logit"].to_numpy() >= gate_cut
        pooled = test["max_score"].to_numpy() >= 0.32
        for name, fired in (
            ("frozen 0.32", pooled),
            ("per-class detector", detector),
            ("SigLIP gate + 0.32", gate & pooled),
        ):
            mitigations.append(
                {"class_name": class_name, "n_pos": int(y_test.sum()), "n_neg": int((~y_test).sum()),
                 "method": name, **rates(y_test, fired)}
            )
    mitigations = pd.DataFrame(mitigations)
    mitigations.to_csv(RESULTS / "audit_mitigations_exploratory.csv", index=False)
    print(mitigations.pivot(index="class_name", columns="method", values=["recall", "fpr"]).round(2).to_string())
    print(mitigations.groupby("method")[["precision", "recall", "fpr"]].mean().round(3).to_string())

    # Which token really wins, rather than the broad "structural" group?
    for role in ("dev", "heldout"):
        tokens = pd.read_csv(RESULTS / f"token_attribution_{role}.csv")
        for prompt in ("a giraffe", "a box", "a photo of a box", "a photo of a forklift"):
            share = tokens[tokens["prompt"] == prompt]["winning_token"].value_counts(normalize=True)
            print(role, prompt, share.round(3).to_dict())

    # Prompts that are the same string make some "conditions" the same experiment.
    prompts = config["prompt_conditions"]
    same = []
    for class_name, fields in config["prompt_classes"].items():
        expanded_prompts = {c: t.format(**fields) for c, t in prompts.items()}
        if expanded_prompts["bare"] == expanded_prompts["article"]:
            same.append(class_name)
    print("bare identical to article:", same)
    print("manifest images missing on disk:", raw["image_id"].nunique() - manifest["image_id"].nunique() - len(excluded))
    print(raw[raw["class_name"] == "Pallet"].groupby(["role", "is_positive"]).size().to_dict())


if __name__ == "__main__":
    main()
