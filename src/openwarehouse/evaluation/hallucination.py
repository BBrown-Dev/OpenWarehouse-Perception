"""Object hallucination metrics, adapted from captioning and VQA to presence decisions.

CHAIR (Rohrbach et al., 2018) and POPE (Li et al., 2023) were defined for generated captions and
yes/no answers. Here a model "mentions" a class when its presence decision for that class is yes.
Only human-verified labels are scored: a yes on a verified-negative class is a hallucination, and
classes never verified for an image are ignored, so no unverified absence is assumed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Columns expected in `decisions`: image_id, class_name, is_positive (bool), says_yes (bool).


def chair(decisions: pd.DataFrame) -> dict[str, float]:
    """Return CHAIR_i (hallucinated mentions / mentions) and CHAIR_s (images with one / images)."""
    mentioned = decisions[decisions["says_yes"]]
    hallucinated = mentioned[~mentioned["is_positive"]]
    images = decisions["image_id"].nunique()
    return {
        "images": images,
        "mentions": len(mentioned),
        "hallucinated": len(hallucinated),
        "chair_i": len(hallucinated) / len(mentioned) if len(mentioned) else float("nan"),
        "chair_s": hallucinated["image_id"].nunique() / images if images else float("nan"),
    }


def pope_questions(
    labels: pd.DataFrame, strategy: str, seed: int, per_image: int = 3
) -> pd.DataFrame:
    """Build balanced yes/no questions from verified labels.

    Each image contributes up to `per_image` positive questions and the same number of negative
    ones, all from its own verified labels. Negatives are chosen by `strategy`:
    `random`, `popular` (classes most often present in this set), or `adversarial` (classes
    that most often co-occur with the image's present classes).
    """
    if strategy not in {"random", "popular", "adversarial"}:
        raise ValueError(f"Unknown POPE strategy: {strategy}")
    rng = np.random.default_rng(seed)
    positives = labels[labels["is_positive"]]
    popularity = positives["class_name"].value_counts()
    by_image = positives.groupby("image_id")["class_name"].agg(set)
    cooccurrence: dict[tuple[str, str], int] = {}
    for present in by_image:
        for a in present:
            for b in present:
                if a != b:
                    cooccurrence[(a, b)] = cooccurrence.get((a, b), 0) + 1

    rows = []
    for image_id, group in labels.groupby("image_id", sort=True):
        present = list(group.loc[group["is_positive"], "class_name"])
        absent = list(group.loc[~group["is_positive"], "class_name"])
        count = min(len(present), len(absent), per_image)
        if count == 0:
            continue
        chosen_present = list(rng.choice(present, count, replace=False))
        if strategy == "random":
            chosen_absent = list(rng.choice(absent, count, replace=False))
        else:
            if strategy == "popular":
                weight = {c: popularity.get(c, 0) for c in absent}
            else:
                weight = {c: sum(cooccurrence.get((p, c), 0) for p in present) for c in absent}
            # Ties are broken randomly so class order in the config does not decide them.
            jitter = rng.random(len(absent))
            ranked = sorted(zip(absent, jitter), key=lambda item: (-weight[item[0]], item[1]))
            chosen_absent = [c for c, _ in ranked[:count]]
        rows += [(image_id, c, True) for c in chosen_present]
        rows += [(image_id, c, False) for c in chosen_absent]
    return pd.DataFrame(rows, columns=["image_id", "class_name", "is_positive"])


def pope_scores(answers: pd.DataFrame) -> dict[str, Any]:
    """Return POPE accuracy, precision, recall, F1, and yes-ratio for yes/no answers."""
    y = answers["is_positive"].to_numpy(bool)
    yes = answers["says_yes"].to_numpy(bool)
    tp, fp = int((yes & y).sum()), int((yes & ~y).sum())
    fn, tn = int((~yes & y).sum()), int((~yes & ~y).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    return {
        "questions": len(y),
        "accuracy": (tp + tn) / len(y) if len(y) else float("nan"),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else float("nan"),
        "yes_ratio": float(yes.mean()) if len(y) else float("nan"),
    }
