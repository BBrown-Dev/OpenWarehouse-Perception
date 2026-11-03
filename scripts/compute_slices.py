"""Performance on the held-out slice nearest the deployment setting, against everything else.

There is no warehouse-camera test set. The nearest proxy is held-out images that humans verified
as showing a warehouse-like scene: positive for Warehouse, Shelving, Aisle, or Inventory. The
slice is defined from labels alone, before looking at any result on it.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Sibling scripts are imported directly, so make them importable from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))


import numpy as np
import pandas as pd

from openwarehouse.config import load_config
from openwarehouse.evaluation.metrics import auroc, bootstrap_ci
from openwarehouse.experiments import require_selection

import compute_metrics as cm

RESULTS = Path("docs/results")
SCENE_CLASSES = ["Warehouse", "Shelving", "Aisle", "Inventory"]


def main() -> None:
    config = load_config("open_images", "poc")
    selection = require_selection()
    frame = cm.decisions(config, selection)
    held = frame[frame["role"] == "heldout"]
    scene_images = set(
        held.loc[held["class_name"].isin(SCENE_CLASSES) & held["is_positive"], "image_id"]
    )
    held = held.assign(slice=np.where(held["image_id"].isin(scene_images), "warehouse-like", "other"))
    rows = []
    for (slice_name, class_name), group in held.groupby(["slice", "class_name"]):
        if class_name in SCENE_CLASSES:
            continue  # these define the slice, so measuring them on it is circular
        y = group["is_positive"].to_numpy(bool)
        row = {
            "slice": slice_name,
            "class_name": class_name,
            "n_pos": int(y.sum()),
            "n_neg": int((~y).sum()),
            "detector_recall": float(group.loc[y, "detector_yes"].mean()) if y.any() else np.nan,
            "detector_fpr": float(group.loc[~y, "detector_yes"].mean()) if (~y).any() else np.nan,
        }
        if y.sum() >= 5 and (~y).sum() >= 5:
            scores = group["siglip_probability"].to_numpy()
            row["siglip_auroc"] = auroc(y, scores)
            row["siglip_auroc_low"], row["siglip_auroc_high"] = bootstrap_ci(
                lambda i: auroc(y[i], scores[i]), len(y), 2000
            )
            det = group["detector_score"].to_numpy()
            row["detector_auroc"] = auroc(y, det)
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "metrics_warehouse_slice.csv", index=False)
    print("warehouse-like held-out images:", len(scene_images))
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
