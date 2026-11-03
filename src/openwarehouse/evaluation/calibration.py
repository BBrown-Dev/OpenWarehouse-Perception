"""Calibration of presence scores: expected calibration error and reliability curves.

Calibration depends on the share of positives. These evaluation sets have an artificial share
(capped positives and negatives per class), so ECE here compares models and conditions on the
same data; it does not predict calibration on a real camera feed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def reliability_table(y_true, confidence, bins: int = 15) -> pd.DataFrame:
    """Return per-bin counts, mean confidence, and observed positive rate (equal-width bins)."""
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(confidence, dtype=float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    # The last bin is closed on the right so a confidence of exactly 1.0 is counted.
    index = np.minimum(np.digitize(p, edges[1:-1], right=False), bins - 1)
    rows = []
    for b in range(bins):
        mask = index == b
        rows.append(
            {
                "bin_low": edges[b],
                "bin_high": edges[b + 1],
                "count": int(mask.sum()),
                "mean_confidence": float(p[mask].mean()) if mask.any() else np.nan,
                "positive_rate": float(y[mask].mean()) if mask.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def calibration_errors(y_true, confidence, bins: int = 15) -> dict[str, float]:
    """Return ECE, maximum calibration error, and Brier score."""
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(confidence, dtype=float), 0.0, 1.0)
    table = reliability_table(y, p, bins)
    filled = table[table["count"] > 0]
    gaps = (filled["positive_rate"] - filled["mean_confidence"]).abs()
    return {
        "n": len(y),
        "ece": float((filled["count"] / len(y) * gaps).sum()) if len(y) else float("nan"),
        "mce": float(gaps.max()) if len(filled) else float("nan"),
        "brier": float(np.mean((p - y) ** 2)) if len(y) else float("nan"),
        "mean_confidence": float(p.mean()) if len(y) else float("nan"),
        "positive_rate": float(y.mean()) if len(y) else float("nan"),
    }
