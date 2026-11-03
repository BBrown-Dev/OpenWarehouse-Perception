"""Find failure modes on dev data, then measure the same probes on held-out data."""

from __future__ import annotations

import json
import sys

from openwarehouse.config import load_config
from openwarehouse.experiments import require_selection
from openwarehouse.failure_analysis import (
    BASE_PROMPTS,
    FAILURE_DIR,
    run_absent_probe,
    run_modifier_probe,
    summarize_absent_probe,
    summarize_modifier_probe,
)
from openwarehouse.segmentation.grounded_sam import GroundedSAM
from openwarehouse.utils import resolve_device, set_seed

PROBE_IMAGES = 60


def main(role: str) -> None:
    config = load_config("open_images", "poc")
    set_seed(config["seed"])
    device = resolve_device(config["device"])
    selection = require_selection()
    threshold = selection["detector_threshold"]
    FAILURE_DIR.mkdir(parents=True, exist_ok=True)
    model = GroundedSAM(config, device=device)

    modifier = run_modifier_probe(model, config, role, PROBE_IMAGES, config["seed"])
    modifier.to_csv(FAILURE_DIR / f"modifier_{role}.csv", index=False)
    print(summarize_modifier_probe(modifier, threshold).round(3).to_string())

    absent = run_absent_probe(model, config, role, PROBE_IMAGES, config["seed"])
    absent.to_json(FAILURE_DIR / f"absent_{role}.json", orient="records")
    print(summarize_absent_probe(absent, threshold).round(3).to_string())

    # SigLIP rescoring and mask evaluation used to run here. Corrected versions now run only in
    # scripts/run_validation.py, which writes them to docs/results/; the copies made here were stale.
    print(json.dumps({"role": role, "selection": selection, "bases": BASE_PROMPTS}))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dev")
