import os
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(os.getcwd()) / "notebooks"
SETUP = """import pandas as pd
from IPython.display import Image, display
RESULTS = "../docs/results/"
FIGURES = "../docs/figures/"
pd.set_option("display.width", 200, "display.max_columns", 30)"""

NOTEBOOKS = {
    "01_data_preparation": [
        ("md", "# 01 - Data Preparation\n\nOpen Images V7 is imported by `scripts/import_open_images.py`. This notebook summarizes the evaluated manifest and the leakage exclusions. Classical preprocessing is not implemented."),
        ("code", SETUP),
        ("md", "## Evaluated images per class\n\nCounts are after photographer-based exclusions."),
        ("code", 'siglip = pd.read_csv(RESULTS + "siglip_heldout.csv")\nsiglip[["class_name", "n_pos", "n_neg"]].drop_duplicates().reset_index(drop=True)'),
        ("md", "## Leakage exclusions"),
        ("code", 'exclusions = pd.read_csv(RESULTS + "leakage_exclusions.csv")\nprint(len(exclusions), "held-out images excluded")\nexclusions.groupby("split")[["same_id", "same_bytes", "near_duplicate", "same_author"]].sum()'),
    ],
    "02_clip_baseline": [
        ("md", "# 02 - Zero-Shot SigLIP Baseline\n\nImage-level AUROC by prompt condition. `scripts/run_poc.py` produces the scores; this notebook loads them. Conditions were selected on dev before held-out summaries could be computed."),
        ("code", SETUP),
        ("code", 'dev = pd.read_csv(RESULTS + "siglip_dev.csv")\ndev.pivot(index="class_name", columns="condition", values="auroc").round(3)'),
        ("code", 'held = pd.read_csv(RESULTS + "siglip_heldout.csv")\nheld.pivot(index="class_name", columns="condition", values="auroc").round(3)'),
        ("code", 'held.groupby("condition")["auroc"].mean().round(3).sort_values(ascending=False)'),
        ("md", "`delta_low` and `delta_high` are the paired CI against the `article` baseline. The frozen `photo` condition does not beat it."),
        ("code", 'held[held["condition"] == "photo"].round(3).reset_index(drop=True)'),
    ],
    "04_grounded_sam": [
        ("md", "# 04 - Grounded SAM\n\nRuns the pipeline live on one dev image, then loads the full evaluation. Rescoring here uses the pretrained SigLIP model."),
        ("code", SETUP),
        ("md", "## Natural-language prompt to masks"),
        ("code", 'import matplotlib.pyplot as plt\nimport numpy as np\nfrom openwarehouse.config import load_config\nfrom openwarehouse.experiments import image_path, load_image, load_manifest\nfrom openwarehouse.pipeline import PerceptionPipeline\nfrom openwarehouse.utils import resolve_device\n\nos_root = __import__("os")\nos_root.chdir("..")\nconfig = load_config("open_images", "poc")\nmanifest = load_manifest(config)\nimage_id = "00a1c9247cf3e24c"\nsplit = manifest[manifest["image_id"] == image_id]["split"].iloc[0]\nimage = load_image(image_path(config, split, image_id))\npipeline = PerceptionPipeline(config, device=resolve_device(config["device"]))\ninstances = pipeline(image, "person", min_score=0.32)\nprint([(i.label, round(i.score, 3), round(i.encoder_score, 4), int(i.mask.sum())) for i in instances])\nos_root.chdir("notebooks")'),
        ("code", 'plt.imshow(image)\nfor instance in instances:\n    overlay = np.zeros((*instance.mask.shape, 4))\n    overlay[instance.mask] = (0.9, 0.2, 0.1, 0.45)\n    plt.imshow(overlay)\nplt.axis("off")\nplt.show()'),
        ("md", "## Detection by prompt condition (held-out)"),
        ("code", 'detector = pd.read_csv(RESULTS + "detector_heldout.csv")\ndetector.dropna(subset=["ap50"]).pivot(index="class_name", columns="condition", values="ap50").round(3)'),
        ("code", 'display(Image(FIGURES + "prompt_sensitivity_ap50.png"))\npd.read_csv(RESULTS + "ap50_bare_vs_article_heldout.csv").round(3)'),
        ("md", "## Operating point at the frozen threshold"),
        ("code", 'pd.read_csv(RESULTS + "operating_point_heldout.csv").round(3)'),
        ("md", "## Masks and rescoring\n\nThe ground-truth-box column is the upper bound for SAM. Person has only 40 instances."),
        ("code", 'masks = pd.read_csv(RESULTS + "masks_heldout.csv")\nmasks.groupby("class_name")[["oracle_box_mask_iou", "detected_mask_iou", "detected_box_iou"]].mean().round(3)'),
        ("code", 'pd.concat([pd.read_csv(RESULTS + f"rescoring_{r}.csv") for r in ("dev", "heldout")]).round(3)'),
    ],
    "09_failure_analysis": [
        ("md", "# 09 - Failure Analysis\n\nEvidence for the failure modes in `docs/RESULTS.md`. Probes were written on dev and run unchanged on held-out."),
        ("code", SETUP),
        ("md", "## Failure mode 1: function words create detections of absent objects"),
        ("code", 'pd.concat([pd.read_csv(RESULTS + f"token_summary_{r}.csv").assign(role=r) for r in ("dev", "heldout")]).round(3)'),
        ("code", 'display(Image(FIGURES + "absent_object_fire_rate.png"))\ndisplay(Image(FIGURES + "hallucination_0.png"))\ndisplay(Image(FIGURES + "hallucination_1.png"))'),
        ("md", "## Failure mode 2: negation and state are ignored"),
        ("code", 'probe = pd.read_csv(RESULTS + "modifier_heldout.csv")\nprobe.assign(fires=probe["score"] >= 0.32, same_box=probe["top_box_iou"] >= 0.9).groupby(["class_name", "prompt"])[["fires", "same_box", "score"]].mean().round(3)'),
        ("code", 'display(pd.read_csv(RESULTS + "siglip_negation_heldout.csv").round(3))\ndisplay(Image(FIGURES + "siglip_negation_auroc.png"))\ndisplay(Image(FIGURES + "negation_0.png"))'),
        ("md", "## Failure mode 3: the detector cannot reject lookalikes or scene-level concepts"),
        ("code", 'siglip = pd.read_csv(RESULTS + "siglip_heldout.csv").query("condition == \'photo\'").set_index("class_name")["auroc"]\ndetector = pd.read_csv(RESULTS + "detector_heldout.csv").query("condition == \'bare\'").set_index("class_name")["image_auroc"]\npoints = pd.read_csv(RESULTS + "operating_point_heldout.csv").set_index("class_name")["false_positive_rate"]\npd.DataFrame({"siglip_auroc": siglip, "detector_auroc": detector, "detector_fpr": points}).round(3).sort_values("detector_auroc")'),
        ("md", "## Mechanism: SigLIP occlusion\n\nThe image is labeled as a forklift in Open Images but shows a street sweeper."),
        ("code", 'display(pd.read_csv(RESULTS + "checks.csv").round(4))\ndisplay(Image(FIGURES + "siglip_occlusion.png"))'),
    ],
}

for name, cells in NOTEBOOKS.items():
    notebook = nbf.v4.new_notebook()
    notebook.cells = [
        nbf.v4.new_markdown_cell(text) if kind == "md" else nbf.v4.new_code_cell(text)
        for kind, text in cells
    ]
    NotebookClient(
        notebook, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}
    ).execute()
    nbf.write(notebook, ROOT / f"{name}.ipynb")
    print("executed", name)
