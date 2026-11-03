# OpenWarehouse-Perception

Open-vocabulary warehouse inspection: find and segment warehouse objects described in plain language ("box", "shelf", "forklift", "person") without training a model for each new item. Built from open-source Grounding DINO, SAM and SigLIP via Hugging Face, and evaluated zero-shot on Open Images V7.

> **Status: proof of concept, not production-ready.** The deployment assessment is **FAIL** with a **NO-GO** recommendation: the system misses about a third of individual people and reports absent objects with high confidence. Start with the [deployment readiness report](docs/DEPLOYMENT_READINESS.md) (criteria, go/no-go, evidence, reproducibility guide), then the [production readiness assessment](docs/EVALUATION.md) (per-use-case verdicts, risk register, examples). Methods and detailed results are in [Results and Failure Analysis](docs/RESULTS.md), design choices and binding safety requirements in [Design Decisions](docs/DESIGN_DECISIONS.md), and the review history in [Review Log](docs/REVIEW.md).

## What is implemented

```
text prompt ──┐
              ├─> Grounding DINO tiny ──(score ≥ 0.32)──> SAM ViT-B ──> SigLIP base/16 ──> boxes, masks, scores
image ────────┘     text-prompted boxes                   mask per box    scores each crop
image ──────────> SigLIP: whole-image presence ranking
```

- **Demo**: `python scripts/demo.py --image photo.jpg --prompt "box" --prompt "damaged box"` writes boxes and scores (JSON), one mask per detection (PNG) and an overlay to `artifacts/demo/`. `--examples` builds the report's example figure.
- **Evaluation**: zero-shot results on 1,705 held-out images across 12 warehouse categories, with choices frozen on development data; failure analysis; hallucination and confident-false-positive rates; calibration; robustness to simulated lighting, glare and motion blur; latency, model size and precision benchmarks; a pre-registered test on fresh images.
- **Out of scope**: training or adapting any model, camera and video input, safety decision logic, and a deployment build for edge hardware. What these would need before any deployment is set out in the [deployment readiness report](docs/DEPLOYMENT_READINESS.md).

## Layout

| Path | Purpose |
| --- | --- |
| `configs/` | Settings merged over `base.yaml` (`open_images`, `poc`, `gate`, `robustness`); `revisions.yaml` pins each model to an exact Hugging Face commit |
| `src/openwarehouse/segmentation/grounded_sam.py` | Grounding DINO detection and SAM segmentation |
| `src/openwarehouse/models/vision_language.py` | SigLIP scoring |
| `src/openwarehouse/pipeline.py` | Prompt in, scored and segmented instances out |
| `src/openwarehouse/experiments.py` | Running and caching the models, freezing dev choices, held-out summaries |
| `src/openwarehouse/data/` | Open Images selection and import, leakage checks |
| `src/openwarehouse/evaluation/` | AP50, IoU and AUROC with bootstrap intervals; CHAIR and POPE; calibration (ECE) |
| `src/openwarehouse/failure_analysis.py`, `gate.py` | Failure probes, mask and rescoring evaluation; the pre-registered presence-check test |
| `src/openwarehouse/robustness/` | Image corruptions and severity sweeps |
| `scripts/` | Every experiment, metric, report and verification step; run order in [REVIEW.md §6](docs/REVIEW.md#6-how-to-reproduce) |
| `docs/` | Reports; `docs/results/` holds every result table, `docs/figures/` every figure |
| `tests/` | Unit tests for metrics, leakage guards, gate rules, robustness and the demo |
| `notebooks/` | Thin drivers that display saved results |
| `data/`, `artifacts/` | Downloaded images and cached model outputs (not committed) |

## Notebooks

| Notebook | Content |
| --- | --- |
| `01_data_preparation` | Evaluated images per class and role (dev or held-out), and leakage exclusions |
| `02_clip_baseline` | Zero-shot SigLIP baseline |
| `04_grounded_sam` | Open-vocabulary detection, segmentation and rescoring |
| `09_failure_analysis` | Failure modes and mechanism evidence |

The notebooks read saved results from `docs/results/` and are rebuilt with `python scripts/build_notebooks.py`. Robustness, latency and the go/no-go decision are in `docs/EVALUATION.md` and `docs/DEPLOYMENT_READINESS.md`.

## Setup

Python 3.10 and an NVIDIA GPU with at least 8 GB memory and a driver supporting CUDA 12.6 (tested on an RTX 2070 SUPER). Install the exact versions the results were produced with:

```
python -m venv .venv
.venv\Scripts\activate            # Linux: source .venv/bin/activate
pip install -r requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cu126
pip install -e . --no-deps
python -m pytest -q               # 50 tests
```

Then download the evaluated images with `python scripts/import_open_images.py` (it reuses the committed manifest) and check them with `python scripts/record_image_hashes.py --verify`. The full step-by-step guide, including which runs another engineer can do and how long each takes, is in [DEPLOYMENT_READINESS.md §7](docs/DEPLOYMENT_READINESS.md#7-reproducibility-guide-for-another-engineer).

## Safety scope

This is a research assessment, not a certified safety function. It does not replace the safety-rated systems on a forklift, such as those required under ISO 3691-4. Low-confidence or failed perception must escalate risk rather than report a clear scene. Model output is advisory only and never confirms that an area is clear; the binding requirements are in [Design Decisions](docs/DESIGN_DECISIONS.md#safety-requirements).
