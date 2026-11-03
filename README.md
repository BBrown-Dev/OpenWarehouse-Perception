# OpenWarehouse-Perception
Open-vocabulary warehouse perception and safety assessment platform for autonomous forklift operations. Combines CLIP/SigLIP, Grounded SAM, LoRA adaptation, classical OpenCV pipelines, robustness testing, edge deployment optimization, and AI safety evaluation.

> **Status: proof of concept.** SigLIP + Grounded SAM zero-shot detection and segmentation is implemented and evaluated on Open Images V7. For decision-makers, start with the [deployment readiness report](docs/DEPLOYMENT_READINESS.md) (PASS/FAIL and go/no-go), then the [production readiness assessment](docs/EVALUATION.md). Try it with `python scripts/demo.py --image photo.jpg --prompt "box"`. Details are in [Results and Failure Analysis](docs/RESULTS.md) and [Design Decisions](docs/DESIGN_DECISIONS.md). LoRA, edge, robustness, classical preprocessing, and the safety assessor remain stubs, so notebooks 03 and 05 to 08 are unimplemented templates.
>
> Reproduce with the ordered steps in [`docs/REVIEW.md`](docs/REVIEW.md#6-how-to-reproduce). Use the project `.venv`, and run `scripts/record_image_hashes.py --verify` after importing the data, before any experiment.

## Pipeline

```
image -> classical preprocessing -> Grounded SAM (text prompts -> boxes + masks)
      -> CLIP/SigLIP + LoRA (re-score each instance) -> rule-based safety assessment
      -> SceneAssessment (risk level + hazards)
```

Robustness testing wraps the whole pipeline. Edge optimization exports and benchmarks its models.

## Layout

| Path | Purpose |
| --- | --- |
| `configs/` | One YAML per stage, merged over `base.yaml` |
| `notebooks/` | Thin, ordered drivers. Logic belongs in `src/` |
| `src/openwarehouse/data/` | Datasets, splits, open-vocabulary prompts |
| `src/openwarehouse/classical/` | OpenCV preprocessing and scene cues |
| `src/openwarehouse/models/` | CLIP/SigLIP encoder and LoRA adaptation |
| `src/openwarehouse/segmentation/` | Grounded SAM |
| `src/openwarehouse/safety/` | Hazard rules and scene risk assessment |
| `src/openwarehouse/robustness/` | Corruptions and severity sweeps |
| `src/openwarehouse/edge/` | Export, quantization, benchmarks |
| `src/openwarehouse/evaluation/` | Perception, calibration, and safety metrics |
| `src/openwarehouse/pipeline.py` | End-to-end inference |
| `data/`, `artifacts/` | Local data and generated outputs (not committed) |

## Notebooks

| Notebook | Stage |
| --- | --- |
| `00_setup_and_config` | Environment, seeds, config, experiment tracking |
| `01_data_preparation` | Ingest, EDA, leakage-safe splits, classical preprocessing |
| `02_clip_baseline` | Zero-shot CLIP/SigLIP baseline |
| `03_lora_adaptation` | LoRA training and comparison with the baseline |
| `04_grounded_sam` | Open-vocabulary detection and segmentation |
| `05_safety_assessment` | Hazard rules and safety metrics |
| `06_robustness_testing` | Corruption sweeps and degradation analysis |
| `07_edge_optimization` | Export, quantization, latency benchmarks |
| `08_evaluation_report` | Consolidated results and go/no-go review |
| `09_failure_analysis` | Failure modes and mechanism evidence, loaded from saved results |

Run them in order. Each stage reads earlier outputs from `artifacts/`.

## Setup

```
pip install -e ".[notebook,edge,dev]"
```

## Safety scope

This is a research assessment, not a certified safety function. It does not replace the safety-rated systems on a forklift, such as those required under ISO 3691-4. Low-confidence or failed perception must escalate risk rather than report a clear scene. Model output is advisory only and never confirms that an area is clear; the binding requirements are in [Design Decisions](docs/DESIGN_DECISIONS.md#safety-requirements).
