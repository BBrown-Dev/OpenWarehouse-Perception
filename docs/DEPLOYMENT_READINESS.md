# Deployment Readiness Report: Open-Vocabulary Warehouse Inspection

**System.** Grounding DINO tiny (text-prompted boxes) → SAM ViT-B (masks) → SigLIP base/16-224 (crop and scene scoring), zero-shot, no training. Prompts are warehouse object names such as "box", "shelf", "forklift" and "person".
**Evaluation data.** Open Images V7: 1,705 held-out images (validation and test splits) across 12 warehouse categories, plus 1,666 unused fresh images for one pre-registered test. No warehouse-camera footage was available.
**Settings.** Choices frozen on development data before any held-out result was seen: detector prompt as a bare noun, detector threshold 0.32, SigLIP wording "a photo of …". Hardware: NVIDIA RTX 2070 SUPER (8 GB), 8-thread Intel desktop CPU.
**Date.** 2026-11-02. Detailed evidence: [EVALUATION.md](EVALUATION.md), [RESULTS.md](RESULTS.md), [REVIEW.md](REVIEW.md); raw tables in `docs/results/`.

---

## Decision

| | |
| --- | --- |
| **Deployment assessment** | **FAIL**: 9 of 11 criteria fail; 2 pass, and only in the fp16 configuration (memory, compression agreement) |
| **Recommendation** | **NO-GO** for production deployment, including any use that triggers an action without a person checking it. Permissible instead: an offline pilot that ranks or pre-labels recorded footage for human reviewers, who make every decision. |
| **Safety rationale** | The system misses about one in three individual people and reports objects that are not there with high confidence, so any automated action based on it could fail to stop for a person or act on a hazard that does not exist. |

---

## 1. Deployment criteria

The targets were set for this report, after the results were known, so they are a stated standard, not a pre-registered test. Where the project's binding requirements already fix a rule ([DESIGN_DECISIONS.md](DESIGN_DECISIONS.md#safety-requirements)), the target follows it. The latency and memory targets are assumptions, marked as such, because no deployment device or frame rate has been specified.

| # | Criterion | Target | Measured (held-out unless noted) | Result |
| --- | --- | --- | --- | --- |
| 1 | Individual people missed (box IoU ≥ 0.5) | ≤ 1%, upper 95% bound | 35.4% [25.8, 44.3] of 1,391 | **FAIL** |
| 2 | False positive rate at the operating point, inspection categories | ≤ 10% per category | Box 79%, shelf 81%, truck 90%, forklift 100%, person 22% | **FAIL** |
| 3 | Confident false positives (score ≥ 0.5) on adversarial absent-object queries | ≤ 5% | 20.5% [17.2, 24.3] of 493 | **FAIL** |
| 4 | Condition prompts ("a damaged box") distinguish condition | Measured accuracy ≥ 90% | Unmeasurable (no condition labels); fires on 97% of ordinary box images | **FAIL** |
| 5 | Robustness: people found drops ≤ 5 points at severity 3 for every measured corruption | ≤ 5 points | Low light −5.8 (67.8% → 62.0%); glare −0.7; motion blur already −7.4 at severity 2 (not measured at 3) | **FAIL** |
| 6 | Calibration before any score is shown or thresholded as a probability | Per-class ECE ≤ 0.05 | Detector mean 0.20, SigLIP mean 0.47 | **FAIL** |
| 7 | Latency per image and prompt, end to end, on the deployment GPU (assumed 10 frames/s) | ≤ 100 ms | 580 to 1,060 ms on the RTX 2070 SUPER (detector alone 238 to 299 ms) | **FAIL** |
| 8 | Total model memory (assumed edge budget) | ≤ 1 GB | 1,877 MB fp32; **939 MB** with fp16 weights | **PASS** (fp16 only) |
| 9 | Compressed model agrees with the evaluated model | ≥ 99% of decisions | fp16 weights 99.0–100%; int8 71–80% | **PASS** for fp16, **FAIL** for int8 |
| 10 | Evaluated on deployment-distribution data | Required | None available; Open Images photographs only | **FAIL** |
| 11 | Results reproducible from the documented procedure | Pre-registered verification passes | Results recompute exactly from cached outputs and SigLIP reproduces, but the detector check failed for 2 of 12 classes, whose original batches included later-excluded images; a clean regeneration would shift their scores by up to 0.086 (Section 7) | **FAIL** |

---

## 2. Model accuracy

Held-out Open Images V7, frozen settings. Box metrics need box labels, which exist for five categories; the others are scored at image level only. Intervals are 95% bootstrap intervals over images.

| Category | Images (with / without) | AP50 [95% CI] | Objects found (IoU ≥ 0.5) | Image recall at 0.32 | False positive rate at 0.32 | Detector AUROC | SigLIP AUROC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Person | 420 / 532 | 0.65 [0.56, 0.75] | 64.6% of 1,391 | 97.1% | 22.0% | 0.97 | 0.76 |
| Truck | 122 / 136 | 0.46 [0.39, 0.55] | 81.7% of 142 | 94.3% | 90.4% | 0.56 | 0.71 |
| Ladder | 117 / 106 | 0.55 [0.43, 0.69] | 70.4% of 54 | 79.5% | 48.1% | 0.73 | 0.49 |
| Box | 185 / 125 | 0.44 [0.36, 0.56] | 70.1% of 154 | 88.1% | 79.2% | 0.61 | 0.60 |
| Shelf | 298 / 186 | 0.16 [0.11, 0.27] | 18.1% of 849 | 98.0% | 80.6% | 0.70 | 0.79 |
| Forklift | 25 / 14 | no box labels | — | 100% | 100% | 0.67 | **0.97** |
| Shelving | 121 / 52 | — | — | 92.6% | 92.3% | 0.49 | 0.93 |
| Hard hat | 17 / 22 | — | — | 100% | 100% | 0.42 | 0.90 |
| Warehouse | 30 / 65 | — | — | 100% | 100% | 0.52 | 0.80 |
| Inventory | 31 / 72 | — | — | 96.8% | 98.6% | 0.47 | 0.79 |
| Aisle | 86 / 53 | — | — | 97.7% | 96.2% | 0.68 | 0.77 |
| Cardboard | 126 / 22 | — | — | 97.6% | 100% | 0.43 | 0.36 |

Mean AP50 over the five box categories is 0.45. Masks: SAM given the correct box reaches mask IoU 0.84 (box), 0.89 (truck), 0.77 (person); from the detector's own boxes, 0.67, 0.74 and 0.72.

**Reading the table.** The detector fires on most images whether or not the object is there (image recall and false positive rate are both high), so its useful signal is in the ranking, not the yes/no decision. SigLIP ranks some scene-level categories well (forklift 0.97, shelving 0.93) but has no boxes. The 0.32 threshold was chosen on pooled development data and suits no single category well.

---

## 3. Robustness

### 3.1 Distribution shifts (measured on real data)

| Shift | Effect |
| --- | --- |
| Development images (Open Images train) → held-out (validation/test) | Person AP50 0.73 → 0.65; people found 82% → 65%; SigLIP ladder AUROC 0.72 → 0.49 |
| General scenes → warehouse-like scenes (247 held-out images verified as warehouse, shelving, aisle or inventory scenes) | Images with people missed entirely 2.9% → 8.2%; individual people missed 35.4% → 46.3% |
| Apparent size of a person (stand-in for distance) | People found: 92% at ≥ 50% of image height, 71% at 10–25%, 53% at 5–10%, **26% below 5%** |
| Thresholds set on development data → fresh images | A SigLIP presence check lost ~15 points of recall for person and shelf, against ≤ 5 expected |
| Prompt wording | Person AP50 0.65 ("person") → 0.13 ("a photo of a person") on the same images |

### 3.2 Corrupted-condition accuracy by shift type

Person detection on 300 held-out images (150 with people, 150 without), frozen settings, synthetic corruptions with reviewer-chosen severities, not calibrated to real cameras. Robustness to corruptions is outside the project scenario, so the sweep was stopped once the conditions below were measured.

| Shift type | Severity | People found (IoU ≥ 0.5) | Images with people flagged | False alarms on images without people | AP50 |
| --- | --- | --- | --- | --- | --- |
| Clean | — | 67.8% | 95.3% | 22.7% | 0.65 |
| Low light | 1 / 2 / 3 / 4 / 5 | 67.8 / 66.4 / 62.0 / 54.9 / **40.0%** | 92.7 / 93.3 / 90.0 / 88.0 / **74.7%** | 17.3 / 18.7 / 19.3 / 16.0 / 16.7% | 0.65 / 0.64 / 0.62 / 0.58 / **0.45** |
| Glare | 1 / 2 / 3 / 4 / 5 | 68.3 / 67.6 / 67.1 / 65.3 / 61.1% | 94.7 / 96.0 / 94.7 / 92.7 / 93.3% | 23.3 / 22.0 / 21.3 / 23.3 / 20.7% | 0.65 / 0.65 / 0.65 / 0.63 / 0.60 |
| Motion blur | 1 / 2 | 66.4 / 60.4% | 94.7 / 91.3% | 19.3 / 16.7% | 0.64 / 0.59 |
| Defocus blur, sensor noise, JPEG compression, dust haze, occlusion | — | **Not measured** (outside scenario scope) | | | |

Low light is the most damaging measured shift: at the darkest setting the detector finds 40% of people instead of 68%, and misses people entirely in a quarter of the images that contain them. Motion blur costs 7 points by severity 2. Glare costs at most 7 points even at severity 5. False alarms do not rise under these corruptions.

---

## 4. Inference latency and model size

Batch size 1, latency includes preprocessing. The benchmark ran each variant as one block and partly overlapped other jobs, so its latencies are indicative; the profiler figures were measured alone with variants interleaved.

| Component | Precision | Size (MB) | Latency per image (ms), GPU | Latency per image (ms), CPU | Agreement with fp32 |
| --- | --- | --- | --- | --- | --- |
| Grounding DINO tiny | fp32 | 689.4 | 267 (forward alone 192) | 5,841 | 100% |
| Grounding DINO tiny | fp16 weights | 344.9 | 238 (forward alone 180) | — | 99.0% |
| Grounding DINO tiny | int8 dynamic | 253.2 | — | 5,445 | 80.0% |
| SigLIP base/16-224 | fp32 | 812.8 | 15.7 | 284 | 100% |
| SigLIP base/16-224 | fp16 weights | 406.5 | 15.7 (profiler) | — | 100% |
| SigLIP base/16-224 | int8 dynamic | 287.4 | — | 404 | 71.0% |
| SAM ViT-B | fp32 / fp16 | 375.1 / 187.6 | not timed separately | — | not measured |
| **Full pipeline** (detect → segment → score crops) | fp16 autocast | 1,877 (fp32 weights) | **580 to 1,060 per image and prompt** (5 demo calls) | — | — |

Batched, fp16 weights are 1.7× (detector, batch 4) to 3.2× (SigLIP, batch 32) faster than fp32 per image; at batch 1 they are no faster, because the GPU is mostly waiting on preprocessing and launch overhead (`profile_precision.csv`).

---

## 5. Tradeoff analysis

| Lever | What it buys | What it costs | Assessment |
| --- | --- | --- | --- |
| **Precision: fp16 weights** | Half the memory (1,877 → 939 MB); 1.7–3.2× faster when batched | ≤ 1% of predictions change; no speed-up at batch 1 | Worth it; passes the agreement rule |
| **Precision: int8 dynamic** | 63–65% smaller than fp32 | 20–29% of predictions change, 9–11% of presence decisions flip; no speed-up (SigLIP 42% slower on CPU) | Rejected |
| **Batching frames or cameras** | The only route to fp16 speed-ups on this GPU | Added latency per frame while a batch fills; scores shift up to 0.05 with batch composition | Needs its own acceptance test |
| **Detector threshold** | Lower: more objects found. Higher: fewer false alarms | No single threshold works: per-class thresholds chosen on development data cut person false alarms only from 22% to 15%, and most other categories stay above 90% | Thresholding cannot fix the false positive problem |
| **SigLIP presence check before detection** (pre-registered test on fresh images) | Person false alarms 31.9% → 19.7%; shelf 80.3% → 59.6% | Person recall 97.1% → 81.5%; shelf 95.7% → 80.7%, beyond the allowed loss; truck gain only 94.4% → 89.7% | Rejected: trades misses for false alarms |
| **SigLIP crop rescoring after detection** | Slightly better box AP for box and ladder on development data | Worse for person, shelf and truck | Rejected |
| **Prompt wording** | Bare nouns: AP50 0.45 (5 categories) | Articles and templates create detections of absent objects; "a photo of …" drops person AP50 to 0.13 | Keep bare nouns; never put negation or condition words in detector prompts |
| **Model choice (tiny models)** | Fits an 8 GB GPU in fp32 and under 1 GB in fp16 | Lower accuracy than larger variants (not measured here) | A larger detector is the obvious next experiment, under the same criteria |

---

## 6. Evidence

### 6.1 Evidence supporting the NO-GO decision

1. **People are missed.** 35.4% [25.8, 44.3] of individual people are not localized; 74% of small or distant people are missed; on warehouse-like scenes 46.3% (`safety_person_miss.csv`, `safety_person_recall_by_height.csv`).
2. **Absent objects are reported confidently.** 30.2% [27.8, 32.7] of queries for absent categories score ≥ 0.5, and 10.4% ≥ 0.7; under adversarial queries 20.5% score ≥ 0.5; "forklift" fires on all 14 forklift-free images, with scores up to 0.84 (`safety_confident_errors.csv`).
3. **Condition and negation words are ignored.** "a damaged box" fires on 97% of ordinary box images; "no forklift" fires on 100% of images that contain a forklift (`modifier_heldout.csv`).
4. **Scores are not probabilities.** Per-class ECE averages 0.20 for the detector and 0.47 for SigLIP (`metrics_ece.csv`).
5. **Too slow for real time.** 580–1,060 ms per image and prompt end to end on the evaluation GPU.
6. **No deployment-distribution evidence exists.** Every number comes from general photographs; the warehouse-like slice is worse on every measure.
7. **Mitigations tried did not fix it.** Per-class thresholds, a SigLIP presence check (pre-registered, tested once on fresh images) and crop rescoring were all measured and none brought any category within the criteria.

### 6.2 Evidence that would reverse the decision

All of the following, measured on a pre-registered, warehouse-camera test set (labeled by site, camera, lighting and distance; split by site; at least 3 sites):

1. **People:** at least 300 independent person instances, spread across sites and shifts, with an upper 95% bound on the miss rate ≤ 1%, including people under 5% of image height within the truck's stopping distance.
2. **False positives:** false positive rate ≤ 10% and confident false positives (≥ 0.5) ≤ 5% for every inspection category, including adversarial absent-object queries.
3. **Conditions:** a labeled set of at least 200 damaged, fallen or crushed packages with condition detection accuracy ≥ 90%; this will likely need a trained condition classifier on detected crops, not prompting.
4. **Robustness:** ≤ 5-point drop in people found at severity 3 for every corruption type, on real low-light and motion footage, not only simulated.
5. **Latency:** p99 ≤ 100 ms per frame for the whole pipeline on the target device, after any compression passes the ≥ 99% agreement rule.
6. **Calibration:** per-class ECE ≤ 0.05 on deployment-prevalence data, if any score is shown or thresholded as a probability.
7. **Reproducibility:** an independent clean rerun matching the committed results, after regenerating the Aisle and Box detector outputs with the clean manifest (Section 7.1).

Even if all seven are met, the system remains advisory: it may add caution, never clear an area (Safety requirements 1–3).

---

## 7. Reproducibility guide for another engineer

### 7.1 Status

Verification run on 2026-11-02 (`docs/results/verification_cache_*`), 76 minutes on the reference machine:

| Check | Result |
| --- | --- |
| Image bytes match the recorded hashes (5,348 images) | **Pass** |
| Every committed results table recomputed from the cached model outputs with the current code | **Pass, exactly.** No committed table changed; the only data change is one new robustness row (motion blur, severity 2) |
| Notebooks re-execute | **Pass.** Only cell IDs, progress text and image bytes changed |
| 50 unit tests | **Pass** |
| SigLIP rerun on a seeded 10% of original batches (3,378 image-prompt pairs), tolerance 0.01 | **Pass.** Largest difference 0.000005 |
| Detector rerun on a seeded 10% of batches rebuilt from the current manifest (3,401 pairs), tolerance 0.001 for 99% of pairs, no decision flips | **Fail.** 94.1% within tolerance, largest difference 0.086, 2 decisions flipped at 0.32 |

**Why the detector check failed.** Every mismatch is in two classes, Aisle and Box; the other 60 of 72 class-wording combinations reproduce exactly. Those two classes were scored before the leakage exclusions were applied, so their original batches also contained 103 images (on the exclusion list) that the current manifest omits. The excluded images' own predictions are dropped before any metric, so nothing leaks, but sharing a batch changed the padding and therefore the scores of their neighbours. Rebuilding Aisle and Box batches in the cache file's original order reproduces all 546 sampled pairs **exactly** (`verify_caches.py --file-order Aisle Box`). That follow-up was designed after the failure was seen.

**What this means.** The committed results follow exactly from the cached model outputs, and the cached outputs match the current code and pinned models. But a clean regeneration from scratch applies the exclusions first and would batch Aisle and Box differently: their scores would move by up to 0.086 and about 1 in 1,700 decisions would flip, so their reported metrics would not reproduce exactly. All other classes would. Fix, if exact clean reproduction is required: regenerate the Aisle and Box detector outputs with the clean manifest (~7 min of GPU) and recompute the summaries (~50 min, mostly single-threaded bootstrap), then re-baseline the affected tables.

### 7.2 Prerequisites

- Windows 10/11 or Linux; Python 3.10 (tested with 3.10.4).
- NVIDIA GPU with at least 8 GB memory and a driver supporting CUDA 12.6 (tested on an RTX 2070 SUPER). CPU-only runs work but the detector takes ~6 s per image.
- About 8 GB free disk: 5.7 GB of images, 0.1 GB of annotations, plus model weights (~1.9 GB) in the Hugging Face cache.
- Network access to Hugging Face and the Open Images bucket for the first run.

### 7.3 Steps

1. **Get the code** at the evaluated commit (the `review/zero-shot-poc` branch must first be pushed to the shared repository; at the time of writing it exists only locally):
   ```
   git clone https://github.com/BBrown-Dev/OpenWarehouse-Perception.git
   cd OpenWarehouse-Perception
   git checkout review/zero-shot-poc
   ```
2. **Create the environment** with the exact versions used:
   ```
   python -m venv .venv
   .venv\Scripts\activate            # Linux: source .venv/bin/activate
   pip install -r requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cu126
   pip install -e . --no-deps
   ```
   Check: `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` prints `2.14.1+cu126 True`.
3. **Run the tests**: `python -m pytest -q`. Expected: 50 passed.
4. **Download the data**: `python scripts/import_open_images.py` (network-bound; uses FiftyOne).
5. **Verify the images**: `python scripts/record_image_hashes.py --verify`. On a fresh machine it lists the 1,666 fresh gate images as missing and exits with an error until those images are downloaded (they are fetched by `run_gate_test.py` in a full regeneration). Any other mismatch, or a mismatch on an image that is present, means the data differs: stop.
6. **Model weights** download automatically at first use, pinned to the exact commits in `configs/revisions.yaml`. Loading an unpinned model fails by design.
7. **Choose a run:**

   | Run | Commands | Time on the reference machine | What it proves |
   | --- | --- | --- | --- |
   | **Demo only** | `python scripts/demo.py --image <photo.jpg> --prompt "box"`, or `--examples` | ~1 min | The pipeline works end to end |
   | **Verification** (recommended) | Cached-model run: `run_poc.py`, `audit_outputs.py`, `build_report_assets.py`, `compute_metrics.py`, `compute_slices.py`, `compute_safety_metrics.py`, `run_robustness.py --cached-only`, `demo.py --examples`; then `verify_caches.py`, `verify_caches.py --file-order Aisle Box`, `build_notebooks.py`, `python -m pytest -q` | ~1.5 h (76 min measured; about 50 min of it is single-threaded bootstrap resampling) | Every reported number follows from the model outputs, and the model outputs still match the code |
   | **Full regeneration** | Empty `artifacts/`, then all steps in [REVIEW.md Section 6](REVIEW.md#6-how-to-reproduce) | ~7.5–9.5 h | Everything from scratch |

   The verification run needs the cached model outputs in `artifacts/poc/` (not in git; about 13 MB plus the robustness cache). Without them, run the full regeneration.
8. **Compare with the committed results**: `git status docs/results` and `git diff --stat docs/results`. Any change to a committed table is a reproduction failure to investigate; new files are expected only where a step adds output. `docs/results/verification_cache_sample.json` reports the model-rerun check: SigLIP must pass; the detector is expected to fail only for Aisle and Box (Section 7.1), and `verification_cache_file_order.json` must then pass with `"max_abs_diff": 0.0`.
9. **Do not** re-freeze `selection.json` (step 3 of a full regeneration re-derives it and must match), and do not treat a rerun of `run_gate_test.py` as a new confirmatory test: it re-selects the same fresh images by seed.

### 7.4 Known sources of small differences

- Detector scores depend on batch composition (up to 0.05); keep `detector_batch_size: 4` and manifest order.
- GPU kernels in fp16 autocast are not bit-identical across GPU models or drivers; expect agreement within the tolerances in `scripts/verify_caches.py`, not exact equality, on other hardware.
- Latency figures are machine-specific.
