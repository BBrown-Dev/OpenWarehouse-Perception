# Review Log

A review of the zero-shot proof of concept, carried out on 2026-11-01 and 2026-11-02. It records what was checked, what was wrong, what changed, and which results moved. Raw outputs are in `docs/results/`.

The code had no version history before this review (the repository held only an initial commit), so the code changes below are described by their effect and the output that shows it, not by diff. The commit that adds this file is the first baseline that can be diffed.

**Scope agreed for the review.** Open Images V7 is accepted as a proxy for warehouse imagery in this prototype. Four follow-up items were approved: finish the corrected re-scoring and update `RESULTS.md`; fix report wording the evidence does not support; commit the work on a branch; and run the SigLIP presence check once on fresh images.

## 1. Errors found and fixed

| Issue | Effect | Fix | Evidence |
| --- | --- | --- | --- |
| Box labels ignored the Open Images class hierarchy. Person AP50 counted only boxes labeled Person, not Man, Woman, Boy or Girl. | Held-out Person AP50 understated (0.570, now 0.651) and 25 held-out images dropped. Mean AP50 0.436, now 0.452. Person masks evaluated: 12 instances, now 40. The other four box classes have no child classes and are unchanged. | Box labels expanded to child classes; detector summaries, rescoring and mask evaluation rerun. | `audit_ap_hierarchy.csv`, `detector_heldout.csv`, `masks_heldout.csv` |
| Frozen choices had to be shown unaffected by the label fix. | None. Dev re-selection gives `bare` and threshold 0.32 again, matching `selection.json`. | Assertion in `scripts/run_validation.py`. | run log |
| SigLIP crop rescoring used a prompt wording other than SigLIP's own frozen choice. | Fused and SigLIP-only AP50 changed for all five classes. Detector AP50 in the same table is unchanged for the four classes without child labels, which isolates the change to the SigLIP side. On dev, fusion is now mixed (slightly higher for Box and Ladder) rather than lower on all five classes. The decision not to adopt it stands. | `evaluate_rescoring` scores crops with the frozen `photo` wording. | `rescoring_dev.csv`, `rescoring_heldout.csv`; pre-review copies remain in `artifacts/poc/failure/` (not tracked) |
| `scripts/build_report_assets.py` copied every CSV from `artifacts/poc/failure/` into `docs/results/`. | Rebuilding the report would have overwritten the corrected rescoring and mask tables with pre-fix ones. | Those three files are now skipped by the copy. | `scripts/build_report_assets.py` |

## 2. Report wording corrected

| Claim before | Problem | Now |
| --- | --- | --- |
| "a giraffe" fired on 86% of verified negative images | The images are verified to have no forklift; giraffe absence was assumed. One noun is also a narrow test. | Marked as assumed. Ten more unrelated control nouns were tested on 60 dev no-forklift images: 78% mean false detection with an article, 30% bare. "Lighthouse", "snowman" and "kangaroo" fire about as often bare, so dropping the article is not a general fix. |
| Occlusion: hiding parts of the vehicle often raises the SigLIP score | Drawn from one image, which is mislabeled (a street sweeper verified as a forklift). | Withdrawn. Only the 20-image map correlation (0.66) is kept, as consistent with, not proof of, no localization. |
| Per-class thresholds "would be a fairer test" and a listed mitigation | Checked: they barely help. False positive rate stays at 92% or higher for 8 of 12 classes. | Reported as checked and largely ineffective. |
| Post-hoc held-out analyses presented alongside confirmatory ones | Per-class thresholds, the SigLIP gate check, content-token scoring and control nouns were designed after held-out results were seen. | Labeled exploratory in the text and in Limitations. |
| Cascade "may help but has not been run" | It had been run exploratorily; the pre-registered test now exists. | Points to the pre-registered result (Section 8 of `RESULTS.md`). |

## 3. Checks added (no change to conclusions)

| Check | Result | File |
| --- | --- | --- |
| Fresh-process repeat of the 18 reproduced examples | Maximum top-score difference 0.00005 | `validation_fresh_process.csv` |
| Batch vs single-image detector scores (100 dev images) | Correlation 0.996, median difference 0.006, maximum 0.05 | `checks.csv` |
| Detector fp32 vs fp16 autocast on GPU (100 images) | Median difference 0.001, maximum 0.012, no decision flips | `validation_quantization.csv` |
| SigLIP fp32 vs dynamic int8 on CPU (151 images) | Dev forklift AUROC 0.959 vs 0.960. This understated the risk; see Section 7 | `validation_quantization.csv` |
| Detector fp32 vs dynamic int8 on CPU (40 images) | Median score difference 0.028, 1 decision flip, median top-box IoU 0.89, no speed gain (5.6 vs 5.2 s per image) | `validation_quantization.csv` |
| Bootstrap clustered by photographer instead of image | Interval bounds move by at most 0.017 | `audit_cluster_bootstrap.csv` |
| Sensitivity to the 415 photographer-linked exclusions | Including them would change SigLIP AUROC by -0.04 to +0.04, raising it for 9 of 12 classes | `audit_exclusion_sensitivity.csv`, `audit_exclusion_shift.csv` |
| Model snapshots and environment | Recorded after the fact; revisions were not pinned when experiments ran | `run_metadata.json`, `requirements.lock.txt` |

## 4. Pre-registered test on fresh images

Rules were fixed in `configs/gate.yaml` before any fresh image was chosen. Cuts were frozen from dev data (`gate_selection.json`). Fresh images come from unused validation and test images whose photographers appear nowhere earlier; the near-duplicate check removed none.

Result: the SigLIP presence check significantly lowers false positives for Person, Shelf and Truck. For Person and Shelf it costs about 15 points of recall, beyond the pre-set limits (5 and 10). Truck meets the rule, but the detector still fires on 90% of its negatives. AP50 does not improve. Forklift had no unused negatives and was checked only on reused images, non-confirmatorily. Details: `RESULTS.md` Section 8, `gate_fresh_test.csv`, `gate_dev.csv`, `gate_reused_heldout_nonconfirmatory.csv`, `gate_fresh_manifest.csv`.

## 5. Not addressed

- Forklift localization still has no box labels, and forklift negatives are too few (14 held-out) for a confirmatory test.
- Label noise in verified positives was observed but not measured.
- Confidence intervals are not corrected for multiple comparisons outside the pre-registered test.
- LoRA adaptation, edge optimization, robustness testing and the safety assessor remain stubs.

## 6. How to reproduce

Use the project environment (`.venv`) and run from the project root, in this order. Times are estimates from the logged runs on an RTX 2070 SUPER and an 8-thread desktop CPU, with images already downloaded.

| Step | Command | Time |
| --- | --- | --- |
| 1 | `scripts/import_open_images.py` (downloads images, labels and held-out masks) | network-bound; skip if `data/` is present |
| 2 | `scripts/record_image_hashes.py --verify`; stop if any image differs. Fresh gate images show as missing until step 10 | 1 min |
| 3 | `scripts/run_poc.py` | ~2 h (detector over all images and six wordings) |
| 4 | `scripts/run_failure_analysis.py dev`, `scripts/run_mechanism_probe.py dev`, `scripts/render_failure_examples.py` | ~30 min for steps 4 and 5 together (rescoring and masks no longer run here) |
| 5 | `scripts/run_failure_analysis.py heldout`, `scripts/run_mechanism_probe.py heldout` | |
| 6 | `scripts/build_report_assets.py` (copies outputs that steps 7 and 8 read) | ~15 min |
| 7 | `scripts/audit_outputs.py` | a few min |
| 8 | `scripts/run_validation.py` | ~1.5–2 h |
| 9 | `scripts/build_report_assets.py` again, to pick up corrected tables | ~15 min |
| 10 | `scripts/run_gate_test.py`, then `scripts/run_quantization_gate_check.py` | ~30 min |
| 11 | `scripts/run_benchmark.py`, then `scripts/profile_precision.py`, **with nothing else running on the machine** | ~1 h |
| 12 | `scripts/compute_metrics.py`, `scripts/compute_slices.py`, `scripts/compute_safety_metrics.py` | ~5 min |
| 13 | `scripts/run_robustness.py` (`--cached-only` reports cached conditions without the GPU) | ~45 min |
| 14 | `scripts/demo.py --examples`, then `scripts/build_notebooks.py` | a few min |
| 15 | `scripts/record_image_hashes.py --verify` again, now including the fresh images | 1 min |

**Total: about 7.5 to 9.5 hours of compute** with images present, almost all of it GPU-bound and sequential. A clean reproduction must start from an empty `artifacts/` directory, because every stage reuses its cache when one exists. Two cautions: step 3 re-freezes `selection.json`, which must match the committed one; and step 10 re-selects the same fresh images by seed, which reproduces the gate test but is not a second confirmatory test.

Tests: `python -m pytest`.

## 7. Second review: MLOps audit (2026-11-02)

A second pass over all code, outputs and assumptions, covering errors, unsupported claims, quantization, reproducibility, leakage and missing validation.

### Verified correct

| Item | How it was checked |
| --- | --- |
| AP50 computation | Read against the Open Images rules: greedy matching by score, duplicates count as false positives, group-of boxes judged by intersection over the detection's area, all-point interpolation. Covered by `tests/test_metrics.py` |
| Ground truth export | Every held-out box for the five box classes in `ground_truth.json` (exported through FiftyOne) was compared with the raw Open Images `detections.csv`: 2,941 boxes in both, no image-class pair differs, no image missing |
| Dev threshold selection | The F1 expression equals 2TP / (2TP + FP + FN) |
| SigLIP text encoding | Pads to 64 tokens, as SigLIP was trained; the common default-padding mistake is absent |
| Fresh-process reproducibility | 18 saved examples reproduce to within 0.00005, both before and after pinning model revisions |
| Held-out gating | Selection freezes once, held-out scoring is refused before it, and a config hash is checked |

### Changes made

| Change | Reason | Files |
| --- | --- | --- |
| Pinned every model to its exact Hugging Face commit; loading an unpinned model now fails | `from_pretrained` without a revision follows the latest Hub commit, so a model update would silently change every result | `configs/revisions.yaml`, `config.py`, `vision_language.py`, `grounded_sam.py` |
| Added a settings check to the SigLIP score cache, as the detector cache already had | `siglip_scores.csv` was reused without checking which model or prompts produced it | `experiments.py` |
| Recorded SHA-256 for all 5,348 evaluated images (3,682 main, 1,666 fresh), with a verify mode | Image IDs alone do not pin the data; a re-download can change or fail | `scripts/record_image_hashes.py`, `docs/results/image_sha256.csv` |
| Measured SigLIP precision drift at the frozen presence-check cuts | All SigLIP results ran under fp16 autocast and this was never checked; the presence check thresholds raw logits | `scripts/run_quantization_gate_check.py`, `validation_siglip_precision_gate.csv` |
| Tests for the revision pins and the SigLIP cache check (25 tests pass) | New behavior | `tests/test_leakage.py` |
| Made the decision-based acceptance rule for quantized or exported models binding | AUROC alone had passed a model that changes 1 in 10 decisions | `docs/DESIGN_DECISIONS.md` |
| Corrected the reproduction order in the README and in Section 6, and put the image-hash check before any run | The README line was out of date, and the earlier Section 6 left out the import, the probes and the second asset build | `README.md`, `docs/REVIEW.md` |
| `RESULTS.md`: added intervals to the 14-negative headline rates, a quantization limitation, the gate sampling deviation, and the photographer-shift observation for Ladder | Unqualified or missing claims | `docs/RESULTS.md` |
| `DESIGN_DECISIONS.md`: added revision and image pinning, and three leakage rows | Documented vectors were incomplete | `docs/DESIGN_DECISIONS.md` |

### Findings

**Quantization (most important).** Dynamic int8 SigLIP keeps AUROC (forklift 0.959 vs 0.960, Shelf 0.825 vs 0.846) but moves logits by a median of 0.9 to 1.3 and up to 9.3. At the frozen presence-check cuts, that flips 9.3% of forklift, 8.6% of truck and 10.9% of shelf decisions on dev images. AUROC does not change when scores shift, so it is the wrong acceptance metric for a thresholded system. fp16 autocast, used for all results, is benign: at most 0.034 logits and 0.5% to 0.7% flips. The int8 detector check is too small to rely on (40 images, one prompt, top score only); it showed one decision flip and a median top-box IoU of 0.89, and gave only about a 7% speed-up because dynamic quantization covers only linear layers. SAM was never checked. This is now a binding acceptance rule in `DESIGN_DECISIONS.md`: thresholds are re-derived on dev data after quantization, and a model is accepted on decision agreement and recall at the operating point, not AUROC. Use static calibration on dev data only, as the `edge/quantize.py` stub already requires.

**Pre-registration deviation.** `configs/gate.yaml` states a cap of 300 per class and label, but sampled images keep all their labels, so Person has 545 fresh positives. Each class's sample is therefore partly chosen through the other classes. This does not invalidate the test, because the selection never looked at model outputs, but it is now disclosed in `RESULTS.md`.

**Statistical weaknesses (documented, not fixed).**
- AP50 intervals use 300 bootstrap samples. At the gate test's Bonferroni level (alpha 0.0167), each tail rests on about 2.5 samples, so those bounds carry Monte Carlo error. Use at least 2,000 samples for any future confirmatory AP interval.
- Every bootstrap call uses seed 42, so intervals across classes and conditions share resampling noise. Harmless for a single interval, but the intervals are correlated.
- The gate cuts sit at the 5th percentile of dev positive logits. On fresh images recall fell by about 15 points for Person and Shelf even though Person had 520 dev positives, so the cause is a shift between dev and fresh data, not small samples. The Ladder exclusion result points the same way.
- Precision values depend on the artificial share of positives, as Section 1 of `RESULTS.md` already warns.

**Reproducibility gaps (documented, not fixed).**
- The selection hash covers `poc.yaml` and `open_images.yaml` but not `base.yaml`, which holds the seed. Widening it now would invalidate the frozen selection.
- Batched detector scores depend on batch composition (correlation 0.996, differences up to 0.05). Batch size is checked, but batch order follows manifest order and is not otherwise recorded.
- No deterministic-algorithms flag is set for CUDA. The measured repeat drift is negligible.
- Run outputs are CSVs; MLflow is configured but unused, and there is no data versioning beyond the new hash snapshot.

**Leakage vectors (reviewed).** Photographer-based exclusion removed all 415 links found; there were no exact or near duplicates. Remaining vectors are now listed in `DESIGN_DECISIONS.md`: pretraining overlap (unmeasurable), held-out clustering by photographer (intervals moved by at most 0.017), review decisions made with held-out results in view, and the limited hash scope. Images with no recorded photographer cannot be matched on photographer.

**Missing validation (recommended).**
1. Score calibration on dev data (reliability curves) before any absolute threshold is used.
2. A recall-at-operating-point acceptance test for any quantized or exported model, on dev data.
3. Latency and memory on the target edge hardware; the only timings are CPU timings on a desktop.
4. Robustness under blur, low light and occlusion; still a stub.
5. Unit tests for `evaluate_masks` and `evaluate_rescoring`, which have none.
6. A warehouse-representative test set with forklift box labels. This is the largest gap, and the reason no forklift localization or safety claim can be made.

## 8. Third review: new metrics, benchmark, robustness and the evaluation document (2026-11-02)

A review of everything added in this round: `docs/EVALUATION.md`, the IoU, CHAIR, POPE and ECE code, the safety metrics, the benchmark and profiler, the corruption sweep, and the safety requirements.

### Errors found and fixed

| Issue | Effect | Fix |
| --- | --- | --- |
| Person localization credited each person with their best-overlapping detection, without one-to-one matching | One box over two people counted as finding both: 42 of 939 held-out matches were duplicates. Held-out individual miss rate was understated (32.5%, now **35.4%**), fresh likewise (22.2%, now 25.9%), and recall for large people overstated (97%, now 92%) | `localization_iou` now matches greedily one-to-one; test added; metrics and safety tables regenerated |
| The corruption sweep had started with the old matching code and saved only summary numbers | Its instance recall would have been inflated and could not be corrected afterwards | Sweep stopped at 12 of 41 conditions and restarted with the fix; it now caches raw detections per condition (`artifacts/poc/robustness/`), so metrics can be recomputed without the GPU. Test added |
| `EVALUATION.md` said the corruption sweep had completed | False status claim | Corrected |
| `EVALUATION.md` said per-class rates transfer to a real feed | Per-class ECE depends on the sampled share of positives just as precision does | Reworded: recall, miss and false positive rates transfer; precision and calibration (pooled or per class) do not. Same qualification added to the safety requirement on ECE |
| Shifts in Section 2 were described as ordered "from smallest to largest effect" | They use different metrics and cannot be ranked | Ordering claim removed |
| The rule-of-three sample size (about 300 people) assumed independent people | People in one frame, shift or site are correlated | Caveat added: plan for several times that, spread across sites |
| New scripts imported each other only when run with `PYTHONPATH=scripts` | A reproduction run from another working directory would fail | Scripts now add their own directory to the import path |

### Disclosed rather than fixed

| Issue | Why it matters | Where disclosed |
| --- | --- | --- |
| The benchmark ran each variant as one block in a fixed order, and some GPU rows overlapped CPU-heavy analysis scripts | Benchmark latencies are indicative only; this is part of why fp16 looked slower | `EVALUATION.md` Section 6; the interleaved profiler, run alone, supersedes them |
| Profiler kernel-time totals for Grounding DINO exceed wall time | That column is unreliable for the detector | Not used; noted in Section 6 |
| One held-out image holds 103 people against the detector's 50-detection cap | That image alone is 7.4% of held-out person instances | Sensitivity rows in `safety_person_miss.csv`; without it the miss rate is 32.1% |
| Top-1 accuracy is defined on images with exactly one verified-positive class; other classes may be present but unverified | Absolute top-1 is not meaningful; only variant comparison is | `EVALUATION.md` Section 6 |
| CHAIR and POPE are adapted from captioning and VQA | Not comparable to published caption-model numbers | `evaluation/hallucination.py` docstring, `EVALUATION.md` Section 5 |
| "Confident" levels 0.5 and 0.7 are conventional fixed values, set in code before the rates were computed, not tuned | A different level gives a different rate; the full table has all three | `safety_confident_errors.csv` |
| "A fallen person/box" queries assume no fallen object in ordinary images | State absence is unverified | Risk register row 7 |
| Corruptions are synthetic, with severities chosen by the reviewer rather than calibrated to real cameras; motion blur is horizontal only; occlusion boxes are placed at random, not over people | Measures sensitivity, not field behavior; severities are not comparable across corruption types | `robustness/corruptions.py` docstring, `EVALUATION.md` Section 2 |
| The fresh gate images were reused for descriptive safety metrics after the one confirmatory test | Nothing was tuned on them, but they are no longer untouched; any future confirmatory test needs new images | This log |
| Held-out data has now been used for many analyses with no correction for multiple comparisons | Some differences will look significant by chance | `RESULTS.md` Limitations |
| No clean end-to-end rerun has been done; the main model caches predate the review's code changes | The spot check (18 examples within 0.00005) is not a full regeneration | `EVALUATION.md` Section 0; Section 6 above gives the procedure and time |

### Added in this round (documented here for completeness)

`docs/EVALUATION.md`; `evaluation/hallucination.py` (CHAIR, POPE), `evaluation/calibration.py` (ECE, Brier); `localization_iou`; `robustness/corruptions.py` and `robustness/runner.py` (previously stubs); scripts `compute_metrics.py`, `compute_slices.py`, `compute_safety_metrics.py`, `run_benchmark.py`, `profile_precision.py`, `run_robustness.py`; the Safety requirements and quantization acceptance sections of `DESIGN_DECISIONS.md`; 22 new tests (47 in total).

## 9. Fourth review: warehouse inspection scope, demo, and removal of superseded logic (2026-11-02)

The deliverable was aligned to the project scenario (an open-vocabulary inspection proof of concept: text prompt in, boxes and masks out, at least 20 images across at least 3 categories, a measured hallucination or confident false positive, and a production readiness table), kept in warehouse scope with the existing Open Images V7 data. This review covers every change since Section 8 and checks that no code path still uses logic the earlier reviews ruled out.

### Superseded logic removed from active code

| Logic | Why it must not be used | Change |
| --- | --- | --- |
| Mask evaluation paired each ground-truth instance with its best-overlapping detection | Same flaw as the box matching fixed in Section 8: one detection could be credited to several objects | `evaluate_masks` now matches one-to-one per image and class through a shared `match_one_to_one`; masks regenerated (`scripts/run_masks.py`). Box detected-mask IoU 0.678 → 0.670, unmatched boxes 23.6% → 25.7%; Person and Truck unchanged |
| SigLIP yes/no decisions at a fixed probability of 0.5 in CHAIR and POPE | Violates Safety requirement 5 | SigLIP removed from CHAIR and POPE; it appears only in per-class ECE and threshold-free AUROC. Requirement 5 now cites per-class ECE instead of the removed POPE numbers |
| `run_failure_analysis.py` still wrote rescoring and mask tables | These were the stale copies that nearly overwrote corrected results; nothing read them any more | Removed from the script; old files moved to `artifacts/poc/failure/superseded/` |
| A non-warehouse prompt ("a giraffe") in the confident-error table | Prompts must match project scope | Removed; only warehouse prompts remain in the safety metrics and demo |
| `metrics_iou.csv` column `mean_best_iou` | Name described the old matching | Renamed `mean_matched_iou` |
| COCO: a benchmark citation, metric naming, and an unused `pycocotools` dependency | The project uses Open Images V7 only | All removed |

Retained on purpose: the research scripts behind `RESULTS.md` failure mode 1 (`run_mechanism_probe.py`, `render_failure_examples.py`, the control nouns in `run_validation.py`) still use non-warehouse nouns, because they test what the detector does with an object that cannot be present. Their results stay in `RESULTS.md` as research evidence; the decision documents (`EVALUATION.md`, the demo) use warehouse prompts only.

### Errors found and fixed in this round

| Issue | Fix |
| --- | --- |
| The demo's first "hallucination" example ("hard hat") was a novelty helmet: Open Images verifies it is not a hard hat, but it looks like one | Not a hallucination. Replaced |
| The replacement image (worker on a walkway) returned **no** detection for the bare prompt "forklift"; the earlier documented failure there used "a photo of a forklift" | Labeling it a hallucination would have been false. Replaced with an image drawn by seed from the 14 verified forklift-free images |
| That seeded image shows a backhoe loader | Relabeled honestly as a lookalike confident false positive, not a hallucination. The figure now shows three confident false positives (0.72, 0.56, 0.76), all labeled accurately |
| The overall readiness verdict claimed every use case had a false positive rate above 75% or no measurable accuracy | False for people (22%) and, across all 12 classes, ladders (48%). Narrowed to the four categories where it holds, with the exceptions named |
| `EVALUATION.md` status said the combined pipeline latency was unmeasured | Now measured through the demo: 0.58 to 1.06 s per image and prompt on the GPU (5 calls; corrected in Section 11 to 0.52 to 1.00 s, measured with the GPU otherwise idle) |

### Weaknesses disclosed rather than fixed

| Issue | Where |
| --- | --- |
| No non-maximum suppression after Grounding DINO (neither the Hugging Face processor nor this project applies one), so overlapping duplicate boxes are possible; AP50 counts them as false positives, and the demo shows them raw | This log; a production pipeline would need it, validated under the acceptance rules |
| The 0.32 threshold was chosen for single-noun prompts; it is unvalidated for longer prompts such as "a damaged box" | `EVALUATION.md` Demo section, `scripts/demo.py` docstring |
| The corruption sweep stopped after 11 of 41 conditions (clean, low light 1–5, glare 1–5, motion blur 1) to save time | `EVALUATION.md` Section 2; resumable from `artifacts/poc/robustness/` |
| Demo latency comes from 5 calls on one machine, including plotting-free pipeline time only | `EVALUATION.md` |
| The forklift row of the readiness table rests on 39 images | Stated in the table |
| Demo example images are held-out images; four were drawn by seed, the fire truck was chosen because `RESULTS.md` already discusses it | Figure caption |
| A clean end-to-end rerun has still not been done | `EVALUATION.md` Section 0 |

### Added

`scripts/demo.py` (prompt in; JSON boxes and scores, PNG masks and overlay out; `--examples` builds `docs/figures/demo_examples.png`), `scripts/run_masks.py`, `match_one_to_one`, the `--cached-only` mode of the corruption sweep, the readiness assessment, scenario checklist, architecture diagram and demo section in `EVALUATION.md`, and 3 new tests (50 in total).

## 10. Deployment readiness report and verification run (2026-11-02)

**Added.** `docs/DEPLOYMENT_READINESS.md`: PASS/FAIL criteria (**FAIL**, 9 of 11), go/no-go (**NO-GO**), a one-sentence safety rationale, accuracy, robustness by shift type, latency, model size (SAM measured: 375.1 MB fp32, 187.6 MB fp16; full pipeline 1,877 MB fp32, 939 MB fp16), tradeoff analysis, the evidence for the decision and the evidence that would reverse it, and a reproducibility guide. `scripts/verify_caches.py` (seeded 10% model rerun with fixed tolerances; `--file-order` option). One more robustness condition (motion blur 2) was cached before the sweep was stopped as out of scope.

**Verification result.** Recomputing the tables derived from the cached model outputs reproduced all of them exactly (corrected in Section 11: this first read "every committed table"); SigLIP reproduced exactly; the pre-registered detector check **failed** (94.1% of 3,401 sampled scores within 0.001, against 99% required; 2 flipped decisions). The cause, found afterwards: Aisle and Box, the first two classes scored, were batched with 103 images that were later excluded for leakage. In their original batches they reproduce exactly (546 of 546 pairs). No metric is affected by leakage, because excluded images' predictions are dropped, but a clean regeneration would not reproduce those two classes' scores exactly.

**Errors caught while writing the report.**

| Issue | Fix |
| --- | --- |
| Decision header counted 9 failing criteria when the table showed 8 (before the verification result) | Corrected; after the verification the count is 9 of 11, with reproducibility failing |
| The guide's install command omitted PyTorch's package index, so the pinned `torch==2.14.1+cu126` could not install | `--extra-index-url https://download.pytorch.org/whl/cu126` added |
| The guide said the hash check passes on a fresh machine; it exits with an error until the fresh gate images exist | Described accurately |
| The guide's clone step relied on a branch that has not been pushed | Stated in the step |

**Open.** Exact clean reproduction for Aisle and Box requires regenerating their detector outputs and re-baselining (Section 7.1 of the report). The bootstrap summary step takes ~50 minutes single-threaded; parallelizing it would cut verification time without changing results.

## 11. Fifth review: verification claims, fp16 criteria and the reproducibility guide (2026-11-02)

### Corrections

| # | Error or unsupported claim | Evidence | Correction |
| --- | --- | --- | --- |
| 1 | "Every committed table recomputed exactly" (report, `EVALUATION.md`, Section 10, and commit `fefe277`'s message) | File timestamps: 18 result files were never regenerated (gate test, validation checks, rescoring, masks, benchmark, profiler), and 5 were only copied from the probe cache | Every statement now names what was verified (tables derived from cached model outputs) and what was not. The commit message cannot be changed; this entry corrects it |
| 2 | Criterion 9 (compression agreement) marked PASS for fp16, and criterion 8 (memory) PASS on that basis | The binding protocol requires per-class decision agreement at the operating point, recall loss, box and mask IoU and ≥ 100 images per class per label; only top-1 agreement on 200 images and partial presence-check flips exist | Both marked **not demonstrated**; assessment reads 9 of 11 FAIL, 2 not demonstrated. Decision unchanged |
| 3 | Aisle/Box impact given as "about 1 in 1,700 decisions" | That averages over all classes; within Aisle and Box it is 2 of 504 sampled decisions | Now "about 1 in 250 of their decisions" |
| 4 | Demo latency 0.58–1.06 s | Measured while the corruption sweep shared the GPU; the verification's demo step, with the GPU otherwise idle, gave 0.52–1.00 s (first call 3.06 s) | Clean figures used; 5 calls noted as too few for a p95 |
| 5 | "Excluded … before any results were read" | Exclusion list written at 17:59 on 2026-11-01, after SigLIP scores (17:13) and after the detector run began | Reworded: the rule is metadata-only, so results could not influence it, but whether outputs were viewed first cannot be confirmed |
| 6 | Offline pilot presented as "permissible" or "defensible" | Its usefulness was never measured | Described as a candidate use needing its own evaluation |

### Reproducibility fixes

| Issue | Fix |
| --- | --- |
| `import_open_images.py` re-drew the sample, checking which development images the bucket still holds, so a new machine could get a different image set | It now reuses the committed `docs/results/manifest.csv` (verified byte-identical to the manifest the results used); `--resample` draws a new sample and says the results will not match |
| The guide recommended a verification run that needs `artifacts/` caches another engineer does not have | The guide now says so, gives the two options (obtain the caches, or run the full regeneration), and states the expected Aisle and Box differences under a full regeneration |

### Still open

Reproduce the model-running steps listed above; run the fp16 acceptance protocol if criteria 8 and 9 are to stand; regenerate Aisle and Box for exact clean reproduction; push the branch; add continuous integration.

## 12. Reproducibility guide points at the evaluated commit (2026-11-02)

The guide's first step checked out the `review/zero-shot-poc` branch, which exists only locally; the work is to be published on `main` instead. The step now checks out commit `e860090`, the commit the results and corrections were finalized in, so `main` can move on without changing what another engineer reproduces. Commits after `e860090` change documentation only.
