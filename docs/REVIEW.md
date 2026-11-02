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

Use the project environment (`.venv`) and run from the project root, in this order:

1. `scripts/import_open_images.py` downloads the images and labels.
2. `scripts/record_image_hashes.py --verify` confirms the images match `docs/results/image_sha256.csv`. Stop if any differ. Fresh gate images show as missing until step 10 has downloaded them; run the check again after it.
3. `scripts/run_poc.py`
4. `scripts/run_failure_analysis.py dev`, `scripts/run_mechanism_probe.py dev`, `scripts/render_failure_examples.py`
5. `scripts/run_failure_analysis.py heldout`, `scripts/run_mechanism_probe.py heldout`
6. `scripts/build_report_assets.py` (copies outputs that steps 7 and 8 read)
7. `scripts/audit_outputs.py`
8. `scripts/run_validation.py` (`--skip-checks` reuses the repeat, control-noun and quantization outputs)
9. `scripts/build_report_assets.py` again, to pick up the corrected tables
10. `scripts/run_gate_test.py`, then `scripts/run_quantization_gate_check.py`
11. `scripts/build_notebooks.py`

Tests: `python -m pytest`. Do not rerun the gate test on new fresh images and report it as the same test; its cached outputs in `artifacts/poc/gate/` are the record of the one confirmatory run.

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
