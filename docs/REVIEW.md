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
| SigLIP fp32 vs dynamic int8 on CPU (151 images) | Dev forklift AUROC 0.959 vs 0.960 | `validation_quantization.csv` |
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

Use the project environment (`.venv`). Run in order: `scripts/run_poc.py`, `scripts/run_failure_analysis.py`, `scripts/audit_outputs.py`, `scripts/run_validation.py` (pass `--skip-checks` to reuse the repeat, control-noun and quantization outputs), `scripts/build_report_assets.py`, `scripts/run_gate_test.py`, `scripts/build_notebooks.py`. Tests: `python -m pytest`.
