# Design Decisions

Each entry gives the choice, the reason, and what it costs. Results are in [RESULTS.md](RESULTS.md).

## Problem and scope

**Choice.** Zero-shot detection and segmentation for warehouse forklift perception, with prompt wording as the experimental variable. LoRA, edge deployment, robustness, and safety assessment stay as documented stubs.

**Why.** Warehouse autonomy is a realistic target domain, and open-vocabulary models promise to avoid labeling every object type. Whether they deliver on that can be tested without training anything. An honest, narrow proof of concept was preferred over shallow coverage of every stage.

**Cost.** There is no adapted model and no safety score. The project shows where zero-shot prompting fails, not a deployable fix.

## Architecture

| Component | Choice | Why | Alternative rejected |
| --- | --- | --- | --- |
| Image-text encoder | `google/siglip-base-patch16-224` | Sigmoid scoring gives each prompt an independent score, which suits a multi-label scene better than softmax over a fixed prompt set | CLIP: softmax scores depend on which other prompts are present, confounding the prompt experiment |
| Detector | `IDEA-Research/grounding-dino-tiny` | Standard open-vocabulary detector with per-token scores, which made the mechanism directly measurable | Larger variants: harder to fit in 8 GB and less relevant to an edge target |
| Segmenter | `facebook/sam-vit-base` | Takes boxes as prompts, and can be tested separately using ground-truth boxes | Larger SAM models: slower for a comparison that needs thousands of masks |
| Composition | Detect, then segment, then optional SigLIP rescoring | Each stage can be measured alone, so errors can be attributed | End-to-end model: errors could not be separated by stage |

Library versions are recorded in the environment (`torch` 2.14.1, `transformers` 5.20.2) and models run on an RTX 2070 SUPER whenever possible.

**Input size.** The detector runs at shortest edge 480, longest 800, with batch size 4. This was needed for throughput on 8 GB across about 38,000 image-prompt pairs. It lowers scores relative to the published 800×1333 setting, so published numbers are context only.

## Dataset

**Choice.** Open Images V7 imported with FiftyOne, using human-verified image-level labels to select images.

**Why.** It has verified positive *and* negative labels, which are needed to measure false detections. It also includes boxes, some masks, and photographer metadata for the leakage checks. FiftyOne's class filter only accepts the 600 boxable classes and silently drops others, so image IDs were chosen from the official label files for classes such as Forklift truck.

**Cost.** It is not warehouse footage. Fallen package, damaged carton, and blocked aisle do not exist as classes, so they are used only as hallucination probes with no accuracy claim. Forklift has image-level labels only, and Pallet has no held-out labels.

**Sizes.** Selection caps were 60 positives and 60 negatives per class for dev, and 150 each for held-out, limited by what exists. Evaluated counts can exceed a cap because an image selected for one class may also be verified for another (for example, 520 dev Person positives).

## Splits

**Choice.** Open Images train is dev, used for all choices. Validation and test together are held-out, used only after freezing. There is no training, so dev exists only to select prompts and a threshold.

**Why.** Rare classes need every available held-out label. Merging validation and test is acceptable because nothing is tuned on either.

## Loss

**This project trains nothing, so it has no loss of its own.** The pretrained losses still matter, because they explain the behavior:

- **SigLIP** uses pairwise sigmoid loss: every image-text pair is an independent binary decision. This is why its scores don't depend on other prompts. Its bias term also explains why raw probabilities are often near zero, for example about 0.000 for "a person" on positive images. Raw probabilities are therefore not used as confidence.
- **Grounding DINO** uses contrastive token-to-region alignment with focal loss, plus box losses. This creates the per-token scores analyzed in the failure analysis.

If the LoRA stage is built, it should use the SigLIP pairwise loss with added hard negatives (negated and state-swapped captions), training on dev data only.

## Evaluation metrics

| Metric | Used for | Why | Rejected |
| --- | --- | --- | --- |
| AUROC | Image-level presence | Does not depend on the artificial positive rate of this dataset, or on a threshold | Accuracy and raw precision: both depend on that artificial rate |
| AP50, Open Images style | Boxable classes | Standard, with correct handling of group-of boxes | COCO AP: needs complete labels. Verified-negative labeling here does not support it, and localization tightness matters less than presence for this question |
| Mask IoU with a ground-truth-box upper bound | SAM | Separates SAM error from detector error | Mask IoU alone: errors could not be attributed |
| False detection rate on verified negatives | Hallucination | Answers the safety question directly | |
| Bootstrap CI, paired across conditions | All comparisons | Small classes need uncertainty, and pairing removes image-difficulty variance | Point estimates only |

The threshold, 0.32, maximizes pooled dev F1. This is simple and leak-free but too permissive, as the false positive rates show. Per-class thresholds were checked afterwards and barely help, and a pre-registered SigLIP presence check was not adopted (RESULTS Sections 3 and 8).

## Prompt conditions

| Condition | Forklift example |
| --- | --- |
| `bare` | forklift |
| `article` (baseline) | a forklift |
| `photo` | a photo of a forklift |
| `context` | a forklift in a warehouse |
| `synonym` | a lift truck |
| `descriptive` | an industrial vehicle with two metal forks for lifting pallets |
| `ensemble` (SigLIP only) | Mean across the other conditions |

Wording varies in a fixed template across all classes, so differences can be attributed to wording rather than to one lucky phrase. `article` is the baseline because it is what a person would most plausibly type.

## Data leakage review

| Vector | Status | Action |
| --- | --- | --- |
| Dev and held-out summaries computed together | Fixed | A gate blocks held-out summaries until `selection.json` is frozen, with a config hash recorded |
| Same photographer across dev and held-out | Fixed | 415 held-out images excluded by Flickr profile URL before any results were read |
| Tuning on validation (stub text and notebooks) | Fixed | Changed to dev |
| Stale cached predictions and default-negative labels | Fixed | `restrict_to_manifest` limits predictions to verified, current images |
| Shared IDs, exact duplicates, dHash near-duplicates | Checked | None found across dev and held-out, with tests in `tests/test_leakage.py` |
| Batch composition affecting scores | Measured, retained | On 100 dev images, batch and single-image scores correlate 0.996, with median difference 0.006 and maximum 0.05. Batches follow image ID rather than labels, so this adds noise rather than leaking labels, but it can flip decisions near the threshold |
| Pretraining overlap (SigLIP on WebLI; Grounding DINO on O365, GoldG, and Cap4M) | Not measurable | Evaluation images might appear in pretraining data, and Flickr-sourced data is part of GoldG. Results could be optimistic |
| Same photographer within held-out | Retained | Inflates confidence only slightly, by making images less independent |
| Failure analysis on held-out data | Controlled | Probes were written and examined on dev, then run on held-out without changes. The attribution write-up was finalized after seeing both |

## Tools

FiftyOne for dataset handling, PyTorch and Hugging Face Transformers for models, pandas and scikit-learn for analysis, pytest for metric and leakage tests, and notebooks as thin drivers over `src/`. MLflow was scaffolded but not used, because the runs are deterministic scripts with outputs saved as CSV in `docs/results/`.

## Image attribution

Figures contain Open Images photographs from Flickr, which were released under their photographers' licenses. Check attribution using the image IDs before publishing the figures outside coursework.
