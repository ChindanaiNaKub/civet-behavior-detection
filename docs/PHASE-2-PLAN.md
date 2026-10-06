# Phase 2 Plan — Short-Clip Civet Behavior Classification

**Target:** October 31, 2026

**Team capacity:** 3 people, about 10 hours per person per week

**Goal:** Deliver a reproducible short-clip behavior-classification baseline, a held-out evaluation, and a notebook or script demo that shows time-coded predictions on a timeline.

## Scope

Adapt the Phase 2 image-classification task to classify five-second video windows. A single image cannot establish movement or duration, which are needed for the agreed Locomotion and Resting definitions. This plan does not include YOLO bounding-box detection or pose/keypoint analysis. See [ADR 0001](adr/0001-classify-short-behavior-clips.md).

The four behavior labels are Eating, Locomotion, Resting, and Climbing. Resting requires at least five seconds of continuous inactivity; when that duration is confirmed, annotate from the inactivity onset. Split annotations at behavior changes. Exclude windows crossing transitions. For overlapping actions, assign the most specific visible behavior; use Uncertain if none is clear. Use Uncertain for low-confidence model predictions as well.

Start with a compact pretrained video classifier. Choose the model size after checking the dataset and available compute. Do not train from scratch unless the audit shows enough labeled clips to support it.

## Work plan

| Dates | Work | Exit criteria |
|---|---|---|
| **Oct 7–11: Audit and protocol** | Inventory the 22 supplied clips (5 Eating, 10 Locomotion, 7 Resting); treat folder names as weak labels only. Find additional clips across all four behaviors. Record source clip, duration, available session/camera/room metadata, lighting, visibility, and occlusion. Confirm compute and set up the annotation guide. Make a clip-level train/validation/test split plan before extracting windows. | Inventory and data gaps are documented; annotation rules and metadata fields are ready; a group-aware split is feasible or its limitations are recorded; model size and data target are set from actual counts. |
| **Oct 12–18: Collect and label** | Gather more examples, prioritizing missing and underrepresented classes. Annotate behavior intervals with start/end times. Have two people independently label a 20% shared sample, reconcile disagreements, then complete the remaining labels. Mark transitions and uncertain intervals explicitly. Lock the held-out source clips before making windows. | Every included window has one reviewed label and a source timecode; held-out clips do not appear in training or validation; per-class clip and window counts are known. |
| **Oct 19–25: Train and evaluate** | Extract five-second windows from the split clips; exclude transition-crossing windows. Use a compact pretrained video classifier, with model size set by the compute audit. Train on the training set and choose settings and the low-confidence threshold on validation data only. Evaluate the locked test set once. | Reproducible training and inference; held-out macro F1, per-class precision/recall/F1, confusion matrix, and Uncertain/coverage rate are recorded. Any class missing adequate test examples is called out rather than hidden in an aggregate score. |
| **Oct 26–31: Demo and report** | Build the notebook or script that accepts a short clip, predicts a label and confidence for each five-second window, and displays a time-coded timeline. Review errors by class and visibility conditions. Document the dataset, split, annotation agreement, model, metric results, limitations, and next steps. Re-run the demo from the documented setup. | The demo runs reproducibly on a clip outside training data; predictions link to original time ranges; the report distinguishes measured results from planned work and states data/evaluation limits. |

## Team workstreams

Assign people by availability and experience; roles can overlap, but keep the test set locked by someone who is not tuning the model.

- **Data and split lead:** inventories source clips, gathers metadata and new footage, tracks class balance, and creates the source-clip/session split before window extraction.
- **Annotation lead:** maintains the label guide, organizes two-person review of the shared sample, reconciles disagreements, and checks segment timestamps.
- **Model and demo lead:** audits compute, trains the pretrained baseline, tunes only on training/validation data, evaluates the locked test set, and builds the timeline demo.

All three review the class definitions and final error analysis. The annotation and split leads should coordinate so uncertain or transition windows do not leak across splits.

## Data and evaluation rules

- The 22 supplied videos currently have folder-level labels, not verified behavior intervals. Manually label time ranges; do not treat every frame in a folder as ground truth.
- Split by complete source clips before creating five-second windows. Group by recording session or camera where metadata permits. If source clips are too few to support a meaningful held-out set for every class, report that limitation and use group-aware cross-validation for development; do not present a weak test score as a reliable estimate.
- Primary metric: four-class macro F1. Also report per-class precision, recall, and F1, a confusion matrix with an Uncertain prediction column, and the share of windows returned as Uncertain. Count Uncertain outputs as abstentions/misses in the four-class metrics; do not silently discard them. Select the Uncertain threshold on validation data, never on the test set.
- Compare annotation agreement on the shared sample and record resolved disagreements. If agreement is poor for a class, clarify its examples before training.
- Report room, camera, lighting, and visibility only when the metadata or footage verifies them. Folder names and the proposed four-room/four-camera setup do not establish coverage.

## Risks and responses

- **Too few independent examples, especially for Climbing:** audit in the first work block, source examples across classes, and set realistic counts after seeing availability. If a class lacks test coverage, state that plainly.
- **Near-duplicate leakage:** split by original clip/session before window extraction and keep the test set locked.
- **Behavior ambiguity:** apply the shared definitions, remove transition windows, use Uncertain when evidence is insufficient, and review the shared annotation sample before full labeling.
- **Scope mismatch with the brief:** explicitly present this as an adaptation of the image-classification option, made because the agreed behavior classes depend on time. Confirm the adaptation with the professor/team if the assignment requires literal still-image inputs.
- **Overstated model performance:** report per-class results and Uncertain coverage; describe this as a baseline, not a validated production system.

## Decisions to make during the first audit

1. Final dataset size and examples needed per class, based on source availability and test coverage.
2. Model variant and compute setup, based on the available hardware.
3. Window stride and whether windows overlap. Start with non-overlapping five-second windows; record timestamps so this can be changed without relabeling.
4. Any numeric performance target, after class counts and held-out test coverage are known.
