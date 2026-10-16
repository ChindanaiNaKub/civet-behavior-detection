# Phase 2 Plan — Short-Clip Civet Behavior Classification

**Target:** October 31, 2026

**Team capacity:** 3 people, about 10 hours per person per week

**Goal:** Deliver a reproducible short-clip behavior-classification baseline, a held-out evaluation, and a notebook or script demo that shows time-coded predictions on a timeline.

## Scope

Adapt the Phase 2 image-classification task to classify five-second video windows. A single image cannot establish movement or duration, which are needed for the agreed Locomotion and Resting definitions. This plan does not include YOLO bounding-box detection or pose/keypoint analysis. See [ADR 0001](adr/0001-classify-short-behavior-clips.md).

The four behavior labels are Eating, Locomotion, Resting, and Climbing. Resting requires at least five seconds of continuous inactivity; when that duration is confirmed, annotate from the inactivity onset. Split annotations at behavior changes. Exclude windows crossing transitions. For overlapping actions, assign the most specific visible behavior; use Uncertain if none is clear. Use Uncertain for low-confidence model predictions as well.

Start with a compact pretrained video classifier. Choose the model size after checking the dataset and available compute. Do not train from scratch unless the audit shows enough labeled clips to support it.

Pre-label intervals automatically rather than hand-labeling every clip: derive Resting from sustained low motion (at least five seconds) and Locomotion from location change, and use a compact zero-shot pretrained video model to propose Eating and Climbing intervals. A human verifies and corrects the proposed timeline. Auto labels are weak supervision only and never enter the reported evaluation; a small, fully human-labeled gold set of whole source clips is the only data used for reported model quality. See [ADR 0003](adr/0003-automate-prelabeling-with-human-gold-set.md).

See the [footage and compute audit](FOOTAGE-AUDIT.md) for the source inventory, metadata limits, class-count targets, and local hardware check. Use the [behavior interval annotation guide](ANNOTATION-GUIDE.md) for operational label rules and review procedure.

## Work plan

| Dates | Work | Exit criteria |
|---|---|---|
| **Oct 7–11: Audit and protocol** | Inventory the 22 supplied clips (5 Eating, 10 Locomotion, 7 Resting); treat folder names as weak labels only. Find additional clips across all four behaviors, prioritizing Climbing. Record source clip, duration, available session/camera/room metadata, lighting, visibility, and occlusion. Confirm compute and use the annotation guide. Make a clip-level train/validation/test split plan before extracting windows. | [Footage audit](FOOTAGE-AUDIT.md) and [annotation guide](ANNOTATION-GUIDE.md) are complete; metadata gaps and data targets are explicit; a group-aware split is feasible or its limitations are recorded. |
| **Oct 12–18: Collect, pre-label, and verify** | Gather more examples, prioritizing missing and underrepresented classes. Run the automated pre-labeler to propose behavior intervals and derive Resting/Locomotion from motion; a human verifies and corrects the timeline. Two people independently label the gold sample and reconcile disagreements; the gold set stays fully human. Mark transitions and uncertain intervals explicitly. Lock the held-out source clips before making windows. | Every included window has one verified label and a source timecode; a fully human gold set is fixed; held-out clips do not appear in training or validation; per-class clip and window counts are known. |
| **Oct 19–25: Train and evaluate** | Extract five-second windows from the split clips; exclude transition-crossing windows. Use a compact pretrained video classifier, with model size set by the compute audit. Train on the verified training set (auto pre-labels only as reduced-weight weak supervision) and choose settings and the low-confidence threshold on validation data only. Evaluate the locked test set once. | Reproducible training and inference; held-out macro F1, per-class precision/recall/F1, confusion matrix, and Uncertain/coverage rate are recorded. Any class missing adequate test examples is called out rather than hidden in an aggregate score. |
| **Oct 26–31: Demo and report** | Build the notebook or script that accepts a short clip, predicts a label and confidence for each five-second window, and displays a time-coded timeline. Review errors by class and visibility conditions. Document the dataset, split, annotation agreement, model, metric results, limitations, and next steps. Re-run the demo from the documented setup. | The demo runs reproducibly on a clip outside training data; predictions link to original time ranges; the report distinguishes measured results from planned work and states data/evaluation limits. |

## Team workstreams

Assign people by availability and experience; roles can overlap, but keep the test set locked by someone who is not tuning the model.

- **Data and split lead:** inventories source clips, gathers metadata and new footage, tracks class balance, and creates the source-clip/session split before window extraction.
- **Annotation lead:** maintains the label guide, oversees human verification of machine pre-labels, curates the human gold set, reconciles disagreements, and checks segment timestamps.
- **Model and demo lead:** audits compute, trains the pretrained baseline, tunes only on training/validation data, evaluates the locked test set, and builds the timeline demo.

All three review the class definitions and final error analysis. The annotation and split leads should coordinate so uncertain or transition windows do not leak across splits.

## Data and evaluation rules

- The 22 supplied videos currently have folder-level labels, not verified behavior intervals. Verify intervals from the pre-labeler or label from scratch; do not treat every frame in a folder as ground truth.
- Automated pre-labels are weak supervision only: they may train the classifier with reduced weight, but reported metrics are computed only on the human gold set. Never score the model against its own pre-labels.
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
- **Automated pre-label quality unproven:** Resting and Climbing are the likely failure modes. Run one smoke test before committing the plan to the pre-labeler; fall back to manual labeling for any class the pre-labeler cannot handle.

## Decisions to make during the first audit

1. The audit sets an initial pilot target of 20 verified source clips per class, with five reserved for test; revise only after source and session availability are confirmed.
2. Model variant and compute setup, based on the available hardware and a short memory smoke run.
3. Window stride and whether windows overlap. Start with non-overlapping five-second windows; record timestamps so this can be changed without relabeling.
4. Any numeric performance target, after class counts and held-out test coverage are known.
5. Pre-labeler choice and the size and scope of the human gold set, after the pre-labeler smoke test.
