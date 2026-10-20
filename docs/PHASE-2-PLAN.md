# Phase 2 Plan — Short-Clip Civet Behavior Classification

**Target:** October 31, 2026

**Team capacity:** 3 people, about 10 hours per person per week

**Goal:** Deliver a reproducible short-clip behavior-classification baseline, a held-out evaluation, and a notebook or script demo that shows time-coded predictions on a timeline.

## Scope

Adapt the Phase 2 image-classification task to classify five-second video windows. A single image cannot establish movement or duration, which are needed for the agreed Locomotion and Resting definitions. This plan does not include YOLO bounding-box detection or pose/keypoint analysis. See [ADR 0001](adr/0001-classify-short-behavior-clips.md).

The model classes are Eating, Locomotion and Resting. Climbing stays a valid annotation label but is out of scope as a model class ([ADR 0007](adr/0007-drop-climbing-from-phase-2-scope.md)). Resting requires at least five seconds of continuous inactivity; when that duration is confirmed, annotate from the inactivity onset. Split annotations at behavior changes. Exclude windows crossing transitions. For overlapping actions, assign the most specific visible behavior; use Uncertain if none is clear. Use Uncertain for low-confidence model predictions as well.

Start with a compact pretrained video classifier. Choose the model size after checking the dataset and available compute. Do not train from scratch unless the audit shows enough labeled clips to support it.

Deliver Task 1 of the brief as a CNN over the frames of each window: an ImageNet-pretrained ResNet-18 backbone, mean and max pooling over the window's eight frames, and only the pooling head trained on project data. It fits the 4 GiB GPU on the audit workstation. See [ADR 0004](adr/0004-train-a-cnn-over-window-frames.md). Do not train a video network from scratch on 22 clips.

The brief names the four classes Walking, Eating, Resting and Climbing. This repository labels the walking class Locomotion, because the Phase 1 glossary widened it to any movement that changes location, including travel along a structure. Report the class as Walking and give Locomotion as its operational definition, so the deliverable uses the brief's word.

Pre-label intervals automatically rather than hand-labeling every clip: derive Resting from sustained low motion (at least five seconds) and Locomotion from location change, and use a compact zero-shot pretrained video model to propose Eating and Climbing intervals. A human verifies and corrects the proposed timeline. Auto labels are weak supervision only and never enter the reported evaluation; a small, fully human-labeled gold set of whole source clips is the only data used for reported model quality. See [ADR 0003](adr/0003-automate-prelabeling-with-human-gold-set.md).

See the [footage and compute audit](FOOTAGE-AUDIT.md) for the source inventory, metadata limits, class-count targets, and local hardware check. Use the [behavior interval annotation guide](ANNOTATION-GUIDE.md) for operational label rules and review procedure.

## Where this plan stands

As of 2026-10-19. A checked box has its artifact in the repository.

- [x] **Audit and protocol.** [Footage audit](FOOTAGE-AUDIT.md) and [annotation guide](ANNOTATION-GUIDE.md) are complete. Metadata gaps are explicit: no session IDs, no room mapping, and camera and class are confounded.
- [x] **Pre-labeler and its smoke test.** `scripts/prelabel.py` proposes a timeline for all 22 clips in 2 min 41 s. Resting is reliable, Locomotion is a weak hint, and Eating and Climbing come out empty. See [the smoke test](PRELABEL-SMOKE-TEST.md).
- [x] **Pipeline for #5 and #6.** `scripts/windows.py` builds five-second windows from verified intervals, `scripts/split.py` writes the split ledger, `scripts/train.py` trains the frame CNN, `scripts/evaluate.py` scores the locked test set once, and `scripts/demo.py` prints a clip's predicted timeline.
- [x] **Review tooling for #4.** `scripts/review.py` serves each clip with its proposed intervals for a human to accept or correct, and `scripts/agreement.py` compares the two independent gold-set passes.
- [x] **Climbing scope decided.** The two candidate intervals were reviewed and are not climbing, so the class list stays at three. See [ADR 0007](adr/0007-drop-climbing-from-phase-2-scope.md) and [#16](https://github.com/ChindanaiNaKub/civet-behavior-detection/issues/16).
- [x] **Human verification of the timeline.** One annotator verified intervals for all 22 clips (62 intervals, `data/annotations.csv`). The first six gold clips were labeled blind.
- [x] **Split re-chosen on verified windows.** The first assignment left the test split with no Resting window. [ADR 0008](adr/0008-resplit-on-verified-windows-gold-is-test.md) re-chose it and makes the test split the eight-clip gold set (`data/gold-clips.txt`).
- [x] **The human gold set.** Two people labeled the eight gold clips independently (the first pass is in Git history, the second in `data/annotations-second.csv`). The passes were compared (Dice 0.597, `data/agreement.csv`) and reconciled with `scripts/reconcile.py`; the final intervals are in `data/annotations.csv` with annotator `reconciled`. The second pass was not fully independent of the first, and Thippharake, who made the first pass, delegated the per-clip reconciliation decisions to Gun, who made the second. Both are recorded in the ledger and in PR #28.
- [ ] **Lock the test set.** `scripts/split.py` passes on the final labels (test windows Eating 7 / Locomotion 2 / Resting 6). A person reviews `data/splits.csv` and fills `test_locked_by`. Do not run `scripts/evaluate.py` before then.
- [ ] **Train, evaluate, and report.** Run the pipeline on the verified windows and write the report from the numbers it produces.

The plan below is unchanged in its target date and its evaluation rules. What changed is the model family (ADR 0004), the split rule (ADR 0005) and the clip assignment (ADR 0008).

## Work plan

| Dates | Work | Exit criteria |
|---|---|---|
| **Oct 7–11: Audit and protocol** | Inventory the 22 supplied clips (5 Eating, 10 Locomotion, 7 Resting); treat folder names as weak labels only. Find additional clips across all four behaviors, prioritizing Climbing. Record source clip, duration, available session/camera/room metadata, lighting, visibility, and occlusion. Confirm compute and use the annotation guide. Make a clip-level train/validation/test split plan before extracting windows. | [Footage audit](FOOTAGE-AUDIT.md) and [annotation guide](ANNOTATION-GUIDE.md) are complete; metadata gaps and data targets are explicit; a group-aware split is feasible or its limitations are recorded. |
| **Oct 12–18: Collect, pre-label, and verify** | Gather more examples, prioritizing missing and underrepresented classes. Run the automated pre-labeler to propose behavior intervals and derive Resting/Locomotion from motion; a human verifies and corrects the timeline. Two people independently label the gold sample and reconcile disagreements; the gold set stays fully human. Mark transitions and uncertain intervals explicitly. Lock the held-out source clips before making windows. | Every included window has one verified label and a source timecode; a fully human gold set is fixed; held-out clips do not appear in training or validation; per-class clip and window counts are known. |
| **Oct 19–25: Train and evaluate** | Extract five-second windows from the split clips; exclude transition-crossing windows. Use the frame CNN from [ADR 0004](adr/0004-train-a-cnn-over-window-frames.md), with the class list from `scripts/labels.py`. Train on the verified training set (auto pre-labels only as reduced-weight weak supervision) and choose settings and the low-confidence threshold on validation data only. Evaluate the locked test set once. | Reproducible training and inference; held-out macro F1, per-class precision/recall/F1, confusion matrix, and Uncertain/coverage rate are recorded. Any class missing adequate test examples is called out rather than hidden in an aggregate score. |
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

## Decisions made

1. **The 20-clips-per-class target is revised.** The supplied archive is the whole dataset, so the ceilings are Eating 5, Locomotion 10, Resting 7 and Climbing 0. The target becomes every clip that review verifies, with the per-class counts reported as they are found. See the revision in the [footage audit](FOOTAGE-AUDIT.md#revised-source-count-targets).
2. **Model and compute.** A frame CNN over an ImageNet-pretrained ResNet-18 backbone. One training run takes about 40 seconds for 12 epochs on the RTX 3050. See [ADR 0004](adr/0004-train-a-cnn-over-window-frames.md).
3. **Windows.** Non-overlapping five-second windows, all exactly five seconds long so the input shape is uniform, with a shorter tail dropped. A window is kept only when one verified interval covers it completely, which excludes transition-crossing windows by construction. The annotation keeps source timecodes, so the rule can change without relabeling.
4. **Numeric target.** None is set. With 5 Eating clips and no Climbing clip, a target would be a guess. Report macro F1 and per-class numbers with their support and abstain rate, and let those be the result.
5. **Pre-labeler scope.** Resting and Locomotion stay auto-derived as hints and Eating and Climbing are labeled by hand, per the smoke test's next step. The zero-shot stage must not carry weight in reported metrics.
6. **Climbing is out of scope.** No Climbing-labelled clip was supplied, the two candidate intervals in [#16](https://github.com/ChindanaiNaKub/civet-behavior-detection/issues/16) were reviewed and are not climbing, and the instructor confirmed three classes are enough. `MODEL_CLASSES` stays at three. Climbing remains a valid annotation label that never becomes a model class, and no Climbing work is planned. The review was done by one person, not the two #16 asked for. See [ADR 0007](adr/0007-drop-climbing-from-phase-2-scope.md), which supersedes [ADR 0006](adr/0006-climbing-has-no-evidence-of-absence.md).
7. **The split is chosen on verified windows, and the test split is the gold set.** The first assignment, made on folder names, left test with no Resting window once the intervals were verified. [ADR 0008](adr/0008-resplit-on-verified-windows-gold-is-test.md) re-chose it, and `scripts/split.py` now refuses a split with no verified window for a class. Test has five to seven windows per class, so every metric is reported with its support.
