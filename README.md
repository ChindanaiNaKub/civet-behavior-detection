# Civet Behavior Detection

Phase 2 project for building a checkable short-clip behavior-classification baseline from civet footage.

## Current status

The pipeline is built and runs end to end. `data/annotations.csv` covers all 22 clips: the first annotator's verified intervals for the 14 train and validation clips, and the reconciled final intervals for the eight gold clips. The two independent gold passes were compared (Dice 0.597 over the eight clips, in `data/agreement.csv`) and reconciled with `scripts/reconcile.py`. What is still missing is the test lock: a person reviews `data/splits.csv` and fills `test_locked_by`, and only then may `scripts/evaluate.py` run. No model has been evaluated on the test split and no reported number exists yet. The Phase 2 goal is a reproducible baseline and timeline demo by **October 31, 2026**.

The model classes are Eating, Locomotion and Resting. The brief calls the third class Walking, and the repository calls it Locomotion because the Phase 1 glossary widened it to any movement that changes location. Climbing is out of scope: no Climbing clip was supplied, the two candidate intervals were reviewed and are not climbing, and the instructor confirmed three classes are enough. It stays a valid annotation label that never becomes a model class, and no Climbing work is planned. See [ADR 0004](docs/adr/0004-train-a-cnn-over-window-frames.md) for the model, [ADR 0005](docs/adr/0005-split-by-clip-not-by-recording-group.md) for the split rule and [ADR 0007](docs/adr/0007-drop-climbing-from-phase-2-scope.md) for Climbing.

## Start here

- [คู่มือหน้าตรวจสำหรับ Windows / Fedora Linux](docs/REVIEWER-QUICKSTART.md)
- [Gold review handoff: second reviewer, reconciliation and test lock](docs/GOLD-REVIEW-HANDOFF.md)
- [Phase 2 plan](docs/PHASE-2-PLAN.md)
- [Footage and compute audit](docs/FOOTAGE-AUDIT.md)
- [Behavior interval annotation guide](docs/ANNOTATION-GUIDE.md)
- [Behavior pre-labeler smoke test](docs/PRELABEL-SMOKE-TEST.md)
- [Behavior glossary](GLOSSARY.md)
- [Decision: classify short behavior clips](docs/adr/0001-classify-short-behavior-clips.md)
- [Decision: track source clips in the repository](docs/adr/0002-track-source-clips-in-repository.md)
- [Decision: automate pre-labeling with a human gold set](docs/adr/0003-automate-prelabeling-with-human-gold-set.md)
- [Decision: train a CNN over the frames of each window](docs/adr/0004-train-a-cnn-over-window-frames.md)
- [Decision: split by clip, not by recording group](docs/adr/0005-split-by-clip-not-by-recording-group.md)
- [Decision: Climbing has no evidence of absence (superseded)](docs/adr/0006-climbing-has-no-evidence-of-absence.md)
- [Decision: drop Climbing from Phase 2 scope](docs/adr/0007-drop-climbing-from-phase-2-scope.md)
- [Decision: re-split on verified windows, gold set is the test split](docs/adr/0008-resplit-on-verified-windows-gold-is-test.md)
- [Behavior dataset ledger and templates](data/README.md)
- [Phase 2 plan issue #1](https://github.com/ChindanaiNaKub/civet-behavior-detection/issues/1)
- [Phase 2 spec issue #2](https://github.com/ChindanaiNaKub/civet-behavior-detection/issues/2)

The pre-labeler for [#9: Auto pre-label behavior timelines (zero-shot)](https://github.com/ChindanaiNaKub/civet-behavior-detection/issues/9) is in `scripts/prelabel.py`. Its [smoke test](docs/PRELABEL-SMOKE-TEST.md) found that motion alone pre-labels Resting reliably and Locomotion partly, while the zero-shot model proposes no Eating interval and only false Climbing ones — so those two classes are labeled by hand. That unblocks [#4: Create reviewed behavior-interval dataset](https://github.com/ChindanaiNaKub/civet-behavior-detection/issues/4). Follow the ticket links for the remaining dataset, classifier, and evaluation work.

## Run the pipeline

Install the pinned versions, then run these from the repository root. Steps 2, 4 and the gold-set passes need a person.

```
python3 -m pip install -r requirements.txt
```

1. Propose a behavior timeline for every clip. Takes about three minutes on the audit GPU and writes `outputs/prelabel/timeline.csv`.

```
python3 scripts/prelabel.py
```

2. Verify those proposals against the video. This is the human step, roughly two hours for all 22 clips. The page serves each clip under the player and saves accepted intervals to `data/annotations.csv`.

```
python3 scripts/review.py
```

3. Write the split ledger and the windows. `scripts/split.py` assigns every clip to train, validation or test and refuses to write a split that leaves a class with no verified window. `scripts/lighting.py` fills the observed lighting, visibility and occlusion in `data/annotations.csv`; re-run it after saving a clip in the review page.

```
python3 scripts/split.py
python3 scripts/windows.py
python3 scripts/lighting.py
```

4. Review `data/splits.csv` and record who locked the test set in `test_locked_by`. Do this before the next step if the numbers are to be reported, because the test set must be fixed before it is read.

5. Train, then score the locked test set once. The threshold is chosen on the validation windows, and the test windows are read once, after the threshold is fixed.

```
python3 scripts/train.py
python3 scripts/evaluate.py
```

6. Show one clip's predicted timeline against its own timecodes. The threshold comes from step 5.

```
python3 scripts/demo.py data/raw/Resting/25690519194809256.mp4
```

7. Check the pipeline logic. No GPU, no video and no model download.

```
python3 -m pytest scripts
```

The gold set is the eight clips in `data/gold-clips.txt`, which are also the test split (ADR 0008). Two different people label them blind, each without opening the other's file, `data/annotations.csv` or `outputs/prelabel/`. `review.py` needs only the Python standard library.

First annotator, who also redoes the two clips that moved into test (`--redo`) so the page serves them again:

```
python3 scripts/review.py --only-file data/gold-clips.txt --blind-file data/gold-clips.txt --redo 25690519193422801 --redo 25690519194200758
```

Second annotator, into their own file:

For a separate folder containing only blind review materials, run
`python scripts/prepare_gold_review.py` and give `outputs/gold-review/` to the
second person. They run `python start_review.py` there and return
`data/annotations-second.csv`. This avoids exposing existing answers in the
repository. See the [handoff guide](docs/GOLD-REVIEW-HANDOFF.md) for receiving,
reconciling and locking the final dataset. If working in the repository instead:

```
python3 scripts/review.py --only-file data/gold-clips.txt --blind-file data/gold-clips.txt --out data/annotations-second.csv
```

Then compare the two independent passes and fill the reconciliation columns in `data/agreement.csv`.

The comparison requires every gold clip and two distinct, consistent annotator
identities. It refuses to overwrite existing reconciliation work. File validation
cannot verify that a reviewer worked independently or watched the whole clip.

```
python3 scripts/agreement.py
```

## Phase 2 scope

The model classes are Eating, Locomotion and Resting; Climbing is out of scope ([ADR 0007](docs/adr/0007-drop-climbing-from-phase-2-scope.md)). Phase 2 adapts the brief's still-image classification option to five-second video windows because movement and duration are needed to identify Locomotion and Resting. The initial model will use a compact pretrained video classifier selected after the data and compute audit.

Behavior intervals are pre-labeled automatically, not hand-labeled from scratch: Resting and Locomotion come from motion, and a zero-shot model proposes Eating and Climbing for human verification. A small human gold set is the only data used for reported model quality. See [decision 0003](docs/adr/0003-automate-prelabeling-with-human-gold-set.md).

The 22 audited clips are tracked in `data/raw/` so collaborators get them on clone. They are stored as delivered and are unverified: folder-level labels only, no verified behavior intervals, no verified Climbing example, and room/camera coverage unknown. See the [footage audit](docs/FOOTAGE-AUDIT.md) for the inventory and compute check, the [annotation guide](docs/ANNOTATION-GUIDE.md) for labeling rules, and the plan for split and evaluation rules.

## Data handling

The 22 source clips are committed under `data/raw/<folder>/<clip_id>.mp4`, matching `relative_path` in [`data/source-clips.csv`](data/source-clips.csv). Rationale is recorded in [decision 0002](docs/adr/0002-track-source-clips-in-repository.md). Source archives, other footage formats, and model checkpoints are not stored here; track those in team-approved storage with source, stable identifier, duration, and verified capture metadata. These clips are deliverable to collaborators with read access to this private repository — do not republish them elsewhere without team approval.

## Repository contents

- `GLOSSARY.md` — agreed behavior language.
- `data/raw/` — the 22 tracked source clips, one subdirectory per folder label.
- `scripts/labels.py` — the behavior label domain and the CSV contracts, so a label rule is stated once.
- `scripts/prelabel.py` — behavior pre-labeler: Resting from sustained low motion, Locomotion from location change, Eating/Climbing proposed by zero-shot X-CLIP; writes a per-interval timeline CSV for review. Logic checks in `scripts/test_prelabel.py`.
- `scripts/review.py` — local review page: plays each clip with its proposed intervals and writes verified intervals to `data/annotations.csv`. Supports blind clips for the gold set.
- `scripts/agreement.py` — compares two independent label passes on the shared sample and writes `data/agreement.csv`.
- `scripts/reconcile.py` — browser page where the two gold reviewers decide one final set of intervals per gold clip; `--apply` writes `data/agreement.csv` and the final gold rows. Logic checks in `scripts/test_reconcile.py`.
- `scripts/split.py` — assigns every clip to train, validation or test, checks each split against the verified windows, and writes `data/splits.csv`.
- `scripts/lighting.py` — fills lighting, visibility and occlusion in `data/annotations.csv` from the footage and the annotator's own reasons.
- `data/gold-clips.txt` — the eight gold clips, one id per line, also the test split.
- `scripts/windows.py` — builds exactly five-second labeled windows from the verified intervals, keeping only windows one verified model-class interval covers completely.
- `scripts/frames.py` — decodes the frames of a window.
- `scripts/model.py` — the frame CNN and its checkpoint format.
- `scripts/train.py` — trains on the train split, early-stops on the validation split.
- `scripts/evaluate.py` — chooses the abstain threshold on validation, scores the locked test set once, and writes the report block.
- `scripts/demo.py` — prints one clip's predicted behavior timeline.
- `scripts/metrics.py` — class metrics that count an abstention as a miss.
- `requirements.txt` — the pinned versions the pipeline was run with.
- `docs/PHASE-2-PLAN.md` — schedule, workstreams, evaluation, and risks.
- `docs/adr/` — project decisions and their rationale.
- `docs/agents/` — issue-tracker and domain-document conventions for agents.

`outputs/` holds generated artifacts and is not tracked: the pre-label timeline, the windows, the checkpoints and the evaluation output. Every one of them is regenerated by the commands above.
