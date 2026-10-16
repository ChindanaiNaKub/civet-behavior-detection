# Civet Behavior Detection

Phase 2 project for building a checkable short-clip behavior-classification baseline from civet footage.

## Current status

The project is in planning and data-preparation. No behavior model has been trained or evaluated yet. The Phase 2 goal is a reproducible baseline and timeline demo by **October 31, 2026**.

## Start here

- [Phase 2 plan](docs/PHASE-2-PLAN.md)
- [Footage and compute audit](docs/FOOTAGE-AUDIT.md)
- [Behavior interval annotation guide](docs/ANNOTATION-GUIDE.md)
- [Behavior pre-labeler smoke test](docs/PRELABEL-SMOKE-TEST.md)
- [Behavior glossary](GLOSSARY.md)
- [Decision: classify short behavior clips](docs/adr/0001-classify-short-behavior-clips.md)
- [Decision: track source clips in the repository](docs/adr/0002-track-source-clips-in-repository.md)
- [Decision: automate pre-labeling with a human gold set](docs/adr/0003-automate-prelabeling-with-human-gold-set.md)
- [Behavior dataset ledger and templates](data/README.md)
- [Phase 2 plan issue #1](https://github.com/ChindanaiNaKub/civet-behavior-detection/issues/1)
- [Phase 2 spec issue #2](https://github.com/ChindanaiNaKub/civet-behavior-detection/issues/2)

The pre-labeler for [#9: Auto pre-label behavior timelines (zero-shot)](https://github.com/ChindanaiNaKub/civet-behavior-detection/issues/9) is in `scripts/prelabel.py`. Its [smoke test](docs/PRELABEL-SMOKE-TEST.md) found that motion alone pre-labels Resting reliably and Locomotion partly, while the zero-shot model proposes no Eating interval and only false Climbing ones — so those two classes are labeled by hand. That unblocks [#4: Create reviewed behavior-interval dataset](https://github.com/ChindanaiNaKub/civet-behavior-detection/issues/4). Follow the ticket links for the remaining dataset, classifier, and evaluation work.

## Phase 2 scope

The agreed labels are Eating, Locomotion, Resting, and Climbing. Phase 2 adapts the brief's still-image classification option to five-second video windows because movement and duration are needed to identify Locomotion and Resting. The initial model will use a compact pretrained video classifier selected after the data and compute audit.

Behavior intervals are pre-labeled automatically, not hand-labeled from scratch: Resting and Locomotion come from motion, and a zero-shot model proposes Eating and Climbing for human verification. A small human gold set is the only data used for reported model quality. See [decision 0003](docs/adr/0003-automate-prelabeling-with-human-gold-set.md).

The 22 audited clips are tracked in `data/raw/` so collaborators get them on clone. They are stored as delivered and are unverified: folder-level labels only, no verified behavior intervals, no verified Climbing example, and room/camera coverage unknown. See the [footage audit](docs/FOOTAGE-AUDIT.md) for the inventory and compute check, the [annotation guide](docs/ANNOTATION-GUIDE.md) for labeling rules, and the plan for split and evaluation rules.

## Data handling

The 22 source clips are committed under `data/raw/<folder>/<clip_id>.mp4`, matching `relative_path` in [`data/source-clips.csv`](data/source-clips.csv). Rationale is recorded in [decision 0002](docs/adr/0002-track-source-clips-in-repository.md). Source archives, other footage formats, and model checkpoints are not stored here; track those in team-approved storage with source, stable identifier, duration, and verified capture metadata. These clips are deliverable to collaborators with read access to this private repository — do not republish them elsewhere without team approval.

## Repository contents

- `GLOSSARY.md` — agreed behavior language.
- `data/raw/` — the 22 tracked source clips, one subdirectory per folder label.
- `scripts/prelabel.py` — behavior pre-labeler: Resting from sustained low motion, Locomotion from location change, Eating/Climbing proposed by zero-shot X-CLIP; writes a per-interval timeline CSV for review. Logic checks in `scripts/test_prelabel.py`.
- `docs/PHASE-2-PLAN.md` — schedule, workstreams, evaluation, and risks.
- `docs/adr/` — project decisions and their rationale.
- `docs/agents/` — issue-tracker and domain-document conventions for agents.
