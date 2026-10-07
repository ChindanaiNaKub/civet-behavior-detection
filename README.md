# Civet Behavior Detection

Phase 2 project for building a checkable short-clip behavior-classification baseline from civet footage.

## Current status

The project is in planning and data-preparation. No behavior model has been trained or evaluated yet. The Phase 2 goal is a reproducible baseline and timeline demo by **October 31, 2026**.

## Start here

- [Phase 2 plan](docs/PHASE-2-PLAN.md)
- [Footage and compute audit](docs/FOOTAGE-AUDIT.md)
- [Behavior interval annotation guide](docs/ANNOTATION-GUIDE.md)
- [Behavior glossary](GLOSSARY.md)
- [Decision: classify short behavior clips](docs/adr/0001-classify-short-behavior-clips.md)
- [Behavior dataset ledger and templates](data/README.md)
- [Phase 2 plan issue #1](https://github.com/ChindanaiNaKub/civet-behavior-detection/issues/1)
- [Phase 2 spec issue #2](https://github.com/ChindanaiNaKub/civet-behavior-detection/issues/2)

The next actionable ticket is [#3: Audit video sources and lock annotation rules](https://github.com/ChindanaiNaKub/civet-behavior-detection/issues/3). Follow the ticket links for the remaining dataset, classifier, and evaluation work.

## Phase 2 scope

The agreed labels are Eating, Locomotion, Resting, and Climbing. Phase 2 adapts the brief's still-image classification option to five-second video windows because movement and duration are needed to identify Locomotion and Resting. The initial model will use a compact pretrained video classifier selected after the data and compute audit.

The current 22 clips have folder-level labels only. They are not verified behavior intervals, no Climbing example has been verified, and room/camera coverage remains unknown. See the [footage audit](docs/FOOTAGE-AUDIT.md) for the inventory and compute check, the [annotation guide](docs/ANNOTATION-GUIDE.md) for labeling rules, and the plan for split and evaluation rules.

## Data handling

Raw videos, source archives, and model checkpoints are not stored in this Git repository. Keep footage in team-approved storage and track its source, stable identifier, duration, and verified capture metadata. Add any derived annotation data only after confirming the team can share it in this private repository.

## Repository contents

- `GLOSSARY.md` — agreed behavior language.
- `docs/PHASE-2-PLAN.md` — schedule, workstreams, evaluation, and risks.
- `docs/adr/` — project decisions and their rationale.
- `docs/agents/` — issue-tracker and domain-document conventions for agents.
