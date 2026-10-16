# Automate behavior pre-labeling; keep a human gold set

Phase 2 will pre-label behavior intervals automatically instead of hand-labeling every clip. Resting is derived from sustained low motion (at least five seconds) and Locomotion from location change; a compact zero-shot pretrained video model proposes Eating and Climbing intervals. A human verifies and corrects the proposed timeline. A small gold set of whole source clips stays fully human-labeled and is the only data used to report model quality.

## Why

- Hand-labeling every interval is the project's largest human cost and the main reason the schedule is human-bound. With 22 clips, one animal, two cameras, and no Climbing example, a hand-labeled dataset is expensive to reach.
- Two of the four classes need no learned labels: Resting follows from duration and Locomotion from displacement. Only Eating-versus-Climbing-versus-idle separation needs a trained classifier.
- If the machine labels the data and the machine is then scored on those same labels, the metric is circular. The human gold set is what keeps it honest.

## Consequences

- Issue #4 changes from "humans label intervals" to "machine pre-labels, human verifies, small gold set stays human." The pre-labeler is tracked separately (issue #9).
- The planned 20% shared sample becomes the seed of the gold hold-out. Two-person labeling is kept only for the gold set and the Climbing gap, not for the whole dataset.
- Auto labels are weak supervision only: they may train the classifier with reduced loss weight, but they never enter reported evaluation.
- The gold set must be labeled without seeing the machine proposal, or the reported metric is contaminated.
- Zero-shot pre-label quality on this footage is unproven; Resting and Climbing are the likely failure modes. One smoke test is budgeted before the plan commits to the pre-labeler.
- Climbing still has no verified example and the pre-labeler cannot invent one. Source acquisition remains the top gap.

This supersedes the "manually label every time range" wording in the Phase 2 plan and annotation guide.
