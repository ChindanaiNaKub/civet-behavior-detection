# Split by clip, not by recording group

> **Assignment superseded by [ADR 0008](0008-resplit-on-verified-windows-gold-is-test.md).** The by-clip rule stands. The clip assignment below was stratified on folder names and left the test split with no Resting window once the intervals were verified, so ADR 0008 re-chose it on the verified windows.

Train, validation and test are split by whole source clip, stratified by class, so every split holds Eating, Locomotion and Resting. The `(camera, date)` recording group is recorded on every row of [`../../data/splits.csv`](../../data/splits.csv) but may appear on both sides of the split. [`../../scripts/split.py`](../../scripts/split.py) generates the ledger, so changing the assignment is one edit there.

## Why

- The Phase 2 plan preferred splitting by recording session or camera, and the evaluation rules still do. This decision departs from that preference for one measured reason.
- Camera and class are confounded in this footage. 597 of the 659 seconds of Resting are Camera 03 on 2026-05-19, and every Locomotion clip is on 2026-05-18. A camera-held-out split would train on 62 seconds of Resting and test on 597 seconds, so the score would measure that mismatch instead of the model.
- No group split puts all three classes on both sides. There are five `(camera, date)` groups, and Locomotion appears in only two of them while Resting appears in only two others, so any group-isolated test loses a class from either training or testing.
- The plan already allows this fallback. It says to report the limitation and use group-aware development when the source clips cannot support a meaningful held-out set for every class.

## Consequences

- Windows from one recording group can land in both train and test. The measured score is an upper bound on generalization to a new recording, not an estimate of it. Report it that way.
- Every split keeps all three classes, so macro F1 is defined for all of them and no class silently contributes a zero.
- The ledger records the straddle on every row, in `assignment_reason`, so a reader sees which groups appear twice.
- The test assignment is not locked by an agent. `test_locked_by` stays empty until a person reviews the ledger and signs it, because a machine that both picks and certifies the test set is not a check.
- The weak-label plumbing run showed the cost directly: validation macro F1 1.000 against test macro F1 0.179, because the model learned the recording group rather than the behavior. That run used folder-name hints, not verified intervals, so it measures the confound and not the model. It is the reason this consequence is stated first.
- When more recording sessions exist, return to a group-held-out test.
