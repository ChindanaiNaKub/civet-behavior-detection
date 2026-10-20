# Re-split on verified windows; the gold set is the test split

The split stays by whole clip (ADR 0005), but the assignment is now chosen and checked on the windows the verified intervals produce, not on folder names. The test split is the gold set: eight whole clips, listed in [`../../data/gold-clips.txt`](../../data/gold-clips.txt), labeled blind by two different people. [`../../scripts/split.py`](../../scripts/split.py) refuses an assignment that leaves a model class with no window in any split. This supersedes the assignment in ADR 0005, not its by-clip rule.

## Why

- The first assignment was stratified on folder names, the only labels that existed. Once the intervals were verified (`data/annotations.csv`, 62 intervals), `scripts/windows.py` gave the test split Eating 1, Locomotion 5 and Resting 0 windows. Both Resting-folder test clips are mostly Uncertain because a plant hides the animal, so they produce no Resting window. A test split with no Resting window leaves macro F1 undefined for that class.
- ADR 0005 already says to confirm every class is represented in the held-out set. This is that confirmation, run against the verified labels instead of the folder names.
- The reported metrics must come from labels a person made without the machine proposal (ADR 0003). Making the test split the gold set means every number that is reported rests on those labels.

## The assignment

| split | clips | windows (Eating / Locomotion / Resting) |
|---|---:|---|
| train | 11 | 14 / 5 / 74 |
| validation | 3 | 10 / 1 / 19 |
| test (gold) | 8 | 7 / 2 / 6 |

Two clips moved into test to give it Eating and Resting: `25690519193422801` (Eating, 50 s) and `25690519194200758` (Resting, 31 s). The gold passes are done: two people labeled the eight clips independently, the passes were reconciled in `scripts/reconcile.py`, and the final intervals are in `data/annotations.csv`. The table shows the windows from those final intervals. Test Locomotion fell from 5 windows (the first annotator's labels alone) to 2, because the reconciled labels mark the stretches where the civet stops, turns or is partly hidden as Transition or Uncertain, and keep Locomotion only where its movement is visible; Eating (7) and Resting (6) are unchanged. Locomotion in test now rests on two windows from two clips.

## Consequences

- **The test split is small.** Five to seven windows per class. A per-class score from it moves a lot with one window, so every metric is reported with its support, and the report calls the result a baseline.
- **Locomotion is thin everywhere.** Eleven windows in total, because most Locomotion intervals are shorter than five seconds and a window needs one interval to cover it fully.
- **Lighting and camera still track the class.** Across the 141 windows, all 99 Resting windows are daylight on Camera 03 and 30 of 31 Eating windows are daylight on Camera 02. Locomotion supplies every infrared window (7 of its 11). A model can score well by recognising the recording rather than the behavior, which is the failure the plumbing run showed (validation macro F1 1.000, test 0.179). The score is an upper bound on generalization to a new recording, as ADR 0005 says.
- **The blind pass on the two moved clips is weaker.** The first annotator saw the proposal and their own earlier labels for those clips. The second annotator has no such history, which is why the agreement score matters most on them.
- **Locked.** Both gold passes exist, the agreement is computed and reconciled, and `split.py` passes on the final labels, so Gun signed `test_locked_by` and `test_locked_at` on the eight test rows of `data/splits.csv` on 2026-10-20, and reports that the repository owner agreed. `split.py` now refuses to overwrite a signed ledger. `scripts/evaluate.py` does not check the lock itself; it must read the test windows once, after the threshold is chosen on validation.
- **Fallback.** If the blind labels leave a class with no window, use group-aware cross-validation for development as the plan allows, and report that no held-out estimate is reliable.

## Update 2026-10-21

The final reconciled ledger gives 138 windows in total (Eating 31, Locomotion 8, Resting 99). The 141-window and 11-Locomotion figures in the consequences above come from an earlier label pass, before the reconciliation. The split assignment and the decision in this ADR are unchanged. The held-out evaluation is in [`../../docs/EVALUATION-REPORT.md`](../../docs/EVALUATION-REPORT.md).
