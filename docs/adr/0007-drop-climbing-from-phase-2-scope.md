# Drop Climbing from Phase 2 scope

Phase 2 classifies three behaviors: Eating, Locomotion and Resting. Climbing is out of scope. No Climbing-labelled clip was delivered, the two candidate intervals raised in [#16](https://github.com/ChindanaiNaKub/civet-behavior-detection/issues/16) were reviewed and are not climbing, and the course instructor confirmed that three classes are enough because no climbing footage was supplied. This supersedes [ADR 0006](0006-climbing-has-no-evidence-of-absence.md), which kept the class open until that review happened.

## Why

- **The review ADR 0006 asked for was done.** One reviewer, a project member, watched both candidate intervals, `Locomotion/25690518090944462.mp4` around 4.5 to 7 s and `Locomotion/25690518091918610.mp4` around 11 to 16 s. The ascent described in #16, crossing the inclined log and going up onto a ledge, was not observed in either. Both intervals stay Locomotion. A contact sheet of the same frames, cut with the command in #16, is consistent with that for the first clip: the animal walks across the floor and the log is not used.
- **The instructor settled the scope question.** ADR 0006 listed whether Climbing is required as a question the repository could not answer. The instructor said three classes are acceptable because no climbing clip was provided.
- **The supplied archive is the whole dataset.** The 22 clips are all the footage the team has, so there is no further source to search for a climb.

## Limits of this decision

- **One reviewer, not two.** #16 asked for two people. This review was done by one person at their request. The conclusion therefore rests on that review plus the instructor's confirmation, not on an independent second pass.
- **Not a claim that the footage cannot contain a climb.** The intervals reviewed were the two candidates raised in #16. The other clips that issue mentioned were not individually reviewed for climbing, and the audit's frame screen remains a sampled check. The report should say a Climbing clip was not supplied and that the candidate intervals were reviewed and ruled out. It should not say the footage has no climbing.

## Consequences

- `MODEL_CLASSES` in [`../../scripts/labels.py`](../../scripts/labels.py) stays at Eating, Locomotion and Resting. No code changes.
- The locked-test-set concern in ADR 0006 no longer applies. The split in [ADR 0005](0005-split-by-clip-not-by-recording-group.md) stands, and `test_locked_by` still waits for a person to review and sign the assignment.
- **Climbing stays a valid annotation label and nothing more.** `labels.ALL_LABELS`, the review page and the annotation guide still accept it, so an annotator who sees a real climb can record it instead of forcing it into Locomotion. A Climbing interval is not a model class: `scripts/windows.py` drops any window it touches as `non_class_label`, and `test_climbing_is_not_a_model_class_yet` in `scripts/test_windows.py` covers that, so a stray Climbing row cannot reach training or the reported evaluation. No Climbing-specific work is planned.
- **The pre-labeler is unchanged.** `scripts/prelabel.py` can still propose Climbing, and its four proposals on the supplied clips are false positives. A reviewer rejects them. Removing Climbing from its prompts is a separate cleanup, not part of this decision.
- The annotation guide records that no Climbing interval is expected in the supplied clips, and what to do if one appears.
- If more footage arrives later, adding the class means adding it to `MODEL_CLASSES`, a split that holds it, and a fresh review of the new clips.
