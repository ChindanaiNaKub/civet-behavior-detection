# Held-out evaluation report (issue #6)

The baseline was scored once on the locked test clips on 2026-10-21. This report
records that run, the data it rests on, the one error it made, and what the
number does not prove.

Test macro F1 is **0.949** over the three model classes, on 15 windows. The model
abstained on 0 windows, so the Uncertain rate is 0.0 percent.

Gun signed the test lock in `data/splits.csv` on 2026-10-20, before this run.
No model setting and no threshold was changed after the test was read.

## What was evaluated

The model is the frame CNN from [ADR 0004](adr/0004-train-a-cnn-over-window-frames.md),
trained by `scripts/train.py` on the train split and early-stopped on the
validation split. It is the checkpoint produced for issue #5.

| Item | Value |
|---|---|
| Checkpoint | `outputs/model/model.pt`, not tracked |
| Backbone | ImageNet ResNet-18, frozen |
| Input | 8 frames per five-second window, 128 by 128 |
| Trainable parameters | 3,075 |
| Train windows | 93 (Eating 14, Locomotion 5, Resting 74) |
| Validation windows | 30 (Eating 10, Locomotion 1, Resting 19) |
| Test windows | 15 (Eating 7, Locomotion 2, Resting 6) |
| Seed | 0 |
| Best epoch | 4 |
| Validation macro F1 | 1.000 at the selected threshold |
| Abstain threshold | 0.00, chosen on validation |

The classes are Eating, Locomotion and Resting. Climbing has no source clip and
is out of scope ([ADR 0007](adr/0007-drop-climbing-from-phase-2-scope.md)), so
this is a three-class result. A four-class macro F1 is not defined for this
baseline. Reporting the same score divided over four classes would add a class
with no support, not a measurement.

## Reproduce this run

From the repository root, after `python3 -m pip install -r requirements.txt`:

```
python3 scripts/windows.py
python3 scripts/train.py
python3 scripts/evaluate.py
```

`scripts/evaluate.py` chooses the threshold on the 30 validation windows, then
decodes and scores the test windows once. It writes `outputs/evaluation.md` and
`outputs/evaluation.json`, and the JSON keeps every test window with its clip,
time range, label, prediction and class probabilities. Those generated files are
not tracked, so the table below is the record of this run.

The run environment was an RTX 3050 Laptop GPU with 4 GiB, Python 3.14,
torch 2.10.0+cu128 and torchvision 0.25.0+cu128. Training used about 40 seconds.
Retraining on another device may not reproduce the weights exactly.

## Verified dataset coverage

The 22 source clips produce 201 five-second grid windows. 138 windows have one
verified model-class label that covers the whole window. The other 63 are
dropped: 46 are not covered completely by one interval and 17 touch a Transition
or Uncertain span. No auto pre-label enters these counts or the scores.

| Split | Clips assigned | Clips with windows | Windows | Eating | Locomotion | Resting |
|---|---:|---:|---:|---:|---:|---:|
| Train | 11 | 6 | 93 | 14 | 5 | 74 |
| Validation | 3 | 3 | 30 | 10 | 1 | 19 |
| Test (locked gold) | 8 | 5 | 15 | 7 | 2 | 6 |
| **Total** | **22** | **14** | **138** | **31** | **8** | **99** |

The test split is the eight-clip gold set, but only five of those clips produce a
kept window. The other three consist mostly of Uncertain, Transition or short
Locomotion intervals, so no five-second window is covered by one class interval.
Their labels still exist; they simply cannot yield a scorable window under the
current window rule.

| Test class | Windows | Clips | Where the support sits |
|---|---:|---:|---|
| Eating | 7 | 2 | 6 windows from `25690519193422801`, 1 from `25690519193808204` |
| Locomotion | 2 | 2 | 1 window each from `25690518090944462` and `25690518091012230` |
| Resting | 6 | 1 | all 6 from `25690519194200758` |
| **Total** | **15** | **5** | |

Camera and class still travel together:

| Class | Camera 02 windows | Camera 03 windows |
|---|---:|---:|
| Eating | 30 | 1 |
| Locomotion | 1 | 7 |
| Resting | 0 | 99 |

Every Resting window is Camera 03 on 2026-05-19. Locomotion is mostly Camera 03
on 2026-05-18, and nearly all Eating windows are Camera 02. The split is by
whole clip, and the train and test sides share recording groups, so the score is
an upper bound on generalization to a new recording, not an estimate of it
([ADR 0005](adr/0005-split-by-clip-not-by-recording-group.md)).

## Annotation agreement

The eight test clips were labeled twice, intended to be blind, and reconciled
into the labels scored here. Agreement before reconciliation is interval Dice
0.597 over the eight clips, from `data/agreement.csv`.

| Clip | Intervals compared | Matched labels | Dice |
|---|---:|---:|---:|
| `25690518090944462` | 5 | 2 | 0.800 |
| `25690518091012230` | 5 | 2 | 0.800 |
| `25690518091918610` | 13 | 2 | 0.308 |
| `25690519193422801` | 13 | 5 | 0.769 |
| `25690519193808204` | 18 | 4 | 0.444 |
| `25690519194200758` | 4 | 1 | 0.500 |
| `25690520152850102` | 5 | 2 | 0.800 |
| `25690520153258104` | 4 | 2 | 1.000 |
| **Overall** | **67** | **20** | **0.597** |

This is Dice over interval counts, not temporal overlap. It rewards two
annotators who mark the same behavior with different boundaries, which is why
the reconciliation kept boundary shifts under 0.5 seconds instead of treating
them as different labels. The reconciliation decisions are recorded per clip in
`data/agreement.csv`.

The independence limit is real. The second pass was not fully independent of the
first, and the first annotator delegated the reconciliation decisions to the
second. The agreement score measures how the two passes compared, not proof that
either person worked blind.

## Held-out results

These are the measured test numbers from the single run. The validation block
follows for context; it is not a held-out result.

### Test clips

Windows: 15. Abstained: 0 (0.0%). Accuracy over decided windows: 0.933.

| Behavior | Support | Predicted | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Eating | 7 | 6 | 1.000 | 0.857 | 0.923 |
| Locomotion | 2 | 2 | 1.000 | 1.000 | 1.000 |
| Resting | 6 | 7 | 0.857 | 1.000 | 0.923 |

**Macro F1 0.949** over 3 classes. Macro F1 over the classes with support: 0.949.

Confusion matrix, rows true, columns predicted, last column abstained:

| true \ pred | Eating | Locomotion | Resting | abstain |
|---|---:|---:|---:|---:|
| Eating | 6 | 0 | 1 | 0 |
| Locomotion | 0 | 2 | 0 | 0 |
| Resting | 0 | 0 | 6 | 0 |

### Validation clips at the same threshold

Windows: 30. Abstained: 0 (0.0%). Accuracy over decided windows: 1.000.
Macro F1 1.000. Per-class F1 is 1.000 for Eating (support 10), Locomotion
(support 1) and Resting (support 19). Validation Locomotion rests on one window,
so its perfect score is not a stable estimate.

### Every test window

| Clip | Start (s) | End (s) | Labeled | Predicted | Confidence |
|---|---:|---:|---|---|---:|
| `25690518090944462` | 5 | 10 | Locomotion | Locomotion | 0.600 |
| `25690518091012230` | 0 | 5 | Locomotion | Locomotion | 0.474 |
| `25690519193422801` | 10 | 15 | Eating | Eating | 0.687 |
| `25690519193422801` | 15 | 20 | Eating | Eating | 0.666 |
| `25690519193422801` | 20 | 25 | Eating | Eating | 0.688 |
| `25690519193422801` | 25 | 30 | Eating | Eating | 0.693 |
| `25690519193422801` | 30 | 35 | Eating | Eating | 0.702 |
| `25690519193422801` | 35 | 40 | Eating | Eating | 0.692 |
| `25690519193808204` | 40 | 45 | Eating | **Resting** | 0.847 |
| `25690519194200758` | 0 | 5 | Resting | Resting | 0.858 |
| `25690519194200758` | 5 | 10 | Resting | Resting | 0.875 |
| `25690519194200758` | 10 | 15 | Resting | Resting | 0.875 |
| `25690519194200758` | 15 | 20 | Resting | Resting | 0.872 |
| `25690519194200758` | 20 | 25 | Resting | Resting | 0.842 |
| `25690519194200758` | 25 | 30 | Resting | Resting | 0.839 |

The threshold is 0.00, so every window is decided and the abstain column is empty
by construction. That is the honest result of a validation split the model
separates perfectly. It also means this run cannot show what a useful Uncertain
rate would be. The two Locomotion windows are the least confident, at 0.474 and
0.600, and both are above the threshold.

## Error example

There is one error. The 40 to 45 second window of
`data/raw/Eating/25690519193808204.mp4` is labeled Eating and was predicted
Resting with confidence 0.847. The window is the only kept window from that clip.

The label is not easy to confirm from a still. Extracted frames at 40.0, 43.0
and 45.0 seconds show the enclosure floor and the plant, with no clearly
identifiable feeding action, and the annotation records partial visibility with
fence or mesh occlusion. The label was not changed after the test was read. Treat
this window as unresolved: it may be a model error on a partly hidden animal, or
an annotation error. It counts as a miss in the numbers above.

One window is 1 of 7 Eating windows and 1 of 15 test windows, so this single
disagreement moves Eating recall by about 14 points. That is the size of the
test set, not a property of the model.

## Demo

The demo was re-run from the documented setup with the threshold from
`outputs/evaluation.json`. The clip is from the validation split, so the demo
does not reuse a test clip.

```
python3 scripts/demo.py data/raw/Eating/25690519193547292.mp4 \
  --evaluation outputs/evaluation.json
```

Output from this run:

```
data/raw/Eating/25690519193547292.mp4  (65.0 s, 13 windows, abstain below 0.00)
  start     end  behavior      conf  leaned toward
   0.00    5.00  Eating        0.57  Eating
   5.00   10.00  Eating        0.71  Eating
  10.00   15.00  Eating        0.71  Eating
  15.00   20.00  Eating        0.71  Eating
  20.00   25.00  Eating        0.70  Eating
  25.00   30.00  Eating        0.65  Eating
  30.00   35.00  Eating        0.69  Eating
  35.00   40.00  Eating        0.67  Eating
  40.00   45.00  Eating        0.70  Eating
  45.00   50.00  Eating        0.66  Eating
  50.00   55.00  Eating        0.68  Eating
  55.00   60.00  Eating        0.66  Eating
  60.00   65.00  Eating        0.66  Eating

timeline, one character per window: EEEEEEEEEEEEE
legend: E Eating, L Locomotion, R Resting, ? Uncertain
```

The output matches the issue #5 demo row for row, which is the reproducibility
check for this run. The header names the source clip and every row names its
original five-second time range, behavior and confidence.

## Limitations

- The test split has 15 windows from 5 clips. Locomotion has 2 windows from 2
  clips, and Resting has 6 windows from 1 clip. A per-class score from this
  support is a baseline, not a stable estimate.
- Eating is dominated by one clip (6 of 7 windows), and Resting by one clip (all
  6). The camera and recording group still track the class, so a high score can
  come from recognising the recording rather than the behavior.
- The split is by clip, not by recording group. It is an upper bound on
  generalization to a new recording, not an estimate of it.
- The threshold is 0.00 and no window abstained. The run reports an Uncertain
  rate of 0.0 percent, but it does not demonstrate a usable abstention rule.
- The one error is on a window whose label could not be confirmed from a still.
  The label and the prediction cannot be separated without a re-review.
- The gold agreement is 0.597, and the second pass was not fully independent.
  The reconciled labels carry those decisions.
- Climbing has no source clip and no test example. No four-class result exists.
- The result comes from one training run at seed 0. No run-to-run spread is
  reported.
- The classifier sees eight frames at 128 by 128. Small or partly hidden
  behavior may be invisible at that resolution.

## Next steps

1. Add footage from recording sessions that are absent from training, then
   repeat the evaluation with a group-held-out test. That would turn the upper
   bound into an estimate.
2. Re-review the gold label for `25690519193808204` at 35.74 to 45.60 seconds.
   Keep the current lock for this report; if the label changes, publish a new
   evaluation as a separate run.
3. Collect more Locomotion and Resting test clips, and more than one clip per
   class, before treating per-class differences as real.
4. Choose an abstain threshold on a larger validation set, or on a separate
   calibration set, so the Uncertain rate is meaningful. Do not choose it on the
   test windows.
5. Repeat training with several seeds and report the mean and spread. The current
   single run cannot say how much of the score is the seed.
6. Revisit Climbing only if verified footage exists. Until then, three classes
   are the scope.

## Measured and planned

| Item | Status |
|---|---|
| Three-class held-out macro F1 | Measured, 0.949 on 15 test windows |
| Per-class precision, recall and F1 | Measured, table above |
| Confusion matrix with abstain column | Measured, empty abstain column |
| Uncertain rate | Measured, 0.0 percent at the validation-selected threshold |
| Annotation agreement | Measured, interval Dice 0.597 |
| Verified window coverage | Measured, 138 windows from 14 clips |
| Demo on a validation clip | Measured, output above |
| Group-held-out generalization | Not measured, no recording group is held out |
| Performance on a new recording | Not measured, requires new footage |
| Calibrated confidence | Not measured, the threshold never abstains |
| Run-to-run variance | Not measured, one seed |
| Climbing performance | Not measured, out of scope |
