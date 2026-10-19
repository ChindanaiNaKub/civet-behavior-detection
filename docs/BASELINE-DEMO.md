# Baseline training and clip demo (issue #5)

This records one real run of the classifier and the timeline demo. It does not
decode or score any test video. The test lock belongs to issue #4. The held-out
evaluation is issue #6.

## What was trained

The model is the frame CNN from ADR 0004. It uses an ImageNet-pretrained
ResNet-18 backbone, eight frames per five-second window at 128 by 128, and mean
and max pooling before the three-class head. Only the 3,075-parameter head
trains. The backbone is frozen and its BatchNorm statistics stay fixed, which is
the split ADR 0004 describes.

Training selects only the train split and early-stops on the validation split.

| Item | Value |
|---|---|
| Train windows | 93 (Eating 14, Locomotion 5, Resting 74) |
| Validation windows | 30 (Eating 10, Locomotion 1, Resting 19) |
| Test windows | 18, never decoded or scored |
| Trainable parameters | 3,075 |
| Requested epochs | 40 |
| Best epoch | 4 |
| Best validation macro F1 | 1.000 |
| Checkpoint | `outputs/model/model.pt`, ignored by Git |

The training loss fell from 2.0564 at epoch 1 to 0.1212 at epoch 14. Validation
macro F1 reached 1.000 at epoch 4 and stayed there, so training stopped at epoch
14 after ten epochs without a gain. The best checkpoint is epoch 4.

The validation split is perfectly separated. That score says the split is easy,
not that the classifier generalizes. The test set is the harder check.

## Threshold chosen on validation

The threshold is chosen on the 30 validation windows. No test video is decoded
and no test window is scored.

```
python3 scripts/evaluate.py --validation-only
```

This writes `outputs/validation.md` and `outputs/threshold.json`. The chosen
threshold was 0.00 with validation macro F1 1.000 and a 0.0 percent abstain rate.
A threshold of 0.00 is the honest result of a validation split the model already
separates perfectly. It also means the demo has no Uncertain windows.

## Demo

The demo clip is `data/raw/Eating/25690519193547292.mp4` from the validation
split. It is not a training clip and not a test clip. The threshold comes from
`outputs/threshold.json`, not a hard-coded default.

```
python3 scripts/demo.py data/raw/Eating/25690519193547292.mp4 \
  --evaluation outputs/threshold.json
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

The header carries the source clip. Every row carries the five-second time
range, the behavior or Uncertain result, and the confidence. The last four
seconds are a shorter tail and are dropped, because every model input is exactly
five seconds.

## Limits to carry into issue #6

- Validation has one Locomotion window. Its perfect score is not a stable
  estimate for that class.
- Resting windows are daylight on Camera 03, most Eating windows are daylight on
  Camera 02, and Locomotion is the only infrared class. Class and recording still
  travel together, so a high score can come from recognising the recording.
- The validation-selected threshold is 0.00, so the demo cannot show an
  Uncertain result. Do not read that as calibrated confidence.
- No test video was decoded and no test window was scored. There is no held-out
  number yet, and the test set is not locked.

## Reproduce

From the repository root, after installing `requirements.txt`:

```
python3 scripts/windows.py
python3 scripts/train.py --out outputs/model/model.pt
python3 scripts/evaluate.py --validation-only
python3 scripts/demo.py data/raw/Eating/25690519193547292.mp4 \
  --evaluation outputs/threshold.json
```

The split ledger is tracked in `data/splits.csv`, so these steps do not need
`scripts/split.py`. Re-run `scripts/split.py` only before the test is locked,
because it rewrites the ledger and clears `test_locked_by` and `test_locked_at`.
After the lock, skip it or restore the lock fields by hand.

The full `python3 scripts/evaluate.py` run is a manual gate on the test lock.
The script does not enforce the lock, so do not run it until a person fills
`test_locked_by` in `data/splits.csv`.

Check the safety property and the rest of the pipeline logic with:

```
python3 -m pytest scripts
```
