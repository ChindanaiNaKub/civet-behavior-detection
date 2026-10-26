# Tasks 2 and 3: detection and silhouette features

The professor confirmed on 2026-10-26 that Phase 2 requires all three tasks of the
brief. [ADR 0009](adr/0009-build-tasks-2-and-3-on-auto-labels.md) records the
scope change and the two substitutions. This page records what was built and what
it measured. Task 1 is unchanged and lives in [the evaluation report](EVALUATION-REPORT.md).

## Task 2: YOLO-based behavior detection

The brief asks for Video to Frame Extraction to YOLO Annotation to YOLO Training
to Behavior Detection. The blocker is the annotation step: no bounding boxes
exist, and no person was available to draw them.

### What failed

| Method | Result |
|---|---|
| Pretrained COCO `yolo11n` on the full frame | Zero detections of the animal. One `potted plant` at 0.344 |
| Frame differencing, largest blob | Locks onto swaying foliage |
| Grounding DINO, open vocabulary, prompt "a civet" | 40+ overlapping boxes at threshold 0.12, top score 0.535 on empty floor |

### What worked

The cameras never move, so each clip supplies its own background. `scripts/detect.py`
takes the per-pixel median frame, measures each pixel's noise, thresholds the
difference, keeps components in a plausible animal size band, and keeps only boxes
that persist for about a second. Swaying foliage appears and vanishes; the animal
stays.

| Step | Count |
|---|---:|
| Clips | 22 |
| Frames sampled at 1 fps | 1061 |
| Candidate boxes | 470 |
| Boxes kept after persistence | 408 |

**Visual audit.** I reviewed 116 of the 408 boxes on contact sheets. 113 contained
the animal. 3 were empty floor, all in the last twelve seconds of
`25690519193547292` after the animal left the view.

### Training

`scripts/train_detector.py` trains `yolo11n` at 640 on 11 train clips, keeps the
3 validation clips for model selection, and holds out the 8 gold clips. Round 2
then lets round 1 label every sampled frame of the train and validation clips,
adds the frames the seeds missed, and retrains.

| Round | Labels (train / val) | Epochs | Wall time | Val mAP50 | Val mAP50-95 |
|---|---|---:|---:|---:|---:|
| 1, seed labels | 255 / 54 | 41 | 2.7 min | 0.537 | 0.284 |
| 2, self-trained | 308 / 115 | 39 | 3.4 min | 0.774 | 0.561 |
| 2, scored on untouched seed val | - | - | - | 0.499 | 0.242 |

The comparable row is the last one, because round 2's own validation labels were
partly written by round 1. Self-training widened coverage and removed the three
known floor false positives, but it did not improve mAP on the untouched seed
labels.

### Held-out behavior

Round 2 fired on **138 of 235** sampled frames of the 8 gold clips (59 percent) at
confidence 0.25 or above. A visual audit of 20 annotated held-out frames found 10
with a box, of which **5 were on the animal**. The misses and false positives are
concentrated on the two clips from 2026-05-20, a recording day absent from
training, where the detector fires on a bright plant.

**Conclusion.** The auto-labeling pipeline works and needs no human boxes. The
detector generalizes across cameras but not across unseen recording days, so the
demo falls back to the per-clip background model when the detector is unsure, and
the slide says so.

## Task 3: behavior analysis from silhouette features

The brief asks for Pose Estimation to Keypoints to Feature Extraction to Behavior
Classification, with Body Angle, Head Angle, Movement Speed and Limb Distance.
There is no civet pose model and no keypoint annotation. `scripts/features.py`
keeps the four feature names and measures the animal's silhouette instead. See
[the feature definitions](FEATURES.md).

### Result

| Set | Windows | Macro F1 | Majority-class macro F1 | Accuracy |
|---|---:|---:|---:|---:|
| Validation | 30 | 0.479 | 0.259 | 0.267 |
| Test (locked) | 15 | 0.422 | 0.190 | 0.267 |

Per-class F1 on the test windows: Eating 0.267, Locomotion 1.000, Resting 0.000.

### Why it is weak

The median background absorbs a still animal. 58 of 93 train windows, 23 of 30
validation windows and 7 of 15 test windows have no silhouette at all, and their
features are train medians. The classifier is effectively an Eating/Locomotion
detector, and Resting is invisible to it. The locomotion signal is strong because
speed separates it by an order of magnitude (134 px/s against 17 and 2).

**Conclusion.** Task 3 runs end to end and reports an honest number. It is a proxy,
not keypoint analysis, and the deck must say so before showing the score.

## One command

```
python3 scripts/demo_all.py data/raw/Eating/25690519193422801.mp4
```

prints Task 1's behavior per window, Task 2's detector box and confidence plus the
crop classification, and Task 3's four features, for the same windows. On that
clip it detects the animal in 10 of 10 windows, 6 by the trained detector and 4 by
the per-clip model.
