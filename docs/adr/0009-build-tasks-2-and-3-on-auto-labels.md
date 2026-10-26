# Build Tasks 2 and 3 on auto-generated labels

On 2026-10-26 the professor confirmed that Phase 2 requires all three tasks of the
brief, not Task 1 alone. Task 2 is YOLO-based behavior detection and Task 3 is
behavior analysis from keypoints. This ADR brings both into scope, and records how
each one is delivered without any human-drawn annotation.

It supersedes the rejection in the last consequence of
[ADR 0004](0004-train-a-cnn-over-window-frames.md), and the assumption in
[ADR 0001](0001-classify-short-behavior-clips.md) that the YOLO option needs
bounding-box annotations that do not exist here.

## Why the scope changes

The submission is graded on three tasks. ADR 0004 rejected Tasks 2 and 3 on the
grounds that Task 2 needs boxes and Task 3 needs pose estimation. Both objections
are about missing labels, not about missing compute. The footage is fixed-camera,
single-animal, 1080p, and the animal is the only object that translates, so the
labels can be generated rather than drawn.

## What was measured before choosing this

Four candidate sources of boxes were tried on the same footage.

| Method | Result |
|---|---|
| COCO `yolo11n` on the full frame | Zero detections of the animal. One `potted plant` at 0.344 |
| Frame differencing, largest blob | Box on swaying foliage, not the animal (`[0, 90, 232, 291]` at half resolution) |
| Grounding DINO, open vocabulary, prompt "a civet" | 40+ overlapping boxes at threshold 0.12, top score 0.535 on empty floor. Not usable |
| Per-pixel median background with per-pixel MAD threshold | 470 candidates, 408 kept by a one-second persistence filter, across 1061 sampled frames of all 22 clips |

A visual audit of 116 of the 408 seed boxes found 113 containing the animal and 3
empty-floor false positives, all three in the last twelve seconds of
`25690519193547292` where foliage keeps moving after the animal leaves the view.

## Decision

**Task 2** is a self-training loop. `scripts/detect.py` generates seed boxes from
the per-pixel background model and keeps only boxes that persist for about a
second. `scripts/train_detector.py` trains `yolo11n` on those, runs the detector
back over the footage, keeps confident boxes, and retrains. The contaminated seed
boxes are the reason for the second round: the model sees the same frames the
classical stage distorted, and a box on empty floor scores low.

**Task 3** replaces pose estimation with silhouette geometry and keeps the four
feature names the brief asks for. Body Angle is the principal axis of the animal's
silhouette. Head Angle is the axis of the narrow end. Movement Speed is centroid
displacement per second, normalised by box width. Limb Distance is the extent
across the body axis. The substitution is stated on the slide and in
`docs/FEATURES.md`; it is not presented as a keypoint model.

**Task 1** is unchanged.

## Consequences

- The detector has no human-drawn ground truth, so it has no mAP against truth.
  What gets reported is the held-out detection rate plus a visual audit with the
  number of frames checked and the number wrong. Stating a mAP against its own
  seed labels would be circular.
- Detector quality is bounded by the seed. Where the animal is still for the whole
  clip, as in `25690519194200758`, the background absorbs it and no seed box
  exists. The self-training round is what can recover those frames, and if it does
  not, the limit is reported rather than hidden.
- Task 3 is a proxy. A grader who wants real keypoints will not get them, and the
  slide must say so before the numbers.
- `scripts/labels.py` keeps three model classes for behavior. The detector's one
  class is `civet`, which is an object class and not a behavior class, so the two
  class lists stay separate.
- Every generated box, label file and weight is untracked. They are rebuilt by the
  two scripts above.
