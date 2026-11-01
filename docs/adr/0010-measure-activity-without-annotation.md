# Measure activity on the banded civet footage without annotation

The second submission adds 26 hours of banded civet footage from four fixed
cameras with no annotation. This ADR records what the system measures on that
footage, how it is labelled, and how the trained model is scored.

## Why the Phase 2 pipeline does not carry over

The classifier in [ADR 0004](0004-train-a-cnn-over-window-frames.md) is trained on
`data/annotations.csv`, which a person verified interval by interval. The new
footage has no annotation and no person to make one, and its clips are five to
eight hours long rather than five to eighty seconds.

Two consequences follow. The three behaviour classes cannot be scored on this
footage, because nothing says when the animal was eating. And the per-clip median
background that `scripts/detect.py` builds from a short clip has to be built from
the whole recording instead: an adaptive background absorbs a still animal within
a minute, which is exactly the case an activity measurement has to see.

## Decision

**The label is a measurement, not an annotation.** `scripts/activity.py` samples
each camera file once per second at quarter resolution, takes the per-pixel median
of the whole clip as the empty enclosure, and keeps a box only when a neighbouring
second corroborates it. A five-second window is:

- **Absent** when fewer than three of its five samples hold a box;
- **Active** when the animal is present and the median speed is at least 0.05 body
  widths per second, or the silhouette churns by 0.20 or more in a second;
- **Inactive** when it is present and neither holds.

Absent is a state of the table and never a training class, because an empty
enclosure is not behaviour.

**The split is by camera.** Cameras 1 and 2 train, camera 3 selects the epoch and
the abstain threshold, camera 4 is held out. This is the same rule as
[ADR 0005](0005-split-by-clip-not-by-recording-group.md) one level up: a whole
viewpoint is held out, so the score cannot come from recognising the view. Only
one recording night exists, so a recording-day holdout is not available.

**The model is unchanged.** `scripts/train.py` trains the same ImageNet ResNet-18
frame CNN with mean and max pooling that Task 1 uses, on Active and Inactive
windows. `scripts/labels.py` gains `ACTIVITY_CLASSES` so the two activity labels
are accepted without entering the behaviour taxonomy.

**The score is agreement with the measurement.** The held-out camera is scored
against the labels the measurement produced. The teacher is a motion rule, not a
person, so the number is a distillation check: it says whether the CNN reproduces
the measurement from appearance on a view it has not seen, and nothing more. It is
reported beside a visual audit of sampled windows, as the Task 2 detector was in
[ADR 0009](0009-build-tasks-2-and-3-on-auto-labels.md).

## Consequences

- Active and Inactive are not Eating, Locomotion and Resting. They are not
  comparable with the Phase 2 test macro F1 of 0.949 and do not replace it.
- The measurement cannot see an animal that is still for most of the recording,
  because the median would absorb it. It cannot tell a still civet from a rock:
  both are a box that does not move, so only the visual audit separates them, and it samples.
- A box on a swaying plant survives persistence. The area band and the audit are
  the only defences, and the audit is what reports the failure rate.
- The model assumes the animal is in view, because it is trained on present
  windows only. Presence stays the measurement's job; the model is not a detector.
- One night, one enclosure. Nothing here estimates a second night.
