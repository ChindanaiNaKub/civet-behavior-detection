# The four Task 3 features

The brief asks for behavior analysis from keypoints. There is no civet pose model
and no keypoint annotation for this footage, so per
[ADR 0009](adr/0009-build-tasks-2-and-3-on-auto-labels.md) the pipeline
measures the animal's **silhouette** instead and keeps the four names. Nothing here
is a keypoint, a joint, or a head detector.

The silhouette is the background-model mask from `scripts/detect.py`, cropped to the
kept box. `scripts/features.py` writes one row per five-second labelled window, with
each feature the **median over the sampled frames** in that window.

| Feature | What is measured | Unit |
|---|---|---|
| Body angle | Principal axis of the whole silhouette (image moments) | degrees, 0-180 |
| Head angle | Principal axis of the *narrower* half of the silhouette, split along the body axis | degrees, 0-180 |
| Movement speed | Silhouette centroid displacement between samples / time | full-res px/s |
| Limb distance | Extent of the silhouette across the body axis | full-res px |

`box_area_px` is the median full-resolution box area, kept as context. `n_frames` is
how many samples contributed; a window with `n_frames` 0 had no silhouette and its
features are left empty in the CSV.

## What a reader must not conclude

- **Not keypoints.** There is no pose model, no joint, and no verification against
  any head, limb or keypoint annotation, because none exists.
- Head angle is a **proxy** and is wrong when the animal is curled, when the tail is
  the narrow end, or when the view is foreshortened.
- An angle is an axis: 0° and 180° are the same orientation, so direction of travel
  is not known from it.
- Speed is pixel motion in 2D, not real-world speed; it changes with distance from
  the camera and with partial occlusion.
- A still animal becomes part of the median background. Most Resting windows have no
  silhouette, so Resting features are largely missing rather than measured as zero.
- The kept box is the largest *moving* component, so when part of the animal is
  static against the background the geometry describes only its moving end.
- These features cannot support claims about social behavior, posture, or any
  behavior the four names do not cover.
