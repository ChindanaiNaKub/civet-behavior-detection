# Behavior dataset

This directory holds the source-clip ledger and CSV templates for the reviewed behavior-interval dataset. Raw footage stays in team-approved storage and is not copied into this repository.

## Files

- [`source-clips.csv`](source-clips.csv) lists the 22 audited source clips. `folder_label_weak` is the original folder name, not a verified behavior. `capture_datetime_overlay` and `camera_id_overlay` are transcribed from the sampled on-screen overlay; timezone, room, and recording session remain unknown.
- [`annotations.csv`](annotations.csv) is the interval-level annotation table. Add one row per annotator and interval. `behavior` accepts Eating, Locomotion, Resting, Climbing, Transition, or Uncertain; Transition and Uncertain are recorded as separate interval labels and are not model classes. Annotators must work independently on the shared sample before discussing it. Use the boundary rules in [`../docs/ANNOTATION-GUIDE.md`](../docs/ANNOTATION-GUIDE.md).
- [`agreement.csv`](agreement.csv) records shared-sample review, disagreements, and their resolution. Record the sample denominator so the reviewed percentage can be checked.
- [`splits.csv`](splits.csv) is the source-clip split ledger. Assign whole original clips before generating windows. Keep known recording sessions and camera groups together where possible; record the grouping evidence and who locked the test assignment. Current clips are marked `unassigned` with the reason in this ledger because the available groups cannot support all three splits.

## Dataset state

The source ledger is an inventory, not a labeled dataset. No behavior intervals, annotator agreement, or adjudications have been entered yet. The footage audit found no verified Climbing interval in the supplied footage; this does not establish that additional footage cannot provide one. Do not treat an empty field as a negative label or treat a folder name as ground truth.

The current source ledger has 11 clips from Camera 02 and 11 from Camera 03, with no recording-session IDs. If each known camera must stay intact as a group, these two groups cannot populate three camera-held-out splits. The split ledger marks all current clips unassigned and records this limitation; no test set is locked. Obtain session metadata or additional independent camera groups before making a split. Behavior labels are also unverified, so held-out class coverage cannot yet be assessed. Record the eventual coverage and any remaining limitation in the split ledger and project report.

Lighting, visibility, camera view, and occlusion vary by interval. Record them in `annotations.csv` only when observed; use `unknown` when evidence is unavailable. Do not infer room, camera, or session from filenames or folder names.
