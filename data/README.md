# Behavior dataset

This directory holds the source-clip ledger and CSV templates for the reviewed behavior-interval dataset. Raw footage stays in team-approved storage and is not copied into this repository.

## Files

- [`source-clips.csv`](source-clips.csv) lists the 22 audited source clips. `folder_label_weak` is the original folder name, not a verified behavior. `capture_datetime_overlay` and `camera_id_overlay` are transcribed from the sampled on-screen overlay; timezone, room, and recording session remain unknown.
- [`annotations.csv`](annotations.csv) is the interval-level annotation table. Add one row per annotator and interval. Annotators must work independently on the shared sample before discussing it. Use the labels and boundary rules in [`../docs/ANNOTATION-GUIDE.md`](../docs/ANNOTATION-GUIDE.md).
- [`agreement.csv`](agreement.csv) records shared-sample review, disagreements, and their resolution. Record the sample denominator so the reviewed percentage can be checked.
- [`splits.csv`](splits.csv) is the source-clip split ledger. Assign whole original clips before generating windows. Keep known recording sessions together; record the grouping evidence and who locked the test assignment.

## Dataset state

The source ledger is an inventory, not a labeled dataset. No behavior intervals, annotator agreement, adjudications, or split assignments have been entered yet. In particular, the supplied footage has no verified Climbing interval. Do not treat an empty field as a negative label or treat a folder name as ground truth.

Before assigning splits, confirm whether source/session metadata can be obtained. Camera overlays identify views but the available clips do not establish recording-session boundaries. If the available grouping evidence cannot support independent train, validation, and locked test sets, document that limitation in the split ledger and project report rather than implying independence.

Lighting, visibility, camera view, and occlusion vary by interval. Record them in `annotations.csv` only when observed; use `unknown` when evidence is unavailable. Do not infer room, camera, or session from filenames or folder names.
