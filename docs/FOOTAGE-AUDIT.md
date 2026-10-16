# Footage and compute audit

**Audit date:** 2026-10-06
**Purpose:** Establish what the supplied footage supports before annotation or model selection.

## Source inventory

The supplied source is `วิดิโอชะมดทำai-20261004T133418Z-1-001.zip`, expanded as `วิดิโอชะมดทำai/<folder>/<filename>.mp4` in the project workspace. The archive has 22 entries, and all 22 match the extracted files byte-for-byte. The extracted files have 22 distinct SHA-256 hashes. The archive is not copied into this repository; keep raw footage in team-approved storage as described in the README.

Durations and stream properties below come from `ffprobe` on the extracted files. The on-screen date/time and camera fields were read from a sampled frame around two seconds into each clip. They are burned-in overlays, not MP4 metadata tags. The timestamp timezone and the meaning of any recording session are not documented. Folder is the existing folder-level label only; it is not a verified interval annotation.

| Source file | Existing folder label | Duration (s) | On-screen date/time | On-screen camera | Format |
|---|---:|---:|---|---|---|
| `Eating/25690518092326646.mp4` | Eating | 80.721 | 2026-05-13 15:28:34 | Camera 02 | 1920×1080, 25 fps, HEVC |
| `Eating/25690518092408866.mp4` | Eating | 8.800 | 2026-05-13 17:25:15 | Camera 02 | 1920×1080, 25 fps, HEVC |
| `Eating/25690519193422801.mp4` | Eating | 50.481 | 2026-05-19 15:51:39 | Camera 02 | 1920×1080, 25 fps, HEVC |
| `Eating/25690519193547292.mp4` | Eating | 68.840 | 2026-05-19 15:59:55 | Camera 02 | 1920×1080, 25 fps, HEVC |
| `Eating/25690519193808204.mp4` | Eating | 45.599 | 2026-05-19 16:39:16 | Camera 03 | 1920×1080, 25 fps, HEVC |
| `Locomotion/25690518090944462.mp4` | Locomotion | 11.080 | 2026-05-18 00:00:02 | Camera 03 | 1920×1080, 25 fps, HEVC |
| `Locomotion/25690518091012230.mp4` | Locomotion | 8.000 | 2026-05-18 00:00:33 | Camera 03 | 1920×1080, 25 fps, HEVC |
| `Locomotion/25690518091030781.mp4` | Locomotion | 10.239 | 2026-05-18 00:00:50 | Camera 03 | 1920×1080, 25 fps, HEVC |
| `Locomotion/25690518091047366.mp4` | Locomotion | 9.639 | 2026-05-18 00:08:54 | Camera 03 | 1920×1080, 25 fps, HEVC |
| `Locomotion/25690518091210873.mp4` | Locomotion | 14.400 | 2026-05-18 01:19:56 | Camera 03 | 1920×1080, 25 fps, HEVC |
| `Locomotion/25690518091711167.mp4` | Locomotion | 10.680 | 2026-05-18 00:00:08 | Camera 02 | 1920×1080, 25 fps, HEVC |
| `Locomotion/25690518091847591.mp4` | Locomotion | 12.840 | 2026-05-18 00:33:09 | Camera 02 | 1920×1080, 25 fps, HEVC |
| `Locomotion/25690518091918610.mp4` | Locomotion | 22.281 | 2026-05-18 00:33:31 | Camera 02 | 1920×1080, 25 fps, HEVC |
| `Locomotion/25690518091948946.mp4` | Locomotion | 10.680 | 2026-05-18 01:51:46 | Camera 02 | 1920×1080, 25 fps, HEVC |
| `Locomotion/25690520153014458.mp4` | Locomotion | 28.880 | 2026-05-18 05:19:41 | Camera 02 | 1920×1080, 25 fps, HEVC |
| `Resting/25690519194109670.mp4` | Resting | 131.518 | 2026-05-19 16:43:15 | Camera 03 | 1920×1080, 25 fps, HEVC |
| `Resting/25690519194200758.mp4` | Resting | 31.319 | 2026-05-19 16:46:39 | Camera 03 | 1920×1080, 25 fps, HEVC |
| `Resting/25690519194418953.mp4` | Resting | 91.718 | 2026-05-19 17:00:09 | Camera 03 | 1920×1080, 25 fps, HEVC |
| `Resting/25690519194809256.mp4` | Resting | 219.916 | 2026-05-19 17:09:23 | Camera 03 | 1920×1080, 25 fps, HEVC |
| `Resting/25690519195143424.mp4` | Resting | 122.237 | 2026-05-19 17:34:29 | Camera 03 | 1920×1080, 25 fps, HEVC |
| `Resting/25690520152850102.mp4` | Resting | 30.881 | 2026-05-19 04:48:16 | Camera 02 | 1920×1080, 25 fps, HEVC |
| `Resting/25690520153258104.mp4` | Resting | 31.081 | 2026-05-18 05:24:16 | Camera 02 | 1920×1080, 25 fps, HEVC |

**Totals:** 22 clips, 1,051.830 seconds (17 minutes 31.830 seconds): 5 Eating-folder clips, 10 Locomotion-folder clips, and 7 Resting-folder clips. No source file is in a Climbing folder.

## What is verified, unknown, and unavailable

| Claim or metadata | Status | Evidence and limit |
|---|---|---|
| Source path and supplied file count | **Verified** | The ZIP contains 22 MP4 files under the three named folders; each extracted file matches its archive entry byte-for-byte. |
| Duration, dimensions, frame rate, codec | **Verified** | Read from each MP4 stream with `ffprobe`; every clip is 1920×1080, 25 fps, HEVC. |
| Folder-level label | **Verified as a folder name only** | Counts are Eating 5, Locomotion 10, Resting 7. These names do not establish the behavior at every timecode. |
| On-screen date/time | **Visible; interpretation partly unknown** | A date/time is burned into every sampled clip frame and is listed in the inventory. The timezone and clock accuracy are not documented; filenames do not consistently establish its meaning. |
| On-screen camera ID | **Verified for sampled frames** | Camera 02 and Camera 03 overlays appear in this set, 11 clips each, matching the `camera_id_overlay` column of [`../data/source-clips.csv`](../data/source-clips.csv). No Camera 01 or Camera 04 overlay was found. These labels identify the recorded view only; no room mapping is supplied. |
| Recording session and room | **Unknown / unavailable** | No session ID or room mapping file is present. Do not infer four-room/four-camera coverage from the project brief. The visible IDs establish two camera labels in these files, not which rooms they cover or whether other footage exists. |
| Light, camera view, visibility, occlusion | **Not inventoried** | These may be observable at specific timecodes, but no complete, reviewed condition annotations exist. Record them per interval rather than extrapolating from examples. |
| Behavior examples | **Screening candidates found for three classes; intervals unverified** | In sampled frames, Eating is visible at `Eating/25690518092326646.mp4` (45 s), Locomotion at `Locomotion/25690518091918610.mp4` (8 s), and a resting posture at `Resting/25690519194809256.mp4` (20 s). These are examples to review, not verified time intervals; use the annotation guide before counting them. |
| Climbing coverage | **No verified example found; true coverage remains unknown** | No Climbing folder label exists. A visual scan sampled frames every five seconds across all 22 clips and every two seconds from all 10 Locomotion-folder clips; no sampled frame showed an unambiguous climb. This is a coverage check, not a full behavior annotation, so it cannot prove climbing is absent. |
| Additional examples in the explainer project | **Not independent sources** | Its script identifies Eating, Locomotion, and Resting examples as illustrative footage from this supplied set. Do not add them to the source-clip counts without tracing them to original files. |

The current evidence supports three **candidate** classes from folder names, not verified labeled intervals. Climbing is the highest-priority source gap. Review full clips against the annotation guide before counting any example toward a class.

## Local compute check

The workstation available during this audit reports an Intel Core i5-12500H CPU (12 cores / 16 logical CPUs), 15.3 GiB system memory, and an NVIDIA GeForce RTX 3050 Laptop GPU with 4 GiB VRAM. This is a check of one machine, not confirmation that every team member has the same hardware or that a shared GPU/cloud resource is available.

**Initial model target:** use a compact pretrained video classifier and confirm the exact variant with a short training/inference smoke run. Start with a small batch and reduced frame count/resolution; use mixed precision if the chosen framework and GPU support it. Do not train from scratch or choose a larger model until its memory use is measured on this machine. No model has been trained or evaluated as part of this audit.

## Initial source-count targets

Set an initial target of **20 manually verified source clips per behavior**. Keep at least 5 whole source clips per class for a locked test set and use the other 15 for development (initially 12 train and 3 validation). These are pilot planning counts, not a claim of statistical adequacy. Count only clips with reviewed intervals for the class. Where capture-session metadata becomes available, split by session; if fewer than five independent sessions support a class, report that limit and use group-aware cross-validation for development instead of implying an independent test estimate.

The table below compares the target to the current **folder-level candidates**. It assumes, only for planning, that every candidate survives manual interval review; the review may reduce those counts.

| Behavior | Current folder-level candidates | Initial target | Additional clips needed if all candidates verify |
|---|---:|---:|---:|
| Eating | 5 | 20 | 15 |
| Locomotion | 10 | 20 | 10 |
| Resting | 7 | 20 | 13 |
| Climbing | 0 | 20 | 20 |

Do not count two segments from one original clip as two source clips. Freeze the held-out source clips before extracting overlapping windows.

## Revised source-count targets

The supplied archive is the whole dataset rather than a sample of a larger one, so the 20-clips-per-class target above cannot be met by collecting more of the same footage. The ceilings are the folder-level candidates themselves.

| Behavior | Ceiling | Initial target | Revised target |
|---|---:|---:|---|
| Eating | 5 clips, 254 s | 20 | every clip that review verifies |
| Locomotion | 10 clips, 139 s | 20 | every clip that review verifies |
| Resting | 7 clips, 659 s | 20 | every clip that review verifies |
| Climbing | 0 clips | 20 | none available |

Report the verified count for each class beside every metric, and state plainly that Climbing has no source clip. Do not present a four-class result. `scripts/split.py` writes the split for the three classes that exist and refuses to write one that leaves a class out of a split.
