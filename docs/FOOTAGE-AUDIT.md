# Footage and compute audit

**Audit date:** 2026-10-06  
**Purpose:** Establish what the supplied footage supports before annotation or model selection.

## Source inventory

The supplied source is `วิดิโอชะมดทำai-20261004T133418Z-1-001.zip`, expanded as `วิดิโอชะมดทำai/<folder>/<filename>.mp4` in the project workspace. The archive has 22 entries, and all 22 match the extracted files byte-for-byte. The extracted files have 22 distinct SHA-256 hashes. The archive is not copied into this repository; keep raw footage in team-approved storage as described in the README.

Durations and stream properties below come from `ffprobe` on the extracted files. Folder is the existing folder-level label only; it is not a verified interval annotation.

| Source file | Existing folder label | Duration (s) | Format |
|---|---:|---:|---|
| `Eating/25690518092326646.mp4` | Eating | 80.721 | 1920×1080, 25 fps, HEVC |
| `Eating/25690518092408866.mp4` | Eating | 8.800 | 1920×1080, 25 fps, HEVC |
| `Eating/25690519193422801.mp4` | Eating | 50.481 | 1920×1080, 25 fps, HEVC |
| `Eating/25690519193547292.mp4` | Eating | 68.840 | 1920×1080, 25 fps, HEVC |
| `Eating/25690519193808204.mp4` | Eating | 45.599 | 1920×1080, 25 fps, HEVC |
| `Locomotion/25690518090944462.mp4` | Locomotion | 11.080 | 1920×1080, 25 fps, HEVC |
| `Locomotion/25690518091012230.mp4` | Locomotion | 8.000 | 1920×1080, 25 fps, HEVC |
| `Locomotion/25690518091030781.mp4` | Locomotion | 10.239 | 1920×1080, 25 fps, HEVC |
| `Locomotion/25690518091047366.mp4` | Locomotion | 9.639 | 1920×1080, 25 fps, HEVC |
| `Locomotion/25690518091210873.mp4` | Locomotion | 14.400 | 1920×1080, 25 fps, HEVC |
| `Locomotion/25690518091711167.mp4` | Locomotion | 10.680 | 1920×1080, 25 fps, HEVC |
| `Locomotion/25690518091847591.mp4` | Locomotion | 12.840 | 1920×1080, 25 fps, HEVC |
| `Locomotion/25690518091918610.mp4` | Locomotion | 22.281 | 1920×1080, 25 fps, HEVC |
| `Locomotion/25690518091948946.mp4` | Locomotion | 10.680 | 1920×1080, 25 fps, HEVC |
| `Locomotion/25690520153014458.mp4` | Locomotion | 28.880 | 1920×1080, 25 fps, HEVC |
| `Resting/25690519194109670.mp4` | Resting | 131.518 | 1920×1080, 25 fps, HEVC |
| `Resting/25690519194200758.mp4` | Resting | 31.319 | 1920×1080, 25 fps, HEVC |
| `Resting/25690519194418953.mp4` | Resting | 91.718 | 1920×1080, 25 fps, HEVC |
| `Resting/25690519194809256.mp4` | Resting | 219.916 | 1920×1080, 25 fps, HEVC |
| `Resting/25690519195143424.mp4` | Resting | 122.237 | 1920×1080, 25 fps, HEVC |
| `Resting/25690520152850102.mp4` | Resting | 30.881 | 1920×1080, 25 fps, HEVC |
| `Resting/25690520153258104.mp4` | Resting | 31.081 | 1920×1080, 25 fps, HEVC |

**Totals:** 22 clips, 1,051.830 seconds (17 minutes 31.830 seconds): 5 Eating-folder clips, 10 Locomotion-folder clips, and 7 Resting-folder clips. No source file is in a Climbing folder.

## What is verified, unknown, and unavailable

| Claim or metadata | Status | Evidence and limit |
|---|---|---|
| Source path and supplied file count | **Verified** | The ZIP contains 22 MP4 files under the three named folders; each extracted file matches its archive entry byte-for-byte. |
| Duration, dimensions, frame rate, codec | **Verified** | Read from each MP4 stream with `ffprobe`; every clip is 1920×1080, 25 fps, HEVC. |
| Folder-level label | **Verified as a folder name only** | Counts are Eating 5, Locomotion 10, Resting 7. These names do not establish the behavior at every timecode. |
| Capture date/time | **Unknown** | File names are numeric and timestamp-like, but no source key or export documentation confirms their meaning or timezone. Do not parse them as capture times yet. |
| Recording session, camera ID, room | **Unavailable in supplied metadata** | No mapping file is present, and MP4 stream tags contain no session, camera, or room fields. The four-room/four-camera description comes from the project brief and does not verify clip coverage. |
| Light, camera view, visibility, occlusion | **Not inventoried** | These may be observable at specific timecodes, but no complete, reviewed condition annotations exist. Record them per interval rather than extrapolating from examples. |
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
