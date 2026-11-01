#!/usr/bin/env python3
"""Presence and activity per fixed camera, for footage that has no annotation.

The Phase 2 classifier is trained on behaviour intervals a person verified. The
banded-civet footage added for the second submission is 26 hours of raw recording
from four fixed cameras with no annotation and no person available to make one,
so this module labels what a fixed camera can measure by itself: whether the
animal is in view, and whether it is moving.

The method is the background model from detect.py, unchanged in spirit: the
camera never moves, so the per-pixel median of the whole clip is the empty
enclosure, and anything that differs from it is the animal or a swaying plant. A
box that survives the size band is presence; the centroid moving between samples
is activity. A window is Active when the animal is present and moving, Inactive
when it is present and still, and Absent when it is not in view. A frame with a
global change in it, such as a dusk shadow or an infrared gain jump, is counted as
unmeasured instead of being read as an animal or as absence. Only Active and
Inactive are behaviour, and only they become training windows.

The median background is the reason a long recording is required rather than
preferred: a five-second clip cannot supply one, and an adaptive background
absorbs a still animal within a minute, which is exactly the case this module has
to measure. So the whole clip is decoded once to a cached greyscale stack, the
median is taken from it, and the stack is traced frame by frame. Thresholds are
re-applied from the cache, so tuning does not decode the footage again.

Every number this module writes is a measurement, not a ground truth. The report
script pairs it with a visual audit, and docs/ACTIVITY-REPORT.md says what the
measurement cannot see.

Usage:
    python3 scripts/activity.py --clips ../ชะมดแผงสันหางดำ
    python3 scripts/activity.py --clips ../ชะมดแผงสันหางดำ --retune
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Analysis is one sample per second at quarter resolution. 1080p quarter is
# 480 by 270, where the animal is 30 to 60 px wide: enough for a centroid and a
# box, small enough that 26 hours fit in a 12 GB cache.
SAMPLE_FPS = 1.0
SCALE = 0.25
ANALYSIS_WH = (480, 270)

# Background differencing, as in detect.py but at quarter resolution, so the area
# band is a quarter of detect.py's: the animal measures 400 to 4000 px here.
ABS_MIN = 12.0
K_MAD = 6.0
MIN_AREA = 700
MAX_AREA = 9000
MIN_SIDE = 10
ASPECT = (0.3, 3.5)
CLOSE_WH = (7, 7)
OPEN_WH = (5, 5)
# The floor of a night-time enclosure is sand and leaves, and the infrared gain
# drifts, so a patch of texture can appear where the median has none. That patch
# is sparse and a civet is solid, so the mask is median-filtered and a component
# must fill at least this share of its own box.
DENOISE_WH = 5
MIN_FILL = 0.35
# A frame where more than a fortieth of the view changed has a global change in it:
# a dusk shadow, an exposure step, an infrared gain jump. On this footage an animal
# covers at most two percent of the view and such a frame covers four to twenty
# four, so the gate separates them. Those frames are counted as unmeasured rather
# than read as an animal or as absence.
MAX_FG_FRAC = 0.04

# A box is presence only when a neighbouring second corroborates it, which is what
# removes a flying insect: it appears in one sample and is gone from the next.
PERSIST_WINDOW_S = 2.0
PERSIST_IOU = 0.1
PERSIST_CENTRE_PX = 40.0

# Two consecutive samples further apart than this many body widths are two
# different objects, so the step is a break in the track, not a speed.
JUMP_BODY = 3.0

# Activity rule, chosen from the measured distributions on camera 1 and 2 and
# then applied unchanged to every camera. See docs/ACTIVITY-REPORT.md.
ACTIVE_SPEED = 0.05      # body widths per second
ACTIVE_CHURN = 0.20      # share of the silhouette that changed in one second

WINDOW_S = 5.0
MIN_PRESENT = 3          # of the five samples a window holds
SPLITS = {"camera1": "train", "camera2": "train", "camera3": "validation", "camera4": "test"}

FRAME_FIELDS = ("clip_id", "camera", "time_s", "present", "x1", "y1", "x2", "y2", "area_px", "speed_bw_s", "churn")
WINDOW_FIELDS = ("clip_id", "start_s", "end_s", "behavior", "split")

Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class Row:
    """One sampled second: what the background model saw, and how it moved."""

    time_s: float
    present: bool
    box: Box | None
    area_px: int
    speed_bw_s: float
    churn: float


@dataclass(frozen=True)
class Clip:
    """One camera file and the split its windows belong to."""

    clip_id: str
    camera: str
    path: Path
    duration_s: float
    split: str


# --- pure detection ----------------------------------------------------------


def threshold_for(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Median background and the per-pixel noise threshold that goes with it.

    The median is taken over at most about two thousand frames spread through the
    clip. The animal is absent from most of a night's footage, so the median is
    the empty enclosure; a frame every ten or twenty seconds is far more than the
    median needs and keeps the pass inside a laptop's memory.

    The threshold is per pixel, as in detect.py: the wire mesh a metre from the
    lens flickers far more than the enclosure floor, and one scalar threshold
    either misses the animal on the floor or reports the mesh as an animal.
    """
    stride = max(8, len(frames) // 2000)
    sample = frames[::stride].astype(np.int16)
    background = np.median(sample, axis=0)
    noise = np.median(np.abs(sample - background), axis=0) + 1.0
    return background.astype(np.float32), np.maximum(ABS_MIN, K_MAD * noise).astype(np.float32)


def largest_box(mask: np.ndarray) -> tuple[Box | None, int]:
    """Largest component whose size and shape can be the animal, or None."""
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best: Box | None = None
    best_area = 0
    for x, y, width, height, area in stats[1:]:
        if area < MIN_AREA or area > MAX_AREA:
            continue
        if width < MIN_SIDE or height < MIN_SIDE:
            continue
        if not ASPECT[0] <= width / height <= ASPECT[1]:
            continue
        if area < MIN_FILL * width * height:
            continue
        if area > best_area:
            best_area = int(area)
            best = (int(x), int(y), int(x + width), int(y + height))
    return best, best_area


def _iou(a: Box, b: Box) -> float:
    """Intersection over union of two xyxy boxes."""
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    overlap = max(0, right - left) * max(0, bottom - top)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap
    return overlap / union if union else 0.0


def _centre_distance(a: Box, b: Box) -> float:
    (ax, ay), (bx, by) = _centre(a), _centre(b)
    return float(np.hypot(ax - bx, ay - by))


def _centre(box: Box) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def _body_width(box: Box) -> float:
    return max(1.0, float(max(box[2] - box[0], box[3] - box[1])))


def corroborated(index: int, times: list[float], boxes: list[Box | None]) -> bool:
    """Whether a neighbouring sample holds a box on the same object.

    Only neighbours inside PERSIST_WINDOW_S are compared, so the cost follows the
    number of boxes rather than its square.
    """
    box = boxes[index]
    if box is None:
        return False
    time_s = times[index]
    for other in (index - 1, index + 1, index - 2, index + 2):
        if not 0 <= other < len(times) or boxes[other] is None:
            continue
        if abs(times[other] - time_s) > PERSIST_WINDOW_S + 1e-6:
            continue
        if _iou(box, boxes[other]) >= PERSIST_IOU or _centre_distance(box, boxes[other]) <= PERSIST_CENTRE_PX:
            return True
    return False


def trace(frames: np.ndarray, background: np.ndarray, threshold: np.ndarray) -> tuple[list[Row], int]:
    """One row per sampled frame: presence, box, speed and churn, and a count of
    the frames a global change made unmeasurable.

    Three passes over the cached stack. The first proposes a box per frame and
    measures how much of the silhouette changed since the previous frame. The
    second drops boxes that no neighbouring second corroborates. The third
    measures speed between the boxes that survived, so a dropped neighbour cannot
    become a large speed.
    """
    close = cv2.getStructuringElement(cv2.MORPH_RECT, CLOSE_WH)
    open_ = cv2.getStructuringElement(cv2.MORPH_RECT, OPEN_WH)

    times = [index / SAMPLE_FPS for index in range(len(frames))]
    boxes: list[Box | None] = []
    areas: list[int] = []
    churns: list[float] = []
    previous_mask: np.ndarray | None = None
    unmeasured = 0
    for index in range(len(frames)):
        mask = (np.abs(frames[index].astype(np.int16) - background.astype(np.int16)) > threshold).astype(np.uint8)
        mask = cv2.medianBlur(mask, DENOISE_WH)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_)
        if mask.mean() > MAX_FG_FRAC:
            unmeasured += 1
            boxes.append(None)
            areas.append(0)
            churns.append(float("nan"))
            previous_mask = None
            continue
        box, area = largest_box(mask)
        churn = float("nan")
        if box is not None and previous_mask is not None:
            union = np.logical_or(previous_mask, mask).sum()
            if union:
                churn = float(np.logical_xor(previous_mask, mask).sum() / union)
        boxes.append(box)
        areas.append(area)
        churns.append(churn)
        previous_mask = mask if box is not None else None

    kept = [corroborated(index, times, boxes) for index in range(len(times))]
    rows: list[Row] = []
    previous: Row | None = None
    for index in range(len(times)):
        if not kept[index]:
            continue
        box = boxes[index]
        assert box is not None
        speed = float("nan")
        if previous is not None and previous.box is not None:
            gap = times[index] - previous.time_s
            distance = _centre_distance(previous.box, box)
            if 0 < gap <= 2.0 / SAMPLE_FPS + 1e-6 and distance <= JUMP_BODY * max(_body_width(previous.box), _body_width(box)):
                speed = distance / gap / _body_width(box)
        churn = churns[index] if index and kept[index - 1] else float("nan")
        row = Row(times[index], True, box, areas[index], speed, churn)
        rows.append(row)
        previous = row
    return rows, unmeasured


# --- windows -----------------------------------------------------------------


def grid(duration_s: float, window: float = WINDOW_S) -> list[tuple[float, float]]:
    """Non-overlapping windows of exactly ``window`` seconds; a short tail is dropped."""
    spans = []
    start = 0.0
    while start + window <= duration_s + 1e-6:
        spans.append((start, start + window))
        start += window
    return spans


def label_window(rows: list[Row]) -> str:
    """Absent, Inactive or Active for the samples of one window."""
    present = [row for row in rows if row.present]
    if len(present) < MIN_PRESENT:
        return "Absent"
    speeds = np.array([row.speed_bw_s for row in present if not np.isnan(row.speed_bw_s)])
    churns = np.array([row.churn for row in present if not np.isnan(row.churn)])
    if len(speeds) and float(np.median(speeds)) >= ACTIVE_SPEED:
        return "Active"
    if len(churns) and float(np.median(churns)) >= ACTIVE_CHURN:
        return "Active"
    return "Inactive"


def windows(rows: list[Row], duration_s: float) -> list[tuple[float, float, str]]:
    """(start, end, label) for every window the clip's duration allows.

    The rows are in time order and only present samples are in them, so one
    pointer walks the list once instead of rescanning it per window.
    """
    out = []
    index = 0
    for start, end in grid(duration_s):
        while index < len(rows) and rows[index].time_s < start:
            index += 1
        stop = index
        while stop < len(rows) and rows[stop].time_s < end:
            stop += 1
        out.append((start, end, label_window(rows[index:stop])))
    return out


def crop_boxes(windows, frames_csv: Path) -> list[tuple[float, float, float, float] | None]:
    """Fractional box of the animal for each window, or None where the scan had none.

    The classifier sees a 128 by 128 view. In a 1080p frame the animal is about
    60 px wide, which is 16 px after that resize, and 16 px cannot show whether a
    civet is standing or lying down. The scan already knows where the animal is, so
    each window is cropped to its box plus a margin before the resize.

    Boxes are returned as fractions of the frame, so the caller does not need to
    know the analysis resolution or the video's. The box is the one nearest the
    middle of the window, and only within two seconds of it.
    """
    samples: dict[str, list[tuple[float, tuple[float, float, float, float]]]] = {}
    with frames_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            width, height = ANALYSIS_WH
            box = (
                int(row["x1"]) / width, int(row["y1"]) / height,
                int(row["x2"]) / width, int(row["y2"]) / height,
            )
            samples.setdefault(row["clip_id"], []).append((float(row["time_s"]), box))

    out: list[tuple[float, float, float, float] | None] = []
    for window in windows:
        middle = (window.start + window.end) / 2.0
        best, distance = None, float("inf")
        for time_s, box in samples.get(window.clip_id, []):
            gap = abs(time_s - middle)
            if gap < distance:
                best, distance = box, gap
        out.append(best if distance <= 2.0 else None)
    return out


# --- clip I/O ----------------------------------------------------------------


def cache_clip(path: Path, cache: Path) -> np.ndarray:
    """Decode one clip to a greyscale stack, once, and return it as a memmap.

    ffmpeg decodes every frame and drops all but one per second, which is several
    times faster than seeking with OpenCV on a seven-hour file. The stack is
    memory-mapped because a whole night at quarter resolution is gigabytes.
    """
    width, height = ANALYSIS_WH
    cache.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-v", "error", "-i", str(path),
        "-vf", f"fps={SAMPLE_FPS},scale={width}:{height}",
        "-f", "rawvideo", "-pix_fmt", "gray", "-y", str(cache),
    ]
    subprocess.run(command, check=True)
    return open_cache(cache)


def open_cache(cache: Path) -> np.ndarray:
    """The cached stack of one clip, already on disk."""
    width, height = ANALYSIS_WH
    count = cache.stat().st_size // (width * height)
    if not count:
        raise RuntimeError(f"no frames in {cache}")
    return np.memmap(cache, dtype=np.uint8, mode="r", shape=(count, height, width))


def discover(clips_root: Path) -> list[Clip]:
    """Every camera file under the root, with its camera name and split."""
    found = []
    for path in sorted(clips_root.rglob("*.mp4")):
        camera = path.parent.name
        if camera not in SPLITS:
            raise SystemExit(f"{path}: folder {camera!r} is not one of {sorted(SPLITS)}")
        found.append(Clip(path.stem, camera, path, 0.0, SPLITS[camera]))
    if not found:
        raise SystemExit(f"no .mp4 files under {clips_root}")
    return found


# --- tables ------------------------------------------------------------------


def write_frames(path: Path, clips: list[Clip], traces: dict[str, list[Row]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(FRAME_FIELDS)
        for clip in clips:
            for row in traces[clip.clip_id]:
                box = row.box or (0, 0, 0, 0)
                writer.writerow(
                    [
                        clip.clip_id, clip.camera, f"{row.time_s:.2f}", int(row.present),
                        box[0], box[1], box[2], box[3], row.area_px,
                        f"{row.speed_bw_s:.4f}" if not np.isnan(row.speed_bw_s) else "",
                        f"{row.churn:.4f}" if not np.isnan(row.churn) else "",
                    ]
                )


def write_windows(path: Path, clips: list[Clip], labelled: dict[str, list[tuple[float, float, str]]]) -> None:
    """Only the behaviour windows: Absent carries no behaviour to learn."""
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(WINDOW_FIELDS)
        for clip in clips:
            for start, end, label in labelled[clip.clip_id]:
                if label == "Absent":
                    continue
                writer.writerow([clip.clip_id, f"{start:.2f}", f"{end:.2f}", label, clip.split])


def capture_datetime(clip_id: str) -> str:
    """The recording start, from the file name the camera gave it.

    ``A01_20260405235925`` is camera 1 starting 2026-04-05 23:59:25. The burned-in
    overlay agrees: camera 2's file starts 19:33:16 and its overlay reads 19:43:16
    ten minutes in. A name that does not carry a timestamp is left empty rather
    than guessed.
    """
    stamp = clip_id.rsplit("_", 1)[-1]
    if len(stamp) != 14 or not stamp.isdigit():
        return ""
    return f"{stamp[0:4]}-{stamp[4:6]}-{stamp[6:8]} {stamp[8:10]}:{stamp[10:12]}:{stamp[12:14]}"


def write_source_clips(path: Path, clips: list[Clip], clips_root: Path) -> None:
    """The ledger train.py and evaluate.py read to find each clip's frames."""
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["clip_id", "relative_path", "folder_label_weak", "duration_s",
             "capture_datetime_overlay", "camera_id_overlay", "session_id", "room_id"]
        )
        for clip in clips:
            writer.writerow(
                [clip.clip_id, str(clip.path.relative_to(clips_root)), clip.camera,
                 f"{clip.duration_s:.2f}", capture_datetime(clip.clip_id),
                 f"Camera 0{clip.camera[-1]}", "banded-civet-2026-04-05", ""]
            )


def summary(
    clips: list[Clip],
    traces: dict[str, list[Row]],
    labelled: dict[str, list[tuple[float, float, str]]],
    unmeasured: dict[str, int],
) -> str:
    lines = [
        "# Activity scan summary",
        "",
        f"One sample per second at {ANALYSIS_WH[0]} by {ANALYSIS_WH[1]}, "
        f"median background over the whole clip, activity rule speed {ACTIVE_SPEED} body widths/s "
        f"or churn {ACTIVE_CHURN}.",
        "",
        "Windows are five seconds. Unmeasured frames hold a global change (a dusk shadow or an "
        "infrared gain jump) and are counted as neither presence nor absence.",
        "",
        "| camera | clip | hours | unmeasured s | present s | Active | Inactive | Absent |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for clip in clips:
        counts = Counter(label for _, _, label in labelled[clip.clip_id])
        present = sum(1 for row in traces[clip.clip_id] if row.present)
        lines.append(
            f"| {clip.camera} | `{clip.clip_id}` | {clip.duration_s / 3600:.2f} | {unmeasured.get(clip.clip_id, 0)} | {present} | "
            f"{counts['Active']} | {counts['Inactive']} | {counts['Absent']} |"
        )
    return "\n".join(lines) + "\n"


# --- driver ------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clips", type=Path, required=True, help="folder of cameraN/*.mp4")
    parser.add_argument("--out", type=Path, default=Path("outputs/activity"))
    parser.add_argument("--retune", action="store_true", help="reuse the cached stacks and skip decoding")
    parser.add_argument("--clean", action="store_true", help="delete the cached stacks when done")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    cache_dir = args.out / "cache"
    clips = discover(args.clips)

    traces: dict[str, list[Row]] = {}
    labelled: dict[str, list[tuple[float, float, str]]] = {}
    unmeasured_s: dict[str, int] = {}
    resolved: list[Clip] = []
    for clip in clips:
        cache = cache_dir / f"{clip.clip_id}.gray"
        if args.retune:
            frames = open_cache(cache)
        else:
            print(f"decoding {clip.camera}/{clip.clip_id}", flush=True)
            frames = cache_clip(clip.path, cache)
        background, threshold = threshold_for(frames)
        rows, unmeasured = trace(frames, background, threshold)
        duration = len(frames) / SAMPLE_FPS
        unmeasured_s[clip.clip_id] = unmeasured
        resolved.append(Clip(clip.clip_id, clip.camera, clip.path, duration, clip.split))
        traces[clip.clip_id] = rows
        labelled[clip.clip_id] = windows(rows, duration)
        counts = Counter(label for _, _, label in labelled[clip.clip_id])
        print(
            f"  {clip.camera}/{clip.clip_id}: {duration / 3600:.2f} h, {len(rows)} samples, "
            f"unmeasured {unmeasured}, present {sum(r.present for r in rows)}, {dict(counts)}",
            flush=True,
        )
        del frames

    write_frames(args.out / "frames.csv", resolved, traces)
    write_windows(args.out / "windows.csv", resolved, labelled)
    write_source_clips(args.out / "source-clips.csv", resolved, args.clips)
    (args.out / "summary.md").write_text(summary(resolved, traces, labelled, unmeasured_s))
    (args.out / "scan.json").write_text(
        json.dumps(
            {
                "sample_fps": SAMPLE_FPS,
                "analysis_wh": list(ANALYSIS_WH),
                "abs_min": ABS_MIN,
                "k_mad": K_MAD,
                "min_area": MIN_AREA,
                "max_area": MAX_AREA,
                "active_speed": ACTIVE_SPEED,
                "active_churn": ACTIVE_CHURN,
                "window_s": WINDOW_S,
                "min_present": MIN_PRESENT,
                "min_fill": MIN_FILL,
                "denoise_wh": DENOISE_WH,
                "max_fg_frac": MAX_FG_FRAC,
                "splits": SPLITS,
                "clips": [
                    {
                        "clip_id": clip.clip_id,
                        "camera": clip.camera,
                        "duration_s": round(clip.duration_s, 2),
                        "unmeasured_s": unmeasured_s[clip.clip_id],
                        "present_s": sum(1 for row in traces[clip.clip_id] if row.present),
                    }
                    for clip in resolved
                ],
            },
            indent=2,
        )
        + "\n"
    )
    if args.clean:
        for path in cache_dir.glob("*.gray"):
            path.unlink()

    print(f"\n{len(clips)} clips -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
