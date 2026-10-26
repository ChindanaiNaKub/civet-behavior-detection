#!/usr/bin/env python3
"""Seed bounding boxes for the Phase 2 YOLO detector.

Task 2 needs bounding boxes, but no person will draw them and a pretrained COCO
detector found zero animals in this footage. This proposes boxes with a classical
detector so YOLO has a first set to train on: per-pixel median background, MAD
thresholding, morphology, then a temporal persistence filter. Swaying foliage
appears for a frame and vanishes; a civet persists across neighbouring seconds.

This is only the seed stage. Later boxes come from YOLO itself, trained on these
and then proposing more, so a noisy seed is acceptable as long as it lands on the
animal often enough to learn from.

The burned-in timestamp and camera text are static, so the median background
removes them; they are not special-cased.

Usage:
    python3 scripts/detect.py
    python3 scripts/detect.py --clips data/raw --out outputs/detection/boxes.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Tunables, calibrated on the 22 clips in data/raw (1061 sampled frames). The
# validated run produced 472 raw candidates; the persistence filter kept the
# animal-shaped majority of them.
SAMPLE_FPS = 1.0            # sampled frames per second
SCALE = 0.5                 # analysis resolution relative to full resolution
ABS_MIN = 14.0              # grey-level difference floor
K_MAD = 6.0                 # noise multiplier on the per-pixel MAD
MIN_AREA = 400              # component area at half resolution, px
MAX_AREA = 25000            # a larger blob is enclosure structure, not the civet
MIN_SIDE = 15               # component width and height at half resolution, px
ASPECT = (0.4, 4.0)         # width / height range for a plausible civet
CLOSE_WH = (7, 7)           # joins a broken animal silhouette
OPEN_WH = (5, 5)            # removes isolated noise pixels
PERSIST_WINDOW_S = 1.0      # a box must have a neighbour this close in time
PERSIST_TOLERANCE_S = 0.5 / SAMPLE_FPS  # consecutive samples drift past 1 s with real fps
PERSIST_IOU = 0.2           # or overlap a neighbour this much
PERSIST_CENTRE_PX = 40.0    # or keep its centre this close, half-res px
JPEG_QUALITY = 90

FIELDS = [
    "clip_id",
    "folder",
    "frame_index",
    "time_s",
    "x1",
    "y1",
    "x2",
    "y2",
    "area_px",
    "score",
    "source",
]

Box = tuple[int, int, int, int]  # x1, y1, x2, y2 at analysis resolution


@dataclass
class Sampled:
    """One clip's sampled frames and where they came from."""

    frames: np.ndarray          # (n, h, w) uint8 greyscale at SCALE
    times: np.ndarray           # seconds
    indices: np.ndarray         # source frame indices
    size: tuple[int, int]       # (width, height) full resolution


@dataclass
class Detection:
    """One kept box, in full-resolution pixels."""

    frame_index: int
    time_s: float
    box: Box


@dataclass
class ClipResult:
    """What one clip contributed, for the summary."""

    sampled: int
    candidates: int
    detections: list[Detection]


# --- pure detection ----------------------------------------------------------


def box_iou(a: Box, b: Box) -> float:
    """Intersection over union of two xyxy boxes."""
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    overlap = max(0, right - left) * max(0, bottom - top)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap
    return overlap / union if union else 0.0


def centre_distance(a: Box, b: Box) -> float:
    """Euclidean distance between box centres."""
    return float(
        np.hypot(
            (a[0] + a[2] - b[0] - b[2]) / 2.0,
            (a[1] + a[3] - b[1] - b[3]) / 2.0,
        )
    )


def _largest_candidate(stats: np.ndarray) -> Box | None:
    """Largest component whose size and shape can be a civet, or None."""
    best: Box | None = None
    best_area = 0
    for x, y, width, height, area in stats[1:]:
        if area < MIN_AREA or area > MAX_AREA:
            continue
        if width < MIN_SIDE or height < MIN_SIDE:
            continue
        if not ASPECT[0] <= width / height <= ASPECT[1]:
            continue
        if area > best_area:
            best_area = int(area)
            best = (int(x), int(y), int(x + width), int(y + height))
    return best


def candidate_boxes(frames: np.ndarray) -> list[Box | None]:
    """Largest plausible box per frame from a median background model.

    ``frames`` is a greyscale stack at analysis resolution. Returns one box or
    None per frame, in frame order.
    """
    background = np.median(frames, axis=0)
    noise = np.median(np.abs(frames.astype(np.int16) - background.astype(np.int16)), axis=0) + 1.0
    threshold = np.maximum(ABS_MIN, K_MAD * noise)
    close = cv2.getStructuringElement(cv2.MORPH_RECT, CLOSE_WH)
    open_ = cv2.getStructuringElement(cv2.MORPH_RECT, OPEN_WH)

    boxes: list[Box | None] = []
    for frame in frames:
        mask = (np.abs(frame.astype(np.int16) - background.astype(np.int16)) > threshold).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_)
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        boxes.append(_largest_candidate(stats))
    return boxes


def persistence_filter(items: list[tuple[float, Box | None]]) -> list[tuple[float, Box]]:
    """Keep boxes that a neighbouring second corroborates.

    ``items`` holds (time in seconds, box or None). A box survives when another
    box within PERSIST_WINDOW_S overlaps it by PERSIST_IOU or keeps its centre
    within PERSIST_CENTRE_PX.
    """
    present = [(time, box) for time, box in items if box is not None]
    kept: list[tuple[float, Box]] = []
    for i, (time, box) in enumerate(present):
        for j, (other_time, other) in enumerate(present):
            if i == j or abs(time - other_time) > PERSIST_WINDOW_S + PERSIST_TOLERANCE_S:
                continue
            if box_iou(box, other) >= PERSIST_IOU or centre_distance(box, other) <= PERSIST_CENTRE_PX:
                kept.append((time, box))
                break
    return kept


def to_full(box: Box, size: tuple[int, int]) -> Box:
    """Scale a half-resolution box to full pixels, clamped to the frame."""
    width, height = size
    x1, y1, x2, y2 = (int(round(value / SCALE)) for value in box)
    x1 = min(max(x1, 0), width - 1)
    y1 = min(max(y1, 0), height - 1)
    x2 = min(max(x2, x1 + 1), width)
    y2 = min(max(y2, y1 + 1), height)
    return x1, y1, x2, y2


# --- clip I/O ----------------------------------------------------------------


def read_clip(path: Path) -> Sampled:
    """Decode a clip once, keeping every ~1/SAMPLE_FPS frame at analysis size."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    step = max(1, int(round(fps / SAMPLE_FPS)))

    frames, times, indices = [], [], []
    size = (0, 0)
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % step == 0:
            if not size[0]:
                size = (frame.shape[1], frame.shape[0])
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(grey, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
            frames.append(small)
            times.append(index / fps)
            indices.append(index)
        index += 1
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return Sampled(np.stack(frames), np.asarray(times, float), np.asarray(indices, int), size)


def detect_clip(path: Path, image_dir: Path, label_dir: Path) -> ClipResult:
    """Read one clip, propose boxes, write the surviving frames and labels."""
    sampled = read_clip(path)
    candidates = candidate_boxes(sampled.frames)
    kept = persistence_filter(list(zip(sampled.times.tolist(), candidates)))
    by_time = dict(zip(sampled.times.tolist(), sampled.indices.tolist()))

    cap = cv2.VideoCapture(str(path))
    detections: list[Detection] = []
    for time_s, box in kept:
        index = by_time[time_s]
        full = to_full(box, sampled.size)
        stem = f"{path.stem}_{time_s:.2f}"

        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            continue
        cv2.imwrite(
            str(image_dir / f"{stem}.jpg"),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
        )
        x1, y1, x2, y2 = full
        width, height = sampled.size
        label = (
            f"0 {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
            f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}\n"
        )
        (label_dir / f"{stem}.txt").write_text(label)
        detections.append(Detection(index, time_s, full))
    cap.release()
    return ClipResult(len(sampled.frames), sum(box is not None for box in candidates), detections)


# --- driver ------------------------------------------------------------------


def write_summary(path: Path, results: list[tuple[str, ClipResult]]) -> None:
    """Table of what each clip contributed, so a thin clip is visible at once."""
    lines = [
        "# Seed detection summary",
        "",
        "| clip | frames sampled | boxes kept | median full-res box area (px^2) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for clip_id, result in results:
        areas = [(d.box[2] - d.box[0]) * (d.box[3] - d.box[1]) for d in result.detections]
        median = f"{np.median(areas):.0f}" if areas else "-"
        lines.append(f"| {clip_id} | {result.sampled} | {len(result.detections)} | {median} |")
    sampled = sum(r.sampled for _, r in results)
    kept = sum(len(r.detections) for _, r in results)
    lines += ["", f"Total: {sampled} frames sampled, {kept} boxes kept."]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clips", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("outputs/detection/boxes.csv"))
    parser.add_argument("--dataset", type=Path, default=Path("outputs/detection/dataset"))
    parser.add_argument("--summary", type=Path, default=Path("outputs/detection/summary.md"))
    args = parser.parse_args()

    clips = sorted(args.clips.rglob("*.mp4"))
    if not clips:
        parser.error(f"no .mp4 files under {args.clips}")

    image_dir = args.dataset / "images"
    label_dir = args.dataset / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    results: list[tuple[str, ClipResult]] = []
    rows: list[list] = []
    for path in clips:
        result = detect_clip(path, image_dir, label_dir)
        results.append((path.stem, result))
        for detection in result.detections:
            x1, y1, x2, y2 = detection.box
            rows.append(
                [
                    path.stem,
                    path.parent.name,
                    detection.frame_index,
                    f"{detection.time_s:.2f}",
                    x1,
                    y1,
                    x2,
                    y2,
                    (x2 - x1) * (y2 - y1),
                    "1.0",
                    "seed",
                ]
            )
        print(
            f"{path.parent.name}/{path.stem}: {result.sampled} frames, "
            f"{result.candidates} candidates, {len(result.detections)} kept",
            flush=True,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        writer.writerows(rows)
    write_summary(args.summary, results)

    print(f"\n{len(clips)} clips -> {len(rows)} boxes -> {args.out}")
    print(f"dataset -> {args.dataset}, summary -> {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
