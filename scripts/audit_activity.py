#!/usr/bin/env python3
"""Contact sheets for checking the activity scan with your own eyes.

The scan in scripts/activity.py is a measurement, not a ground truth, so the
report has to say how often it was right. This script samples windows of one
state spread evenly across the clips, takes the frame in the middle of each
window from the cached analysis stack, draws the box the scan found, and tiles
them into one image. Reading the sheet is the audit; the count of tiles that hold
the animal is the number the report quotes.

The frames come from the same 480 by 270 stack the measurement saw, so the sheet
answers "did the measurement's own view hold the animal", which is the question
the audit has to answer. A full-resolution crop is a different check and this
script does not make one.

Absent is the state worth auditing most: a window the scan called empty that
holds a civet is a miss, and nothing else in the pipeline can see one.

Usage:
    python3 scripts/audit_activity.py --state Active
    python3 scripts/audit_activity.py --state Absent --count 36
"""

from __future__ import annotations

import argparse
import csv
from itertools import groupby
from pathlib import Path

import cv2
import numpy as np

import activity as A

TILE_W, TILE_H = 480, 270
COLUMNS = 4


def read_frames(path: Path) -> dict[str, list[dict]]:
    """Samples per clip, so a window can find the box nearest its middle."""
    by_clip: dict[str, list[dict]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            by_clip.setdefault(row["clip_id"], []).append(row)
    return by_clip


def windows_of(path: Path, durations: dict[str, float], state: str) -> list[dict]:
    """Windows of one state, with Absent rebuilt from the recording's full grid."""
    out = []
    covered: dict[str, list[tuple[float, float]]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            begin, end = float(row["start_s"]), float(row["end_s"])
            covered.setdefault(row["clip_id"], []).append((begin, end))
            if row["behavior"] == state:
                out.append({"clip_id": row["clip_id"], "start_s": begin})
    if state != "Absent":
        return out
    for clip_id, duration in durations.items():
        start = 0.0
        for begin, end in sorted(covered.get(clip_id, [])):
            while start + A.WINDOW_S <= begin + 1e-6:
                out.append({"clip_id": clip_id, "start_s": start})
                start += A.WINDOW_S
            start = max(start, end)
        while start + A.WINDOW_S <= duration + 1e-6:
            out.append({"clip_id": clip_id, "start_s": start})
            start += A.WINDOW_S
    return out


def box_at(samples: list[dict], time_s: float) -> tuple[int, int, int, int] | None:
    """The box the scan held nearest this moment, in analysis pixels."""
    best, distance = None, float("inf")
    for row in samples:
        gap = abs(float(row["time_s"]) - time_s)
        if gap < distance:
            best, distance = row, gap
    if best is None or distance > 2.0:
        return None
    return tuple(int(best[key]) for key in ("x1", "y1", "x2", "y2"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("outputs/activity"))
    parser.add_argument("--clips", type=Path, required=True, help="folder of cameraN/*.mp4")
    parser.add_argument("--state", default="Active", choices=("Active", "Inactive", "Absent"))
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--image", type=Path, default=None)
    args = parser.parse_args()

    durations = {}
    with (args.out / "source-clips.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            durations[row["clip_id"]] = float(row["duration_s"])
    videos = {path.stem: path for path in sorted(args.clips.rglob("*.mp4"))}

    samples = read_frames(args.out / "frames.csv")
    chosen = windows_of(args.out / "windows.csv", durations, args.state)
    if not chosen:
        raise SystemExit(f"no {args.state} window in {args.out / 'windows.csv'}")
    step = max(1, len(chosen) // args.count)
    chosen = chosen[::step][: args.count]

    tiles = []
    for clip_id, group in groupby(chosen, key=lambda row: row["clip_id"]):
        cap = cv2.VideoCapture(str(videos[clip_id]))
        for row in group:
            middle = row["start_s"] + A.WINDOW_S / 2
            cap.set(cv2.CAP_PROP_POS_MSEC, middle * 1000.0)
            ok, frame = cap.read()
            if not ok:
                continue
            box = box_at(samples.get(clip_id, []), middle)
            if box is not None:
                scale = frame.shape[1] / A.ANALYSIS_WH[0]
                x1, y1, x2, y2 = (int(value * scale) for value in box)
                # A crop twice the box wide, so the audit sees the animal and what
                # is beside it: a box on a shadow is obvious only with its context.
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                half_w, half_h = int((x2 - x1) * 1.6), int((y2 - y1) * 1.6)
                left = max(0, min(cx - half_w, frame.shape[1] - 2 * half_w))
                top = max(0, min(cy - half_h, frame.shape[0] - 2 * half_h))
                frame = frame[top : top + 2 * half_h, left : left + 2 * half_w].copy()
                cv2.rectangle(frame, (x1 - left, y1 - top), (x2 - left, y2 - top), (0, 255, 0), 2)
            tile = cv2.resize(frame, (TILE_W, TILE_H))
            cv2.putText(
                tile, f"{clip_id[:3]} {row['start_s'] / 3600:5.2f}h", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA,
            )
            tiles.append(tile)
        cap.release()

    rows = (len(tiles) + COLUMNS - 1) // COLUMNS
    sheet = np.zeros((rows * TILE_H, COLUMNS * TILE_W, 3), np.uint8)
    for index, tile in enumerate(tiles):
        y, x = (index // COLUMNS) * TILE_H, (index % COLUMNS) * TILE_W
        sheet[y : y + TILE_H, x : x + TILE_W] = tile

    image = args.image or args.out / f"audit-{args.state.lower()}.png"
    cv2.imwrite(str(image), sheet)
    print(f"{len(tiles)} {args.state} windows -> {image}")
    for row in chosen:
        print(f"  {row['clip_id']} {row['start_s']:8.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
