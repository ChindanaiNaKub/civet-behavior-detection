"""Build five-second labeled windows from the verified interval annotations.

The split happens by whole source clip and before any window exists, so one
clip's windows can never land on both sides of it (evaluation rules, issue #1).
A window is kept only when a single verified interval covers it completely and
that interval's behavior is a model class. Windows that cross an annotation
boundary, a Transition or an Uncertain span are dropped, which is what keeps
every training example single-label.

Overlapping intervals are an annotation error, not a case to resolve, because
docs/ANNOTATION-GUIDE.md allows one label per interval. Those windows are dropped
and counted so the error surfaces instead of being given an arbitrary label.

Usage:
    python3 scripts/windows.py --out outputs/windows/windows.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import labels

# Coverage is judged on decoded timestamps, so compare with a tolerance.
EPS = 1e-6


@dataclass(frozen=True)
class Interval:
    clip_id: str
    start: float
    end: float
    behavior: str


@dataclass(frozen=True)
class Window:
    clip_id: str
    start: float
    end: float
    behavior: str
    split: str


def grid(duration: float, window: float = labels.WINDOW_S):
    """Non-overlapping windows of exactly ``window`` seconds; a shorter tail is dropped.

    A short trailing window would be a differently shaped input, so it is dropped
    rather than padded. Every training, evaluation and demo window is therefore
    five seconds long, and the same grid is used in all three.
    """
    spans = []
    start = 0.0
    while start + window <= duration + EPS:
        spans.append((start, start + window))
        start += window
    return spans


def label_window(start: float, end: float, intervals: list[Interval]):
    """The behavior of the one interval covering [start, end), or why none does."""
    covering = [iv for iv in intervals if iv.start <= start + EPS and iv.end >= end - EPS]
    if not covering:
        return None, "not_covered"
    if len(covering) > 1:
        return None, "conflict"
    behavior = covering[0].behavior
    if behavior not in labels.MODEL_CLASSES:
        return None, "non_class_label"
    return behavior, ""


def windows_for_clip(
    clip_id: str,
    duration: float,
    split: str,
    intervals: list[Interval],
    window: float = labels.WINDOW_S,
):
    """Kept windows and a count of why each dropped window was dropped."""
    kept: list[Window] = []
    dropped: Counter = Counter()
    for start, end in grid(duration, window):
        behavior, reason = label_window(start, end, intervals)
        if behavior is None:
            dropped[reason] += 1
            continue
        kept.append(Window(clip_id, start, end, behavior, split))
    return kept, dropped


def build(
    intervals: list[Interval],
    durations: dict[str, float],
    splits: dict[str, str],
    window: float = labels.WINDOW_S,
):
    """Windows for every assigned clip, plus the reason counts for the rest."""
    by_clip: dict[str, list[Interval]] = {}
    for interval in intervals:
        by_clip.setdefault(interval.clip_id, []).append(interval)

    windows: list[Window] = []
    dropped: Counter = Counter()
    for clip_id in sorted(durations):
        split = splits.get(clip_id, "unassigned")
        if split not in ("train", "validation", "test"):
            dropped["clip_unassigned"] += 1
            continue
        clip_intervals = by_clip.get(clip_id)
        if not clip_intervals:
            dropped["clip_unlabelled"] += 1
            continue
        kept, clip_dropped = windows_for_clip(clip_id, durations[clip_id], split, clip_intervals, window)
        windows.extend(kept)
        dropped.update(clip_dropped)
    return windows, dropped


# --- csv boundary ------------------------------------------------------------


def read_intervals(path: Path) -> list[Interval]:
    intervals = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            clip_id = (row.get("clip_id") or "").strip()
            behavior = (row.get("behavior") or "").strip()
            if not clip_id or not behavior:
                raise ValueError(f"{path}: row {row} needs clip_id and behavior")
            if behavior not in labels.ALL_LABELS:
                raise ValueError(f"{path}: unknown behavior {behavior!r}")
            start, end = float(row["start_s"]), float(row["end_s"])
            if end <= start:
                raise ValueError(f"{path}: {clip_id} has end_s {end} <= start_s {start}")
            intervals.append(Interval(clip_id, start, end, behavior))
    return intervals


def read_durations(path: Path) -> dict[str, float]:
    with path.open(newline="") as handle:
        return {row["clip_id"].strip(): float(row["duration_s"]) for row in csv.DictReader(handle)}


def read_splits(path: Path) -> dict[str, str]:
    with path.open(newline="") as handle:
        return {row["clip_id"].strip(): row["split"].strip() for row in csv.DictReader(handle)}


def read_relative_paths(path: Path) -> dict[str, str]:
    with path.open(newline="") as handle:
        return {row["clip_id"].strip(): row["relative_path"].strip() for row in csv.DictReader(handle)}


def read_windows(path: Path) -> list[Window]:
    with path.open(newline="") as handle:
        return [
            Window(row["clip_id"].strip(), float(row["start_s"]), float(row["end_s"]), row["behavior"].strip(), row["split"].strip())
            for row in csv.DictReader(handle)
        ]


def write_windows(path: Path, windows: list[Window]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=labels.WINDOW_FIELDS)
        writer.writeheader()
        for win in windows:
            writer.writerow(
                {
                    "clip_id": win.clip_id,
                    "start_s": f"{win.start:.2f}",
                    "end_s": f"{win.end:.2f}",
                    "behavior": win.behavior,
                    "split": win.split,
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations.csv"))
    parser.add_argument("--source-clips", type=Path, default=Path("data/source-clips.csv"))
    parser.add_argument("--splits", type=Path, default=Path("data/splits.csv"))
    parser.add_argument("--out", type=Path, default=Path("outputs/windows/windows.csv"))
    args = parser.parse_args()

    intervals = read_intervals(args.annotations)
    if not intervals:
        parser.error(f"no verified intervals in {args.annotations}; annotate before building windows")
    windows, dropped = build(intervals, read_durations(args.source_clips), read_splits(args.splits))
    if not windows:
        parser.error("no window had one verified model-class label; check annotations.csv and splits.csv")

    write_windows(args.out, windows)

    counts: Counter = Counter((w.split, w.behavior) for w in windows)
    print(f"{len(windows)} windows from {len({w.clip_id for w in windows})} clips -> {args.out}")
    for split in labels.SPLITS:
        per_class = {c: counts[(split, c)] for c in labels.MODEL_CLASSES}
        print(f"  {split:<11} {sum(per_class.values()):>4}  {per_class}")
    if dropped:
        print("  dropped:", ", ".join(f"{reason} {count}" for reason, count in sorted(dropped.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
