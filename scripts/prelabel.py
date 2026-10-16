#!/usr/bin/env python3
"""Zero-shot behavior pre-labeler for source clips (issue #9).

Derives Resting from at least five seconds of sustained low motion and Locomotion
from location change (the centroid of the moving pixels), then uses the pretrained
zero-shot video-text model X-CLIP to propose Eating and Climbing intervals.

``confidence`` is an interval's weakest evidence margin, 0..1: for a motion row,
how close its worst sample sits to the motion threshold; for a zero-shot row, the
proposed class's softmax probability.

Output is weak supervision for human review. It is never evaluation ground truth:
reported metrics come only from the human gold set (ADR 0003).

Reviewer use: accept a row, or edit ``behavior``/``start_s``/``end_s`` and fill
``review``. Boundaries between two different behaviors are Transition candidates,
not a proposed class. ``zeroshot_top``/``zeroshot_prob`` record what the model
would have said on the strongest overlapping window, including inside the
motion-derived Resting/Locomotion rows, for the failure analysis.

Usage:
    python scripts/prelabel.py                        # all clips under data/raw
    python scripts/prelabel.py --clips data/raw --out outputs/prelabel/timeline.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Tunables, calibrated on data/raw (see docs/PRELABEL-SMOKE-TEST.md).
SAMPLE_FPS = 8.0            # frames kept per second of video
MOTION_WH = (160, 120)      # (w, h) for motion analysis
MODEL_WH = (224, 224)       # (w, h) fed to X-CLIP
MOTION_PIXEL_DELTA = 8      # grey levels for a pixel to count as moving
MOTION_MIN_FRACTION = 0.002 # moving pixels needed before a centroid is usable
REST_MAX_MOTION = 0.002     # mean moving-pixel fraction over REST_MIN_S, max for Resting
REST_MIN_S = 5.0            # sustained low motion needed for Resting
MOVE_SPAN_S = 1.0           # window for centroid displacement
MOVE_MIN = 0.08             # displacement / frame width, min for Locomotion
MIN_RUN_S = 1.0             # shorter runs inherit the previous label
WINDOW_S = 5.0              # zero-shot window length
PARTIAL_MIN_S = 2.0         # trailing window shorter than this is dropped
ZS_MIN_PROB = 0.35          # softmax below this -> Uncertain
MODEL_ID = "microsoft/xclip-base-patch16-zero-shot"
PROMPTS = [
    ("Eating", "a video of a civet eating food from a bowl"),
    ("Locomotion", "a video of a civet walking across the floor"),
    ("Resting", "a video of a civet lying still on the ground"),
    ("Climbing", "a video of a civet climbing up a vertical structure"),
]
BEHAVIORS = [name for name, _ in PROMPTS]

CSV_FIELDS = [
    "clip_id",
    "start_s",
    "end_s",
    "behavior",
    "confidence",
    "method",
    "uncertain_reason",
    "zeroshot_top",
    "zeroshot_prob",
    "review",
]


@dataclass
class Window:
    """One zero-shot window and its top prompt."""

    start: float
    end: float
    label: str
    prob: float


@dataclass
class Row:
    """One proposed interval, ready for CSV."""

    clip_id: str
    start: float
    end: float
    behavior: str
    confidence: float
    method: str
    uncertain_reason: str
    zeroshot_top: str
    zeroshot_prob: float
    review: str = ""


# --- motion signal -----------------------------------------------------------


def read_clip(path: Path, sample_fps: float = SAMPLE_FPS):
    """Decode a clip once, keeping every ~1/sample_fps frame at two sizes.

    Returns (small grey frames, RGB model frames, sample times, clip end), all
    numpy. Small frames serve motion analysis; model frames serve X-CLIP.
    """
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps / sample_fps)))

    small, model_frames, times = [], [], []
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % step == 0:
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small.append(cv2.resize(grey, MOTION_WH, interpolation=cv2.INTER_AREA))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            model_frames.append(cv2.resize(rgb, MODEL_WH, interpolation=cv2.INTER_AREA))
            times.append(index / fps)
        index += 1
    cap.release()
    if not times:
        raise RuntimeError(f"no frames decoded from {path}")
    return (
        np.stack(small),
        np.stack(model_frames),
        np.asarray(times, dtype=float),
        index / fps,
    )


def motion_and_disp(small: np.ndarray, sample_fps: float):
    """Per-sample motion and location-change signals.

    ``motion[t]`` is the fraction of pixels whose grey value changed by more than
    ``MOTION_PIXEL_DELTA`` since the previous sample, so it stays sensitive when
    the animal covers a small part of the frame. ``disp[t]`` is how far the
    centroid of those moving pixels has moved in the last ``MOVE_SPAN_S``
    seconds, in frame widths. ``disp`` is NaN while too few pixels move to place
    a centroid, so a still animal yields no location change.
    """
    count, height, width = small.shape
    diff = np.abs(small[1:].astype(np.int16) - small[:-1].astype(np.int16))
    diff = np.vstack([np.zeros((1, height, width), np.int16), diff])
    mask = diff > MOTION_PIXEL_DELTA
    motion = mask.mean(axis=(1, 2))

    rows = np.repeat(np.arange(count), mask.sum(axis=(1, 2)))
    columns = np.nonzero(mask.reshape(count, -1))[1]
    total = np.bincount(rows, minlength=count)
    usable = (total > 0) & (motion >= MOTION_MIN_FRACTION)
    centroids = np.full((count, 2), np.nan)
    centroids[:, 0] = np.where(
        usable, np.bincount(rows, weights=columns % width, minlength=count) / np.maximum(total, 1), np.nan
    )
    centroids[:, 1] = np.where(
        usable, np.bincount(rows, weights=columns // width, minlength=count) / np.maximum(total, 1), np.nan
    )

    span = max(1, int(round(MOVE_SPAN_S * sample_fps)))
    disp = np.full(count, np.nan)
    for t in range(count):
        seen = [i for i in range(max(0, t - span + 1), t + 1) if not math.isnan(centroids[i, 0])]
        if len(seen) > 1:
            disp[t] = np.hypot(*(centroids[seen[-1]] - centroids[seen[0]])) / width
    return motion, disp


def label_samples(motion: np.ndarray, disp: np.ndarray, sample_fps: float) -> np.ndarray:
    """Per-sample behavior label from motion only: Resting, Locomotion or other.

    Resting needs the mean motion over the trailing REST_MIN_S window below
    REST_MAX_MOTION, so it can only start REST_MIN_S seconds into the clip.
    Locomotion needs the moving-pixel centroid to have moved MOVE_MIN frame
    widths in the last MOVE_SPAN_S.
    """
    count = len(motion)
    labels = np.array(["other"] * count, dtype=object)
    rest_span = max(1, int(round(REST_MIN_S * sample_fps)))
    for t in range(rest_span - 1, count):
        if motion[t - rest_span + 1 : t + 1].mean() < REST_MAX_MOTION:
            labels[t] = "Resting"
    for t in range(count):
        if labels[t] == "other" and not math.isnan(disp[t]) and disp[t] >= MOVE_MIN:
            labels[t] = "Locomotion"
    return merge_short_runs(labels, max(1, int(round(MIN_RUN_S * sample_fps))))


def merge_short_runs(labels: np.ndarray, min_run: int) -> np.ndarray:
    """Give runs shorter than ``min_run`` the previous run's label."""
    out = labels.copy()
    starts = [0] + [i for i in range(1, len(labels)) if labels[i] != labels[i - 1]]
    starts.append(len(labels))
    for a, b in zip(starts, starts[1:]):
        if b - a >= min_run or a == 0:
            continue
        out[a:b] = out[a - 1]
    return out


def motion_confidence(label: str, motion: np.ndarray, disp: np.ndarray) -> float:
    """0..1 margin of the weakest sample in the interval, min at the threshold."""
    if label == "Resting":
        worst = float(np.mean(motion)) / REST_MAX_MOTION
    else:
        values = disp[~np.isnan(disp)]
        worst = MOVE_MIN / float(np.min(values)) if len(values) else 1.0
    return round(min(max(1.0 - worst, 0.0), 1.0), 3)


# --- intervals ---------------------------------------------------------------


def runs(labels: np.ndarray):
    """Yield (label, first index, last index) for each maximal run."""
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            yield labels[start], start, i - 1
            start = i


def grid_windows(clip_end: float, window: float = WINDOW_S, partial_min: float = PARTIAL_MIN_S):
    """Non-overlapping windows covering the clip; the trailing one may be short."""
    windows = []
    start = 0.0
    while start < clip_end:
        end = min(start + window, clip_end)
        if end - start >= partial_min:
            windows.append((start, end))
        start += window
    return windows


def zs_diagnostic(windows: list[Window], start: float, end: float):
    """Highest-probability zero-shot window overlapping [start, end)."""
    best = None
    for w in windows:
        if min(w.end, end) > max(w.start, start) and (best is None or w.prob > best.prob):
            best = w
    return (best.label, round(best.prob, 3)) if best else ("", "")


def build_rows(
    clip_id: str,
    times: np.ndarray,
    labels: np.ndarray,
    motion: np.ndarray,
    disp: np.ndarray,
    windows: list[Window],
    clip_end: float,
) -> list[Row]:
    """Turn per-sample labels plus zero-shot windows into proposed intervals.

    Zero-shot rows carry the label of the full zero-shot window, so a run
    boundary that cuts a window yields a shorter interval labelled from that
    wider window, and window pieces shorter than PARTIAL_MIN_S are dropped.
    """
    dt = float(times[1] - times[0]) if len(times) > 1 else 0.0
    rows: list[Row] = []
    for label, first, last in runs(labels):
        start = float(times[first])
        end = min(float(times[last]) + dt, clip_end)
        if label != "other":
            top, prob = zs_diagnostic(windows, start, end)
            rows.append(
                Row(
                    clip_id,
                    start,
                    end,
                    label,
                    motion_confidence(label, motion[first : last + 1], disp[first : last + 1]),
                    "motion",
                    "",
                    top,
                    prob,
                )
            )
            continue
        for w in windows:
            span_start, span_end = max(start, w.start), min(end, w.end)
            if span_end - span_start < PARTIAL_MIN_S:
                continue
            behavior, reason = w.label, ""
            if w.label not in ("Eating", "Climbing") or w.prob < ZS_MIN_PROB:
                behavior = "Uncertain"
                reason = f"zero-shot top {w.label} {w.prob:.2f}"
            rows.append(
                Row(
                    clip_id,
                    span_start,
                    span_end,
                    behavior,
                    round(w.prob, 3),
                    "zero-shot",
                    reason,
                    w.label,
                    round(w.prob, 3),
                )
            )
    return merge_rows(rows)


def merge_rows(rows: list[Row]) -> list[Row]:
    """Join neighbours that ended up with the same behavior; keep the weakest confidence."""
    merged: list[Row] = []
    for row in rows:
        if merged and merged[-1].behavior == row.behavior and merged[-1].end == row.start:
            last = merged[-1]
            last.end = row.end
            last.confidence = min(last.confidence, row.confidence)
            if (row.zeroshot_prob or 0) > (last.zeroshot_prob or 0):
                last.zeroshot_top, last.zeroshot_prob = row.zeroshot_top, row.zeroshot_prob
                last.uncertain_reason = row.uncertain_reason
            continue
        merged.append(row)
    return merged


# --- zero-shot model ---------------------------------------------------------


def classify_windows(model, processor, frames: np.ndarray, times: np.ndarray, spans, device: str):
    """Score each span against the prompts; return Window list in span order."""
    import torch

    window_results: list[Window] = []
    texts = [text for _, text in PROMPTS]
    for start, end in spans:
        picks = np.searchsorted(times, np.linspace(start, end, 32, endpoint=False))
        picks = np.clip(picks, 0, len(times) - 1)
        clips = [frames[i] for i in picks]
        inputs = processor(text=texts, videos=clips, return_tensors="pt", padding=True)
        if "pixel_values" not in inputs:  # some transformers builds route video frames via images
            inputs = processor(text=texts, images=clips, return_tensors="pt", padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits_per_video
        probs = logits.softmax(dim=-1)[0]
        best = int(probs.argmax())
        window_results.append(Window(start, end, BEHAVIORS[best], float(probs[best])))
    return window_results


def load_model(model_id: str, device: str):
    import torch
    from transformers import XCLIPModel, XCLIPProcessor

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available; running on CPU")
        device = "cpu"
    processor = XCLIPProcessor.from_pretrained(model_id)
    model = XCLIPModel.from_pretrained(model_id).to(device).eval()
    return model, processor


def prelabel_clip(path: Path, model, processor, device: str) -> list[Row]:
    small, frames, times, clip_end = read_clip(path)
    motion, disp = motion_and_disp(small, SAMPLE_FPS)
    labels = label_samples(motion, disp, SAMPLE_FPS)
    spans = grid_windows(clip_end)
    windows = classify_windows(model, processor, frames, times, spans, device)
    return build_rows(path.stem, times, labels, motion, disp, windows, clip_end)


# --- driver ------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clips", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("outputs/prelabel/timeline.csv"))
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0, help="only the first N clips")
    args = parser.parse_args()

    clips = sorted(args.clips.rglob("*.mp4"))
    if args.limit:
        clips = clips[: args.limit]
    if not clips:
        parser.error(f"no .mp4 files under {args.clips}")

    model, processor = load_model(args.model, args.device)
    rows: list[Row] = []
    for path in clips:
        clip_rows = prelabel_clip(path, model, processor, args.device)
        rows.extend(clip_rows)
        print(f"{path.parent.name}/{path.name}: {len(clip_rows)} intervals", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_FIELDS)
        for row in rows:
            writer.writerow(
                [
                    row.clip_id,
                    f"{row.start:.2f}",
                    f"{row.end:.2f}",
                    row.behavior,
                    row.confidence,
                    row.method,
                    row.uncertain_reason,
                    row.zeroshot_top,
                    row.zeroshot_prob,
                    row.review,
                ]
            )

    counts: dict[str, int] = {}
    for row in rows:
        counts[row.behavior] = counts.get(row.behavior, 0) + 1
    print(f"\n{len(clips)} clips -> {len(rows)} intervals -> {args.out}")
    print("proposed:", ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
