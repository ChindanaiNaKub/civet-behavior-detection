#!/usr/bin/env python3
"""Task 3 features over the labelled five-second windows, and a small classifier.

The brief asks for behavior analysis from keypoints. There is no civet pose model
and no keypoint annotation, so ADR 0009 replaces pose estimation with silhouette
geometry and keeps the four feature names: body angle is the principal axis of the
silhouette, head angle is the axis of its narrow end, movement speed is centroid
displacement per second, limb distance is the extent across the body axis.

The silhouette is the background-model mask from detect.py, cropped to the box that
function already keeps. The animal is the only thing that translates, so the median
background catches it; where it lies still for most of a clip the median absorbs it
and a window has no silhouette at all. Such a window is written with n_frames 0
rather than filled with a guess.

The classifier sees train and validation windows only, stops on validation macro F1,
and touches the test clips once, at the end.

Usage:
    python3 scripts/features.py
    python3 scripts/features.py --clips data/raw --out outputs/features/features.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

import detect
import labels
import metrics
import windows as W

# Tunables. Two samples per second, so a five-second window has ~10 chances at a
# silhouette and a walking animal moves far enough between samples to measure.
SAMPLE_FPS = 2.0
MIN_SILHOUETTE_PX = 3       # fewer foreground pixels cannot give an axis
HIDDEN = 16                 # MLP width; the input is four numbers, so it stays tiny
LEARNING_RATE = 0.01
WEIGHT_DECAY = 1e-4
EPOCHS = 600
PATIENCE = 60

FEATURES = ("body_angle_deg", "head_angle_deg", "speed_px_s", "limb_extent_px")
FIELDS = ["clip_id", "start_s", "end_s", *FEATURES, "box_area_px", "n_frames"]

KERNEL_CLOSE = cv2.getStructuringElement(cv2.MORPH_RECT, detect.CLOSE_WH)
KERNEL_OPEN = cv2.getStructuringElement(cv2.MORPH_RECT, detect.OPEN_WH)


@dataclass
class Sample:
    """Geometry of one silhouette, at full-resolution units."""

    time_s: float
    body_angle_deg: float
    head_angle_deg: float
    speed_px_s: float
    limb_extent_px: float
    box_area_px: float


@dataclass
class WindowRow:
    """One labelled window and the medians of its samples."""

    window: W.Window
    values: dict[str, float]
    n_frames: int


# --- pure geometry -----------------------------------------------------------


def _points(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column and row coordinates of every set pixel."""
    rows, columns = np.nonzero(mask)
    return columns.astype(float), rows.astype(float)


def principal_angle_deg(columns: np.ndarray, rows: np.ndarray) -> float:
    """Axis of a point set in degrees, 0..180 measured counter-clockwise from +x."""
    if len(columns) < MIN_SILHOUETTE_PX:
        return float("nan")
    dx = columns - columns.mean()
    dy = rows - rows.mean()
    angle = 0.5 * math.degrees(math.atan2(2.0 * (dx * dy).mean(), (dx * dx).mean() - (dy * dy).mean()))
    return angle % 180.0


def body_angle_deg(mask: np.ndarray) -> float:
    """Principal axis of the whole silhouette, 0..180; 0 and 180 are the same axis."""
    return principal_angle_deg(*_points(mask))


def head_angle_deg(mask: np.ndarray) -> float:
    """Principal axis of the narrow end of the silhouette.

    The silhouette is split at its centroid along the body axis and each half's mean
    width is its mean distance from that axis. The narrower half is called the head.

    This is a proxy, not a head detector. A curled animal folds both ends together
    and the split is meaningless; a tail can be the narrow end instead of the snout;
    a resting animal seen from above has no narrow end at all.
    """
    columns, rows = _points(mask)
    if len(columns) < 2 * MIN_SILHOUETTE_PX:
        return float("nan")
    angle = principal_angle_deg(columns, rows)
    if math.isnan(angle):
        return angle
    theta = math.radians(angle)
    along = columns * math.cos(theta) + rows * math.sin(theta)
    across = -columns * math.sin(theta) + rows * math.cos(theta)

    halves = [along >= along.mean(), along < along.mean()]
    widths = [np.abs(across[half]).mean() if half.sum() >= MIN_SILHOUETTE_PX else math.inf for half in halves]
    narrow = halves[int(np.argmin(widths))]
    return principal_angle_deg(columns[narrow], rows[narrow])


def limb_extent_px(mask: np.ndarray, scale: float = 1.0) -> float:
    """Extent across the body axis in pixels; ``scale`` turns analysis px into full px."""
    columns, rows = _points(mask)
    if len(columns) < MIN_SILHOUETTE_PX:
        return float("nan")
    theta = math.radians(principal_angle_deg(columns, rows))
    across = -columns * math.sin(theta) + rows * math.cos(theta)
    return float(across.max() - across.min()) / scale


def speed_px_s(before, after, dt_s: float) -> float:
    """Centroid displacement between two samples over the time between them."""
    if dt_s <= 0:
        return float("nan")
    return float(np.hypot(after[0] - before[0], after[1] - before[1]) / dt_s)


# --- silhouette --------------------------------------------------------------


def background_model(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Median background and MAD threshold, the model detect.candidate_boxes fits.

    That function returns boxes but not the model, and a silhouette needs both. The
    constants are imported from detect so the mask cannot drift from the boxes.
    """
    background = np.median(frames, axis=0)
    noise = np.median(np.abs(frames.astype(np.int16) - background.astype(np.int16)), axis=0) + 1.0
    return background, np.maximum(detect.ABS_MIN, detect.K_MAD * noise)


def silhouette_mask(frame, background, threshold, box) -> np.ndarray:
    """The background-model mask inside a box, as detect.py builds it."""
    x1, y1, x2, y2 = box
    mask = (np.abs(frame.astype(np.int16) - background.astype(np.int16)) > threshold).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL_CLOSE)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL_OPEN)
    return mask[y1:y2, x1:x2].astype(bool)


def read_clip(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Grey frames at SAMPLE_FPS and detect.SCALE, their times, and full frame size.

    Same decode as detect.read_clip at twice its rate: speed needs samples closer
    together than a second.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    step = max(1, int(round(fps / SAMPLE_FPS)))

    frames, times = [], []
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
            frames.append(cv2.resize(grey, None, fx=detect.SCALE, fy=detect.SCALE, interpolation=cv2.INTER_AREA))
            times.append(index / fps)
        index += 1
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return np.stack(frames), np.asarray(times, float), size


def clip_samples(path: Path) -> list[Sample]:
    """One geometry sample per frame where detect.py keeps a box and a silhouette."""
    frames, times, size = read_clip(path)
    boxes = detect.candidate_boxes(frames)
    kept = dict(detect.persistence_filter(list(zip(times.tolist(), boxes))))
    background, threshold = background_model(frames)

    samples: list[Sample] = []
    previous_centroid, previous_time = None, None
    for time, frame in zip(times.tolist(), frames):
        box = kept.get(time)
        if box is None:
            continue
        mask = silhouette_mask(frame, background, threshold, box)
        if mask.sum() < MIN_SILHOUETTE_PX:
            continue
        rows, columns = np.nonzero(mask)
        centroid = (float(columns.mean()) + box[0], float(rows.mean()) + box[1])

        speed = float("nan")
        if previous_centroid is not None:
            # Analysis pixels per second scaled up to full-resolution pixels per second.
            speed = speed_px_s(previous_centroid, centroid, time - previous_time) / detect.SCALE
        previous_centroid, previous_time = centroid, time

        x1, y1, x2, y2 = detect.to_full(box, size)
        samples.append(
            Sample(
                time,
                body_angle_deg(mask),
                head_angle_deg(mask),
                speed,
                limb_extent_px(mask, detect.SCALE),
                float((x2 - x1) * (y2 - y1)),
            )
        )
    return samples


def window_row(window: W.Window, samples: list[Sample]) -> WindowRow:
    """Medians over the samples whose time falls inside the window."""
    inside = [s for s in samples if window.start <= s.time_s < window.end]
    values: dict[str, float] = {}
    for name in (*FEATURES, "box_area_px"):
        column = np.array([getattr(s, name) for s in inside], float)
        column = column[np.isfinite(column)]
        values[name] = float(np.median(column)) if len(column) else float("nan")
    return WindowRow(window, values, len(inside))


# --- classifier --------------------------------------------------------------


class MLP(nn.Module):
    """Four standardised features to one behavior; small enough for 90 windows."""

    def __init__(self, inputs: int, classes: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inputs, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def prepare(train_rows: list[WindowRow]):
    """Train medians fill a missing feature; train mean and std scale every split.

    A window with no silhouette cannot be left out of a run over the locked test
    grid, so its four numbers are the training medians: the model sees a typical
    animal rather than an invented one, and the report counts how many that was.
    """
    medians = {}
    for name in FEATURES:
        column = np.array([r.values[name] for r in train_rows], float)
        column = column[np.isfinite(column)]
        medians[name] = float(np.median(column)) if len(column) else 0.0

    def raw(chosen: list[WindowRow]) -> np.ndarray:
        return np.array(
            [[r.values[name] if math.isfinite(r.values[name]) else medians[name] for name in FEATURES] for r in chosen],
            dtype=np.float32,
        ).reshape(len(chosen), len(FEATURES))

    train = raw(train_rows)
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std < 1e-8] = 1.0

    def standardise(chosen: list[WindowRow]) -> np.ndarray:
        return (raw(chosen) - mean) / std

    return train, standardise


def train_mlp(x: np.ndarray, y: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, classes: list[str], seed: int):
    """Full-batch Adam on train windows, best validation macro F1 kept."""
    torch.manual_seed(seed)
    model = MLP(x.shape[1], len(classes))
    counts = np.array([max(int((y == i).sum()), 1) for i in range(len(classes))], float)
    weight = torch.tensor(len(y) / (len(classes) * counts), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    inputs = torch.tensor(x)
    targets = torch.tensor(y, dtype=torch.long)
    val_inputs = torch.tensor(x_val)

    best = (-1.0, -1, None)
    stale = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            predictions = model(val_inputs).argmax(dim=1).numpy()
        score = metrics.report([classes[i] for i in y_val], [classes[i] for i in predictions], classes)["macro_f1"]
        if score > best[0]:
            best = (score, epoch, {k: v.detach().clone() for k, v in model.state_dict().items()})
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    model.load_state_dict(best[2])
    model.eval()
    return model, best


def predict(model, x: np.ndarray, classes: list[str]) -> list[str]:
    with torch.no_grad():
        return [classes[i] for i in model(torch.tensor(x)).argmax(dim=1).tolist()]


# --- report ------------------------------------------------------------------


def medians_table(rows: list[WindowRow], classes: list[str]) -> list[str]:
    """Median of each feature per class, over the windows that had a silhouette."""
    lines = [
        "| Behavior | windows with a silhouette | Body angle | Head angle | Speed (px/s) | Limb extent (px) | Box area (px^2) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    columns = (*FEATURES, "box_area_px")
    for name in classes:
        chosen = [r for r in rows if r.window.behavior == name]
        usable = [r for r in chosen if r.n_frames]
        cells = []
        for column in columns:
            values = np.array([r.values[column] for r in usable], float)
            values = values[np.isfinite(values)]
            cells.append(f"{np.median(values):.1f}" if len(values) else "-")
        lines.append(f"| {name} | {len(usable)}/{len(chosen)} | " + " | ".join(cells) + " |")
    return lines


def build_report(
    rows: list[WindowRow],
    classes: list[str],
    val_metrics: dict,
    test_metrics: dict,
    baseline_val: dict,
    baseline_test: dict,
    best: tuple,
    imputed: dict[str, int],
) -> str:
    """Markdown: what was measured, then what it cannot mean."""
    n_by_split = {split: sum(1 for r in rows if r.window.split == split) for split in labels.SPLITS}
    empty = {split: sum(1 for r in rows if r.window.split == split and not r.n_frames) for split in labels.SPLITS}
    if val_metrics["accuracy_decided"] < baseline_val["accuracy_decided"]:
        verdict = (
            "This is below the majority guess on accuracy and above it on macro F1: the gain is "
            "in the rare classes, not in getting most windows right."
        )
    else:
        verdict = "This is above the majority guess on both accuracy and macro F1."
    lines = [
        "# Task 3 features and classifier",
        "",
        "Generated by `scripts/features.py`. Silhouette geometry replaces pose estimation "
        "(ADR 0009); there are no keypoints and no keypoint model behind these numbers.",
        "",
        "## Data",
        "",
        f"Windows: train {n_by_split['train']}, validation {n_by_split['validation']}, "
        f"test {n_by_split['test']}. Windows with no silhouette at all: "
        f"train {empty['train']}, validation {empty['validation']}, test {empty['test']}; "
        "their four features are the training medians, and because those rows also take part in "
        "the fit the model learns one shared vector for every window whose animal was invisible.",
        "",
        f"Classifier: MLP {len(FEATURES)} -> {HIDDEN} -> {len(classes)} over {', '.join(FEATURES)}; "
        f"seed 0, Adam, class-weighted cross-entropy, early stop after {PATIENCE} epochs without "
        f"validation gain. Best epoch {best[1]} at validation macro F1 {best[0]:.3f}.",
        "",
        "## Feature medians per class",
        "",
        *medians_table([r for r in rows if r.window.split != "test"], classes),
        "",
        "Medians are over the train and validation windows that had at least one silhouette. "
        "A cell of `-` means no window of that class had any.",
        "",
        "## Validation",
        "",
        metrics.markdown(val_metrics, "Small MLP on validation windows"),
        "",
        f"Majority-class baseline on the same windows: macro F1 {baseline_val['macro_f1']:.3f}, "
        f"accuracy {baseline_val['accuracy_decided']:.3f}. {verdict}",
        "",
        "## Test",
        "",
        metrics.markdown(test_metrics, "Small MLP on the 15 locked test windows"),
        "",
        f"Majority-class baseline on the same windows: macro F1 {baseline_test['macro_f1']:.3f}, "
        f"accuracy {baseline_test['accuracy_decided']:.3f}. "
        f"Windows whose features were imputed: validation {imputed['validation']}, test {imputed['test']}.",
        "",
        "## What these features cannot see",
        "",
        "These are not keypoints, and the report should never be read as if they were. "
        "Body angle and head angle are two axes of one blob: the narrow end can be the tail, "
        "and a curled animal has no unambiguous long axis, so head angle is a guess about which "
        "end of the silhouette the head is on. The box is the largest *moving* component, so when "
        "the animal's head is static in the background and its rear moves, body angle is the rear's "
        "axis and head angle is not the head at all. Because the silhouette is a projection, a civet "
        "turning in place and a civet walking toward the camera can look equally short, and a median "
        "over samples that straddle 0 and 180 degrees collapses two opposite orientations toward 90. "
        "Speed is image motion in pixels, not distance travelled, so it changes with how far the "
        "animal is from the camera and with how much of it the background model absorbed. The "
        f"largest gap is Resting: {empty['train']}/{n_by_split['train']} train windows have no "
        "silhouette at all because a still animal becomes part of the median background, so those "
        "rows, and every imputed validation or test row, are the training median vector rather than "
        "resting geometry, and their predictions are a prior guess. Foliage that moves inside a kept "
        "box pollutes body angle, head angle and limb extent. With one animal and no depth there is "
        "nothing here about social behavior, posture under occlusion, or any behavior the brief's "
        "four features would not have covered either.",
    ]
    return "\n".join(lines) + "\n"


# --- driver ------------------------------------------------------------------


def write_csv(path: Path, rows: list[WindowRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        for row in rows:
            writer.writerow(
                [
                    row.window.clip_id,
                    f"{row.window.start:.2f}",
                    f"{row.window.end:.2f}",
                    *("" if not math.isfinite(row.values[name]) else f"{row.values[name]:.3f}" for name in FEATURES),
                    "" if not math.isfinite(row.values["box_area_px"]) else f"{row.values['box_area_px']:.1f}",
                    row.n_frames,
                ]
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clips", type=Path, default=Path("data/raw"))
    parser.add_argument("--windows", type=Path, default=Path("outputs/windows/windows.csv"))
    parser.add_argument("--splits", type=Path, default=Path("data/splits.csv"))
    parser.add_argument("--out", type=Path, default=Path("outputs/features/features.csv"))
    parser.add_argument("--report", type=Path, default=Path("outputs/features/features.md"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    windows = [w for w in W.read_windows(args.windows) if w.split in labels.SPLITS]
    if not windows:
        parser.error(f"no labelled windows in {args.windows}")
    splits = W.read_splits(args.splits)
    wrong = [w.clip_id for w in windows if splits.get(w.clip_id) != w.split]
    if wrong:
        parser.error(f"{args.windows} disagrees with {args.splits} on {sorted(set(wrong))}")

    paths = {path.stem: path for path in sorted(args.clips.rglob("*.mp4"))}
    clip_ids = sorted({w.clip_id for w in windows})
    absent = [clip for clip in clip_ids if clip not in paths]
    if absent:
        parser.error(f"no video under {args.clips} for {absent}")

    samples: dict[str, list[Sample]] = {}
    for clip_id in clip_ids:
        samples[clip_id] = clip_samples(paths[clip_id])
        print(f"{clip_id}: {len(samples[clip_id])} silhouette samples", flush=True)

    rows = [window_row(window, samples[window.clip_id]) for window in windows]
    write_csv(args.out, rows)
    print(f"\n{len(rows)} windows -> {args.out}")

    classes = list(labels.MODEL_CLASSES)
    train_rows = [r for r in rows if r.window.split == "train"]
    val_rows = [r for r in rows if r.window.split == "validation"]
    test_rows = [r for r in rows if r.window.split == "test"]

    # Features for the fit come only from the train clips. The assertion is the
    # guard: a test clip in the fit would invalidate every number below.
    test_clips = {r.window.clip_id for r in test_rows}
    assert not test_clips & {r.window.clip_id for r in train_rows + val_rows}

    x_train, standardise = prepare(train_rows)
    x_val = standardise(val_rows)
    x_test = standardise(test_rows)
    index_of = {name: i for i, name in enumerate(classes)}
    y_train = np.array([index_of[r.window.behavior] for r in train_rows])
    y_val = np.array([index_of[r.window.behavior] for r in val_rows])

    model, best = train_mlp(x_train, y_train, x_val, y_val, classes, args.seed)
    val_pred = predict(model, x_val, classes)
    test_pred = predict(model, x_test, classes)

    val_labels = [r.window.behavior for r in val_rows]
    test_labels = [r.window.behavior for r in test_rows]
    val_metrics = metrics.report(val_labels, val_pred, classes)
    test_metrics = metrics.report(test_labels, test_pred, classes)

    majority = classes[int(np.bincount(y_train, minlength=len(classes)).argmax())]
    baseline_val = metrics.report(val_labels, [majority] * len(val_rows), classes)
    baseline_test = metrics.report(test_labels, [majority] * len(test_rows), classes)
    imputed = {
        "validation": sum(1 for r in val_rows if not r.n_frames),
        "test": sum(1 for r in test_rows if not r.n_frames),
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        build_report(rows, classes, val_metrics, test_metrics, baseline_val, baseline_test, best, imputed)
    )

    print(f"train {len(train_rows)} windows | validation {len(val_rows)} | test {len(test_rows)}")
    print(f"majority class {majority}")
    print(f"validation macro F1 {val_metrics['macro_f1']:.3f} (baseline {baseline_val['macro_f1']:.3f})")
    print(f"test macro F1 {test_metrics['macro_f1']:.3f} (baseline {baseline_test['macro_f1']:.3f})")
    print(f"report -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
