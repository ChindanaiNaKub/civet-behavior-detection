"""Run the three Phase 2 tasks on one clip, in one command.

Task 1 is the behavior classifier over five-second windows. Task 2 detects the
animal's box with the trained YOLO detector, falls back to the per-clip
background model when the detector is unsure, and classifies the cropped animal.
Task 3 reports the four silhouette features that replace keypoints (ADR 0009).

The fallback exists because the detector generalizes across cameras but not
across recording days: on the eight held-out clips it fires on 59 percent of
frames, and a visual audit put half of those boxes on the animal. The per-clip
background model adapts to the clip in front of it, so the demo still finds the
animal on a day the detector never saw.

Usage:
    python3 scripts/demo_all.py data/raw/Eating/25690519193547292.mp4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

import detect
import features
import frames
import model as M
import windows as W

DETECT_CONF = 0.25
YOLO_TRUST = 0.50       # above this the YOLO box wins, below it the per-clip model does
FALLBACK_TOLERANCE_S = 2.0
CROP_PAD = 0.10
UNKNOWN = "-"


def clip_duration(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if not count:
        raise SystemExit(f"{path}: no frame count, cannot time the timeline")
    return float(count) / fps


def read_frame(path: Path, time_s: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, time_s) * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"{path}: nothing decoded at {time_s:.2f} s")
    return frame


def yolo_top_box(detector, frame: np.ndarray) -> tuple[list[int] | None, float]:
    if detector is None:
        return None, 0.0
    result = detector(frame, imgsz=640, conf=DETECT_CONF, verbose=False)[0]
    confidences = result.boxes.conf.tolist()
    if not confidences:
        return None, 0.0
    best = int(np.argmax(confidences))
    return [int(v) for v in result.boxes.xyxy.tolist()[best]], float(confidences[best])


def per_clip_boxes(path: Path) -> dict[float, list[int]]:
    """The background model's boxes for this clip, keyed by sample time."""
    sampled = detect.read_clip(path)
    candidates = detect.candidate_boxes(sampled.frames)
    kept = detect.persistence_filter(list(zip(sampled.times.tolist(), candidates)))
    return {time: list(detect.to_full(box, sampled.size)) for time, box in kept}


def nearest_box(boxes: dict[float, list[int]], time_s: float) -> list[int] | None:
    if not boxes:
        return None
    closest = min(boxes, key=lambda t: abs(t - time_s))
    return boxes[closest] if abs(closest - time_s) <= FALLBACK_TOLERANCE_S else None


def pad_box(box: list[int], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    height, width = shape
    x1, y1, x2, y2 = box
    dx, dy = int((x2 - x1) * CROP_PAD), int((y2 - y1) * CROP_PAD)
    return (max(0, x1 - dx), max(0, y1 - dy), min(width, x2 + dx), min(height, y2 + dy))


def crop_windows(path: Path, spans, boxes) -> np.ndarray:
    """Eight cropped frames per window, as the classifier's input.

    A window without a box keeps the whole frame, so the row still classifies and
    the table can show that the detector found nothing rather than dropping it.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    decoded = []
    for (start, end), box in zip(spans, boxes):
        span = (end - start) / frames.SAMPLES
        step = max(1, int(round(span * fps)))
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
        kept, last = [], None
        for _ in range(frames.SAMPLES):
            ok, frame = cap.read()
            if ok:
                last = frame
            if last is None:
                raise SystemExit(f"{path}: nothing decoded at {start:.2f} s")
            patch = last
            if box is not None:
                x1, y1, x2, y2 = pad_box(box, last.shape[:2])
                if x2 - x1 > 8 and y2 - y1 > 8:
                    patch = last[y1:y2, x1:x2]
            kept.append(cv2.resize(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB), frames.WH,
                                   interpolation=cv2.INTER_AREA))
            for _ in range(step - 1):
                if not cap.grab():
                    break
        decoded.append(np.stack(kept))
    cap.release()
    return np.stack(decoded)


def fmt(value: float, spec: str = ".0f") -> str:
    return UNKNOWN if value is None or not np.isfinite(value) else format(value, spec)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("clip", type=Path)
    parser.add_argument("--model", type=Path, default=Path("outputs/model/model.pt"))
    parser.add_argument("--detector", type=Path, default=Path("outputs/detection/model/round2.pt"))
    parser.add_argument("--evaluation", type=Path, default=Path("outputs/evaluation.json"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-detector", action="store_true", help="skip YOLO and use the per-clip model only")
    args = parser.parse_args()

    if not args.evaluation.exists():
        raise SystemExit(f"{args.evaluation} not found; run scripts/evaluate.py first")
    threshold = float(json.loads(args.evaluation.read_text())["threshold"])

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model, config = M.load(args.model, str(device))
    detector = None
    if not args.no_detector:
        if not args.detector.exists():
            raise SystemExit(f"{args.detector} not found; run scripts/train_detector.py first")
        from ultralytics import YOLO

        detector = YOLO(str(args.detector))

    clip_id = args.clip.stem
    spans = W.grid(clip_duration(args.clip))
    windows = [W.Window(clip_id, start, end, "", "demo") for start, end in spans]

    decoded = np.stack(frames.clip_windows(args.clip, spans))
    probabilities = M.probabilities(model, decoded, range(len(windows)), device)
    classes = config["classes"]

    fallback = per_clip_boxes(args.clip)
    boxes, detector_conf, box_source = [], [], []
    for start, end in spans:
        frame = read_frame(args.clip, (start + end) / 2.0)
        box, conf = yolo_top_box(detector, frame)
        if box is not None and conf >= YOLO_TRUST:
            boxes.append(box); detector_conf.append(conf); box_source.append("yolo")
            continue
        fallback_box = nearest_box(fallback, (start + end) / 2.0)
        if fallback_box is not None:
            boxes.append(fallback_box); detector_conf.append(conf); box_source.append("motion")
        else:
            boxes.append(None); detector_conf.append(conf); box_source.append("none")

    cropped = crop_windows(args.clip, spans, boxes)
    crop_probabilities = M.probabilities(model, cropped, range(len(windows)), device)

    samples = features.clip_samples(args.clip)
    geometry = [features.window_row(window, samples) for window in windows]

    print(f"{args.clip}  ({spans[-1][1]:.1f} s, {len(windows)} windows, abstain below {threshold:.2f})")
    print(f"{'start':>7} {'end':>7}  {'T1':<11} {'conf':>5}  {'T2 box':<20} {'via':<6} {'conf':>5}  "
          f"{'crop':<11} {'body':>5} {'head':>5} {'speed':>7} {'limb':>5}")
    symbols = []
    for window, row, crop_row, geo in zip(windows, probabilities, crop_probabilities, geometry):
        best = int(np.argmax(row))
        label = classes[best] if float(row[best]) >= threshold else "Uncertain"
        leaned = classes[best]
        shown = label if label != "Uncertain" else f"?{leaned}"
        crop_best = classes[int(np.argmax(crop_row))]
        box = boxes[windows.index(window)]
        box_text = UNKNOWN if box is None else f"{box[0]},{box[1]},{box[2]},{box[3]}"
        values = geo.values
        print(f"{window.start:7.2f} {window.end:7.2f}  {shown:<11} {float(row[best]):5.2f}  {box_text:<20} "
              f"{box_source[windows.index(window)]:<6} {detector_conf[windows.index(window)]:5.2f}  "
              f"{crop_best:<11} {fmt(values['body_angle_deg'])} {fmt(values['head_angle_deg'])} "
              f"{fmt(values['speed_px_s'], '.0f'):>7} {fmt(values['limb_extent_px']):>5}")
        symbols.append("?" if label == "Uncertain" else label[0])

    found = sum(1 for source in box_source if source in {"yolo", "motion"})
    agree = sum(1 for row, crop_row in zip(probabilities, crop_probabilities)
                if classes[int(np.argmax(row))] == classes[int(np.argmax(crop_row))])
    print(f"\ntimeline, one character per window: {''.join(symbols)}")
    print("legend: E Eating, L Locomotion, R Resting, ? Uncertain")
    print(f"detection: {found}/{len(windows)} windows found the animal directly "
          f"({sum(1 for s in box_source if s == 'yolo')} by the trained detector, "
          f"{sum(1 for s in box_source if s == 'motion')} by the per-clip model)")
    print(f"full-frame and crop labels agree on {agree}/{len(windows)} windows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
