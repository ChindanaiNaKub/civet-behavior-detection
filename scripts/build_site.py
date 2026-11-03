"""Build the static presentation site for Phase 2.

Renders one annotated clip per split for the three tasks, then writes a page
that plays them with the prediction timeline and the reported numbers. The site
needs no server and no model at presentation time, so it can be hosted as static
files and opened by anyone.

Everything with a number on the page is read from the pipeline's own output:
`outputs/evaluation.json` for Task 1, `outputs/features/features.md` for Task 3,
the README sentences for Task 2 and the banded civet activity summary, and the
activity figures from `docs/figures/`. The build fails rather than inventing one.

Usage:
    python3 scripts/build_site.py
    python3 scripts/build_site.py --clips data/raw/Resting/25690519195143424.mp4
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import re
import shutil
import subprocess
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import torch

import demo_all as D
import detect
import features
import frames
import model as M
import windows as W

ROOT = Path(__file__).resolve().parent.parent
RENDER_WIDTH, RENDER_HEIGHT = 960, 540
MAX_WINDOWS = 8                      # 40 s per clip keeps the page light
CRF = 28
FPS_FALLBACK = 25.0

INK = (0x22, 0x24, 0x1F)
BG = (0xF3, 0xEE, 0xDF)
CLASS_BGR = {"Eating": (0x3B, 0x5A, 0xC4), "Locomotion": (0x6F, 0x96, 0x87),
             "Resting": (0x36, 0x42, 0x28), "Uncertain": (0x60, 0x63, 0x64)}
CLASS_HEX = {"Eating": "#C45A3B", "Locomotion": "#87966F", "Resting": "#284236",
             "Uncertain": "#64675E"}
INSET_SIZE, INSET_MARGIN, INSET_TOP = 300, 18, 78
INSET_MIN_REGION = 170   # a box smaller than this is a head crop, so widen it around the animal
# cv2 wants BGR, so the page palette is flipped here.
RUST_BGR, MOSS_BGR, FOREST_BGR = (0x3B, 0x5A, 0xC4), (0x6F, 0x96, 0x87), (0x36, 0x42, 0x28)


def video_size(path: Path) -> tuple[float, float, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS_FALLBACK
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return float(count) / fps, fps, int(count)


def analyse(clip: Path, model, config, detector, threshold: float) -> dict:
    duration, _, _ = video_size(clip)
    spans = W.grid(duration)[:MAX_WINDOWS]
    clip_id = clip.stem
    if not spans:
        raise SystemExit(f"{clip}: shorter than one five-second window")

    windows = [W.Window(clip_id, start, end, "", "demo") for start, end in spans]
    decoded = np.stack(frames.clip_windows(clip, spans))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probabilities = M.probabilities(model, decoded, range(len(windows)), device)
    intervals = [iv for iv in W.read_intervals(ROOT / "data/annotations.csv") if iv.clip_id == clip_id]

    fallback = D.per_clip_boxes(clip)
    samples = features.clip_samples(clip)

    rows = []
    for window, probability in zip(windows, probabilities):
        middle = (window.start + window.end) / 2.0
        box, detector_conf = D.yolo_top_box(detector, D.read_frame(clip, middle))
        source = "yolo"
        if box is None or detector_conf < D.YOLO_TRUST:
            motion_box = D.nearest_box(fallback, middle)
            if motion_box is not None:
                box, source = motion_box, "motion"
            else:
                box, source = None, "none"
        best = int(np.argmax(probability))
        label = config["classes"][best] if float(probability[best]) >= threshold else "Uncertain"
        verified, reason = W.label_window(window.start, window.end, intervals)
        geometry = features.window_row(window, samples).values
        rows.append({
            "start": round(window.start, 2),
            "end": round(window.end, 2),
            "label": label,
            "leaned": config["classes"][best],
            "confidence": round(float(probability[best]), 3),
            "probabilities": [round(float(p), 3) for p in probability],
            "verified": verified,
            "excluded": reason,
            "box": box,
            "source": source,
            "detector_conf": round(float(detector_conf), 3),
            "features": {key: (None if not np.isfinite(value) else round(float(value), 1))
                         for key, value in geometry.items()},
        })
    return {"id": clip_id, "path": str(clip), "group": clip.parent.name,
            "duration": round(duration, 2),
            "rendered_s": round(spans[-1][1], 2), "windows": rows}


def banner(frame: np.ndarray, text: str, sub: str) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (RENDER_WIDTH, 64), (18, 20, 16), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    cv2.putText(frame, text, (18, 28), cv2.FONT_HERSHEY_DUPLEX, 0.72, (243, 238, 223), 1, cv2.LINE_AA)
    cv2.putText(frame, sub, (18, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (196, 200, 186), 1, cv2.LINE_AA)


def timeline_strip(frame: np.ndarray, rows: list[dict], current: int) -> None:
    """Two bands: the verified annotation and the model's prediction, window by window."""
    top = RENDER_HEIGHT - 84
    gutter, band, gap = 72, 32, 4
    width = (RENDER_WIDTH - gutter) / len(rows)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, top), (RENDER_WIDTH, RENDER_HEIGHT), (18, 20, 16), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    for label, key, y in (("verified", "verified", top + gap), ("model", "label", top + gap * 2 + band)):
        cv2.putText(frame, label, (8, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (196, 200, 186), 1, cv2.LINE_AA)
        for index, row in enumerate(rows):
            x1, x2 = int(gutter + index * width), int(gutter + (index + 1) * width)
            value = row[key] or "excluded"
            cv2.rectangle(frame, (x1 + 1, y), (x2 - 1, y + band), CLASS_BGR.get(value, (0xB4, 0xC6, 0xC9)), -1)
            if index == current:
                cv2.rectangle(frame, (x1 + 1, y), (x2 - 1, y + band), (243, 238, 223), 2)
            cv2.putText(frame, value[:4], (x1 + 8, y + 21), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (243, 238, 223) if value != "excluded" else (0x5E, 0x67, 0x64), 1, cv2.LINE_AA)


def start_encoder(out: Path, fps: float) -> subprocess.Popen:
    out.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{RENDER_WIDTH}x{RENDER_HEIGHT}", "-r", f"{fps}", "-i", "-",
         "-c:v", "libx264", "-preset", "medium", "-crf", str(CRF), "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(out)],
        stdin=subprocess.PIPE,
    )


def finish(encoder: subprocess.Popen, out: Path) -> None:
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise SystemExit(f"ffmpeg failed writing {out}")
    if out.stat().st_size < 100_000:
        raise SystemExit(f"{out} looks empty ({out.stat().st_size} bytes)")


def box_label(frame: np.ndarray, x1: int, y1: int, text: str, colour: tuple[int, int, int]) -> None:
    """A filled tab above the box. Never above the banner."""
    width = 10 + 9 * len(text)
    top = max(66, y1 - 22)
    cv2.rectangle(frame, (x1, top), (x1 + width, top + 20), colour, -1)
    cv2.putText(frame, text, (x1 + 5, top + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (243, 238, 223), 1, cv2.LINE_AA)


def draw_inset(frame: np.ndarray, region: tuple[int, int, int, int], source: np.ndarray | None = None):
    """Paste an enlarged crop top right; returns (x, y, zoom, origin) for mapping points.

    ``source`` is cropped instead of ``frame`` when the box and its tab are already drawn
    on ``frame`` but should not be enlarged into the inset.
    """
    x1, y1, x2, y2 = region
    span = max(x2 - x1, y2 - y1, INSET_MIN_REGION)
    middle_x, middle_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    x1 = max(0, int(middle_x - span / 2))
    y1 = max(0, int(middle_y - span / 2))
    x2 = min(frame.shape[1], x1 + span)
    y2 = min(frame.shape[0], y1 + span)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    crop = (frame if source is None else source)[y1:y2, x1:x2].copy()
    zoom = min(INSET_SIZE / crop.shape[1], INSET_SIZE / crop.shape[0])
    view = cv2.resize(crop, (int(crop.shape[1] * zoom), int(crop.shape[0] * zoom)),
                      interpolation=cv2.INTER_NEAREST)
    left, top = RENDER_WIDTH - view.shape[1] - INSET_MARGIN, INSET_TOP
    frame[top:top + view.shape[0], left:left + view.shape[1]] = view
    cv2.rectangle(frame, (left - 2, top - 2), (left + view.shape[1] + 1, top + view.shape[0] + 1),
                  (243, 238, 223), 2)
    return left, top, zoom, (x1, y1)


def render(info: dict, out: Path) -> None:
    source = Path(info["path"])
    rows = info["windows"]
    _, fps, _ = video_size(source)
    limit = int(round(info["rendered_s"] * fps))
    encoder = start_encoder(out, fps)
    cap = cv2.VideoCapture(str(source))
    source_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or RENDER_WIDTH
    scale = RENDER_WIDTH / float(source_width)
    index = 0
    while index < limit:
        ok, frame = cap.read()
        if not ok:
            break
        time_s = index / fps
        current = min(int(time_s // 5.0), len(rows) - 1)
        row = rows[current]
        frame = cv2.resize(frame, (RENDER_WIDTH, RENDER_HEIGHT), interpolation=cv2.INTER_AREA)
        if row["box"] is not None:
            x1, y1, x2, y2 = [int(value * scale) for value in row["box"]]
            colour = (0x3B, 0x5A, 0xC4) if row["source"] == "yolo" else (0x8F, 0x96, 0x87)
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 3)
            label = f"yolo {row['detector_conf']:.2f}" if row["source"] == "yolo" else "background"
            box_label(frame, x1, y1, label, colour)
        banner(frame, f"Task 1  {row['label']}  {row['confidence']:.2f}",
               f"{row['start']:.0f}-{row['end']:.0f} s   Task 2 box: {row['source']}")
        timeline_strip(frame, rows, current)
        encoder.stdin.write(frame.tobytes())
        index += 1
    cap.release()
    finish(encoder, out)


def render_detection(info: dict, detector, out: Path) -> None:
    """Task 2 on its own: the detector runs on every rendered frame, so the box tracks the animal."""
    source = Path(info["path"])
    rows = info["windows"]
    _, fps, _ = video_size(source)
    limit = int(round(info["rendered_s"] * fps))
    fallback = D.per_clip_boxes(source)
    encoder = start_encoder(out, fps)
    cap = cv2.VideoCapture(str(source))
    source_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or RENDER_WIDTH
    scale = RENDER_WIDTH / float(source_width)
    index = 0
    while index < limit:
        ok, frame = cap.read()
        if not ok:
            break
        time_s = index / fps
        current = min(int(time_s // 5.0), len(rows) - 1)
        box, confidence = D.yolo_top_box(detector, frame)
        label, colour = "no box", RUST_BGR
        if box is not None and confidence >= D.YOLO_TRUST:
            label = f"YOLO {confidence:.2f}"
        else:
            box = D.nearest_box(fallback, time_s)
            label, colour = "background model", MOSS_BGR
        frame = cv2.resize(frame, (RENDER_WIDTH, RENDER_HEIGHT), interpolation=cv2.INTER_AREA)
        if box is not None:
            clean = frame.copy()
            x1, y1, x2, y2 = [int(value * scale) for value in box]
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 3)
            box_label(frame, x1, y1, label, colour)
            pad = max(12, int(0.3 * max(x2 - x1, y2 - y1)))
            inset = draw_inset(frame, (x1 - pad, y1 - pad, x2 + pad, y2 + pad), clean)
            if inset is not None:
                left, top, zoom, (cx, cy) = inset
                cv2.rectangle(frame,
                              (int(left + (x1 - cx) * zoom), int(top + (y1 - cy) * zoom)),
                              (int(left + (x2 - cx) * zoom), int(top + (y2 - cy) * zoom)),
                              colour, 2)
        banner(frame, f"Task 2  animal detection  {label}",
               f"{rows[current]['start']:.0f}-{rows[current]['end']:.0f} s   "
               f"rust: trained YOLO   moss: per-clip background model")
        encoder.stdin.write(frame.tobytes())
        index += 1
    cap.release()
    finish(encoder, out)


def silhouette_geometry(sample: features.Sample, scale: float) -> dict:
    """The mask's contour, centroid and two axes, mapped from analysis pixels to render pixels."""
    factor = scale / detect.SCALE
    x1, y1 = sample.box[0], sample.box[1]
    contours, _ = cv2.findContours(sample.mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    mapped = [np.round((contour.reshape(-1, 2) + (x1, y1)) * factor).astype(np.int32)
              for contour in contours]
    rows, columns = np.nonzero(sample.mask)
    centroid = ((float(columns.mean()) + x1) * factor, (float(rows.mean()) + y1) * factor)
    theta = math.radians(sample.body_angle_deg)
    along = 0.5 * max(sample.box[2] - sample.box[0], sample.box[3] - sample.box[1]) * factor
    limb = 0.5 * sample.limb_extent_px * scale
    body = ((centroid[0] - math.cos(theta) * along, centroid[1] - math.sin(theta) * along),
            (centroid[0] + math.cos(theta) * along, centroid[1] + math.sin(theta) * along))
    across = ((centroid[0] + math.sin(theta) * limb, centroid[1] - math.cos(theta) * limb),
              (centroid[0] - math.sin(theta) * limb, centroid[1] + math.cos(theta) * limb))
    return {"contours": mapped, "centroid": centroid, "body": body, "limb": across}


def draw_geometry(frame: np.ndarray, parts: dict, mapper=None) -> None:
    """Draw the silhouette, the body axis and the limb extent, optionally through an inset mapping."""
    point = mapper or (lambda p: (int(p[0]), int(p[1])))
    for contour in parts["contours"]:
        cv2.polylines(frame, [np.array([point(p) for p in contour], np.int32)], True,
                      MOSS_BGR, 2, cv2.LINE_AA)
    cv2.line(frame, point(parts["body"][0]), point(parts["body"][1]), RUST_BGR, 3, cv2.LINE_AA)
    cv2.line(frame, point(parts["limb"][0]), point(parts["limb"][1]), (0xDF, 0xEE, 0xF3), 2, cv2.LINE_AA)
    cv2.circle(frame, point(parts["centroid"]), 4, (0xDF, 0xEE, 0xF3), -1, cv2.LINE_AA)


def render_features(info: dict, samples: list[features.Sample], out: Path) -> None:
    """Task 3 on its own: the silhouette, the body axis, the limb extent and the four numbers."""
    source = Path(info["path"])
    rows = info["windows"]
    _, fps, _ = video_size(source)
    limit = int(round(info["rendered_s"] * fps))
    times = [sample.time_s for sample in samples]
    encoder = start_encoder(out, fps)
    cap = cv2.VideoCapture(str(source))
    source_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or RENDER_WIDTH
    scale = RENDER_WIDTH / float(source_width)
    index = 0
    while index < limit:
        ok, frame = cap.read()
        if not ok:
            break
        time_s = index / fps
        current = min(int(time_s // 5.0), len(rows) - 1)
        row = rows[current]
        frame = cv2.resize(frame, (RENDER_WIDTH, RENDER_HEIGHT), interpolation=cv2.INTER_AREA)
        found = bisect.bisect_right(times, time_s) - 1
        if found >= 0:
            sample = samples[found]
            parts = silhouette_geometry(sample, scale)
            factor = scale / detect.SCALE
            x1, y1, x2, y2 = [int(value * factor) for value in sample.box]
            clean = frame.copy()
            cv2.rectangle(frame, (x1, y1), (x2, y2), FOREST_BGR, 2)
            draw_geometry(frame, parts)
            pad = max(12, int(0.3 * max(x2 - x1, y2 - y1)))
            inset = draw_inset(frame, (x1 - pad, y1 - pad, x2 + pad, y2 + pad), clean)
            if inset is not None:
                left, top, zoom, (cx, cy) = inset
                draw_geometry(frame, parts, lambda p: (int(left + (p[0] - cx) * zoom),
                                                       int(top + (p[1] - cy) * zoom)))
        values = row["features"]
        shown = {name: "-" if value is None else f"{value:.0f}" for name, value in values.items()
                 if name != "box_area_px"}
        banner(frame, "Task 3  silhouette features",
               f"window median   body {shown['body_angle_deg']}  head {shown['head_angle_deg']}"
               f"  speed {shown['speed_px_s']} px/s  limb {shown['limb_extent_px']} px")
        encoder.stdin.write(frame.tobytes())
        index += 1
    cap.release()
    finish(encoder, out)


def reported_numbers() -> dict:
    """Read every number the page states, so it cannot drift from the reports."""
    readme = (ROOT / "README.md").read_text()
    features_md = (ROOT / "outputs/features/features.md").read_text()
    patterns = {
        "detector_map50": (readme, r"mAP50 ([\d.]+) on validation"),
        "feature_f1": (readme, r"macro F1 ([\d.]+) on the locked test windows"),
        "heldout_hit": (readme, r"finds the animal in (\d+) percent of held-out frames"),
        "feature_windows": (features_md, r"Windows: train \d+, validation \d+, test (\d+)"),
        "feature_missing": (features_md, r"no silhouette at all: train \d+, validation \d+, test (\d+)"),
        "activity_hours": (readme, r"([\d.]+)\s+hours of banded civet"),
        "activity_present": (readme, r"in view ([\d.]+) percent of the recorded night"),
        "activity_active": (readme, r"moving ([\d.]+) percent of\s+it"),
        "activity_f1": (readme, r"macro F1 ([\d.]+) on the camera it never saw"),
        "activity_majority": (readme, r"against ([\d.]+) for\s+always answering Active"),
    }
    found = {}
    for key, (text, pattern) in patterns.items():
        match = re.search(pattern, text)
        if not match:
            raise SystemExit(f"cannot read {key} from the reports; update build_site.py")
        found[key] = match.group(1)
    return found


def verified_segments(clip: dict) -> str:
    out = []
    for row in clip["windows"]:
        value = row["verified"]
        if value is None:
            out.append(f'<span class="seg excluded" title="{row["start"]}-{row["end"]} s: {row["excluded"]}">'
                       f'<span>excluded</span></span>')
        else:
            out.append(f'<span class="seg" style="--c:{CLASS_HEX[value]}" '
                       f'title="{row["start"]}-{row["end"]} s: {value}"><span>{value}</span></span>')
    return "".join(out)


def model_segments(clip: dict) -> str:
    return "".join(
        f'<button class="seg" data-at="{row["start"]}" style="--c:{CLASS_HEX[row["label"]]}" '
        f'title="{row["start"]}-{row["end"]} s: {row["label"]} {row["confidence"]:.2f}">'
        f'<span>{row["label"]}</span></button>' for row in clip["windows"])


def page(clips: list[dict], evaluation: dict, numbers: dict, task2: dict, task3: dict) -> str:
    test = evaluation["test"]
    classes = test["classes"]
    clip_json = json.dumps(clips, separators=(",", ":"))
    task_json = json.dumps({"task2": task2["windows"], "task3": task3["windows"]},
                           separators=(",", ":"))
    tabs = "\n".join(
        f'<button class="tab" data-clip="{clip["id"]}" title="{clip["id"]}">{clip["group"]}'
        f' &middot; {clip["id"][-6:]}</button>' for clip in clips)
    stages = "\n".join(
        f'''<div class="stage" data-clip="{clip["id"]}">
  <video src="media/{clip["id"]}.mp4" muted loop playsinline preload="metadata"></video>
  <div class="timeline">
    <div class="band"><span class="bl">verified</span><div class="segs">{verified_segments(clip)}</div></div>
    <div class="band"><span class="bl">model</span><div class="segs">{model_segments(clip)}</div></div>
  </div>
  <div class="readout"><b class="now">-</b><span class="meta"></span></div>
</div>''' for clip in clips)
    table_rows = "\n".join(
        f"<tr><td>{name}</td><td>{test['support'][name]}</td><td>{test['precision'][name]:.3f}</td>"
        f"<td>{test['recall'][name]:.3f}</td><td>{test['f1'][name]:.3f}</td></tr>" for name in classes)
    matrix_rows = "\n".join(
        f"<tr><td>{name}</td>{''.join(f'<td class=num>{value}</td>' for value in test['matrix'][i])}</tr>"
        for i, name in enumerate(classes))

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Civet behavior detection: held-out demo</title>
<style>
  :root {{ --bg:#F3EEDF; --panel:#E1E6DA; --forest:#284236; --rust:#C45A3B; --moss:#87966F;
           --ink:#22241F; --muted:#64675E; --rule:#C9C6B4; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
    font:16px/1.55 "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif; }}
  .serif {{ font-family:inherit; }}
  .sans {{ font-family:"Helvetica Neue", Inter, system-ui, sans-serif; }}
  .wrap {{ max-width:1180px; margin:0 auto; padding:0 32px 96px; }}
  header {{ padding:56px 0 28px; border-bottom:2px solid var(--forest); }}
  .kicker {{ font:600 11px/1 "Helvetica Neue", sans-serif; letter-spacing:.18em;
    text-transform:uppercase; color:var(--rust); }}
  h1 {{ font-size:clamp(34px,4.4vw,54px); line-height:1.05; margin:14px 0 12px; font-weight:400; }}
  .lead {{ max-width:62ch; color:var(--muted); }}
  h2 {{ font-size:26px; font-weight:400; margin:0 0 6px; }}
  h2 .num {{ color:var(--rust); font:600 12px/1 "Helvetica Neue", sans-serif;
    letter-spacing:.16em; margin-right:12px; vertical-align:3px; }}
  section {{ padding:44px 0; border-bottom:1px solid var(--rule); }}
  section p {{ max-width:74ch; }}
  .row {{ display:grid; grid-template-columns:1.35fr .9fr; gap:28px; align-items:start; }}
  @media (max-width:900px) {{ .row {{ grid-template-columns:1fr; }} }}
  .tabs {{ display:flex; gap:8px; flex-wrap:wrap; margin:18px 0 12px; }}
  .tab {{ font:500 12px/1 "Helvetica Neue", sans-serif; letter-spacing:.06em; padding:9px 14px;
    border:1px solid var(--rule); background:transparent; color:var(--muted); cursor:pointer; }}
  .tab.on {{ background:var(--forest); border-color:var(--forest); color:var(--bg); }}
  .stage {{ display:none; }}
  .stage.on {{ display:block; }}
  video {{ width:100%; display:block; background:#10130F; }}
  .timeline {{ margin-top:8px; display:grid; gap:4px; }}
  .band {{ display:grid; grid-template-columns:62px 1fr; gap:8px; align-items:center; }}
  .bl {{ font:500 10px/1 "Helvetica Neue", sans-serif; letter-spacing:.12em; text-transform:uppercase;
    color:var(--muted); }}
  .segs {{ display:flex; gap:3px; }}
  .seg {{ flex:1 1 0; height:38px; border:0; padding:0 5px; cursor:pointer; background:var(--c);
    color:#F3EEDF; font:500 10px/1.15 "Helvetica Neue", sans-serif; text-align:left;
    display:flex; align-items:center; overflow:hidden; opacity:.85; }}
  .seg.excluded {{ background:var(--rule); color:var(--muted); cursor:default; }}
  .seg.current {{ opacity:1; outline:2px solid var(--ink); outline-offset:-2px; }}
  .readout {{ display:flex; align-items:baseline; gap:14px; flex-wrap:wrap; margin-top:12px;
    font-family:"Helvetica Neue", sans-serif; }}
  .readout b {{ font-size:19px; font-weight:500; color:var(--ink); }}
  .readout span {{ font-size:12.5px; line-height:1.5; color:var(--muted); }}
  .readout .meta {{ flex:1 1 300px; }}
  table {{ width:100%; border-collapse:collapse; margin:16px 0 6px;
    font:14px/1.4 "Helvetica Neue", sans-serif; }}
  th, td {{ text-align:left; padding:9px 10px; border-bottom:1px solid var(--rule); }}
  th {{ font-size:11px; letter-spacing:.1em; text-transform:uppercase; color:var(--muted);
    font-weight:600; }}
  td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  td.dim {{ color:var(--muted); }}
  .big {{ display:flex; gap:36px; flex-wrap:wrap; margin:20px 0 8px; }}
  .big div {{ min-width:150px; }}
  .big b {{ display:block; font-size:44px; line-height:1; font-weight:400; color:var(--forest); }}
  .big span {{ font:12px/1.4 "Helvetica Neue", sans-serif; color:var(--muted); }}
  .note {{ background:var(--panel); padding:16px 18px; font-size:14px; color:var(--muted);
    font-family:"Helvetica Neue", sans-serif; }}
  figure {{ margin:20px 0 6px; }}
  figure img {{ width:100%; display:block; border:1px solid var(--rule); }}
  figcaption, .caption {{ margin-top:8px; font:12.5px/1.5 "Helvetica Neue", sans-serif; color:var(--muted); max-width:74ch; }}
  .figgrid {{ display:grid; gap:22px; }}
  code {{ font:13px/1.4 ui-monospace, "SF Mono", Menlo, monospace; background:var(--panel);
    padding:2px 5px; }}
  pre {{ background:var(--panel); padding:14px 16px; overflow:auto; font-size:13px; }}
  footer {{ padding-top:26px; font:12px/1.6 "Helvetica Neue", sans-serif; color:var(--muted); }}
  a {{ color:var(--forest); }}
</style>
</head>
<body>
<div class="wrap">
<header>
  <div class="kicker">CMZ / Phase 2 / held-out demo</div>
  <h1>Civet behavior detection</h1>
  <p class="lead">Three tasks on five-second windows of camera-trap footage: behavior
  classification, YOLO detection of the animal, and silhouette features. A separate section
  measures presence and movement on {numbers['activity_hours']} hours of banded civet footage with no annotation.
  The clips below are precomputed from committed model outputs, so this page runs on any
  machine with a browser.</p>
</header>

<section>
  <div class="row">
    <div>
      <h2><span class="num">01</span>Prediction timeline</h2>
      <div class="tabs">{tabs}</div>
      {stages}
      <div class="readout"><span>Upper band is the verified annotation, lower band is what the model
      predicted. Grey windows cross an annotation boundary or a Transition, so the evaluation
      excludes them. Click a lower segment to jump to that window.</span></div>
    </div>
    <div>
      <p>Every five-second window is classified and shown against the clip's own timecodes.
      The lower band is the model's prediction, the upper band is the verified annotation.</p>
      <p>Three validation clips, one per class. The test clips are scored once in the evaluation
      and are not shown here.</p>
      <p>The page plays precomputed output. It needs a browser, not this project's GPU.</p>
    </div>
  </div>
</section>

<section>
  <h2><span class="num">02</span>Held-out behavior classification</h2>
  <p>{test['n']} locked test windows over 8 clips. The threshold was chosen on the validation
  windows and the test clips were decoded and scored once, after it was fixed.</p>
  <div class="big">
    <div><b>{test['macro_f1']:.3f}</b><span>macro F1 over 3 classes</span></div>
    <div><b>{test['n_abstain']}</b><span>abstentions</span></div>
    <div><b>{test['accuracy_decided']:.3f}</b><span>accuracy over decided windows</span></div>
  </div>
  <table>
    <thead><tr><th>Behavior</th><th>Support</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead>
    <tbody>{table_rows}</tbody>
  </table>
  <table>
    <thead><tr><th>true \\ pred</th>{"".join(f"<th class=num>{name}</th>" for name in classes)}<th class=num>abstain</th></tr></thead>
    <tbody>{matrix_rows}</tbody>
  </table>
</section>

<section>
  <h2><span class="num">03</span>YOLO detection</h2>
  <div class="row">
    <div>
      <div class="player" data-rows="task2">
        <video src="media/task2-detection.mp4" muted loop playsinline preload="metadata"></video>
        <div class="readout"><b class="now">-</b><span class="meta"></span></div>
      </div>
      <p class="caption">The detector runs on every frame here, not only on the middle of a
      window, so the box tracks the animal. Rust is the trained YOLO detector with its
      confidence; moss is the per-clip background model, which takes over under 0.50. The
      inset is the same frame enlarged.</p>
    </div>
    <div>
      <div class="big">
        <div><b>{numbers['detector_map50']}</b><span>mAP50 on validation</span></div>
        <div><b>{numbers['heldout_hit']}%</b><span>held-out frames where the animal is found</span></div>
      </div>
      <p>No bounding box was drawn by hand. A per-clip background model proposes the boxes,
      116 of 408 were audited visually and 113 contained the animal, then YOLO trains on them.
      The detector does not carry across recording days, so the demo falls back to the per-clip
      model and the box colour on the video says which one fired.</p>
      <p><a href="https://github.com/ChindanaiNaKub/civet-behavior-detection/blob/main/docs/TASKS-2-AND-3.md">Tasks 2 and 3 report</a></p>
    </div>
  </div>
</section>

<section>
  <h2><span class="num">04</span>Silhouette features</h2>
  <div class="row">
    <div>
      <div class="player" data-rows="task3">
        <video src="media/task3-features.mp4" muted loop playsinline preload="metadata"></video>
        <div class="readout"><b class="now">-</b><span class="meta"></span></div>
      </div>
      <p class="caption">Moss outline is the silhouette Task 3 measures, rust is its body axis,
      cream is the limb extent across that axis. The numbers under the clip are the window
      medians the classifier sees; the geometry is redrawn from the nearest sampled frame.
      This clip is from the locked test split, scored once before the overlay was drawn, and
      the overlay shows the measurement only, not the classifier's guess.</p>
    </div>
    <div>
      <div class="big">
        <div><b>{numbers['feature_f1']}</b><span>macro F1, locked test</span></div>
        <div><b>0.190</b><span>majority-guess baseline</span></div>
      </div>
      <p>Keypoints are replaced by four silhouette features: body angle, head angle, image
      speed and limb extent. The gain over the majority guess is in the rare classes, not in
      getting most windows right.</p>
      <p><a href="https://github.com/ChindanaiNaKub/civet-behavior-detection/blob/main/docs/FEATURES.md">The four features</a></p>
    </div>
  </div>
</section>

<section>
  <h2><span class="num">05</span>Banded civet activity, with no annotation</h2>
  <p>{numbers['activity_hours']} hours from four fixed cameras on the night of 2026-04-05, and not one
  human label. The Phase 2 behavior classes cannot be scored on it, so this scan measures what a fixed
  camera can see by itself: is the animal in view, and is it moving. The same classifier is then trained
  on that measurement. These numbers are not comparable with the three-class result above.</p>
  <div class="big">
    <div><b>{numbers['activity_hours']}</b><span>hours, one night, four cameras</span></div>
    <div><b>{numbers['activity_present']}%</b><span>of the night in view</span></div>
    <div><b>{numbers['activity_active']}%</b><span>of the night moving</span></div>
    <div><b>{numbers['activity_f1']}</b><span>macro F1 on camera 4, held out</span></div>
  </div>
  <div class="figgrid">
    <figure>
      <img src="media/activity-timeline.png" alt="One night on four fixed cameras. Teal is the animal moving, amber is present and still, grey is absent.">
      <figcaption>Each row is one camera on the same clock. The empty room is the per-pixel median of the
      whole clip, so a box survives only when the next second corroborates it.</figcaption>
    </figure>
    <figure>
      <img src="media/activity-by-hour.png" alt="Active minutes per clock hour, four cameras, peaking between 23:00 and 02:00.">
      <figcaption>Active minutes per clock hour. Activity clusters between 23:00 and 02:00, with a
      second cluster near 21:00.</figcaption>
    </figure>
  </div>
  <p>Always answering Active would score {numbers['activity_majority']}, so the model carries 0.21 macro F1
  on a viewpoint it never saw. The visual audit read 24 windows of each state at full resolution: 24 Active,
  23 Inactive and 22 Absent held what the label said.
  <a href="https://github.com/ChindanaiNaKub/civet-behavior-detection/blob/main/docs/ACTIVITY-REPORT.md">Activity report</a></p>
</section>

<section>
  <h2><span class="num">06</span>What this does not prove</h2>
  <div class="note">
    <p>The test split holds {test['n']} windows, so a single window moves macro F1 by several
    points. The validation split is perfectly separated, which says the split is easy, not that
    the classifier generalizes.</p>
    <p>Task 2's detector fires on {numbers['heldout_hit']} percent of held-out frames and about
    half of those boxes sit on the animal.</p>
    <p>Task 3's features are not keypoints. Body angle and head angle are two axes of one blob,
    and {numbers['feature_missing']} of the {numbers['feature_windows']} test windows have no
    silhouette at all, so their rows are the training median rather than measured geometry.</p>
    <p>The banded civet score is measured against the labels the scan produced, on one night from four
    cameras. It is a check that the classifier learned that measurement, not accuracy against a person,
    and it says nothing about a second night.</p>
  </div>
</section>

<section>
  <h2><span class="num">07</span>Reproduce</h2>
  <pre>python3 -m pip install -r requirements.txt
python3 scripts/evaluate.py
python3 scripts/demo_all.py data/raw/Eating/25690519193547292.mp4
python3 scripts/activity.py --clips &lt;folder&gt; --out outputs/activity
python3 scripts/activity_report.py
python3 scripts/build_site.py</pre>
  <p><a href="https://chindanainakub.github.io/civet-behavior-detection/presentation/CMZ-Phase-2-Deck.pdf">Deck (PDF, 28 slides)</a> ·
  <a href="https://chindanainakub.github.io/civet-behavior-detection/presentation/">Deck (reveal.js)</a> ·
  <a href="https://github.com/ChindanaiNaKub/civet-behavior-detection">Repository</a> ·
  <a href="https://github.com/ChindanaiNaKub/civet-behavior-detection/blob/main/docs/EVALUATION-REPORT.md">Evaluation report</a> ·
  <a href="https://github.com/ChindanaiNaKub/civet-behavior-detection/blob/main/docs/TASKS-2-AND-3.md">Tasks 2 and 3</a> ·
  <a href="https://github.com/ChindanaiNaKub/civet-behavior-detection/blob/main/docs/ACTIVITY-REPORT.md">Activity report</a></p>
</section>

<footer>Built {date.today().isoformat()} by <code>scripts/build_site.py</code>.
Numbers come from <code>outputs/evaluation.json</code> and the pipeline reports.</footer>
</div>
<script>
const CLIPS = {clip_json};
const TASKS = {task_json};
const stages = [...document.querySelectorAll('.stage')];
const tabs = [...document.querySelectorAll('.tab')];
let active = CLIPS[0].id;

function show(id) {{
  active = id;
  tabs.forEach(t => t.classList.toggle('on', t.dataset.clip === id));
  stages.forEach(s => {{
    const on = s.dataset.clip === id;
    s.classList.toggle('on', on);
    const v = s.querySelector('video');
    if (on) {{ v.play().catch(() => {{}}); }} else {{ v.pause(); v.currentTime = 0; }}
  }});
  paint();
}}

function rowsOf(id) {{
  const clip = CLIPS.find(c => c.id === active);
  return clip.windows;
}}

function paint() {{
  const stage = stages.find(s => s.dataset.clip === active);
  const video = stage.querySelector('video');
  const rows = rowsOf(active);
  const index = rows.findIndex(r => video.currentTime >= r.start && video.currentTime < r.end);
  const at = index < 0 ? rows.length - 1 : index;
  const row = rows[at];
  stage.querySelectorAll('.seg').forEach(seg => seg.classList.remove('current'));
  stage.querySelectorAll('.segs').forEach(segs => segs.children[at].classList.add('current'));
  stage.querySelector('b.now').textContent = `${{row.label}} ${{row.confidence.toFixed(2)}}`;
  stage.querySelector('.meta').textContent =
    `${{row.start.toFixed(0)}}-${{row.end.toFixed(0)}} s · verified ${{row.verified || 'excluded from scoring'}}` +
    ` · box ${{row.source}}` +
    (row.features.speed_px_s === null ? ' · no silhouette' : ` · speed ${{row.features.speed_px_s}} px/s`) +
    (row.excluded ? ` · not scored: ${{row.excluded.replace('_', ' ')}}` : '');
}}

document.querySelectorAll('.player').forEach(player => {{
  const rows = TASKS[player.dataset.rows];
  const video = player.querySelector('video');
  const paintOne = () => {{
    const index = rows.findIndex(r => video.currentTime >= r.start && video.currentTime < r.end);
    const row = rows[Math.max(0, index)];
    const f = row.features;
    const value = v => (v === null ? '-' : v);
    const span = `${{row.start.toFixed(0)}}-${{row.end.toFixed(0)}} s`;
    if (player.dataset.rows === 'task2') {{
      player.querySelector('b.now').textContent = row.source === 'yolo'
        ? `YOLO ${{row.detector_conf.toFixed(2)}}` : row.source === 'motion' ? 'background model' : 'no box';
      player.querySelector('.meta').textContent = span
        + (row.box ? ` · box ${{row.box[2] - row.box[0]}}×${{row.box[3] - row.box[1]}} px` : ' · nothing found');
    }} else {{
      player.querySelector('b.now').textContent = span;
      player.querySelector('.meta').textContent = 'window median'
        + ` · body ${{value(f.body_angle_deg)}}° · head ${{value(f.head_angle_deg)}}°`
        + ` · speed ${{value(f.speed_px_s)}} px/s · limb ${{value(f.limb_extent_px)}} px`;
    }}
  }};
  video.addEventListener('timeupdate', paintOne);
  paintOne();
}});

tabs.forEach(t => t.addEventListener('click', () => show(t.dataset.clip)));
stages.forEach(s => {{
  s.querySelectorAll('.seg').forEach(seg => seg.addEventListener('click', () => {{
    s.querySelector('video').currentTime = Number(seg.dataset.at) + 0.4;
    paint();
  }}));
  s.querySelector('video').addEventListener('timeupdate', paint);
}});
show(CLIPS[0].id);
</script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "site")
    parser.add_argument("--clips", type=Path, nargs="*", default=None)
    parser.add_argument("--model", type=Path, default=ROOT / "outputs/model/model.pt")
    parser.add_argument("--detector", type=Path, default=ROOT / "outputs/detection/model/round2.pt")
    parser.add_argument("--evaluation", type=Path, default=ROOT / "outputs/evaluation.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task2-clip", type=Path,
                        default=ROOT / "data/raw/Resting/25690519195143424.mp4",
                        help="clip for the Task 2 detection video")
    parser.add_argument("--task3-clip", type=Path,
                        default=ROOT / "data/raw/Eating/25690519193422801.mp4",
                        help="clip for the Task 3 features video")
    args = parser.parse_args()

    clips = args.clips or [
        ROOT / "data/raw/Eating/25690519193547292.mp4",
        ROOT / "data/raw/Locomotion/25690518091030781.mp4",
        ROOT / "data/raw/Resting/25690519195143424.mp4",
    ]
    for path in clips:
        if not path.exists():
            raise SystemExit(f"{path} not found")

    evaluation = json.loads(args.evaluation.read_text())
    threshold = float(evaluation["threshold"])
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model, config = M.load(args.model, str(device))
    from ultralytics import YOLO
    detector = YOLO(str(args.detector))

    args.out.mkdir(parents=True, exist_ok=True)

    infos = []
    for path in clips:
        print(f"analysing {path.name}", flush=True)
        info = analyse(path, model, config, detector, threshold)
        print(f"  rendering {info['rendered_s']:.0f} s", flush=True)
        render(info, args.out / "media" / f"{info['id']}.mp4")
        infos.append(info)

    print(f"analysing {args.task2_clip.name}", flush=True)
    task2 = analyse(args.task2_clip, model, config, detector, threshold)
    render_detection(task2, detector, args.out / "media" / "task2-detection.mp4")

    print(f"analysing {args.task3_clip.name}", flush=True)
    task3 = analyse(args.task3_clip, model, config, detector, threshold)
    samples = features.clip_samples(args.task3_clip)
    render_features(task3, samples, args.out / "media" / "task3-features.mp4")

    for name in ("activity-timeline.png", "activity-by-hour.png"):
        shutil.copy(ROOT / "docs/figures" / name, args.out / "media" / name)

    (args.out / "index.html").write_text(page(infos, evaluation, reported_numbers(), task2, task3))
    print(f"wrote {args.out}/index.html and {len(infos) + 2} clips")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
