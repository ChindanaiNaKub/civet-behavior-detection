#!/usr/bin/env python3
"""Train and self-train the Phase 2 civet detector.

Task 2 needs a detector, but the only boxes that exist are the classical seeds
from scripts/detect.py, and those include false positives on empty floor. Round 1
learns from the seeds; round 2 then lets round 1 re-label every sampled frame of
the train and validation clips, so the retrained model sees far more frames and
mostly self-agreed boxes. Test clips never enter either round: they are only
scored at the end, on frames no round has seen.

The dataset listings and data.yaml produced here are the reproducible input for
round 1. Round 2 needs a separate copy under /tmp because it mixes in new frames
and replaced labels, and the seed labels must stay intact.

Usage:
    python3 scripts/train_detector.py
    python3 scripts/train_detector.py --epochs 40 --work /tmp/opencode/yolo
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

# Tunables. 60 epochs at 8 images per batch fits the 4 GiB laptop GPU; patience
# stops a run that has already plateaued.
SAMPLE_FPS = 1.0
IMGSZ = 640
EPOCHS = 60
BATCH = 8
PATIENCE = 15
SEED = 0
WORKERS = 2
DEVICE = 0

SEED_CONF = 0.25            # inference floor, so weak boxes are still candidates
PSEUDO_CONF = 0.6           # a box needs this to be written without seed agreement
AGREE_IOU = 0.3             # model box this close to the seed label agrees with it
HELDOUT_FRAMES = 20         # annotated frames spread over the test clips

SPLITS = Path("data/splits.csv")
DATASET = Path("outputs/detection/dataset")
MODEL_DIR = Path("outputs/detection/model")
HELDOUT_DIR = Path("outputs/detection/heldout")
HELDOUT_CSV = Path("outputs/detection/heldout.csv")
TRAIN_LOG = Path("outputs/detection/train-log.md")
CLIPS = Path("data/raw")
WEIGHTS = "yolo11n.pt"

# The three floor-only boxes confirmed by eye in the validation clip. 56.04 has
# no seed label; the clip's seed label nearest that second is 66.04.
KNOWN_FALSE_POSITIVES = ("25690519193547292", (56.04, 67.04, 68.04))


# --- dataset plumbing --------------------------------------------------------


def read_splits(path: Path) -> dict[str, str]:
    """clip_id to split, refusing duplicate or unknown split values."""
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    splits: dict[str, str] = {}
    for row in rows:
        clip_id, split = row["clip_id"].strip(), row["split"].strip()
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"{clip_id}: unknown split {split!r}")
        if clip_id in splits:
            raise ValueError(f"{clip_id}: listed twice")
        splits[clip_id] = split
    return splits


def split_of(stem: str, splits: dict[str, str]) -> str:
    """Split of an image stem named <clip_id>_<time_s>."""
    clip_id = stem.rsplit("_", 1)[0]
    if clip_id not in splits:
        raise KeyError(f"{stem}: clip not in {SPLITS}")
    return splits[clip_id]


def write_yaml(path: Path, root: Path, train: Path, val: Path) -> None:
    """Single-class dataset description, with absolute listings."""
    path.write_text(
        yaml.safe_dump(
            {
                "path": str(root.resolve()),
                "train": str(train.resolve()),
                "val": str(val.resolve()),
                "names": ["civet"],
            },
            sort_keys=False,
        )
    )


def build_seed_dataset(splits: dict[str, str]) -> dict[str, int]:
    """Write train.txt/val.txt/data.yaml from the seed images, excluding test."""
    images = sorted((DATASET / "images").glob("*.jpg"))
    train, val, test = [], [], []
    for image in images:
        split = split_of(image.stem, splits)
        {"train": train, "validation": val, "test": test}[split].append(image)
    if len(test) + len(train) + len(val) != len(images):
        raise AssertionError("an image landed in no split")
    train_clips = {p.stem.rsplit("_", 1)[0] for p in train}
    val_clips = {p.stem.rsplit("_", 1)[0] for p in val}
    test_clips = {p.stem.rsplit("_", 1)[0] for p in test}
    if train_clips & test_clips or val_clips & test_clips:
        raise AssertionError("a test clip leaked into train or validation")

    # Seed crops without an image would train on a dangling label; fail loudly.
    for group in (train, val):
        for image in group:
            if not (DATASET / "labels" / f"{image.stem}.txt").is_file():
                raise AssertionError(f"missing label for {image.name}")

    (DATASET / "train.txt").write_text("".join(f"{p.resolve()}\n" for p in train))
    (DATASET / "val.txt").write_text("".join(f"{p.resolve()}\n" for p in val))
    write_yaml(DATASET / "data.yaml", DATASET, DATASET / "train.txt", DATASET / "val.txt")

    counts = {
        "train_clips": len(train_clips),
        "val_clips": len(val_clips),
        "test_clips": len(test_clips),
        "train_images": len(train),
        "val_images": len(val),
        "test_images": len(test),
    }
    print(
        "seed dataset: {train_images} train images from {train_clips} clips, "
        "{val_images} val images from {val_clips} clips, "
        "{test_images} test images from {test_clips} clips excluded".format(**counts),
        flush=True,
    )
    return counts


def label_to_xyxy(line: str) -> tuple[float, float, float, float] | None:
    """Normalised YOLO label to normalised corners."""
    parts = line.split()
    if len(parts) != 5:
        return None
    cx, cy, width, height = (float(value) for value in parts[1:])
    return cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2


def iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """IoU of two normalised xyxy boxes; the axes cancel, so aspect does not matter."""
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    overlap = max(0.0, right - left) * max(0.0, bottom - top)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap
    return overlap / union if union > 0 else 0.0


def sampled_frames(path: Path):
    """Yield (time_s, full-resolution frame) at SAMPLE_FPS, like scripts/detect.py."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    step = max(1, int(round(fps / SAMPLE_FPS)))
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % step == 0:
            yield index / fps, frame
        index += 1
    cap.release()


def detections(result, frame_shape) -> list[tuple[float, tuple[float, float, float, float]]]:
    """Every detection as (conf, normalised xyxy), highest confidence first."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    height, width = frame_shape[:2]
    found = []
    for conf, (x1, y1, x2, y2) in zip(boxes.conf.cpu().numpy(), boxes.xyxy.cpu().numpy()):
        found.append(
            (
                float(conf),
                (
                    min(max(x1 / width, 0.0), 1.0),
                    min(max(y1 / height, 0.0), 1.0),
                    min(max(x2 / width, 0.0), 1.0),
                    min(max(y2 / height, 0.0), 1.0),
                ),
            )
        )
    return sorted(found, key=lambda item: item[0], reverse=True)


def best_box(result, frame_shape) -> tuple[float, tuple[float, float, float, float]] | None:
    """Highest-confidence detection as (conf, normalised xyxy), or None."""
    found = detections(result, frame_shape)
    return found[0] if found else None


def write_label(path: Path, box: tuple[float, float, float, float]) -> None:
    """Normalised xyxy corners to one class-0 YOLO row."""
    x1, y1, x2, y2 = box
    path.write_text(f"0 {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}\n")


# --- self-training -----------------------------------------------------------


def build_round2(
    weights: Path,
    splits: dict[str, str],
    work: Path,
) -> tuple[Path, dict[str, int]]:
    """Re-label every sampled train/validation frame with round 1 and build its dataset."""
    root = work / "dataset-round2"
    for directory in (root / "images", root / "labels"):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)

    # Seed labels stay in the repo; this copy may be replaced or extended.
    seed_labels = {p.stem: p for p in (DATASET / "labels").glob("*.txt")}
    counts = {"train_seed": 0, "val_seed": 0, "kept": 0, "replaced": 0, "added": 0, "left": 0}

    model = YOLO(str(weights))
    clips = sorted(
        path
        for path in CLIPS.rglob("*.mp4")
        if splits.get(path.stem) in {"train", "validation"}
    )
    for clip in clips:
        split = splits[clip.stem]
        frames = 0
        for time_s, frame in sampled_frames(clip):
            stem = f"{clip.stem}_{time_s:.2f}"
            seed_path = seed_labels.get(stem)
            seed_box = label_to_xyxy(seed_path.read_text()) if seed_path else None
            if seed_box is not None:
                counts[f"{'val' if split == 'validation' else 'train'}_seed"] += 1

            result = model.predict(
                frame, conf=SEED_CONF, imgsz=IMGSZ, device=DEVICE, verbose=False
            )[0]
            found = best_box(result, frame.shape)
            seed_target = root / "images" / f"{stem}.jpg"
            label_target = root / "labels" / f"{stem}.txt"

            if seed_box is not None:
                if not seed_target.exists():
                    source = DATASET / "images" / f"{stem}.jpg"
                    if not source.is_file():
                        raise AssertionError(f"missing seed image for {stem}")
                    seed_target.symlink_to(source.resolve())
                label_target.write_text(seed_path.read_text())
                if found is not None and iou(found[1], seed_box) < AGREE_IOU:
                    if found[0] > PSEUDO_CONF:
                        write_label(label_target, found[1])
                        counts["replaced"] += 1
                    else:
                        counts["left"] += 1
                else:
                    counts["kept"] += 1
            elif found is not None and found[0] >= PSEUDO_CONF:
                ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if not ok:
                    raise RuntimeError(f"cannot encode {stem}")
                encoded.tofile(str(root / "images" / f"{stem}.jpg"))
                write_label(label_target, found[1])
                counts["added"] += 1
            frames += 1
        print(f"round2 {split} {clip.stem}: {frames} frames labelled", flush=True)

    train_txt, val_txt = root / "train.txt", root / "val.txt"
    train_images, val_images = [], []
    for image in sorted((root / "images").glob("*.jpg")):
        target = val_images if splits[image.stem.rsplit("_", 1)[0]] == "validation" else train_images
        target.append(image)
    train_txt.write_text("".join(f"{p.resolve()}\n" for p in train_images))
    val_txt.write_text("".join(f"{p.resolve()}\n" for p in val_images))
    write_yaml(root / "data.yaml", root, train_txt, val_txt)

    counts["train_images"] = len(train_images)
    counts["val_images"] = len(val_images)
    print(
        "round2 dataset: {train_images} train / {val_images} val images "
        "(kept {kept}, replaced {replaced}, added {added} seed labels, left {left} disagreed seeds)".format(
            **counts
        ),
        flush=True,
    )
    return root / "data.yaml", counts


# --- training and scoring ----------------------------------------------------


def train_round(weights: Path, data: Path, name: str, work: Path, epochs: int) -> dict:
    """Train one round from an image list and copy the best weights into the repo."""
    model = YOLO(str(weights))
    started = time.time()
    model.train(
        data=str(data),
        epochs=epochs,
        imgsz=IMGSZ,
        batch=BATCH,
        patience=PATIENCE,
        seed=SEED,
        workers=WORKERS,
        device=DEVICE,
        project=str(work),
        name=name,
        exist_ok=True,
        verbose=True,
    )
    wall = time.time() - started

    with (work / name / "results.csv").open(newline="") as handle:
        rows = [{key.strip(): value for key, value in row.items()} for row in csv.DictReader(handle)]
    # best.pt is the highest-fitness epoch, not the last one; report what was saved.
    best_row = max(
        rows,
        key=lambda row: 0.1 * float(row["metrics/mAP50(B)"]) + 0.9 * float(row["metrics/mAP50-95(B)"]),
    )
    best = work / name / "weights" / "best.pt"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    target = MODEL_DIR / f"{name}.pt"
    shutil.copy2(best, target)
    stats = {
        "name": name,
        "epochs": len(rows),
        "best_epoch": int(best_row["epoch"]),
        "wall_s": wall,
        "mAP50": float(best_row["metrics/mAP50(B)"]),
        "mAP50-95": float(best_row["metrics/mAP50-95(B)"]),
        "weights": target,
    }
    print(
        f"{name}: {stats['epochs']} epochs (best {stats['best_epoch']}), {wall / 60:.1f} min, "
        f"mAP50 {stats['mAP50']:.3f}, mAP50-95 {stats['mAP50-95']:.3f}",
        flush=True,
    )
    return stats


def save_annotated(image: np.ndarray, box: tuple[float, ...], conf: float, path: Path) -> None:
    """Draw the best box so a human can see what the held-out run saw."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in (box[0] * width, box[1] * height, box[2] * width, box[3] * height))
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 0), 2)
    cv2.putText(
        image, f"civet {conf:.2f}", (x1, max(y1 - 6, 12)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2, cv2.LINE_AA,
    )
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise RuntimeError(f"cannot write {path}")


def heldout_check(weights: Path, splits: dict[str, str]) -> dict:
    """Score round 2 on the test clips and annotate a spread of their frames."""
    model = YOLO(str(weights))
    test_clips = sorted(path for path in CLIPS.rglob("*.mp4") if splits.get(path.stem) == "test")
    if len(test_clips) != 8:
        raise AssertionError(f"expected 8 test clips, found {len(test_clips)}")

    rows: list[dict] = []
    sampled: list[tuple[str, float]] = []
    frames_with_box = 0
    for clip in test_clips:
        for time_s, frame in sampled_frames(clip):
            sampled.append((clip.stem, time_s))
            result = model.predict(
                frame, conf=SEED_CONF, imgsz=IMGSZ, device=DEVICE, verbose=False
            )[0]
            boxes = result.boxes
            if boxes is not None and len(boxes):
                frames_with_box += 1
            for det in range(len(boxes) if boxes is not None else 0):
                x1, y1, x2, y2 = boxes.xyxy.cpu().numpy()[det]
                rows.append(
                    {
                        "clip_id": clip.stem,
                        "time_s": f"{time_s:.2f}",
                        "conf": f"{float(boxes.conf[det]):.4f}",
                        "x1": f"{x1:.1f}",
                        "y1": f"{y1:.1f}",
                        "x2": f"{x2:.1f}",
                        "y2": f"{y2:.1f}",
                    }
                )
    if not frames_with_box:
        print("held-out: round 2 fired on no test frame; the run is unusable", flush=True)

    HELDOUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with HELDOUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["clip_id", "time_s", "conf", "x1", "y1", "x2", "y2"])
        writer.writeheader()
        writer.writerows(rows)

    # Spread the 20 annotated frames over the whole held-out set, misses included.
    picks = np.linspace(0, len(sampled) - 1, HELDOUT_FRAMES).round().astype(int)
    picks = sorted(set(int(p) for p in picks))
    wanted = {sampled[p] for p in picks}
    HELDOUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in HELDOUT_DIR.glob("*.jpg"):
        old.unlink()

    hits = 0
    for clip in test_clips:
        for time_s, frame in sampled_frames(clip):
            if (clip.stem, time_s) not in wanted:
                continue
            result = model.predict(
                frame, conf=SEED_CONF, imgsz=IMGSZ, device=DEVICE, verbose=False
            )[0]
            found = best_box(result, frame.shape)
            path = HELDOUT_DIR / f"{clip.stem}_{time_s:.2f}.jpg"
            if found is not None:
                hits += 1
                save_annotated(frame, found[1], found[0], path)
            else:
                cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

    stats = {
        "test_frames": len(sampled),
        "frames_with_box": frames_with_box,
        "annotated": len(wanted),
        "annotated_hits": hits,
        "detections": len(rows),
    }
    print(
        f"held-out: {frames_with_box}/{len(sampled)} sampled frames fired; "
        f"{hits}/{len(wanted)} annotated frames fired",
        flush=True,
    )
    return stats


def false_positive_check(weights: Path, clip_id: str, times: tuple[float, ...]) -> list[dict]:
    """Report round 2's confidence on the nearest sampled frame to each given time.

    The seed label records where the false positive was, so the strongest box
    overlapping it is the one that would repeat the mistake; the best box anywhere
    is reported too, since on 56.04 there is no seed label to compare against.
    """
    path = next(CLIPS.rglob(f"{clip_id}.mp4"))
    sampled = [time_s for time_s, _ in sampled_frames(path)]
    nearest = {wanted: min(sampled, key=lambda value: abs(wanted - value)) for wanted in times}

    model = YOLO(str(weights))
    results = []
    for time_s, frame in sampled_frames(path):
        wanted = [w for w, value in nearest.items() if abs(value - time_s) < 1e-6]
        if not wanted:
            continue
        stem = f"{clip_id}_{time_s:.2f}"
        seed_path = DATASET / "labels" / f"{stem}.txt"
        seed_box = label_to_xyxy(seed_path.read_text()) if seed_path.is_file() else None
        found = detections(
            model.predict(frame, conf=SEED_CONF, imgsz=IMGSZ, device=DEVICE, verbose=False)[0],
            frame.shape,
        )
        at_seed = [item for item in found if seed_box is not None and iou(item[1], seed_box) >= AGREE_IOU]
        results.append(
            {
                "wanted": wanted[0],
                "time_s": time_s,
                "best": found[0][0] if found else None,
                "at_seed": at_seed[0][0] if at_seed else None,
            }
        )
    return results


# --- log ---------------------------------------------------------------------


def write_log(
    seed: dict[str, int],
    round1: dict,
    round2: dict,
    round2_seed_eval: dict[str, float],
    round2_counts: dict[str, int],
    heldout: dict,
    false_positives: list[dict],
    epochs: int,
    command: str,
) -> None:
    """One table per question, so a later reader does not have to dig in /tmp."""
    lines = [
        "# Detector training log",
        "",
        f"Command: `{command}`",
        "",
        "Round 1 trains on the classical seed boxes. Round 2 trains on those plus "
        "round 1's own boxes on every 1 fps frame of the train and validation clips.",
        "Test clips are excluded from both rounds and only scored at the end.",
        "",
        "## Validation",
        "",
        "| round | epochs run | best epoch | wall time | mAP50 | mAP50-95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| 1 (seed labels) | {round1['epochs']} | {round1['best_epoch']} | {round1['wall_s'] / 60:.1f} min | {round1['mAP50']:.3f} | {round1['mAP50-95']:.3f} |",
        f"| 2 (self-trained) | {round2['epochs']} | {round2['best_epoch']} | {round2['wall_s'] / 60:.1f} min | {round2['mAP50']:.3f} | {round2['mAP50-95']:.3f} |",
        f"| 2, scored on the untouched seed val set | - | - | - | {round2_seed_eval['mAP50']:.3f} | {round2_seed_eval['mAP50-95']:.3f} |",
        "",
        "Scores are from each round's best checkpoint (highest Ultralytics fitness, "
        "0.1 * mAP50 + 0.9 * mAP50-95), not its last epoch.",
        "The round 2 val score is against val labels that round 1 partly wrote, so "
        "the last row is the comparable one.",
        "",
        "## Labels",
        "",
        "| round | train labels | val labels |",
        "| --- | ---: | ---: |",
        f"| 1 | {seed['train_images']} | {seed['val_images']} |",
        f"| 2 | {round2_counts['train_images']} | {round2_counts['val_images']} |",
        "",
        "Round 2 reused {kept} seed labels, replaced {replaced} that disagreed with a "
        "confident model box, added {added} frames the seeds had missed, and left "
        "{left} disagreed seeds alone because the model was not confident.".format(**round2_counts),
        "",
        "## Held-out test clips",
        "",
        f"Round 2 fired on {heldout['frames_with_box']} of {heldout['test_frames']} sampled "
        f"frames ({100 * heldout['frames_with_box'] / heldout['test_frames']:.0f}%) at conf >= {SEED_CONF}.",
        f"The {heldout['annotated']} annotated frames are in `outputs/detection/heldout/`; "
        f"{heldout['annotated_hits']} produced a box at conf >= {SEED_CONF} "
        f"({100 * heldout['annotated_hits'] / heldout['annotated']:.0f}%).",
        f"All {heldout['detections']} boxes above the floor are in `outputs/detection/heldout.csv`.",
        "",
        "### Known false positives",
        "",
        "| requested t (s) | sampled frame t (s) | best conf | conf on seed box |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for item in false_positives:
        best = "-" if item["best"] is None else f"{item['best']:.2f}"
        at_seed = "-" if item["at_seed"] is None else f"{item['at_seed']:.2f}"
        lines.append(f"| {item['wanted']:.2f} | {item['time_s']:.2f} | {best} | {at_seed} |")
    lines += [
        "",
        "\"conf on seed box\" is the strongest box overlapping the seed label by "
        f"IoU >= {AGREE_IOU}; it is the one that would repeat the false positive. "
        "A dash means no box there, or no seed label at that second.",
        "",
        f"The brief listed 56.04, but that clip's seed labels are at 66.04, 67.04 "
        "and 68.04; 56.04 has no seed label, so only the best box anywhere can be "
        "reported there.",
        "",
        "## Artifacts",
        "",
        f"Ultralytics output: `/tmp/opencode/yolo` (weights, logs, round 2 dataset).",
        "Final weights: `outputs/detection/model/round1.pt`, `outputs/detection/model/round2.pt`.",
        f"Epoch budget: {epochs}.",
    ]
    TRAIN_LOG.write_text("\n".join(lines) + "\n")


# --- driver ------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--work", type=Path, default=Path("/tmp/opencode/yolo"))
    args = parser.parse_args()

    # Python 3.14 defaults to forkserver, which relaunches sys.executable; inside
    # the T3 AppImage that is Electron, not Python, so workers need fork.
    multiprocessing.set_start_method("fork", force=True)

    if not DATASET.is_dir():
        parser.error(f"{DATASET} not found; run scripts/detect.py first")
    splits = read_splits(SPLITS)
    counts = {split: sum(1 for value in splits.values() if value == split) for split in set(splits.values())}
    print(
        f"splits: {counts['train']} train, {counts['validation']} validation, "
        f"{counts['test']} test clips",
        flush=True,
    )

    args.work.mkdir(parents=True, exist_ok=True)
    seed = build_seed_dataset(splits)
    data1 = DATASET / "data.yaml"

    # Download into the work dir so no weights land in the repo.
    round1 = train_round(args.work / WEIGHTS, data1, "round1", args.work, args.epochs)
    data2, round2_counts = build_round2(round1["weights"], splits, args.work)
    round2 = train_round(round1["weights"], data2, "round2", args.work, args.epochs)

    # Same weights, seed val labels: the only fair round 1 vs round 2 comparison.
    metrics = YOLO(str(round2["weights"])).val(
        data=str(data1), split="val", imgsz=IMGSZ, batch=BATCH, device=DEVICE,
        project=str(args.work), name="round2-seed-val", exist_ok=True, verbose=False,
    )
    round2_seed_eval = {
        "mAP50": float(metrics.results_dict["metrics/mAP50(B)"]),
        "mAP50-95": float(metrics.results_dict["metrics/mAP50-95(B)"]),
    }

    heldout = heldout_check(round2["weights"], splits)
    false_positives = false_positive_check(round2["weights"], *KNOWN_FALSE_POSITIVES)
    write_log(
        seed, round1, round2, round2_seed_eval, round2_counts,
        heldout, false_positives, args.epochs,
        " ".join(["python3", "scripts/train_detector.py", *sys.argv[1:]]),
    )

    print(f"\nlog -> {TRAIN_LOG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
