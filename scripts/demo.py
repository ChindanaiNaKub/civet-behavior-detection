"""Show the classifier's behavior timeline for one clip.

Predicts a behavior and a confidence for every five-second window of a clip and
prints them against the clip's own timecodes, so a reviewer can compare the
predicted sequence with the footage (issue #5). A window below the abstain
threshold prints as Uncertain with the class the model leaned toward, which is
the same abstention the evaluation counts.

The threshold comes from the evaluation run rather than a default, so the demo
cannot quietly disagree with the reported numbers.

Usage:
    python3 scripts/demo.py data/raw/Resting/25690519194809256.mp4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import frames
import model as M
import windows as W


def clip_duration(path: Path) -> float:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if not count:
        raise SystemExit(f"{path}: no frame count, cannot time the timeline")
    return float(count) / fps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("clip", type=Path)
    parser.add_argument("--model", type=Path, default=Path("outputs/model/model.pt"))
    parser.add_argument("--evaluation", type=Path, default=Path("outputs/evaluation.json"))
    parser.add_argument("--threshold", type=float, default=None, help="override the threshold from the evaluation")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.threshold is None:
        if not args.evaluation.exists():
            raise SystemExit(f"{args.evaluation} not found; run scripts/evaluate.py first, or pass --threshold")
        args.threshold = json.loads(args.evaluation.read_text())["threshold"]

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model, config = M.load(args.model, str(device))
    classes = config["classes"]

    clip_id = args.clip.stem
    spans = W.grid(clip_duration(args.clip))
    windows = [W.Window(clip_id, start, end, "", "demo") for start, end in spans]
    decoded = frames.decode_windows(windows, args.clip.parent.parent, {clip_id: args.clip.parent.name + "/" + args.clip.name})
    probabilities = M.probabilities(model, decoded, range(len(windows)), device)

    print(f"{args.clip}  ({spans[-1][1]:.1f} s, {len(windows)} windows, abstain below {args.threshold:.2f})")
    print(f"{'start':>7} {'end':>7}  {'behavior':<12} {'conf':>5}  leaned toward")
    symbols = []
    for window, row in zip(windows, probabilities):
        best = int(row.argmax())
        confident = row[best] >= args.threshold
        behavior = classes[best] if confident else "Uncertain"
        symbols.append(behavior[0] if confident else "?")
        print(f"{window.start:>7.2f} {window.end:>7.2f}  {behavior:<12} {row[best]:>5.2f}  {classes[best]}")

    print("\ntimeline, one character per window:", "".join(symbols))
    print("legend:", ", ".join(f"{c[0]} {c}" for c in classes) + ", ? Uncertain")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
