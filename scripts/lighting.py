"""Fill the observed capture conditions in data/annotations.csv.

Issue #4 asks for lighting, visibility and occlusion where they can be verified.
The review page leaves all three as `unknown`. This fills what the footage or the
annotator's own words establish and leaves everything else `unknown`:

- lighting: a clip is infrared/grayscale when its sampled frames carry no colour
  (every pixel has R = G = B) and daylight/colour otherwise. Across the 22 clips the
  measured colour is either 0.0 or above 6, so the threshold is not sensitive.
- visibility and occlusion: only from the annotator's `uncertain_reason`. A reason
  that says the civet cannot be seen sets visibility to `out of view`, and one that
  names a tree or plant sets occlusion to `tree/plant`.

Only values still `unknown` are written, so a value a reviewer set by hand stays.
Re-run it after saving a clip in the review page, which writes `unknown` again.

Usage:
    python3 scripts/lighting.py            # rewrite data/annotations.csv
    python3 scripts/lighting.py --dry-run  # print per-clip lighting only
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

import review

SAMPLES = 5
SIZE = (320, 180)
# Mean of (max - min) over a pixel's three channels, on a 0-255 scale. Infrared
# frames measure 0.0 here and daylight frames measure 6 to 10.
GRAYSCALE_MAX_CHROMA = 2.0

INFRARED = "infrared/grayscale"
DAYLIGHT = "daylight/color"


def chroma(frame: np.ndarray) -> float:
    """How much colour a BGR frame carries: the mean spread across its channels."""
    small = cv2.resize(frame, SIZE).astype(np.int16)
    return float((small.max(axis=2) - small.min(axis=2)).mean())


def classify_lighting(chromas: list[float]) -> str:
    """Infrared/grayscale or daylight/colour from the colour of the sampled frames."""
    if not chromas:
        raise ValueError("no frame could be sampled")
    return INFRARED if float(np.mean(chromas)) <= GRAYSCALE_MAX_CHROMA else DAYLIGHT


def clip_lighting(path: Path) -> str:
    capture = cv2.VideoCapture(str(path))
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        values = []
        for k in range(SAMPLES):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(total * (k + 0.5) / SAMPLES))
            ok, frame = capture.read()
            if ok:
                values.append(chroma(frame))
    finally:
        capture.release()
    return classify_lighting(values)


def conditions_from_reason(reason: str) -> dict[str, str]:
    """Visibility and occlusion that the annotator's own reason states, nothing inferred."""
    text = reason.lower().replace("’", "'")
    found: dict[str, str] = {}
    if "can't see" in text or "cannot see" in text or "cant see" in text:
        found["visibility"] = "out of view"
    if "tree" in text or "plant" in text:
        found["occlusion"] = "tree/plant"
    return found


def fill(rows: list[dict], lighting: dict[str, str]) -> int:
    """Set each still-unknown condition in place; return how many fields changed."""
    changed = 0
    for row in rows:
        wanted = {"lighting": lighting.get(row["clip_id"], "unknown")}
        wanted.update(conditions_from_reason(row.get("uncertain_reason", "")))
        for name, value in wanted.items():
            if value != "unknown" and row[name] == "unknown":
                row[name] = value
                changed += 1
    return changed


def write_rows(path: Path, rows: list[dict]) -> None:
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=review.CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations.csv"))
    parser.add_argument("--clips-file", type=Path, default=Path("data/source-clips.csv"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = review.read_rows(args.annotations)
    clip_ids = sorted({row["clip_id"] for row in rows})
    paths = {clip.clip_id: clip.path for clip in review.load_clips(args.clips_file, args.raw)}
    lighting = {clip_id: clip_lighting(paths[clip_id]) for clip_id in clip_ids}
    for clip_id in clip_ids:
        print(f"{clip_id}: {lighting[clip_id]}")
    if args.dry_run:
        return 0

    changed = fill(rows, lighting)
    write_rows(args.annotations, rows)
    print(f"\n{changed} fields set in {args.annotations}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
