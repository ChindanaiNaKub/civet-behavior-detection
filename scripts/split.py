"""Assign every source clip to train, validation or test, and write the ledger.

The split is by whole clip and happens before any window exists, which is the
rule in issue #1. It is not by camera, although holding a camera out would be the
cleaner generalization test, because camera and class are confounded in this
footage: 597 of the 659 resting seconds are Camera 03 on 2026-05-19, and every
locomotion clip is on 2026-05-18. A camera-held-out split would train on 62 s of
resting and test on 597 s, so it would measure that mismatch rather than the
model. Splitting by clip instead means a recording group can appear on both sides
of the split, which the ledger records on every row.

Every split holds all three model classes, and the class is stratified by runs of
footage rather than by clip count, because a clip is between 8 and 220 seconds.
Climbing is absent because no verified climb exists in this footage.

Usage:
    python3 scripts/split.py            # rewrite data/splits.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import labels

# Chosen so every split holds Eating, Locomotion and Resting, and so the quiet
# class does not sit entirely on one side. See the module docstring for why this
# is by clip and not by camera.
ASSIGNMENT = {
    "train": [
        # Eating, 158 s
        "25690518092326646",
        "25690518092408866",
        "25690519193547292",
        # Locomotion, 87 s
        "25690518091047366",
        "25690518091210873",
        "25690518091711167",
        "25690518091847591",
        "25690518091948946",
        "25690520153014458",
        # Resting, 565 s
        "25690519194109670",
        "25690519194418953",
        "25690519194809256",
        "25690519195143424",
    ],
    "validation": [
        "25690519193422801",  # Eating, 50 s
        "25690518091030781",  # Locomotion, 10 s
        "25690519194200758",  # Resting, 31 s
    ],
    "test": [
        "25690519193808204",  # Eating, 46 s
        "25690518090944462",  # Locomotion, 11 s
        "25690518091012230",  # Locomotion, 8 s
        "25690518091918610",  # Locomotion, 22 s
        "25690520152850102",  # Resting, 31 s
        "25690520153258104",  # Resting, 31 s
    ],
}


def read_ledger(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def check(clips: list[dict]) -> None:
    """Fail loudly rather than write a ledger a later step has to guess about."""
    assigned = [clip_id for split in ASSIGNMENT.values() for clip_id in split]
    duplicates = [clip_id for clip_id, n in Counter(assigned).items() if n > 1]
    if duplicates:
        raise SystemExit(f"assigned more than once: {sorted(duplicates)}")
    known = {row["clip_id"] for row in clips}
    unknown = [clip_id for clip_id in assigned if clip_id not in known]
    if unknown:
        raise SystemExit(f"not in the source ledger: {sorted(unknown)}")
    missing = sorted(known - set(assigned))
    if missing:
        raise SystemExit(f"not assigned to any split: {missing}")

    behavior = {row["clip_id"]: row["folder_label_weak"] for row in clips}
    for split, clip_ids in ASSIGNMENT.items():
        present = {behavior[c] for c in clip_ids}
        gap = [c for c in labels.MODEL_CLASSES if c not in present]
        if gap:
            raise SystemExit(f"{split} has no clip for {gap}; group-aware or class-aware split needed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-clips", type=Path, default=Path("data/source-clips.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/splits.csv"))
    args = parser.parse_args()

    clips = read_ledger(args.source_clips)
    check(clips)

    by_clip = {row["clip_id"]: row for row in clips}
    elsewhere = {clip_id: sorted(s for s, ids in ASSIGNMENT.items() if s != split and clip_id in ids) for split, ids in ASSIGNMENT.items() for clip_id in ids}

    rows = []
    for split in labels.SPLITS:
        for clip_id in ASSIGNMENT[split]:
            clip = by_clip[clip_id]
            group = f"{clip['camera_id_overlay']} {clip['capture_datetime_overlay'].split()[0]}"
            straddles = elsewhere[clip_id]
            reason = "class-stratified clip split, not a recording-group split"
            if straddles:
                reason += f"; {group} also appears in {' and '.join(straddles)}"
            else:
                reason += f"; {group} sits entirely in {split}"
            rows.append(
                {
                    "clip_id": clip_id,
                    "split": split,
                    "grouping_key": group,
                    "grouping_evidence": "on-screen camera and date overlay, read in the footage audit",
                    "assignment_reason": reason,
                    "test_locked_by": "",
                    "test_locked_at": "",
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=labels.SPLIT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    durations = {row["clip_id"]: float(row["duration_s"]) for row in clips}
    print(f"{len(rows)} clips -> {args.out}")
    for split in labels.SPLITS:
        ids = ASSIGNMENT[split]
        per_class = Counter(by_clip[c]["folder_label_weak"] for c in ids)
        seconds = {c: sum(durations[i] for i in ids if by_clip[i]["folder_label_weak"] == c) for c in labels.MODEL_CLASSES}
        print(
            f"  {split:<11} {len(ids):>2} clips {sum(durations[i] for i in ids):>7.1f} s  "
            + " ".join(f"{c}={per_class[c]}/{seconds[c]:.0f}s" for c in labels.MODEL_CLASSES)
        )
    print("\ntest_locked_by is left empty on purpose: lock the test set after reviewing this file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
