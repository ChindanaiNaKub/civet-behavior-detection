"""Assign every source clip to train, validation or test, and write the ledger.

The split is by whole clip and happens before any window exists, which is the
rule in issue #1. It is not by camera, although holding a camera out would be the
cleaner generalization test, because camera and class are confounded in this
footage: 597 of the 659 resting seconds are Camera 03 on 2026-05-19, and every
locomotion clip is on 2026-05-18. A camera-held-out split would train on 62 s of
resting and test on 597 s, so it would measure that mismatch rather than the
model. Splitting by clip instead means a recording group can appear on both sides
of the split, which the ledger records on every row.

Every split must hold all three model classes. The first assignment was stratified
on the weak folder names, before any interval was verified, and left the test split
with no Resting window and one Eating window (ADR 0008). The assignment below is
checked against the windows the verified intervals actually produce, so a split
that looks complete by folder name but is empty of a class in practice is refused.
Climbing is not a model class (ADR 0007).

Usage:
    python3 scripts/split.py            # rewrite data/splits.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import labels
import windows as W

# Test is the gold set: whole clips labelled by a person without the machine
# proposal, twice (data/gold-clips.txt). Validation and train are verified against
# the proposal. See ADR 0008 for why these clips and not the first assignment.
ASSIGNMENT = {
    "train": [
        # Eating
        "25690518092326646",
        "25690518092408866",
        # Locomotion
        "25690518091047366",
        "25690518091210873",
        "25690518091711167",
        "25690518091847591",
        "25690518091948946",
        "25690520153014458",
        # Resting
        "25690519194109670",
        "25690519194418953",
        "25690519194809256",
    ],
    "validation": [
        "25690519193547292",  # Eating, 69 s
        "25690518091030781",  # Locomotion, 10 s
        "25690519195143424",  # Resting, 122 s
    ],
    "test": [
        "25690519193808204",  # Eating, 46 s
        "25690519193422801",  # Eating, 50 s
        "25690518090944462",  # Locomotion, 11 s
        "25690518091012230",  # Locomotion, 8 s
        "25690518091918610",  # Locomotion, 22 s
        "25690520152850102",  # Resting, 31 s, animal mostly hidden
        "25690520153258104",  # Resting, 31 s, animal mostly hidden
        "25690519194200758",  # Resting, 31 s
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


def window_counts(intervals: list[W.Interval], durations: dict[str, float], assignment: dict[str, list[str]]) -> Counter:
    """Windows per (split, class) that the verified intervals produce under an assignment."""
    splits = {clip_id: split for split, clip_ids in assignment.items() for clip_id in clip_ids}
    built, _ = W.build(intervals, durations, splits)
    return Counter((w.split, w.behavior) for w in built)


def verified_gaps(counts: Counter) -> dict[str, list[str]]:
    """The model classes each split has no window for, from the verified intervals."""
    return {
        split: [c for c in labels.MODEL_CLASSES if counts[(split, c)] == 0]
        for split in labels.SPLITS
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-clips", type=Path, default=Path("data/source-clips.csv"))
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/splits.csv"))
    args = parser.parse_args()

    clips = read_ledger(args.source_clips)
    check(clips)

    intervals = W.read_intervals(args.annotations) if args.annotations.exists() else []
    counts: Counter = Counter()
    if intervals:
        counts = window_counts(intervals, W.read_durations(args.source_clips), ASSIGNMENT)
        gaps = {split: gap for split, gap in verified_gaps(counts).items() if gap}
        if gaps:
            raise SystemExit(f"no verified window for {gaps}; change the assignment or record the limitation (ADR 0005)")
    else:
        print(f"warning: no verified intervals in {args.annotations}; class coverage was checked on folder names only")

    by_clip = {row["clip_id"]: row for row in clips}
    elsewhere = {clip_id: sorted(s for s, ids in ASSIGNMENT.items() if s != split and clip_id in ids) for split, ids in ASSIGNMENT.items() for clip_id in ids}

    rows = []
    for split in labels.SPLITS:
        for clip_id in ASSIGNMENT[split]:
            clip = by_clip[clip_id]
            group = f"{clip['camera_id_overlay']} {clip['capture_datetime_overlay'].split()[0]}"
            straddles = elsewhere[clip_id]
            reason = "clip split chosen on verified windows, not a recording-group split"
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
        line = f"  {split:<11} {len(ids):>2} clips {sum(durations[i] for i in ids):>7.1f} s"
        if intervals:
            line += "  windows " + " ".join(f"{c}={counts[(split, c)]}" for c in labels.MODEL_CLASSES)
        print(line)
    print("\ntest_locked_by is left empty on purpose: lock the test set after reviewing this file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
