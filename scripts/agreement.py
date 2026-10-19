"""Compare two independent label sets on the shared sample and write the ledger.

The gold sample is labelled twice without either annotator seeing the other's
work or the machine proposal (ADR 0003). This measures how far the two sets agree
before anyone reconciles them, so an unclear label rule shows up while it is
still cheap to fix (docs/ANNOTATION-GUIDE.md review procedure).

Agreement is a Dice score over interval counts: two annotators can describe the
same behavior with slightly different boundaries, and counting that as two
disagreements would understate how well the rules were understood. The
reconciliation columns are left blank for a person to fill when the disagreements
are discussed, because a script cannot decide which label was right.

Usage:
    python3 scripts/agreement.py --a data/annotations.csv --b data/annotations-second.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import labels
import review

# Two boundaries count as the same when they land within this many seconds.
BOUNDARY_TOLERANCE_S = 0.5


def gold_pass(rows: list[dict], gold: set[str], name: str, *, allow_other: bool = False) -> list[dict]:
    """Require every gold clip and one consistent identity before comparison.

    This checks the submitted file, not whether a human actually worked blind.
    The coordinator must confirm independence with the reviewers separately.
    """
    present = {row["clip_id"] for row in rows}
    if not gold:
        raise ValueError("gold clip list is empty")
    if gold - present:
        raise ValueError(f"{name} missing gold clips: {sorted(gold - present)}")
    if not allow_other and present - gold:
        raise ValueError(f"{name} contains non-gold clips: {sorted(present - gold)}")
    selected = [dict(row) for row in rows if row["clip_id"] in gold]
    identities = {row["annotator"].strip().casefold() for row in selected}
    if len(identities) != 1 or "" in identities:
        raise ValueError(f"{name} needs exactly one non-empty annotator identity across the gold clips")
    for row in selected:
        start, end = float(row["start_s"]), float(row["end_s"])
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            raise ValueError(f"{name}: invalid interval in {row['clip_id']}")
        if row["behavior"] not in labels.ALL_LABELS:
            raise ValueError(f"{name}: unknown behavior in {row['clip_id']}")
        if row["behavior"] == "Uncertain" and not row.get("uncertain_reason", "").strip():
            raise ValueError(f"{name}: Uncertain needs a reason in {row['clip_id']}")
    for clip_id in gold:
        intervals = sorted((float(r["start_s"]), float(r["end_s"])) for r in selected if r["clip_id"] == clip_id)
        if any(current[0] < previous[1] for previous, current in zip(intervals, intervals[1:])):
            raise ValueError(f"{name}: overlapping intervals in {clip_id}")
    return selected


def overlap(first, second) -> float:
    return max(0.0, min(first["end_s"], second["end_s"]) - max(first["start_s"], second["start_s"]))


def match_intervals(a_intervals: list[dict], b_intervals: list[dict]) -> list[tuple[dict, dict]]:
    """Greedy one-to-one pairing of intervals that overlap and share a behavior."""
    candidates = sorted(
        (
            (overlap(a, b), i, j)
            for i, a in enumerate(a_intervals)
            for j, b in enumerate(b_intervals)
            if a["behavior"] == b["behavior"] and overlap(a, b) > 0
        ),
        reverse=True,
    )
    taken_a, taken_b, pairs = set(), set(), []
    for _, i, j in candidates:
        if i in taken_a or j in taken_b:
            continue
        taken_a.add(i)
        taken_b.add(j)
        pairs.append((a_intervals[i], b_intervals[j]))
    return pairs


def compare(a_rows: list[dict], b_rows: list[dict]) -> tuple[list[dict], list[str]]:
    """Ledger rows for every clip labelled in both sets, plus the clips that are not."""
    by_clip = lambda rows: {  # noqa: E731 - a one-line grouping, not a function worth naming
        clip_id: [row for row in rows if row["clip_id"] == clip_id] for clip_id in {r["clip_id"] for r in rows}
    }
    a_by_clip, b_by_clip = by_clip(a_rows), by_clip(b_rows)
    only_a = sorted(set(a_by_clip) - set(b_by_clip))
    only_b = sorted(set(b_by_clip) - set(a_by_clip))

    rows = []
    for clip_id in sorted(set(a_by_clip) & set(b_by_clip)):
        a_intervals, b_intervals = a_by_clip[clip_id], b_by_clip[clip_id]
        for interval in a_intervals + b_intervals:
            interval["start_s"], interval["end_s"] = float(interval["start_s"]), float(interval["end_s"])
        pairs = match_intervals(a_intervals, b_intervals)
        boundaries = sum(
            1
            for a, b in pairs
            if abs(a["start_s"] - b["start_s"]) <= BOUNDARY_TOLERANCE_S and abs(a["end_s"] - b["end_s"]) <= BOUNDARY_TOLERANCE_S
        )
        total = len(a_intervals) + len(b_intervals)
        rows.append(
            {
                "clip_id": clip_id,
                "annotator_a": a_intervals[0]["annotator"],
                "annotator_b": b_intervals[0]["annotator"],
                "intervals_compared": total,
                "matching_labels": len(pairs),
                "matching_boundaries": boundaries,
                "disagreements": total - 2 * len(pairs),
                "resolved_labels": "",
                "resolution_reason": "",
                "reviewed_by": "",
                "reviewed_at": "",
            }
        )
    return rows, [f"{clip_id} only in the first file" for clip_id in only_a] + [
        f"{clip_id} only in the second file" for clip_id in only_b
    ]


def write_ledger(path: Path, rows: list[dict]) -> None:
    fields = ("clip_id", "annotator_a", "annotator_b", "intervals_compared", "matching_labels",
              "matching_boundaries", "disagreements", "resolved_labels", "resolution_reason",
              "reviewed_by", "reviewed_at")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--a", type=Path, default=Path("data/annotations.csv"))
    parser.add_argument("--b", type=Path, default=Path("data/annotations-second.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/agreement.csv"))
    parser.add_argument("--gold-file", type=Path, default=Path("data/gold-clips.txt"))
    args = parser.parse_args()

    if not args.b.exists():
        parser.error(f"{args.b} not found; the second annotator runs scripts/review.py --out {args.b} on the gold clips")

    try:
        gold = review.id_set([], args.gold_file)
        a_rows = gold_pass(review.read_rows(args.a), gold, "first pass", allow_other=True)
        b_rows = gold_pass(review.read_rows(args.b), gold, "second pass")
        if a_rows[0]["annotator"].strip().casefold() == b_rows[0]["annotator"].strip().casefold():
            raise ValueError("gold passes need two different annotators")
        if args.out.resolve() in {args.a.resolve(), args.b.resolve()}:
            raise ValueError("output must not overwrite either annotation file")
        existing = review_ledger(args.out)
        if any(row.get(field, "").strip() for row in existing for field in
               ("resolved_labels", "resolution_reason", "reviewed_by", "reviewed_at")):
            raise ValueError("output contains reconciliation work; choose a new --out to preserve it")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    rows, orphans = compare(a_rows, b_rows)
    if not rows:
        parser.error("no clip was labelled in both files, so there is nothing to compare")

    write_ledger(args.out, rows)
    merged = [f"{row['clip_id']} {row['matching_labels']}/{row['intervals_compared']} matched, "
              f"{row['matching_boundaries']} boundary matches, {row['disagreements']} disagreements" for row in rows]
    print("\n".join(merged) + "\n")
    compared = sum(row["intervals_compared"] for row in rows)
    matched = sum(row["matching_labels"] for row in rows)
    print(f"agreement (Dice over intervals): {2 * matched / compared:.3f} on {len(rows)} shared clips -> {args.out}")
    if orphans:
        print("not compared: " + "; ".join(orphans))
    print("reconciliation columns are blank for the two annotators to fill.")
    print(f"model classes: {', '.join(labels.MODEL_CLASSES)}")
    return 0


def review_ledger(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    raise SystemExit(main())
