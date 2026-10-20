#!/usr/bin/env python3
"""Reconcile the two independent gold passes in the browser (issue #4).

Two people labelled the same gold clips without seeing each other's work, so they
differ. This page lines the two passes up against the video, shows where they
disagree, and lets the two reviewers decide one final set of intervals per clip,
with a reason for each clip. Nothing is decided for them: the page only records
what they choose.

Decisions are kept in a draft file (outputs/reconcile/decisions.json) until the
reviewers run ``--apply``. Applying writes data/agreement.csv (the measured columns
from the original two passes plus the reviewers' resolution) and replaces the gold
rows of data/annotations.csv with the final intervals. The original first pass
stays in Git history and data/annotations-second.csv is not touched, per
docs/GOLD-REVIEW-HANDOFF.md.

Usage:
    python scripts/reconcile.py                 # open the reconciliation page
    python scripts/reconcile.py --apply         # after every gold clip is decided
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import agreement
import labels
import review

# The identity written on the final rows. They belong to neither annotator.
RECONCILED = "reconciled"
# A disagreement shorter than this is almost always a boundary shift, not a label dispute.
SHORT_SPAN_S = agreement.BOUNDARY_TOLERANCE_S
MAX_BODY = 1 << 20
LOCK = threading.Lock()


# --- pure functions ----------------------------------------------------------


def intervals_of(rows: list[dict]) -> list[dict]:
    """Rows as intervals with float times, ordered by start."""
    found = [
        {**row, "start_s": float(row["start_s"]), "end_s": float(row["end_s"])}
        for row in rows
    ]
    return sorted(found, key=lambda interval: (interval["start_s"], interval["end_s"]))


def label_at(intervals: list[dict], moment: float) -> str | None:
    for interval in intervals:
        if interval["start_s"] <= moment < interval["end_s"]:
            return interval["behavior"]
    return None


def spans(a: list[dict], b: list[dict], duration: float) -> list[dict]:
    """The clip cut at every boundary of either pass, with what each pass says there.

    Consecutive stretches that say the same thing are joined. ``agree`` is true when
    both passes give the same label, or when neither labels the stretch.
    """
    cuts = {0.0, float(duration)}
    for interval in a + b:
        cuts.update((interval["start_s"], interval["end_s"]))
    points = sorted(point for point in cuts if 0.0 <= point <= duration)
    joined: list[dict] = []
    for start, end in zip(points, points[1:]):
        if end - start < 1e-6:
            continue
        middle = (start + end) / 2
        first, second = label_at(a, middle), label_at(b, middle)
        if joined and (joined[-1]["a"], joined[-1]["b"]) == (first, second):
            joined[-1]["end"] = end
            continue
        joined.append({"start": start, "end": end, "a": first, "b": second})
    for span in joined:
        span["agree"] = span["a"] == span["b"]
        span["short"] = (span["end"] - span["start"]) < SHORT_SPAN_S
    return joined


def check_final(clip_id: str, intervals: list[dict], duration: float) -> list[dict]:
    """Annotation rows for a clip's final intervals, or ValueError saying what is wrong."""
    if not isinstance(intervals, list) or not intervals:
        raise ValueError("a clip needs at least one final interval")
    cleaned = []
    for number, interval in enumerate(intervals, 1):
        if not isinstance(interval, dict):
            raise ValueError(f"interval {number}: invalid record")
        cleaned.append(dict(interval))
    review.validate_conditions(cleaned)
    rows = review.annotation_rows(clip_id, RECONCILED, cleaned)
    ordered = sorted(rows, key=lambda row: (float(row["start_s"]), float(row["end_s"])))
    for row in ordered:
        if float(row["end_s"]) > duration + 0.05:
            raise ValueError(f"{row['behavior']} ends at {row['end_s']} s, past the {duration:.2f} s clip")
    for previous, current in zip(ordered, ordered[1:]):
        if float(current["start_s"]) < float(previous["end_s"]) - 1e-6:
            raise ValueError(
                f"{previous['behavior']} {previous['start_s']}-{previous['end_s']} overlaps "
                f"{current['behavior']} {current['start_s']}-{current['end_s']}"
            )
    return ordered


def resolved_labels(rows: list[dict]) -> str:
    """The final labels and their time ranges, as one line for data/agreement.csv."""
    ordered = sorted(rows, key=lambda row: (float(row["start_s"]), float(row["end_s"])))
    return "; ".join(f"{row['behavior']} {float(row['start_s']):.2f}-{float(row['end_s']):.2f}" for row in ordered)


def load_decisions(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {"reviewers": ["", ""], "clips": {}, "durations": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "reviewers": list(data.get("reviewers", ["", ""])),
        "clips": dict(data.get("clips", {})),
        "durations": dict(data.get("durations", {})),
    }


def save_decisions(path: Path, decisions: dict) -> None:
    """Replace the draft file whole, so a crash never leaves half a decision."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(decisions, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def reviewers_of(decisions: dict) -> list[str]:
    names = [str(name).strip() for name in decisions.get("reviewers", [])]
    if len(names) != 2 or not all(names) or names[0].casefold() == names[1].casefold():
        raise ValueError("two different reviewer names are required")
    return names


def build_ledger(a_rows: list[dict], b_rows: list[dict], decisions: dict, gold: set[str], moment: str) -> list[dict]:
    """The agreement ledger: measured from the original passes, resolved from the decisions."""
    names = reviewers_of(decisions)
    undecided = sorted(gold - set(decisions["clips"]))
    if undecided:
        raise ValueError(f"no final intervals yet for: {undecided}")
    ledger, orphans = agreement.compare(a_rows, b_rows)
    if orphans or {row["clip_id"] for row in ledger} != gold:
        raise ValueError(f"the two passes do not cover the same gold clips: {orphans}")
    for row in ledger:
        decision = decisions["clips"][row["clip_id"]]
        reason = str(decision.get("reason", "")).strip()
        if not reason:
            raise ValueError(f"{row['clip_id']} has no reason; write why it changed, or that both passes agree")
        row["resolved_labels"] = resolved_labels(decision["rows"])
        row["resolution_reason"] = reason
        row["reviewed_by"] = "; ".join(names)
        row["reviewed_at"] = moment
    return ledger


def replace_gold_rows(rows: list[dict], gold: set[str], final: list[dict]) -> list[dict]:
    """Every row for a non-gold clip stays; each gold clip gets its one final set."""
    return [row for row in rows if row["clip_id"] not in gold] + final


def write_annotations(path: Path, rows: list[dict]) -> None:
    path = Path(path)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=review.CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def apply(
    a_path: Path,
    b_path: Path,
    ledger_path: Path,
    store_path: Path,
    gold_file: Path,
    moment: str,
) -> list[dict]:
    """Write the agreement ledger and swap the final gold rows into the first-pass file."""
    gold = review.id_set([], gold_file)
    decisions = load_decisions(store_path)
    all_a = review.read_rows(a_path)
    if any(row["annotator"] == RECONCILED and row["clip_id"] in gold for row in all_a):
        raise ValueError(f"{a_path} already holds reconciled gold rows; nothing to apply")
    existing = agreement.review_ledger(ledger_path)
    if any(row.get(field, "").strip() for row in existing for field in
           ("resolved_labels", "resolution_reason", "reviewed_by", "reviewed_at")):
        raise ValueError(f"{ledger_path} already holds reconciliation work; not overwriting it")

    a_rows = agreement.gold_pass(all_a, gold, "first pass", allow_other=True)
    b_rows = agreement.gold_pass(review.read_rows(b_path), gold, "second pass")
    ledger = build_ledger(a_rows, b_rows, decisions, gold, moment)

    final: list[dict] = []
    durations = decisions.get("durations", {})
    for clip_id in sorted(gold):
        final.extend(check_final(clip_id, decisions["clips"][clip_id]["intervals"], float(durations.get(clip_id, 1e9))))
    agreement.write_ledger(ledger_path, ledger)
    write_annotations(a_path, replace_gold_rows(all_a, gold, final))
    return ledger


# --- server ------------------------------------------------------------------


class Config:
    def __init__(self, clips, a_rows, b_rows, store_path, gold):
        self.clips = {clip.clip_id: clip for clip in clips}
        self.order = [clip.clip_id for clip in clips]
        self.a = {clip_id: intervals_of([r for r in a_rows if r["clip_id"] == clip_id]) for clip_id in self.order}
        self.b = {clip_id: intervals_of([r for r in b_rows if r["clip_id"] == clip_id]) for clip_id in self.order}
        self.store_path = Path(store_path)
        self.gold = gold
        self.annotators = (
            a_rows[0]["annotator"] if a_rows else "first pass",
            b_rows[0]["annotator"] if b_rows else "second pass",
        )

    def stats(self, clip_id: str) -> dict:
        a, b = self.a[clip_id], self.b[clip_id]
        matched = len(agreement.match_intervals(a, b))
        return {"a_intervals": len(a), "b_intervals": len(b), "matched": matched, "to_discuss": len(a) + len(b) - 2 * matched}


class Handler(BaseHTTPRequestHandler):
    server_version = "civet-reconcile"

    def log_message(self, *args) -> None:
        pass

    @property
    def config(self) -> Config:
        return self.server.config

    def _json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        route = urlparse(self.path)
        args = parse_qs(route.query)
        config = self.config
        if route.path in ("/", "/index.html"):
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif route.path == "/api/state":
            decisions = load_decisions(config.store_path)
            self._json(
                {
                    "annotators": config.annotators,
                    "reviewers": decisions["reviewers"],
                    "clips": [
                        {"clip_id": clip_id, "duration_s": config.clips[clip_id].duration_s,
                         "done": clip_id in decisions["clips"], **config.stats(clip_id)}
                        for clip_id in config.order
                    ],
                }
            )
        elif route.path == "/api/clip":
            clip_id = args.get("id", [""])[0]
            if clip_id not in config.clips:
                self._json({"error": "unknown clip"}, 404)
                return
            clip = config.clips[clip_id]
            decision = load_decisions(config.store_path)["clips"].get(clip_id)
            self._json(
                {
                    "clip_id": clip_id,
                    "duration_s": clip.duration_s,
                    "video_url": f"/video?id={clip_id}",
                    "a": config.a[clip_id],
                    "b": config.b[clip_id],
                    "spans": spans(config.a[clip_id], config.b[clip_id], clip.duration_s),
                    "final": decision["intervals"] if decision else None,
                    "reason": decision["reason"] if decision else "",
                    **config.stats(clip_id),
                }
            )
        elif route.path == "/video":
            clip = config.clips.get(args.get("id", [""])[0])
            if clip is None or not clip.path.exists():
                self._json({"error": "unknown clip"}, 404)
                return
            review.Handler._serve_video(self, clip.path)
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        route = urlparse(self.path).path
        if route not in ("/api/final", "/api/reviewers"):
            self._json({"error": "not found"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= MAX_BODY:
                raise ValueError("request body is empty or too large")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("invalid request")
            with LOCK:
                decisions = load_decisions(self.config.store_path)
                if route == "/api/reviewers":
                    names = body.get("reviewers")
                    if not isinstance(names, list) or len(names) != 2:
                        raise ValueError("send two reviewer names")
                    decisions["reviewers"] = [str(name).strip() for name in names]
                    save_decisions(self.config.store_path, decisions)
                    self._json({"saved": True})
                    return
                clip_id = str(body.get("clip_id", ""))
                if clip_id not in self.config.clips:
                    raise ValueError("unknown clip")
                duration = self.config.clips[clip_id].duration_s
                rows = check_final(clip_id, body.get("intervals"), duration)
                reason = str(body.get("reason", "")).strip()
                if not reason:
                    raise ValueError("write why this differs from a pass, or that both passes agree")
                decisions["clips"][clip_id] = {"intervals": body["intervals"], "rows": rows, "reason": reason,
                                                "saved_at": datetime.now().astimezone().isoformat(timespec="seconds")}
                decisions["durations"][clip_id] = duration
                save_decisions(self.config.store_path, decisions)
            self._json({"saved": True, "resolved_labels": resolved_labels(rows),
                        "done": len(decisions["clips"]), "total": len(self.config.order)})
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            self._json({"error": str(exc)}, 400)


# --- page --------------------------------------------------------------------

PAGE = Path(__file__).with_name("reconcile_page.html").read_text(encoding="utf-8")
PAGE = PAGE.replace("__BEHAVIORS__", json.dumps(list(review.BEHAVIORS)))
PAGE = PAGE.replace("__CONDITION_OPTIONS__", json.dumps(review.CONDITION_OPTIONS))


# --- driver ------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--a", type=Path, default=Path("data/annotations.csv"), help="first pass")
    parser.add_argument("--b", type=Path, default=Path("data/annotations-second.csv"), help="second pass")
    parser.add_argument("--clips-file", type=Path, default=Path("data/source-clips.csv"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--gold-file", type=Path, default=Path("data/gold-clips.txt"))
    parser.add_argument("--ledger", type=Path, default=Path("data/agreement.csv"))
    parser.add_argument("--store", type=Path, default=Path("outputs/reconcile/decisions.json"))
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--apply", action="store_true", help="write agreement.csv and the final gold rows, then exit")
    args = parser.parse_args(argv)

    if args.apply:
        try:
            moment = datetime.now().astimezone().isoformat(timespec="seconds")
            ledger = apply(args.a, args.b, args.ledger, args.store, args.gold_file, moment)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        print(f"{len(ledger)} gold clips reconciled -> {args.ledger}; final gold rows written to {args.a}")
        print("Re-run scripts/lighting.py, scripts/split.py and scripts/windows.py, then have a person sign the test lock.")
        return 0

    try:
        gold = review.id_set([], args.gold_file)
        all_a = review.read_rows(args.a)
        if any(row["annotator"] == RECONCILED and row["clip_id"] in gold for row in all_a):
            raise ValueError(f"{args.a} already holds reconciled gold rows")
        a_rows = agreement.gold_pass(all_a, gold, "first pass", allow_other=True)
        b_rows = agreement.gold_pass(review.read_rows(args.b), gold, "second pass")
        clips = [clip for clip in review.load_clips(args.clips_file, args.raw) if clip.clip_id in gold]
        if {clip.clip_id for clip in clips} != gold:
            raise ValueError("some gold clips are not in the source ledger")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.config = Config(clips, a_rows, b_rows, args.store, gold)
    print(f"Reconcile: http://127.0.0.1:{server.server_port}/\n"
          f"{len(clips)} gold clips. Draft decisions: {args.store}\nKeep this terminal open. Stop with Ctrl+C.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
