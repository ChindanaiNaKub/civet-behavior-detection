#!/usr/bin/env python3
"""Local review tool for machine-proposed behavior intervals (issue #9 output).

A reviewer watches each clip in data/raw, sees the intervals proposed in
outputs/prelabel/timeline.csv overlaid under the video, and accepts or corrects
them. Verdicts are POSTed here and appended to data/annotations.csv.

Reviewing, not first-pass labeling, is the job for proposed clips, so every clip
is review mode by default. A clip listed in ``--blind``/``--blind-file`` hides
the proposal everywhere: the page, the JSON, the video are all proposal-free, and
the human labels it from scratch. That blindness is the point; a visible proposal
would make the reported gold-set metric measure model agreement, not accuracy
(ADR 0003, docs/ANNOTATION-GUIDE.md).

Writes are idempotent per clip and annotator: re-saving a clip replaces that
annotator's rows for it instead of adding a second copy, which the window builder
would drop as an overlapping-annotation conflict. Standard library only.

Usage:
    python scripts/review.py --port 8001
    python scripts/review.py --blind 25690519194109670 --blind-file gold.txt
    python scripts/review.py --only-file data/gold-clips.txt --blind-file data/gold-clips.txt
    python scripts/review.py --redo CLIP_ID      # serve a clip that already has rows again
    python scripts/review.py --counts            # per-clip verified row counts
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
import threading
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import labels

# Behavior terms come from GLOSSARY.md; Transition and Uncertain are real
# interval labels but never model classes.
BEHAVIORS = labels.ALL_LABELS
CSV_FIELDS = list(labels.ANNOTATION_FIELDS)
CONDITIONS = ("visibility", "lighting", "camera_view", "occlusion")
CONDITION_OPTIONS = {
    "visibility": ("clear", "partial", "fully occluded", "out of view", "unknown"),
    "lighting": ("daylight/color", "infrared/grayscale", "unknown"),
    "camera_view": ("elevated oblique", "overhead", "animal-level side view", "unknown"),
    "occlusion": ("none", "tree/plant", "fence/mesh", "enclosure structure", "multiple obstructions", "unknown"),
}
WRITE_LOCK = threading.Lock()


# --- pure functions ----------------------------------------------------------


def annotation_rows(clip_id: str, annotator: str, intervals: list[dict]) -> list[dict]:
    """Map reviewer verdict intervals to annotations.csv rows.

    Raises ValueError on an unknown behavior, ``end_s <= start_s``, or an
    ``Uncertain`` interval with no ``uncertain_reason``. Condition fields the
    reviewer left blank default to ``unknown``.
    """
    annotator = annotator.strip()
    if not clip_id:
        raise ValueError("clip_id is required")
    if not annotator:
        raise ValueError("annotator is required")
    rows = []
    for interval in intervals:
        behavior = str(interval.get("behavior", "")).strip()
        if behavior not in BEHAVIORS:
            raise ValueError(f"unknown behavior {behavior!r}")
        start, end = float(interval["start_s"]), float(interval["end_s"])
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            raise ValueError(f"end_s must exceed start_s for {behavior} interval")
        reason = str(interval.get("uncertain_reason", "")).strip()
        if behavior != "Uncertain":
            reason = ""
        if behavior == "Uncertain" and not reason:
            raise ValueError("Uncertain interval needs an uncertain_reason")
        row = {name: str(interval.get(name) or "").strip() or "unknown" for name in CONDITIONS}
        row.update(
            clip_id=clip_id, start_s=f"{start:.2f}", end_s=f"{end:.2f}", behavior=behavior,
            uncertain_reason=reason, annotator=annotator, notes=str(interval.get("notes", "")).strip(),
        )
        rows.append({name: row[name] for name in CSV_FIELDS})
    return rows


def read_rows(path: Path) -> list[dict]:
    """Existing annotation rows, or none when the file is absent or empty."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != CSV_FIELDS:
            raise ValueError(f"{path}: header is {reader.fieldnames}, expected {CSV_FIELDS}")
        return [{name: row.get(name, "") for name in CSV_FIELDS} for row in reader]


def validate_conditions(rows: list[dict]) -> None:
    """Every submitted interval needs an explicit choice from all four menus."""
    if not isinstance(rows, list):
        raise ValueError("intervals must be a list")
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"interval {index}: invalid record")
        for name, choices in CONDITION_OPTIONS.items():
            if row.get(name) not in choices:
                raise ValueError(f"interval {index}: select a valid {name} before saving")


def replace_rows(path: Path, clip_id: str, annotator: str, rows: list[dict]) -> int:
    """Write one annotator's rows for one clip, keeping every other row.

    Idempotent, because a re-saved clip must not leave two copies of its
    intervals: two copies overlap, and the window builder drops overlapping
    windows as an annotation conflict (scripts/windows.py). Rows another
    annotator wrote for the same clip stay, because the gold sample is labelled
    independently by two people (docs/ANNOTATION-GUIDE.md).
    """
    with WRITE_LOCK:
        return _replace_rows(path, clip_id, annotator, rows)


def _replace_rows(path: Path, clip_id: str, annotator: str, rows: list[dict]) -> int:
    kept = [row for row in read_rows(path) if not (row["clip_id"] == clip_id and row["annotator"] == annotator)]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # LF endings match the tracked data/annotations.csv; csv.writer would
    # otherwise emit CRLF and make the file mixed.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", newline="", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(kept + rows)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            shutil.copyfile(path, path.with_suffix(path.suffix + ".bak"))
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return len(rows)


def check_output(out: Path, protected: list[Path]) -> None:
    for path in protected:
        if out.resolve() == path.resolve() or (out.exists() and path.exists() and os.path.samefile(out, path)):
            raise ValueError(f"output is a protected annotation file: {path}")


@dataclass
class Clip:
    """One source clip from the ledger."""

    clip_id: str
    path: Path
    duration_s: float


def clip_payload(clip: Clip, proposals: list[dict], blind: bool) -> dict:
    """Clip metadata for the page. A blind clip carries no proposals at all."""
    payload = {
        "clip_id": clip.clip_id,
        "video_url": f"/video?id={clip.clip_id}",
        "duration_s": clip.duration_s,
        "blind": blind,
    }
    if not blind:
        payload["proposals"] = proposals
    return payload


def load_clips(clips_file: Path, raw_dir: Path) -> list[Clip]:
    """Clip list from data/source-clips.csv, with paths under the raw root."""
    with Path(clips_file).open(newline="", encoding="utf-8-sig") as handle:
        return [
            Clip(row["clip_id"], Path(raw_dir) / row["relative_path"], float(row["duration_s"]))
            for row in csv.DictReader(handle)
        ]


def load_proposals(timeline_file: Path) -> dict[str, list[dict]]:
    """Group proposed intervals by clip_id; empty when the timeline is absent."""
    proposals: dict[str, list[dict]] = {}
    path = Path(timeline_file)
    if not path.exists():
        return proposals
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            proposals.setdefault(row["clip_id"], []).append(
                {
                    "start_s": float(row["start_s"]),
                    "end_s": float(row["end_s"]),
                    "behavior": row["behavior"],
                    "confidence": row["confidence"],
                }
            )
    return proposals


def verified_counts(out_file: Path) -> dict[str, int]:
    """Verified rows per clip_id in the annotations file."""
    counts: dict[str, int] = {}
    path = Path(out_file)
    if not path.exists():
        return counts
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            counts[row["clip_id"]] = counts.get(row["clip_id"], 0) + 1
    return counts


def id_set(ids: list[str], id_file: Path | None) -> set[str]:
    """Clip ids from repeated flags and/or a file with one id per line."""
    found = set(ids)
    if id_file:
        found.update(line.strip() for line in Path(id_file).read_text(encoding="utf-8-sig").splitlines() if line.strip())
    return found


# Clip ids to hide proposals from.
blind_ids = id_set


def restrict_clips(clips: list[Clip], only: set[str]) -> list[Clip]:
    """Only the listed clips, in ledger order; every clip when the list is empty."""
    return [clip for clip in clips if clip.clip_id in only] if only else clips


def pending_counts(counts: dict[str, int], redo: set[str], saved: set[str]) -> dict[str, int]:
    """Verified rows as the page should see them.

    A clip marked for redo counts as unverified until it is saved in this run, so
    the page serves it again even though an earlier pass left rows for it. Saving
    replaces that annotator's earlier rows (replace_rows), so nothing is duplicated.
    """
    return {clip_id: (0 if clip_id in redo and clip_id not in saved else n) for clip_id, n in counts.items()}


# --- server ------------------------------------------------------------------


@dataclass
class Config:
    clips: list[Clip]
    proposals: dict[str, list[dict]]
    blind: set[str] = field(default_factory=set)
    out_file: Path = Path("data/annotations.csv")
    redo: set[str] = field(default_factory=set)
    saved: set[str] = field(default_factory=set)
    edit_existing: bool = False
    protected: list[Path] = field(default_factory=list)


class Handler(BaseHTTPRequestHandler):
    server_version = "civet-review"

    def log_message(self, *args) -> None:
        pass  # progress prints matter more than per-request noise

    @property
    def config(self) -> Config:
        return self.server.config

    def _json(self, payload: dict | list, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _clip(self, clip_id: str) -> Clip | None:
        return next((clip for clip in self.config.clips if clip.clip_id == clip_id), None)

    def _clips_payload(self) -> list[dict]:
        redo = {c.clip_id for c in self.config.clips} if self.config.edit_existing else self.config.redo
        counts = pending_counts(verified_counts(self.config.out_file), redo, self.config.saved)
        return [
            {"clip_id": c.clip_id, "blind": c.clip_id in self.config.blind, "verified": counts.get(c.clip_id, 0)}
            for c in self.config.clips
        ]

    def _reviewed(self, counts: dict[str, int]) -> int:
        return sum(1 for c in self.config.clips if counts.get(c.clip_id))

    def do_GET(self) -> None:
        route = urlparse(self.path)
        args = parse_qs(route.query)
        if route.path in ("/", "/index.html"):
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif route.path == "/api/clips":
            self._json(self._clips_payload())
        elif route.path == "/api/session":
            # Scope browser drafts to this output file, never to another pass.
            identity = str(self.config.out_file.resolve()).encode("utf-8")
            self._json({"review_id": hashlib.sha256(identity).hexdigest(), "output_name": self.config.out_file.name})
        elif route.path == "/api/clip":
            clip = self._clip(args.get("id", [""])[0])
            if clip is None:
                self._json({"error": "unknown clip"}, 404)
                return
            blind = clip.clip_id in self.config.blind
            payload = clip_payload(clip, self.config.proposals.get(clip.clip_id, []), blind)
            if self.config.edit_existing:
                if blind:
                    self._json({"error": "editing existing answers is not a blind pass"}, 400)
                    return
                annotator = args.get("annotator", [""])[0].strip()
                rows = [dict(row) for row in read_rows(self.config.out_file)
                        if row["clip_id"] == clip.clip_id and row["annotator"] == annotator]
                if not rows:
                    self._json({"error": "No saved intervals for this name and clip. Enter the exact original annotator name and select clips from that submission."}, 400)
                    return
                for row in rows:
                    row["start_s"], row["end_s"] = float(row["start_s"]), float(row["end_s"])
                payload.pop("proposals", None)
                payload.update(edit_existing=True, intervals=rows)
            self._json(payload)
        elif route.path == "/video":
            clip = self._clip(args.get("id", [""])[0])
            if clip is None or not clip.path.exists():
                self._json({"error": "unknown clip"}, 404)
                return
            self._serve_video(clip.path)
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/verdict":
            self._json({"error": "not found"}, 404)
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            clip_id = str(body.get("clip_id", ""))
            clip = self._clip(clip_id)
            if clip is None:
                self._json({"error": "unknown clip"}, 404)
                return
            annotator = str(body.get("annotator", "")).strip()
            validate_conditions(body.get("intervals", []))
            rows = annotation_rows(clip_id, annotator, body.get("intervals", []))
            if not rows:
                raise ValueError("add at least one interval before saving")
            ordered = sorted(rows, key=lambda row: float(row["start_s"]))
            if any(float(row["end_s"]) > clip.duration_s + 0.005 for row in ordered):
                raise ValueError("interval ends beyond the clip duration")
            if any(float(b["start_s"]) < float(a["end_s"]) for a, b in zip(ordered, ordered[1:])):
                raise ValueError("intervals must not overlap")
            check_output(self.config.out_file, self.config.protected)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            self._json({"error": str(exc)}, 400)
            return
        try:
            replace_rows(self.config.out_file, clip_id, annotator, rows)
        except OSError as exc:
            self._json({"error": f"Could not save; existing records retained: {exc}"}, 500)
            return
        self.config.saved.add(clip_id)
        counts = verified_counts(self.config.out_file)
        reviewed = self._reviewed(counts)
        print(f"saved {clip_id}: +{len(rows)} rows, {reviewed}/{len(self.config.clips)} clips reviewed -> {self.config.out_file}")
        self._json({"saved": len(rows), "verified": counts.get(clip_id, 0), "reviewed": reviewed, "total": len(self.config.clips)})

    def _serve_video(self, path: Path) -> None:
        """Stream the mp4, honouring a single Range so the browser can seek."""
        size = path.stat().st_size
        start, end, status = 0, size - 1, 200
        header = self.headers.get("Range", "")
        if header:
            try:
                if not header.startswith("bytes=") or "," in header:
                    raise ValueError("unsupported range")
                first, separator, last = header[len("bytes="):].partition("-")
                if not separator:
                    raise ValueError("missing range separator")
                if first:
                    start = int(first)
                    end = min(int(last), size - 1) if last else size - 1
                else:
                    suffix = int(last)
                    if suffix <= 0:
                        raise ValueError("empty suffix")
                    start, end = max(0, size - suffix), size - 1
                if start < 0 or start >= size or end < start:
                    raise ValueError("range outside video")
                status = 206
            except ValueError:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining > 0:
                chunk = handle.read(min(1 << 16, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                    return  # Seeking or closing the player cancels this stream.
                remaining -= len(chunk)


# --- page --------------------------------------------------------------------

PAGE = Path(__file__).with_name("review_page.html").read_text(encoding="utf-8")

# The page's number keys follow labels.ALL_LABELS, so a label added there needs no
# second edit in the JavaScript.
PAGE = PAGE.replace("__BEHAVIORS__", json.dumps(list(BEHAVIORS)))
PAGE = PAGE.replace("__CONDITION_OPTIONS__", json.dumps(CONDITION_OPTIONS))
TRANSLATIONS = json.loads(Path(__file__).with_name("review_i18n.json").read_text(encoding="utf-8"))
PAGE = PAGE.replace("__TRANSLATIONS__", json.dumps(TRANSLATIONS, ensure_ascii=True))


# --- driver ------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clips-file", type=Path, default=Path("data/source-clips.csv"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--timeline", type=Path, default=Path("outputs/prelabel/timeline.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/annotations.csv"))
    parser.add_argument("--protect-file", action="append", type=Path, default=[], help="refuse to write to this annotation file")
    parser.add_argument("--blind", action="append", default=[], metavar="CLIP_ID")
    parser.add_argument("--blind-file", type=Path)
    parser.add_argument("--only", action="append", default=[], metavar="CLIP_ID", help="serve only these clips")
    parser.add_argument("--only-file", type=Path, help="file of clip ids to serve, one per line")
    parser.add_argument("--redo", action="append", default=[], metavar="CLIP_ID", help="serve this clip again even if it already has rows")
    parser.add_argument("--redo-file", type=Path, help="file of clip ids to serve again, one per line")
    parser.add_argument("--edit-existing", action="store_true", help="edit this annotator's saved rows in --out, without machine proposals; not a blind pass")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--auto-port", action="store_true", help="choose a free port when the preferred port is busy")
    parser.add_argument("--open-browser", action="store_true", help="open the system default browser after startup")
    parser.add_argument("--counts", action="store_true", help="print per-clip verified row counts and exit")
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("port must be between 0 and 65535")
    try:
        check_output(args.out, args.protect_file)
    except ValueError as exc:
        parser.error(str(exc))

    clips = load_clips(args.clips_file, args.raw)
    known = {clip.clip_id for clip in clips}
    blind = blind_ids(args.blind, args.blind_file)
    if args.edit_existing and blind:
        parser.error("--edit-existing cannot be combined with blind clips; use a separate editing session")
    only = id_set(args.only, args.only_file)
    redo = id_set(args.redo, args.redo_file)
    for name, ids in (("blind", blind), ("only", only), ("redo", redo)):
        unknown = ids - known
        if unknown:
            parser.error(f"{name} clips not in {args.clips_file}: {sorted(unknown)}")
    clips = restrict_clips(clips, only)

    if args.counts:
        counts = verified_counts(args.out)
        for clip in clips:
            print(f"{clip.clip_id}: {counts.get(clip.clip_id, 0)}")
        print(f"reviewed {sum(1 for c in clips if counts.get(c.clip_id))}/{len(clips)} clips -> {args.out}")
        return 0

    config = Config(clips=clips, proposals=load_proposals(args.timeline), blind=blind, out_file=args.out, redo=redo, protected=args.protect_file, edit_existing=args.edit_existing)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError:
        if not args.auto_port:
            parser.error(f"cannot listen on port {args.port}; try --port 0 for a free port")
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.config = config
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"Review: {url}\n{len(clips)} clips, {len(blind)} blind. Save to: {args.out}\nKeep this terminal open. Stop with Ctrl+C.", flush=True)
    if args.open_browser:
        def open_browser():
            try:
                if not webbrowser.open(url):
                    print(f"Open this URL manually in a browser: {url}", flush=True)
            except webbrowser.Error:
                print(f"Open this URL manually in a browser: {url}", flush=True)
        threading.Thread(target=open_browser, daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
