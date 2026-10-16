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
    python scripts/review.py --counts            # per-clip verified row counts
"""

from __future__ import annotations

import argparse
import csv
import json
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
        if end <= start:
            raise ValueError(f"end_s must exceed start_s for {behavior} interval")
        reason = str(interval.get("uncertain_reason", "")).strip()
        if behavior == "Uncertain" and not reason:
            raise ValueError("Uncertain interval needs an uncertain_reason")
        row = {name: str(interval.get(name) or "unknown") for name in CONDITIONS}
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
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != CSV_FIELDS:
            raise ValueError(f"{path}: header is {reader.fieldnames}, expected {CSV_FIELDS}")
        return [{name: row.get(name, "") for name in CSV_FIELDS} for row in reader]


def replace_rows(path: Path, clip_id: str, annotator: str, rows: list[dict]) -> int:
    """Write one annotator's rows for one clip, keeping every other row.

    Idempotent, because a re-saved clip must not leave two copies of its
    intervals: two copies overlap, and the window builder drops overlapping
    windows as an annotation conflict (scripts/windows.py). Rows another
    annotator wrote for the same clip stay, because the gold sample is labelled
    independently by two people (docs/ANNOTATION-GUIDE.md).
    """
    kept = [row for row in read_rows(path) if not (row["clip_id"] == clip_id and row["annotator"] == annotator)]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # LF endings match the tracked data/annotations.csv; csv.writer would
    # otherwise emit CRLF and make the file mixed.
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(kept + rows)
    return len(rows)


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
    with Path(clips_file).open(newline="") as handle:
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
    with path.open(newline="") as handle:
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
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            counts[row["clip_id"]] = counts.get(row["clip_id"], 0) + 1
    return counts


def blind_ids(blind: list[str], blind_file: Path | None) -> set[str]:
    """Clip ids to hide proposals from, from repeated flags and/or a file."""
    ids = set(blind)
    if blind_file:
        ids.update(line.strip() for line in Path(blind_file).read_text().splitlines() if line.strip())
    return ids


# --- server ------------------------------------------------------------------


@dataclass
class Config:
    clips: list[Clip]
    proposals: dict[str, list[dict]]
    blind: set[str] = field(default_factory=set)
    out_file: Path = Path("data/annotations.csv")


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
        counts = verified_counts(self.config.out_file)
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
        elif route.path == "/api/clip":
            clip = self._clip(args.get("id", [""])[0])
            if clip is None:
                self._json({"error": "unknown clip"}, 404)
                return
            blind = clip.clip_id in self.config.blind
            self._json(clip_payload(clip, self.config.proposals.get(clip.clip_id, []), blind))
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
            if self._clip(clip_id) is None:
                self._json({"error": "unknown clip"}, 404)
                return
            annotator = str(body.get("annotator", "")).strip()
            rows = annotation_rows(clip_id, annotator, body.get("intervals", []))
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            self._json({"error": str(exc)}, 400)
            return
        replace_rows(self.config.out_file, clip_id, annotator, rows)
        counts = verified_counts(self.config.out_file)
        reviewed = self._reviewed(counts)
        print(f"saved {clip_id}: +{len(rows)} rows, {reviewed}/{len(self.config.clips)} clips reviewed -> {self.config.out_file}")
        self._json({"saved": len(rows), "verified": counts.get(clip_id, 0), "reviewed": reviewed, "total": len(self.config.clips)})

    def _serve_video(self, path: Path) -> None:
        """Stream the mp4, honouring a single Range so the browser can seek."""
        size = path.stat().st_size
        start, end, status = 0, size - 1, 200
        header = self.headers.get("Range", "")
        if header.startswith("bytes="):
            first, _, last = header[len("bytes="):].split(",")[0].partition("-")
            start = int(first) if first else 0
            end = min(int(last), size - 1) if last else size - 1
            status = 206
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
                self.wfile.write(chunk)
                remaining -= len(chunk)


# --- page --------------------------------------------------------------------

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>civet interval review</title><style>
body{font:14px/1.4 system-ui,sans-serif;margin:1rem;background:#181818;color:#eee}
video{width:100%;max-height:60vh;background:#000}
#strip{position:relative;height:44px;background:#2a2a2a;border:1px solid #444;margin:.5rem 0}
.blk{position:absolute;top:4px;height:36px;overflow:hidden;border-radius:3px;font-size:11px;line-height:36px;text-align:center;color:#111;cursor:pointer}
.blk.sel{outline:2px solid #fff;z-index:2}#play{position:absolute;top:0;width:2px;height:100%;background:#f33;pointer-events:none}
table{border-collapse:collapse;width:100%}td,th{border:1px solid #444;padding:2px 4px;font-size:12px}
.badge{background:#b33;padding:2px 6px;border-radius:3px}#help,#status{color:#bbb;white-space:pre-wrap;margin:.4rem 0}
</style></head><body>
<h3>civet interval review</h3><div id="status"></div>
<video id="player" controls preload="metadata"></video><div id="strip"></div>
<div id="help"></div>
<table id="list"><thead><tr><th>#</th><th>behavior</th><th>start_s</th><th>end_s</th><th>reason</th></tr></thead><tbody></tbody></table>
<script>
const COLORS={Eating:"#8fd694",Locomotion:"#8fb8f0",Resting:"#c9a0dc",Climbing:"#f0b070",Transition:"#ddd",Uncertain:"#f08f8f"};
const KEYS=__BEHAVIORS__;
const HELP="space play/pause | left/right select interval | 1-6 set behavior | , . start -/+0.1 | ; ' end -/+0.1 | n add at playhead | d delete selected | r set uncertain reason | Enter accept + save + next clip | p skip clip";
let clips=[],clip=null,intervals=[],sel=0,annotator="";
const $=id=>document.getElementById(id),fmt=n=>n.toFixed(2),api=async(p,o)=>(await fetch(p,o)).json();
const reviewedCount=()=>clips.filter(c=>c.verified>0).length;
const nextClipId=()=>{const c=clips.find(x=>x.verified===0);return c?c.clip_id:null;};
async function start(){
  annotator=prompt("Annotator name:")||"";
  if(!annotator.trim()){alert("An annotator name is required.");return start();}
  clips=await api("/api/clips");load(nextClipId());
}
async function load(clip_id){
  if(!clip_id){$("status").textContent="all clips reviewed";$("help").textContent="";return;}
  clip=await api("/api/clip?id="+encodeURIComponent(clip_id));
  intervals=(clip.proposals||[]).map(p=>({start_s:p.start_s,end_s:p.end_s,behavior:p.behavior,visibility:"unknown",lighting:"unknown",camera_view:"unknown",occlusion:"unknown",notes:"",uncertain_reason:p.behavior==="Uncertain"?"machine proposal":""}));
  sel=0;$("player").src=clip.video_url;render();
}
function render(){
  $("status").innerHTML=clip.clip_id+(clip.blind?' <span class="badge">BLIND</span>':"")+" — "+reviewedCount()+"/"+clips.length+" clips reviewed";
  const dur=clip.duration_s||1,strip=$("strip");strip.innerHTML="";
  intervals.forEach((iv,i)=>{const b=document.createElement("div");b.className="blk"+(i===sel?" sel":"");
    b.style.left=iv.start_s/dur*100+"%";b.style.width=Math.max((iv.end_s-iv.start_s)/dur*100,.5)+"%";
    b.style.background=COLORS[iv.behavior]||"#999";b.textContent=iv.behavior;b.onclick=()=>{sel=i;render();};strip.appendChild(b);});
  const ph=document.createElement("div");ph.id="play";ph.style.left=(($("player").currentTime||0)/dur*100)+"%";strip.appendChild(ph);
  const tb=$("list").querySelector("tbody");tb.innerHTML="";
  intervals.forEach((iv,i)=>{const tr=document.createElement("tr");tr.style.background=i===sel?"#333":"";
    tr.innerHTML="<td>"+(i+1)+"</td><td>"+iv.behavior+"</td><td>"+fmt(iv.start_s)+"</td><td>"+fmt(iv.end_s)+"</td><td>"+(iv.uncertain_reason||"")+"</td>";
    tr.onclick=()=>{sel=i;render();};tb.appendChild(tr);});
  $("help").textContent=HELP;
}
async function save(){
  if(!intervals.length){alert("Add at least one interval before saving.");return;}
  const res=await api("/api/verdict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({clip_id:clip.clip_id,annotator:annotator,intervals:intervals})});
  if(res.error){alert("Rejected: "+res.error);return;}
  const c=clips.find(x=>x.clip_id===clip.clip_id);if(c)c.verified=res.verified;load(nextClipId());
}
$("player").addEventListener("timeupdate",()=>{const p=document.getElementById("play");if(p&&clip)p.style.left=($("player").currentTime/clip.duration_s*100)+"%";});
document.addEventListener("keydown",e=>{
  if(!clip||e.target.tagName==="INPUT")return;
  const iv=intervals[sel],nudge=(k,d)=>{if(iv){iv[k]=Math.max(0,+(iv[k]+d).toFixed(2));render();}};
  if(e.key===" "){e.preventDefault();const v=$("player");v.paused?v.play():v.pause();}
  else if(e.key==="ArrowLeft"){sel=Math.max(0,sel-1);render();}
  else if(e.key==="ArrowRight"){sel=Math.min(intervals.length-1,sel+1);render();}
  else if(e.key>="1"&&e.key<="6"&&iv){iv.behavior=KEYS[+e.key-1];if(iv.behavior==="Uncertain"&&!iv.uncertain_reason)iv.uncertain_reason="unspecified";render();}
  else if(e.key===",")nudge("start_s",-0.1);
  else if(e.key===".")nudge("start_s",0.1);
  else if(e.key===";")nudge("end_s",-0.1);
  else if(e.key==="'")nudge("end_s",0.1);
  else if(e.key==="d"&&iv){intervals.splice(sel,1);sel=Math.max(0,sel-1);render();}
  else if(e.key==="r"&&iv){const r=prompt("Uncertain reason:",iv.uncertain_reason||"");if(r!==null){iv.uncertain_reason=r;render();}}
  else if(e.key==="n"){const t=+$("player").currentTime;intervals.push({start_s:+t.toFixed(2),end_s:+(t+2).toFixed(2),behavior:"Uncertain",uncertain_reason:"to label",visibility:"unknown",lighting:"unknown",camera_view:"unknown",occlusion:"unknown",notes:""});sel=intervals.length-1;render();}
  else if(e.key==="Enter"){e.preventDefault();save();}
  else if(e.key==="p")load(nextClipId());
});
start();
</script></body></html>"""

# The page's number keys follow labels.ALL_LABELS, so a label added there needs no
# second edit in the JavaScript.
PAGE = PAGE.replace("__BEHAVIORS__", json.dumps(list(BEHAVIORS)))


# --- driver ------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clips-file", type=Path, default=Path("data/source-clips.csv"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--timeline", type=Path, default=Path("outputs/prelabel/timeline.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/annotations.csv"))
    parser.add_argument("--blind", action="append", default=[], metavar="CLIP_ID")
    parser.add_argument("--blind-file", type=Path)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--counts", action="store_true", help="print per-clip verified row counts and exit")
    args = parser.parse_args()

    clips = load_clips(args.clips_file, args.raw)
    blind = blind_ids(args.blind, args.blind_file)
    unknown = blind - {clip.clip_id for clip in clips}
    if unknown:
        parser.error(f"blind clips not in {args.clips_file}: {sorted(unknown)}")

    if args.counts:
        counts = verified_counts(args.out)
        for clip in clips:
            print(f"{clip.clip_id}: {counts.get(clip.clip_id, 0)}")
        print(f"reviewed {sum(1 for c in clips if counts.get(c.clip_id))}/{len(clips)} clips -> {args.out}")
        return 0

    config = Config(clips=clips, proposals=load_proposals(args.timeline), blind=blind, out_file=args.out)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.config = config
    print(f"review http://127.0.0.1:{args.port}/  ({len(clips)} clips, {len(blind)} blind) -> {args.out}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
