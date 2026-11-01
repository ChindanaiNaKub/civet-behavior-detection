#!/usr/bin/env python3
"""The active and inactive table and its two figures.

Reads the scan from scripts/activity.py and turns it into the table the brief asks
for: per camera, how much of the night the animal was in view, how much of that it
was moving, and how the activity was broken into bouts. Absent is a column of its
own, because a camera that sees an empty enclosure is the normal case and folding
it into Inactive would make every camera look like a resting animal.

The figures are the same numbers as a picture: one row per camera over the clock,
and the active minutes of each hour side by side. Everything on them comes from
the scan, nothing is drawn by hand.

Usage:
    python3 scripts/activity_report.py
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import activity as A  # noqa: E402

COLORS = {"Active": "#1b7f79", "Inactive": "#e0a33a", "Absent": "#dedede"}
ORDER = ("Active", "Inactive", "Absent")
# One colour per camera, so the four series in the hourly figure stay apart.
CAMERA_COLORS = ("#9fc6c3", "#1b7f79", "#5b6bb5", "#b5485d")
NIGHT_HOURS = (18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5)


def read_windows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_clips(path: Path) -> list[dict]:
    """The camera ledger, with the folder name read as the camera name."""
    with path.open(newline="") as handle:
        clips = list(csv.DictReader(handle))
    for clip in clips:
        clip["camera"] = clip["folder_label_weak"]
    return clips


def timeline(clip: dict, windows: list[dict]) -> list[tuple[float, float, str]]:
    """Every five-second cell of the recording, with the ones no window covers absent."""
    duration = float(clip["duration_s"])
    mine = [w for w in windows if w["clip_id"] == clip["clip_id"]]
    out: list[tuple[float, float, str]] = []
    start = 0.0
    for row in mine:
        begin = float(row["start_s"])
        if begin > start:
            out.append((start, begin, "Absent"))
        out.append((begin, float(row["end_s"]), row["behavior"]))
        start = float(row["end_s"])
    if start < duration:
        out.append((start, duration, "Absent"))
    return out


def bouts(spans: list[tuple[float, float, str]], label: str) -> list[float]:
    """Durations of consecutive runs of one label."""
    runs, length = [], 0.0
    for start, end, state in spans:
        if state == label:
            length += end - start
            continue
        if length:
            runs.append(length)
        length = 0.0
    if length:
        runs.append(length)
    return runs


def summarise(clips: list[dict], windows: list[dict], unmeasured: dict[str, float]) -> list[dict]:
    rows = []
    for clip in clips:
        spans = timeline(clip, windows)
        total = float(clip["duration_s"])
        seconds = defaultdict(float)
        for start, end, state in spans:
            seconds[state] += end - start
        # Unmeasured seconds are counted out of Absent rather than left in it: the
        # scan could not see those frames, so it cannot call them empty.
        skipped = min(unmeasured.get(clip["clip_id"], 0.0), seconds["Absent"])
        seconds["Absent"] -= skipped
        runs = bouts(spans, "Active")
        rows.append(
            {
                "camera": clip["camera"],
                "clip_id": clip["clip_id"],
                "hours": total / 3600.0,
                "present": (seconds["Active"] + seconds["Inactive"]) / total,
                "active": seconds["Active"] / total,
                "inactive": seconds["Inactive"] / total,
                "absent": seconds["Absent"] / total,
                "unmeasured": skipped / total,
                "active_s": seconds["Active"],
                "bouts": len(runs),
                "median_bout_s": sorted(runs)[len(runs) // 2] if runs else 0.0,
                "longest_bout_s": max(runs) if runs else 0.0,
            }
        )
    return rows


def table_markdown(rows: list[dict]) -> str:
    lines = [
        "# Active and inactive table",
        "",
        "Shares are of the recorded time of each camera. Present is Active plus Inactive. "
        "A bout is a run of consecutive Active windows; one window is five seconds.",
        "",
        "| Camera | Clip | Recorded (h) | Present | Active | Inactive | Absent | Unmeasured | Active bouts | Median bout (s) | Longest bout (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['camera']} | `{row['clip_id']}` | {row['hours']:.2f} | {row['present']:.1%} | "
            f"{row['active']:.1%} | {row['inactive']:.1%} | {row['absent']:.1%} | {row['unmeasured']:.1%} | "
            f"{row['bouts']} | {row['median_bout_s']:.0f} | {row['longest_bout_s']:.0f} |"
        )
    total = sum(row["hours"] for row in rows)
    active = sum(row["active_s"] for row in rows) / (total * 3600)
    lines.append(
        f"| **All** | 4 clips | {total:.2f} | | {active:.1%} | | | | "
        f"{sum(row['bouts'] for row in rows)} | | |"
    )
    return "\n".join(lines) + "\n"


def write_table_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in row.items()})


def plot_timeline(path: Path, clips: list[dict], windows: list[dict]) -> None:
    """One row per camera, coloured by state, on the clock of the recording night.

    All four cameras are drawn on one clock axis, so the rows line up by wall time
    and a camera that started later simply has a shorter row.
    """
    figure, axis = plt.subplots(figsize=(12, 3.6), dpi=160)
    starts = [datetime.fromisoformat(clip["capture_datetime_overlay"]) for clip in clips]
    ends = [start + timedelta(seconds=float(clip["duration_s"])) for start, clip in zip(starts, clips)]
    for index, (clip, start) in enumerate(zip(clips, starts)):
        runs: list[tuple[float, float, str]] = []
        for begin, end, state in timeline(clip, windows):
            if runs and runs[-1][2] == state and abs(runs[-1][1] - begin) < 1e-6:
                runs[-1] = (runs[-1][0], end, state)
            else:
                runs.append((begin, end, state))
        for begin, end, state in runs:
            left = mdates.date2num(start + timedelta(seconds=begin))
            right = mdates.date2num(start + timedelta(seconds=end))
            axis.barh(index, right - left, left=left, height=0.62, color=COLORS[state], linewidth=0)
        axis.annotate(
            clip["camera"], xy=(0, index), xycoords=("axes fraction", "data"),
            xytext=(-8, 0), textcoords="offset points", ha="right", va="center", fontsize=9,
        )

    axis.set_yticks([])
    axis.set_ylim(len(clips) - 0.4, -0.9)
    axis.set_xlim(mdates.date2num(min(starts)), mdates.date2num(max(ends)))
    axis.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axis.set_xlabel("clock time (Asia/Bangkok), night of 2026-04-05")
    axis.set_title("Banded civet, one night on four fixed cameras", loc="left", fontsize=11, pad=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[state]) for state in ORDER]
    figure.legend(
        handles, ORDER, loc="upper right", ncol=3, frameon=False, fontsize=9,
        handlelength=1.2, borderaxespad=0.4,
    )
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.tick_params(axis="y", length=0)
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(path)
    plt.close(figure)


def plot_hours(path: Path, clips: list[dict], windows: list[dict]) -> None:
    """Active minutes in each clock hour, so the nightly rhythm is readable."""
    hours = list(NIGHT_HOURS)
    figure, axis = plt.subplots(figsize=(12, 3.2), dpi=160)
    width = 0.8 / len(clips)
    for index, clip in enumerate(clips):
        start = datetime.fromisoformat(clip["capture_datetime_overlay"])
        active = defaultdict(float)
        for begin, end, state in timeline(clip, windows):
            if state != "Active":
                continue
            # A five-second window can straddle an hour, so it is split at the hour.
            clock = start + timedelta(seconds=begin)
            end_clock = start + timedelta(seconds=end)
            while clock < end_clock:
                boundary = clock.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
                stop = min(boundary, end_clock)
                active[clock.hour] += (stop - clock).total_seconds() / 60.0
                clock = stop
        axis.bar(
            [position + index * width - 0.4 for position in range(len(hours))],
            [active[hour] for hour in hours],
            width=width, color=CAMERA_COLORS[index % len(CAMERA_COLORS)],
            label=clip["camera"], linewidth=0,
        )
    axis.set_xticks(range(len(hours)))
    axis.set_xticklabels([f"{hour:02d}" for hour in hours], fontsize=8)
    axis.set_xlabel("clock hour (Asia/Bangkok)")
    axis.set_ylabel("active minutes")
    axis.set_title("When the animal was moving", loc="left", fontsize=11)
    axis.legend(frameon=False, ncol=len(clips), fontsize=8, loc="upper left")
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("outputs/activity"))
    parser.add_argument("--figures", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()

    clips = read_clips(args.out / "source-clips.csv")
    windows = read_windows(args.out / "windows.csv")
    scan = json.loads((args.out / "scan.json").read_text())
    unmeasured = {clip["clip_id"]: float(clip["unmeasured_s"]) for clip in scan["clips"]}
    rows = summarise(clips, windows, unmeasured)

    (args.out / "table.md").write_text(table_markdown(rows))
    write_table_csv(args.out / "table.csv", rows)
    args.figures.mkdir(parents=True, exist_ok=True)
    plot_timeline(args.figures / "activity-timeline.png", clips, windows)
    plot_hours(args.figures / "activity-by-hour.png", clips, windows)

    print(table_markdown(rows))
    print(f"figures -> {args.figures}/activity-timeline.png and activity-by-hour.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
