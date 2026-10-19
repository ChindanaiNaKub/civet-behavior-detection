"""Build a standalone blind-review folder without first-pass labels or proposals.

Run from the repository root: python scripts/prepare_gold_review.py
The output stays private and is not committed. Give it to a second person who
has not seen the existing annotations or machine proposals.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import review


def prepare(root: Path, out: Path) -> int:
    gold = review.id_set([], root / "data/gold-clips.txt")
    if not gold:
        raise ValueError("gold clip list is empty")
    clips = review.load_clips(root / "data/source-clips.csv", root / "data/raw")
    selected = [clip for clip in clips if clip.clip_id in gold]
    missing = gold - {clip.clip_id for clip in selected}
    if missing:
        raise ValueError(f"gold clips not in source ledger: {sorted(missing)}")
    for clip in selected:
        if not clip.path.is_file():
            raise ValueError(f"source video missing: {clip.clip_id}")
    # Refuse reuse: an existing folder may contain answers or a review in progress.
    out.mkdir(parents=True, exist_ok=False)
    (out / "scripts").mkdir()
    (out / "data/raw").mkdir(parents=True)
    shutil.copyfile(root / "scripts/review.py", out / "scripts/review.py")
    shutil.copyfile(root / "scripts/review_page.html", out / "scripts/review_page.html")
    # The server only needs these contracts. Omit model metadata and comments
    # about which behaviors have been found in the existing first pass.
    (out / "scripts/labels.py").write_text(
        f"ALL_LABELS = {tuple(review.BEHAVIORS)!r}\n"
        f"ANNOTATION_FIELDS = {tuple(review.CSV_FIELDS)!r}\n", encoding="utf-8",
    )
    shutil.copyfile(root / "GLOSSARY.md", out / "GLOSSARY.md")
    # Do not copy the full annotation guide: it includes conclusions about clips.
    with (out / "data/clips.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["clip_id", "relative_path", "duration_s"], lineterminator="\n")
        writer.writeheader()
        for clip in selected:
            filename = f"{clip.clip_id}.mp4"
            shutil.copyfile(clip.path, out / "data/raw" / filename)
            writer.writerow({"clip_id": clip.clip_id, "relative_path": filename, "duration_s": clip.duration_s})
    (out / "data/gold-clips.txt").write_text("\n".join(sorted(gold)) + "\n", encoding="utf-8")
    (out / "start_review.py").write_text(
        '"""Start blind review from any working directory; Python standard library only."""\n'
        "import os\nfrom pathlib import Path\nimport runpy\nimport sys\n\n"
        "root = Path(__file__).resolve().parent\nos.chdir(root)\n"
        "sys.path.insert(0, str(root / 'scripts'))\n"
        "sys.argv = ['review.py', '--clips-file', 'data/clips.csv', "
        "'--only-file', 'data/gold-clips.txt', '--blind-file', 'data/gold-clips.txt', "
        "'--out', 'data/annotations-second.csv', "
        "'--protect-file', str(root.parents[1] / 'data/annotations.csv')]\n"
        "runpy.run_path(str(root / 'scripts/review.py'), run_name='__main__')\n",
        encoding="utf-8",
    )
    (out / "README.md").write_text(
        "# Independent blind behavior review\n\n"
        "Use Python 3.10 or later. No packages need installing. Keep this folder private; "
        "it includes project footage. Do not publish or upload it outside team-approved storage.\n\n"
        "1. Work independently. Do not open the project's first-pass annotations, "
        "machine proposals, or issue discussions about individual clips before submitting. "
        "If you have already seen those answers, tell the coordinator.\n"
        "2. Read GLOSSARY.md. Assign one behavior per interval. Use the most specific "
        "visible behavior when actions overlap, or Uncertain with a reason. Mark changes "
        "between behaviors as Transition. For Resting, require five continuous inactive "
        "seconds and start the interval at inactivity onset once confirmed.\n"
        "3. Run `python start_review.py`, then open http://127.0.0.1:8000/. "
        "Enter your own name consistently for all clips. Watch each whole clip. "
        "Click Start after entering your name. Use Add record, the behavior selector, "
        "the start/end time fields, and Save clip. Keyboard shortcuts are optional. "
        "Cover every period you can assess; label obstructed or ambiguous periods "
        "Uncertain instead of guessing.\n"
        "4. Review all eight clips. You can stop and restart; saved clips are skipped. "
        "Do not change the output file or use a different name to resume.\n"
        "5. Return only data/annotations-second.csv to the coordinator, with confirmation "
        "that you worked without seeing existing answers. Do not discuss labels until "
        "both independent passes are submitted.\n\n"
        "Saving records your intervals; it does not certify that review or reconciliation "
        "is complete. The coordinator checks coverage, compares both passes, reconciles "
        "differences with the reviewers, and arranges a human test-set lock afterward.\n",
        encoding="utf-8",
    )
    return len(selected)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("outputs/gold-review"))
    args = parser.parse_args()
    try:
        count = prepare(Path(__file__).resolve().parents[1], args.out)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    print(f"{count} blind clips -> {args.out}; return data/annotations-second.csv after independent review")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
