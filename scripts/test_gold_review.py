"""Check blind handoff contents and complete, independent gold-pass comparison."""

import csv
import subprocess
import sys
from pathlib import Path

import pytest

import agreement
import prepare_gold_review
import review


def test_packet_contains_neutral_metadata_and_no_answers(tmp_path):
    root = tmp_path / "repo"
    (root / "data/raw/Eating").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "data/gold-clips.txt").write_text("c\n")
    (root / "data/raw/Eating/c.mp4").write_bytes(b"private video")
    (root / "data/source-clips.csv").write_text(
        "clip_id,relative_path,duration_s,folder_label_weak\nc,Eating/c.mp4,10,Eating\n"
    )
    (root / "data/annotations.csv").write_text("FIRST PASS ANSWERS")
    for name in ("review.py", "labels.py", "review_page.html"):
        (root / "scripts" / name).write_text("# review tool\n")
    (root / "GLOSSARY.md").write_text("# Definitions\n")
    out = tmp_path / "packet"
    assert prepare_gold_review.prepare(root, out) == 1
    files = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
    assert files == {
        "scripts/review.py", "scripts/review_page.html", "scripts/labels.py", "GLOSSARY.md", "README.md",
        "data/clips.csv", "data/gold-clips.txt", "data/raw/c.mp4", "start_review.py",
    }
    assert "Eating" not in (out / "data/clips.csv").read_text()
    assert (out / "data/raw/c.mp4").read_bytes() == b"private video"
    assert all("FIRST PASS ANSWERS" not in p.read_text() for p in out.rglob("*.csv"))
    with pytest.raises(FileExistsError):
        prepare_gold_review.prepare(root, out)


def rows(annotator="Ada", clip="c", **overrides):
    interval = {"start_s": 0, "end_s": 5, "behavior": "Eating", **overrides}
    return review.annotation_rows(clip, annotator, [interval])


def test_gold_pass_requires_all_clips_and_consistent_identity():
    with pytest.raises(ValueError, match="missing gold"):
        agreement.gold_pass(rows(), {"c", "d"}, "second")
    with pytest.raises(ValueError, match="identity"):
        agreement.gold_pass(rows() + rows("Bob", "d"), {"c", "d"}, "second")
    with pytest.raises(ValueError, match="non-gold"):
        agreement.gold_pass(rows() + rows(clip="d"), {"c"}, "second")
    assert len(agreement.gold_pass(rows() + rows(clip="d"), {"c"}, "first", allow_other=True)) == 1


@pytest.mark.parametrize("start,end", [(-1, 5), (5, 5), (0, float("nan")), (0, float("inf"))])
def test_gold_pass_rejects_invalid_times(start, end):
    bad = rows()
    bad[0].update(start_s=str(start), end_s=str(end))
    with pytest.raises(ValueError, match="invalid interval"):
        agreement.gold_pass(bad, {"c"}, "second")


def test_gold_pass_rejects_overlap():
    with pytest.raises(ValueError, match="overlapping"):
        agreement.gold_pass(rows() + rows(start_s=4, end_s=7), {"c"}, "second")


def test_agreement_cli_rejects_same_person_and_preserves_reconciliation(tmp_path):
    a, b, gold, out = [tmp_path / name for name in ("a.csv", "b.csv", "gold.txt", "agreement.csv")]
    gold.write_text("c\n")
    review.replace_rows(a, "c", "Ada", rows())
    review.replace_rows(b, "c", " ada ", rows(" ada "))
    script = Path(agreement.__file__)
    command = [sys.executable, str(script), "--a", str(a), "--b", str(b),
               "--gold-file", str(gold), "--out", str(out)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0 and "different annotators" in result.stderr
    assert not out.exists()
    b.unlink()
    review.replace_rows(b, "c", "Bob", rows("Bob"))
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "1.000 on 1 shared clips" in result.stdout
    with out.open(newline="") as handle:
        ledger = list(csv.DictReader(handle))
    ledger[0]["reviewed_by"] = "Human reviewer"
    agreement.write_ledger(out, ledger)
    before = out.read_bytes()
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0 and "reconciliation work" in result.stderr
    assert out.read_bytes() == before
