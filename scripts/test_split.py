"""Self-check for the split ledger rules, with no video, GPU or model download.

Run with `python -m pytest scripts`.
"""

import csv
from collections import Counter
from pathlib import Path

import pytest

import labels
import split as S
import windows as W


def _iv(clip_id, start, end, behavior):
    return W.Interval(clip_id, start, end, behavior)


def test_every_source_clip_is_assigned_exactly_once():
    path = Path(__file__).resolve().parent.parent / "data" / "source-clips.csv"
    with path.open(newline="") as handle:
        known = {row["clip_id"] for row in csv.DictReader(handle)}
    assigned = [c for ids in S.ASSIGNMENT.values() for c in ids]
    assert len(assigned) == len(set(assigned))
    assert set(assigned) == known


def test_the_gold_list_is_the_test_split():
    path = Path(__file__).resolve().parent.parent / "data" / "gold-clips.txt"
    gold = {line.strip() for line in path.read_text().splitlines() if line.strip()}
    assert gold == set(S.ASSIGNMENT["test"])


def test_window_counts_follow_the_assignment():
    intervals = [_iv("a", 0.0, 10.0, "Eating"), _iv("b", 0.0, 10.0, "Resting")]
    counts = S.window_counts(intervals, {"a": 10.0, "b": 10.0}, {"train": ["a"], "validation": [], "test": ["b"]})
    assert counts[("train", "Eating")] == 2
    assert counts[("test", "Resting")] == 2
    assert counts[("test", "Eating")] == 0


def test_a_split_with_no_window_for_a_class_is_reported():
    counts = Counter({("train", "Eating"): 2, ("train", "Locomotion"): 1, ("train", "Resting"): 3, ("test", "Eating"): 1})
    gaps = S.verified_gaps(counts)
    assert gaps["train"] == []
    assert gaps["test"] == ["Locomotion", "Resting"]
    assert gaps["validation"] == list(labels.MODEL_CLASSES)


def test_uncertain_footage_gives_no_resting_window():
    # A Resting-folder clip the annotator could not see produces no Resting window,
    # which is how the first split ended up with an empty Resting test class.
    intervals = [_iv("c", 0.0, 30.0, "Uncertain")]
    counts = S.window_counts(intervals, {"c": 30.0}, {"train": [], "validation": [], "test": ["c"]})
    assert S.verified_gaps(counts)["test"] == list(labels.MODEL_CLASSES)


def test_an_unsigned_ledger_has_no_signed_lock(tmp_path):
    path = tmp_path / "splits.csv"
    path.write_text("clip_id,split,grouping_key,grouping_evidence,assignment_reason,test_locked_by,test_locked_at\n"
                    "a,test,k,e,r,,\n")
    assert S.signed_lock(path) == []
    assert S.signed_lock(tmp_path / "missing.csv") == []


def test_a_signed_ledger_is_reported_so_it_is_not_overwritten(tmp_path):
    path = tmp_path / "splits.csv"
    path.write_text("clip_id,split,grouping_key,grouping_evidence,assignment_reason,test_locked_by,test_locked_at\n"
                    "a,test,k,e,r,Ada,2026-10-20T22:00:00+07:00\nb,train,k,e,r,,\n")
    signed = S.signed_lock(path)
    assert [row["clip_id"] for row in signed] == ["a"]
    assert signed[0]["test_locked_by"] == "Ada"
