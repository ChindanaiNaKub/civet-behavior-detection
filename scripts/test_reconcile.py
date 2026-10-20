"""Self-check for the gold reconciliation rules in reconcile.py.

No server, video or GPU: run with `python -m pytest scripts`.
"""

import csv
import json
from pathlib import Path

import pytest

import agreement
import reconcile as R
import review


def _row(clip, start, end, behavior, annotator, **extra):
    row = {
        "clip_id": clip, "start_s": f"{start:.2f}", "end_s": f"{end:.2f}", "behavior": behavior,
        "visibility": "clear", "lighting": "daylight/color", "camera_view": "elevated oblique",
        "occlusion": "none", "uncertain_reason": "", "annotator": annotator, "notes": "",
    }
    row.update(extra)
    return row


def _write(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=review.CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _iv(start, end, behavior):
    return {"start_s": start, "end_s": end, "behavior": behavior}


def test_spans_join_equal_stretches_and_flag_a_label_dispute():
    a = [_iv(0.0, 10.0, "Resting")]
    b = [_iv(0.0, 6.0, "Resting"), _iv(6.0, 10.0, "Locomotion")]
    found = R.spans(R.intervals_of([]) or a, b, 10.0)
    assert [(s["start"], s["end"], s["a"], s["b"], s["agree"]) for s in found] == [
        (0.0, 6.0, "Resting", "Resting", True),
        (6.0, 10.0, "Resting", "Locomotion", False),
    ]


def test_a_boundary_shift_is_a_short_disagreement():
    a = [_iv(0.0, 5.0, "Locomotion"), _iv(5.0, 10.0, "Resting")]
    b = [_iv(0.0, 5.3, "Locomotion"), _iv(5.3, 10.0, "Resting")]
    disagreeing = [s for s in R.spans(a, b, 10.0) if not s["agree"]]
    assert len(disagreeing) == 1
    assert disagreeing[0]["short"] is True


def test_a_stretch_only_one_pass_labelled_is_a_disagreement():
    found = R.spans([_iv(0.0, 4.0, "Eating")], [], 6.0)
    assert [(s["a"], s["b"], s["agree"]) for s in found] == [("Eating", None, False), (None, None, True)]


def _final(*items):
    return [{"behavior": b, "start_s": s, "end_s": e, "visibility": "clear", "lighting": "daylight/color",
             "camera_view": "elevated oblique", "occlusion": "none", "uncertain_reason": "unclear" if b == "Uncertain" else ""}
            for b, s, e in items]


def test_final_rows_get_the_reconciled_identity_and_are_ordered():
    rows = R.check_final("clip", _final(("Resting", 5.0, 9.0), ("Locomotion", 0.0, 5.0)), 10.0)
    assert [r["behavior"] for r in rows] == ["Locomotion", "Resting"]
    assert {r["annotator"] for r in rows} == {R.RECONCILED}


def test_overlapping_final_intervals_are_rejected():
    with pytest.raises(ValueError, match="overlaps"):
        R.check_final("clip", _final(("Locomotion", 0.0, 6.0), ("Resting", 5.0, 9.0)), 10.0)


def test_a_final_interval_past_the_clip_is_rejected():
    with pytest.raises(ValueError, match="past"):
        R.check_final("clip", _final(("Resting", 0.0, 11.0)), 10.0)


def test_an_empty_final_set_is_rejected():
    with pytest.raises(ValueError):
        R.check_final("clip", [], 10.0)


def test_a_missing_condition_is_rejected():
    interval = _final(("Resting", 0.0, 5.0))[0]
    interval["visibility"] = ""
    with pytest.raises(ValueError, match="visibility"):
        R.check_final("clip", [interval], 10.0)


def test_uncertain_without_a_reason_is_rejected():
    interval = _final(("Uncertain", 0.0, 5.0))[0]
    interval["uncertain_reason"] = ""
    with pytest.raises(ValueError, match="reason"):
        R.check_final("clip", [interval], 10.0)


def test_resolved_labels_lists_labels_with_their_time_ranges():
    rows = R.check_final("clip", _final(("Resting", 5.0, 9.5), ("Locomotion", 0.0, 5.0)), 10.0)
    assert R.resolved_labels(rows) == "Locomotion 0.00-5.00; Resting 5.00-9.50"


def test_two_different_reviewer_names_are_required():
    with pytest.raises(ValueError):
        R.reviewers_of({"reviewers": ["Ada", ""]})
    with pytest.raises(ValueError):
        R.reviewers_of({"reviewers": ["Ada", "ada"]})
    assert R.reviewers_of({"reviewers": ["Ada", "Bo"]}) == ["Ada", "Bo"]


def test_decisions_round_trip_keeps_durations(tmp_path):
    path = tmp_path / "d.json"
    R.save_decisions(path, {"reviewers": ["A", "B"], "clips": {"c": {"intervals": [], "rows": [], "reason": "x"}}, "durations": {"c": 9.5}})
    loaded = R.load_decisions(path)
    assert loaded["durations"] == {"c": 9.5} and loaded["reviewers"] == ["A", "B"]
    assert R.load_decisions(tmp_path / "missing.json") == {"reviewers": ["", ""], "clips": {}, "durations": {}}


def _setup(tmp_path):
    a = tmp_path / "annotations.csv"
    b = tmp_path / "second.csv"
    ledger = tmp_path / "agreement.csv"
    gold = tmp_path / "gold.txt"
    store = tmp_path / "decisions.json"
    gold.write_text("g1\ng2\n")
    _write(a, [
        _row("n1", 0, 10, "Resting", "Gun"),
        _row("g1", 0, 5, "Locomotion", "Thip"), _row("g1", 5, 10, "Resting", "Thip"),
        _row("g2", 0, 8, "Eating", "Thip"),
    ])
    _write(b, [_row("g1", 0, 4, "Locomotion", "Gun"), _row("g1", 4, 10, "Resting", "Gun"), _row("g2", 0, 8, "Eating", "Gun")])
    ledger.write_text(",".join(["clip_id", "annotator_a", "annotator_b", "intervals_compared", "matching_labels", "matching_boundaries",
                                "disagreements", "resolved_labels", "resolution_reason", "reviewed_by", "reviewed_at"]) + "\n")
    return a, b, ledger, gold, store


def _decide(store, clips=("g1", "g2")):
    decisions = {"reviewers": ["Ada", "Bo"], "clips": {}, "durations": {}}
    for clip in clips:
        rows = R.check_final(clip, _final(("Locomotion", 0.0, 4.5), ("Resting", 4.5, 10.0)) if clip == "g1" else _final(("Eating", 0.0, 8.0)), 10.0)
        decisions["clips"][clip] = {"intervals": _final(("Locomotion", 0.0, 4.5), ("Resting", 4.5, 10.0)) if clip == "g1" else _final(("Eating", 0.0, 8.0)),
                                    "rows": rows, "reason": "boundary at 4.5 after watching" if clip == "g1" else "both passes agree"}
        decisions["durations"][clip] = 10.0
    R.save_decisions(store, decisions)


def test_apply_writes_the_ledger_and_swaps_only_the_gold_rows(tmp_path):
    a, b, ledger, gold, store = _setup(tmp_path)
    _decide(store)
    result = R.apply(a, b, ledger, store, gold, "2026-10-20T20:00:00+07:00")
    assert {row["clip_id"] for row in result} == {"g1", "g2"}
    rows = review.read_rows(a)
    assert [r["annotator"] for r in rows if r["clip_id"] == "n1"] == ["Gun"]
    assert {r["annotator"] for r in rows if r["clip_id"] in ("g1", "g2")} == {R.RECONCILED}
    assert [(r["start_s"], r["end_s"]) for r in rows if r["clip_id"] == "g1"] == [("0.00", "4.50"), ("4.50", "10.00")]
    written = {r["clip_id"]: r for r in agreement.review_ledger(ledger)}
    assert written["g1"]["resolved_labels"] == "Locomotion 0.00-4.50; Resting 4.50-10.00"
    assert written["g1"]["reviewed_by"] == "Ada; Bo" and written["g1"]["reviewed_at"].startswith("2026-10-20")
    assert written["g1"]["matching_labels"] == "2"


def test_the_measured_columns_come_from_the_original_passes(tmp_path):
    a, b, ledger, gold, store = _setup(tmp_path)
    _decide(store)
    result = {row["clip_id"]: row for row in R.apply(a, b, ledger, store, gold, "t")}
    assert result["g1"]["intervals_compared"] == 4 and result["g1"]["matching_labels"] == 2
    assert result["g1"]["matching_boundaries"] == 0  # the passes differ by 1 s at the boundary, past the 0.5 s tolerance
    assert result["g2"]["disagreements"] == 0


def test_apply_refuses_until_every_gold_clip_is_decided(tmp_path):
    a, b, ledger, gold, store = _setup(tmp_path)
    _decide(store, clips=("g1",))
    with pytest.raises(ValueError, match="g2"):
        R.apply(a, b, ledger, store, gold, "t")
    assert all(r["annotator"] != R.RECONCILED for r in review.read_rows(a))


def test_apply_refuses_a_clip_with_no_reason(tmp_path):
    a, b, ledger, gold, store = _setup(tmp_path)
    _decide(store)
    decisions = R.load_decisions(store)
    decisions["clips"]["g2"]["reason"] = " "
    R.save_decisions(store, decisions)
    with pytest.raises(ValueError, match="reason"):
        R.apply(a, b, ledger, store, gold, "t")


def test_apply_refuses_without_two_reviewers(tmp_path):
    a, b, ledger, gold, store = _setup(tmp_path)
    _decide(store)
    decisions = R.load_decisions(store)
    decisions["reviewers"] = ["Ada", "Ada"]
    R.save_decisions(store, decisions)
    with pytest.raises(ValueError, match="reviewer"):
        R.apply(a, b, ledger, store, gold, "t")


def test_a_second_apply_does_not_overwrite_the_first(tmp_path):
    a, b, ledger, gold, store = _setup(tmp_path)
    _decide(store)
    R.apply(a, b, ledger, store, gold, "t")
    before = Path(ledger).read_text()
    with pytest.raises(ValueError, match="already"):
        R.apply(a, b, ledger, store, gold, "t2")
    assert Path(ledger).read_text() == before


def test_non_gold_rows_are_kept_exactly(tmp_path):
    a, b, ledger, gold, store = _setup(tmp_path)
    _decide(store)
    kept = [r for r in review.read_rows(a) if r["clip_id"] == "n1"]
    R.apply(a, b, ledger, store, gold, "t")
    assert [r for r in review.read_rows(a) if r["clip_id"] == "n1"] == kept
