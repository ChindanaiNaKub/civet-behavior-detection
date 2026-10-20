"""Self-check for the two-annotator comparison. Pure data: `python -m pytest scripts`."""

import agreement as A


def row(clip_id, start, end, behavior, annotator="a"):
    return {"clip_id": clip_id, "start_s": start, "end_s": end, "behavior": behavior, "annotator": annotator}


def test_identical_sets_agree_on_every_interval():
    rows, orphans = A.compare(
        [row("c", 0, 5, "Eating"), row("c", 5, 10, "Resting")],
        [row("c", 0, 5, "Eating", "b"), row("c", 5, 10, "Resting", "b")],
    )
    assert orphans == []
    assert rows[0]["matching_labels"] == 2
    assert rows[0]["matching_boundaries"] == 2
    assert rows[0]["disagreements"] == 0
    assert rows[0]["annotator_a"] == "a"
    assert rows[0]["annotator_b"] == "b"


def test_shifted_boundaries_inside_the_tolerance_still_count():
    rows, _ = A.compare([row("c", 0, 5, "Eating")], [row("c", 0.2, 4.9, "Eating", "b")])
    assert rows[0]["matching_labels"] == 1
    assert rows[0]["matching_boundaries"] == 1


def test_boundaries_beyond_the_tolerance_keep_the_label_match_only():
    rows, _ = A.compare([row("c", 0, 5, "Eating")], [row("c", 2.0, 5.0, "Eating", "b")])
    assert rows[0]["matching_labels"] == 1
    assert rows[0]["matching_boundaries"] == 0


def test_different_behaviors_are_two_disagreements():
    rows, _ = A.compare([row("c", 0, 5, "Eating")], [row("c", 0, 5, "Locomotion", "b")])
    assert rows[0]["matching_labels"] == 0
    assert rows[0]["disagreements"] == 2


def test_two_intervals_cannot_both_match_one():
    rows, _ = A.compare(
        [row("c", 0, 3, "Eating"), row("c", 3, 6, "Eating")],
        [row("c", 0, 6, "Eating", "b")],
    )
    assert rows[0]["matching_labels"] == 1
    assert rows[0]["disagreements"] == 1


def test_a_clip_labelled_in_only_one_file_is_reported_not_compared():
    rows, orphans = A.compare([row("c1", 0, 5, "Eating")], [row("c2", 0, 5, "Eating", "b")])
    assert rows == []
    assert orphans == ["c1 only in the first file", "c2 only in the second file"]


def test_the_ledger_round_trips_dashes_and_thai_text(tmp_path):
    path = tmp_path / "agreement.csv"
    row = {name: "" for name in ("clip_id", "annotator_a", "annotator_b", "intervals_compared", "matching_labels",
                                 "matching_boundaries", "disagreements", "resolved_labels", "resolution_reason",
                                 "reviewed_by", "reviewed_at")}
    row.update(clip_id="c1", resolved_labels="Eating 5.53\u201342.10", resolution_reason="ตัดสินตามคู่มือ \u2014 both agree")
    A.write_ledger(path, [row])
    loaded = A.review_ledger(path)
    assert loaded[0]["resolved_labels"] == "Eating 5.53\u201342.10"
    assert loaded[0]["resolution_reason"] == "ตัดสินตามคู่มือ \u2014 both agree"
    assert not path.with_suffix(".csv.tmp").exists()
