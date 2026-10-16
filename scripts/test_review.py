"""Self-check for verdict mapping, blind stripping and write safety in review.py.

Pure standard library, no HTTP server, no video and no GPU: run with
`python -m pytest scripts`.
"""

import json
from pathlib import Path

import pytest

import review as R


def _interval(**overrides):
    interval = {"start_s": 1.0, "end_s": 2.5, "behavior": "Eating"}
    interval.update(overrides)
    return interval


def test_verdict_maps_to_annotation_row_in_header_order():
    rows = R.annotation_rows("clip-1", "ada", [_interval()])
    assert rows == [
        {
            "clip_id": "clip-1",
            "start_s": "1.00",
            "end_s": "2.50",
            "behavior": "Eating",
            "visibility": "unknown",
            "lighting": "unknown",
            "camera_view": "unknown",
            "occlusion": "unknown",
            "uncertain_reason": "",
            "annotator": "ada",
            "notes": "",
        }
    ]
    assert list(rows[0]) == R.CSV_FIELDS


def test_conditions_are_kept_when_the_reviewer_sets_them():
    rows = R.annotation_rows(
        "clip-1", "ada", [_interval(visibility="partial", lighting="infrared", notes="fence edge")]
    )
    assert rows[0]["visibility"] == "partial"
    assert rows[0]["lighting"] == "infrared"
    assert rows[0]["notes"] == "fence edge"


def test_uncertain_without_reason_is_rejected():
    with pytest.raises(ValueError):
        R.annotation_rows("clip-1", "ada", [_interval(behavior="Uncertain")])


def test_uncertain_with_reason_is_accepted():
    rows = R.annotation_rows("clip-1", "ada", [_interval(behavior="Uncertain", uncertain_reason="too dark")])
    assert rows[0]["behavior"] == "Uncertain"
    assert rows[0]["uncertain_reason"] == "too dark"


def test_transition_is_a_valid_interval_label():
    rows = R.annotation_rows("clip-1", "ada", [_interval(behavior="Transition")])
    assert rows[0]["behavior"] == "Transition"


def test_invented_behavior_is_rejected():
    with pytest.raises(ValueError):
        R.annotation_rows("clip-1", "ada", [_interval(behavior="Sleeping")])


def test_end_must_exceed_start():
    with pytest.raises(ValueError):
        R.annotation_rows("clip-1", "ada", [_interval(start_s=2.0, end_s=2.0)])


def test_annotator_is_required():
    with pytest.raises(ValueError):
        R.annotation_rows("clip-1", "  ", [_interval()])


def test_blind_payload_strips_every_proposal_trace():
    clip = R.Clip("clip-1", Path("clip-1.mp4"), 10.0)
    proposals = [{"start_s": 3.0, "end_s": 7.0, "behavior": "Eating", "confidence": "0.90"}]

    blind = R.clip_payload(clip, proposals, True)
    assert "proposals" not in blind
    text = json.dumps(blind)
    assert "Eating" not in text
    assert "0.90" not in text
    assert "3.0" not in text

    reviewed = R.clip_payload(clip, proposals, False)
    assert reviewed["proposals"] == proposals


def test_replace_writes_the_header_once_and_keeps_other_clips(tmp_path):
    out = tmp_path / "annotations.csv"
    R.replace_rows(out, "clip-a", "ada", R.annotation_rows("clip-a", "ada", [_interval(behavior="Resting")]))
    R.replace_rows(out, "clip-b", "ada", R.annotation_rows("clip-b", "ada", [_interval(behavior="Locomotion")]))

    assert out.read_text().splitlines().count(",".join(R.CSV_FIELDS)) == 1
    assert [(row["clip_id"], row["behavior"]) for row in R.read_rows(out)] == [
        ("clip-a", "Resting"),
        ("clip-b", "Locomotion"),
    ]


def test_resaving_a_clip_replaces_its_rows_instead_of_duplicating_them(tmp_path):
    out = tmp_path / "annotations.csv"
    R.replace_rows(out, "clip-a", "ada", R.annotation_rows("clip-a", "ada", [_interval(), _interval(start_s=3.0, end_s=4.0)]))
    R.replace_rows(out, "clip-a", "ada", R.annotation_rows("clip-a", "ada", [_interval(behavior="Resting")]))

    rows = R.read_rows(out)
    assert len(rows) == 1
    assert rows[0]["behavior"] == "Resting"


def test_a_second_annotators_rows_for_the_same_clip_survive(tmp_path):
    out = tmp_path / "annotations.csv"
    R.replace_rows(out, "clip-a", "ada", R.annotation_rows("clip-a", "ada", [_interval(behavior="Eating")]))
    R.replace_rows(out, "clip-a", "bob", R.annotation_rows("clip-a", "bob", [_interval(behavior="Resting")]))

    assert sorted((row["annotator"], row["behavior"]) for row in R.read_rows(out)) == [("ada", "Eating"), ("bob", "Resting")]


def test_writes_use_lf_like_the_tracked_annotations_file(tmp_path):
    out = tmp_path / "annotations.csv"
    R.replace_rows(out, "clip-a", "ada", R.annotation_rows("clip-a", "ada", [_interval()]))
    assert b"\r\n" not in out.read_bytes()


def test_read_rows_rejects_a_file_with_a_different_header(tmp_path):
    out = tmp_path / "annotations.csv"
    out.write_text("clip,start\nx,1\n")
    with pytest.raises(ValueError, match="header"):
        R.read_rows(out)


def test_verified_counts_groups_by_clip(tmp_path):
    out = tmp_path / "annotations.csv"
    R.replace_rows(out, "clip-a", "ada", R.annotation_rows("clip-a", "ada", [_interval(), _interval(start_s=3.0, end_s=4.0)]))
    R.replace_rows(out, "clip-b", "ada", R.annotation_rows("clip-b", "ada", [_interval()]))
    assert R.verified_counts(out) == {"clip-a": 2, "clip-b": 1}


def test_verified_counts_of_missing_file_is_empty(tmp_path):
    assert R.verified_counts(tmp_path / "nope.csv") == {}


def test_blind_ids_merges_flag_and_file(tmp_path):
    blind_file = tmp_path / "blind.txt"
    blind_file.write_text("clip-b\n\n clip-c \n")
    assert R.blind_ids(["clip-a", "clip-b"], blind_file) == {"clip-a", "clip-b", "clip-c"}
