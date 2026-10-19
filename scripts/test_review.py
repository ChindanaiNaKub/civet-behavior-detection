"""Self-check for verdict mapping, blind stripping and write safety in review.py.

Pure standard library, no HTTP server, no video and no GPU: run with
`python -m pytest scripts`.
"""

import json
import io
from pathlib import Path
from types import SimpleNamespace

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


def _clip(clip_id):
    return R.Clip(clip_id, Path(f"{clip_id}.mp4"), 10.0)


def test_id_set_is_what_blind_ids_uses(tmp_path):
    id_file = tmp_path / "ids.txt"
    id_file.write_text("clip-b\n clip-c \n")
    assert R.id_set(["clip-a"], id_file) == {"clip-a", "clip-b", "clip-c"}
    assert R.id_set([], None) == set()


def test_restrict_clips_keeps_ledger_order_and_an_empty_list_keeps_everything():
    clips = [_clip("a"), _clip("b"), _clip("c")]
    assert [c.clip_id for c in R.restrict_clips(clips, {"c", "a"})] == ["a", "c"]
    assert R.restrict_clips(clips, set()) == clips


def test_a_redo_clip_counts_as_unverified_until_saved_in_this_run():
    counts = {"a": 3, "b": 2}
    assert R.pending_counts(counts, {"a"}, set()) == {"a": 0, "b": 2}
    assert R.pending_counts(counts, {"a"}, {"a"}) == {"a": 3, "b": 2}
    assert R.pending_counts(counts, set(), set()) == counts


def test_atomic_save_preserves_original_if_replacement_fails(tmp_path, monkeypatch):
    out = tmp_path / "second.csv"
    R.replace_rows(out, "clip-a", "bob", R.annotation_rows("clip-a", "bob", [_interval()]))
    original = out.read_bytes()

    def fail(*args):
        raise PermissionError("file is open elsewhere")

    monkeypatch.setattr(R.os, "replace", fail)
    with pytest.raises(PermissionError):
        R.replace_rows(out, "clip-a", "bob", R.annotation_rows("clip-a", "bob", [_interval(behavior="Resting")]))
    assert out.read_bytes() == original
    assert out.with_suffix(".csv.bak").read_bytes() == original
    assert {p.name for p in tmp_path.iterdir()} == {"second.csv", "second.csv.bak"}


def test_saving_second_pass_does_not_touch_first_pass_and_keeps_backup(tmp_path):
    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    R.replace_rows(first, "clip-a", "friend", R.annotation_rows("clip-a", "friend", [_interval()]))
    original = first.read_bytes()
    R.check_output(second, [first])
    R.replace_rows(second, "clip-a", "bob", R.annotation_rows("clip-a", "bob", [_interval()]))
    before_second = second.read_bytes()
    R.replace_rows(second, "clip-a", "bob", R.annotation_rows("clip-a", "bob", [_interval(behavior="Resting")]))
    assert first.read_bytes() == original
    assert second.with_suffix(".csv.bak").read_bytes() == before_second
    with pytest.raises(ValueError, match="protected"):
        R.check_output(first.parent / "." / first.name, [first])


@pytest.mark.parametrize("start,end", [(-1, 2), (0, float("nan")), (0, float("inf"))])
def test_invalid_timestamps_are_rejected(start, end):
    with pytest.raises(ValueError):
        R.annotation_rows("clip-a", "bob", [_interval(start_s=start, end_s=end)])


def test_thai_reason_roundtrips_and_old_reason_is_cleared(tmp_path):
    out = tmp_path / "second.csv"
    rows = R.annotation_rows("clip-a", "bob", [_interval(behavior="Uncertain", uncertain_reason="มองไม่เห็นตัวสัตว์")])
    R.replace_rows(out, "clip-a", "bob", rows)
    assert R.read_rows(out)[0]["uncertain_reason"] == "มองไม่เห็นตัวสัตว์"
    assert R.annotation_rows("clip-a", "bob", [_interval(uncertain_reason="to label")])[0]["uncertain_reason"] == ""


@pytest.mark.parametrize("intervals", [[], [_interval(end_s=11)], [_interval(), _interval(start_s=2, end_s=4)]])
def test_invalid_http_submission_leaves_both_files_untouched(tmp_path, intervals):
    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    first.write_bytes(b"friend's original records")
    second.write_bytes(b"second review must survive rejected request")
    handler = object.__new__(R.Handler)
    handler.server = SimpleNamespace(config=R.Config(
        clips=[R.Clip("clip-a", Path("video.mp4"), 10)], proposals={},
        out_file=second, protected=[first],
    ))
    handler.path = "/api/verdict"
    payload = json.dumps({"clip_id": "clip-a", "annotator": "bob", "intervals": intervals}).encode()
    handler.rfile = io.BytesIO(payload)
    handler.headers = {"Content-Length": str(len(payload))}
    responses = []
    handler._json = lambda data, status=200: responses.append((status, data))
    handler.do_POST()
    assert responses[0][0] == 400
    assert first.read_bytes() == b"friend's original records"
    assert second.read_bytes() == b"second review must survive rejected request"


@pytest.mark.parametrize("failure", [ConnectionResetError, ConnectionAbortedError, BrokenPipeError])
def test_cancelled_video_stream_returns_without_traceback(tmp_path, failure):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video stream")
    handler = object.__new__(R.Handler)
    handler.headers = {}
    handler.send_response = lambda *args: None
    handler.send_header = lambda *args: None
    handler.end_headers = lambda: None

    def disconnect(*args):
        raise failure("browser cancelled")

    handler.wfile = SimpleNamespace(write=disconnect)
    handler._serve_video(video)


def test_http_save_uses_server_output_not_client_supplied_path(tmp_path):
    first, second = tmp_path / "friend.csv", tmp_path / "second.csv"
    first.write_bytes(b"friend's unchanged original records")
    handler = object.__new__(R.Handler)
    handler.server = SimpleNamespace(config=R.Config(
        clips=[R.Clip("clip-a", Path("video.mp4"), 10)], proposals={},
        out_file=second, protected=[first],
    ))
    handler.path = "/api/verdict"
    payload = json.dumps({"clip_id": "clip-a", "annotator": "bob", "out": str(first), "intervals": [_interval()]}).encode()
    handler.rfile = io.BytesIO(payload)
    handler.headers = {"Content-Length": str(len(payload))}
    responses = []
    handler._json = lambda data, status=200: responses.append((status, data))
    handler.do_POST()
    assert responses[0][0] == 200
    assert R.read_rows(second)[0]["annotator"] == "bob"
    assert first.read_bytes() == b"friend's unchanged original records"
