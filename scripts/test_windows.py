"""Self-check for the window builder. Pure data, no video: `python -m pytest scripts`."""

import pytest

import windows as W


def iv(start, end, behavior, clip="clip"):
    return W.Interval(clip, start, end, behavior)


def test_grid_gives_full_windows_and_drops_a_short_tail():
    assert W.grid(11.5) == [(0.0, 5.0), (5.0, 10.0)]
    assert W.grid(10.0) == [(0.0, 5.0), (5.0, 10.0)]
    assert W.grid(12.0) == [(0.0, 5.0), (5.0, 10.0)]
    assert W.grid(4.0) == []


def test_window_fully_inside_one_interval_keeps_that_label():
    assert W.label_window(0.0, 5.0, [iv(0.0, 10.0, "Resting")]) == ("Resting", "")


def test_window_crossing_a_boundary_is_dropped():
    behavior, reason = W.label_window(3.0, 8.0, [iv(0.0, 5.0, "Eating"), iv(5.0, 10.0, "Locomotion")])
    assert behavior is None
    assert reason == "not_covered"


def test_window_over_a_transition_is_dropped():
    assert W.label_window(0.0, 5.0, [iv(0.0, 10.0, "Transition")]) == (None, "non_class_label")


def test_window_over_an_uncertain_span_is_dropped():
    assert W.label_window(0.0, 5.0, [iv(0.0, 10.0, "Uncertain")]) == (None, "non_class_label")


def test_climbing_is_not_a_model_class_yet():
    assert W.label_window(0.0, 5.0, [iv(0.0, 10.0, "Climbing")]) == (None, "non_class_label")


def test_overlapping_intervals_are_reported_not_guessed():
    assert W.label_window(0.0, 5.0, [iv(0.0, 10.0, "Eating"), iv(0.0, 10.0, "Resting")]) == (None, "conflict")


def test_windows_a_short_interval_cannot_cover_are_dropped():
    kept, dropped = W.windows_for_clip("clip", 10.0, "train", [iv(0.0, 4.0, "Eating")])
    assert kept == []
    assert dropped["not_covered"] == 2


def test_split_is_carried_onto_every_window():
    kept, _ = W.windows_for_clip("clip", 5.0, "test", [iv(0.0, 5.0, "Eating")])
    assert [w.split for w in kept] == ["test"]


def test_build_skips_unassigned_and_unlabelled_clips():
    intervals = [iv(0.0, 5.0, "Eating", clip="a"), iv(0.0, 5.0, "Eating", clip="b")]
    durations = {"a": 5.0, "b": 5.0, "c": 5.0}
    splits = {"a": "train", "b": "unassigned", "c": "test"}
    kept, dropped = W.build(intervals, durations, splits)
    assert [w.clip_id for w in kept] == ["a"]
    assert dropped["clip_unassigned"] == 1
    assert dropped["clip_unlabelled"] == 1


def test_read_intervals_rejects_an_unknown_label_and_a_bad_span(tmp_path):
    bad_label = tmp_path / "bad.csv"
    bad_label.write_text("clip_id,start_s,end_s,behavior\na,0,5,Flying\n")
    with pytest.raises(ValueError, match="unknown behavior"):
        W.read_intervals(bad_label)

    bad_span = tmp_path / "span.csv"
    bad_span.write_text("clip_id,start_s,end_s,behavior\na,5,5,Eating\n")
    with pytest.raises(ValueError, match="end_s"):
        W.read_intervals(bad_span)
