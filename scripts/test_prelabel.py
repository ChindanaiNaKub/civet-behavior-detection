"""Self-check for the motion stage and interval assembly in prelabel.py.

Pure numpy, no video and no model download: run with `python -m pytest scripts`.
"""

import numpy as np
import pytest

import prelabel as P


def _labels(values):
    labels = np.array(["other"] * len(values), dtype=object)
    for start, end, label in values:
        labels[start:end] = label
    return labels


def test_merge_short_runs_inherits_previous_label():
    labels = _labels([(5, 7, "Resting")])
    assert set(P.merge_short_runs(labels, 5)) == {"other"}


def test_merge_short_runs_keeps_long_runs():
    labels = _labels([(5, 15, "Resting")])
    merged = P.merge_short_runs(labels, 5)
    assert (merged[5:15] == "Resting").all()
    assert (merged[:5] == "other").all()


def test_resting_needs_five_seconds_of_low_motion():
    fps = P.SAMPLE_FPS
    quiet = np.zeros(int(10 * fps))
    disp = np.full(quiet.size, np.nan)
    labels = P.label_samples(quiet, disp, fps)
    assert (labels[: int(P.REST_MIN_S * fps) - 1] == "other").all()
    assert (labels[int(P.REST_MIN_S * fps) :] == "Resting").all()


def test_movement_while_awake_is_not_resting():
    fps = P.SAMPLE_FPS
    motion = np.full(int(10 * fps), P.REST_MAX_MOTION * 10)
    disp = np.full(motion.size, np.nan)
    assert (P.label_samples(motion, disp, fps) == "other").all()


def test_locomotion_comes_from_location_change():
    fps = P.SAMPLE_FPS
    motion = np.full(int(10 * fps), P.MOTION_MIN_FRACTION)
    disp = np.full(motion.size, np.nan)
    disp[int(3 * fps) :] = P.MOVE_MIN * 2
    labels = P.label_samples(motion, disp, fps)
    assert labels[-1] == "Locomotion"
    assert labels[0] == "other"


def test_moving_blob_gives_motion_and_displacement():
    height, width = 120, 160
    frames = np.zeros((int(5 * P.SAMPLE_FPS), height, width), np.uint8)
    for i in range(len(frames)):
        x = 3 * i
        frames[i, 50:70, x : x + 20] = 255
    motion, disp = P.motion_and_disp(frames, P.SAMPLE_FPS)
    assert motion[1:].min() > P.MOTION_MIN_FRACTION
    assert np.isnan(disp[0])
    # 3 px per sample over a MOVE_SPAN_S window, in frame widths.
    assert np.nanmax(disp) == pytest.approx(3 * P.SAMPLE_FPS * P.MOVE_SPAN_S / width, abs=0.02)


def test_static_scene_has_no_motion_or_location_change():
    frames = np.full((int(10 * P.SAMPLE_FPS), 120, 160), 90, np.uint8)
    motion, disp = P.motion_and_disp(frames, P.SAMPLE_FPS)
    assert motion.max() == 0
    assert np.isnan(disp).all()


def test_grid_windows_covers_clip_and_drops_short_tail():
    assert P.grid_windows(11.5) == [(0.0, 5.0), (5.0, 10.0)]
    assert P.grid_windows(12.0) == [(0.0, 5.0), (5.0, 10.0), (10.0, 12.0)]


def test_build_rows_proposes_and_abstains_on_low_probability():
    times = np.arange(int(10 * P.SAMPLE_FPS)) / P.SAMPLE_FPS
    labels = np.array(["other"] * len(times), dtype=object)
    motion = np.zeros(len(times))
    disp = np.full(len(times), np.nan)
    windows = [P.Window(0.0, 5.0, "Eating", 0.8), P.Window(5.0, 10.0, "Climbing", 0.2)]
    rows = P.build_rows("clip", times, labels, motion, disp, windows, 10.0)
    assert [(r.start, r.end, r.behavior, r.confidence) for r in rows] == [
        (0.0, 5.0, "Eating", 0.8),
        (5.0, 10.0, "Uncertain", 0.2),
    ]
    assert rows[1].zeroshot_top == "Climbing"


def test_build_rows_merges_neighbours_and_keeps_weakest_confidence():
    times = np.arange(int(10 * P.SAMPLE_FPS)) / P.SAMPLE_FPS
    labels = np.array(["other"] * len(times), dtype=object)
    windows = [P.Window(0.0, 5.0, "Eating", 0.8), P.Window(5.0, 10.0, "Eating", 0.6)]
    rows = P.build_rows("clip", times, labels, np.zeros(len(times)), np.full(len(times), np.nan), windows, 10.0)
    assert [(r.start, r.end, r.behavior, r.confidence) for r in rows] == [(0.0, 10.0, "Eating", 0.6)]
    assert rows[0].zeroshot_prob == 0.8


def test_motion_interval_carries_zero_shot_diagnostic():
    times = np.arange(int(10 * P.SAMPLE_FPS)) / P.SAMPLE_FPS
    labels = np.array(["Resting"] * len(times), dtype=object)
    windows = [P.Window(0.0, 5.0, "Locomotion", 0.9)]
    rows = P.build_rows(
        "clip", times, labels, np.zeros(len(times)), np.full(len(times), np.nan), windows, 10.0
    )
    assert len(rows) == 1
    assert rows[0].method == "motion"
    assert (rows[0].zeroshot_top, rows[0].zeroshot_prob) == ("Locomotion", 0.9)
