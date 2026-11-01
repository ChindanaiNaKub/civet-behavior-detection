"""Self-check for the activity scan, with synthetic frames and rows only.

Run with `python -m pytest scripts`.
"""

import numpy as np

import activity as A

HEIGHT, WIDTH = 270, 480


def _row(time_s, box, speed=float("nan"), churn=float("nan")):
    return A.Row(time_s, box is not None, box, 0 if box is None else 1000, speed, churn)


def _square(x, y, side=40):
    return (x, y, x + side, y + side)


def test_a_moving_square_is_present_and_active():
    # Thirty frames sweeping the whole width keeps the median background empty:
    # a synthetic square that stays put would be absorbed into it, which is the
    # limitation a long recording exists to avoid.
    frames = np.full((30, HEIGHT, WIDTH), 40, np.uint8)
    for i in range(30):
        x = 20 + 10 * i
        frames[i, 100:140, x : x + 40] = 220

    background, threshold = A.threshold_for(frames)
    rows, unmeasured = A.trace(frames, background, threshold)

    assert len(rows) == 30
    assert all(row.present for row in rows)
    assert A.label_window(rows[5:10]) == "Active"


def test_a_still_square_is_present_and_inactive():
    frames = np.full((12, HEIGHT, WIDTH), 40, np.uint8)
    frames[:, 100:140, 100:140] = 220

    # A still object is not in the median only if it is absent from most frames,
    # so the background is supplied directly here.
    background = np.full((HEIGHT, WIDTH), 40, np.float32)
    rows, unmeasured = A.trace(frames, background, np.float32(A.ABS_MIN))

    assert all(row.present for row in rows)
    assert A.label_window(rows[:5]) == "Inactive"


def test_a_single_frame_blob_is_dropped_as_an_insect():
    frames = np.full((12, HEIGHT, WIDTH), 40, np.uint8)
    frames[5, 100:140, 100:140] = 220

    background = np.full((HEIGHT, WIDTH), 40, np.float32)
    rows, unmeasured = A.trace(frames, background, np.float32(A.ABS_MIN))

    assert rows == []


def test_a_window_with_too_few_samples_is_absent():
    rows = [
        _row(0.0, _square(10, 10)),
        _row(1.0, None),
        _row(2.0, None),
        _row(3.0, _square(10, 10)),
        _row(4.0, None),
    ]
    assert A.label_window(rows) == "Absent"


def test_speed_alone_makes_a_window_active():
    rows = [_row(float(i), _square(10 + i, 10), speed=A.ACTIVE_SPEED + 0.01) for i in range(5)]
    assert A.label_window(rows) == "Active"


def test_churn_alone_makes_a_window_active():
    rows = [_row(float(i), _square(10, 10), speed=0.0, churn=A.ACTIVE_CHURN + 0.01) for i in range(5)]
    assert A.label_window(rows) == "Active"


def test_present_but_still_and_unmoving_is_inactive():
    rows = [_row(float(i), _square(10, 10), speed=0.0, churn=0.0) for i in range(5)]
    assert A.label_window(rows) == "Inactive"


def test_grid_drops_the_short_tail():
    assert A.grid(12.0) == [(0.0, 5.0), (5.0, 10.0)]
    assert A.grid(4.9) == []


def test_the_area_band_rejects_an_insect_and_a_wall():
    mask = np.zeros((HEIGHT, WIDTH), np.uint8)
    mask[50:56, 50:56] = 1                      # 36 px, a flying insect
    assert A.largest_box(mask)[0] is None

    mask[:] = 1                                 # the whole view
    assert A.largest_box(mask)[0] is None

    mask[:] = 0
    mask[100:140, 100:140] = 1                  # 1600 px, the animal
    box, area = A.largest_box(mask)
    assert box == (100, 100, 140, 140)
    assert area == 1600


def test_the_capture_time_comes_from_the_file_name():
    assert A.capture_datetime("A02_20260405193316") == "2026-04-05 19:33:16"
    assert A.capture_datetime("clip-without-a-stamp") == ""
