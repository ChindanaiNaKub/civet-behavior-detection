"""Self-check for the seed box detector, with synthetic frames only.

Run with `python -m pytest scripts`.
"""

import numpy as np

import detect as D

HEIGHT, WIDTH = 96, 128


def _persist(boxes):
    return D.persistence_filter([(float(i), box) for i, box in enumerate(boxes)])


def test_a_moving_square_yields_one_persistent_box():
    frames = np.full((20, HEIGHT, WIDTH), 40, np.uint8)
    side = 24
    for i in range(20):
        x, y = 10 + 3 * i, 36
        frames[i, y : y + side, x : x + side] = 220

    boxes = D.candidate_boxes(frames)
    assert len(boxes) == 20
    assert all(box is not None for box in boxes)

    kept = _persist(boxes)
    assert len(kept) == 20
    for i, box in kept:
        known = (10 + 3 * int(i), 36, 10 + 3 * int(i) + side, 36 + side)
        assert D.box_iou(box, known) > 0.5


def test_a_static_scene_yields_no_boxes():
    frames = np.full((10, HEIGHT, WIDTH), 60, np.uint8)
    frames[:, 20:70, 30:80] = 180
    frames[:, 10:14, 5:120] = 20

    assert D.candidate_boxes(frames) == [None] * 10
    assert _persist(D.candidate_boxes(frames)) == []


def test_random_blobs_at_new_positions_yield_no_boxes_after_persistence():
    # Six non-overlapping slots so the median background stays clean, then a
    # different slot each frame: every blob is a candidate, none is corroborated.
    slots = [(8, 8), (52, 8), (96, 8), (8, 60), (52, 60), (96, 60)]
    rng = np.random.default_rng(0)
    frames = np.full((12, HEIGHT, WIDTH), 40, np.uint8)
    side = 24
    previous = None
    for i in range(12):
        draw = int(rng.integers(len(slots) - 1))
        slot = draw if previous is None or draw < previous else draw + 1
        previous = slot
        x, y = slots[slot]
        frames[i, y : y + side, x : x + side] = 220

    boxes = D.candidate_boxes(frames)
    assert all(box is not None for box in boxes)
    assert _persist(boxes) == []
