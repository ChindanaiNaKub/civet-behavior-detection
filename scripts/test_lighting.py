"""Self-check for the capture-condition rules in lighting.py.

No video file is read: frames are built in memory. Run with `python -m pytest scripts`.
"""

import numpy as np
import pytest

import lighting as L


def _row(clip_id="c", reason="", **conditions):
    row = {"clip_id": clip_id, "uncertain_reason": reason, "visibility": "unknown", "lighting": "unknown", "occlusion": "unknown"}
    row.update(conditions)
    return row


def test_a_gray_frame_has_no_chroma_and_a_colour_frame_has_some():
    gray = np.full((180, 320, 3), 120, dtype=np.uint8)
    colour = gray.copy()
    colour[..., 2] = 200
    assert L.chroma(gray) == 0.0
    assert L.chroma(colour) > L.GRAYSCALE_MAX_CHROMA


def test_lighting_follows_the_mean_colour_of_the_samples():
    assert L.classify_lighting([0.0, 0.0, 0.0]) == L.INFRARED
    assert L.classify_lighting([7.0, 8.5, 6.2]) == L.DAYLIGHT


def test_no_sampled_frame_is_an_error_not_a_guess():
    with pytest.raises(ValueError):
        L.classify_lighting([])


def test_only_what_the_annotator_wrote_is_read_from_the_reason():
    assert L.conditions_from_reason("can't see civet because the tree is intercept") == {
        "visibility": "out of view",
        "occlusion": "tree/plant",
    }
    assert L.conditions_from_reason("animal hidden behind plant") == {"occlusion": "tree/plant"}
    assert L.conditions_from_reason("machine proposal") == {}
    assert L.conditions_from_reason("") == {}


def test_fill_sets_unknown_fields_and_keeps_a_value_set_by_hand():
    rows = [_row("a", "can't see civet"), _row("b", lighting="hand set")]
    changed = L.fill(rows, {"a": L.INFRARED, "b": L.DAYLIGHT})
    assert rows[0]["lighting"] == L.INFRARED and rows[0]["visibility"] == "out of view"
    assert rows[1]["lighting"] == "hand set"
    assert changed == 2


def test_fill_leaves_a_clip_with_no_measurement_unknown():
    rows = [_row("z")]
    assert L.fill(rows, {}) == 0
    assert rows[0]["lighting"] == "unknown"
