"""Self-check for the Task 3 silhouette geometry, with synthetic masks only.

Run with `python -m pytest scripts/test_features.py -q`.
"""

import numpy as np
import pytest

import features as F


def rectangle(height: int, width: int) -> np.ndarray:
    """A filled rectangle, rows downward."""
    mask = np.zeros((height + 4, width + 4), bool)
    mask[2 : 2 + height, 2 : 2 + width] = True
    return mask


def axial(angle: float) -> float:
    """Distance from an axis angle to horizontal, treating 0 and 180 as the same axis."""
    return min(angle, 180.0 - angle)


def test_horizontal_silhouette_has_body_angle_near_zero():
    angle = F.body_angle_deg(rectangle(20, 120))
    assert axial(angle) < 5.0, angle


def test_vertical_silhouette_has_body_angle_near_ninety():
    angle = F.body_angle_deg(rectangle(120, 20))
    assert abs(angle - 90.0) < 5.0, angle


def test_ten_pixels_per_sample_at_two_samples_per_second_is_twenty_px_s():
    assert F.speed_px_s((50.0, 60.0), (60.0, 60.0), 0.5) == pytest.approx(20.0)


def test_limb_extent_of_a_rectangle_is_its_extent_across_the_body_axis():
    # 120 px long, 20 px tall: the body axis is horizontal, so the extent across it
    # is the height (19 coordinate steps between the first and last row).
    assert F.limb_extent_px(rectangle(20, 120)) == pytest.approx(20.0, abs=1.0)


def test_head_angle_follows_the_narrow_end():
    # Wide at the left, tapering to a narrow snout at the right: the right half is
    # the narrow end, so the head axis is still roughly horizontal.
    mask = np.zeros((60, 120), bool)
    for x in range(120):
        half = max(1, int(28 - (24 * x / 119)))
        mask[30 - half : 30 + half, x] = True
    angle = F.head_angle_deg(mask)
    assert axial(angle) < 10.0, angle


def test_a_mask_too_small_for_an_axis_is_not_a_number():
    dot = np.zeros((10, 10), bool)
    dot[5, 5] = True
    assert np.isnan(F.body_angle_deg(dot))
    assert np.isnan(F.limb_extent_px(dot))
