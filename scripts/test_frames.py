"""The motion windows must carry the differences of the frames, not more frames."""

import numpy as np

import frames


class FakeCapture:
    """A capture that returns the given frames once, one read per sample."""

    def __init__(self, images):
        self.images = list(images)
        self.index = 0

    def set(self, prop, value):
        return True

    def read(self):
        image = self.images[min(self.index, len(self.images) - 1)]
        self.index += 1
        return True, image

    def grab(self):
        return True


def still_window():
    return [np.full((4, 4, 3), value, dtype=np.uint8) for value in range(0, 80, 10)]


def test_motion_window_is_frames_then_differences():
    out = frames.window_frames(FakeCapture(still_window()), 1.0, 0.0, 5.0, wh=(4, 4), motion=True)
    assert out.shape == (16, 4, 4, 3)
    assert out[:8].sum() > 0


def test_first_difference_frame_is_zero():
    out = frames.window_frames(FakeCapture(still_window()), 1.0, 0.0, 5.0, wh=(4, 4), motion=True)
    assert not out[8].any()


def test_difference_frame_is_the_step_between_samples():
    out = frames.window_frames(FakeCapture(still_window()), 1.0, 0.0, 5.0, wh=(4, 4), motion=True)
    assert np.all(out[9] == 10)


def test_without_motion_the_window_stays_eight_frames():
    out = frames.window_frames(FakeCapture(still_window()), 1.0, 0.0, 5.0, wh=(4, 4))
    assert out.shape == (8, 4, 4, 3)
