"""Decode the frames of a five-second window, for the frame-CNN classifier.

Windows of one clip are decoded in a single sequential pass, so the cost is one
seek per window instead of one per frame. Frames are kept as uint8 and scaled to
the model input size here, which keeps a whole dataset of windows in a few tens
of megabytes while training.

The classifier sees eight evenly spaced frames from the window rather than all
25 frames per second: with 22 source clips the risk is overfitting, not temporal
resolution, and the even spacing still shows a change of posture or position.
"""

from __future__ import annotations

import numpy as np

# Frames kept per window, and the (width, height) the classifier sees.
SAMPLES = 8
WH = (128, 128)

# torchvision's ImageNet statistics, because the backbone is pretrained on it.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def window_frames(cap, fps: float, start: float, end: float, samples: int = SAMPLES, wh=WH) -> np.ndarray:
    """``samples`` evenly spaced frames of [start, end) as (samples, h, w, 3) uint8.

    ``cap`` is an open ``cv2.VideoCapture`` positioned at or before ``start``.
    A frame that fails to decode repeats the previous one, so a short or damaged
    read still yields a full window rather than a differently shaped array.
    """
    import cv2

    span = (end - start) / samples
    step = max(1, int(round(span * fps)))
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
    frames = []
    last = None
    for _ in range(samples):
        ok, frame = cap.read()
        if ok:
            last = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if last is None:
            raise RuntimeError("no frame decoded at the start of the window")
        frames.append(cv2.resize(last, wh, interpolation=cv2.INTER_AREA))
        for _ in range(step - 1):
            if not cap.grab():
                break
    return np.stack(frames)


def clip_windows(path, spans, samples: int = SAMPLES, wh=WH) -> list[np.ndarray]:
    """Frames for every (start, end) span of one clip, in the order given."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    try:
        return [window_frames(cap, fps, start, end, samples, wh) for start, end in spans]
    finally:
        cap.release()


def to_tensor(batch: np.ndarray):
    """(n, t, h, w, 3) uint8 to a normalised (n, t, 3, h, w) float tensor."""
    import torch

    array = batch.astype(np.float32) / 255.0
    array = (array - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(array.transpose(0, 1, 4, 2, 3)))


def decode_windows(windows, clips_root, relative, samples: int = SAMPLES, wh=WH) -> np.ndarray:
    """Frames for every window, in the order given, one decode pass per clip.

    ``relative`` maps a clip id to its path under ``clips_root`` (the ledger's
    ``relative_path`` column), so a caller never rebuilds that layout by hand.
    """
    by_clip: dict[str, list[int]] = {}
    for i, win in enumerate(windows):
        by_clip.setdefault(win.clip_id, []).append(i)

    decoded = np.zeros((len(windows), samples, wh[1], wh[0], 3), dtype=np.uint8)
    for clip_id, indices in by_clip.items():
        spans = [(windows[i].start, windows[i].end) for i in indices]
        for i, window in zip(indices, clip_windows(clips_root / relative[clip_id], spans, samples, wh)):
            decoded[i] = window
    return decoded
