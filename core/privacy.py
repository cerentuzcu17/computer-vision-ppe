"""
Privacy: remove the face from a frame once PPE detection is done with it.

KVKK / privacy-by-design intent: the raw face is only ever in memory during PPE
inference; the moment we have the verdict, the face region is destroyed from the
frame that gets stored / shown / streamed. Pose keypoints are kept (they are not
identifying), so downstream logic still knows where each person is.

`mask` (solid fill) is the default because it is IRREVERSIBLE - unlike a blur,
which can sometimes be inverted or re-identified. Use `blur` only when a human
still needs to sanity-check the region and reversibility risk is acceptable.
"""
from __future__ import annotations

import cv2
import numpy as np


def anonymize_face(frame: np.ndarray, face_box: tuple[int, int, int, int], method: str = "mask") -> None:
    """Irreversibly remove the face ROI from `frame`, in place. Call AFTER PPE detection."""
    x1, y1, x2, y2 = face_box
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
    if x2 <= x1 or y2 <= y1:
        return

    if method == "mask":
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
    elif method == "blur":
        roi = frame[y1:y2, x1:x2]
        # Kernel scales with the region so small faces are still unrecognizable.
        k = max(11, (min(x2 - x1, y2 - y1) // 2) | 1)
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
    else:
        raise ValueError(f"unknown anonymize method: {method!r} (use 'mask' or 'blur')")
