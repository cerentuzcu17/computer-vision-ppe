"""
The "things" our pipeline detects, kept separate from how we detect them.

Right now there's just Person, since the pose model is the only thing running.
Once the helmet model is wired in, a Helmet entity (and maybe a combined
PersonWithHelmet result) belongs here too - this file is meant to grow with
the project instead of everything living inside pose_prototype.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Person:
    """Everything we know about one person in a frame."""

    person_id: int
    box: tuple[int, int, int, int]          # the full-body box: x1, y1, x2, y2
    keypoints: np.ndarray                    # raw (17, 3) array from the model: x, y, confidence
    head_box: tuple[int, int, int, int]
    head_keypoints_found: int                # how many head keypoints we actually trusted
    used_fallback_box: bool                  # True if we had to guess the head box instead of measuring it
