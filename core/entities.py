"""
The "things" our pipeline detects, kept separate from how we detect them.

Person is the base unit (from pose). PpeStatus is the per-person safety verdict,
and PersonResult ties them together with the ROIs the pipeline used.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Every PPE verdict is one of these three. "unknown" is deliberately distinct
# from "no": it means we couldn't judge (head turned away, face not located),
# NOT that the equipment is absent - conflating them causes false alarms.
YES, NO, UNKNOWN = "yes", "no", "unknown"


@dataclass
class Person:
    """Everything we know about one person in a frame."""

    person_id: int
    box: tuple[int, int, int, int]          # the full-body box: x1, y1, x2, y2
    keypoints: np.ndarray                    # raw (17, 3) array from the model: x, y, confidence
    head_box: tuple[int, int, int, int]
    head_keypoints_found: int                # how many head keypoints we actually trusted
    used_fallback_box: bool                  # True if we had to guess the head box instead of measuring it


@dataclass
class PpeStatus:
    """Per-person PPE verdict. Each field is YES / NO / UNKNOWN."""

    helmet: str = UNKNOWN
    goggle: str = UNKNOWN
    helmet_iou: float = 0.0                   # IoU of the matched helmet box with the head ROI
    goggle_iou: float = 0.0                   # IoU of the matched goggle box with the face ROI


@dataclass
class PersonResult:
    """One person after the full pipeline: pose + ROIs + PPE verdict."""

    person: Person
    face_box: tuple[int, int, int, int]       # tight face ROI (goggle region); also what we anonymize
    ppe: PpeStatus = field(default_factory=PpeStatus)
