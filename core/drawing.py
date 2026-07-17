"""
Everything that turns a Person into pixels on screen or a line in the console.

Kept separate from the detection logic so we can change how results look
without touching how they're computed.
"""

from __future__ import annotations

import cv2
import numpy as np

from entities import Person
from geometry import HEAD_KEYPOINTS, MIN_KEYPOINT_CONFIDENCE

# Which keypoints connect to which, so we can draw a stick-figure skeleton.
# Also COCO's standard layout - nothing custom here.
SKELETON_BONES = (
    (0, 1), (0, 2), (1, 3), (2, 4),            # face
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),   # shoulders + arms
    (5, 11), (6, 12), (11, 12),                # torso
    (11, 13), (13, 15), (12, 14), (14, 16),    # legs
)


def draw_person(frame: np.ndarray, person: Person) -> None:
    """Draw everything for one person onto the frame: body box, skeleton, head keypoints, head box."""

    # Body box in green, with a small id label above it.
    x1, y1, x2, y2 = person.box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
    cv2.putText(frame, f"id {person.person_id}", (x1, max(y1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)

    # Skeleton lines, but only between points we actually trust - otherwise
    # you get lines flying off to (0, 0) for occluded joints.
    for start, end in SKELETON_BONES:
        x_start, y_start, conf_start = person.keypoints[start]
        x_end, y_end, conf_end = person.keypoints[end]
        if conf_start >= MIN_KEYPOINT_CONFIDENCE and conf_end >= MIN_KEYPOINT_CONFIDENCE:
            cv2.line(frame, (int(x_start), int(y_start)), (int(x_end), int(y_end)), (255, 160, 0), 2, cv2.LINE_AA)

    # Head keypoints as little red dots.
    for kpt_index in HEAD_KEYPOINTS:
        x, y, confidence = person.keypoints[kpt_index]
        if confidence >= MIN_KEYPOINT_CONFIDENCE:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1, cv2.LINE_AA)

    # Head box: bright cyan when it's a real measurement, a duller orange-ish
    # cyan when it's just our top-25%-of-the-body guess, so you can tell at a
    # glance which heads to trust.
    box_color = (0, 220, 220) if not person.used_fallback_box else (0, 140, 220)
    hx1, hy1, hx2, hy2 = person.head_box
    cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), box_color, 2)


def print_person_summary(person: Person) -> None:
    """Quick console line per person, mostly for debugging while we tune the head-box logic."""
    source = "fallback guess" if person.used_fallback_box else "from keypoints"
    print(f"  person {person.person_id:>2}: head_box={person.head_box}, "
          f"trusted head keypoints={person.head_keypoints_found} ({source})")
