"""
Pure geometry: figuring out where a head is, and how much two boxes overlap.

Nothing in this file touches YOLO, OpenCV, or the filesystem - it's just math
on tuples of numbers, which makes it easy to unit test on its own.
"""

from __future__ import annotations

import numpy as np

# Ultralytics' pose model always returns 17 keypoints in COCO order.
# We only care about the first 5 - everything above the shoulders.
NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR = 0, 1, 2, 3, 4
HEAD_KEYPOINTS = (NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR)

# Ultralytics gives every keypoint a confidence score. Below this, we don't
# trust it enough to use it (could be occluded, could be a bad guess).
MIN_KEYPOINT_CONFIDENCE = 0.30

# The head keypoints (eyes, ears, nose) all sit on the FACE, but a helmet rests
# on the crown - above and around the whole skull. So we can't just box the
# keypoints; we grow that box outward, using the eye-to-ear spread as our unit
# of scale (it stays sensible even when only two keypoints survive).
#
#   - UP: how far above the face keypoints to reach, to cover the crown + a
#     helmet sitting on top of it. This is the important one for helmet matching.
#   - DOWN: a little below the lowest face point, for the chin.
#   - HALF_WIDTH: half the box width, left and right of the face center.
HEAD_BOX_UP_SPANS = 1.4
HEAD_BOX_DOWN_SPANS = 0.35
HEAD_BOX_HALF_WIDTH_SPANS = 0.9


def find_head_box(keypoints: np.ndarray, person_box: tuple[int, int, int, int]) -> tuple[tuple[int, int, int, int], int, bool]:
    """
    Work out where this person's head is.

    Best case: we have a couple of confident head keypoints (eyes, ears,
    nose), so we just take a box around them and pad it out a bit.

    Worst case: the person is facing away from the camera, or too far/blurry
    for the model to trust any head keypoint. In that case we don't have much
    to go on, so we just guess: the top quarter of their body box is probably
    their head. It's crude, but it's better than nothing, and it keeps the
    downstream helmet-matching logic from breaking on hard frames.
    """
    trusted_points = []
    for kpt_index in HEAD_KEYPOINTS:
        x, y, confidence = keypoints[kpt_index]
        if confidence >= MIN_KEYPOINT_CONFIDENCE:
            trusted_points.append((x, y))

    # Two points is really the minimum needed to make a sensible box - one
    # point alone has no width or height.
    if len(trusted_points) >= 2:
        points = np.array(trusted_points, dtype=np.float32)
        left, top = points.min(axis=0)
        right, bottom = points.max(axis=0)

        center_x = (left + right) / 2.0
        # Our unit of scale: the wider of the horizontal/vertical keypoint spread.
        # Taking the max keeps two horizontally-aligned eyes (or ears) from
        # collapsing the box to a thin line - a real problem on people facing
        # away, where the ears are the only trustworthy head keypoints.
        span = max(right - left, bottom - top, 1.0)

        head_box = (
            int(round(center_x - span * HEAD_BOX_HALF_WIDTH_SPANS)),
            int(round(top - span * HEAD_BOX_UP_SPANS)),
            int(round(center_x + span * HEAD_BOX_HALF_WIDTH_SPANS)),
            int(round(bottom + span * HEAD_BOX_DOWN_SPANS)),
        )
        return clip_to_box(head_box, person_box), len(trusted_points), False

    # Fallback: no reliable keypoints, so just take the top of the body box.
    x1, y1, x2, y2 = person_box
    guessed_box = (x1, y1, x2, y1 + int((y2 - y1) * 0.25))
    return clip_to_box(guessed_box, person_box), len(trusted_points), True


def helmet_iou(head_box: tuple[int, int, int, int], helmet_box: tuple[int, int, int, int]) -> float:
    """
    How much a head box and a helmet box overlap, from 0 (no overlap) to 1
    (identical boxes). This is just standard IoU - nothing helmet-specific
    happens here yet.

    TODO: once we actually have a helmet detector, use this to match helmets
    to people. Rough plan:
      1. For every detected helmet, compute IoU against every person's head box.
      2. Assign each helmet to whichever person scores highest (as long as
         it's above some minimum, maybe ~0.1 - heads and helmets don't
         perfectly line up).
      3. Anyone left with no helmet assigned gets flagged as "no helmet".
    """
    return intersection_over_union(head_box, helmet_box)


def intersection_over_union(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    """Standard IoU between two boxes. Kept separate from helmet_iou so it's easy to unit test on its own."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    overlap_left = max(ax1, bx1)
    overlap_top = max(ay1, by1)
    overlap_right = min(ax2, bx2)
    overlap_bottom = min(ay2, by2)

    overlap_width = max(0, overlap_right - overlap_left)
    overlap_height = max(0, overlap_bottom - overlap_top)
    overlap_area = overlap_width * overlap_height
    if overlap_area == 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union_area = area_a + area_b - overlap_area

    return overlap_area / union_area if union_area > 0 else 0.0


def clip_to_box(box: tuple[int, int, int, int], bounds: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Make sure a box never sticks out past its parent box (the head can't be bigger than the body)."""
    x1, y1, x2, y2 = box
    bound_x1, bound_y1, bound_x2, bound_y2 = bounds
    return (
        max(x1, bound_x1),
        max(y1, bound_y1),
        min(x2, bound_x2),
        min(y2, bound_y2),
    )
