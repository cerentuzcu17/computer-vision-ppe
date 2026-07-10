"""
Pose prototype for the helmet-detection project.

The idea here is simple: before we can tell whether someone is wearing a
helmet, we first need to know where each person's HEAD actually is in the
frame. So this script runs YOLO's pose model on an image/video/webcam feed,
finds every person, looks at their head keypoints (nose, eyes, ears), and
draws a box around the head.

That head box is the thing we'll later compare against a detected helmet box
(using IoU) to decide "this helmet belongs to this person". The helmet model
isn't wired in yet - see the TODO next to helmet_iou() below.

How to run it:
    python pose_prototype.py --source sample.jpg
    python pose_prototype.py --source video.mp4
    python pose_prototype.py --source 0        (0 = webcam)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO

# Ultralytics' pose model always returns 17 keypoints in COCO order.
# We only care about the first 5 - everything above the shoulders.
NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR = 0, 1, 2, 3, 4
HEAD_KEYPOINTS = (NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR)

# Ultralytics gives every keypoint a confidence score. Below this, we don't
# trust it enough to use it (could be occluded, could be a bad guess).
MIN_KEYPOINT_CONFIDENCE = 0.30

# The head keypoints sit near the center of the face, not at the edges of the
# head, so we need to pad the box outward or it'll look too tight.
HEAD_BOX_PADDING = 0.20

# Which keypoints connect to which, so we can draw a stick-figure skeleton.
# Also COCO's standard layout - nothing custom here.
SKELETON_BONES = (
    (0, 1), (0, 2), (1, 3), (2, 4),            # face
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),   # shoulders + arms
    (5, 11), (6, 12), (11, 12),                # torso
    (11, 13), (13, 15), (12, 14), (14, 16),    # legs
)


@dataclass
class Person:
    """Everything we know about one person in a frame."""

    person_id: int
    box: tuple[int, int, int, int]          # the full-body box: x1, y1, x2, y2
    keypoints: np.ndarray                    # raw (17, 3) array from the model: x, y, confidence
    head_box: tuple[int, int, int, int]
    head_keypoints_found: int                # how many head keypoints we actually trusted
    used_fallback_box: bool                  # True if we had to guess the head box instead of measuring it


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

        width = max(right - left, 1.0)
        height = max(bottom - top, 1.0)
        pad_x = width * HEAD_BOX_PADDING + width * 0.5
        pad_y = height * HEAD_BOX_PADDING + height * 0.7  # extra room for forehead/chin

        head_box = (
            int(round(left - pad_x)),
            int(round(top - pad_y)),
            int(round(right + pad_x)),
            int(round(bottom + pad_y)),
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


def extract_people(yolo_result) -> list[Person]:
    """Turn one YOLO prediction into a list of Person objects we can actually work with."""
    if yolo_result.keypoints is None or yolo_result.boxes is None:
        return []

    boxes = yolo_result.boxes.xyxy.cpu().numpy()
    all_keypoints = yolo_result.keypoints.data.cpu().numpy()  # shape: (num_people, 17, 3)

    people = []
    for person_id, (box, keypoints) in enumerate(zip(boxes, all_keypoints)):
        person_box = tuple(int(value) for value in box[:4])
        head_box, head_kpt_count, used_fallback = find_head_box(keypoints, person_box)

        people.append(Person(
            person_id=person_id,
            box=person_box,
            keypoints=keypoints,
            head_box=head_box,
            head_keypoints_found=head_kpt_count,
            used_fallback_box=used_fallback,
        ))

    return people


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


def process_frame(model: YOLO, frame: np.ndarray, confidence_threshold: float) -> np.ndarray:
    """Run the model on one frame and draw the results directly onto it."""
    result = model.predict(frame, conf=confidence_threshold, verbose=False)[0]
    people = extract_people(result)

    print(f"[frame] found {len(people)} people")
    for person in people:
        draw_person(frame, person)
        print_person_summary(person)

    return frame


def looks_like_an_image(source: str) -> bool:
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
    return source.lower().endswith(image_extensions)


def run(source: str, weights_path: str, confidence_threshold: float, save_path: str | None, show_window: bool) -> None:
    """Main entry point: figures out whether we're dealing with an image, a video, or a webcam, and handles each."""
    print(f"[model] loading weights from: {weights_path}")
    model = YOLO(weights_path)  # if the file isn't there yet, ultralytics downloads it for us

    if looks_like_an_image(source):
        _run_on_image(model, source, confidence_threshold, save_path, show_window)
    else:
        _run_on_video_or_webcam(model, source, confidence_threshold, save_path, show_window)


def _run_on_image(model: YOLO, source: str, confidence_threshold: float, save_path: str | None, show_window: bool) -> None:
    frame = cv2.imread(source)
    if frame is None:
        raise FileNotFoundError(f"couldn't read image: {source}")

    annotated = process_frame(model, frame, confidence_threshold)

    if save_path:
        cv2.imwrite(save_path, annotated)
        print(f"[saved] {save_path}")

    if show_window:
        cv2.imshow("pose_prototype", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def _run_on_video_or_webcam(model: YOLO, source: str, confidence_threshold: float, save_path: str | None, show_window: bool) -> None:
    # A plain digit like "0" means webcam index, anything else is a file path.
    capture = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not capture.isOpened():
        raise RuntimeError(f"couldn't open source: {source}")

    video_writer = None
    try:
        while True:
            got_frame, frame = capture.read()
            if not got_frame:
                break

            annotated = process_frame(model, frame, confidence_threshold)

            if save_path:
                if video_writer is None:
                    height, width = annotated.shape[:2]
                    fps = capture.get(cv2.CAP_PROP_FPS) or 25
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
                video_writer.write(annotated)

            if show_window:
                cv2.imshow("pose_prototype", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        if video_writer is not None:
            video_writer.release()
            print(f"[saved] {save_path}")
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find people and their head regions using YOLO-pose.")
    parser.add_argument("--source", required=True,
                         help="path to an image or video, or '0' to use the webcam")
    parser.add_argument("--weights", default="model/yolo26n-pose.pt",
                         help="YOLO-pose weights file (downloaded automatically into model/ if missing)")
    parser.add_argument("--conf", type=float, default=0.25,
                         help="minimum confidence for a person detection to count")
    parser.add_argument("--save", default=None,
                         help="where to write the annotated output (image or video)")
    parser.add_argument("--show", action="store_true",
                         help="pop up a window and show the result live")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.source, args.weights, args.conf, args.save, args.show)


if __name__ == "__main__":
    main()
