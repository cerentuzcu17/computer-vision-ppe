"""YOLO-pose prototype: per-person head region extraction.

Detects every person, reduces them to 17 COCO keypoints, builds a "head box"
from the head keypoints, and lays the groundwork for (later) matching that
head box against a helmet box via IoU to assign the helmet to the right person.

Usage:
    python pose_prototype.py --source sample.jpg
    python pose_prototype.py --source video.mp4
    python pose_prototype.py --source 0            # webcam
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO

import os
from core.security import ImageEncryptor

# Static secret key for prototyping. Must be moved to a secure .env file in production.
SECRET_KEY = b'vK9_X1zY7b_NMoP83fL_TkWq4vE9mZaR_BcDeFgHiJk='
encryptor = ImageEncryptor(key=SECRET_KEY)

# --- COCO pose keypoint indices ---------------------------------------------
# 0=nose, 1=left eye, 2=right eye, 3=left ear, 4=right ear
HEAD_KPT_IDS = (0, 1, 2, 3, 4)
KPT_CONF_THRESH = 0.30          # keypoints below this confidence are untrusted
HEAD_PADDING_RATIO = 0.20       # margin added around the head keypoint cluster

# Bone connections in the COCO skeleton (0-indexed keypoint pairs)
SKELETON = (
    (0, 1), (0, 2), (1, 3), (2, 4),           # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms / shoulders
    (5, 11), (6, 12), (11, 12),               # torso
    (11, 13), (13, 15), (12, 14), (14, 16),   # legs
)


@dataclass
class Person:
    """A single detected person."""

    person_id: int
    person_box: tuple[int, int, int, int]           # x1, y1, x2, y2
    keypoints: np.ndarray                            # (17, 3) -> x, y, conf
    head_box: tuple[int, int, int, int] | None
    head_kpt_count: int                              # number of confident head kpts
    head_from_fallback: bool                         # whether the box used the fallback


# --- Geometry functions (pure, testable) ------------------------------------
def head_box_from_keypoints(
    keypoints: np.ndarray,
    person_box: tuple[int, int, int, int],
    conf_thresh: float = KPT_CONF_THRESH,
    padding_ratio: float = HEAD_PADDING_RATIO,
) -> tuple[tuple[int, int, int, int], int, bool]:
    """Build a head box from head keypoints.

    If enough confident head keypoints exist, returns their bounding box
    (with padding). Otherwise falls back to the top 25% of the person box.

    Returns: (head_box, confident_head_kpt_count, fallback_used)
    """
    head_pts = []
    for idx in HEAD_KPT_IDS:
        x, y, conf = keypoints[idx]
        if conf >= conf_thresh:
            head_pts.append((x, y))

    if len(head_pts) >= 2:
        pts = np.array(head_pts, dtype=np.float32)
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        # Keypoints cluster near the head's center; pad the box outward.
        w = max(x2 - x1, 1.0)
        h = max(y2 - y1, 1.0)
        pad_x = w * padding_ratio + w * 0.5
        pad_y = h * padding_ratio + h * 0.7   # a bit more vertically (forehead/chin)
        box = (
            int(round(x1 - pad_x)),
            int(round(y1 - pad_y)),
            int(round(x2 + pad_x)),
            int(round(y2 + pad_y)),
        )
        return _clip_box(box, person_box), len(head_pts), False

    # Fallback: top 25% of the person box.
    px1, py1, px2, py2 = person_box
    box = (px1, py1, px2, py1 + int((py2 - py1) * 0.25))
    return _clip_box(box, person_box), len(head_pts), True


def helmet_iou(
    head_box: tuple[int, int, int, int],
    helmet_box: tuple[int, int, int, int],
) -> float:
    """IoU (0..1) between a head box and a helmet box.

    In the helmet step, each detected helmet box will be compared against
    every person's head box, and assigned to the person with the highest IoU.

    TODO(helmet): once the helmet model is added:
        - Match helmet detections to heads with this function (Hungarian / greedy).
        - Calibrate the IoU threshold (e.g. > 0.1); adjust head box padding accordingly.
        - Label a person "no helmet" when there is no overlap.
    """
    return _iou(head_box, helmet_box)


def _iou(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
) -> float:
    """Intersection-over-Union between two axis-aligned boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _clip_box(
    box: tuple[int, int, int, int],
    bounds: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Clip a box to the given bounds (the person box)."""
    x1, y1, x2, y2 = box
    bx1, by1, bx2, by2 = bounds
    return (
        max(x1, bx1),
        max(y1, by1),
        min(x2, bx2),
        min(y2, by2),
    )


# --- Inference + drawing -----------------------------------------------------
def build_people(result) -> list[Person]:
    """Build a list of Person from an Ultralytics result."""
    people: list[Person] = []
    if result.keypoints is None or result.boxes is None:
        return people

    boxes = result.boxes.xyxy.cpu().numpy()
    kpts = result.keypoints.data.cpu().numpy()   # (N, 17, 3)

    for i, (box, kp) in enumerate(zip(boxes, kpts)):
        person_box = tuple(int(v) for v in box[:4])
        head_box, head_count, fallback = head_box_from_keypoints(kp, person_box)
        people.append(
            Person(
                person_id=i,
                person_box=person_box,
                keypoints=kp,
                head_box=head_box,
                head_kpt_count=head_count,
                head_from_fallback=fallback,
            )
        )
    return people


def draw_person(frame: np.ndarray, person: Person) -> None:
    """Draw the skeleton, head keypoints, head box, and person box."""
    kp = person.keypoints

    # Person box (green)
    px1, py1, px2, py2 = person.person_box
    cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 200, 0), 2)
    cv2.putText(
        frame, f"id {person.person_id}", (px1, max(py1 - 6, 12)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA,
    )

    # Skeleton bones (blue)
    for a, b in SKELETON:
        xa, ya, ca = kp[a]
        xb, yb, cb = kp[b]
        if ca >= KPT_CONF_THRESH and cb >= KPT_CONF_THRESH:
            cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)),
                     (255, 160, 0), 2, cv2.LINE_AA)

    # Head keypoints (red dot)
    for idx in HEAD_KPT_IDS:
        x, y, conf = kp[idx]
        if conf >= KPT_CONF_THRESH:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1, cv2.LINE_AA)

    # Head box (cyan; a darker shade when it came from the fallback)
    if person.head_box is not None:
        hx1, hy1, hx2, hy2 = person.head_box
        color = (0, 220, 220) if not person.head_from_fallback else (0, 140, 220)
        cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), color, 2)


def log_person(person: Person) -> None:
    """Print a per-person summary to the console."""
    tag = "fallback" if person.head_from_fallback else "keypoint"
    print(
        f"  person {person.person_id:>2} | head_box={person.head_box} "
        f"| confident_head_kpts={person.head_kpt_count} ({tag})"
    )


def process_frame(model: YOLO, frame: np.ndarray, conf: float) -> np.ndarray:
    """Process a single frame: infer -> draw -> log. Returns the annotated frame."""
    result = model.predict(frame, conf=conf, verbose=False)[0]
    people = build_people(result)
    print(f"[frame] people detected: {len(people)}")
    for person in people:
        draw_person(frame, person)
        log_person(person)
    return frame


# --- Input handling -----------------------------------------------------
def is_image(source: str) -> bool:
    return source.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
    )


def run(source: str, weights: str, conf: float, save: str | None, show: bool) -> None:
    """Process the source (image/video/webcam)."""
    print(f"[model] loading: {weights}")
    model = YOLO(weights)          # weights auto-download if missing

    if is_image(source):
        frame = cv2.imread(source)
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {source}")
        out = process_frame(model, frame, conf)
        if save:
            cv2.imwrite(save, out)
            print(f"[saved] {save}")
        if show:
            cv2.imshow("pose_prototype", out)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    # Video or webcam
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    writer = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out = process_frame(model, frame, conf)
        if save:
            if writer is None:
                h, w = out.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                fps = cap.get(cv2.CAP_PROP_FPS) or 25
                writer = cv2.VideoWriter(save, fourcc, fps, (w, h))
            writer.write(out)
        if show:
            cv2.imshow("pose_prototype", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer is not None:
        writer.release()
        print(f"[saved] {save}")
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO-pose head region prototype")
    p.add_argument("--source", required=True,
                   help="image/video path, or 0 for webcam")
    p.add_argument("--weights", default="model/yolo11n-pose.pt",
                   help="YOLO-pose weights (auto-downloads into model/)")
    p.add_argument("--conf", type=float, default=0.25,
                   help="detection confidence threshold")
    p.add_argument("--save", default=None,
                   help="output file (image/video)")
    p.add_argument("--show", action="store_true",
                   help="show in a window")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(args.source, args.weights, args.conf, args.save, args.show)


if __name__ == "__main__":
    main()
