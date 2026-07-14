"""
Pose + ROI(head_box) + IoU helmet matching, detector slot = SH17 YOLOv9-e.

Design (per the pipe spec):
  - The ANCHOR is the pose face keypoints (nose/eyes/ears). find_head_box turns
    them into a head ROI; the helmet relationship is established from THAT.
    (Pose stays central - it's also what we'll use later for KVKK face blurring.)
  - From SH17 we use ONLY the `helmet` class. We deliberately IGNORE its `head`
    class - the "no helmet" signal comes from pose, not from SH17.

Per-person label:
  - a `helmet` box matches the head ROI (center-in + IoU)  -> "helmet"
  - head located from real keypoints but no helmet matched -> "no_helmet"
  - head could only be guessed (no trustworthy face keypoints, fallback box)
                                                            -> "unknown"

Output: observe/pipeline_match_sh17/  (annotated per image + tally)
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "core"))
from geometry import find_head_box, intersection_over_union  # noqa: E402

SAMPLES = ROOT / "test" / "samples" / "v1"
OUT = ROOT / "observe" / "pipeline_match_sh17"
CONF = 0.25
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

POSE_W = ROOT / "model" / "yolo26n-pose.pt"
SH17_W = ROOT / "model" / "external" / "sh17_yolo9e.pt"
SH17_HELMET = 10          # the only SH17 class we use (its `head`=12 is ignored)

COLOR = {"helmet": (0, 200, 0), "no_helmet": (0, 0, 230), "unknown": (150, 150, 150)}


def helmet_boxes(model, frame):
    """Run SH17, keep ONLY helmet boxes."""
    r = model.predict(frame, conf=CONF, classes=[SH17_HELMET], verbose=False)[0]
    out = []
    if r.boxes is not None:
        for b in r.boxes:
            out.append(tuple(int(v) for v in b.xyxy[0].tolist()))
    return out


def center_inside(box, head_box):
    cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
    hx1, hy1, hx2, hy2 = head_box
    return hx1 <= cx <= hx2 and hy1 <= cy <= hy2


def match_one(head_box, is_fallback, helmets):
    """helmet if a helmet box matches the head ROI; else no_helmet (or unknown if the head was only guessed)."""
    inside = [(box, intersection_over_union(head_box, box)) for box in helmets if center_inside(box, head_box)]
    if inside:
        best = max(inside, key=lambda d: d[1])
        return "helmet", best[0]
    return ("unknown" if is_fallback else "no_helmet"), None


def draw(frame, pbox, hbox, status, mbox, pid):
    col = COLOR[status]
    cv2.rectangle(frame, pbox[:2], pbox[2:], col, 2)
    cv2.rectangle(frame, hbox[:2], hbox[2:], col, 2)
    if mbox is not None:
        cv2.rectangle(frame, mbox[:2], mbox[2:], col, 1)
    cv2.putText(frame, f"P{pid}:{status}", (pbox[0], max(pbox[1] - 8, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("Loading pose + SH17...")
    pose = YOLO(str(POSE_W))
    sh17 = YOLO(str(SH17_W))
    images = sorted(p for p in SAMPLES.iterdir() if p.suffix.lower() in IMAGE_EXTS)

    tally = Counter()
    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        # pose -> per-person head ROI from face keypoints
        pr = pose.predict(frame, conf=CONF, verbose=False)[0]
        persons = []
        if pr.keypoints is not None and pr.boxes is not None:
            for pid, (box, kp) in enumerate(zip(pr.boxes.xyxy.cpu().numpy(),
                                                pr.keypoints.data.cpu().numpy())):
                pbox = tuple(int(v) for v in box[:4])
                hbox, nkpt, fb = find_head_box(kp, pbox)
                persons.append((pid, pbox, hbox, fb))

        helmets = helmet_boxes(sh17, frame)       # SH17 helmet boxes only
        canvas = frame.copy()
        counts = Counter()
        for pid, pbox, hbox, fb in persons:
            status, mbox = match_one(hbox, fb, helmets)
            draw(canvas, pbox, hbox, status, mbox, pid)
            counts[status] += 1
            tally[status] += 1
        cv2.imwrite(str(OUT / img_path.name), canvas)
        print(f"{img_path.name}: {len(persons)} persons | SH17 helmets={len(helmets)}"
              f" | helmet={counts['helmet']} no_helmet={counts['no_helmet']} unknown={counts['unknown']}")

    print(f"\nTOTAL  helmet={tally['helmet']}  no_helmet={tally['no_helmet']}  unknown={tally['unknown']}")
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
