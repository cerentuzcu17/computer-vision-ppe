"""
Same ROI + IoU matching pipeline, but the detector slot is now the SH17
YOLOv9-e model instead of the 2-class hardhat placeholder.

Key difference in class semantics:
  SH17 has separate `helmet` (id 10) and `head` (id 12) classes (no "NO-Hardhat").
    helmet box matched to a person's head_box -> "helmet"   (wearing one)
    bare head box matched                     -> "no_helmet" (none detected on the head)
    nothing matched                           -> "unknown"

Conflict rule here is PROVISIONAL (pending the pipe design): if both a helmet and a
head box land on the same person, we treat it as "helmet" - a detected helmet on the
head means they're wearing it. (This is the opposite of the safety-first tie-break we
used for the Hardhat/NO-Hardhat placeholder, because `head` != `NO-Hardhat`.)

Output: observe/pipeline_match_sh17/  (annotated + combined vs hardhat + tally)
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
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
SH17_HELMET, SH17_HEAD = 10, 12          # class ids in the SH17 model

COLOR = {"helmet": (0, 200, 0), "no_helmet": (0, 0, 230), "unknown": (150, 150, 150)}


def detections(model, frame):
    """Run SH17, keep only helmet / head boxes as (kind, box)."""
    r = model.predict(frame, conf=CONF, classes=[SH17_HELMET, SH17_HEAD], verbose=False)[0]
    out = []
    if r.boxes is not None:
        for b in r.boxes:
            kind = "helmet" if int(b.cls[0]) == SH17_HELMET else "head"
            out.append((kind, tuple(int(v) for v in b.xyxy[0].tolist())))
    return out


def center_inside(box, head_box):
    cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
    hx1, hy1, hx2, hy2 = head_box
    return hx1 <= cx <= hx2 and hy1 <= cy <= hy2


def match_one(head_box, dets):
    """Return (status, matched_box). Provisional: a helmet on the head wins over a bare head."""
    inside = [(kind, box, intersection_over_union(head_box, box))
              for kind, box in dets if center_inside(box, head_box)]
    if not inside:
        return "unknown", None
    helmets = [d for d in inside if d[0] == "helmet"]
    if helmets:
        best = max(helmets, key=lambda d: d[2])
        return "helmet", best[1]
    best = max(inside, key=lambda d: d[2])   # only head boxes left
    return "no_helmet", best[1]


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
        pr = pose.predict(frame, conf=CONF, verbose=False)[0]
        persons = []
        if pr.keypoints is not None and pr.boxes is not None:
            for pid, (box, kp) in enumerate(zip(pr.boxes.xyxy.cpu().numpy(),
                                                pr.keypoints.data.cpu().numpy())):
                pbox = tuple(int(v) for v in box[:4])
                hbox, _, _ = find_head_box(kp, pbox)
                persons.append((pid, pbox, hbox))

        dets = detections(sh17, frame)
        n_helmet_box = sum(1 for k, _ in dets if k == "helmet")
        n_head_box = sum(1 for k, _ in dets if k == "head")
        canvas = frame.copy()
        counts = Counter()
        for pid, pbox, hbox in persons:
            status, mbox = match_one(hbox, dets)
            draw(canvas, pbox, hbox, status, mbox, pid)
            counts[status] += 1
            tally[status] += 1
        cv2.imwrite(str(OUT / img_path.name), canvas)
        print(f"{img_path.name}: {len(persons)} persons | SH17 boxes helmet={n_helmet_box} head={n_head_box}"
              f" | matched helmet={counts['helmet']} no_helmet={counts['no_helmet']} unknown={counts['unknown']}")

    print(f"\nTOTAL  helmet={tally['helmet']}  no_helmet={tally['no_helmet']}  unknown={tally['unknown']}")
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
