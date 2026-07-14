"""
The combined pipeline, run with each detector so we can compare them.

  pose  -> per-person head_box
  detector (hardhat placeholder OR ppe_v1) -> Hardhat / NO-Hardhat boxes
  hybrid match -> assign a helmet box to each person's head, label them

Matching (our agreed design):
  - primary: a detection whose CENTER falls inside the head_box is a candidate
  - if several, tie-break by IoU (highest wins)
  - safety-first: on a near-tie between Hardhat and NO-Hardhat, NO-Hardhat wins
    (a false "no helmet" is a safer error on a construction site)
  - no candidate at all -> "unknown" (NOT "no_helmet")

For ppe_v1 (14 classes) we only feed the Hardhat / NO-Hardhat boxes into the
helmet matcher here; the other PPE classes (vest, gloves, ...) are a later step.

Output (observe/pipeline_match/, git-ignored):
  hardhat/<photo>, ppe_v1/<photo>   annotated per detector
  combined/<photo>                  the two side by side
  match_summary.txt / .csv          helmet / no_helmet / unknown tallies
"""
from __future__ import annotations

import csv
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "core"))

from geometry import find_head_box, intersection_over_union  # noqa: E402

SAMPLES = ROOT / "test" / "samples" / "v1"
OUT = ROOT / "observe" / "pipeline_match"
CONF = 0.25
SAFETY_TIE = 0.10          # NO-Hardhat wins if within this IoU of the best Hardhat

DETECTORS = {
    "hardhat": ROOT / "model" / "test" / "hardhat_yolov8n.pt",
    "ppe_v1":  ROOT / "model" / "ppe_v1.pt",
}
HELMET_CLASSES = {"Hardhat", "NO-Hardhat"}

COLOR = {"helmet": (0, 200, 0), "no_helmet": (0, 0, 230), "unknown": (150, 150, 150)}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def helmet_boxes(detector, frame):
    """Run a detector, keep only Hardhat / NO-Hardhat as (cls, conf, box)."""
    r = detector.predict(frame, conf=CONF, verbose=False)[0]
    out = []
    if r.boxes is not None:
        for b in r.boxes:
            cls = r.names[int(b.cls[0])]
            if cls in HELMET_CLASSES:
                out.append((cls, float(b.conf[0]), tuple(int(v) for v in b.xyxy[0].tolist())))
    return out


def center_inside(box, head_box):
    cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
    hx1, hy1, hx2, hy2 = head_box
    return hx1 <= cx <= hx2 and hy1 <= cy <= hy2


def match_one(head_box, dets):
    """Hybrid match: return (status, matched_box) for one head."""
    inside = [(cls, conf, box, intersection_over_union(head_box, box))
              for cls, conf, box in dets if center_inside(box, head_box)]
    if not inside:
        return "unknown", None
    inside.sort(key=lambda d: d[3], reverse=True)   # by IoU
    best = inside[0]
    # safety-first: prefer a near-tied NO-Hardhat over a Hardhat
    if best[0] == "Hardhat":
        for cls, conf, box, iou in inside:
            if cls == "NO-Hardhat" and iou >= best[3] - SAFETY_TIE:
                best = (cls, conf, box, iou)
                break
    status = "helmet" if best[0] == "Hardhat" else "no_helmet"
    return status, best[2]


def draw(frame, person_box, head_box, status, matched_box, pid):
    col = COLOR[status]
    x1, y1, x2, y2 = person_box
    cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
    hx1, hy1, hx2, hy2 = head_box
    cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), col, 2)
    if matched_box is not None:
        cv2.rectangle(frame, matched_box[:2], matched_box[2:], col, 1)
    label = f"P{pid}:{status}"
    cv2.putText(frame, label, (x1, max(y1 - 8, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)


def label_panel(img, text):
    bar = np.zeros((46, img.shape[1], 3), np.uint8)
    cv2.putText(bar, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([bar, img])


def fit_height(img, h):
    return cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h))


def main():
    for name in DETECTORS:
        (OUT / name).mkdir(parents=True, exist_ok=True)
    (OUT / "combined").mkdir(parents=True, exist_ok=True)

    print("Loading pose + detectors...")
    pose = YOLO(str(ROOT / "model" / "yolo26n-pose.pt"))
    detectors = {name: YOLO(str(p)) for name, p in DETECTORS.items()}

    images = sorted(p for p in SAMPLES.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    print(f"{len(images)} photos\n")

    per_image_rows = []
    tally = {name: Counter() for name in DETECTORS}   # helmet/no_helmet/unknown per detector

    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        # pose once, reused for both detectors
        pr = pose.predict(frame, conf=CONF, verbose=False)[0]
        persons = []
        if pr.keypoints is not None and pr.boxes is not None:
            boxes = pr.boxes.xyxy.cpu().numpy()
            kpts = pr.keypoints.data.cpu().numpy()
            for pid, (box, kp) in enumerate(zip(boxes, kpts)):
                pbox = tuple(int(v) for v in box[:4])
                hbox, _, _ = find_head_box(kp, pbox)
                persons.append((pid, pbox, hbox))

        panels = []
        line = f"{img_path.name}: {len(persons)} persons"
        for name in DETECTORS:
            canvas = frame.copy()
            dets = helmet_boxes(detectors[name], frame)
            counts = Counter()
            for pid, pbox, hbox in persons:
                status, mbox = match_one(hbox, dets)
                draw(canvas, pbox, hbox, status, mbox, pid)
                counts[status] += 1
                tally[name][status] += 1
            cv2.imwrite(str(OUT / name / img_path.name), canvas)
            panels.append(label_panel(fit_height(canvas, 620),
                          f"pose + {name}  H:{counts['helmet']} N:{counts['no_helmet']} ?:{counts['unknown']}"))
            per_image_rows.append([img_path.name, name, len(persons),
                                   counts['helmet'], counts['no_helmet'], counts['unknown']])
            line += f"  |  {name}: helmet={counts['helmet']} no={counts['no_helmet']} unk={counts['unknown']}"
        print(line)

        gap = np.full((panels[0].shape[0], 6, 3), 60, np.uint8)
        combined = panels[0]
        for p in panels[1:]:
            combined = np.hstack([combined, gap, p])
        cv2.imwrite(str(OUT / "combined" / img_path.name), combined)

    # ---- write outputs ----
    with open(OUT / "match_per_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "detector", "persons", "helmet", "no_helmet", "unknown"])
        w.writerows(per_image_rows)

    lines = ["Combined pipeline (pose + detector + hybrid match) on test/samples/v1",
             f"{len(images)} photos, conf>={CONF}, safety-tie IoU={SAFETY_TIE}\n",
             f"{'detector':10} {'helmet':>7} {'no_helmet':>10} {'unknown':>8}",
             "-" * 40]
    with open(OUT / "match_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["detector", "helmet", "no_helmet", "unknown"])
        for name in DETECTORS:
            t = tally[name]
            w.writerow([name, t['helmet'], t['no_helmet'], t['unknown']])
            lines.append(f"{name:10} {t['helmet']:>7} {t['no_helmet']:>10} {t['unknown']:>8}")
    text = "\n".join(lines)
    (OUT / "match_summary.txt").write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
