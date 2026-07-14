"""
Compare the three models we have on the v1 sample photos, side by side.

  - ppe_v1   : our trained 14-class PPE detector          (model/ppe_v1.pt)
  - hardhat  : third-party 2-class placeholder            (model/test/hardhat_yolov8n.pt)
  - pose     : YOLO26 pose + our head_box logic           (model/yolo26n-pose.pt)

There are no ground-truth labels for these photos, so this is NOT an accuracy
(mAP) benchmark. It's a descriptive comparison: what each model fires on, how
confidently, and how fast - plus annotated images to eyeball.

Outputs (under observe/compare_v1/, git-ignored):
  <model>/<photo>        annotated image per model
  combined/<photo>       the three annotations side by side
  metrics_per_image.csv  one row per (photo, model)
  metrics_summary.csv    one row per model (aggregate)
  metrics_summary.txt    same aggregate, human-readable

Run:  uv run test/compare_models.py
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

from drawing import draw_person          # noqa: E402
from entities import Person              # noqa: E402
from geometry import find_head_box       # noqa: E402

SAMPLES = ROOT / "test" / "samples" / "v1"
OUT = ROOT / "observe" / "compare_v1"
CONF = 0.25

MODELS = {
    "ppe_v1":  ROOT / "model" / "ppe_v1.pt",
    "hardhat": ROOT / "model" / "test" / "hardhat_yolov8n.pt",
    "pose":    ROOT / "model" / "yolo26n-pose.pt",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images() -> list[Path]:
    return sorted(p for p in SAMPLES.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def run_detect(model, frame):
    """Detection model: return (annotated BGR, list of (class, conf), infer_ms)."""
    t0 = time.perf_counter()
    result = model.predict(frame, conf=CONF, verbose=False)[0]
    infer_ms = (time.perf_counter() - t0) * 1000
    dets = []
    if result.boxes is not None:
        for b in result.boxes:
            dets.append((result.names[int(b.cls[0])], float(b.conf[0])))
    return result.plot(), dets, infer_ms


def run_pose(model, frame):
    """Pose model: draw skeleton + head_box, return (annotated, n_persons, infer_ms)."""
    t0 = time.perf_counter()
    result = model.predict(frame, conf=CONF, verbose=False)[0]
    infer_ms = (time.perf_counter() - t0) * 1000
    annotated = frame.copy()
    n = 0
    if result.keypoints is not None and result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        kpts = result.keypoints.data.cpu().numpy()
        for pid, (box, kp) in enumerate(zip(boxes, kpts)):
            pbox = tuple(int(v) for v in box[:4])
            hbox, nkpt, fb = find_head_box(kp, pbox)
            draw_person(annotated, Person(pid, pbox, kp, hbox, nkpt, fb))
            n += 1
    return annotated, n, infer_ms


def label_panel(img, text):
    """Put a title bar on top of a panel."""
    bar = np.zeros((46, img.shape[1], 3), np.uint8)
    cv2.putText(bar, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([bar, img])


def fit_height(img, h):
    return cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h))


def main():
    for name in MODELS:
        (OUT / name).mkdir(parents=True, exist_ok=True)
    (OUT / "combined").mkdir(parents=True, exist_ok=True)

    print("Loading models...")
    models = {name: YOLO(str(path)) for name, path in MODELS.items()}

    images = list_images()
    print(f"{len(images)} sample photos in {SAMPLES}\n")

    per_image_rows = []
    # aggregate accumulators per model
    agg = {name: {"images": 0, "dets": 0, "conf_sum": 0.0, "ms_sum": 0.0,
                  "classes": Counter()} for name in MODELS}

    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  [skip] unreadable: {img_path.name}")
            continue
        panels = []
        print(f"{img_path.name}")

        # --- ppe_v1 ---
        ann, dets, ms = run_detect(models["ppe_v1"], frame)
        cv2.imwrite(str(OUT / "ppe_v1" / img_path.name), ann)
        classes = Counter(c for c, _ in dets)
        mean_conf = sum(cf for _, cf in dets) / len(dets) if dets else 0.0
        per_image_rows.append([img_path.name, "ppe_v1", len(dets), f"{mean_conf:.3f}",
                               f"{ms:.1f}", "; ".join(f"{k}:{v}" for k, v in classes.most_common())])
        agg["ppe_v1"]["images"] += 1; agg["ppe_v1"]["dets"] += len(dets)
        agg["ppe_v1"]["conf_sum"] += sum(cf for _, cf in dets); agg["ppe_v1"]["ms_sum"] += ms
        agg["ppe_v1"]["classes"] += classes
        panels.append(label_panel(fit_height(ann, 600), f"ppe_v1  ({len(dets)} det)"))
        print(f"  ppe_v1 : {len(dets)} det, mean_conf={mean_conf:.2f}, {ms:.0f}ms  {dict(classes)}")

        # --- hardhat placeholder ---
        ann, dets, ms = run_detect(models["hardhat"], frame)
        cv2.imwrite(str(OUT / "hardhat" / img_path.name), ann)
        classes = Counter(c for c, _ in dets)
        mean_conf = sum(cf for _, cf in dets) / len(dets) if dets else 0.0
        per_image_rows.append([img_path.name, "hardhat", len(dets), f"{mean_conf:.3f}",
                               f"{ms:.1f}", "; ".join(f"{k}:{v}" for k, v in classes.most_common())])
        agg["hardhat"]["images"] += 1; agg["hardhat"]["dets"] += len(dets)
        agg["hardhat"]["conf_sum"] += sum(cf for _, cf in dets); agg["hardhat"]["ms_sum"] += ms
        agg["hardhat"]["classes"] += classes
        panels.append(label_panel(fit_height(ann, 600), f"hardhat  ({len(dets)} det)"))
        print(f"  hardhat: {len(dets)} det, mean_conf={mean_conf:.2f}, {ms:.0f}ms  {dict(classes)}")

        # --- pose ---
        ann, n_persons, ms = run_pose(models["pose"], frame)
        cv2.imwrite(str(OUT / "pose" / img_path.name), ann)
        per_image_rows.append([img_path.name, "pose", n_persons, "", f"{ms:.1f}", f"persons:{n_persons}"])
        agg["pose"]["images"] += 1; agg["pose"]["dets"] += n_persons; agg["pose"]["ms_sum"] += ms
        panels.append(label_panel(fit_height(ann, 600), f"pose  ({n_persons} person)"))
        print(f"  pose   : {n_persons} persons, {ms:.0f}ms")

        # --- combined side by side ---
        gap = np.full((panels[0].shape[0], 6, 3), 60, np.uint8)
        combined = panels[0]
        for p in panels[1:]:
            combined = np.hstack([combined, gap, p])
        cv2.imwrite(str(OUT / "combined" / img_path.name), combined)

    # ---- write per-image csv ----
    with open(OUT / "metrics_per_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "model", "detections", "mean_conf", "infer_ms", "classes"])
        w.writerows(per_image_rows)

    # ---- aggregate ----
    summary_rows = []
    for name, a in agg.items():
        imgs = a["images"] or 1
        dets = a["dets"]
        mean_det = dets / imgs
        mean_conf = (a["conf_sum"] / dets) if dets else 0.0
        mean_ms = a["ms_sum"] / imgs
        top = "; ".join(f"{k}:{v}" for k, v in a["classes"].most_common(6)) or "-"
        summary_rows.append([name, a["images"], dets, f"{mean_det:.2f}",
                             f"{mean_conf:.3f}", f"{mean_ms:.1f}", top])

    with open(OUT / "metrics_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "images", "total_det", "det_per_img", "mean_conf", "mean_infer_ms", "top_classes"])
        w.writerows(summary_rows)

    # ---- human-readable summary ----
    lines = []
    lines.append("PPE model comparison on test/samples/v1")
    lines.append(f"{len(images)} photos, conf>={CONF}. No ground truth -> descriptive stats, not mAP.\n")
    header = f"{'model':10} {'imgs':>4} {'det':>5} {'det/img':>8} {'meanconf':>9} {'ms/img':>8}   top classes"
    lines.append(header)
    lines.append("-" * len(header))
    for r in summary_rows:
        lines.append(f"{r[0]:10} {r[1]:>4} {r[2]:>5} {r[3]:>8} {r[4]:>9} {r[5]:>8}   {r[6]}")
    text = "\n".join(lines)
    (OUT / "metrics_summary.txt").write_text(text, encoding="utf-8")

    print("\n" + text)
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
