"""
Run the privacy-aware PPE pipeline on test/samples/v1 and draw the results.

Each output frame has the face already removed (anonymized) - the pose skeleton
and keypoints are drawn ON TOP, showing that we keep the geometry but not the
identity. head ROI is colored by the helmet verdict, face ROI by the goggle
verdict (yes=green, no=red, unknown=gray).

Output: observe/privacy_pipeline/
Run:    uv run test/run_privacy_pipeline.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "core"))
from ppe_pipeline import PpePipeline    # noqa: E402
from drawing import draw_person         # noqa: E402
from entities import YES, NO, UNKNOWN   # noqa: E402

SAMPLES = ROOT / "test" / "samples" / "v1"
OUT = ROOT / "observe" / "privacy_pipeline"
POSE_W = ROOT / "model" / "yolo26n-pose.pt"
DETECTOR_W = ROOT / "model" / "external" / "sh17_yolo9e.pt"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

COLOR = {YES: (0, 200, 0), NO: (0, 0, 230), UNKNOWN: (150, 150, 150)}


def draw_result(canvas, result):
    draw_person(canvas, result.person)                 # pose skeleton + keypoints (kept) + id
    p, ppe = result.person, result.ppe
    cv2.rectangle(canvas, p.head_box[:2], p.head_box[2:], COLOR[ppe.helmet], 3)   # helmet verdict
    cv2.rectangle(canvas, result.face_box[:2], result.face_box[2:], COLOR[ppe.goggle], 2)  # goggle verdict
    label = f"P{p.person_id} helmet:{ppe.helmet} goggle:{ppe.goggle}"
    cv2.putText(canvas, label, (p.box[0], max(p.box[1] - 8, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR[ppe.helmet], 2, cv2.LINE_AA)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("Loading pipeline (pose + SH17)...")
    pipeline = PpePipeline(str(POSE_W), str(DETECTOR_W), anonymize_method="mask")

    images = sorted(p for p in SAMPLES.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    helmet_tally, goggle_tally = Counter(), Counter()

    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        anonymized, results = pipeline.process(frame)     # raw `frame` no longer needed after this
        for r in results:
            draw_result(anonymized, r)
            helmet_tally[r.ppe.helmet] += 1
            goggle_tally[r.ppe.goggle] += 1
        cv2.imwrite(str(OUT / img_path.name), anonymized)
        hs = Counter(r.ppe.helmet for r in results)
        gs = Counter(r.ppe.goggle for r in results)
        print(f"{img_path.name}: {len(results)} persons | "
              f"helmet {dict(hs)} | goggle {dict(gs)}")

    print(f"\nHELMET  {dict(helmet_tally)}")
    print(f"GOGGLE  {dict(goggle_tally)}")
    print(f"Faces anonymized, saved to {OUT}")


if __name__ == "__main__":
    main()
