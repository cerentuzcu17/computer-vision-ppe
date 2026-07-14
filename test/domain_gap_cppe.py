"""
Domain-gap check: evaluate ppe_v1 on the Construction-PPE test set.

ppe_v1 was trained on a merged Roboflow dataset. Construction-PPE is a
different source (real construction sites, cleanly labelled). If ppe_v1's
accuracy drops a lot here vs its own reported mAP50 (0.786), that gap is the
domain gap - the model learned its own dataset, not "PPE in general".

The two datasets use different class ids, so we remap Construction-PPE's
ground-truth labels onto ppe_v1's class ids, keep only the 8 shared classes,
and run a proper `model.val()` restricted to those classes.

Outputs (observe/domain_gap_cppe/): val plots + a few annotated predictions.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
CPPE = ROOT / "datasets" / "construction-ppe"
EVAL = ROOT / "datasets" / "cppe_eval"           # remapped copy for evaluation
OUT = ROOT / "observe" / "domain_gap_cppe"
WEIGHTS = ROOT / "model" / "ppe_v1.pt"

# ppe_v1 class ids (14)
PPE_NAMES = {0: "Fall-Detected", 1: "Gloves", 2: "Goggles", 3: "Hardhat", 4: "Ladder",
             5: "Mask", 6: "NO-Gloves", 7: "NO-Goggles", 8: "NO-Hardhat", 9: "NO-Mask",
             10: "NO-Safety Vest", 11: "Person", 12: "Safety Cone", 13: "Safety Vest"}

# Construction-PPE id -> ppe_v1 id, for the 8 shared classes (others dropped)
REMAP = {0: 3, 1: 1, 2: 13, 4: 2, 6: 11, 7: 8, 8: 7, 9: 6}
SHARED = sorted(set(REMAP.values()))   # ppe_v1 ids we evaluate on


def build_eval_split():
    """Copy the test images and write GT labels remapped to ppe_v1 ids."""
    for sub in ("images/test", "labels/test"):
        (EVAL / sub).mkdir(parents=True, exist_ok=True)
    # images
    for img in (CPPE / "images" / "test").iterdir():
        shutil.copy2(img, EVAL / "images" / "test" / img.name)
    # labels (remap)
    kept_lines = dropped = 0
    for lab in (CPPE / "labels" / "test").glob("*.txt"):
        out_lines = []
        for line in lab.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            cid = int(parts[0])
            if cid in REMAP:
                out_lines.append(" ".join([str(REMAP[cid])] + parts[1:]))
                kept_lines += 1
            else:
                dropped += 1
        (EVAL / "labels" / "test" / lab.name).write_text("\n".join(out_lines))
    # data.yaml
    names_block = "\n".join(f"  {i}: {n}" for i, n in PPE_NAMES.items())
    (EVAL / "data.yaml").write_text(
        f"path: {EVAL.as_posix()}\ntrain: images/test\nval: images/test\ntest: images/test\n"
        f"names:\n{names_block}\n")
    print(f"eval split: kept {kept_lines} GT boxes (shared classes), dropped {dropped} (non-shared)")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    build_eval_split()

    model = YOLO(str(WEIGHTS))
    print(f"\nEvaluating ppe_v1 on Construction-PPE test ({len(list((EVAL/'images/test').iterdir()))} imgs), "
          f"restricted to shared classes {[PPE_NAMES[i] for i in SHARED]}\n")

    metrics = model.val(data=str(EVAL / "data.yaml"), split="test", classes=SHARED,
                        project=str(OUT), name="val", exist_ok=True, plots=True, verbose=True)

    # per-class table
    print("\n=== ppe_v1 on Construction-PPE (shared classes) ===")
    print(f"{'class':16} {'mAP50':>7} {'mAP50-95':>9}")
    for i in SHARED:
        # metrics.box.maps is indexed by class id; ap_class_index tells which have data
        try:
            ci = list(metrics.box.ap_class_index).index(i)
            ap50 = metrics.box.ap50[ci]
            ap = metrics.box.ap[ci]
            print(f"{PPE_NAMES[i]:16} {ap50:>7.3f} {ap:>9.3f}")
        except (ValueError, IndexError):
            print(f"{PPE_NAMES[i]:16} {'--':>7} {'--':>9}  (no GT/pred)")
    print(f"\nOVERALL  mAP50={metrics.box.map50:.3f}  mAP50-95={metrics.box.map:.3f}  "
          f"P={metrics.box.mp:.3f}  R={metrics.box.mr:.3f}")
    print("(ppe_v1 on its OWN dataset test set: mAP50=0.786, mAP50-95=0.524)")

    # a few annotated predictions for eyeballing
    (OUT / "preds").mkdir(exist_ok=True)
    sample = sorted((EVAL / "images" / "test").iterdir())[:8]
    for img in sample:
        r = model.predict(str(img), conf=0.25, classes=SHARED, verbose=False)[0]
        cv2.imwrite(str(OUT / "preds" / img.name), r.plot())
    print(f"\nSaved val plots + {len(sample)} annotated preds to {OUT}")


if __name__ == "__main__":
    main()
