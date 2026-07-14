"""
Three-way comparison on the Construction-PPE test set (neutral ground):

  ppe_v1   : our trained 14-class PPE detector
  sh17     : SH17-trained YOLOv9-e (17 classes)   [NON-COMMERCIAL, eval only]
  hardhat  : third-party 2-class placeholder

Each model uses different class ids, so for every model we remap Construction-PPE's
ground-truth labels onto THAT model's ids (for the classes it supports) and run a
proper mAP eval restricted to those classes. Then we line the per-class mAP50 up
in one table. The one class all three share is 'helmet'.

Output: observe/three_way_cppe/  (table + a few side-by-side predictions)
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
CPPE = ROOT / "datasets" / "construction-ppe"
WORK = ROOT / "datasets" / "_threeway"
OUT = ROOT / "observe" / "three_way_cppe"

# semantic class -> Construction-PPE ground-truth id
SEM_TO_CPPE = {"helmet": 0, "gloves": 1, "vest": 2, "goggles": 4, "person": 6}
CPPE_TO_SEM = {v: k for k, v in SEM_TO_CPPE.items()}
SEMANTICS = ["helmet", "vest", "gloves", "goggles", "person"]

# per model: weights + semantic -> that model's class id
PPE_NAMES = {0: "Fall-Detected", 1: "Gloves", 2: "Goggles", 3: "Hardhat", 4: "Ladder",
             5: "Mask", 6: "NO-Gloves", 7: "NO-Goggles", 8: "NO-Hardhat", 9: "NO-Mask",
             10: "NO-Safety Vest", 11: "Person", 12: "Safety Cone", 13: "Safety Vest"}
MODELS = {
    "ppe_v1": {
        "weights": ROOT / "model" / "ppe_v1.pt",
        "names": PPE_NAMES,
        "map": {"helmet": 3, "gloves": 1, "vest": 13, "goggles": 2, "person": 11},
    },
    "sh17": {
        "weights": ROOT / "model" / "external" / "sh17_yolo9e.pt",
        "names": None,  # filled from the model
        "map": {"helmet": 10, "gloves": 9, "vest": 16, "goggles": 8, "person": 0},
    },
    "hardhat": {
        "weights": ROOT / "model" / "test" / "hardhat_yolov8n.pt",
        "names": {0: "Hardhat", 1: "NO-Hardhat"},
        "map": {"helmet": 0},
    },
}


def build_gt(model_name, cfg):
    """Copy test images + write GT labels remapped to this model's class ids."""
    root = WORK / model_name
    (root / "images" / "test").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "test").mkdir(parents=True, exist_ok=True)
    for img in (CPPE / "images" / "test").iterdir():
        dst = root / "images" / "test" / img.name
        if not dst.exists():
            shutil.copy2(img, dst)
    for lab in (CPPE / "labels" / "test").glob("*.txt"):
        out = []
        for line in lab.read_text().splitlines():
            p = line.split()
            if not p:
                continue
            sem = CPPE_TO_SEM.get(int(p[0]))
            if sem and sem in cfg["map"]:
                out.append(" ".join([str(cfg["map"][sem])] + p[1:]))
        (root / "labels" / "test" / lab.name).write_text("\n".join(out))
    names_block = "\n".join(f"  {i}: {n}" for i, n in cfg["names"].items())
    (root / "data.yaml").write_text(
        f"path: {root.as_posix()}\ntrain: images/test\nval: images/test\ntest: images/test\n"
        f"names:\n{names_block}\n")
    return root


def evaluate(model_name, cfg):
    model = YOLO(str(cfg["weights"]))
    if cfg["names"] is None:
        cfg["names"] = model.names
    root = build_gt(model_name, cfg)
    ids = sorted(cfg["map"].values())
    print(f"\n--- {model_name}: eval on {[s for s in cfg['map']]} ---")
    m = model.val(data=str(root / "data.yaml"), split="test", classes=ids,
                  project=str(OUT), name=f"val_{model_name}", exist_ok=True, plots=False, verbose=False)
    # per-semantic mAP50
    result = {}
    for sem, cid in cfg["map"].items():
        try:
            ci = list(m.box.ap_class_index).index(cid)
            result[sem] = float(m.box.ap50[ci])
        except (ValueError, IndexError):
            result[sem] = float("nan")
    result["_overall"] = float(m.box.map50)
    return model, result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    models = {}
    for name, cfg in MODELS.items():
        models[name], results[name] = evaluate(name, cfg)

    # ---- comparison table ----
    print("\n\n================  THREE-WAY mAP50 on Construction-PPE test  ================")
    hdr = f"{'class':10} " + " ".join(f"{n:>10}" for n in MODELS)
    print(hdr); print("-" * len(hdr))
    lines = ["THREE-WAY mAP50 on Construction-PPE test set (neutral ground)",
             "ppe_v1 = ours | sh17 = SH17 YOLOv9-e (NC, eval only) | hardhat = placeholder\n", hdr, "-" * len(hdr)]
    for sem in SEMANTICS:
        row = f"{sem:10} " + " ".join(
            (f"{results[n].get(sem, float('nan')):>10.3f}" if not np.isnan(results[n].get(sem, float('nan'))) else f"{'--':>10}")
            for n in MODELS)
        print(row); lines.append(row)
    ov = f"{'OVERALL':10} " + " ".join(f"{results[n]['_overall']:>10.3f}" for n in MODELS)
    print(ov); lines.append("-" * len(hdr)); lines.append(ov)
    note = "\nNote: 'overall' spans each model's supported classes (hardhat = helmet only)."
    print(note); lines.append(note)
    (OUT / "three_way_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    # ---- visual: 4 images, all three models side by side ----
    (OUT / "combined").mkdir(exist_ok=True)
    sample = sorted((CPPE / "images" / "test").iterdir())[:4]

    def panel(img, title):
        bar = np.zeros((44, img.shape[1], 3), np.uint8)
        cv2.putText(bar, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        return np.vstack([bar, img])

    for img_path in sample:
        frame = cv2.imread(str(img_path))
        panels = []
        for name in MODELS:
            ids = sorted(MODELS[name]["map"].values())
            r = models[name].predict(str(img_path), conf=0.25, classes=ids, verbose=False)[0]
            h = 600
            ann = cv2.resize(r.plot(), (int(frame.shape[1] * h / frame.shape[0]), h))
            panels.append(panel(ann, name))
        gap = np.full((panels[0].shape[0], 6, 3), 60, np.uint8)
        combo = panels[0]
        for p in panels[1:]:
            combo = np.hstack([combo, gap, p])
        cv2.imwrite(str(OUT / "combined" / img_path.name), combo)

    print(f"\nSaved table + {len(sample)} side-by-side images to {OUT}")


if __name__ == "__main__":
    main()
