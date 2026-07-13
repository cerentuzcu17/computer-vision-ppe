# PPE Detector — `ppe_v1.pt`

![type](https://img.shields.io/badge/task-detect-blue)
![base](https://img.shields.io/badge/base-YOLO26s-informational)
![classes](https://img.shields.io/badge/classes-14-blueviolet)
![mAP50](https://img.shields.io/badge/mAP@50-0.786-success)
![mAP50-95](https://img.shields.io/badge/mAP@50--95-0.524-yellowgreen)
![precision](https://img.shields.io/badge/precision-0.720-green)
![recall](https://img.shields.io/badge/recall-0.837-green)
![license](https://img.shields.io/badge/dataset-CC--BY--4.0-lightgrey)

Our **first in-house trained model** (v1). A 14-class Personal Protective
Equipment object detector — the project's own weight, as opposed to the pose
model (`yolo26n-pose.pt`) and the third-party placeholder
(`test/hardhat_yolov8n.pt`). It supersedes the placeholder hardhat detector:
it covers `Hardhat`/`NO-Hardhat` **and** vests, gloves, goggles, masks, etc.

## Classes (14)

`Fall-Detected`, `Gloves`, `Goggles`, `Hardhat`, `Ladder`, `Mask`,
`NO-Gloves`, `NO-Goggles`, `NO-Hardhat`, `NO-Mask`, `NO-Safety Vest`,
`Person`, `Safety Cone`, `Safety Vest`.

## Metrics (validation)

| mAP@50 | mAP@50-95 | Precision | Recall |
|--------|-----------|-----------|--------|
| 0.786  | 0.524     | 0.720     | 0.837  |

Strongest classes (mAP@50 > 0.9): `Ladder`, `Goggles`, `Person`, `Hardhat`.
Weakest: `NO-Safety Vest` (~0.22), `Mask` (~0.54), `Safety Cone` (~0.71) —
candidates for more data / augmentation in a future version.

## Provenance

- **Base:** `yolo26s.pt`, 640px, batch 32, 100 epochs (RTX 5090).
- **Dataset:** Roboflow "PPE combined model", 44,002 images (30765/8814/4423),
  CC BY 4.0.
- **Training run:** [`train/runs/run-1/`](../train/runs/run-1) holds the full
  record (`args.yaml`, `results.csv`, curves, confusion matrix). This weight is
  a copy of that run's `best.pt`, promoted here as the labelled production v1.
- **Reproduce:** see [`train/README.md`](../train/README.md).

## Usage

```python
from ultralytics import YOLO
model = YOLO("model/ppe_v1.pt")
results = model.predict("test/samples/v1/", conf=0.25)
```

Weight is tracked with Git LFS — run `git lfs install` before cloning/pulling
or you'll get a pointer file instead of the real weight.
