# train/

Training pipeline for a 14-class PPE object detector (YOLO26), separate from
`core/pose_prototype.py`'s pose + IoU-matching approach. This directory holds
the training script and a record of past runs; it does not hold the dataset
or model weights (see below).

## Dataset

Roboflow export, **44,002 images**, YOLO format, pre-split train/valid/test
(30765 / 8814 / 4423), CC BY 4.0:
https://app.roboflow.com/cerens-workspace-rpo0u/personal-protective-equipment-combined-model-o7hds

Classes (14): `Fall-Detected`, `Gloves`, `Goggles`, `Hardhat`, `Ladder`,
`Mask`, `NO-Gloves`, `NO-Goggles`, `NO-Hardhat`, `NO-Mask`, `NO-Safety Vest`,
`Person`, `Safety Cone`, `Safety Vest`.

Not committed to this repo (same policy as `dataset/` and `model/` — see the
root `.gitignore`). To reproduce a run, download the export above into
`dataset/` at the project root so you have:

```
dataset/
├── data.yaml
├── train/{images,labels}
├── valid/{images,labels}
└── test/{images,labels}
```

## Reproducing a model

Requires a CUDA GPU (the script hard-fails rather than falling back to CPU)
and `torch` + `ultralytics` installed for your CUDA version.

```bash
python train/train.py --model yolo26s.pt --epochs 100 --imgsz 640 --batch 32
```

Outputs (weights + full plots/logs) go to `runs/train/<name>/` at the project
root — also not committed; only the lightweight metrics below are kept here
as a record.

## Past runs

Trained on an RTX 5090, `yolo26s.pt` base, 640px, batch 32.

| | test-1 | test-2 |
|---|---|---|
| Epochs | 100 (full run) | 100 requested, early-stopped at 100 (best: epoch 75) |
| mAP50 | 0.786 | 0.787 |
| mAP50-95 | 0.524 | 0.523 |
| Precision | 0.720 | 0.718 |
| Recall | 0.837 | 0.842 |

Both runs converge to the same result (training is seeded/deterministic).
Strongest classes: `Ladder`, `Goggles`, `Person`, `Hardhat` (mAP50 > 0.9).
Weakest: `NO-Safety Vest` (~0.22), `Mask` (~0.54), `Safety Cone` (~0.71) —
likely candidates for more data, augmentation, or class-balancing if this
model is taken further.

Each `test-N/` folder holds `args.yaml` (full run config), `results.csv`
(per-epoch metrics), `results.png`, `confusion_matrix(_normalized).png`, and
the box precision/recall/F1 curves. Weights (`best.pt`/`last.pt`) are not
included — regenerate them by running `train.py` with the args in
`test-N/args.yaml`.
