# computer-vision-ppe

Per-person PPE (Personal Protective Equipment) detection. Two complementary
tracks:

1. **Pose + IoU matching** (`core/`) — detect every person, reduce them to
   pose keypoints, extract the head region, and match it against a helmet box
   via IoU to label each person "helmet / no helmet". Aimed at correctly
   assigning helmets to people in crowded scenes.
2. **Trained PPE detector** (`model/ppe_v1.pt`) — our own 14-class YOLO26 model
   (hardhat, vest, gloves, goggles, mask, … + `NO-` variants) that detects PPE
   directly. See its [model card](model/ppe_v1.md).

The plan is to combine them: the trained detector finds PPE boxes, pose gives
per-person body regions, and the matcher assigns each PPE item to a person.

## Folder structure

```
.
├── core/
│   ├── entities.py         # Detected "things": Person now, Helmet later
│   ├── geometry.py         # Pure math: head box from keypoints, IoU
│   ├── drawing.py          # Turning a Person into pixels / console output
│   └── pose_prototype.py   # CLI glue: load model, read source, run frames
├── model/                  # Model weights (.pt via Git LFS)
│   ├── yolo26n-pose.pt     #   pose model (our standard)
│   ├── ppe_v1.pt           #   our trained 14-class PPE detector (see ppe_v1.md)
│   └── test/               #   third-party placeholder weights (hardhat)
├── train/                  # PPE model training: pipeline, run records, metrics
│   └── runs/run-1/         #   record behind ppe_v1.pt (weights, curves, args)
├── test/samples/v1/        # Committed inference test photos (sanity checks)
├── dataset/                # Raw input media (not committed, see .gitignore)
├── observe/                # Run outputs: processed images/videos, logs (not committed)
└── ui/                     # Future: visualization / dashboard layer (empty for now)
```

This split exists so each piece can grow in its own folder: algorithm
(`core`), weights (`model`), model training (`train`), fixed test media
(`test`), scratch data (`dataset`), result tracking (`observe`), presentation
(`ui`).

## Environment — uv

The project is managed with [uv](https://docs.astral.sh/uv/). The Python
version is pinned in `.python-version` (**3.12.5**).

**Prerequisite — Git LFS.** The model weights (`model/*.pt`) are versioned with
[Git LFS](https://git-lfs.com/), so you must have it installed *before* you
clone/pull, or you'll get tiny pointer files instead of the real weights:

```bash
# One-time per machine, then clone (or re-pull) the repo
git lfs install
git lfs pull        # if you already cloned before installing LFS
```

```bash
# Install dependencies (from pyproject.toml + uv.lock, .venv is created automatically)
uv sync

# Run commands inside the .venv
uv run core/pose_prototype.py --source dataset/sample.jpg --save observe/sample_out.jpg
```

### Dependencies

- `ultralytics` — YOLO-pose inference (see the model weight note below)
- `opencv-python` — image/video I/O, drawing
- `numpy` — geometry math

**Model weights:** the project standardizes on the **YOLO26** family (the
successor to YOLO11, NMS-free / `end2end` inference — a good fit for CPU).
`yolo26n-pose.pt` is committed via Git LFS, so `git lfs pull` gives you the
exact same weight every other dev has — no per-machine download needed. (If
the file is ever missing, `ultralytics` will also auto-download it into
`model/` on first run.) For a more accurate but slower model, use
`--weights model/yolo26s-pose.pt`.

Any other computer-vision component added to this repo later (e.g. the
helmet detector) should default to a `yolo26*` weight too, so the whole
pipeline stays on one consistent model generation.

### Trained PPE model — `ppe_v1.pt`

Our own 14-class PPE detector (YOLO26s), the project's first in-house weight.
Full details, metrics, and provenance are in its
[model card](model/ppe_v1.md); the training pipeline and run record live in
[`train/`](train/README.md).

### Test hardhat model (placeholder, superseded)

Before `ppe_v1.pt` existed, `model/test/hardhat_yolov8n.pt` was the placeholder
used to validate the head-box-vs-helmet-box matching logic end to end. It's now
superseded by `ppe_v1.pt` (which covers hardhats and much more) but kept as a
lightweight 2-class baseline. It lives under `model/test/`, separate from our
own weights, so it's obvious at a glance which is a borrowed stand-in:

- Source: [keremberke/yolov8n-hard-hat-detection](https://huggingface.co/keremberke/yolov8n-hard-hat-detection)
  (YOLOv8n, trained on Roboflow's "Hard Hats" dataset — ~19.7k images)
- Classes: `Hardhat`, `NO-Hardhat`
- Reported mAP@0.5: 0.836
- Loads with plain `ultralytics.YOLO(...)` — no extra dependency needed,
  despite the model card showing the `ultralyticsplus` wrapper

Shared via Git LFS, so `git lfs pull` fetches it with the rest of the repo.
If you ever need to re-fetch it straight from the source:

```bash
curl -L -o model/test/hardhat_yolov8n.pt \
  https://huggingface.co/keremberke/yolov8n-hard-hat-detection/resolve/main/best.pt
```

This is explicitly a **test/placeholder** weight, not the model we'd ship:
it's a different YOLO generation (v8, not v26) and its license isn't
confirmed. Treat it as good enough to validate our own code, not as a
production hardhat detector.

## Running (core/pose_prototype.py)

```bash
# Image
uv run core/pose_prototype.py --source dataset/sample.jpg --save observe/sample_out.jpg

# Video
uv run core/pose_prototype.py --source dataset/video.mp4 --save observe/video_out.mp4

# Webcam
uv run core/pose_prototype.py --source 0 --show
```

Arguments:

| Flag         | Description                                            | Default                   |
|--------------|---------------------------------------------------------|----------------------------|
| `--source`   | Image/video path, or `0` for webcam                     | *(required)*               |
| `--weights`  | YOLO-pose weights                                        | `model/yolo26n-pose.pt`   |
| `--conf`     | Detection confidence threshold                           | `0.25`                     |
| `--save`     | Output file (image/video)                                | none                        |
| `--show`     | Show in a window                                          | off                         |

## Status

- [x] Person + keypoint detection with YOLO-pose
- [x] Head box from head keypoints (fallback: top 25% of the person box)
- [x] `helmet_iou()` skeleton (helmet model to be added later)
- [x] Test hardhat model picked and verified (`model/hardhat_yolov8n.pt`, see above)
- [ ] Helmet detection + head-to-helmet matching wired into the pipeline
- [ ] `ui/` layer
