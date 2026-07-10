# computer-vision-ppe

Per-person helmet overlap detection. Detects every person in an image,
reduces them to pose keypoints, extracts the head region, and matches that
head region against a helmet box via IoU (Intersection-over-Union) to label
each person "helmet / no helmet". Goal: correctly assign helmets to people
in crowded scenes.

## Folder structure

```
.
├── core/
│   ├── entities.py         # Detected "things": Person now, Helmet later
│   ├── geometry.py         # Pure math: head box from keypoints, IoU
│   ├── drawing.py          # Turning a Person into pixels / console output
│   └── pose_prototype.py   # CLI glue: load model, read source, run frames
├── dataset/    # Sample/test input media (not committed, see .gitignore)
├── model/      # Model weights — .pt files auto-download here (not committed)
│   └── test/   # Third-party placeholder weights, separate from our own (not committed)
├── observe/    # Run outputs: processed images/videos, logs (not committed)
└── ui/         # Future: visualization / dashboard layer (empty for now)
```

This split exists so each piece can grow in its own folder as the helmet
model and UI are added: algorithm (`core`), data (`dataset`/`model`), result
tracking (`observe`), presentation (`ui`).

## Environment — uv

The project is managed with [uv](https://docs.astral.sh/uv/). The Python
version is pinned in `.python-version` (**3.12.5**).

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
The `ultralytics` package auto-downloads `yolo26n-pose.pt` on first run and
saves it into `model/`. No separate pip dependency is needed — just internet
access and write permission on `model/`. For a more accurate but slower
model, use `--weights model/yolo26s-pose.pt`.

Any other computer-vision component added to this repo later (e.g. the
helmet detector) should default to a `yolo26*` weight too, so the whole
pipeline stays on one consistent model generation.

### Test hardhat model

Until we train our own, `model/test/hardhat_yolov8n.pt` is the placeholder we
use to test the head-box-vs-helmet-box matching logic end to end. It lives
under `model/test/`, separate from `model/yolo26n-pose.pt`, so it's obvious
at a glance which weight is "ours" (the pose model we standardized on) and
which is a borrowed stand-in we're only using to validate the pipeline:

- Source: [keremberke/yolov8n-hard-hat-detection](https://huggingface.co/keremberke/yolov8n-hard-hat-detection)
  (YOLOv8n, trained on Roboflow's "Hard Hats" dataset — ~19.7k images)
- Classes: `Hardhat`, `NO-Hardhat`
- Reported mAP@0.5: 0.836
- Loads with plain `ultralytics.YOLO(...)` — no extra dependency needed,
  despite the model card showing the `ultralyticsplus` wrapper

Not committed (see `.gitignore`) — download it once per machine:

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
