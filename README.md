# computer-vision-ppe

Per-person helmet overlap detection. Detects every person in an image,
reduces them to pose keypoints, extracts the head region, and matches that
head region against a helmet box via IoU (Intersection-over-Union) to label
each person "helmet / no helmet". Goal: correctly assign helmets to people
in crowded scenes.

## Folder structure

```
.
├── core/       # Algorithm logic: pose inference, head box, IoU (pose_prototype.py)
├── dataset/    # Sample/test input media (not committed, see .gitignore)
├── model/      # Model weights — .pt files auto-download here (not committed)
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

**Model weights:** the `ultralytics` package auto-downloads `yolo11n-pose.pt`
(or larger variants like `yolo11s-pose.pt`) on first run and saves it into
`model/`. No separate pip dependency is needed — just internet access and
write permission on `model/`. For a more accurate but slower model, use
`--weights model/yolo11s-pose.pt`.

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
| `--weights`  | YOLO-pose weights                                        | `model/yolo11n-pose.pt`   |
| `--conf`     | Detection confidence threshold                           | `0.25`                     |
| `--save`     | Output file (image/video)                                | none                        |
| `--show`     | Show in a window                                          | off                         |

## Status

- [x] Person + keypoint detection with YOLO-pose
- [x] Head box from head keypoints (fallback: top 25% of the person box)
- [x] `helmet_iou()` skeleton (helmet model to be added later)
- [ ] Helmet detection + head-to-helmet matching
- [ ] `ui/` layer
