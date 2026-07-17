# computer-vision-ppe

Per-person PPE (Personal Protective Equipment) detection, built privacy-first:
a person's face is only ever processed in memory, long enough to detect their
safety equipment, and is destroyed (anonymized) before any frame is stored,
shown, or streamed.

## Pipeline (`core/ppe_pipeline.py`)

```
frame
  → pose             one pass: person detection + body keypoints (+ track_id)
  → ROIs             head_box (helmet region) + face_box (goggle region),
                      both derived from the pose face keypoints
  → PPE detection    helmet + goggle boxes (currently: SH17 model, see below)
  → match            center-in-ROI + IoU tie-break, per person
  → anonymize        face region destroyed AFTER detection (core/privacy.py)
  → (optional) track + temporal vote to stabilize the verdict across frames
```

Each person gets a `helmet`/`goggle` verdict of **yes / no / unknown** — never
collapsed to a boolean. `unknown` means "couldn't judge" (head turned away,
keypoints unreliable), which is deliberately different from `no` ("judged and
absent") to avoid false alarms. See [`core/entities.py`](core/entities.py)
(`PpeStatus`) and [`core/tracking.py`](core/tracking.py) (`TemporalVoter`) for
how a track's verdict is stabilized over a sliding window of recent frames
instead of trusting any single frame.

We also have a separate, from-scratch **trained PPE detector**
(`model/ppe_v1.pt`, 14 classes) — see its [model card](model/ppe_v1.md) and
[`train/`](train/README.md). It is not yet wired into the main pipeline above
(current detector is SH17, see below); domain-gap evaluation against it lives
in [`test/eval/`](test/eval).

## Folder structure

```
.
├── core/
│   ├── entities.py             # Person, PpeStatus (yes/no/unknown), PersonResult
│   ├── geometry.py             # head/face ROIs from keypoints, IoU, best_match
│   ├── drawing.py               # pose skeleton + keypoints -> pixels
│   ├── privacy.py               # anonymize_face: irreversible mask / blur
│   ├── tracking.py              # TemporalVoter: per-track majority vote
│   ├── ppe_pipeline.py          # PpePipeline: wires the whole flow together
│   └── pose_prototype.py        # standalone pose-only CLI (pre-PPE prototype)
├── model/                       # Model weights (.pt via Git LFS)
│   ├── yolo26n-pose.pt          #   pose model (our standard)
│   ├── ppe_v1.pt                #   our trained 14-class PPE detector (see ppe_v1.md)
│   ├── test/                    #   third-party placeholder weights (hardhat, LFS)
│   └── external/                #   non-commercial eval-only weights (e.g. SH17, gitignored)
├── train/                       # PPE model training: pipeline, run records, metrics
│   └── runs/run-1/              #   record behind ppe_v1.pt (weights, curves, args)
├── test/
│   ├── samples/v1/               # committed inference test photos (sanity checks)
│   ├── run_privacy_pipeline.py   # run the full pipeline over samples/v1 (stills)
│   ├── run_privacy_pipeline_video.py  # video: tracking + temporal voting
│   ├── pipeline_match*.py        # earlier pose+ROI+IoU matcher iterations
│   └── eval/                     # one-off studies: model comparisons, domain-gap
├── dataset/                     # Raw input media (not committed, see .gitignore)
├── datasets/                    # Downloaded eval datasets, e.g. Construction-PPE (gitignored)
├── observe/                     # Run outputs: processed images/videos, logs (not committed)
└── ui/                          # Future: visualization / dashboard layer (empty for now)
```

This split exists so each piece can grow in its own folder: algorithm
(`core`), weights (`model`), model training (`train`), fixed test media +
runners (`test`), scratch/downloaded data (`dataset`/`datasets`), result
tracking (`observe`), presentation (`ui`).

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

- `ultralytics` — YOLO-pose inference and detection (see the model weight note below)
- `opencv-python` — image/video I/O, drawing
- `numpy` — geometry math
- `lap` — linear assignment solver required by Ultralytics' built-in tracker
  (`pose.track(...)`, used for the video pipeline's `track_id` + temporal voting)

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

### Current PPE detector — SH17 (eval only, not for production)

The main pipeline currently detects helmets and goggles with a pretrained
**SH17 YOLOv9-e** model (`model/external/sh17_yolo9e.pt`, not committed — see
[SH17dataset](https://github.com/ahmadmughees/SH17dataset)). It generalizes
far better than `ppe_v1.pt` to real-world scenes (domain-gap study in
[`test/eval/`](test/eval)), but its dataset/weights are
**CC BY-NC-SA 4.0 — non-commercial**. It's there to validate the pipeline
end to end; it must be swapped for a commercially-licensed or in-house model
before this ships as a product. Download it yourself into `model/external/`
if you need to reproduce results (path is gitignored).

## Running

```bash
# Full privacy-aware pipeline (pose + helmet + goggle + anonymization) on stills
uv run test/run_privacy_pipeline.py

# Same, on video, with tracking + temporal voting
uv run test/run_privacy_pipeline_video.py --source path/to/video.mp4 --save out.mp4
uv run test/run_privacy_pipeline_video.py --source 0 --show     # webcam

# Standalone pose-only prototype (no PPE, no privacy layer)
uv run core/pose_prototype.py --source dataset/sample.jpg --save observe/sample_out.jpg
```

`run_privacy_pipeline_video.py` flags: `--window` (temporal-vote frame window,
default 15), `--log-track <id>` (print raw-vs-voted verdict for one track),
`--show` (live window).

## Status

- [x] Person + keypoint detection with YOLO-pose
- [x] Head box (helmet ROI) and face box (goggle ROI) from keypoints
- [x] Helmet + goggle detection wired into the pipeline (ROI + IoU matching)
- [x] Privacy: face anonymized (irreversible mask) immediately after PPE detection
- [x] Tracking (Ultralytics built-in) + temporal voting to stabilize verdicts on video
- [x] In-house trained PPE detector `ppe_v1.pt` (14 classes) — not yet the
      pipeline's active detector; domain-gap studied against SH17/Construction-PPE
- [ ] Replace the SH17 (non-commercial) detector with a shippable one
- [ ] `ui/` layer
