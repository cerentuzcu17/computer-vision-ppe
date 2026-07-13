"""
YOLO26 training script for the PPE (Personal Protective Equipment) dataset.
Run from the project root, e.g.:

    python train/train.py --model yolo26s.pt --epochs 100 --imgsz 640 --batch 32

GPU is required: the script aborts if CUDA is not visible to torch, rather
than silently falling back to CPU.

See train/README.md for the dataset source and how to reproduce best.pt
(training outputs, including weights, are not committed to this repo).
"""
import argparse
import os
import sys

import torch
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_YAML = os.path.join(ROOT, "dataset", "data.yaml")


def assert_gpu():
    if not torch.cuda.is_available():
        sys.exit(
            "CUDA GPU not available to torch. Training aborted so it doesn't "
            "silently fall back to CPU. Check `nvidia-smi` and that your torch "
            "build matches your driver's CUDA version."
        )
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"Using GPU 0: {name} (compute capability {cap[0]}.{cap[1]})")


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLO26 on the PPE dataset")
    p.add_argument("--model", default="yolo26s.pt",
                    help="yolo26n.pt | yolo26s.pt | yolo26m.pt | yolo26l.pt | yolo26x.pt "
                         "(or a .yaml to train from scratch, or a prior .pt checkpoint to resume/fine-tune)")
    p.add_argument("--data", default=DATA_YAML, help="path to data.yaml")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=32, help="use -1 for Ultralytics auto batch size")
    p.add_argument("--workers", type=int, default=4,
                    help="dataloader workers; lower this (or set 0) if you see "
                         "'DataLoader worker exited unexpectedly' on Windows")
    p.add_argument("--patience", type=int, default=25, help="early stopping patience (epochs)")
    p.add_argument("--project", default=os.path.join(ROOT, "runs", "train"))
    p.add_argument("--name", default="ppe_yolo26")
    p.add_argument("--resume", action="store_true", help="resume from the last checkpoint in --name")
    return p.parse_args()


def main():
    args = parse_args()
    assert_gpu()

    if args.resume:
        ckpt = os.path.join(args.project, args.name, "weights", "last.pt")
        if not os.path.isfile(ckpt):
            sys.exit(f"--resume given but no checkpoint found at {ckpt}")
        print(f"Resuming from {ckpt}")
        model = YOLO(ckpt)
        model.train(resume=True, device=0)  # rest of the args come from the checkpoint's saved args.yaml
    else:
        model = YOLO(args.model)
        model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            patience=args.patience,
            device=0,          # force GPU 0; never "cpu"
            amp=True,           # mixed precision, uses tensor cores + saves VRAM
            project=args.project,
            name=args.name,
            plots=True,
        )

    metrics = model.val(data=args.data, device=0)
    print(metrics.box.maps)


if __name__ == "__main__":
    main()
