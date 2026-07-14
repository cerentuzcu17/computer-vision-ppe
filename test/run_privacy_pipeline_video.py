"""
Privacy-aware PPE pipeline on VIDEO, with tracking + temporal voting.

Per frame:  pose.track (stable track_id)  ->  helmet/goggle verdict  ->
temporal vote over the track's recent frames  ->  stabilized verdict, drawn on
the anonymized frame. One noisy frame can't flip a track's result anymore.

Run:  uv run test/run_privacy_pipeline_video.py --source path/to/video.mp4 --save out.mp4
      uv run test/run_privacy_pipeline_video.py --source 0        (webcam)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "core"))
from ppe_pipeline import PpePipeline    # noqa: E402
from tracking import TemporalVoter      # noqa: E402
from drawing import draw_person         # noqa: E402
from entities import YES, NO, UNKNOWN   # noqa: E402

POSE_W = ROOT / "model" / "yolo26n-pose.pt"
DETECTOR_W = ROOT / "model" / "external" / "sh17_yolo9e.pt"
COLOR = {YES: (0, 200, 0), NO: (0, 0, 230), UNKNOWN: (150, 150, 150)}


def draw(canvas, result, stable):
    draw_person(canvas, result.person)
    p = result.person
    cv2.rectangle(canvas, p.head_box[:2], p.head_box[2:], COLOR[stable.helmet], 3)
    cv2.rectangle(canvas, result.face_box[:2], result.face_box[2:], COLOR[stable.goggle], 2)
    tid = result.track_id if result.track_id is not None else "?"
    cv2.putText(canvas, f"#{tid} helmet:{stable.helmet} goggle:{stable.goggle}",
                (p.box[0], max(p.box[1] - 8, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                COLOR[stable.helmet], 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="video file, or '0' for webcam")
    ap.add_argument("--save", default=None, help="output video path")
    ap.add_argument("--window", type=int, default=15, help="temporal vote window (frames)")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--log-track", type=int, default=None, help="print raw vs voted for this track_id")
    args = ap.parse_args()

    pipeline = PpePipeline(str(POSE_W), str(DETECTOR_W), anonymize_method="mask")
    voter = TemporalVoter(window=args.window)

    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    if not cap.isOpened():
        raise SystemExit(f"couldn't open source: {args.source}")

    writer = None
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        anonymized, results = pipeline.process(frame, track=True)
        for r in results:
            stable = voter.vote(r.track_id, r.ppe)
            draw(anonymized, r, stable)
            if args.log_track is not None and r.track_id == args.log_track:
                print(f"frame {frame_idx:3d} track#{r.track_id}: "
                      f"raw helmet={r.ppe.helmet:<7} -> voted={stable.helmet}")
        if args.save:
            if writer is None:
                h, w = anonymized.shape[:2]
                fps = cap.get(cv2.CAP_PROP_FPS) or 12
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(anonymized)
        if args.show:
            cv2.imshow("ppe", anonymized)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
        print(f"[saved] {args.save}")
    cv2.destroyAllWindows()
    print(f"processed {frame_idx} frames")


if __name__ == "__main__":
    main()
