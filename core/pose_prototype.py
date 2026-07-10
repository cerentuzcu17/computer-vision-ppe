"""
Pose prototype for the helmet-detection project.

The idea here is simple: before we can tell whether someone is wearing a
helmet, we first need to know where each person's HEAD actually is in the
frame. So this script runs YOLO's pose model on an image/video/webcam feed,
finds every person, looks at their head keypoints (nose, eyes, ears), and
draws a box around the head.

That head box is the thing we'll later compare against a detected helmet box
(using IoU) to decide "this helmet belongs to this person". The helmet model
isn't wired in yet - see the TODO next to helmet_iou() in geometry.py.

The actual detection/geometry logic lives in this folder's other modules:
  - entities.py  - the Person dataclass (and future entities like Helmet)
  - geometry.py  - head box math, IoU
  - drawing.py   - turning a Person into pixels/console output

This file is just the glue: load the model, read a source, run frames
through it, save/show the result.

How to run it:
    python pose_prototype.py --source sample.jpg
    python pose_prototype.py --source video.mp4
    python pose_prototype.py --source 0        (0 = webcam)
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np
from ultralytics import YOLO

from drawing import draw_person, print_person_summary
from entities import Person
from geometry import find_head_box


def extract_people(yolo_result) -> list[Person]:
    """Turn one YOLO prediction into a list of Person objects we can actually work with."""
    if yolo_result.keypoints is None or yolo_result.boxes is None:
        return []

    boxes = yolo_result.boxes.xyxy.cpu().numpy()
    all_keypoints = yolo_result.keypoints.data.cpu().numpy()  # shape: (num_people, 17, 3)

    people = []
    for person_id, (box, keypoints) in enumerate(zip(boxes, all_keypoints)):
        person_box = tuple(int(value) for value in box[:4])
        head_box, head_kpt_count, used_fallback = find_head_box(keypoints, person_box)

        people.append(Person(
            person_id=person_id,
            box=person_box,
            keypoints=keypoints,
            head_box=head_box,
            head_keypoints_found=head_kpt_count,
            used_fallback_box=used_fallback,
        ))

    return people


def process_frame(model: YOLO, frame: np.ndarray, confidence_threshold: float) -> np.ndarray:
    """Run the model on one frame and draw the results directly onto it."""
    result = model.predict(frame, conf=confidence_threshold, verbose=False)[0]
    people = extract_people(result)

    print(f"[frame] found {len(people)} people")
    for person in people:
        draw_person(frame, person)
        print_person_summary(person)

    return frame


# True if the source path's file extension looks like a still image, not a video.
def looks_like_an_image(source: str) -> bool:
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
    return source.lower().endswith(image_extensions)


def run(source: str, weights_path: str, confidence_threshold: float, save_path: str | None, show_window: bool) -> None:
    """Main entry point: figures out whether we're dealing with an image, a video, or a webcam, and handles each."""
    print(f"[model] loading weights from: {weights_path}")
    model = YOLO(weights_path)  # if the file isn't there yet, ultralytics downloads it for us

    if looks_like_an_image(source):
        _run_on_image(model, source, confidence_threshold, save_path, show_window)
    else:
        _run_on_video_or_webcam(model, source, confidence_threshold, save_path, show_window)


def _run_on_image(model: YOLO, source: str, confidence_threshold: float, save_path: str | None, show_window: bool) -> None:
    frame = cv2.imread(source)
    if frame is None:
        raise FileNotFoundError(f"couldn't read image: {source}")

    annotated = process_frame(model, frame, confidence_threshold)

    if save_path:
        cv2.imwrite(save_path, annotated)
        print(f"[saved] {save_path}")

    if show_window:
        cv2.imshow("pose_prototype", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def _run_on_video_or_webcam(model: YOLO, source: str, confidence_threshold: float, save_path: str | None, show_window: bool) -> None:
    # A plain digit like "0" means webcam index, anything else is a file path.
    capture = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not capture.isOpened():
        raise RuntimeError(f"couldn't open source: {source}")

    video_writer = None
    try:
        while True:
            got_frame, frame = capture.read()
            if not got_frame:
                break

            annotated = process_frame(model, frame, confidence_threshold)

            if save_path:
                if video_writer is None:
                    height, width = annotated.shape[:2]
                    fps = capture.get(cv2.CAP_PROP_FPS) or 25
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
                video_writer.write(annotated)

            if show_window:
                cv2.imshow("pose_prototype", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        if video_writer is not None:
            video_writer.release()
            print(f"[saved] {save_path}")
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find people and their head regions using YOLO-pose.")
    parser.add_argument("--source", required=True,
                         help="path to an image or video, or '0' to use the webcam")
    parser.add_argument("--weights", default="model/yolo26n-pose.pt",
                         help="YOLO-pose weights file (downloaded automatically into model/ if missing)")
    parser.add_argument("--conf", type=float, default=0.25,
                         help="minimum confidence for a person detection to count")
    parser.add_argument("--save", default=None,
                         help="where to write the annotated output (image or video)")
    parser.add_argument("--show", action="store_true",
                         help="pop up a window and show the result live")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.source, args.weights, args.conf, args.save, args.show)


if __name__ == "__main__":
    main()
