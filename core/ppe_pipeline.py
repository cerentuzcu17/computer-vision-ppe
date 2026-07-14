"""
The privacy-aware PPE pipeline, end to end.

    frame
      -> pose            (person detection + body keypoints, one pass)
      -> ROIs            (head_box for helmet, face_box for goggle, from keypoints)
      -> PPE detection   (helmet + goggle boxes)
      -> match           (helmet<->head_box, goggle<->face_box; goggle gated on a frontal face)
      -> anonymize       (destroy the face region AFTER detection)
      -> (anonymized frame, per-person PPE verdicts)

Ordering note: goggles live on the face, so we cannot remove the face *before*
detection - we detect first, then immediately anonymize. The raw face therefore
only exists in memory during inference; process() never mutates the input frame
and returns an already-anonymized copy, so the caller can drop the raw frame.
"""
from __future__ import annotations

from ultralytics import YOLO

from entities import Person, PersonResult, PpeStatus, YES, NO, UNKNOWN
from geometry import find_head_box, find_face_box, is_face_frontal, best_match
from privacy import anonymize_face

# SH17 class ids (the detector we use for now). helmet=10, glasses=8.
SH17_HELMET_CLASS = 10
SH17_GOGGLE_CLASS = 8


class PpePipeline:
    def __init__(self, pose_weights: str, detector_weights: str,
                 helmet_class_id: int = SH17_HELMET_CLASS, goggle_class_id: int = SH17_GOGGLE_CLASS,
                 conf: float = 0.25, anonymize_method: str = "mask"):
        self.pose = YOLO(pose_weights)
        self.detector = YOLO(detector_weights)
        self.helmet_class_id = helmet_class_id
        self.goggle_class_id = goggle_class_id
        self.conf = conf
        self.anonymize_method = anonymize_method

    def process(self, frame, track: bool = False):
        """Run the whole pipeline on one BGR frame. Returns (anonymized_frame, list[PersonResult]).

        track=True uses Ultralytics' built-in tracker so each person carries a
        stable track_id across frames (needed for temporal voting on video).
        """
        persons = self._detect_people(frame, track=track)
        helmet_boxes, goggle_boxes = self._detect_ppe(frame)

        results = []
        for person, face_box, face_is_guess, track_id in persons:
            ppe = PpeStatus()
            self._judge_helmet(ppe, person, helmet_boxes)
            self._judge_goggle(ppe, person, face_box, face_is_guess, goggle_boxes)
            results.append(PersonResult(person, face_box, ppe, track_id))

        # Privacy: destroy every face AFTER we're done detecting on the raw frame.
        anonymized = frame.copy()
        for result in results:
            anonymize_face(anonymized, result.face_box, self.anonymize_method)

        return anonymized, results

    # --- stages -------------------------------------------------------------

    def _detect_people(self, frame, track: bool = False):
        """Pose pass -> (Person, face_box, face_is_guess, track_id) per detected person."""
        if track:
            result = self.pose.track(frame, conf=self.conf, persist=True, verbose=False)[0]
        else:
            result = self.pose.predict(frame, conf=self.conf, verbose=False)[0]
        people = []
        if result.keypoints is None or result.boxes is None:
            return people
        boxes = result.boxes.xyxy.cpu().numpy()
        keypoints = result.keypoints.data.cpu().numpy()
        track_ids = result.boxes.id.int().cpu().tolist() if result.boxes.id is not None else None
        for i, (box, kp) in enumerate(zip(boxes, keypoints)):
            person_box = tuple(int(v) for v in box[:4])
            head_box, n_kpts, head_is_guess = find_head_box(kp, person_box)
            face_box, face_is_guess = find_face_box(kp, person_box)
            track_id = track_ids[i] if track_ids is not None else None
            # id doubles as the person label when tracking; else just the index
            person = Person(track_id if track_id is not None else i,
                            person_box, kp, head_box, n_kpts, head_is_guess)
            people.append((person, face_box, face_is_guess, track_id))
        return people

    def _detect_ppe(self, frame):
        """One detector pass -> (helmet_boxes, goggle_boxes)."""
        result = self.detector.predict(
            frame, conf=self.conf, classes=[self.helmet_class_id, self.goggle_class_id], verbose=False)[0]
        helmets, goggles = [], []
        if result.boxes is not None:
            for b in result.boxes:
                box = tuple(int(v) for v in b.xyxy[0].tolist())
                cid = int(b.cls[0])
                if cid == self.helmet_class_id:
                    helmets.append(box)
                elif cid == self.goggle_class_id:
                    goggles.append(box)
        return helmets, goggles

    # --- per-person verdicts ------------------------------------------------

    def _judge_helmet(self, ppe, person, helmet_boxes):
        idx, iou = best_match(person.head_box, helmet_boxes)
        if idx >= 0:
            ppe.helmet, ppe.helmet_iou = YES, iou
        else:
            # Head was located from real keypoints -> we can see there's no helmet.
            # Head was only guessed -> we can't be sure -> unknown.
            ppe.helmet = UNKNOWN if person.used_fallback_box else NO

    def _judge_goggle(self, ppe, person, face_box, face_is_guess, goggle_boxes):
        # Can't judge goggles on a face we can't see head-on.
        if face_is_guess or not is_face_frontal(person.keypoints):
            ppe.goggle = UNKNOWN
            return
        idx, iou = best_match(face_box, goggle_boxes)
        if idx >= 0:
            ppe.goggle, ppe.goggle_iou = YES, iou
        else:
            ppe.goggle = NO
