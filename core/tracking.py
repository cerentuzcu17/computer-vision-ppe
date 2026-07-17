"""
Temporal voting: stabilize a track's PPE verdict over time.

A single frame's verdict is noisy - a helmet can flicker to "no" for one blurry
frame. Tracking (Ultralytics' built-in ByteTrack/BoTSORT via model.track) gives
each person a stable track_id across frames; this module keeps a sliding window
of that track's recent verdicts and returns the majority, so one bad frame can't
flip the result.

Rules:
  - Only YES / NO votes count; UNKNOWN is ignored (can't-judge shouldn't sway it).
  - A track with no YES/NO votes yet, or a tie, stays UNKNOWN.
  - track_id None (e.g. stills, or a frame the tracker didn't id) -> pass through.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque

from entities import PpeStatus, YES, NO, UNKNOWN

DEFAULT_WINDOW = 15


def _majority(votes) -> str:
    counts = Counter(v for v in votes if v in (YES, NO))
    if not counts:
        return UNKNOWN
    ranked = counts.most_common()
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return UNKNOWN                      # tie -> don't commit
    return ranked[0][0]


class TemporalVoter:
    """Per-track sliding-window majority vote over PPE verdicts."""

    def __init__(self, window: int = DEFAULT_WINDOW):
        self.window = window
        self._helmet: dict[int, deque] = defaultdict(lambda: deque(maxlen=window))
        self._goggle: dict[int, deque] = defaultdict(lambda: deque(maxlen=window))

    def vote(self, track_id: int | None, ppe: PpeStatus) -> PpeStatus:
        """Record this frame's raw verdict for the track and return the stabilized one."""
        if track_id is None:
            return ppe                      # nothing to vote over
        self._helmet[track_id].append(ppe.helmet)
        self._goggle[track_id].append(ppe.goggle)
        return PpeStatus(
            helmet=_majority(self._helmet[track_id]),
            goggle=_majority(self._goggle[track_id]),
            helmet_iou=ppe.helmet_iou,
            goggle_iou=ppe.goggle_iou,
        )

    def drop(self, track_id: int) -> None:
        """Forget a track once it leaves the scene (free memory)."""
        self._helmet.pop(track_id, None)
        self._goggle.pop(track_id, None)
