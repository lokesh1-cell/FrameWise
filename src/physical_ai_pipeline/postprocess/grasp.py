"""Grasp quality scoring and interaction state machine (annotation only).

Per hand (left / right / unknown), each kept frame is labelled:

* IDLE        - no object near the hand.
* APPROACHING - nearest object within ``approach_distance`` hand-diagonals
                and the distance is shrinking, no contact yet.
* GRASPING    - contact (hand-object IoU >= ``contact_iou``) for fewer than
                ``hold_after_frames`` frames.
* HOLDING     - sustained contact.
* RELEASING   - contact lost after GRASPING/HOLDING, for up to
                ``release_grace_frames`` frames.

A manipulation ActionSegment spans from the first APPROACHING/GRASPING frame
to the end of RELEASING (or the last contact frame).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from physical_ai_pipeline.config import GraspConfig
from physical_ai_pipeline.frame import FrameRecord, GraspInfo, Hand
from physical_ai_pipeline.inference.coco import PERSON_CLASS_ID
from physical_ai_pipeline.inference.geometry import center, diagonal, iou


class InteractionState(StrEnum):
    IDLE = "IDLE"
    APPROACHING = "APPROACHING"
    GRASPING = "GRASPING"
    HOLDING = "HOLDING"
    RELEASING = "RELEASING"


@dataclass(slots=True)
class ActionSegmentData:
    start_frame: int
    end_frame: int
    start_pts_ns: int
    end_pts_ns: int
    hand: Hand
    object_class: str
    mean_grasp_confidence: float
    phase_counts: dict[InteractionState, int]

    @property
    def label(self) -> str:
        return f"grasp:{self.object_class}"


@dataclass(slots=True)
class _HandTrack:
    state: InteractionState = InteractionState.IDLE
    contact_frames: int = 0
    release_frames: int = 0
    last_index: int = -1
    last_distance: float | None = None
    # Open segment bookkeeping.
    seg_start: int | None = None
    seg_start_pts: int = 0
    seg_last: int = -1
    seg_last_pts: int = 0
    seg_object: str = ""
    seg_conf: list[float] = field(default_factory=list)
    seg_phases: dict[InteractionState, int] = field(default_factory=dict)
    seg_had_contact: bool = False


def grasp_confidence(
    hand_conf: float, object_conf: float, contact_iou: float, contact_frames: int, hold_after: int
) -> float:
    """Detection confidence x overlap strength x temporal stability, in [0, 1]."""
    overlap = min(1.0, contact_iou / 0.3) if contact_iou > 0 else 0.0
    stability = min(1.0, contact_frames / max(1, hold_after))
    return float(max(0.0, min(1.0, (hand_conf * object_conf) ** 0.5 * (0.5 * overlap + 0.5 * stability))))


class GraspScorer:
    def __init__(self, config: GraspConfig) -> None:
        self.config = config
        self._tracks: dict[Hand, _HandTrack] = {}
        self.segments: list[ActionSegmentData] = []

    def score(self, record: FrameRecord) -> list[GraspInfo]:
        """Annotate one kept frame (in index order). Returns per-hand grasp info."""
        inference = record.inference
        if inference is None:
            return []
        objects = [(i, d) for i, d in enumerate(inference.objects) if d.class_id != PERSON_CLASS_ID]
        infos: list[GraspInfo] = []
        seen: set[Hand] = set()
        for hand_index, hand in enumerate(inference.hands):
            key = hand.handedness if hand.handedness not in seen else Hand.UNKNOWN
            seen.add(key)
            track = self._tracks.setdefault(key, _HandTrack())
            if track.last_index >= 0 and record.index - track.last_index > self.config.max_frame_gap:
                self._close(track, key)
                self._tracks[key] = track = _HandTrack()
            best_idx, best_iou, best_dist, best_conf, best_class = -1, 0.0, float("inf"), 0.0, ""
            hand_diag = max(diagonal(hand.bbox), 1.0)
            hc = center(hand.bbox)
            for obj_idx, obj in objects:
                overlap = iou(hand.bbox, obj.bbox)
                oc = center(obj.bbox)
                dist = ((hc[0] - oc[0]) ** 2 + (hc[1] - oc[1]) ** 2) ** 0.5 / hand_diag
                if (overlap, -dist) > (best_iou, -best_dist):
                    best_idx, best_iou, best_dist, best_conf, best_class = (
                        obj_idx,
                        overlap,
                        dist,
                        obj.confidence,
                        obj.class_name,
                    )
            state = self._step(track, best_iou, best_dist)
            conf = grasp_confidence(
                hand.confidence, best_conf, best_iou, track.contact_frames, self.config.hold_after_frames
            )
            if state is InteractionState.IDLE or state is InteractionState.APPROACHING:
                conf = 0.0
            self._update_segment(track, key, state, record, best_class, conf)
            track.last_index = record.index
            infos.append(
                GraspInfo(hand_index, best_idx, best_iou, state.value, conf, best_dist if best_idx >= 0 else -1.0)
            )
        record.grasps = infos
        return infos

    def _step(self, track: _HandTrack, contact_iou: float, distance: float) -> InteractionState:
        cfg = self.config
        in_contact = contact_iou >= cfg.contact_iou
        approaching = (
            distance <= cfg.approach_distance and track.last_distance is not None and distance < track.last_distance
        )
        track.last_distance = distance if distance != float("inf") else None
        if in_contact:
            track.contact_frames += 1
            track.release_frames = 0
            track.state = (
                InteractionState.HOLDING if track.contact_frames > cfg.hold_after_frames else InteractionState.GRASPING
            )
            return track.state
        if track.state in (InteractionState.GRASPING, InteractionState.HOLDING) or (
            track.state is InteractionState.RELEASING and track.release_frames < cfg.release_grace_frames
        ):
            track.release_frames += 1
            track.contact_frames = 0
            if track.release_frames <= cfg.release_grace_frames:
                track.state = InteractionState.RELEASING
                return track.state
        track.contact_frames = 0
        track.release_frames = 0
        track.state = InteractionState.APPROACHING if approaching else InteractionState.IDLE
        return track.state

    def _update_segment(
        self, track: _HandTrack, hand: Hand, state: InteractionState, record: FrameRecord, obj: str, conf: float
    ) -> None:
        if state is InteractionState.IDLE:
            self._close(track, hand)
            return
        if track.seg_start is None:
            if state is InteractionState.RELEASING:
                return
            track.seg_start, track.seg_start_pts = record.index, record.pts_ns
            track.seg_phases, track.seg_conf, track.seg_had_contact = {}, [], False
        track.seg_last, track.seg_last_pts = record.index, record.pts_ns
        track.seg_phases[state] = track.seg_phases.get(state, 0) + 1
        if state in (InteractionState.GRASPING, InteractionState.HOLDING):
            track.seg_had_contact = True
            track.seg_object = obj or track.seg_object
            track.seg_conf.append(conf)

    def _close(self, track: _HandTrack, hand: Hand) -> None:
        if track.seg_start is not None and track.seg_had_contact:
            length = track.seg_last - track.seg_start + 1
            if length >= self.config.min_segment_frames:
                self.segments.append(
                    ActionSegmentData(
                        start_frame=track.seg_start,
                        end_frame=track.seg_last,
                        start_pts_ns=track.seg_start_pts,
                        end_pts_ns=track.seg_last_pts,
                        hand=hand,
                        object_class=track.seg_object or "unknown",
                        mean_grasp_confidence=sum(track.seg_conf) / max(1, len(track.seg_conf)),
                        phase_counts=dict(track.seg_phases),
                    )
                )
        track.seg_start = None
        track.seg_had_contact = False

    def finish(self) -> list[ActionSegmentData]:
        for hand, track in self._tracks.items():
            self._close(track, hand)
        return self.segments
