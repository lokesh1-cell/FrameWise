"""Fake work: fast wrist motion with zero hand-object overlap for >= 60 frames."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from physical_ai_pipeline.config import FakeWorkConfig
from physical_ai_pipeline.filters.base import FilterBase, FilterEventListener, FilterHit
from physical_ai_pipeline.frame import BBox, DropReason, FrameRecord
from physical_ai_pipeline.inference.coco import LEFT_WRIST, PERSON_CLASS_ID, RIGHT_WRIST
from physical_ai_pipeline.inference.geometry import iou, point_box


def max_hand_object_iou(record: FrameRecord, wrist_box_fraction: float = 0.12, min_conf: float = 0.5) -> float:
    """Max IoU between any hand region and any non-person object box."""
    inference = record.inference
    if inference is None:
        return 0.0
    hands: list[BBox] = [h.bbox for h in inference.hands]
    if not hands:
        for pose in inference.poses:
            side = max(1.0, (pose.bbox[3] - pose.bbox[1]) * wrist_box_fraction)
            for idx in (LEFT_WRIST, RIGHT_WRIST):
                x, y, c = pose.keypoints[idx]
                if c >= min_conf:
                    hands.append(point_box(float(x), float(y), side))
    objects = [d.bbox for d in inference.objects if d.class_id != PERSON_CLASS_ID]
    return max((iou(h, o) for h in hands for o in objects), default=0.0)


class FakeWorkFilter(FilterBase[FakeWorkConfig]):
    """Tracks the primary person's wrists across GPU frames. A frame is
    "suspicious" when the fastest wrist moves >= ``min_wrist_velocity``
    person-heights per frame while hand-object IoU is exactly 0. A run of
    ``min_consecutive_frames`` suspicious frames is dropped retroactively."""

    name: ClassVar[str] = "fake_work"

    def __init__(self, settings: FakeWorkConfig, listener: FilterEventListener | None = None) -> None:
        super().__init__(settings, listener)
        self._prev_index: int | None = None
        self._prev_wrists: dict[int, np.ndarray] = {}
        self._run_start: int | None = None
        self._run_length = 0

    def wrist_speed(self, record: FrameRecord) -> float | None:
        inference = record.inference
        if inference is None or not inference.poses:
            self._prev_wrists, self._prev_index = {}, record.index
            return None
        pose = max(inference.poses, key=lambda p: p.confidence)
        height = max(1.0, pose.bbox[3] - pose.bbox[1])
        wrists = {
            idx: pose.keypoints[idx, :2].astype(np.float64)
            for idx in (LEFT_WRIST, RIGHT_WRIST)
            if pose.keypoints[idx, 2] >= self.settings.keypoint_min_confidence
        }
        speed: float | None = None
        if self._prev_index is not None and self._prev_wrists:
            gap = max(1, record.index - self._prev_index)
            speeds = [
                float(np.linalg.norm(wrists[i] - self._prev_wrists[i])) / gap / height
                for i in wrists
                if i in self._prev_wrists
            ]
            speed = max(speeds) if speeds else None
        self._prev_wrists, self._prev_index = wrists, record.index
        return speed

    def evaluate(self, record: FrameRecord) -> list[FilterHit]:
        speed = self.wrist_speed(record)
        overlap = max_hand_object_iou(record, min_conf=self.settings.keypoint_min_confidence)
        suspicious = speed is not None and speed >= self.settings.min_wrist_velocity and overlap == 0.0
        if not suspicious:
            self._run_start, self._run_length = None, 0
            return []
        if self._run_start is None:
            self._run_start = record.index
        self._run_length += 1
        need = self.settings.min_consecutive_frames
        detail = {"wrist_velocity": float(speed or 0.0), "run_length": float(self._run_length)}
        if self._run_length == need:
            return [FilterHit(DropReason.FAKE_WORK, self._run_start, record.index, detail)]
        if self._run_length > need:
            return [FilterHit(DropReason.FAKE_WORK, record.index, record.index, detail)]
        return []
