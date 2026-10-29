"""Phone usage: a YOLO "cell phone" box overlapping a wrist or ear keypoint."""

from __future__ import annotations

from typing import ClassVar

from physical_ai_pipeline.config import PhoneUsageConfig
from physical_ai_pipeline.filters.base import FilterBase, FilterHit
from physical_ai_pipeline.frame import BBox, DropReason, FrameRecord
from physical_ai_pipeline.inference.coco import LEFT_EAR, LEFT_WRIST, RIGHT_EAR, RIGHT_WRIST
from physical_ai_pipeline.inference.geometry import iou, point_box

_KEYPOINTS = (LEFT_WRIST, RIGHT_WRIST, LEFT_EAR, RIGHT_EAR)


class PhoneUsageFilter(FilterBase[PhoneUsageConfig]):
    """Each wrist/ear keypoint becomes a square box whose side is
    ``keypoint_box_fraction`` of the person's box height; the frame drops
    when any phone box has IoU > ``iou_threshold`` with one of them. Hand
    boxes from Model D count as wrist regions too."""

    name: ClassVar[str] = "phone_usage"

    def keypoint_regions(self, record: FrameRecord) -> list[BBox]:
        inference = record.inference
        if inference is None:
            return []
        regions: list[BBox] = []
        for pose in inference.poses:
            side = max(1.0, (pose.bbox[3] - pose.bbox[1]) * self.settings.keypoint_box_fraction)
            for idx in _KEYPOINTS:
                x, y, conf = pose.keypoints[idx]
                if conf >= self.settings.keypoint_min_confidence:
                    regions.append(point_box(float(x), float(y), side))
        regions.extend(hand.bbox for hand in inference.hands)
        return regions

    def evaluate(self, record: FrameRecord) -> list[FilterHit]:
        inference = record.inference
        if inference is None:
            return []
        phones = [
            d for d in inference.objects
            if d.class_name == "cell phone" and d.confidence >= self.settings.phone_min_confidence
        ]  # fmt: skip
        if not phones:
            return []
        regions = self.keypoint_regions(record)
        best = max((iou(p.bbox, r) for p in phones for r in regions), default=0.0)
        if best > self.settings.iou_threshold:
            return [FilterHit(DropReason.PHONE_USAGE, record.index, record.index, {"iou": best})]
        return []
