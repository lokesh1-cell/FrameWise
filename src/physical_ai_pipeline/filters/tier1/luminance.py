"""Luminance (under/over-exposure) check."""

from __future__ import annotations

from typing import ClassVar

from physical_ai_pipeline.config import LuminanceConfig
from physical_ai_pipeline.filters.base import FilterBase, FilterHit
from physical_ai_pipeline.frame import DropReason, FrameRecord


class LuminanceFilter(FilterBase[LuminanceConfig]):
    name: ClassVar[str] = "luminance"

    def evaluate(self, record: FrameRecord) -> list[FilterHit]:
        if record.tier1 is None or not record.tier1.decode_ok:
            return []
        mean = record.tier1.luminance_mean
        record.quality.luminance_mean = mean
        if mean < self.settings.min_mean:
            return [FilterHit(DropReason.UNDEREXPOSED, record.index, record.index, {"luminance_mean": mean})]
        if mean > self.settings.max_mean:
            return [FilterHit(DropReason.OVEREXPOSED, record.index, record.index, {"luminance_mean": mean})]
        return []
