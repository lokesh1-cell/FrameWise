"""Laplacian-variance blur detection."""

from __future__ import annotations

from typing import ClassVar

from physical_ai_pipeline.config import BlurConfig
from physical_ai_pipeline.filters.base import FilterBase, FilterHit
from physical_ai_pipeline.frame import DropReason, FrameRecord


class BlurFilter(FilterBase[BlurConfig]):
    name: ClassVar[str] = "blur"

    def evaluate(self, record: FrameRecord) -> list[FilterHit]:
        if record.tier1 is None or not record.tier1.decode_ok:
            return []
        variance = record.tier1.laplacian_variance
        record.quality.laplacian_variance = variance
        if variance < self.settings.min_laplacian_variance:
            return [FilterHit(DropReason.BLURRY, record.index, record.index, {"laplacian_variance": variance})]
        return []
