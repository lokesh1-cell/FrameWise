"""Perceptual-hash duplicate frame detection."""

from __future__ import annotations

from typing import ClassVar

from physical_ai_pipeline.config import DuplicateConfig
from physical_ai_pipeline.filters.base import FilterBase, FilterHit
from physical_ai_pipeline.frame import DropReason, FrameRecord


class DuplicateFilter(FilterBase[DuplicateConfig]):
    """Drops a frame whose 64-bit pHash is within ``min_hamming_distance``
    bits of the previous decoded frame."""

    name: ClassVar[str] = "duplicate"

    def evaluate(self, record: FrameRecord) -> list[FilterHit]:
        if record.tier1 is None or record.tier1.phash_hamming < 0:
            return []
        distance = record.tier1.phash_hamming
        record.quality.phash_hamming = distance
        if distance < self.settings.min_hamming_distance:
            return [FilterHit(DropReason.DUPLICATE, record.index, record.index, {"hamming": float(distance)})]
        return []
