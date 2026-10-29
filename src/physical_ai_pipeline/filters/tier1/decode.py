"""Corrupt frame & decode check."""

from __future__ import annotations

from typing import ClassVar

from physical_ai_pipeline.config import DecodeCheckConfig
from physical_ai_pipeline.filters.base import FilterBase, FilterHit
from physical_ai_pipeline.frame import DropReason, FrameRecord


class DecodeCheckFilter(FilterBase[DecodeCheckConfig]):
    """Drops frames PyAV failed to decode and frames whose flat-green
    macroblock ratio exceeds ``max_corrupt_block_ratio``."""

    name: ClassVar[str] = "decode_check"

    def evaluate(self, record: FrameRecord) -> list[FilterHit]:
        m = record.tier1
        if m is None or not m.decode_ok:
            record.quality.decode_ok = False
            return [FilterHit(DropReason.DECODE_ERROR, record.index, record.index, {"decoded": 0.0})]
        record.quality.corrupt_block_ratio = m.corrupt_block_ratio
        if m.corrupt_block_ratio > self.settings.max_corrupt_block_ratio:
            record.quality.decode_ok = False
            return [
                FilterHit(
                    DropReason.DECODE_ERROR, record.index, record.index, {"corrupt_block_ratio": m.corrupt_block_ratio}
                )
            ]
        return []
