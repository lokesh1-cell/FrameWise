"""Tier 1 CPU filters (fail fast, fail cheap)."""

from __future__ import annotations

from physical_ai_pipeline.config import Tier1Config
from physical_ai_pipeline.filters.base import FilterChain, FilterEventListener
from physical_ai_pipeline.filters.tier1.blur import BlurFilter
from physical_ai_pipeline.filters.tier1.decode import DecodeCheckFilter
from physical_ai_pipeline.filters.tier1.duplicate import DuplicateFilter
from physical_ai_pipeline.filters.tier1.idle import IdleFilter
from physical_ai_pipeline.filters.tier1.luminance import LuminanceFilter
from physical_ai_pipeline.filters.tier1.metrics import InlineMetrics, Tier1Metrics, Tier1Precompute
from physical_ai_pipeline.frame import FrameRecord
from physical_ai_pipeline.timeline import FrameWindow

__all__ = [
    "BlurFilter",
    "DecodeCheckFilter",
    "DuplicateFilter",
    "IdleFilter",
    "LuminanceFilter",
    "Tier1Metrics",
    "Tier1Precompute",
    "Tier1Stage",
]


class Tier1Stage:
    """Decode -> luminance -> blur -> duplicate -> idle, in strict order.

    Measurements come from ``record.tier1`` (filled by
    :class:`Tier1Precompute` on a thread pool) or are computed inline.
    """

    def __init__(self, config: Tier1Config, listener: FilterEventListener | None = None) -> None:
        self.config = config
        self.metrics = InlineMetrics(config)
        self.chain = FilterChain(
            [
                DecodeCheckFilter(config.decode, listener),
                LuminanceFilter(config.luminance, listener),
                BlurFilter(config.blur, listener),
                DuplicateFilter(config.duplicate, listener),
                IdleFilter(config.idle, listener),
            ]
        )

    def process(self, record: FrameRecord, window: FrameWindow) -> bool:
        """Returns True if the frame should continue to GPU inference."""
        self.metrics.ensure(record)
        return self.chain.process(record, window)
