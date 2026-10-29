"""Farneback optical-flow idle / static scene detection."""

from __future__ import annotations

from typing import ClassVar

from physical_ai_pipeline.config import IdleConfig
from physical_ai_pipeline.filters.base import FilterBase, FilterEventListener, FilterHit
from physical_ai_pipeline.frame import DropReason, FrameRecord


class IdleFilter(FilterBase[IdleConfig]):
    """Drops runs of >= ``min_consecutive_frames`` frames whose mean flow
    magnitude (vs. the previous decoded frame, in full-resolution pixels) is
    below ``min_flow_magnitude``.

    When a run reaches the minimum length the whole run is dropped
    retroactively, then each further low-motion frame as it arrives. Frames
    removed by an earlier Tier 1 filter (e.g. duplicates, which are static by
    definition) do not reach this filter and do not break a run.
    """

    name: ClassVar[str] = "idle"

    def __init__(self, settings: IdleConfig, listener: FilterEventListener | None = None) -> None:
        super().__init__(settings, listener)
        self._run_start: int | None = None
        self._run_length = 0

    def evaluate(self, record: FrameRecord) -> list[FilterHit]:
        if record.tier1 is None or record.tier1.flow_magnitude < 0:
            return []
        magnitude = record.tier1.flow_magnitude
        record.quality.flow_magnitude = magnitude
        if magnitude >= self.settings.min_flow_magnitude:
            self._run_start, self._run_length = None, 0
            return []
        if self._run_start is None:
            self._run_start = record.index
        self._run_length += 1
        need = self.settings.min_consecutive_frames
        detail = {"flow_magnitude": magnitude, "run_length": float(self._run_length)}
        if self._run_length == need:
            return [FilterHit(DropReason.IDLE, self._run_start, record.index, detail)]
        if self._run_length > need:
            return [FilterHit(DropReason.IDLE, record.index, record.index, detail)]
        return []
