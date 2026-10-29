"""Off-task behaviour: high-risk irrelevant objects present while none of the
session's task-whitelist objects are visible."""

from __future__ import annotations

from typing import ClassVar

from physical_ai_pipeline.config import OffTaskConfig
from physical_ai_pipeline.filters.base import FilterBase, FilterEventListener, FilterHit
from physical_ai_pipeline.frame import DropReason, FrameRecord


class OffTaskFilter(FilterBase[OffTaskConfig]):
    name: ClassVar[str] = "off_task"

    def __init__(
        self,
        settings: OffTaskConfig,
        listener: FilterEventListener | None = None,
        task_whitelist: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__(settings, listener)
        whitelist = task_whitelist if task_whitelist is not None else settings.task_whitelist
        self.whitelist = frozenset(whitelist)
        self.high_risk = frozenset(settings.high_risk_classes) - self.whitelist

    @property
    def enabled(self) -> bool:
        # Without a whitelist there is no notion of "on task".
        return self.settings.enabled and bool(self.whitelist)

    def evaluate(self, record: FrameRecord) -> list[FilterHit]:
        inference = record.inference
        if inference is None:
            return []
        present = {d.class_name for d in inference.objects if d.confidence >= self.settings.min_confidence}
        risky = present & self.high_risk
        if risky and not (present & self.whitelist):
            return [
                FilterHit(DropReason.OFF_TASK, record.index, record.index, {"high_risk_objects": float(len(risky))})
            ]
        return []
