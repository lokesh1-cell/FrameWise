"""Run-length aggregation of dropped frames into DropEvent ranges."""

from __future__ import annotations

from dataclasses import dataclass, field

from physical_ai_pipeline.config import FilterMode
from physical_ai_pipeline.frame import DropReason, FrameRecord


@dataclass(slots=True)
class DropRange:
    start_frame: int
    end_frame: int
    start_pts_ns: int
    reason: DropReason
    mode: FilterMode
    filter_name: str
    all_reasons: set[DropReason] = field(default_factory=set)

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame + 1


class DropLogAggregator:
    """Feed finalized records in index order; closed ranges are returned."""

    def __init__(self) -> None:
        self._open: DropRange | None = None
        self.closed: list[DropRange] = []
        self.dropped_by_reason: dict[DropReason, int] = {}

    def push(self, record: FrameRecord) -> DropRange | None:
        reason = record.primary_reason if record.dropped else None
        if reason is not None:
            self.dropped_by_reason[reason] = self.dropped_by_reason.get(reason, 0) + 1
        current = self._open
        if current is not None and (reason != current.reason or record.index != current.end_frame + 1):
            self._open = None
            self.closed.append(current)
            finished: DropRange | None = current
        else:
            finished = None
        if reason is not None:
            verdict = next(v for v in record.verdicts if v.reason is reason)
            if self._open is None:
                self._open = DropRange(
                    record.index, record.index, record.pts_ns, reason, verdict.mode, verdict.filter_name
                )
            self._open.end_frame = record.index
            self._open.all_reasons.update(v.reason for v in record.verdicts)
        return finished

    def flush(self) -> DropRange | None:
        current, self._open = self._open, None
        if current is not None:
            self.closed.append(current)
        return current
