"""Child detection: 15-frame sampling + 2-second temporal exclusion buffer.

Rule (all defaults from :class:`ChildDetectionConfig`):

* A child check (InsightFace face detection + age) runs on frames
  0, 15, 30, ... If a scheduled frame was removed by Tier 1 before it could be
  checked, the check is deferred to the next frame that reaches the GPU.
* Out-of-cycle checks run whenever the YOLO person count changes between
  consecutive GPU frames (person-change safety override).
* A face with estimated age < 18 and confidence > 0.85 at frame F_k drops
  [F_k - 30, F_k + 30] clamped to [0, total_frames - 1] (61 frames). Ranges
  from multiple detections overlap-merge into a single exclusion block.
* Pixel buffers of every excluded frame are freed immediately (including
  frames that have not been decoded yet, as they arrive).
* Face boxes, crops, embeddings and ages are never stored: only
  ``{"reason": "CHILD_DETECTED", "anchor_frame": F_k, "dropped_range": [a, b]}``
  leaves this module.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import ClassVar

from physical_ai_pipeline.config import ChildDetectionConfig
from physical_ai_pipeline.filters.base import FilterBase, FilterEventListener, FilterHit
from physical_ai_pipeline.frame import DropReason, FaceObservation, FrameRecord
from physical_ai_pipeline.timeline import FrameWindow


def exclusion_range(anchor: int, before: int, after: int, total_frames: int | None) -> tuple[int, int]:
    """Inclusive [anchor - before, anchor + after] clamped to the session."""
    start = max(0, anchor - before)
    end = anchor + after
    if total_frames is not None:
        end = min(end, total_frames - 1)
    return start, end


def merge_ranges(ranges: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping or touching inclusive ranges."""
    merged: list[list[int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


@dataclass(frozen=True, slots=True)
class ChildEvent:
    """The only record of a child detection that may be persisted."""

    anchor_frame: int
    dropped_range: tuple[int, int]
    out_of_cycle: bool

    def to_audit(self) -> dict[str, object]:
        return {
            "reason": DropReason.CHILD_DETECTED.value,
            "anchor_frame": self.anchor_frame,
            "dropped_range": [self.dropped_range[0], self.dropped_range[1]],
        }


class ChildDetectionFilter(FilterBase[ChildDetectionConfig]):
    name: ClassVar[str] = "child_detection"
    release_pixels_on_hit: ClassVar[bool] = True

    def __init__(
        self,
        settings: ChildDetectionConfig,
        listener: FilterEventListener | None = None,
        total_frames: int | None = None,
    ) -> None:
        super().__init__(settings, listener)
        self.total_frames = total_frames
        self.events: list[ChildEvent] = []
        self.checks_scheduled = 0
        self.checks_out_of_cycle = 0
        self._faces: Sequence[FaceObservation] = ()
        self._out_of_cycle = False
        self._deferred = False
        self._last_person_count: int | None = None

    # -- scheduling ---------------------------------------------------------

    def is_scheduled(self, index: int) -> bool:
        return index % self.settings.sample_interval == 0

    def on_frame_arrival(self, record: FrameRecord) -> None:
        """Mark a scheduled frame as owing a check (before Tier 1)."""
        if self.is_scheduled(record.index):
            record.child_check_pending = True

    def on_tier1_result(self, record: FrameRecord, survived: bool, in_exclusion: bool) -> None:
        """Defer a scheduled check whose frame Tier 1 removed; consume a
        deferred check on the next surviving frame."""
        if in_exclusion:
            return
        if not survived:
            if record.child_check_pending:
                record.child_check_pending = False
                self._deferred = True
                if record.release_deferred:
                    record.release_pixels(force=True)
            return
        if self._deferred and not record.child_check_pending:
            record.child_check_pending = True
        self._deferred = False

    def on_person_count(self, record: FrameRecord, count: int) -> bool:
        """Person-change override. Returns True when an out-of-cycle check was added."""
        previous, self._last_person_count = self._last_person_count, count
        if not self.settings.person_change_override or previous is None or previous == count:
            return False
        if record.child_check_pending:
            return False
        record.child_check_pending = True
        record.child_out_of_cycle = True
        return True

    # -- decision -----------------------------------------------------------

    def check(self, record: FrameRecord, faces: Sequence[FaceObservation], window: FrameWindow) -> bool:
        """Apply the child rule to ``record`` with transient ``faces``.
        Returns True when a child was detected."""
        out_of_cycle = bool(record.child_out_of_cycle)
        if out_of_cycle:
            self.checks_out_of_cycle += 1
        else:
            self.checks_scheduled += 1
        self._faces, self._out_of_cycle = faces, out_of_cycle
        before = len(self.events)
        try:
            record.child_check_performed = True
            self.process(record, window)
        finally:
            self._faces = ()
            record.child_out_of_cycle = False
            record.finish_child_check()
        return len(self.events) > before

    def is_child(self, face: FaceObservation) -> bool:
        return face.age < self.settings.max_age_exclusive and face.det_score > self.settings.min_confidence

    def evaluate(self, record: FrameRecord) -> list[FilterHit]:
        if not any(self.is_child(face) for face in self._faces):
            return []
        start, end = exclusion_range(
            record.index, self.settings.buffer_frames_before, self.settings.buffer_frames_after, self.total_frames
        )
        self.events.append(ChildEvent(record.index, (start, end), self._out_of_cycle))
        # No face geometry, score, or age in the hit detail.
        return [FilterHit(DropReason.CHILD_DETECTED, start, end, {}, anchor_frame=record.index)]

    def clamp_events(self, last_frame_index: int) -> list[ChildEvent]:
        """Final events with ranges clamped to the decoded session length."""
        return [
            ChildEvent(e.anchor_frame, (e.dropped_range[0], min(e.dropped_range[1], last_frame_index)), e.out_of_cycle)
            for e in self.events
        ]

    def merged_blocks(self, last_frame_index: int) -> list[tuple[int, int]]:
        return merge_ranges(e.dropped_range for e in self.clamp_events(last_frame_index))
