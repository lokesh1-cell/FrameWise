"""Bounded in-memory window of in-flight frames.

Retroactive filters (child buffer, idle runs, fake-work runs) mark frames
behind the processing frontier, and the child buffer also marks frames that
have not been decoded yet. :class:`FrameWindow` owns both: records that are
still in RAM, and verdict ranges that reach into the future.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass

from physical_ai_pipeline.frame import FrameRecord, Verdict


@dataclass(slots=True)
class _PendingRange:
    start: int
    end: int
    verdict: Verdict
    release_pixels: bool


class FrameWindow:
    def __init__(self) -> None:
        self._records: OrderedDict[int, FrameRecord] = OrderedDict()
        self._pending: list[_PendingRange] = []
        self._max_index = -1

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[FrameRecord]:
        return iter(self._records.values())

    def get(self, index: int) -> FrameRecord | None:
        return self._records.get(index)

    def add(self, record: FrameRecord) -> None:
        if record.index <= self._max_index:
            raise ValueError(f"frame {record.index} added out of order (max {self._max_index})")
        self._max_index = record.index
        self._records[record.index] = record
        self._apply_pending(record)

    def apply_range(self, start: int, end: int, verdict: Verdict, *, release_pixels: bool = False) -> None:
        """Apply ``verdict`` to frames [start, end] inclusive.

        Frames already in the window are marked now; frames beyond the
        newest added index are marked when they arrive. With
        ``release_pixels`` the image buffers of every affected in-window frame
        are freed immediately.
        """
        if end < start:
            return
        for index in range(max(start, self.oldest_index()), min(end, self._max_index) + 1):
            record = self._records.get(index)
            if record is None:
                continue
            record.add_verdict(verdict)
            if release_pixels:
                record.release_pixels()
        if end > self._max_index:
            self._pending.append(_PendingRange(max(start, self._max_index + 1), end, verdict, release_pixels))

    def pending_covers(self, index: int, release_only: bool = False) -> bool:
        return any(p.start <= index <= p.end and (p.release_pixels or not release_only) for p in self._pending)

    def oldest_index(self) -> int:
        return next(iter(self._records)) if self._records else self._max_index + 1

    def pop_oldest(self) -> FrameRecord:
        _, record = self._records.popitem(last=False)
        return record

    def pop_until(self, max_index: int) -> list[FrameRecord]:
        out: list[FrameRecord] = []
        while self._records and next(iter(self._records)) <= max_index:
            out.append(self.pop_oldest())
        return out

    def _apply_pending(self, record: FrameRecord) -> None:
        if not self._pending:
            return
        keep: list[_PendingRange] = []
        for pending in self._pending:
            if pending.start <= record.index <= pending.end:
                record.add_verdict(pending.verdict)
                if pending.release_pixels:
                    record.release_pixels()
            if pending.end > record.index:
                keep.append(pending)
        self._pending = keep
