"""Tier 4 temporal consistency smoothing (streaming, bounded lookahead).

1. Gap fill: a run of <= ``max_gap_fill_frames`` dropped frames between kept
   frames is restored when every frame in it is restorable (SOFT verdicts
   only, pixels still in RAM, never CHILD_DETECTED or DECODE_ERROR).
2. Island removal: a "cluster" (kept runs joined by filled gaps) shorter than
   ``min_island_frames`` is dropped as ISOLATED_ISLAND.

Records must arrive in index order with all upstream verdicts final. The
smoother emits them in order once their fate is decided; at most about
``min_island_frames + max_gap_fill_frames`` records are held.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator

from physical_ai_pipeline.config import FilterMode, SmoothingConfig
from physical_ai_pipeline.frame import DropReason, FrameRecord, Verdict

ISLAND_VERDICT = Verdict(DropReason.ISOLATED_ISLAND, FilterMode.HARD, "temporal_smoother")


def _runs(records: list[FrameRecord]) -> list[tuple[bool, int, int]]:
    """(dropped, start_offset, length) runs over ``records``."""
    runs: list[tuple[bool, int, int]] = []
    for offset, record in enumerate(records):
        state = record.dropped
        if runs and runs[-1][0] == state:
            dropped, start, length = runs[-1]
            runs[-1] = (dropped, start, length + 1)
        else:
            runs.append((state, offset, 1))
    return runs


class TemporalSmoother:
    def __init__(self, config: SmoothingConfig) -> None:
        self.config = config
        self._buffer: deque[FrameRecord] = deque()
        self._in_confirmed_cluster = False
        self.restored = 0
        self.islands_dropped = 0

    def push(self, record: FrameRecord) -> Iterator[FrameRecord]:
        if not self.config.enabled:
            yield record
            return
        if self._buffer and record.index != self._buffer[-1].index + 1:
            raise ValueError(f"smoother expects contiguous frames, got {record.index} after {self._buffer[-1].index}")
        self._buffer.append(record)
        yield from self._drain(final=False)

    def flush(self) -> Iterator[FrameRecord]:
        yield from self._drain(final=True)

    def _fillable(self, records: list[FrameRecord]) -> bool:
        return len(records) <= self.config.max_gap_fill_frames and all(r.restorable for r in records)

    def _restore(self, records: list[FrameRecord]) -> None:
        for record in records:
            record.restored = True
            self.restored += 1

    def _emit(self, count: int) -> Iterator[FrameRecord]:
        for _ in range(count):
            yield self._buffer.popleft()

    def _drain(self, final: bool) -> Iterator[FrameRecord]:
        cfg = self.config
        while self._buffer:
            records = list(self._buffer)
            runs = _runs(records)
            dropped, _, length = runs[0]
            if dropped:
                if not self._in_confirmed_cluster:
                    # Not bracketed by a kept cluster on the left: never filled.
                    yield from self._emit(length)
                    continue
                if len(runs) == 1 and not final and length <= cfg.max_gap_fill_frames:
                    return  # need to see whether a kept frame follows
                gap = records[:length]
                if len(runs) > 1 and self._fillable(gap):
                    self._restore(gap)
                else:
                    self._in_confirmed_cluster = False
                yield from self._emit(length)
                continue

            # Kept run at the head.
            if self._in_confirmed_cluster:
                yield from self._emit(length)
                continue

            decided = self._decide_new_cluster(records, runs, final)
            if decided is None:
                return
            keep, span = decided
            cluster = records[:span]
            if keep:
                for is_drop, start, run_len in runs:
                    if start >= span:
                        break
                    if is_drop:
                        self._restore(records[start : start + run_len])
                self._in_confirmed_cluster = True
            else:
                for record in cluster:
                    if not record.dropped:
                        record.add_verdict(ISLAND_VERDICT)
                        self.islands_dropped += 1
            yield from self._emit(span)

    def _decide_new_cluster(
        self, records: list[FrameRecord], runs: list[tuple[bool, int, int]], final: bool
    ) -> tuple[bool, int] | None:
        """Walk kept runs joined by fillable gaps from the buffer head.

        Returns (keep, span) where span covers the cluster (kept runs and the
        gaps that would be filled), or None when more frames are needed."""
        cfg = self.config
        total = 0
        span = 0
        for i, (is_drop, start, length) in enumerate(runs):
            last = i == len(runs) - 1
            if not is_drop:
                total += length
                span = start + length
                if total >= cfg.min_island_frames:
                    return True, span
                if last and not final:
                    return None
                continue
            gap = records[start : start + length]
            if last:
                if not final and length <= cfg.max_gap_fill_frames and self._fillable(gap):
                    return None  # gap may still be bracketed by a kept frame
                return False, span
            if not self._fillable(gap):
                return False, span
            total += length
        return (total >= cfg.min_island_frames), span
