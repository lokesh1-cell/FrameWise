"""frame_audit_log rows and sinks (BigQuery, JSONL, memory).

Row schema (deploy/bigquery/frame_audit_log.json):
    session_id, source_uri, event_time, reason, anchor_frame, dropped_range,
    frame_count, filter_name, filter_mode, pipeline_version

For CHILD_DETECTED rows only ``reason``, ``anchor_frame`` and
``dropped_range`` carry event data, matching the compliance contract:
``{"reason": "CHILD_DETECTED", "anchor_frame": F_k, "dropped_range": [a, b]}``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from physical_ai_pipeline.config import FilterMode
from physical_ai_pipeline.filters.base import FilterEvent
from physical_ai_pipeline.filters.tier3.child import ChildEvent
from physical_ai_pipeline.frame import DropReason
from physical_ai_pipeline.output.drop_log import DropRange

AuditRow = dict[str, Any]


class AuditSink(Protocol):
    def write_rows(self, rows: Sequence[AuditRow]) -> None: ...

    def flush(self) -> None: ...


@dataclass(slots=True)
class MemoryAuditSink:
    rows: list[AuditRow] = field(default_factory=list)

    def write_rows(self, rows: Sequence[AuditRow]) -> None:
        self.rows.extend(rows)

    def flush(self) -> None:
        return None


class JsonlAuditSink:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")  # noqa: SIM115

    def write_rows(self, rows: Sequence[AuditRow]) -> None:
        for row in rows:
            self._fh.write(json.dumps(row, default=str) + "\n")

    def flush(self) -> None:
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def _iso(ns: int) -> str:
    return datetime.fromtimestamp(ns / 1e9, tz=UTC).isoformat()


@dataclass(slots=True)
class _ShadowRun:
    filter_name: str
    reason: str
    start: int
    end: int
    pts_ns: int


class AuditCollector:
    """Builds audit rows. Implements FilterEventListener to aggregate SHADOW
    hits (which never reach the drop log) into ranges."""

    def __init__(self, session_id: str, source_uri: str, pipeline_version: str, base_ns: int) -> None:
        self.session_id = session_id
        self.source_uri = source_uri
        self.pipeline_version = pipeline_version
        self.base_ns = base_ns
        self.rows: list[AuditRow] = []
        self._shadow: dict[str, _ShadowRun] = {}

    def _row(
        self, *, reason: str, pts_ns: int, start: int, end: int, anchor: int | None, filter_name: str, mode: str
    ) -> AuditRow:
        return {
            "insert_id": f"{self.session_id}:{reason}:{start}:{end}:{anchor if anchor is not None else ''}:{mode}",
            "session_id": self.session_id,
            "source_uri": self.source_uri,
            "event_time": _iso(self.base_ns + pts_ns),
            "reason": reason,
            "anchor_frame": anchor,
            "dropped_range": [start, end],
            "frame_count": end - start + 1,
            "filter_name": filter_name,
            "filter_mode": mode,
            "pipeline_version": self.pipeline_version,
        }

    def on_filter_event(self, event: FilterEvent) -> None:
        if event.mode is not FilterMode.SHADOW:
            return
        hit = event.hit
        run = self._shadow.get(event.filter_name)
        if run is not None and run.reason == hit.reason.value and hit.start <= run.end + 1:
            run.end = max(run.end, hit.end)
            return
        if run is not None:
            self._close_shadow(run)
        self._shadow[event.filter_name] = _ShadowRun(
            event.filter_name, hit.reason.value, hit.start, hit.end, event.pts_ns
        )

    def _close_shadow(self, run: _ShadowRun) -> None:
        self.rows.append(
            self._row(reason=run.reason, pts_ns=run.pts_ns, start=run.start, end=run.end, anchor=None,
                      filter_name=run.filter_name, mode=FilterMode.SHADOW.value)
        )  # fmt: skip

    def add_drop_range(self, drop: DropRange) -> None:
        if drop.reason is DropReason.CHILD_DETECTED:
            return  # child rows come from add_child_events, one per detection
        self.rows.append(
            self._row(reason=drop.reason.value, pts_ns=drop.start_pts_ns, start=drop.start_frame, end=drop.end_frame,
                      anchor=None, filter_name=drop.filter_name, mode=drop.mode.value)
        )  # fmt: skip

    def add_child_events(self, events: Sequence[ChildEvent], pts_lookup: dict[int, int]) -> None:
        for event in events:
            start, end = event.dropped_range
            self.rows.append(
                self._row(
                    reason=DropReason.CHILD_DETECTED.value,
                    pts_ns=pts_lookup.get(event.anchor_frame, 0),
                    start=start,
                    end=end,
                    anchor=event.anchor_frame,
                    filter_name="child_detection",
                    mode=FilterMode.HARD.value,
                )
            )

    def finish(self) -> list[AuditRow]:
        for run in self._shadow.values():
            self._close_shadow(run)
        self._shadow.clear()
        return self.rows
