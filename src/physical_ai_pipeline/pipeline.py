"""Session orchestrator: decode -> Tier 1 -> Tier 2 (batched) -> Tier 3 ->
Tier 4 -> MCAP, with bounded memory.

Memory model: pixel buffers exist only for frames inside the in-flight
window (decision lag + one GPU batch + smoother lookahead, ~130 frames at
the default thresholds). Child-excluded frames are freed the moment the
exclusion is known, including frames that arrive later inside the buffer.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from physical_ai_pipeline.audit import AuditCollector, AuditRow, AuditSink
from physical_ai_pipeline.config import PipelineConfig
from physical_ai_pipeline.filters.base import FilterEvent, FilterEventListener
from physical_ai_pipeline.filters.tier1 import Tier1Precompute, Tier1Stage
from physical_ai_pipeline.filters.tier3 import ChildDetectionFilter, ChildEvent, build_decision_chain
from physical_ai_pipeline.frame import DropReason, FrameRecord
from physical_ai_pipeline.inference.backend import InferenceBackend, ModelInfo
from physical_ai_pipeline.inference.runner import BatchedInferenceRunner
from physical_ai_pipeline.logging_utils import get_logger, log_event
from physical_ai_pipeline.output import convert
from physical_ai_pipeline.output.drop_log import DropLogAggregator, DropRange
from physical_ai_pipeline.output.mcap_writer import SessionMcapWriter
from physical_ai_pipeline.postprocess.grasp import GraspScorer
from physical_ai_pipeline.postprocess.smoother import TemporalSmoother
from physical_ai_pipeline.proto_gen import physical_ai_pb2 as pb
from physical_ai_pipeline.timeline import FrameWindow

_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    distortion: tuple[float, ...] = ()

    @property
    def calibrated(self) -> bool:
        return self.fx > 0 and self.fy > 0


@dataclass(frozen=True, slots=True)
class SessionInfo:
    session_id: str
    source_uri: str
    source_sha256: str = ""
    session_start_ns: int = 0
    fps: float = 30.0
    total_frames_hint: int | None = None
    task_whitelist: tuple[str, ...] | None = None
    intrinsics: CameraIntrinsics | None = None
    git_commit: str = "unknown"


@dataclass(slots=True)
class SessionResult:
    session_id: str
    mcap_path: Path
    stats: dict[str, Any]
    child_events: list[ChildEvent]
    drop_ranges: list[DropRange]
    audit_rows: list[AuditRow]
    models: list[ModelInfo]
    message_counts: dict[str, int] = field(default_factory=dict)


class SessionCancelled(RuntimeError):
    """Raised when the cancel event is set (e.g. Spot preemption SIGTERM)."""


class _Listeners:
    def __init__(self, *listeners: FilterEventListener) -> None:
        self._listeners = listeners

    def on_filter_event(self, event: FilterEvent) -> None:
        for listener in self._listeners:
            listener.on_filter_event(event)


class _EventLogger:
    def on_filter_event(self, event: FilterEvent) -> None:
        hit = event.hit
        fields: dict[str, Any] = {
            "filter": event.filter_name,
            "mode": event.mode.value,
            "reason": hit.reason.value,
            "start_frame": hit.start,
            "end_frame": hit.end,
        }
        if hit.reason is DropReason.CHILD_DETECTED:
            fields["anchor_frame"] = hit.anchor_frame  # never face data
        log_event(_log, "filter_hit", level=10, **fields)


class SessionPipeline:
    def __init__(self, config: PipelineConfig, backend: InferenceBackend, audit_sink: AuditSink | None = None) -> None:
        self.config = config
        self.backend = backend
        self.audit_sink = audit_sink

    def run(
        self,
        frames: Iterable[FrameRecord],
        info: SessionInfo,
        mcap_path: str | Path,
        cancel: threading.Event | None = None,
    ) -> SessionResult:
        return _SessionRun(self.config, self.backend, self.audit_sink, info, Path(mcap_path), cancel).run(frames)


class _SessionRun:
    def __init__(
        self,
        config: PipelineConfig,
        backend: InferenceBackend,
        audit_sink: AuditSink | None,
        info: SessionInfo,
        mcap_path: Path,
        cancel: threading.Event | None = None,
    ) -> None:
        self.cfg = config
        self.cancel = cancel
        self.info = info
        self.audit_sink = audit_sink
        self.audit = AuditCollector(info.session_id, info.source_uri, config.pipeline_version, info.session_start_ns)
        listener = _Listeners(self.audit, _EventLogger())
        self.window = FrameWindow()
        self.tier1 = Tier1Stage(config.tier1, listener)
        self.child = ChildDetectionFilter(config.tier3.child, listener, info.total_frames_hint)
        self.decisions = build_decision_chain(config.tier3, listener, info.task_whitelist)
        self.runner = BatchedInferenceRunner(backend, config.inference)
        self.smoother = TemporalSmoother(config.smoothing)
        self.grasp = GraspScorer(config.grasp)
        self.drop_log = DropLogAggregator()
        self.writer = SessionMcapWriter(mcap_path, config.output, info.session_start_ns)
        self.lag = config.decision_lag_frames
        self.queue: list[FrameRecord] = []
        self.child_pts: dict[int, int] = {}
        self.total = 0
        self.kept = 0
        self.last_index = -1
        self.first_pts_ns = 0
        self.last_pts_ns = 0
        self.width = 0
        self.height = 0

    # -- main loop --------------------------------------------------------------

    def run(self, frames: Iterable[FrameRecord]) -> SessionResult:
        started = time.perf_counter()
        log_event(_log, "session_started", session_id=self.info.session_id, source_uri=self.info.source_uri)
        precompute = Tier1Precompute(self.cfg.tier1, workers=self.cfg.tier1.metric_workers)
        try:
            for record in precompute(iter(frames)):
                if self.cancel is not None and self.cancel.is_set():
                    raise SessionCancelled(f"session {self.info.session_id} cancelled at frame {record.index}")
                self._ingest(record)
                if len(self.queue) >= self.cfg.inference.batch_size:
                    self._process_batch()
                self._finalize(self._processed_frontier() - self.lag)
            self._process_batch()
            self._finalize(self.last_index)
            for record in self.smoother.flush():
                self._emit(record)
            closed = self.drop_log.flush()
            if closed is not None:
                self._log_drop(closed)
            segments = self.grasp.finish()
            self.writer.write_actions(segments)
            child_events = self.child.clamp_events(self.last_index)
            self.audit.add_child_events(child_events, self.child_pts)
            stats = self._stats(time.perf_counter() - started, child_events)
            self.writer.write_metadata(self._metadata(stats))
            self.writer.close()
        except BaseException:
            self.writer.abort()
            log_event(_log, "session_failed", level=40, session_id=self.info.session_id)
            raise
        rows = self.audit.finish()
        if self.audit_sink is not None:
            self.audit_sink.write_rows(rows)
            self.audit_sink.flush()
        log_event(_log, "session_finished", session_id=self.info.session_id, **stats)
        return SessionResult(
            session_id=self.info.session_id,
            mcap_path=self.writer.path,
            stats=stats,
            child_events=child_events,
            drop_ranges=list(self.drop_log.closed),
            audit_rows=rows,
            models=self.runner.backend.model_infos(),
            message_counts=dict(self.writer.message_counts),
        )

    def _ingest(self, record: FrameRecord) -> None:
        if record.index != self.last_index + 1:
            raise ValueError(f"frames must be contiguous: got {record.index} after {self.last_index}")
        self.total += 1
        self.last_index = record.index
        if record.index == 0:
            self.first_pts_ns = record.pts_ns
        self.last_pts_ns = record.pts_ns
        if record.image is not None and not self.width:
            self.height, self.width = record.image.shape[:2]

        self.child.on_frame_arrival(record)
        self.window.add(record)  # applies pending child exclusions (+ pixel release)
        if record.child_dropped:
            # Inside an exclusion: no Tier 1 / detection work, but a scheduled
            # check still runs so a child who stays extends the block.
            if record.child_check_pending and record.image is not None:
                self.queue.append(record)
            else:
                record.finish_child_check()
            return
        survived = self.tier1.process(record, self.window)
        self.child.on_tier1_result(record, survived, in_exclusion=False)
        if survived:
            self.queue.append(record)

    def _process_batch(self) -> None:
        if not self.queue:
            return
        batch, self.queue = self.queue, []
        gpu = [r for r in batch if not r.dropped and r.image is not None]
        self.runner.run(gpu)
        for record in gpu:
            if record.inference is not None:
                self.child.on_person_count(record, record.inference.person_count)
        checks = [r for r in batch if r.child_check_pending and r.image is not None]
        faces = self.runner.analyze_faces(checks)
        try:
            for record in checks:
                if self.child.check(record, faces.pop(record.index, ()), self.window):
                    self.child_pts[record.index] = record.pts_ns
        finally:
            faces.clear()  # face observations never outlive the decision
        for record in batch:
            if record.child_check_pending:
                record.finish_child_check()
        for record in gpu:
            if not record.dropped and record.inference is not None:
                self.decisions.process(record, self.window)

    def _processed_frontier(self) -> int:
        return self.queue[0].index - 1 if self.queue else self.last_index

    def _finalize(self, up_to: int) -> None:
        for record in self.window.pop_until(up_to):
            for out in self.smoother.push(record):
                self._emit(out)

    def _emit(self, record: FrameRecord) -> None:
        closed = self.drop_log.push(record)
        if closed is not None:
            self._log_drop(closed)
        if record.dropped:
            self.writer.write_dropped_frame(record)
            return
        self.kept += 1
        self.grasp.score(record)
        self.writer.write_kept_frame(record)

    def _log_drop(self, drop: DropRange) -> None:
        anchors: list[int] = []
        if drop.reason is DropReason.CHILD_DETECTED:
            anchors = [
                e.anchor_frame for e in self.child.events if drop.start_frame <= e.anchor_frame <= drop.end_frame
            ]
        self.writer.write_drop_event(drop, anchors)
        self.audit.add_drop_range(drop)

    # -- reporting --------------------------------------------------------------

    def _stats(self, wall: float, child_events: Sequence[ChildEvent]) -> dict[str, Any]:
        dropped = {r.value: n for r, n in sorted(self.drop_log.dropped_by_reason.items())}
        return {
            "total_frames": self.total,
            "kept_frames": self.kept,
            "dropped_frames": self.total - self.kept,
            "dropped_by_reason": dropped,
            "child_detection_events": len(child_events),
            "child_exclusion_blocks": len(self.child.merged_blocks(self.last_index)),
            "child_checks_scheduled": self.child.checks_scheduled,
            "child_checks_out_of_cycle": self.child.checks_out_of_cycle,
            "gpu_frames": self.runner.frames_inferred,
            "face_frames": self.runner.face_frames,
            "gpu_seconds": round(self.runner.gpu_seconds, 3),
            "restored_by_smoothing": self.smoother.restored,
            "wall_seconds": round(wall, 3),
            "frames_per_second": round(self.total / wall, 2) if wall > 0 else 0.0,
        }

    def _metadata(self, stats: dict[str, Any]) -> pb.SessionMetadata:
        info = self.info
        intr = info.intrinsics or CameraIntrinsics(self.width, self.height)
        models = self.runner.backend.model_infos()
        meta = pb.SessionMetadata(
            timestamp=convert.timestamp(info.session_start_ns),
            session_id=info.session_id,
            source_uri=info.source_uri,
            source_sha256=info.source_sha256,
            pipeline_version=self.cfg.pipeline_version,
            git_commit=info.git_commit,
            intrinsics=pb.CameraIntrinsics(
                width=intr.width, height=intr.height, fx=intr.fx, fy=intr.fy, cx=intr.cx, cy=intr.cy,
                distortion_model="plumb_bob" if intr.distortion else "none",
                distortion=list(intr.distortion), calibrated=intr.calibrated,
            ),
            models=[pb.ModelInfo(name=m.name, role=m.role, engine_path=m.engine_path, sha256=m.sha256, precision=m.precision) for m in models],
            stats=pb.ProcessingStats(
                total_frames=stats["total_frames"], kept_frames=stats["kept_frames"],
                dropped_frames=stats["dropped_frames"], dropped_by_reason=stats["dropped_by_reason"],
                child_detection_events=stats["child_detection_events"],
                child_checks_scheduled=stats["child_checks_scheduled"],
                child_checks_out_of_cycle=stats["child_checks_out_of_cycle"],
                gpu_frames=stats["gpu_frames"], wall_seconds=stats["wall_seconds"],
                frames_per_second=stats["frames_per_second"], restored_by_smoothing=stats["restored_by_smoothing"],
            ),
            task_whitelist=list(info.task_whitelist or self.cfg.tier3.off_task.task_whitelist),
            source_fps=info.fps,
            session_start=convert.timestamp(info.session_start_ns + self.first_pts_ns),
            session_end=convert.timestamp(info.session_start_ns + self.last_pts_ns),
            config_json=self.cfg.thresholds_json(),
        )  # fmt: skip
        return meta
