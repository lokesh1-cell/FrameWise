"""Process one raw-video GCS object into session.mcap + lineage.json."""

from __future__ import annotations

import re
import shutil
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from physical_ai_pipeline.audit import AuditSink
from physical_ai_pipeline.cloud.gcs import GcsRef, ObjectStore, sha256_path
from physical_ai_pipeline.cloud.notifications import OBJECT_FINALIZE, GcsEvent
from physical_ai_pipeline.config import PipelineConfig
from physical_ai_pipeline.decode.video_reader import VideoReader, prefetch
from physical_ai_pipeline.inference.backend import InferenceBackend
from physical_ai_pipeline.logging_utils import get_logger, log_event
from physical_ai_pipeline.output.lineage_writer import (
    build_lineage,
    collect_node_info,
    git_commit,
    utcnow,
    write_lineage,
)
from physical_ai_pipeline.output.mp4_export import export_mp4
from physical_ai_pipeline.output.sensor_export import export_sensors
from physical_ai_pipeline.pipeline import SessionInfo, SessionPipeline

_log = get_logger(__name__)


class Outcome(StrEnum):
    PROCESSED = "PROCESSED"
    SKIPPED = "SKIPPED"  # not ours / already processed -> ack
    INVALID = "INVALID"  # permanent failure -> ack (dead-letter by log)


class InvalidSourceError(ValueError):
    """The object can never be processed (not a decodable video)."""


@dataclass(frozen=True, slots=True)
class ProcessResult:
    outcome: Outcome
    session_id: str | None = None
    detail: str = ""
    mcap_uri: str | None = None


def session_id_for(name: str, generation: str | None) -> str:
    stem = name.rsplit(".", 1)[0]
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_") or "session"
    return f"{slug}-g{generation}" if generation else slug


def _parse_start_ns(custom: dict[str, Any], time_created: str | None) -> int:
    """Capture start from ``x-goog-meta-capture-start`` (RFC3339 or epoch
    milliseconds), falling back to the object's creation time."""
    raw = custom.get("capture-start") or custom.get("capture_start")
    if raw:
        text = str(raw).strip()
        if text.isdigit():
            return int(text) * 1_000_000
        return int(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp() * 1e9)
    if time_created:
        return int(datetime.fromisoformat(time_created.replace("Z", "+00:00")).timestamp() * 1e9)
    return int(datetime.now(tz=UTC).timestamp() * 1e9)


class SessionProcessor:
    def __init__(
        self,
        config: PipelineConfig,
        store: ObjectStore,
        backend: InferenceBackend,
        audit_sink: AuditSink | None,
        query_metadata_server: bool = True,
    ) -> None:
        self.config = config
        self.store = store
        self.backend = backend
        self.audit_sink = audit_sink
        self.query_metadata_server = query_metadata_server

    def output_prefix(self, session_id: str) -> GcsRef:
        cloud = self.config.cloud
        return GcsRef(cloud.output_bucket, f"{cloud.output_prefix.strip('/')}/{session_id}/")

    def accepts(self, event: GcsEvent) -> str | None:
        """Reason to skip, or None if the event should be processed."""
        cloud = self.config.cloud
        if event.event_type != OBJECT_FINALIZE:
            return f"event type {event.event_type}"
        if event.bucket != cloud.raw_bucket:
            return f"bucket {event.bucket} is not {cloud.raw_bucket}"
        if not event.name.lower().endswith(tuple(cloud.video_extensions)):
            return "not a video extension"
        return None

    def _export_sensors(
        self, event: GcsEvent, workdir: Path, local_video: Path, mcap_path: Path, fps: float, prefix: GcsRef
    ) -> None:
        """Cut companion ``<stem>.txt`` (IMU) / ``<stem>.aac`` (audio) uploaded next to the video."""
        stem = event.name.rsplit(".", 1)[0]
        found: dict[str, Path] = {}
        for ext in ("txt", "aac"):
            ref = GcsRef(event.bucket, f"{stem}.{ext}")
            if self.store.exists(ref):
                self.store.download(ref, workdir / f"companion.{ext}")
                found[ext] = workdir / f"companion.{ext}"
        if "txt" not in found:
            log_event(_log, "imu_missing", level=30, object=f"{stem}.txt")
        # No .aac: use the audio track inside the video itself.
        audio = found.get("aac", local_video)
        mp4_path = workdir / "session.mp4"
        counts = export_sensors(
            mcap_path, workdir, fps, self.config.output.mp4_timing, found.get("txt"), audio,
            self.config.output.imu_offset_s, mp4_path,
        )  # fmt: skip
        uploads = {"session.imu.jsonl": "application/x-ndjson", "session.aac": "audio/aac"}
        if self.config.output.write_mp4 and mp4_path.exists():
            uploads["session.mp4"] = "video/mp4"  # re-upload: it now carries the cut audio track
        for name, content_type in uploads.items():
            if (workdir / name).exists() and (name != "session.imu.jsonl" or "imu_samples" in counts):
                self.store.upload(workdir / name, GcsRef(prefix.bucket, prefix.name + name), content_type)

    def process(self, event: GcsEvent, cancel: threading.Event | None = None) -> ProcessResult:
        reason = self.accepts(event)
        if reason is not None:
            return ProcessResult(Outcome.SKIPPED, detail=reason)
        session_id = session_id_for(event.name, event.generation)
        prefix = self.output_prefix(session_id)
        lineage_ref = GcsRef(prefix.bucket, prefix.name + "lineage.json")
        mcap_ref = GcsRef(prefix.bucket, prefix.name + "session.mcap")
        if self.store.exists(lineage_ref):
            # lineage.json is uploaded last, so its presence means the session is complete.
            return ProcessResult(Outcome.SKIPPED, session_id, "already processed", mcap_ref.uri)

        workdir = Path(self.config.cloud.scratch_dir) / session_id
        shutil.rmtree(workdir, ignore_errors=True)
        workdir.mkdir(parents=True)
        started = utcnow()
        try:
            source = GcsRef(event.bucket, event.name, event.generation)
            local_video = workdir / Path(event.name).name
            log_event(_log, "download_started", session_id=session_id, source=source.uri)
            source_sha = self.store.download(source, local_video)
            meta = self.store.metadata(GcsRef(event.bucket, event.name))
            custom = {**dict(event.metadata), **dict(meta.get("custom") or {})}
            try:
                reader = VideoReader(local_video, self.config.fallback_fps)
            except Exception as exc:  # PyAV raises a family of FFmpeg errors
                raise InvalidSourceError(f"cannot open video: {exc}") from exc
            whitelist_raw = custom.get("task-whitelist") or custom.get("task_whitelist")
            whitelist = tuple(w.strip() for w in str(whitelist_raw).split(",") if w.strip()) if whitelist_raw else None
            info = SessionInfo(
                session_id=session_id,
                source_uri=source.uri,
                source_sha256=source_sha,
                session_start_ns=_parse_start_ns(custom, event.time_created or meta.get("time_created")),
                fps=reader.info.fps,
                total_frames_hint=reader.info.frame_count,
                task_whitelist=whitelist,
                git_commit=git_commit(),
            )
            mcap_path = workdir / "session.mcap"
            pipeline = SessionPipeline(self.config, self.backend, self.audit_sink)
            result = pipeline.run(prefetch(reader.frames(), maxsize=32), info, mcap_path, cancel=cancel)
            if result.stats["total_frames"] == 0:
                raise InvalidSourceError("video contains no decodable frames")
            mcap_sha = sha256_path(mcap_path)
            self.store.upload(mcap_path, mcap_ref, "application/octet-stream")
            if self.config.output.write_mp4:
                mp4_path = workdir / "session.mp4"
                if export_mp4(mcap_path, mp4_path, fps=reader.info.fps, timing=self.config.output.mp4_timing):
                    self.store.upload(mp4_path, GcsRef(prefix.bucket, prefix.name + "session.mp4"), "video/mp4")
            self._export_sensors(event, workdir, local_video, mcap_path, reader.info.fps, prefix)
            lineage = build_lineage(
                session_id=session_id,
                source_uri=source.uri,
                source_sha256=source_sha,
                source_generation=event.generation,
                mcap_uri=mcap_ref.uri,
                mcap_sha256=mcap_sha,
                config=self.config,
                models=result.models,
                git_commit=info.git_commit,
                stats={**result.stats, "decode_errors": reader.decode_errors, "source_video": {
                    "width": reader.info.width, "height": reader.info.height, "fps": reader.info.fps,
                    "codec": reader.info.codec, "frame_count": reader.info.frame_count,
                }},
                node_info=collect_node_info(self.query_metadata_server),
                started_at=started,
                finished_at=utcnow(),
            )  # fmt: skip
            lineage_path = write_lineage(workdir / "lineage.json", lineage)
            self.store.upload(lineage_path, lineage_ref, "application/json")
            log_event(_log, "session_uploaded", session_id=session_id, mcap=mcap_ref.uri, lineage=lineage_ref.uri)
            return ProcessResult(Outcome.PROCESSED, session_id, mcap_uri=mcap_ref.uri)
        except InvalidSourceError as exc:
            log_event(_log, "session_invalid", level=40, session_id=session_id, error=str(exc))
            return ProcessResult(Outcome.INVALID, session_id, str(exc))
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
