from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest

from physical_ai_pipeline.audit import MemoryAuditSink
from physical_ai_pipeline.cloud.bigquery_sink import BigQueryAuditSink, BigQueryInsertError
from physical_ai_pipeline.cloud.gcs import GcsRef, LocalObjectStore, sha256_path
from physical_ai_pipeline.cloud.models import pin_hashes, sync_engines
from physical_ai_pipeline.cloud.notifications import parse_notification
from physical_ai_pipeline.cloud.processor import Outcome, SessionProcessor, session_id_for
from physical_ai_pipeline.cloud.worker import MessageHandler
from physical_ai_pipeline.config import PipelineConfig
from physical_ai_pipeline.inference.fake_backend import ScriptedBackend
from physical_ai_pipeline.synthetic import SceneGenerator, write_video
from physical_ai_pipeline.verify import verify_mcap

ATTRS = {"eventType": "OBJECT_FINALIZE", "bucketId": "project-raw", "objectId": "site-a/cam1/2026-10-21.mp4",
         "objectGeneration": "1700000000"}  # fmt: skip


def test_parse_notification() -> None:
    body = json.dumps({"size": "123", "contentType": "video/mp4", "timeCreated": "2026-10-21T10:00:00Z",
                       "metadata": {"task-whitelist": "cup,bowl"}}).encode()  # fmt: skip
    event = parse_notification(ATTRS, body)
    assert event.uri == "gs://project-raw/site-a/cam1/2026-10-21.mp4"
    assert event.generation == "1700000000" and event.size == 123
    assert event.metadata["task-whitelist"] == "cup,bowl"
    with pytest.raises(ValueError):
        parse_notification({"foo": "bar"}, b"")


def test_session_id() -> None:
    assert session_id_for("site-a/cam 1/x.mp4", "42") == "site-a_cam_1_x-g42"


@pytest.fixture()
def cloud_env(tmp_path: Path) -> tuple[PipelineConfig, LocalObjectStore]:
    store = LocalObjectStore(tmp_path / "gcs")
    video = write_video(tmp_path / "v.mp4", SceneGenerator(320, 180, seed=2).frames(120))
    target = store.path(GcsRef("project-raw", ATTRS["objectId"]))
    target.parent.mkdir(parents=True)
    target.write_bytes(video.read_bytes())
    target.with_name(target.name + ".meta.json").write_text(
        json.dumps({"capture-start": "2026-10-21T10:00:00Z", "task-whitelist": "cup"})
    )
    config = PipelineConfig.load(cloud={"scratch_dir": str(tmp_path / "scratch")})
    return config, store


def _processor(
    config: PipelineConfig, store: LocalObjectStore, sink: MemoryAuditSink | None = None
) -> SessionProcessor:
    backend = ScriptedBackend.from_mapping({"segments": [
        {"start": 0, "end": 119, "objects": [{"class": "person", "bbox": [40, 20, 160, 175]}]},
        {"start": 75, "end": 75, "faces": [{"bbox": [1, 1, 5, 5], "score": 0.99, "age": 6}]}]})  # fmt: skip
    return SessionProcessor(config, store, backend, sink or MemoryAuditSink(), query_metadata_server=False)


def test_processor_end_to_end(cloud_env: tuple[PipelineConfig, LocalObjectStore]) -> None:
    config, store = cloud_env
    sink = MemoryAuditSink()
    result = _processor(config, store, sink).process(parse_notification(ATTRS, b""))
    assert result.outcome is Outcome.PROCESSED
    prefix = f"sessions/{result.session_id}/"
    mcap = store.path(GcsRef("project-clean", prefix + "session.mcap"))
    lineage = json.loads(store.path(GcsRef("project-clean", prefix + "lineage.json")).read_text())
    assert verify_mcap(mcap).ok
    assert lineage["output"]["mcap_sha256"] == sha256_path(mcap)
    assert lineage["source"]["sha256"] == sha256_path(store.path(GcsRef("project-raw", ATTRS["objectId"])))
    assert lineage["source"]["gcs_generation"] == "1700000000"
    assert lineage["processing"]["stats"]["total_frames"] == 120
    child = [r for r in sink.rows if r["reason"] == "CHILD_DETECTED"]
    assert child[0]["dropped_range"] == [45, 105] and child[0]["event_time"].startswith("2026-10-21T10:00:02.5")
    assert not any((Path(config.cloud.scratch_dir)).iterdir())  # scratch cleaned


def test_processor_idempotent_and_filters(cloud_env: tuple[PipelineConfig, LocalObjectStore]) -> None:
    config, store = cloud_env
    proc = _processor(config, store)
    assert proc.process(parse_notification(ATTRS, b"")).outcome is Outcome.PROCESSED
    again = proc.process(parse_notification(ATTRS, b""))
    assert again.outcome is Outcome.SKIPPED and again.detail == "already processed"
    other = parse_notification({**ATTRS, "objectId": "notes.txt"}, b"")
    assert proc.process(other).outcome is Outcome.SKIPPED
    delete = parse_notification({**ATTRS, "eventType": "OBJECT_DELETE"}, b"")
    assert proc.process(delete).outcome is Outcome.SKIPPED


def test_processor_invalid_video(cloud_env: tuple[PipelineConfig, LocalObjectStore]) -> None:
    config, store = cloud_env
    bad = store.path(GcsRef("project-raw", "broken.mp4"))
    bad.write_bytes(b"not a video")
    result = _processor(config, store).process(parse_notification({**ATTRS, "objectId": "broken.mp4"}, b""))
    assert result.outcome is Outcome.INVALID


class FakeMessage:
    def __init__(self, attributes: dict[str, str], data: bytes = b"") -> None:
        self.attributes, self.data, self.message_id = attributes, data, "m1"
        self.acked = self.nacked = False

    def ack(self) -> None:
        self.acked = True

    def nack(self) -> None:
        self.nacked = True


def test_message_handler_ack_nack(cloud_env: tuple[PipelineConfig, LocalObjectStore]) -> None:
    config, store = cloud_env
    shutdown = threading.Event()
    handler = MessageHandler(_processor(config, store), shutdown)
    ok = FakeMessage(ATTRS)
    handler(ok)
    assert ok.acked
    junk = FakeMessage({"hello": "world"})
    handler(junk)
    assert junk.acked  # unparseable -> never retried
    shutdown.set()
    late = FakeMessage({**ATTRS, "objectGeneration": "2"})
    handler(late)
    assert late.nacked  # preempted pod hands the work back


def test_message_handler_nacks_on_cancel(cloud_env: tuple[PipelineConfig, LocalObjectStore]) -> None:
    config, store = cloud_env
    shutdown = threading.Event()
    handler = MessageHandler(_processor(config, store), shutdown)
    original = handler.processor.process

    def process_then_cancel(event: Any, cancel: threading.Event | None = None) -> Any:
        shutdown.set()  # SIGTERM arrives mid-session
        return original(event, cancel=cancel)

    handler.processor.process = process_then_cancel  # type: ignore[method-assign]
    msg = FakeMessage({**ATTRS, "objectGeneration": "3"})
    shutdown.clear()
    handler(msg)
    assert msg.nacked and not msg.acked
    assert not store.exists(GcsRef("project-clean", "sessions/site-a_cam1_2026-10-21-g3/lineage.json"))


class FakeBigQuery:
    def __init__(self, errors: list[Any] | None = None) -> None:
        self.calls: list[tuple[str, list[dict[str, Any]], list[str | None]]] = []
        self.errors = errors or []

    def insert_rows_json(self, table: str, rows: list[dict[str, Any]], row_ids: list[str | None]) -> list[Any]:
        self.calls.append((table, rows, row_ids))
        return self.errors


def test_bigquery_sink_batches_and_strips_insert_id() -> None:
    client = FakeBigQuery()
    sink = BigQueryAuditSink("p.d.frame_audit_log", batch_rows=2, client=client)
    rows = [{"insert_id": f"id{i}", "reason": "BLURRY", "dropped_range": [i, i]} for i in range(5)]
    sink.write_rows(rows)
    sink.flush()
    assert [len(c[1]) for c in client.calls] == [2, 2, 1]
    assert all("insert_id" not in r for c in client.calls for r in c[1])
    assert client.calls[0][2] == ["id0", "id1"]


def test_bigquery_sink_raises_on_rejected_rows() -> None:
    sink = BigQueryAuditSink("t", client=FakeBigQuery(errors=[{"index": 0, "errors": ["bad"]}]))
    sink.write_rows([{"insert_id": "x"}])
    with pytest.raises(BigQueryInsertError):
        sink.flush()


def test_sync_engines_and_pin(tmp_path: Path) -> None:
    store = LocalObjectStore(tmp_path / "gcs")
    base = store.path(GcsRef("project-models", "engines/trt-10.7/x"))
    base.parent.mkdir(parents=True)
    (base.parent / "yolov8n_fp16.engine").write_bytes(b"engine-bytes")
    sha = sha256_path(base.parent / "yolov8n_fp16.engine")
    (base.parent / "manifest.json").write_text(
        json.dumps({"engines": [{"engine": "yolov8n_fp16.engine", "sha256": sha}]})
    )
    hashes = sync_engines(store, "gs://project-models/engines/trt-10.7", tmp_path / "engines")
    assert hashes == {"yolov8n_fp16.engine": sha}
    pinned = pin_hashes(PipelineConfig().inference, hashes)
    assert pinned.detector.sha256 == sha and pinned.pose.sha256 is None
    (base.parent / "manifest.json").write_text(
        json.dumps({"engines": [{"engine": "yolov8n_fp16.engine", "sha256": "0" * 64}]})
    )
    (tmp_path / "engines" / "yolov8n_fp16.engine").unlink()
    with pytest.raises(ValueError):
        sync_engines(store, "gs://project-models/engines/trt-10.7", tmp_path / "engines")
