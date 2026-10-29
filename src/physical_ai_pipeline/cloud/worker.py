"""GKE worker: consumes GCS OBJECT_FINALIZE notifications from Pub/Sub.

One session at a time per pod (one L4 GPU per pod). Spot preemption sends
SIGTERM ~30 s before the node goes away: the worker stops pulling, cancels
the running session (partial outputs are deleted, nothing is uploaded) and
nacks the message so another pod redelivers it.

    python -m physical_ai_pipeline.cloud.worker --config /etc/pap/config.yaml
"""

from __future__ import annotations

import argparse
import signal
import threading
import time
from pathlib import Path
from types import FrameType
from typing import Any

from physical_ai_pipeline.cloud.bigquery_sink import BigQueryAuditSink
from physical_ai_pipeline.cloud.gcs import GoogleCloudStorage, ObjectStore
from physical_ai_pipeline.cloud.models import pin_hashes, sync_engines
from physical_ai_pipeline.cloud.notifications import parse_notification
from physical_ai_pipeline.cloud.processor import Outcome, SessionProcessor
from physical_ai_pipeline.config import PipelineConfig
from physical_ai_pipeline.inference.backend import InferenceBackend
from physical_ai_pipeline.logging_utils import configure_logging, get_logger, log_event
from physical_ai_pipeline.pipeline import SessionCancelled

_log = get_logger(__name__)
HEARTBEAT_PATH = Path("/tmp/pap-heartbeat")


class MessageHandler:
    """Pub/Sub callback logic, independent of the client library."""

    def __init__(self, processor: SessionProcessor, shutdown: threading.Event) -> None:
        self.processor = processor
        self.shutdown = shutdown
        self.busy = threading.Lock()

    def __call__(self, message: Any) -> None:
        with self.busy:
            if self.shutdown.is_set():
                message.nack()
                return
            try:
                event = parse_notification(dict(message.attributes), bytes(message.data or b""))
            except ValueError as exc:
                log_event(
                    _log, "message_ignored", level=30, error=str(exc), message_id=getattr(message, "message_id", None)
                )
                message.ack()
                return
            try:
                result = self.processor.process(event, cancel=self.shutdown)
            except SessionCancelled:
                log_event(_log, "session_cancelled_nack", level=30, object=event.uri)
                message.nack()
                return
            except Exception:
                _log.exception("session_error", extra={"fields": {"event": "session_error", "object": event.uri}})
                message.nack()  # transient: retried, then dead-lettered by the subscription policy
                return
            log_event(
                _log, "message_done", outcome=result.outcome.value, session_id=result.session_id, detail=result.detail
            )
            message.ack()
            if result.outcome is Outcome.INVALID:
                log_event(_log, "source_rejected", level=40, object=event.uri, detail=result.detail)


def _heartbeat(stop: threading.Event, interval: float = 15.0) -> None:
    while not stop.is_set():
        HEARTBEAT_PATH.touch()
        stop.wait(interval)


def build_backend(config: PipelineConfig) -> InferenceBackend:
    from physical_ai_pipeline.inference.tensorrt_backend import TensorRTBackend

    return TensorRTBackend(config.inference)


def run(config: PipelineConfig, store: ObjectStore | None = None) -> int:
    from google.cloud import pubsub_v1

    store = store or GoogleCloudStorage(config.cloud.project_id)
    hashes = sync_engines(store, config.cloud.models_uri, config.inference.engine_dir)
    config = config.model_copy(update={"inference": pin_hashes(config.inference, hashes)})
    backend = build_backend(config)
    audit = BigQueryAuditSink(config.cloud.bigquery_table, config.cloud.bigquery_batch_rows)
    processor = SessionProcessor(config, store, backend, audit)

    shutdown = threading.Event()
    handler = MessageHandler(processor, shutdown)

    def on_signal(signum: int, _frame: FrameType | None) -> None:
        log_event(_log, "shutdown_signal", level=30, signal=signum)
        shutdown.set()

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    threading.Thread(target=_heartbeat, args=(shutdown,), daemon=True, name="heartbeat").start()

    subscriber = pubsub_v1.SubscriberClient()
    path = subscriber.subscription_path(config.cloud.project_id, config.cloud.subscription)
    flow = pubsub_v1.types.FlowControl(max_messages=1, max_lease_duration=config.cloud.max_lease_seconds)
    future = subscriber.subscribe(path, callback=handler, flow_control=flow)
    log_event(_log, "worker_started", subscription=path, pipeline_version=config.pipeline_version)
    try:
        while not shutdown.is_set():
            shutdown.wait(1.0)
            if future.done():
                future.result()  # surfaces streaming-pull errors (pod restarts)
    finally:
        future.cancel()
        # Wait for the in-flight session to observe the cancel flag and nack.
        deadline = time.monotonic() + 20
        while handler.busy.locked() and time.monotonic() < deadline:
            time.sleep(0.2)
        subscriber.close()
        backend.close()
        log_event(_log, "worker_stopped")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    return run(PipelineConfig.load(args.config))


if __name__ == "__main__":
    raise SystemExit(main())
