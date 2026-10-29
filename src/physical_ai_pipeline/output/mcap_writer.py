"""Protobuf MCAP writer: one indexed, chunked, zstd-compressed .mcap per session.

Channels
  /camera/image_compressed  foxglove.CompressedImage   JPEG kept frames
  /camera/annotations       foxglove.ImageAnnotations  bbox / keypoint overlay
  /camera/depth             foxglove.RawImage          16UC1 relative depth (optional)
  /detections               physical_ai.v1.FrameDetections
  /quality                  physical_ai.v1.FrameQuality
  /actions                  physical_ai.v1.ActionSegment
  /pipeline/drop_log        physical_ai.v1.DropEvent
  /session/metadata         physical_ai.v1.SessionMetadata

All messages are stamped ``log_time = publish_time = session_start_ns + pts_ns``.
JPEG encoding runs on a thread pool (OpenCV releases the GIL); frames are
written in index order and their pixel buffers freed right after encoding.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import BinaryIO

import cv2
import numpy as np
from mcap.writer import CompressionType
from mcap_protobuf.writer import Writer

from physical_ai_pipeline.config import OutputConfig
from physical_ai_pipeline.frame import FrameRecord
from physical_ai_pipeline.inference.postprocess import depth_to_uint16
from physical_ai_pipeline.output import convert
from physical_ai_pipeline.output.drop_log import DropRange
from physical_ai_pipeline.postprocess.grasp import ActionSegmentData
from physical_ai_pipeline.proto_gen import physical_ai_pb2 as pb
from physical_ai_pipeline.proto_gen.foxglove import foxglove_pb2 as fg

TOPIC_IMAGE = "/camera/image_compressed"
TOPIC_ANNOTATIONS = "/camera/annotations"
TOPIC_DEPTH = "/camera/depth"
TOPIC_DETECTIONS = "/detections"
TOPIC_QUALITY = "/quality"
TOPIC_ACTIONS = "/actions"
TOPIC_DROP_LOG = "/pipeline/drop_log"
TOPIC_METADATA = "/session/metadata"


def encode_jpeg(image: np.ndarray, quality: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return bytes(buf.tobytes())


class SessionMcapWriter:
    def __init__(self, path: str | Path, config: OutputConfig, session_start_ns: int) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.base_ns = int(session_start_ns)
        self._fh: BinaryIO = open(self.path, "wb")  # noqa: SIM115 - closed in close()
        self._writer = Writer(self._fh, chunk_size=config.mcap_chunk_size, compression=CompressionType.ZSTD)
        self._pool = ThreadPoolExecutor(max_workers=config.encode_threads, thread_name_prefix="jpeg")
        self._inflight: deque[tuple[FrameRecord, Future[bytes] | None]] = deque()
        self._max_inflight = config.encode_threads * 4
        self.frames_written = 0
        self.message_counts: dict[str, int] = {}
        self._closed = False

    def log_ns(self, record: FrameRecord) -> int:
        return self.base_ns + record.pts_ns

    def _write(self, topic: str, message: object, log_ns: int) -> None:
        self._writer.write_message(topic, message, log_time=log_ns, publish_time=log_ns)
        self.message_counts[topic] = self.message_counts.get(topic, 0) + 1

    # -- frames ---------------------------------------------------------------

    def write_kept_frame(self, record: FrameRecord) -> None:
        if record.image is None:
            raise ValueError(f"kept frame {record.index} has no pixels")
        future = self._pool.submit(encode_jpeg, record.image, self.config.jpeg_quality)
        self._inflight.append((record, future))
        self._drain(block=len(self._inflight) > self._max_inflight)

    def write_dropped_frame(self, record: FrameRecord) -> None:
        """Quality metrics for non-child drops; child-excluded frames emit nothing.
        Queued behind in-flight JPEG encodes so /quality stays time-ordered."""
        record.release_pixels(force=True)
        self._inflight.append((record, None))
        self._drain(block=len(self._inflight) > self._max_inflight)

    def _drain(self, block: bool, everything: bool = False) -> None:
        while self._inflight:
            record, future = self._inflight[0]
            if future is not None and not (block or future.done()):
                return
            self._inflight.popleft()
            if future is None:
                if not record.child_dropped:
                    ns = self.log_ns(record)
                    self._write(TOPIC_QUALITY, convert.frame_quality(record, ns), ns)
            else:
                self._emit_frame(record, future.result())
            block = everything or len(self._inflight) > self._max_inflight

    def _emit_frame(self, record: FrameRecord, jpeg: bytes) -> None:
        ns = self.log_ns(record)
        ts = convert.timestamp(ns)
        frame_id = self.config.frame_id
        self._write(TOPIC_IMAGE, fg.CompressedImage(timestamp=ts, frame_id=frame_id, data=jpeg, format="jpeg"), ns)
        self._write(TOPIC_DETECTIONS, convert.frame_detections(record, ns, frame_id), ns)
        self._write(TOPIC_QUALITY, convert.frame_quality(record, ns), ns)
        if self.config.write_annotations and record.inference is not None:
            self._write(TOPIC_ANNOTATIONS, convert.image_annotations(record, ns), ns)
        depth = record.inference.depth if record.inference is not None else None
        if depth is not None:
            d16 = depth_to_uint16(depth)
            h, w = d16.shape
            self._write(
                TOPIC_DEPTH,
                fg.RawImage(
                    timestamp=ts, frame_id=frame_id, width=w, height=h, encoding="16UC1", step=w * 2, data=d16.tobytes()
                ),
                ns,
            )
        self.frames_written += 1
        record.release_pixels(force=True)

    # -- session-level channels ----------------------------------------------

    def write_drop_event(self, drop: DropRange, anchor_frames: Sequence[int] = ()) -> None:
        ns = self.base_ns + drop.start_pts_ns
        msg = pb.DropEvent(
            timestamp=convert.timestamp(ns),
            start_frame=drop.start_frame,
            end_frame=drop.end_frame,
            reason=convert.DROP_REASON_TO_PB[drop.reason],
            all_reasons=sorted(convert.DROP_REASON_TO_PB[r] for r in drop.all_reasons),
            mode=convert.FILTER_MODE_TO_PB[drop.mode],
            anchor_frames=list(anchor_frames),
            frame_count=drop.frame_count,
        )
        self._write(TOPIC_DROP_LOG, msg, ns)

    def write_actions(self, segments: Sequence[ActionSegmentData]) -> None:
        for seg in segments:
            msg = convert.action_segment(seg, self.base_ns)
            self._write(TOPIC_ACTIONS, msg, self.base_ns + seg.start_pts_ns)

    def write_metadata(self, metadata: pb.SessionMetadata) -> None:
        self._write(TOPIC_METADATA, metadata, self.base_ns)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._drain(block=True, everything=True)
        finally:
            self._pool.shutdown(wait=True)
            self._writer.finish()  # type: ignore[no-untyped-call]
            self._fh.close()

    def abort(self) -> None:
        """Close without flushing pending frames and delete the partial file."""
        self._closed = True
        for _, future in self._inflight:
            if future is not None:
                future.cancel()
        self._inflight.clear()
        self._pool.shutdown(wait=True, cancel_futures=True)
        try:
            self._fh.close()
        finally:
            self.path.unlink(missing_ok=True)
