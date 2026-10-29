"""Structural verification of a session .mcap (what Foxglove Studio needs to
open it, plus the pipeline's own invariants).

Checks: indexed summary present; every channel protobuf-encoded with an
embedded schema; Foxglove well-known schema names on image channels; every
message decodes; per-topic timestamps non-decreasing; JPEGs decode; image /
detections / metadata counts agree; no kept frame falls inside a
CHILD_DETECTED drop range; no schema carries face/age/embedding fields.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

from physical_ai_pipeline.output.mcap_writer import (
    TOPIC_ACTIONS,
    TOPIC_ANNOTATIONS,
    TOPIC_DEPTH,
    TOPIC_DETECTIONS,
    TOPIC_DROP_LOG,
    TOPIC_IMAGE,
    TOPIC_METADATA,
    TOPIC_QUALITY,
)
from physical_ai_pipeline.proto_gen import physical_ai_pb2 as pb

REQUIRED_TOPICS = {
    TOPIC_IMAGE: "foxglove.CompressedImage",
    TOPIC_DETECTIONS: "physical_ai.v1.FrameDetections",
    TOPIC_QUALITY: "physical_ai.v1.FrameQuality",
    TOPIC_METADATA: "physical_ai.v1.SessionMetadata",
}
OPTIONAL_TOPICS = {
    TOPIC_ANNOTATIONS: "foxglove.ImageAnnotations",
    TOPIC_DEPTH: "foxglove.RawImage",
    TOPIC_ACTIONS: "physical_ai.v1.ActionSegment",
    TOPIC_DROP_LOG: "physical_ai.v1.DropEvent",
}
FORBIDDEN_FIELDS = {"age", "age_estimate", "estimated_age", "embedding", "face_embedding", "face_crop", "face_image"}


@dataclass(slots=True)
class VerifyReport:
    path: str
    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    topics: dict[str, dict[str, Any]] = field(default_factory=dict)
    kept_frames: int = 0
    child_blocks: list[list[int]] = field(default_factory=list)

    def fail(self, message: str) -> None:
        self.ok = False
        self.errors.append(message)

    def to_json(self) -> str:
        return json.dumps(
            {"path": self.path, "ok": self.ok, "errors": self.errors, "warnings": self.warnings,
             "topics": self.topics, "kept_frames": self.kept_frames, "child_blocks": self.child_blocks},
            indent=2,
        )  # fmt: skip


def _schema_field_names(schema_data: bytes) -> set[str]:
    from google.protobuf import descriptor_pb2

    fds = descriptor_pb2.FileDescriptorSet.FromString(schema_data)
    names: set[str] = set()

    def walk(messages: Any) -> None:
        for message in messages:
            names.update(f.name for f in message.field)
            walk(message.nested_type)

    for file in fds.file:
        walk(file.message_type)
    return names


def verify_mcap(path: str | Path, decode_images: int | None = None) -> VerifyReport:
    """``decode_images``: decode at most this many JPEGs (None = all)."""
    report = VerifyReport(str(path))
    with open(path, "rb") as fh:
        reader = make_reader(fh, decoder_factories=[DecoderFactory()])
        summary = reader.get_summary()
        if summary is None:
            report.fail("missing summary section (file not finalized / not indexed)")
            return report
        if not summary.chunk_indexes:
            report.warnings.append("no chunk indexes: seeking in Foxglove will be slow")
        schemas = summary.schemas
        by_topic = {ch.topic: ch for ch in summary.channels.values()}
        for topic, expected in {**REQUIRED_TOPICS, **OPTIONAL_TOPICS}.items():
            channel = by_topic.get(topic)
            if channel is None:
                if topic in REQUIRED_TOPICS:
                    report.fail(f"missing required topic {topic}")
                continue
            schema = schemas[channel.schema_id]
            if schema.name != expected:
                report.fail(f"{topic}: schema {schema.name!r}, expected {expected!r}")
            if schema.encoding != "protobuf" or channel.message_encoding != "protobuf":
                report.fail(f"{topic}: not protobuf-encoded ({schema.encoding}/{channel.message_encoding})")
        for schema in schemas.values():
            bad = _schema_field_names(schema.data) & FORBIDDEN_FIELDS
            if bad:
                report.fail(f"schema {schema.name} carries forbidden face/age fields: {sorted(bad)}")

        last_time: dict[str, int] = {}
        counts: dict[str, int] = defaultdict(int)
        image_frames: list[int] = []
        image_ns: list[int] = []
        detection_frames: list[int] = []
        child_blocks: list[tuple[int, int]] = []
        metadata: list[pb.SessionMetadata] = []
        decoded_images = 0
        shape: tuple[int, ...] | None = None
        for _schema, channel, message, proto in reader.iter_decoded_messages(log_time_order=False):
            topic = channel.topic
            counts[topic] += 1
            if topic in (TOPIC_IMAGE, TOPIC_DETECTIONS, TOPIC_QUALITY, TOPIC_ANNOTATIONS, TOPIC_DEPTH):
                if message.log_time < last_time.get(topic, -1):
                    report.fail(f"{topic}: log_time decreases at {message.log_time}")
                last_time[topic] = message.log_time
            if topic == TOPIC_IMAGE:
                image_ns.append(message.log_time)
                if proto.format != "jpeg":
                    report.fail(f"image format {proto.format!r}")
                if decode_images is None or decoded_images < decode_images:
                    img = cv2.imdecode(np.frombuffer(proto.data, np.uint8), cv2.IMREAD_COLOR)
                    decoded_images += 1
                    if img is None:
                        report.fail(f"undecodable JPEG at log_time {message.log_time}")
                    elif shape is None:
                        shape = img.shape
                    elif img.shape != shape:
                        report.fail(f"image size changed {shape} -> {img.shape}")
            elif topic == TOPIC_DETECTIONS:
                detection_frames.append(int(proto.frame_index))
                if proto.timestamp.ToNanoseconds() != message.log_time:
                    report.fail(f"/detections frame {proto.frame_index}: header stamp != log_time")
            elif topic == TOPIC_DROP_LOG and proto.reason == pb.DROP_REASON_CHILD_DETECTED:
                child_blocks.append((int(proto.start_frame), int(proto.end_frame)))
            elif topic == TOPIC_METADATA:
                metadata.append(proto)
        image_frames = detection_frames

        report.kept_frames = counts[TOPIC_IMAGE]
        report.child_blocks = [list(b) for b in child_blocks]
        report.topics = {t: {"messages": n} for t, n in sorted(counts.items())}
        if counts[TOPIC_IMAGE] != counts[TOPIC_DETECTIONS]:
            report.fail(f"{counts[TOPIC_IMAGE]} images vs {counts[TOPIC_DETECTIONS]} detections")
        if len(metadata) != 1:
            report.fail(f"expected exactly one SessionMetadata, found {len(metadata)}")
        elif metadata[0].stats.kept_frames != counts[TOPIC_IMAGE]:
            report.fail(f"metadata kept_frames {metadata[0].stats.kept_frames} != images {counts[TOPIC_IMAGE]}")
        leaked = [f for f in image_frames for s, e in child_blocks if s <= f <= e]
        if leaked:
            report.fail(f"{len(leaked)} kept frames inside CHILD_DETECTED ranges, e.g. {leaked[:5]}")
        if image_ns != sorted(image_ns):
            report.fail("image timestamps not monotonic")
    return report
