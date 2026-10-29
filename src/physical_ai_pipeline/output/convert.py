"""Conversions from pipeline dataclasses to compiled Protobuf messages."""

from __future__ import annotations

from typing import Any

from google.protobuf.timestamp_pb2 import Timestamp

from physical_ai_pipeline.config import FilterMode
from physical_ai_pipeline.frame import BBox, DropReason, FrameRecord, Hand
from physical_ai_pipeline.inference.coco import COCO_KEYPOINTS, HAND_KEYPOINTS
from physical_ai_pipeline.postprocess.grasp import ActionSegmentData, InteractionState
from physical_ai_pipeline.proto_gen import physical_ai_pb2 as pb
from physical_ai_pipeline.proto_gen.foxglove import foxglove_pb2 as fg

# Enum values are typed loosely: the generated .pyi wraps them in enum types.
DROP_REASON_TO_PB: dict[DropReason, Any] = {r: pb.DropReason.Value(f"DROP_REASON_{r.value}") for r in DropReason}
FILTER_MODE_TO_PB: dict[FilterMode, Any] = {m: pb.FilterMode.Value(f"FILTER_MODE_{m.value}") for m in FilterMode}
HAND_TO_PB: dict[Hand, Any] = {
    Hand.LEFT: pb.HANDEDNESS_LEFT,
    Hand.RIGHT: pb.HANDEDNESS_RIGHT,
    Hand.UNKNOWN: pb.HANDEDNESS_UNSPECIFIED,
}
STATE_TO_PB: dict[str, Any] = {
    s.value: pb.InteractionState.Value(f"INTERACTION_STATE_{s.value}") for s in InteractionState
}


def timestamp(ns: int) -> Timestamp:
    ts = Timestamp()
    ts.FromNanoseconds(int(ns))
    return ts


def bbox(b: BBox) -> pb.BoundingBox:
    return pb.BoundingBox(x_min=b[0], y_min=b[1], x_max=b[2], y_max=b[3])


def frame_detections(record: FrameRecord, log_ns: int, frame_id: str) -> pb.FrameDetections:
    msg = pb.FrameDetections(
        timestamp=timestamp(log_ns),
        frame_id=frame_id,
        frame_index=record.index,
        child_check_performed=record.child_check_performed,
    )
    inference = record.inference
    if inference is None:
        return msg
    msg.person_count = inference.person_count
    for det in inference.objects:
        msg.objects.add(
            class_id=det.class_id, class_name=det.class_name, confidence=det.confidence, bbox=bbox(det.bbox)
        )
    for pose in inference.poses:
        p = msg.poses.add(confidence=pose.confidence, bbox=bbox(pose.bbox))
        for name, (x, y, c) in zip(COCO_KEYPOINTS, pose.keypoints, strict=False):
            p.keypoints.add(x=float(x), y=float(y), confidence=float(c), name=name)
    for hand in inference.hands:
        h = msg.hands.add(confidence=hand.confidence, bbox=bbox(hand.bbox), handedness=HAND_TO_PB[hand.handedness])
        for name, (x, y, c) in zip(HAND_KEYPOINTS, hand.keypoints, strict=False):
            h.keypoints.add(x=float(x), y=float(y), confidence=float(c), name=name)
    for g in record.grasps:
        msg.grasps.add(
            hand_index=g.hand_index,
            object_index=g.object_index,
            hand_object_iou=g.iou,
            state=STATE_TO_PB[g.state],
            grasp_confidence=g.confidence,
            normalized_distance=g.normalized_distance,
        )
    return msg


def frame_quality(record: FrameRecord, log_ns: int) -> pb.FrameQuality:
    q = record.quality
    return pb.FrameQuality(
        timestamp=timestamp(log_ns),
        frame_index=record.index,
        decode_ok=q.decode_ok,
        corrupt_block_ratio=q.corrupt_block_ratio,
        luminance_mean=q.luminance_mean,
        laplacian_variance=q.laplacian_variance,
        phash_hamming_distance=q.phash_hamming,
        optical_flow_magnitude=q.flow_magnitude,
        kept=not record.dropped,
    )


_COLORS = {
    "object": fg.Color(r=0.1, g=0.6, b=1.0, a=1.0),
    "person": fg.Color(r=0.2, g=0.9, b=0.3, a=1.0),
    "hand": fg.Color(r=1.0, g=0.6, b=0.1, a=1.0),
    "text_bg": fg.Color(r=0.0, g=0.0, b=0.0, a=0.6),
    "text": fg.Color(r=1.0, g=1.0, b=1.0, a=1.0),
}


def _box_annotation(b: BBox, ts: Timestamp, color: fg.Color) -> fg.PointsAnnotation:
    return fg.PointsAnnotation(
        timestamp=ts,
        type=fg.PointsAnnotation.LINE_LOOP,
        points=[
            fg.Point2(x=b[0], y=b[1]),
            fg.Point2(x=b[2], y=b[1]),
            fg.Point2(x=b[2], y=b[3]),
            fg.Point2(x=b[0], y=b[3]),
        ],
        outline_color=color,
        thickness=2.0,
    )


def image_annotations(record: FrameRecord, log_ns: int) -> fg.ImageAnnotations:
    """Foxglove overlay: object/hand boxes, body + hand keypoints, labels."""
    ts = timestamp(log_ns)
    ann = fg.ImageAnnotations()
    inference = record.inference
    if inference is None:
        return ann
    for det in inference.objects:
        ann.points.append(_box_annotation(det.bbox, ts, _COLORS["object"]))
        ann.texts.append(
            fg.TextAnnotation(
                timestamp=ts, position=fg.Point2(x=det.bbox[0], y=det.bbox[1]),
                text=f"{det.class_name} {det.confidence:.2f}", font_size=14.0,
                text_color=_COLORS["text"], background_color=_COLORS["text_bg"],
            )
        )  # fmt: skip
    for pose in inference.poses:
        pts = [fg.Point2(x=float(x), y=float(y)) for x, y, c in pose.keypoints if c >= 0.5]
        if pts:
            ann.points.append(
                fg.PointsAnnotation(
                    timestamp=ts,
                    type=fg.PointsAnnotation.POINTS,
                    points=pts,
                    outline_color=_COLORS["person"],
                    thickness=6.0,
                )
            )
    for hand in inference.hands:
        ann.points.append(_box_annotation(hand.bbox, ts, _COLORS["hand"]))
        pts = [fg.Point2(x=float(x), y=float(y)) for x, y, c in hand.keypoints if c >= 0.5]
        if pts:
            ann.points.append(
                fg.PointsAnnotation(
                    timestamp=ts,
                    type=fg.PointsAnnotation.POINTS,
                    points=pts,
                    outline_color=_COLORS["hand"],
                    thickness=4.0,
                )
            )
    return ann


def action_segment(seg: ActionSegmentData, base_ns: int) -> pb.ActionSegment:
    counts = seg.phase_counts
    return pb.ActionSegment(
        start_time=timestamp(base_ns + seg.start_pts_ns),
        end_time=timestamp(base_ns + seg.end_pts_ns),
        start_frame=seg.start_frame,
        end_frame=seg.end_frame,
        label=seg.label,
        hand=HAND_TO_PB[seg.hand],
        object_class=seg.object_class,
        mean_grasp_confidence=seg.mean_grasp_confidence,
        approaching_frames=counts.get(InteractionState.APPROACHING, 0),
        grasping_frames=counts.get(InteractionState.GRASPING, 0),
        holding_frames=counts.get(InteractionState.HOLDING, 0),
        releasing_frames=counts.get(InteractionState.RELEASING, 0),
    )
