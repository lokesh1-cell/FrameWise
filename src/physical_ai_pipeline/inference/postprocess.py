"""Decoders for raw engine outputs (pure NumPy, unit-testable without a GPU)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from physical_ai_pipeline.frame import Detection, Pose
from physical_ai_pipeline.inference.geometry import nms
from physical_ai_pipeline.inference.preprocess import LetterboxParams

Array = npt.NDArray[np.float32]


def _xywh_to_xyxy(xywh: Array) -> Array:
    out = np.empty_like(xywh)
    out[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    out[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    out[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    out[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
    return out


def _clip(box: tuple[float, float, float, float], width: int, height: int) -> tuple[float, float, float, float]:
    return (
        min(max(box[0], 0.0), float(width)),
        min(max(box[1], 0.0), float(height)),
        min(max(box[2], 0.0), float(width)),
        min(max(box[3], 0.0), float(height)),
    )


def decode_yolo_detections(
    output: Array,
    params: Sequence[LetterboxParams],
    image_sizes: Sequence[tuple[int, int]],
    class_names: Sequence[str],
    conf_threshold: float,
    iou_threshold: float,
    max_detections: int,
) -> list[list[Detection]]:
    """YOLOv8 detect head: ``output`` is (B, 4 + num_classes, N), xywh in
    letterboxed pixels. Class-aware NMS. ``image_sizes`` are (height, width)."""
    results: list[list[Detection]] = []
    for b in range(output.shape[0]):
        pred = output[b].T  # (N, 4 + C)
        scores_all = pred[:, 4:]
        class_ids = scores_all.argmax(axis=1)
        scores = scores_all[np.arange(pred.shape[0]), class_ids]
        mask = scores > conf_threshold
        boxes = _xywh_to_xyxy(pred[mask, :4])
        scores, class_ids = scores[mask], class_ids[mask]
        # Offset boxes per class so one NMS pass is class-aware.
        offset = class_ids[:, None].astype(np.float32) * 4096.0
        keep = nms(boxes + offset, scores, iou_threshold)[:max_detections]
        h, w = image_sizes[b]
        dets = [
            Detection(
                bbox=_clip(params[b].to_source(boxes[i]), w, h),
                confidence=float(scores[i]),
                class_id=int(class_ids[i]),
                class_name=class_names[int(class_ids[i])]
                if int(class_ids[i]) < len(class_names)
                else str(int(class_ids[i])),
            )
            for i in keep
        ]
        results.append(dets)
    return results


def decode_yolo_pose(
    output: Array,
    params: Sequence[LetterboxParams],
    image_sizes: Sequence[tuple[int, int]],
    num_keypoints: int,
    conf_threshold: float,
    iou_threshold: float,
    max_detections: int,
) -> list[list[Pose]]:
    """YOLOv8 pose head: ``output`` is (B, 5 + K*3, N): xywh, conf, K x (x, y, v)."""
    expected = 5 + num_keypoints * 3
    if output.shape[1] != expected:
        raise ValueError(f"pose output has {output.shape[1]} channels, expected {expected}")
    results: list[list[Pose]] = []
    for b in range(output.shape[0]):
        pred = output[b].T
        scores = pred[:, 4]
        mask = scores > conf_threshold
        boxes = _xywh_to_xyxy(pred[mask, :4])
        scores = scores[mask]
        kpts = pred[mask, 5:].reshape(-1, num_keypoints, 3)
        keep = nms(boxes, scores, iou_threshold)[:max_detections]
        h, w = image_sizes[b]
        poses: list[Pose] = []
        for i in keep:
            kp = kpts[i].copy()
            kp[:, :2] = params[b].points_to_source(kp[:, :2])
            poses.append(
                Pose(
                    bbox=_clip(params[b].to_source(boxes[i]), w, h),
                    confidence=float(scores[i]),
                    keypoints=kp.astype(np.float32),
                )
            )
        results.append(poses)
    return results


# ---------------------------------------------------------------------------
# InsightFace SCRFD (det_10g) and genderage
# ---------------------------------------------------------------------------

SCRFD_STRIDES = (8, 16, 32)
SCRFD_ANCHORS = 2


def _anchor_centers(size: int, stride: int, num_anchors: int) -> Array:
    n = size // stride
    ys, xs = np.mgrid[:n, :n]
    centers = (np.stack([xs, ys], axis=-1).reshape(-1, 2) * stride).astype(np.float32)
    return np.repeat(centers, num_anchors, axis=0)


def decode_scrfd(
    outputs: Sequence[Array],
    params: LetterboxParams,
    input_size: int,
    conf_threshold: float,
    iou_threshold: float,
) -> tuple[Array, Array]:
    """Decode SCRFD outputs ordered [score_8, score_16, score_32, bbox_8, ...,
    kps_8, ...] for a single image. Returns (boxes xyxy in source px, scores)."""
    if len(outputs) not in (6, 9):
        raise ValueError(f"SCRFD expects 6 or 9 outputs, got {len(outputs)}")
    fmc = len(SCRFD_STRIDES)
    all_boxes: list[Array] = []
    all_scores: list[Array] = []
    for idx, stride in enumerate(SCRFD_STRIDES):
        scores = np.asarray(outputs[idx]).reshape(-1)
        deltas = np.asarray(outputs[idx + fmc]).reshape(-1, 4) * stride
        centers = _anchor_centers(input_size, stride, SCRFD_ANCHORS)
        if centers.shape[0] != scores.shape[0]:
            raise ValueError(f"SCRFD stride {stride}: {scores.shape[0]} scores vs {centers.shape[0]} anchors")
        mask = scores >= conf_threshold
        c, d = centers[mask], deltas[mask]
        boxes = np.stack([c[:, 0] - d[:, 0], c[:, 1] - d[:, 1], c[:, 0] + d[:, 2], c[:, 1] + d[:, 3]], axis=1)
        all_boxes.append(boxes.astype(np.float32))
        all_scores.append(scores[mask].astype(np.float32))
    boxes = np.concatenate(all_boxes) if all_boxes else np.zeros((0, 4), np.float32)
    scores = np.concatenate(all_scores) if all_scores else np.zeros((0,), np.float32)
    keep = nms(boxes, scores, iou_threshold)
    boxes, scores = boxes[keep], scores[keep]
    if boxes.size:
        boxes = ((boxes - np.array([params.pad_x, params.pad_y] * 2, np.float32)) / params.scale).astype(np.float32)
    return boxes, scores.astype(np.float32)


def decode_genderage(output: Array) -> Array:
    """genderage head: (N, 3) = [female_logit, male_logit, age / 100]. Returns ages."""
    return (np.asarray(output, dtype=np.float32).reshape(-1, 3)[:, 2] * 100.0).astype(np.float32)


def depth_to_uint16(depth: Array) -> npt.NDArray[np.uint16]:
    """Normalise relative inverse depth to the full uint16 range for 16UC1 output."""
    lo, hi = float(np.min(depth)), float(np.max(depth))
    if hi - lo < 1e-6:
        return np.zeros(depth.shape, dtype=np.uint16)
    return ((depth - lo) / (hi - lo) * 65535.0).astype(np.uint16)
