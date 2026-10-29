"""Box geometry helpers shared by inference post-processing and filters."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from physical_ai_pipeline.frame import BBox


def iou(a: BBox, b: BBox) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def area(b: BBox) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def center(b: BBox) -> tuple[float, float]:
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def diagonal(b: BBox) -> float:
    return float(np.hypot(b[2] - b[0], b[3] - b[1]))


def point_box(x: float, y: float, side: float) -> BBox:
    half = side / 2.0
    return (x - half, y - half, x + half, y + half)


def nms(boxes: npt.NDArray[np.float32], scores: npt.NDArray[np.float32], iou_threshold: float) -> list[int]:
    """Greedy NMS over xyxy boxes. Returns kept indices sorted by score."""
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        ious = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-9)
        order = rest[ious <= iou_threshold]
    return keep
