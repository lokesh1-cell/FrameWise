"""CPU pre-processing into NCHW float tensors for the TensorRT engines."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

from physical_ai_pipeline.frame import BBox

Tensor = npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class LetterboxParams:
    scale: float
    pad_x: float
    pad_y: float

    def to_source(self, box: npt.NDArray[np.float32]) -> BBox:
        x1 = (float(box[0]) - self.pad_x) / self.scale
        y1 = (float(box[1]) - self.pad_y) / self.scale
        x2 = (float(box[2]) - self.pad_x) / self.scale
        y2 = (float(box[3]) - self.pad_y) / self.scale
        return (x1, y1, x2, y2)

    def points_to_source(self, xy: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        out = xy.astype(np.float32).copy()
        out[..., 0] = (out[..., 0] - self.pad_x) / self.scale
        out[..., 1] = (out[..., 1] - self.pad_y) / self.scale
        return out


def letterbox(
    image: npt.NDArray[np.uint8], size: int, *, center: bool = True, pad_value: int = 114
) -> tuple[npt.NDArray[np.uint8], LetterboxParams]:
    """Resize keeping aspect ratio into a ``size`` x ``size`` canvas.

    ``center=True`` matches Ultralytics; ``center=False`` (top-left placement,
    pad 0) matches InsightFace SCRFD.
    """
    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    nw, nh = round(w * scale), round(h * scale)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), pad_value, dtype=np.uint8)
    pad_x = (size - nw) / 2 if center else 0.0
    pad_y = (size - nh) / 2 if center else 0.0
    top, left = int(round(pad_y - 0.1)), int(round(pad_x - 0.1))
    canvas[top : top + nh, left : left + nw] = resized
    return canvas, LetterboxParams(scale, float(left), float(top))


def yolo_batch(images: Sequence[npt.NDArray[np.uint8]], size: int) -> tuple[Tensor, list[LetterboxParams]]:
    """BGR uint8 -> RGB float32 NCHW in [0, 1], Ultralytics letterbox."""
    tensors: list[Tensor] = []
    params: list[LetterboxParams] = []
    for image in images:
        boxed, p = letterbox(image, size)
        tensors.append(boxed[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0)
        params.append(p)
    return np.ascontiguousarray(np.stack(tensors)), params


def scrfd_input(image: npt.NDArray[np.uint8], size: int) -> tuple[Tensor, LetterboxParams]:
    """SCRFD: top-left letterbox, RGB, (x - 127.5) / 128."""
    boxed, p = letterbox(image, size, center=False, pad_value=0)
    rgb = boxed[:, :, ::-1].astype(np.float32)
    return np.ascontiguousarray(((rgb - 127.5) / 128.0).transpose(2, 0, 1)[None]), p


def genderage_crops(image: npt.NDArray[np.uint8], boxes: Sequence[BBox], size: int) -> Tensor:
    """InsightFace attribute crop: centre on the face box, scale = size / (1.5 * max side),
    RGB, raw 0-255 (the genderage graph normalises internally).

    The returned tensor lives only in RAM for the duration of one inference call.
    """
    crops: list[Tensor] = []
    for box in boxes:
        w, h = box[2] - box[0], box[3] - box[1]
        cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        scale = size / (max(w, h) * 1.5 + 1e-6)
        matrix = np.array(
            [[scale, 0.0, size / 2.0 - cx * scale], [0.0, scale, size / 2.0 - cy * scale]], dtype=np.float32
        )
        crop = cv2.warpAffine(image, matrix, (size, size), borderValue=0.0)
        crops.append(crop[:, :, ::-1].transpose(2, 0, 1).astype(np.float32))
    if not crops:
        return np.zeros((0, 3, size, size), dtype=np.float32)
    return np.ascontiguousarray(np.stack(crops))


def depth_batch(images: Sequence[npt.NDArray[np.uint8]], size: int) -> Tensor:
    """Depth Anything V2: resize to size x size, RGB, ImageNet mean/std."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    out = []
    for image in images:
        rgb = cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)[:, :, ::-1].astype(np.float32) / 255.0
        out.append(((rgb - mean) / std).transpose(2, 0, 1))
    return np.ascontiguousarray(np.stack(out))
