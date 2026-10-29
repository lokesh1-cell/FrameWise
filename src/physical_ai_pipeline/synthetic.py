"""Synthetic test frames for unit tests, the CLI demo and benchmarks."""

from __future__ import annotations

from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import numpy.typing as npt

Image = npt.NDArray[np.uint8]


class SceneGenerator:
    """A textured, scrolling scene: sharp, well exposed, never duplicate and
    never idle, so every frame passes Tier 1 unless modified."""

    def __init__(self, width: int = 640, height: int = 360, seed: int = 0, speed_px: int = 0) -> None:
        rng = np.random.default_rng(seed)
        cell = max(8, width // 27)
        self.width, self.height, self.speed = width, height, speed_px or cell
        cells = rng.integers(50, 210, size=(height // cell + 2, (width * 3) // cell + 2, 3), dtype=np.uint8)
        texture = cv2.resize(cells, (cells.shape[1] * cell, cells.shape[0] * cell), interpolation=cv2.INTER_NEAREST)
        noise = rng.integers(-20, 20, size=texture.shape, dtype=np.int16)
        self.texture = np.clip(texture.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    def frame(self, index: int) -> Image:
        offset = (index * self.speed) % (self.texture.shape[1] - self.width)
        image = np.ascontiguousarray(self.texture[: self.height, offset : offset + self.width])
        cv2.putText(image, str(index), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        return image

    def frames(self, count: int) -> Iterator[Image]:
        for i in range(count):
            yield self.frame(i)


def dark(image: Image, level: int = 15) -> Image:
    return np.full_like(image, level)


def overexposed(image: Image, level: int = 245) -> Image:
    return np.full_like(image, level)


def blurred(image: Image, ksize: int = 31) -> Image:
    return cast(Image, cv2.GaussianBlur(image, (ksize, ksize), 0))


def corrupt(image: Image, fraction: float = 0.2, block: int = 16) -> Image:
    """Paint flat green macroblocks, mimicking FFmpeg error concealment."""
    out = image.copy()
    h, w = out.shape[:2]
    rows = int((h // block) * fraction) or 1
    out[: rows * block, :] = (0, 135, 0)
    return out


def write_video(path: str | Path, frames: Iterator[Image], fps: float = 30.0) -> Path:
    """Encode frames to H.264 MP4 with PyAV."""
    import av

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    first = next(frames)
    with av.open(str(target), mode="w") as container:
        stream = container.add_stream("libx264", rate=Fraction(fps).limit_denominator(100000))
        stream.width, stream.height = first.shape[1], first.shape[0]
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "veryfast"}
        for image in _chain(first, frames):
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="bgr24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return target


def _chain(first: Image, rest: Iterator[Image]) -> Iterator[Image]:
    yield first
    yield from rest
