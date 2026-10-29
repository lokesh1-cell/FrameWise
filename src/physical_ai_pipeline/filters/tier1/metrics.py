"""Tier 1 measurements, computed once per frame.

All measurements for frame i depend only on frame i and frame i-1, so
:class:`Tier1Precompute` computes them on a thread pool (OpenCV releases the
GIL) ahead of the sequential filter decisions. Without precompute,
:class:`InlineMetrics` computes them on demand with identical results.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np
import numpy.typing as npt

from physical_ai_pipeline.config import Tier1Config
from physical_ai_pipeline.frame import FrameRecord, ImageBGR

Gray = npt.NDArray[np.uint8]

# Farneback parameters tuned for a ~320 px wide input (see README benchmarks).
FARNEBACK_PARAMS = {
    "pyr_scale": 0.5,
    "levels": 2,
    "winsize": 9,
    "iterations": 2,
    "poly_n": 5,
    "poly_sigma": 1.1,
    "flags": 0,
}


@dataclass(slots=True)
class Tier1Metrics:
    decode_ok: bool
    corrupt_block_ratio: float = 0.0
    luminance_mean: float = -1.0
    laplacian_variance: float = -1.0
    phash: int | None = None
    phash_hamming: int = -1
    flow_magnitude: float = -1.0


def _resize_to_width(image: npt.NDArray[np.uint8], width: int) -> npt.NDArray[np.uint8]:
    h, w = image.shape[:2]
    if w <= width:
        return image
    return cast(
        npt.NDArray[np.uint8], cv2.resize(image, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)
    )


def corrupt_block_ratio(image_bgr: ImageBGR, block: int, flat_max_std: float, green_dominance: float) -> float:
    """Fraction of ``block`` x ``block`` tiles that are flat and green-dominant
    (FFmpeg error concealment paints unrecoverable macroblocks flat green)."""
    h, w = image_bgr.shape[:2]
    bh, bw = h // block, w // block
    if bh == 0 or bw == 0:
        return 0.0
    tiles = image_bgr[: bh * block, : bw * block].astype(np.float32)
    means = cv2.resize(tiles, (bw, bh), interpolation=cv2.INTER_AREA)
    sq = cv2.resize(tiles * tiles, (bw, bh), interpolation=cv2.INTER_AREA)
    std = np.sqrt(np.maximum(sq - means * means, 0.0)).max(axis=-1)
    b, g, r = means[..., 0], means[..., 1], means[..., 2]
    hits = ((g - np.maximum(b, r)) > green_dominance) & (std <= flat_max_std)
    return float(np.count_nonzero(hits)) / float(bh * bw)


def laplacian_variance(gray: Gray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def phash(gray: Gray) -> int:
    """64-bit DCT perceptual hash (32x32 DCT, top-left 8x8, median threshold)."""
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(small)[:8, :8].flatten()
    bits = np.packbits(low > np.median(low[1:]))
    return int.from_bytes(bits.tobytes(), "big")


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def flow_gray(image: ImageBGR, flow_width: int) -> Gray:
    return cast(Gray, cv2.cvtColor(_resize_to_width(image, flow_width), cv2.COLOR_BGR2GRAY))


def mean_flow_magnitude(prev_gray: Gray, gray: Gray) -> float:
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **FARNEBACK_PARAMS)  # type: ignore[call-overload]
    return float(np.mean(cv2.magnitude(flow[..., 0], flow[..., 1])))


def compute_metrics(image: ImageBGR | None, prev_image: ImageBGR | None, config: Tier1Config) -> Tier1Metrics:
    """Every Tier 1 measurement for one frame. ``phash_hamming`` is filled
    later (it needs the previous frame's hash)."""
    if image is None or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or image.size == 0:
        return Tier1Metrics(decode_ok=False)
    small = _resize_to_width(image, config.analysis_width)
    gray = cast(Gray, cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
    block = max(2, round(config.decode.macroblock_size * small.shape[1] / image.shape[1]))
    metrics = Tier1Metrics(
        decode_ok=True,
        corrupt_block_ratio=corrupt_block_ratio(
            small, block, config.decode.flat_block_max_std, config.decode.green_dominance
        ),
        luminance_mean=float(cv2.mean(gray)[0]),
        laplacian_variance=laplacian_variance(gray),
        phash=phash(gray),
    )
    if config.idle.enabled and prev_image is not None and prev_image.shape == image.shape:
        flow_w = min(config.idle.flow_width, image.shape[1])
        magnitude = mean_flow_magnitude(flow_gray(prev_image, flow_w), flow_gray(image, flow_w))
        metrics.flow_magnitude = magnitude * image.shape[1] / flow_w
    return metrics


class InlineMetrics:
    """Sequential metric computation (used when no precompute stage ran)."""

    def __init__(self, config: Tier1Config) -> None:
        self.config = config
        self._prev_image: ImageBGR | None = None
        self._prev_hash: int | None = None

    def ensure(self, record: FrameRecord) -> Tier1Metrics:
        if record.tier1 is None:
            record.tier1 = compute_metrics(record.image, self._prev_image, self.config)
            _link_hash(record.tier1, self._prev_hash)
        self._prev_image = record.image
        self._prev_hash = record.tier1.phash
        return record.tier1


def _link_hash(metrics: Tier1Metrics, prev_hash: int | None) -> None:
    if metrics.phash is not None and prev_hash is not None:
        metrics.phash_hamming = hamming(prev_hash, metrics.phash)


class Tier1Precompute:
    """Wraps a frame iterator; yields the same records, in order, with
    ``record.tier1`` populated by a thread pool."""

    def __init__(self, config: Tier1Config, workers: int = 4, max_inflight: int = 16) -> None:
        self.config = config
        self.workers = workers
        self.max_inflight = max(max_inflight, workers)

    def __call__(self, frames: Iterator[FrameRecord]) -> Iterator[FrameRecord]:
        prev_image: ImageBGR | None = None
        prev_hash: int | None = None
        inflight: deque[tuple[FrameRecord, Future[Tier1Metrics]]] = deque()
        with ThreadPoolExecutor(self.workers, thread_name_prefix="tier1") as pool:
            for record in frames:
                inflight.append((record, pool.submit(compute_metrics, record.image, prev_image, self.config)))
                prev_image = record.image
                while len(inflight) > self.max_inflight or (inflight and inflight[0][1].done()):
                    done, future = inflight.popleft()
                    done.tier1 = future.result()
                    _link_hash(done.tier1, prev_hash)
                    prev_hash = done.tier1.phash
                    yield done
            while inflight:
                done, future = inflight.popleft()
                done.tier1 = future.result()
                _link_hash(done.tier1, prev_hash)
                prev_hash = done.tier1.phash
                yield done
