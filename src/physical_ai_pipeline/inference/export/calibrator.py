"""INT8 entropy calibrator fed from a directory of representative frames.

Calibration images are read from local disk (frames sampled from consented,
child-free footage). Face crops produced during calibration live only in
host/device memory.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import numpy.typing as npt

from physical_ai_pipeline.inference.trt_engine import _check, _cudart

Preprocess = Callable[[npt.NDArray[np.uint8]], npt.NDArray[np.float32]]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def iter_images(directory: Path, limit: int) -> Iterator[npt.NDArray[np.uint8]]:
    paths = sorted(p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)[:limit]
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            yield cast(npt.NDArray[np.uint8], image)


def make_calibrator(
    image_dir: Path, preprocess: Preprocess, batch_size: int, cache_file: Path, limit: int = 512
) -> Any:
    import tensorrt as trt

    cudart = _cudart()

    class EntropyCalibrator(trt.IInt8EntropyCalibrator2):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self._images = iter_images(image_dir, limit)
            self._device: int | None = None
            self._nbytes = 0

        def get_batch_size(self) -> int:
            return batch_size

        def get_batch(self, names: list[str]) -> list[int] | None:
            batch: list[npt.NDArray[np.float32]] = []
            for image in self._images:
                batch.append(preprocess(image))
                if len(batch) == batch_size:
                    break
            if len(batch) < batch_size:
                return None
            host = np.ascontiguousarray(np.concatenate(batch, axis=0).astype(np.float32))
            if self._device is None or self._nbytes < host.nbytes:
                self._device = int(_check(cudart.cudaMalloc(host.nbytes)))
                self._nbytes = host.nbytes
            _check(
                cudart.cudaMemcpy(
                    self._device, host.ctypes.data, host.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
                )
            )
            return [self._device]

        def read_calibration_cache(self) -> bytes | None:
            return cache_file.read_bytes() if cache_file.is_file() else None

        def write_calibration_cache(self, cache: bytes) -> None:
            cache_file.write_bytes(bytes(cache))

    return EntropyCalibrator()
