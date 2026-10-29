"""Thin TensorRT 10 engine wrapper (name-based I/O, dynamic shapes).

Requires ``tensorrt>=10`` and ``cuda-python`` at runtime; both are imported
lazily so the rest of the package (and its tests) runs on CPU-only machines.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from physical_ai_pipeline.logging_utils import get_logger, log_event

_log = get_logger(__name__)


def sha256_file(path: str | Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _cudart() -> Any:
    try:
        from cuda.bindings import runtime as cudart  # cuda-python >= 12.6
    except ImportError:  # pragma: no cover - older cuda-python layout
        from cuda import cudart
    return cudart


def _check(result: Any) -> Any:
    err, *rest = result if isinstance(result, tuple) else (result,)
    if int(err) != 0:
        raise RuntimeError(f"CUDA error {err}")
    return rest[0] if len(rest) == 1 else rest


@dataclass(slots=True)
class _DeviceBuffer:
    ptr: int
    nbytes: int


class TrtEngine:
    def __init__(self, engine_path: str | Path, expected_sha256: str | None = None) -> None:
        import tensorrt as trt

        path = Path(engine_path)
        if not path.is_file():
            raise FileNotFoundError(f"TensorRT engine not found: {path}")
        self.sha256 = sha256_file(path)
        if expected_sha256 and expected_sha256 != self.sha256:
            raise ValueError(f"engine {path} sha256 {self.sha256} != expected {expected_sha256}")
        self.path = path
        self._trt = trt
        self._cudart = _cudart()
        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)
        self.engine = self._runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize engine {path}")
        self.context = self.engine.create_execution_context()
        self.stream = _check(self._cudart.cudaStreamCreate())
        self.inputs: list[str] = []
        self.outputs: list[str] = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(name)
            else:
                self.outputs.append(name)
        self._buffers: dict[str, _DeviceBuffer] = {}
        log_event(
            _log, "trt_engine_loaded", path=str(path), sha256=self.sha256, inputs=self.inputs, outputs=self.outputs
        )

    def _dtype(self, name: str) -> np.dtype[Any]:
        dtype: np.dtype[Any] = np.dtype(self._trt.nptype(self.engine.get_tensor_dtype(name)))
        return dtype

    def _ensure(self, name: str, nbytes: int) -> int:
        buf = self._buffers.get(name)
        if buf is None or buf.nbytes < nbytes:
            if buf is not None:
                _check(self._cudart.cudaFree(buf.ptr))
            ptr = _check(self._cudart.cudaMalloc(max(nbytes, 1)))
            buf = _DeviceBuffer(int(ptr), nbytes)
            self._buffers[name] = buf
        return buf.ptr

    @property
    def max_batch(self) -> int:
        """Largest batch the first input's optimisation profile accepts."""
        _, _, max_shape = self.engine.get_tensor_profile_shape(self.inputs[0], 0)
        return max(1, int(max_shape[0]))

    def infer_batched(self, tensor: npt.NDArray[Any]) -> list[npt.NDArray[Any]]:
        """Single-input inference split into chunks of at most ``max_batch``;
        outputs are concatenated along axis 0 in ``self.outputs`` order."""
        step = self.max_batch
        if tensor.shape[0] <= step:
            out = self.infer({self.inputs[0]: tensor})
            return [out[name] for name in self.outputs]
        parts = [self.infer({self.inputs[0]: tensor[i : i + step]}) for i in range(0, tensor.shape[0], step)]
        return [np.concatenate([p[name] for p in parts], axis=0) for name in self.outputs]

    def infer(self, feeds: dict[str, npt.NDArray[Any]]) -> dict[str, npt.NDArray[Any]]:
        cudart = self._cudart
        h2d = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        d2h = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        hosts: list[npt.NDArray[Any]] = []
        for name in self.inputs:
            array = np.ascontiguousarray(feeds[name].astype(self._dtype(name), copy=False))
            hosts.append(array)
            if not self.context.set_input_shape(name, tuple(array.shape)):
                raise ValueError(f"input {name} shape {array.shape} outside optimisation profile")
            ptr = self._ensure(name, array.nbytes)
            _check(cudart.cudaMemcpyAsync(ptr, array.ctypes.data, array.nbytes, h2d, self.stream))
            self.context.set_tensor_address(name, ptr)
        results: dict[str, npt.NDArray[Any]] = {}
        for name in self.outputs:
            shape = tuple(self.context.get_tensor_shape(name))
            out = np.empty(shape, dtype=self._dtype(name))
            ptr = self._ensure(name, out.nbytes)
            self.context.set_tensor_address(name, ptr)
            results[name] = out
        if not self.context.execute_async_v3(self.stream):
            raise RuntimeError(f"TensorRT execution failed for {self.path}")
        for name, out in results.items():
            _check(cudart.cudaMemcpyAsync(out.ctypes.data, self._buffers[name].ptr, out.nbytes, d2h, self.stream))
        _check(cudart.cudaStreamSynchronize(self.stream))
        del hosts
        return results

    def close(self) -> None:
        for buf in self._buffers.values():
            self._cudart.cudaFree(buf.ptr)
        self._buffers.clear()
        self._cudart.cudaStreamDestroy(self.stream)
