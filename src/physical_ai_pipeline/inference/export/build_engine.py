"""Build TensorRT engines for the NVIDIA L4 (sm_89) from ONNX graphs.

Engines are GPU-architecture and TensorRT-version specific: run this on an
L4 node with the same TensorRT version as the worker image (see
deploy/gke/engine-build-job.yaml). It writes ``manifest.json`` with each
engine's SHA256, which the worker verifies at load.

    python -m physical_ai_pipeline.inference.export.build_engine \
        --onnx-dir /work/onnx --out /work/engines --calib-dir /work/calib
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from physical_ai_pipeline.inference.preprocess import genderage_crops, scrfd_input, yolo_batch
from physical_ai_pipeline.inference.trt_engine import sha256_file
from physical_ai_pipeline.logging_utils import configure_logging, get_logger, log_event

_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class EngineSpec:
    onnx: str
    engine: str
    precision: str  # fp16 | int8
    input_hw: tuple[int, int]
    min_batch: int
    opt_batch: int
    max_batch: int
    calib: str | None = None  # preprocessing kind for INT8 calibration


DEFAULT_SPECS: tuple[EngineSpec, ...] = (
    EngineSpec("yolov8n.onnx", "yolov8n_fp16.engine", "fp16", (640, 640), 1, 32, 32),
    EngineSpec("yolov8n-pose.onnx", "yolov8n-pose_fp16.engine", "fp16", (640, 640), 1, 32, 32),
    EngineSpec("hands-yolov8n-pose.onnx", "yolov8n-hands_fp16.engine", "fp16", (640, 640), 1, 32, 32),
    EngineSpec("det_10g.onnx", "scrfd_10g_fp16.engine", "fp16", (640, 640), 1, 1, 1),
    EngineSpec("genderage.onnx", "genderage_int8.engine", "int8", (96, 96), 1, 8, 32, calib="genderage"),
    EngineSpec("depth_anything_v2_vits.onnx", "depth_anything_v2_vits_fp16.engine", "fp16", (518, 518), 1, 16, 32),
)


def _calib_preprocess(kind: str, size: int):  # type: ignore[no-untyped-def]
    if kind == "yolo":
        return lambda img: yolo_batch([img], size)[0]
    if kind == "scrfd":
        return lambda img: scrfd_input(img, size)[0]
    if kind == "genderage":
        # Calibrate on centred crops of the full frame; the detector supplies
        # real face boxes at inference time.
        def crop(img: np.ndarray) -> np.ndarray:
            h, w = img.shape[:2]
            side = min(h, w) / 1.5
            box = (w / 2 - side / 2, h / 2 - side / 2, w / 2 + side / 2, h / 2 + side / 2)
            return genderage_crops(img, [box], size)

        return crop
    raise ValueError(f"unknown calibration kind {kind}")


def build(spec: EngineSpec, onnx_dir: Path, out_dir: Path, calib_dir: Path | None, workspace_gb: int) -> Path:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)  # explicit batch is the only mode in TRT 10
    parser = trt.OnnxParser(network, logger)
    onnx_path = onnx_dir / spec.onnx
    if not parser.parse(onnx_path.read_bytes()):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError(f"ONNX parse failed for {onnx_path}: {errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    config.set_flag(trt.BuilderFlag.FP16)  # INT8 engines fall back to FP16 per layer

    profile = builder.create_optimization_profile()
    tensor = network.get_input(0)
    h, w = spec.input_hw
    lo, opt, hi = spec.min_batch, spec.opt_batch, spec.max_batch
    if tensor.shape[0] != -1:
        # Static batch in the ONNX graph (e.g. det_10g): the runtime splits
        # larger batches into chunks of this size.
        lo = opt = hi = int(tensor.shape[0])
        log_event(_log, "engine_static_batch", engine=spec.engine, batch=lo)
    profile.set_shape(tensor.name, (lo, 3, h, w), (opt, 3, h, w), (hi, 3, h, w))
    config.add_optimization_profile(profile)
    if spec.precision == "int8":
        if calib_dir is None:
            raise ValueError(f"{spec.engine}: INT8 requires --calib-dir")
        from physical_ai_pipeline.inference.export.calibrator import make_calibrator

        # Implicit (calibrator-based) INT8 is deprecated in TensorRT 10 but
        # still supported; explicit Q/DQ via NVIDIA ModelOpt is the successor.
        out_dir.mkdir(parents=True, exist_ok=True)
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = make_calibrator(
            calib_dir,
            _calib_preprocess(spec.calib or "yolo", spec.input_hw[0]),
            batch_size=opt,
            cache_file=out_dir / f"{Path(spec.engine).stem}.calib",
        )
        config.set_calibration_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"engine build failed for {spec.engine}")
    out_dir.mkdir(parents=True, exist_ok=True)
    engine_path = out_dir / spec.engine
    engine_path.write_bytes(bytes(serialized))
    log_event(_log, "engine_built", engine=str(engine_path), precision=spec.precision, trt_version=trt.__version__)
    return engine_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--calib-dir", type=Path)
    parser.add_argument("--workspace-gb", type=int, default=8)
    parser.add_argument("--only", nargs="*", help="engine filenames to build (default: all with ONNX present)")
    args = parser.parse_args(argv)
    configure_logging()

    import tensorrt as trt

    engines: list[dict[str, object]] = []
    manifest: dict[str, object] = {"tensorrt_version": trt.__version__, "engines": engines}
    for spec in DEFAULT_SPECS:
        if args.only and spec.engine not in args.only:
            continue
        if not (args.onnx_dir / spec.onnx).is_file():
            log_event(_log, "engine_skipped_missing_onnx", onnx=spec.onnx, level=30)
            continue
        path = build(spec, args.onnx_dir, args.out, args.calib_dir, args.workspace_gb)
        engines.append({**asdict(spec), "sha256": sha256_file(path)})
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
