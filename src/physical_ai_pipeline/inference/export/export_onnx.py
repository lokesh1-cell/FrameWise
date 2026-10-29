"""Export PyTorch YOLOv8 checkpoints to dynamic-batch ONNX.

    python -m physical_ai_pipeline.inference.export.export_onnx \
        --weights yolov8n.pt yolov8n-pose.pt hands-yolov8n-pose.pt --out /work/onnx

The hand model is a YOLOv8n-pose network fine-tuned on Ultralytics'
``hand-keypoints`` dataset (21 keypoints), e.g.
``yolo pose train data=hand-keypoints.yaml model=yolov8n-pose.pt``.
Requires the ``export`` extra (ultralytics, onnx, onnxslim).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from physical_ai_pipeline.logging_utils import configure_logging, get_logger, log_event

_log = get_logger(__name__)


def export_yolo(weights: Path, out_dir: Path, imgsz: int = 640, opset: int = 17) -> Path:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    exported = Path(
        model.export(format="onnx", imgsz=imgsz, dynamic=True, simplify=True, opset=opset, half=False, batch=1)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{weights.stem}.onnx"
    shutil.move(str(exported), target)
    log_event(_log, "onnx_exported", weights=str(weights), onnx=str(target), imgsz=imgsz, opset=opset)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args(argv)
    configure_logging()
    for weights in args.weights:
        export_yolo(weights, args.out, args.imgsz, args.opset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
