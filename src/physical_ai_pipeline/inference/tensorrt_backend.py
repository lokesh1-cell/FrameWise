"""TensorRT backend for the NVIDIA L4: YOLOv8n, YOLOv8n-pose, YOLO hands,
InsightFace SCRFD + genderage, optional Depth Anything V2."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from physical_ai_pipeline.config import InferenceConfig, ModelConfig
from physical_ai_pipeline.frame import FaceObservation, FrameInference, HandDetection
from physical_ai_pipeline.inference.backend import InferenceInput, ModelInfo
from physical_ai_pipeline.inference.coco import COCO_CLASSES, COCO_KEYPOINTS, HAND_KEYPOINTS
from physical_ai_pipeline.inference.postprocess import (
    decode_genderage,
    decode_scrfd,
    decode_yolo_detections,
    decode_yolo_pose,
)
from physical_ai_pipeline.inference.preprocess import depth_batch, genderage_crops, scrfd_input, yolo_batch
from physical_ai_pipeline.inference.trt_engine import TrtEngine


class TensorRTBackend:
    def __init__(self, config: InferenceConfig) -> None:
        self.config = config
        self._engines: dict[str, tuple[ModelConfig, TrtEngine]] = {}
        roles = {
            "detector": config.detector,
            "pose": config.pose,
            "hands": config.hands,
            "face_detector": config.face_detector,
            "face_attributes": config.face_attributes,
            "depth": config.depth,
        }
        for role, model in roles.items():
            if model.enabled:
                path = Path(model.engine_path)
                if not path.is_absolute():
                    path = Path(config.engine_dir) / path
                self._engines[role] = (model, TrtEngine(path, model.sha256))

    def _run_single_input(self, role: str, tensor: np.ndarray) -> list[np.ndarray]:
        _, engine = self._engines[role]
        return engine.infer_batched(tensor)

    def infer(self, batch: Sequence[InferenceInput]) -> list[FrameInference]:
        images = [item.image for item in batch]
        sizes = [(img.shape[0], img.shape[1]) for img in images]
        det_cfg = self.config.detector
        tensor, params = yolo_batch(images, det_cfg.input_size)
        detections = decode_yolo_detections(
            self._run_single_input("detector", tensor)[0], params, sizes, COCO_CLASSES,
            det_cfg.confidence_threshold, det_cfg.nms_iou_threshold, det_cfg.max_detections,
        )  # fmt: skip
        results = [FrameInference(objects=d) for d in detections]
        if "pose" in self._engines:
            cfg = self.config.pose
            if cfg.input_size != det_cfg.input_size:
                tensor, params = yolo_batch(images, cfg.input_size)
            poses = decode_yolo_pose(
                self._run_single_input("pose", tensor)[0], params, sizes, len(COCO_KEYPOINTS),
                cfg.confidence_threshold, cfg.nms_iou_threshold, cfg.max_detections,
            )  # fmt: skip
            for result, p in zip(results, poses, strict=True):
                result.poses = p
        if "hands" in self._engines:
            cfg = self.config.hands
            if cfg.input_size != det_cfg.input_size:
                tensor, params = yolo_batch(images, cfg.input_size)
            hands = decode_yolo_pose(
                self._run_single_input("hands", tensor)[0], params, sizes, len(HAND_KEYPOINTS),
                cfg.confidence_threshold, cfg.nms_iou_threshold, cfg.max_detections,
            )  # fmt: skip
            for result, hs in zip(results, hands, strict=True):
                result.hands = [HandDetection(h.bbox, h.confidence, h.keypoints) for h in hs]
        if "depth" in self._engines:
            cfg = self.config.depth
            depth = self._run_single_input("depth", depth_batch(images, cfg.input_size))[0]
            for result, d in zip(results, depth.reshape(len(images), cfg.input_size, cfg.input_size), strict=True):
                result.depth = d.astype(np.float32)
        return results

    def analyze_faces(self, batch: Sequence[InferenceInput]) -> list[list[FaceObservation]]:
        det_cfg = self.config.face_detector
        attr_cfg = self.config.face_attributes
        results: list[list[FaceObservation]] = []
        for item in batch:
            # SCRFD det_10g is exported with batch 1; faces run only on sampled
            # frames (~1/15 of kept frames) so this is not the bottleneck.
            tensor, params = scrfd_input(item.image, det_cfg.input_size)
            boxes, scores = decode_scrfd(
                self._run_single_input("face_detector", tensor), params, det_cfg.input_size,
                det_cfg.confidence_threshold, det_cfg.nms_iou_threshold,
            )  # fmt: skip
            if boxes.shape[0] == 0:
                results.append([])
                continue
            box_list = [tuple(float(v) for v in b) for b in boxes[: det_cfg.max_detections]]
            crops = genderage_crops(item.image, box_list, attr_cfg.input_size)  # type: ignore[arg-type]
            ages = decode_genderage(self._run_single_input("face_attributes", crops)[0])
            del crops  # face crops never leave this scope
            results.append(
                [
                    FaceObservation(bbox=b, det_score=float(s), age=float(a))  # type: ignore[arg-type]
                    for b, s, a in zip(box_list, scores, ages, strict=True)
                ]
            )
        return results

    def model_infos(self) -> list[ModelInfo]:
        return [
            ModelInfo(Path(engine.path).name, role, str(engine.path), engine.sha256, model.precision.value)
            for role, (model, engine) in self._engines.items()
        ]

    def close(self) -> None:
        for _, engine in self._engines.values():
            engine.close()
