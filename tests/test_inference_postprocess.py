"""CPU-only tests for TensorRT pre/post-processing (no GPU needed)."""

from __future__ import annotations

import numpy as np

from physical_ai_pipeline.config import InferenceConfig
from physical_ai_pipeline.frame import FrameRecord, Hand, HandDetection, Pose
from physical_ai_pipeline.inference.coco import COCO_CLASSES
from physical_ai_pipeline.inference.fake_backend import ScriptedBackend
from physical_ai_pipeline.inference.geometry import iou, nms
from physical_ai_pipeline.inference.postprocess import (
    SCRFD_STRIDES,
    decode_genderage,
    decode_scrfd,
    decode_yolo_detections,
    decode_yolo_pose,
    depth_to_uint16,
)
from physical_ai_pipeline.inference.preprocess import genderage_crops, letterbox, scrfd_input, yolo_batch
from physical_ai_pipeline.inference.runner import BatchedInferenceRunner, assign_handedness


def test_letterbox_roundtrip() -> None:
    image = np.zeros((1080, 1920, 3), np.uint8)
    boxed, params = letterbox(image, 640)
    assert boxed.shape == (640, 640, 3)
    assert abs(params.scale - 640 / 1920) < 1e-9 and params.pad_x == 0 and params.pad_y == 140
    src = params.to_source(np.array([0, 140, 640, 500], np.float32))
    assert np.allclose(src, (0, 0, 1920, 1080), atol=1)


def test_yolo_batch_layout() -> None:
    tensor, params = yolo_batch([np.full((360, 640, 3), 255, np.uint8)] * 3, 640)
    assert tensor.shape == (3, 3, 640, 640) and tensor.dtype == np.float32
    assert tensor.max() == 1.0 and len(params) == 3


def _yolo_output(boxes: list[tuple[float, float, float, float, int, float]], n: int = 50) -> np.ndarray:
    out = np.zeros((1, 4 + 80, n), np.float32)
    for i, (cx, cy, w, h, cls, score) in enumerate(boxes):
        out[0, :4, i] = (cx, cy, w, h)
        out[0, 4 + cls, i] = score
    return out


def test_decode_yolo_detections_nms_and_rescale() -> None:
    _, params = letterbox(np.zeros((360, 640, 3), np.uint8), 640)
    out = _yolo_output([
        (320, 320, 100, 100, 0, 0.9),
        (322, 321, 100, 100, 0, 0.8),  # duplicate person -> suppressed
        (322, 321, 100, 100, 67, 0.7),  # cell phone at same place -> kept (class-aware)
        (100, 300, 20, 20, 41, 0.1),  # below threshold
    ])  # fmt: skip
    dets = decode_yolo_detections(out, [params], [(360, 640)], COCO_CLASSES, 0.25, 0.45, 100)[0]
    assert [d.class_name for d in dets] == ["person", "cell phone"]
    x1, y1, x2, y2 = dets[0].bbox
    assert np.allclose((x1, y1, x2, y2), (270, 130, 370, 230), atol=1)


def test_decode_yolo_pose() -> None:
    _, params = letterbox(np.zeros((640, 640, 3), np.uint8), 640)
    out = np.zeros((1, 5 + 17 * 3, 10), np.float32)
    out[0, :5, 0] = (320, 320, 100, 200, 0.9)
    out[0, 5 + 9 * 3 : 5 + 9 * 3 + 3, 0] = (300, 400, 0.95)  # left wrist
    poses = decode_yolo_pose(out, [params], [(640, 640)], 17, 0.25, 0.45, 10)[0]
    assert len(poses) == 1 and poses[0].keypoints.shape == (17, 3)
    assert np.allclose(poses[0].keypoints[9], (300, 400, 0.95))


def test_decode_scrfd_single_face() -> None:
    size = 640
    outputs: list[np.ndarray] = []
    scores, deltas = [], []
    for stride in SCRFD_STRIDES:
        n = (size // stride) ** 2 * 2
        scores.append(np.zeros((n, 1), np.float32))
        deltas.append(np.zeros((n, 4), np.float32))
    # anchor (x=10, y=5) at stride 16, first anchor -> centre (160, 80)
    idx = (5 * (size // 16) + 10) * 2
    scores[1][idx] = 0.97
    deltas[1][idx] = (1, 1, 1, 1)  # x16 -> 16 px each side
    outputs = scores + deltas
    image = np.zeros((1280, 1280, 3), np.uint8)
    _, params = scrfd_input(image, size)
    boxes, confs = decode_scrfd(outputs, params, size, 0.5, 0.4)
    assert confs.tolist() == [np.float32(0.97)]
    assert np.allclose(boxes[0], (288, 128, 352, 192))  # scale 0.5 back to source


def test_genderage_crop_and_decode() -> None:
    image = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    crops = genderage_crops(image, [(100, 100, 160, 180)], 96)
    assert crops.shape == (1, 3, 96, 96) and crops.dtype == np.float32
    ages = decode_genderage(np.array([[0.1, 0.9, 0.07], [0.8, 0.2, 0.31]], np.float32))
    assert np.allclose(ages, (7, 31))


def test_nms_and_iou() -> None:
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], np.float32)
    assert nms(boxes, np.array([0.9, 0.8, 0.7], np.float32), 0.5) == [0, 2]
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0 and iou((0, 0, 1, 1), (2, 2, 3, 3)) == 0.0


def test_depth_to_uint16() -> None:
    d = depth_to_uint16(np.array([[0.0, 0.5], [1.0, 0.25]], np.float32))
    assert d.dtype == np.uint16 and d.max() == 65535 and d.min() == 0


def test_handedness_from_body_wrists() -> None:
    kps = np.zeros((17, 3), np.float32)
    kps[9] = (100, 100, 0.9)  # left wrist
    kps[10] = (300, 100, 0.9)  # right wrist
    hands = [HandDetection((290, 90, 310, 110), 0.9, np.zeros((21, 3), np.float32)),
             HandDetection((90, 90, 110, 110), 0.9, np.zeros((21, 3), np.float32))]  # fmt: skip
    labelled = assign_handedness(hands, [Pose((0, 0, 400, 400), 0.9, kps)])
    assert [h.handedness for h in labelled] == [Hand.RIGHT, Hand.LEFT]


def test_runner_batches_and_counts_people() -> None:
    backend = ScriptedBackend.from_mapping({"segments": [{"start": 0, "end": 99, "objects": [
        {"class": "person", "bbox": [0, 0, 10, 10], "conf": 0.9},
        {"class": "person", "bbox": [20, 0, 30, 10], "conf": 0.3}]}]})  # fmt: skip
    runner = BatchedInferenceRunner(backend, InferenceConfig(batch_size=16))
    records = [FrameRecord(i, 0, np.zeros((8, 8, 3), np.uint8)) for i in range(40)]
    runner.run(records)
    assert [len(c) for c in backend.infer_calls] == [16, 16, 8]
    assert all(r.inference is not None and r.inference.person_count == 1 for r in records)
