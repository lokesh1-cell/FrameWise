"""Deterministic scripted backend for tests and GPU-less local runs.

A script maps frame ranges to detections, so synthetic videos can exercise
every decision filter without model weights. Script JSON format::

    {"segments": [
        {"start": 0, "end": 99, "objects": [{"class": "cup", "bbox": [..], "conf": 0.9}],
         "poses": [{"bbox": [..], "keypoints": {"left_wrist": [x, y, c]}}],
         "hands": [{"bbox": [..], "conf": 0.8}],
         "faces": [{"bbox": [..], "score": 0.95, "age": 9}],
         "motion": {"all": [dx, dy]}, "oscillate": 0.0}
    ]}
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from physical_ai_pipeline.frame import Detection, FaceObservation, FrameInference, HandDetection, Pose
from physical_ai_pipeline.inference.backend import InferenceInput, ModelInfo
from physical_ai_pipeline.inference.coco import COCO_CLASSES, COCO_KEYPOINTS


@dataclass(slots=True)
class ScriptSegment:
    start: int
    end: int
    objects: list[dict[str, Any]] = field(default_factory=list)
    poses: list[dict[str, Any]] = field(default_factory=list)
    hands: list[dict[str, Any]] = field(default_factory=list)
    faces: list[dict[str, Any]] = field(default_factory=list)
    # Per-keypoint linear motion in px/frame applied to poses and hands, so
    # wrist velocity is non-zero for fake-work tests.
    motion: dict[str, list[float]] = field(default_factory=dict)
    # Oscillating motion: amplitude in px; wrists move back and forth.
    oscillate: float = 0.0


def _bbox(raw: Sequence[float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(v) for v in raw)
    return (x1, y1, x2, y2)


class ScriptedBackend:
    def __init__(self, segments: Sequence[ScriptSegment] = ()) -> None:
        self.segments = list(segments)
        self.face_calls: list[int] = []
        self.infer_calls: list[list[int]] = []

    @classmethod
    def from_json(cls, path: str | Path) -> ScriptedBackend:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_mapping(data)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> ScriptedBackend:
        segments = [ScriptSegment(**seg) for seg in data.get("segments", [])]
        return cls(segments)

    def _active(self, index: int) -> list[ScriptSegment]:
        return [s for s in self.segments if s.start <= index <= s.end]

    def _offset(self, seg: ScriptSegment, index: int) -> tuple[float, float]:
        if seg.oscillate:
            return (seg.oscillate if (index - seg.start) % 2 else -seg.oscillate, 0.0)
        motion = seg.motion.get("all", [0.0, 0.0])
        return (motion[0] * (index - seg.start), motion[1] * (index - seg.start))

    def infer(self, batch: Sequence[InferenceInput]) -> list[FrameInference]:
        self.infer_calls.append([b.index for b in batch])
        out: list[FrameInference] = []
        for item in batch:
            inference = FrameInference()
            for seg in self._active(item.index):
                dx, dy = self._offset(seg, item.index)
                for obj in seg.objects:
                    name = str(obj["class"])
                    class_id = COCO_CLASSES.index(name) if name in COCO_CLASSES else -1
                    inference.objects.append(Detection(_bbox(obj["bbox"]), float(obj.get("conf", 0.9)), class_id, name))
                for pose in seg.poses:
                    kps = np.zeros((len(COCO_KEYPOINTS), 3), dtype=np.float32)
                    for name, (x, y, c) in pose.get("keypoints", {}).items():
                        kps[COCO_KEYPOINTS.index(name)] = (x + dx, y + dy, c)
                    inference.poses.append(Pose(_bbox(pose["bbox"]), float(pose.get("conf", 0.9)), kps))
                for hand in seg.hands:
                    x1, y1, x2, y2 = _bbox(hand["bbox"])
                    kps = np.zeros((21, 3), dtype=np.float32)
                    kps[:, :2] = ((x1 + x2) / 2 + dx, (y1 + y2) / 2 + dy)
                    kps[:, 2] = 0.9
                    inference.hands.append(
                        HandDetection((x1 + dx, y1 + dy, x2 + dx, y2 + dy), float(hand.get("conf", 0.9)), kps)
                    )
            out.append(inference)
        return out

    def analyze_faces(self, batch: Sequence[InferenceInput]) -> list[list[FaceObservation]]:
        results: list[list[FaceObservation]] = []
        for item in batch:
            self.face_calls.append(item.index)
            faces = [
                FaceObservation(_bbox(f["bbox"]), float(f["score"]), float(f["age"]))
                for seg in self._active(item.index)
                for f in seg.faces
            ]
            results.append(faces)
        return results

    def model_infos(self) -> list[ModelInfo]:
        return [ModelInfo("scripted", "all", "", "", "none")]

    def close(self) -> None:
        return None
