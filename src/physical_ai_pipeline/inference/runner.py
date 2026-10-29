"""Batched Tier 2 runner: executes each model once per batch and caches the
results on the frame records."""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np

from physical_ai_pipeline.config import InferenceConfig
from physical_ai_pipeline.frame import FaceObservation, FrameRecord, Hand, HandDetection, Pose
from physical_ai_pipeline.inference.backend import InferenceBackend, InferenceInput
from physical_ai_pipeline.inference.coco import LEFT_WRIST, PERSON_CLASS_ID, RIGHT_WRIST
from physical_ai_pipeline.inference.geometry import center, diagonal
from physical_ai_pipeline.logging_utils import get_logger, log_event

_log = get_logger(__name__)


def assign_handedness(hands: list[HandDetection], poses: list[Pose], min_conf: float = 0.3) -> list[HandDetection]:
    """Label each hand left/right by its nearest body-pose wrist keypoint."""
    wrists: list[tuple[float, float, Hand]] = []
    for pose in poses:
        for idx, side in ((LEFT_WRIST, Hand.LEFT), (RIGHT_WRIST, Hand.RIGHT)):
            x, y, c = pose.keypoints[idx]
            if c >= min_conf:
                wrists.append((float(x), float(y), side))
    if not wrists:
        return hands
    labelled: list[HandDetection] = []
    for hand in hands:
        hx, hy = center(hand.bbox)
        best = min(wrists, key=lambda w: (w[0] - hx) ** 2 + (w[1] - hy) ** 2)
        within = np.hypot(best[0] - hx, best[1] - hy) <= max(diagonal(hand.bbox), 1.0)
        labelled.append(HandDetection(hand.bbox, hand.confidence, hand.keypoints, best[2] if within else Hand.UNKNOWN))
    return labelled


class BatchedInferenceRunner:
    def __init__(self, backend: InferenceBackend, config: InferenceConfig) -> None:
        self.backend = backend
        self.config = config
        self.frames_inferred = 0
        self.face_frames = 0
        self.gpu_seconds = 0.0

    def run(self, records: Sequence[FrameRecord]) -> None:
        """Populate ``record.inference`` for every record that still has pixels."""
        todo = [r for r in records if r.image is not None]
        size = self.config.batch_size
        for start in range(0, len(todo), size):
            chunk = todo[start : start + size]
            t0 = time.perf_counter()
            outputs = self.backend.infer([InferenceInput(r.index, r.image) for r in chunk if r.image is not None])
            self.gpu_seconds += time.perf_counter() - t0
            if len(outputs) != len(chunk):
                raise RuntimeError(f"backend returned {len(outputs)} results for {len(chunk)} frames")
            for record, inference in zip(chunk, outputs, strict=True):
                inference.person_count = sum(
                    1
                    for d in inference.objects
                    if d.class_id == PERSON_CLASS_ID and d.confidence >= self.config.person_confidence
                )
                inference.hands = assign_handedness(inference.hands, inference.poses)
                record.inference = inference
            self.frames_inferred += len(chunk)
            log_event(_log, "gpu_batch", level=10, frames=len(chunk), first_frame=chunk[0].index)

    def analyze_faces(self, records: Sequence[FrameRecord]) -> dict[int, list[FaceObservation]]:
        """Run the face models on ``records``. Returned observations are
        transient and must be discarded by the caller after the decision."""
        todo = [r for r in records if r.image is not None]
        if not todo:
            return {}
        t0 = time.perf_counter()
        results = self.backend.analyze_faces([InferenceInput(r.index, r.image) for r in todo if r.image is not None])
        self.gpu_seconds += time.perf_counter() - t0
        self.face_frames += len(todo)
        return {r.index: faces for r, faces in zip(todo, results, strict=True)}
