"""Inference backend interface."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from physical_ai_pipeline.frame import FaceObservation, FrameInference, ImageBGR


@dataclass(frozen=True, slots=True)
class InferenceInput:
    index: int
    image: ImageBGR


@dataclass(frozen=True, slots=True)
class ModelInfo:
    name: str
    role: str
    engine_path: str
    sha256: str
    precision: str


class InferenceBackend(Protocol):
    def infer(self, batch: Sequence[InferenceInput]) -> list[FrameInference]:
        """Run detector, pose, hands (and depth when enabled) once per batch."""
        ...

    def analyze_faces(self, batch: Sequence[InferenceInput]) -> list[list[FaceObservation]]:
        """Face detection + age estimation. Results are transient: callers
        must not store them beyond the child decision."""
        ...

    def model_infos(self) -> list[ModelInfo]: ...

    def close(self) -> None: ...
