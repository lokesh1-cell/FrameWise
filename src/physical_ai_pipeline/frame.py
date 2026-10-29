"""Core per-frame data structures shared by every pipeline stage."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from physical_ai_pipeline.config import FilterMode

if TYPE_CHECKING:
    from physical_ai_pipeline.filters.tier1.metrics import Tier1Metrics

ImageBGR = npt.NDArray[np.uint8]
BBox = tuple[float, float, float, float]  # x1, y1, x2, y2 in source pixels


class DropReason(StrEnum):
    DECODE_ERROR = "DECODE_ERROR"
    UNDEREXPOSED = "UNDEREXPOSED"
    OVEREXPOSED = "OVEREXPOSED"
    BLURRY = "BLURRY"
    DUPLICATE = "DUPLICATE"
    IDLE = "IDLE"
    CHILD_DETECTED = "CHILD_DETECTED"
    PHONE_USAGE = "PHONE_USAGE"
    OFF_TASK = "OFF_TASK"
    FAKE_WORK = "FAKE_WORK"
    ISOLATED_ISLAND = "ISOLATED_ISLAND"


# Reasons a smoother may never undo, regardless of filter mode.
NON_RESTORABLE: frozenset[DropReason] = frozenset({DropReason.DECODE_ERROR, DropReason.CHILD_DETECTED})


@dataclass(frozen=True, slots=True)
class Verdict:
    reason: DropReason
    mode: FilterMode
    filter_name: str
    detail: tuple[tuple[str, float], ...] = ()


# ---------------------------------------------------------------------------
# Inference outputs (Tier 2 cache)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Detection:
    bbox: BBox
    confidence: float
    class_id: int
    class_name: str


@dataclass(frozen=True, slots=True)
class Pose:
    bbox: BBox
    confidence: float
    keypoints: npt.NDArray[np.float32]  # (17, 3): x, y, conf


class Hand(StrEnum):
    LEFT = "left"
    RIGHT = "right"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class HandDetection:
    bbox: BBox
    confidence: float
    keypoints: npt.NDArray[np.float32]  # (21, 3)
    handedness: Hand = Hand.UNKNOWN


@dataclass(frozen=True, slots=True)
class FaceObservation:
    """Transient face-model output. PRIVACY: never cached on a FrameRecord,
    serialised, logged, or written anywhere. Consumed by the child filter and
    discarded in the same call."""

    bbox: BBox
    det_score: float
    age: float


@dataclass(slots=True)
class FrameInference:
    objects: list[Detection] = field(default_factory=list)
    poses: list[Pose] = field(default_factory=list)
    hands: list[HandDetection] = field(default_factory=list)
    depth: npt.NDArray[np.float32] | None = None
    person_count: int = 0


@dataclass(slots=True)
class QualityMetrics:
    decode_ok: bool = True
    corrupt_block_ratio: float = 0.0
    luminance_mean: float = -1.0
    laplacian_variance: float = -1.0
    phash_hamming: int = -1
    flow_magnitude: float = -1.0


@dataclass(slots=True)
class GraspInfo:
    hand_index: int
    object_index: int
    iou: float
    state: str
    confidence: float
    normalized_distance: float


@dataclass(slots=True)
class FrameRecord:
    index: int
    pts_ns: int
    image: ImageBGR | None
    quality: QualityMetrics = field(default_factory=QualityMetrics)
    inference: FrameInference | None = None
    verdicts: list[Verdict] = field(default_factory=list)
    shadow_verdicts: list[Verdict] = field(default_factory=list)
    child_check_performed: bool = False
    # Set while a scheduled / out-of-cycle child check is owed on this frame;
    # pixel release is deferred until the check has run.
    child_check_pending: bool = False
    release_deferred: bool = False
    child_out_of_cycle: bool = False
    restored: bool = False
    grasps: list[GraspInfo] = field(default_factory=list)
    # Tier 1 measurements (see filters.tier1.metrics.Tier1Metrics).
    tier1: Tier1Metrics | None = None

    @property
    def dropped(self) -> bool:
        return bool(self.verdicts) and not self.restored

    @property
    def hard_dropped(self) -> bool:
        return any(v.mode is FilterMode.HARD or v.reason in NON_RESTORABLE for v in self.verdicts)

    @property
    def child_dropped(self) -> bool:
        return any(v.reason is DropReason.CHILD_DETECTED for v in self.verdicts)

    @property
    def restorable(self) -> bool:
        return bool(self.verdicts) and not self.hard_dropped and self.image is not None

    @property
    def primary_reason(self) -> DropReason | None:
        if not self.verdicts:
            return None
        if self.child_dropped:
            return DropReason.CHILD_DETECTED
        return self.verdicts[0].reason

    def add_verdict(self, verdict: Verdict) -> None:
        if verdict.mode is FilterMode.SHADOW:
            self.shadow_verdicts.append(verdict)
            return
        if any(v.reason is verdict.reason for v in self.verdicts):
            return
        self.verdicts.append(verdict)
        self.restored = False

    def release_pixels(self, *, force: bool = False) -> None:
        """Free image buffers. While a child check is pending (and not
        ``force``) the release is deferred until :meth:`finish_child_check`."""
        if self.child_check_pending and not force:
            self.release_deferred = True
            return
        self.release_deferred = False
        self.image = None
        if self.inference is not None:
            self.inference.depth = None

    def finish_child_check(self) -> None:
        self.child_check_pending = False
        if self.release_deferred or self.child_dropped:
            self.release_pixels(force=True)
