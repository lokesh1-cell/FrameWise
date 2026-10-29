"""Pydantic configuration models for every pipeline threshold.

Configuration resolves in this order (later wins):
  1. Field defaults below (the values from the pipeline specification).
  2. A YAML or JSON file passed to :meth:`PipelineConfig.load`.
  3. Environment variables prefixed ``PAP_`` with ``__`` as the nested
     delimiter, e.g. ``PAP_TIER1__BLUR__MIN_LAPLACIAN_VARIANCE=80``.
"""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

PIPELINE_VERSION = "1.0.0"


class FilterMode(StrEnum):
    """How a filter's verdict is applied.

    SHADOW: evaluate and log only; the frame is kept.
    SOFT:   drop the frame, but the temporal smoother may restore it when it
            sits in a short gap inside an otherwise clean sequence.
    HARD:   drop the frame unconditionally; never restored.
    """

    SHADOW = "SHADOW"
    SOFT = "SOFT"
    HARD = "HARD"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class FilterSettings(_Model):
    enabled: bool = True
    mode: FilterMode = FilterMode.SOFT


# ---------------------------------------------------------------------------
# Tier 1: CPU filters
# ---------------------------------------------------------------------------


class DecodeCheckConfig(FilterSettings):
    mode: FilterMode = FilterMode.HARD
    # Fraction of (downscaled) macroblocks that are flat and green-dominant,
    # the signature of FFmpeg error concealment on missing references.
    max_corrupt_block_ratio: float = Field(default=0.02, ge=0.0, le=1.0)
    macroblock_size: int = Field(default=16, ge=4)
    flat_block_max_std: float = Field(default=1.0, ge=0.0)
    green_dominance: float = Field(default=40.0, ge=0.0)

    @field_validator("mode")
    @classmethod
    def _decode_is_hard(cls, v: FilterMode) -> FilterMode:
        # A frame that failed to decode has no pixels; it cannot be kept.
        if v is FilterMode.SOFT:
            raise ValueError("decode check cannot be SOFT: undecodable frames cannot be restored")
        return v


class LuminanceConfig(FilterSettings):
    min_mean: float = Field(default=40.0, ge=0, le=255)
    max_mean: float = Field(default=230.0, ge=0, le=255)

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.min_mean >= self.max_mean:
            raise ValueError("luminance.min_mean must be < luminance.max_mean")
        return self


class BlurConfig(FilterSettings):
    # Calibrated against the Tier 1 analysis resolution (Tier1Config.analysis_width).
    min_laplacian_variance: float = Field(default=100.0, ge=0)


class DuplicateConfig(FilterSettings):
    # Drop when hamming(pHash(prev), pHash(cur)) < min_hamming_distance.
    min_hamming_distance: int = Field(default=3, ge=0, le=64)


class IdleConfig(FilterSettings):
    # Mean Farneback magnitude, expressed in full-resolution pixels.
    min_flow_magnitude: float = Field(default=0.5, ge=0)
    min_consecutive_frames: int = Field(default=30, ge=1)
    flow_width: int = Field(default=320, ge=32)


class Tier1Config(_Model):
    # Frames are downscaled to this width once for luminance/blur/pHash/corruption.
    analysis_width: int = Field(default=640, ge=64)
    # Threads computing Tier 1 measurements ahead of the sequential decisions.
    metric_workers: int = Field(default=4, ge=1)
    decode: DecodeCheckConfig = DecodeCheckConfig()
    luminance: LuminanceConfig = LuminanceConfig()
    blur: BlurConfig = BlurConfig()
    duplicate: DuplicateConfig = DuplicateConfig()
    idle: IdleConfig = IdleConfig()


# ---------------------------------------------------------------------------
# Tier 2: GPU inference
# ---------------------------------------------------------------------------


class Precision(StrEnum):
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"


class ModelConfig(_Model):
    enabled: bool = True
    engine_path: str
    # Expected SHA256 of the engine file; verified at load when set.
    sha256: str | None = None
    precision: Precision = Precision.FP16
    input_size: int = Field(default=640, ge=32)
    confidence_threshold: float = Field(default=0.25, ge=0, le=1)
    nms_iou_threshold: float = Field(default=0.45, ge=0, le=1)
    max_detections: int = Field(default=100, ge=1)


class InferenceConfig(_Model):
    batch_size: Literal[16, 32] = 32
    engine_dir: str = "/models/engines"
    detector: ModelConfig = ModelConfig(engine_path="yolov8n_fp16.engine")
    pose: ModelConfig = ModelConfig(engine_path="yolov8n-pose_fp16.engine")
    face_detector: ModelConfig = ModelConfig(
        engine_path="scrfd_10g_fp16.engine", confidence_threshold=0.5, nms_iou_threshold=0.4
    )
    face_attributes: ModelConfig = ModelConfig(
        engine_path="genderage_int8.engine", precision=Precision.INT8, input_size=96
    )
    hands: ModelConfig = ModelConfig(engine_path="yolov8n-hands_fp16.engine", confidence_threshold=0.35)
    depth: ModelConfig = ModelConfig(enabled=False, engine_path="depth_anything_v2_vits_fp16.engine", input_size=518)
    # Detector confidence for counting persons (person-change override).
    person_confidence: float = Field(default=0.5, ge=0, le=1)

    @model_validator(mode="after")
    def _face_required(self) -> Self:
        if not (self.face_detector.enabled and self.face_attributes.enabled):
            raise ValueError("face models are required for child detection and cannot be disabled")
        if not self.detector.enabled:
            raise ValueError("object detector is required (person counting, phone, off-task)")
        return self


# ---------------------------------------------------------------------------
# Tier 3: decision filters
# ---------------------------------------------------------------------------


class ChildDetectionConfig(FilterSettings):
    mode: FilterMode = FilterMode.HARD
    sample_interval: int = Field(default=15, ge=1)
    buffer_frames_before: int = Field(default=30, ge=0)
    buffer_frames_after: int = Field(default=30, ge=0)
    # Child when estimated age < max_age_exclusive AND confidence > min_confidence.
    max_age_exclusive: float = Field(default=18.0, gt=0)
    min_confidence: float = Field(default=0.85, ge=0, le=1)
    person_change_override: bool = True

    @field_validator("mode")
    @classmethod
    def _child_is_hard(cls, v: FilterMode) -> FilterMode:
        if v is not FilterMode.HARD:
            raise ValueError("child detection is a legal-compliance filter and must be HARD")
        return v

    @field_validator("enabled")
    @classmethod
    def _child_enabled(cls, v: bool) -> bool:
        if not v:
            raise ValueError("child detection cannot be disabled")
        return v


class PhoneUsageConfig(FilterSettings):
    iou_threshold: float = Field(default=0.3, ge=0, le=1)
    # Wrist/ear keypoints become square boxes of side = fraction * person bbox height.
    keypoint_box_fraction: float = Field(default=0.12, gt=0, le=1)
    keypoint_min_confidence: float = Field(default=0.5, ge=0, le=1)
    phone_min_confidence: float = Field(default=0.4, ge=0, le=1)


class OffTaskConfig(FilterSettings):
    high_risk_classes: tuple[str, ...] = (
        "tv",
        "remote",
        "sports ball",
        "frisbee",
        "kite",
        "skateboard",
        "wine glass",
        "teddy bear",
        "dog",
        "cat",
    )
    # Session-level task whitelist; empty disables the filter.
    task_whitelist: tuple[str, ...] = ()
    min_confidence: float = Field(default=0.5, ge=0, le=1)


class FakeWorkConfig(FilterSettings):
    # Wrist speed in person-bbox-heights per frame.
    min_wrist_velocity: float = Field(default=0.02, ge=0)
    min_consecutive_frames: int = Field(default=60, ge=1)
    keypoint_min_confidence: float = Field(default=0.5, ge=0, le=1)


class Tier3Config(_Model):
    child: ChildDetectionConfig = ChildDetectionConfig()
    phone: PhoneUsageConfig = PhoneUsageConfig()
    off_task: OffTaskConfig = OffTaskConfig()
    fake_work: FakeWorkConfig = FakeWorkConfig()


# ---------------------------------------------------------------------------
# Tier 4: post-processing
# ---------------------------------------------------------------------------


class SmoothingConfig(_Model):
    enabled: bool = True
    max_gap_fill_frames: int = Field(default=2, ge=0)
    min_island_frames: int = Field(default=30, ge=1)


class GraspConfig(_Model):
    contact_iou: float = Field(default=0.05, ge=0, le=1)
    approach_distance: float = Field(default=1.5, gt=0)
    hold_after_frames: int = Field(default=5, ge=1)
    release_grace_frames: int = Field(default=3, ge=0)
    max_frame_gap: int = Field(default=5, ge=1)
    min_segment_frames: int = Field(default=5, ge=1)


# ---------------------------------------------------------------------------
# Output / cloud
# ---------------------------------------------------------------------------


class OutputConfig(_Model):
    jpeg_quality: int = Field(default=90, ge=1, le=100)
    frame_id: str = "camera"
    write_annotations: bool = True
    encode_threads: int = Field(default=4, ge=1)
    mcap_chunk_size: int = Field(default=4 * 1024 * 1024, ge=1024)
    write_mp4: bool = True
    mp4_timing: Literal["continuous", "original"] = "continuous"
    imu_offset_s: float = 0.0  # shift IMU time (relative to its first sample) against video time


class CloudConfig(_Model):
    project_id: str = "project"
    subscription: str = "raw-video-finalize-sub"
    raw_bucket: str = "project-raw"
    output_bucket: str = "project-clean"
    output_prefix: str = "sessions"
    models_uri: str = "gs://project-models/engines"
    bigquery_table: str = "project.physical_ai.frame_audit_log"
    scratch_dir: str = "/scratch"
    video_extensions: tuple[str, ...] = (".mp4", ".mov", ".mkv", ".avi", ".webm")
    max_lease_seconds: int = Field(default=4 * 3600, ge=60)
    bigquery_batch_rows: int = Field(default=500, ge=1)


class PipelineConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PAP_", env_nested_delimiter="__", extra="forbid", frozen=True)

    pipeline_version: str = PIPELINE_VERSION
    fallback_fps: float = Field(default=30.0, gt=0)
    tier1: Tier1Config = Tier1Config()
    inference: InferenceConfig = InferenceConfig()
    tier3: Tier3Config = Tier3Config()
    smoothing: SmoothingConfig = SmoothingConfig()
    grasp: GraspConfig = GraspConfig()
    output: OutputConfig = OutputConfig()
    cloud: CloudConfig = CloudConfig()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Environment overrides file values (passed via init kwargs).
        return (env_settings, init_settings)

    @classmethod
    def load(cls, path: str | Path | None = None, **overrides: Any) -> PipelineConfig:
        data: dict[str, Any] = {}
        if path is not None:
            data = _read_mapping(Path(path))
        data.update(overrides)
        return cls(**data)

    @property
    def decision_lag_frames(self) -> int:
        """Frames a record must trail the processing frontier before every
        retroactive filter (child buffer, idle run, fake-work run) is final."""
        t3 = self.tier3
        return max(
            t3.child.buffer_frames_before,
            self.tier1.idle.min_consecutive_frames,
            t3.fake_work.min_consecutive_frames,
        )

    def thresholds_json(self) -> str:
        return self.model_dump_json(exclude={"cloud"})


def _read_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        loaded = yaml.safe_load(text) or {}
    else:
        loaded = json.loads(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"config file {path} must contain a mapping")
    return loaded
