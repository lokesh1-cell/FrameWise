from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from physical_ai_pipeline.config import ChildDetectionConfig, FilterMode, LuminanceConfig, PipelineConfig


def test_defaults_match_specification() -> None:
    cfg = PipelineConfig()
    assert cfg.tier1.luminance.min_mean == 40 and cfg.tier1.luminance.max_mean == 230
    assert cfg.tier1.blur.min_laplacian_variance == 100
    assert cfg.tier1.duplicate.min_hamming_distance == 3
    assert cfg.tier1.idle.min_flow_magnitude == 0.5 and cfg.tier1.idle.min_consecutive_frames == 30
    child = cfg.tier3.child
    assert (child.sample_interval, child.buffer_frames_before, child.buffer_frames_after) == (15, 30, 30)
    assert child.max_age_exclusive == 18 and child.min_confidence == 0.85
    assert cfg.tier3.phone.iou_threshold == 0.3
    assert cfg.tier3.fake_work.min_consecutive_frames == 60
    assert cfg.smoothing.max_gap_fill_frames == 2 and cfg.smoothing.min_island_frames == 30
    assert cfg.inference.batch_size in (16, 32)
    assert cfg.decision_lag_frames == 60


def test_child_filter_cannot_be_weakened() -> None:
    with pytest.raises(ValidationError):
        ChildDetectionConfig(mode=FilterMode.SOFT)
    with pytest.raises(ValidationError):
        ChildDetectionConfig(mode=FilterMode.SHADOW)
    with pytest.raises(ValidationError):
        ChildDetectionConfig(enabled=False)


def test_invalid_luminance_range_rejected() -> None:
    with pytest.raises(ValidationError):
        LuminanceConfig(min_mean=200, max_mean=100)


def test_batch_size_restricted() -> None:
    with pytest.raises(ValidationError):
        PipelineConfig.load(inference={"batch_size": 24})


def test_yaml_file_and_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text("tier1:\n  blur:\n    min_laplacian_variance: 55\n  luminance:\n    mode: HARD\n")
    cfg = PipelineConfig.load(path)
    assert cfg.tier1.blur.min_laplacian_variance == 55
    assert cfg.tier1.luminance.mode is FilterMode.HARD
    monkeypatch.setenv("PAP_TIER1__BLUR__MIN_LAPLACIAN_VARIANCE", "70")
    assert PipelineConfig.load(path).tier1.blur.min_laplacian_variance == 70


def test_unknown_keys_rejected() -> None:
    with pytest.raises(ValidationError):
        PipelineConfig.load(tier1={"blurr": {}})
