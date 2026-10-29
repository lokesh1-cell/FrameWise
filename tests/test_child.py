"""Child detection: 15-frame sampling, [F_k-30, F_k+30] exclusion, merging,
person-change override, zero-storage guarantees."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import PERSON, child_face, run_session
from physical_ai_pipeline.config import ChildDetectionConfig
from physical_ai_pipeline.filters.tier3.child import ChildDetectionFilter, exclusion_range, merge_ranges
from physical_ai_pipeline.frame import DropReason, FaceObservation, FrameRecord
from physical_ai_pipeline.synthetic import SceneGenerator, blurred
from physical_ai_pipeline.timeline import FrameWindow
from physical_ai_pipeline.verify import verify_mcap


@pytest.mark.parametrize(
    ("anchor", "total", "expected"),
    [(45, 300, (15, 75)), (10, 300, (0, 40)), (0, 300, (0, 30)), (290, 300, (260, 299)), (45, None, (15, 75))],
)
def test_exclusion_range_clamped(anchor: int, total: int | None, expected: tuple[int, int]) -> None:
    assert exclusion_range(anchor, 30, 30, total) == expected


def test_full_buffer_is_61_frames() -> None:
    start, end = exclusion_range(100, 30, 30, 1000)
    assert end - start + 1 == 61


def test_merge_ranges() -> None:
    assert merge_ranges([(15, 75), (60, 120)]) == [(15, 120)]
    assert merge_ranges([(60, 120), (15, 75), (200, 260)]) == [(15, 120), (200, 260)]
    assert merge_ranges([(0, 10), (11, 20)]) == [(0, 20)]  # touching -> one block
    assert merge_ranges([]) == []


def test_threshold_boundaries() -> None:
    flt = ChildDetectionFilter(ChildDetectionConfig())
    assert flt.is_child(FaceObservation((0, 0, 1, 1), 0.86, 17.9))
    assert not flt.is_child(FaceObservation((0, 0, 1, 1), 0.85, 5.0))  # confidence must be > 0.85
    assert not flt.is_child(FaceObservation((0, 0, 1, 1), 0.99, 18.0))  # age must be < 18


def test_scheduled_detection_drops_exactly_61_frames(tmp_path: Path, scene: SceneGenerator) -> None:
    images = [scene.frame(i) for i in range(300)]
    run = run_session(tmp_path, images, [
        {"start": 0, "end": 299, "objects": [PERSON]},
        {"start": 104, "end": 106, "faces": [child_face()]},  # only sampled frame 105 sees it
    ])  # fmt: skip
    dropped = sorted(set(range(300)) - set(run.kept))
    assert dropped == list(range(75, 136))
    assert run.backend.face_calls == list(range(0, 300, 15))
    rows = [r for r in run.audit.rows if r["reason"] == "CHILD_DETECTED"]
    assert len(rows) == 1 and rows[0]["anchor_frame"] == 105 and rows[0]["dropped_range"] == [75, 135]


def test_detection_near_start_and_end_clamped(tmp_path: Path, scene: SceneGenerator) -> None:
    images = [scene.frame(i) for i in range(130)]
    run = run_session(tmp_path, images, [
        {"start": 0, "end": 129, "objects": [PERSON]},
        {"start": 0, "end": 0, "faces": [child_face()]},
        {"start": 120, "end": 120, "faces": [child_face()]},
    ])  # fmt: skip
    assert run.kept == list(range(31, 90))
    ranges = sorted(r["dropped_range"] for r in run.audit.rows if r["reason"] == "CHILD_DETECTED")
    assert ranges == [[0, 30], [90, 129]]


def test_overlapping_detections_merge_into_one_block(tmp_path: Path, scene: SceneGenerator) -> None:
    images = [scene.frame(i) for i in range(400)]
    run = run_session(tmp_path, images, [
        {"start": 0, "end": 399, "objects": [PERSON]},
        {"start": 100, "end": 140, "faces": [child_face()]},  # sampled at 105, 120, 135
    ])  # fmt: skip
    child_blocks = [d for d in run.result.drop_ranges if d.reason is DropReason.CHILD_DETECTED]
    assert [(d.start_frame, d.end_frame) for d in child_blocks] == [(75, 165)]
    assert set(range(75, 166)).isdisjoint(run.kept)
    assert len([r for r in run.audit.rows if r["reason"] == "CHILD_DETECTED"]) == 3
    report = verify_mcap(run.mcap)
    assert report.ok, report.errors
    assert report.child_blocks == [[75, 165]]


def test_child_staying_extends_block_via_checks_inside_exclusion(tmp_path: Path, scene: SceneGenerator) -> None:
    images = [scene.frame(i) for i in range(500)]
    run = run_session(tmp_path, images, [
        {"start": 0, "end": 499, "objects": [PERSON]},
        {"start": 60, "end": 300, "faces": [child_face()]},
    ])  # fmt: skip
    # Sampled detections at 60..300 -> union [30, 330].
    assert run.kept == list(range(0, 30)) + list(range(331, 500))


def test_person_change_triggers_out_of_cycle_check(tmp_path: Path, scene: SceneGenerator) -> None:
    images = [scene.frame(i) for i in range(300)]
    child_person = {"class": "person", "bbox": [200, 20, 260, 175], "conf": 0.9}
    run = run_session(tmp_path, images, [
        {"start": 0, "end": 299, "objects": [PERSON]},
        # Child visible only on frames 112-116: no sampled frame (105, 120) sees it.
        {"start": 112, "end": 116, "objects": [child_person], "faces": [child_face()]},
    ])  # fmt: skip
    assert 112 in run.backend.face_calls  # person count 1 -> 2 at frame 112
    assert 117 in run.backend.face_calls  # and 2 -> 1 at frame 117
    assert sorted(set(range(300)) - set(run.kept)) == list(range(82, 143))
    assert run.result.stats["child_checks_out_of_cycle"] >= 1


def test_without_override_the_child_is_missed(tmp_path: Path, scene: SceneGenerator) -> None:
    from physical_ai_pipeline.config import PipelineConfig

    images = [scene.frame(i) for i in range(300)]
    cfg = PipelineConfig.load(tier3={"child": {"person_change_override": False}})
    run = run_session(tmp_path, images, [
        {"start": 0, "end": 299, "objects": [PERSON]},
        {"start": 112, "end": 116, "faces": [child_face()]},
    ], config=cfg)  # fmt: skip
    assert run.result.stats["child_detection_events"] == 0


def test_scheduled_frame_dropped_by_tier1_defers_check(tmp_path: Path, scene: SceneGenerator) -> None:
    images = [scene.frame(i) for i in range(200)]
    images[60] = blurred(images[60])
    run = run_session(tmp_path, images, [
        {"start": 0, "end": 199, "objects": [PERSON]},
        {"start": 61, "end": 61, "faces": [child_face()]},
    ])  # fmt: skip
    assert 60 not in run.backend.face_calls and 61 in run.backend.face_calls
    assert sorted(set(range(200)) - set(run.kept)) == list(range(31, 92))


def test_adult_and_low_confidence_faces_ignored(tmp_path: Path, scene: SceneGenerator) -> None:
    images = [scene.frame(i) for i in range(120)]
    run = run_session(tmp_path, images, [
        {"start": 0, "end": 119, "objects": [PERSON], "faces": [child_face(age=34), child_face(age=9, score=0.6)]},
    ])  # fmt: skip
    assert run.kept == list(range(120))


def test_pixels_released_immediately_including_future_frames() -> None:
    cfg = ChildDetectionConfig()
    flt = ChildDetectionFilter(cfg, total_frames=200)
    window = FrameWindow()
    import numpy as np

    records = [FrameRecord(i, i * 33_333_333, np.zeros((4, 4, 3), np.uint8)) for i in range(50)]
    for r in records:
        flt.on_frame_arrival(r)
        window.add(r)
    anchor = records[45]
    assert flt.check(anchor, [FaceObservation((0, 0, 1, 1), 0.95, 6.0)], window)
    owed = {15, 30}  # scheduled frames whose own check has not run yet
    assert all(r.image is None for r in records[15:50] if r.index not in owed)
    assert all(records[i].image is not None for i in owed)
    for i in owed:
        flt.check(records[i], [], window)
        assert records[i].image is None
    assert all(r.image is not None for r in records[:15])
    # Frames 50..75 arrive later and are released on arrival.
    late = FrameRecord(60, 0, np.zeros((4, 4, 3), np.uint8))
    window.add(late)
    assert late.child_dropped and late.image is None


def test_scheduled_frame_inside_exclusion_keeps_pixels_until_checked() -> None:
    import numpy as np

    flt = ChildDetectionFilter(ChildDetectionConfig(), total_frames=200)
    window = FrameWindow()
    records = [FrameRecord(i, 0, np.zeros((4, 4, 3), np.uint8)) for i in range(61)]
    for r in records:
        flt.on_frame_arrival(r)
        window.add(r)
    flt.check(records[45], [FaceObservation((0, 0, 1, 1), 0.95, 6.0)], window)
    assert records[60].child_dropped and records[60].image is not None  # check still owed
    flt.check(records[60], [], window)
    assert records[60].image is None


def test_no_face_data_anywhere(tmp_path: Path, scene: SceneGenerator) -> None:
    images = [scene.frame(i) for i in range(150)]
    run = run_session(tmp_path, images, [
        {"start": 0, "end": 149, "objects": [PERSON]},
        {"start": 75, "end": 75, "faces": [child_face(age=7.25, score=0.9731)]},
    ])  # fmt: skip
    child_rows = [r for r in run.audit.rows if r["reason"] == "CHILD_DETECTED"]
    assert {k: child_rows[0][k] for k in ("reason", "anchor_frame", "dropped_range")} == {
        "reason": "CHILD_DETECTED", "anchor_frame": 75, "dropped_range": [45, 105]}  # fmt: skip
    blob = json.dumps(run.audit.rows) + run.mcap.read_bytes().decode("latin-1")
    assert "7.25" not in blob and "0.9731" not in blob
    for record_field in FrameRecord.__dataclass_fields__:
        assert "face" not in record_field and "age" not in record_field.split("_")
