from __future__ import annotations

from pathlib import Path

import numpy as np

from conftest import PERSON, run_session
from physical_ai_pipeline.config import FakeWorkConfig, OffTaskConfig, PhoneUsageConfig
from physical_ai_pipeline.filters.tier3 import FakeWorkFilter, OffTaskFilter, PhoneUsageFilter
from physical_ai_pipeline.frame import Detection, DropReason, FrameInference, FrameRecord, HandDetection, Pose
from physical_ai_pipeline.inference.coco import COCO_CLASSES, COCO_KEYPOINTS
from physical_ai_pipeline.synthetic import SceneGenerator
from physical_ai_pipeline.timeline import FrameWindow


def det(name: str, box: tuple[float, float, float, float], conf: float = 0.9) -> Detection:
    return Detection(box, conf, COCO_CLASSES.index(name), name)


def pose(keypoints: dict[str, tuple[float, float]], box: tuple[float, float, float, float] = (0, 0, 100, 200)) -> Pose:
    kps = np.zeros((17, 3), np.float32)
    for name, (x, y) in keypoints.items():
        kps[COCO_KEYPOINTS.index(name)] = (x, y, 0.9)
    return Pose(box, 0.9, kps)


def record(index: int, inference: FrameInference) -> FrameRecord:
    r = FrameRecord(index, index, np.zeros((2, 2, 3), np.uint8))
    r.inference = inference
    return r


def test_phone_at_ear_dropped() -> None:
    flt = PhoneUsageFilter(PhoneUsageConfig())
    r = record(0, FrameInference(objects=[det("cell phone", (40, 10, 60, 34))], poses=[pose({"left_ear": (50, 22)})]))
    assert flt.process(r, FrameWindow())
    assert r.verdicts[0].reason is DropReason.PHONE_USAGE


def test_phone_on_table_kept() -> None:
    flt = PhoneUsageFilter(PhoneUsageConfig())
    r = record(
        0, FrameInference(objects=[det("cell phone", (300, 300, 320, 324))], poses=[pose({"left_wrist": (50, 120)})])
    )
    assert not flt.process(r, FrameWindow())


def test_off_task_requires_whitelist_absent() -> None:
    flt = OffTaskFilter(OffTaskConfig(), task_whitelist=("cup",))
    risky = record(0, FrameInference(objects=[det("sports ball", (0, 0, 10, 10))]))
    on_task = record(1, FrameInference(objects=[det("sports ball", (0, 0, 10, 10)), det("cup", (20, 20, 30, 30))]))
    assert flt.process(risky, FrameWindow())
    assert not flt.process(on_task, FrameWindow())


def test_off_task_disabled_without_whitelist() -> None:
    flt = OffTaskFilter(OffTaskConfig())
    assert not flt.enabled


def test_fake_work_after_60_frames_retroactive() -> None:
    flt = FakeWorkFilter(FakeWorkConfig())
    window = FrameWindow()
    records = []
    for i in range(80):
        x = 50 + (20 if i % 2 else -20)  # wrist oscillates 40 px/frame, person 200 px tall
        inf = FrameInference(objects=[det("cup", (400, 400, 420, 420))], poses=[pose({"left_wrist": (x, 100)})])
        r = record(i, inf)
        records.append(r)
        window.add(r)
        flt.process(r, window)
    fake = [r.index for r in records if r.dropped]
    assert fake == list(range(1, 80))  # frame 0 has no previous position


def test_real_work_not_flagged() -> None:
    flt = FakeWorkFilter(FakeWorkConfig())
    window = FrameWindow()
    for i in range(100):
        x = 50 + (20 if i % 2 else -20)
        inf = FrameInference(
            objects=[det("cup", (x - 10, 90, x + 10, 110))],
            poses=[pose({"left_wrist": (x, 100)})],
            hands=[HandDetection((x - 8, 92, x + 8, 108), 0.9, np.zeros((21, 3), np.float32))],
        )
        r = record(i, inf)
        window.add(r)
        assert not flt.process(r, window)


def test_phone_usage_end_to_end(tmp_path: Path, scene: SceneGenerator) -> None:
    images = [scene.frame(i) for i in range(200)]
    worker = {"bbox": [40, 20, 160, 175], "keypoints": {"left_ear": [90, 40, 0.9]}}
    run = run_session(tmp_path, images, [
        {"start": 0, "end": 199, "objects": [PERSON], "poses": [worker]},
        {"start": 80, "end": 119, "objects": [{"class": "cell phone", "bbox": [82, 32, 98, 50], "conf": 0.8}]},
    ])  # fmt: skip
    assert sorted(set(range(200)) - set(run.kept)) == list(range(80, 120))
