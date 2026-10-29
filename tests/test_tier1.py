from __future__ import annotations

import numpy as np

from physical_ai_pipeline.config import FilterMode, PipelineConfig, Tier1Config
from physical_ai_pipeline.decode.video_reader import iter_arrays
from physical_ai_pipeline.filters.tier1 import Tier1Precompute, Tier1Stage
from physical_ai_pipeline.filters.tier1.metrics import compute_metrics, corrupt_block_ratio, hamming, phash
from physical_ai_pipeline.frame import DropReason, FrameRecord
from physical_ai_pipeline.synthetic import SceneGenerator, blurred, corrupt, dark, overexposed
from physical_ai_pipeline.timeline import FrameWindow


def run_stage(images: list[np.ndarray | None], config: Tier1Config | None = None) -> list[FrameRecord]:
    cfg = config or PipelineConfig().tier1
    stage, window = Tier1Stage(cfg), FrameWindow()
    records = list(iter_arrays(images))
    for record in records:
        window.add(record)
        stage.process(record, window)
    return records


def reasons(record: FrameRecord) -> set[DropReason]:
    return {v.reason for v in record.verdicts}


def test_clean_scene_passes(scene: SceneGenerator) -> None:
    records = run_stage([scene.frame(i) for i in range(40)])
    assert all(not r.dropped for r in records)
    assert all(r.quality.flow_magnitude > 0.5 for r in records[1:])


def test_decode_failure_dropped_hard(scene: SceneGenerator) -> None:
    records = run_stage([scene.frame(0), None, scene.frame(2)])
    assert reasons(records[1]) == {DropReason.DECODE_ERROR}
    assert records[1].hard_dropped
    assert not records[2].dropped


def test_corrupt_macroblocks_detected(scene: SceneGenerator) -> None:
    img = corrupt(scene.frame(3), fraction=0.2)
    assert corrupt_block_ratio(img, 16, 1.0, 40.0) > 0.1
    records = run_stage([scene.frame(2), img])
    assert reasons(records[1]) == {DropReason.DECODE_ERROR}


def test_luminance_under_and_over(scene: SceneGenerator) -> None:
    records = run_stage([scene.frame(0), dark(scene.frame(1)), scene.frame(2), overexposed(scene.frame(3))])
    assert reasons(records[1]) == {DropReason.UNDEREXPOSED}
    assert reasons(records[3]) == {DropReason.OVEREXPOSED}
    assert not records[0].dropped and not records[2].dropped


def test_blur_detection(scene: SceneGenerator) -> None:
    records = run_stage([scene.frame(0), blurred(scene.frame(1), 31)])
    assert reasons(records[1]) == {DropReason.BLURRY}
    assert records[1].quality.laplacian_variance < 100 < records[0].quality.laplacian_variance


def test_duplicate_detection(scene: SceneGenerator) -> None:
    f = scene.frame(5)
    records = run_stage([scene.frame(4), f, f.copy()])
    assert reasons(records[2]) == {DropReason.DUPLICATE}
    assert records[2].quality.phash_hamming == 0


def test_phash_properties(scene: SceneGenerator) -> None:
    import cv2

    a = cv2.cvtColor(scene.frame(0), cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(scene.frame(10), cv2.COLOR_BGR2GRAY)
    assert hamming(phash(a), phash(a)) == 0
    assert hamming(phash(a), phash(b)) >= 3
    assert 0 <= phash(a) < 2**64


def test_idle_run_dropped_retroactively(scene: SceneGenerator) -> None:
    cfg = PipelineConfig.load(tier1={"duplicate": {"enabled": False}}).tier1
    still = scene.frame(0)
    rng = np.random.default_rng(0)
    # Static camera with sensor noise: not duplicates by pixel, but no motion.
    static = [
        np.clip(still.astype(np.int16) + rng.integers(-1, 2, still.shape), 0, 255).astype(np.uint8) for _ in range(45)
    ]
    images = [scene.frame(i) for i in range(10)] + static + [scene.frame(i) for i in range(60, 70)]
    records = run_stage(images, cfg)
    idle = [r.index for r in records if DropReason.IDLE in reasons(r)]
    # frame 10 differs from frame 9 (scene moved), so the run starts at 11.
    assert idle == list(range(11, 55))


def test_short_static_run_not_dropped(scene: SceneGenerator) -> None:
    cfg = PipelineConfig.load(tier1={"duplicate": {"enabled": False}}).tier1
    images = [scene.frame(i) for i in range(5)] + [scene.frame(5)] * 20 + [scene.frame(i) for i in range(30, 35)]
    assert not any(DropReason.IDLE in reasons(r) for r in run_stage(images, cfg))


def test_shadow_mode_keeps_frame(scene: SceneGenerator) -> None:
    cfg = PipelineConfig.load(tier1={"blur": {"mode": "SHADOW"}}).tier1
    records = run_stage([scene.frame(0), blurred(scene.frame(1))], cfg)
    assert not records[1].dropped
    assert {v.reason for v in records[1].shadow_verdicts} == {DropReason.BLURRY}


def test_fail_fast_stops_at_first_drop(scene: SceneGenerator) -> None:
    # A dark frame is also blurry-looking; only the first filter in order fires.
    records = run_stage([scene.frame(0), dark(scene.frame(1))])
    assert [v.filter_name for v in records[1].verdicts] == ["luminance"]


def test_precompute_matches_inline(scene: SceneGenerator) -> None:
    cfg = PipelineConfig().tier1
    images = [scene.frame(i) for i in range(12)] + [blurred(scene.frame(12))]
    inline = [compute_metrics(images[i], images[i - 1] if i else None, cfg) for i in range(len(images))]
    threaded = [r.tier1 for r in Tier1Precompute(cfg, workers=3, max_inflight=4)(iter_arrays(images))]
    for a, b in zip(inline, threaded, strict=True):
        assert b is not None
        assert a.phash == b.phash and abs(a.laplacian_variance - b.laplacian_variance) < 1e-6
        assert abs(a.flow_magnitude - b.flow_magnitude) < 1e-4


def test_filter_modes_soft_vs_hard(scene: SceneGenerator) -> None:
    cfg = PipelineConfig.load(tier1={"blur": {"mode": "HARD"}}).tier1
    soft = run_stage([scene.frame(0), blurred(scene.frame(1))])[1]
    hard = run_stage([scene.frame(0), blurred(scene.frame(1))], cfg)[1]
    assert soft.verdicts[0].mode is FilterMode.SOFT and soft.restorable
    assert hard.verdicts[0].mode is FilterMode.HARD and not hard.restorable
