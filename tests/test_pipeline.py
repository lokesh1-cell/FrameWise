"""End-to-end: synthetic video -> pipeline -> MCAP -> verifier."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

from conftest import PERSON, child_face, run_session
from physical_ai_pipeline.cli import main as cli_main
from physical_ai_pipeline.config import PipelineConfig
from physical_ai_pipeline.decode.video_reader import VideoReader, iter_arrays
from physical_ai_pipeline.inference.fake_backend import ScriptedBackend
from physical_ai_pipeline.pipeline import SessionCancelled, SessionInfo, SessionPipeline
from physical_ai_pipeline.synthetic import SceneGenerator, blurred, dark, write_video
from physical_ai_pipeline.verify import verify_mcap


def test_mixed_session_outputs(tmp_path: Path, scene: SceneGenerator) -> None:
    images = [scene.frame(i) for i in range(400)]
    for i in range(200, 204):
        images[i] = dark(images[i])  # 4 soft drops -> not fillable (> 2)
    images[300] = blurred(images[300])  # single soft drop -> filled by smoother
    run = run_session(tmp_path, images, [
        {"start": 0, "end": 399, "objects": [PERSON, {"class": "cup", "bbox": [200, 100, 230, 130]}],
         "hands": [{"bbox": [198, 102, 228, 128]}]},
        {"start": 120, "end": 121, "faces": [child_face()]},
    ])  # fmt: skip
    stats = run.result.stats
    assert stats["total_frames"] == 400
    assert stats["dropped_by_reason"] == {"CHILD_DETECTED": 61, "UNDEREXPOSED": 4}
    assert stats["restored_by_smoothing"] >= 1  # frame 300 (+ any natural near-duplicate)
    assert 300 in run.kept
    report = verify_mcap(run.mcap)
    assert report.ok, report.errors
    assert report.kept_frames == 400 - 65
    with open(run.mcap, "rb") as fh:
        reader = make_reader(fh, decoder_factories=[DecoderFactory()])
        meta = [p for _, _, _, p in reader.iter_decoded_messages(topics=["/session/metadata"])][0]
        assert meta.stats.kept_frames == 335 and json.loads(meta.config_json)["tier3"]["child"]["sample_interval"] == 15
    with open(run.mcap, "rb") as fh:
        reader = make_reader(fh, decoder_factories=[DecoderFactory()])
        actions = [p for _, _, _, p in reader.iter_decoded_messages(topics=["/actions"])]
        assert actions and all(a.object_class == "cup" for a in actions)


def test_fake_work_end_to_end(tmp_path: Path, scene: SceneGenerator) -> None:
    images = [scene.frame(i) for i in range(250)]
    worker = {"bbox": [40, 20, 160, 175], "keypoints": {"left_wrist": [100, 100, 0.9]}}
    run = run_session(tmp_path, images, [
        {"start": 0, "end": 249, "objects": [PERSON, {"class": "cup", "bbox": [260, 140, 290, 170]}]},
        {"start": 0, "end": 99, "poses": [worker]},
        {"start": 100, "end": 179, "poses": [worker], "oscillate": 15.0},  # empty waving, nothing touched
        {"start": 180, "end": 249, "poses": [worker]},
    ], config=PipelineConfig())  # fmt: skip
    # Wrist moves on frames 100..180 (180 snaps back to rest) = 81 frames >= 60.
    assert run.result.stats["dropped_by_reason"].get("FAKE_WORK", 0) == 81
    report = verify_mcap(run.mcap)
    assert report.ok, report.errors


def test_gpu_skipped_for_dropped_frames(tmp_path: Path, scene: SceneGenerator) -> None:
    images = [scene.frame(i) for i in range(200)]
    images[50] = dark(images[50])
    run = run_session(tmp_path, images, [
        {"start": 0, "end": 199, "objects": [PERSON]},
        {"start": 105, "end": 105, "faces": [child_face()]},
    ])  # fmt: skip
    inferred = {i for call in run.backend.infer_calls for i in call}
    assert 50 not in inferred  # Tier 1 drop -> no GPU work
    # Batches of 32 surviving frames: the 4th is 97..128 and contains the
    # detection at 105. Frames 129..135 arrive inside the exclusion and skip
    # detection inference entirely; frame 135 is scheduled, so only the face
    # check runs on it.
    assert not inferred & set(range(129, 136))
    assert 135 in run.backend.face_calls


def test_real_video_roundtrip(tmp_path: Path) -> None:
    gen = SceneGenerator(320, 180, seed=3)
    video = write_video(tmp_path / "in.mp4", gen.frames(90))
    reader = VideoReader(video)
    assert reader.info.width == 320 and abs(reader.info.fps - 30) < 0.01
    records = list(reader.frames())
    assert [r.index for r in records] == list(range(90))
    assert all(b.pts_ns > a.pts_ns for a, b in zip(records, records[1:], strict=False))
    assert abs(records[30].pts_ns - 1_000_000_000) < 1_000_000


def test_cancellation_aborts_and_removes_partial_file(tmp_path: Path, scene: SceneGenerator) -> None:
    cancel = threading.Event()
    cancel.set()
    mcap = tmp_path / "s.mcap"
    with pytest.raises(SessionCancelled):
        SessionPipeline(PipelineConfig(), ScriptedBackend()).run(
            iter_arrays(scene.frame(i) for i in range(50)), SessionInfo("s", "mem://"), mcap, cancel=cancel
        )
    assert not mcap.exists()


def test_cli_synth_run_verify(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    video = tmp_path / "demo.mp4"
    assert cli_main(["synth", str(video), "--frames", "300", "--width", "320", "--height", "180"]) == 0
    out = tmp_path / "out"
    script = video.with_suffix(".script.json")
    assert cli_main(["run", str(video), "--out", str(out), "--fake-script", str(script), "--whitelist", "cup"]) == 0
    assert cli_main(["verify", str(out / "session.mcap")]) == 0
    lineage = json.loads((out / "lineage.json").read_text())
    assert len(lineage["source"]["sha256"]) == 64 and len(lineage["output"]["mcap_sha256"]) == 64
    assert lineage["filter_thresholds"]["tier3"]["child"]["buffer_frames_before"] == 30
    rows = [json.loads(line) for line in (out / "frame_audit_log.jsonl").read_text().splitlines()]
    child = [r for r in rows if r["reason"] == "CHILD_DETECTED"]
    assert child and all(r["dropped_range"][1] - r["dropped_range"][0] == 60 for r in child)
