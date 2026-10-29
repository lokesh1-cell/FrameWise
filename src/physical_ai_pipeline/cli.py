"""Fast CLI runner.

pap synth demo.mp4 --frames 600            # synthetic video + matching fake-model script
pap run demo.mp4 --out out/ --fake-script demo.script.json
pap verify out/session.mcap
pap bench --width 1920 --height 1080       # Tier 1 CPU cost per frame
pap worker --config config.yaml            # Pub/Sub worker (GKE entrypoint)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from physical_ai_pipeline.config import PipelineConfig
from physical_ai_pipeline.logging_utils import configure_logging


def _cmd_run(args: argparse.Namespace) -> int:
    from physical_ai_pipeline.audit import JsonlAuditSink
    from physical_ai_pipeline.cloud.gcs import sha256_path
    from physical_ai_pipeline.decode.video_reader import VideoReader, prefetch
    from physical_ai_pipeline.inference.backend import InferenceBackend
    from physical_ai_pipeline.output.lineage_writer import (
        build_lineage,
        collect_node_info,
        git_commit,
        utcnow,
        write_lineage,
    )
    from physical_ai_pipeline.pipeline import SessionInfo, SessionPipeline

    config = PipelineConfig.load(args.config)
    backend: InferenceBackend
    if args.backend == "tensorrt":
        from physical_ai_pipeline.inference.tensorrt_backend import TensorRTBackend

        backend = TensorRTBackend(config.inference)
    else:
        from physical_ai_pipeline.inference.fake_backend import ScriptedBackend

        backend = ScriptedBackend.from_json(args.fake_script) if args.fake_script else ScriptedBackend()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    source = Path(args.input)
    started = utcnow()
    source_sha = sha256_path(source)
    reader = VideoReader(source, config.fallback_fps)
    whitelist = tuple(w.strip() for w in args.whitelist.split(",") if w.strip()) if args.whitelist else None
    info = SessionInfo(
        session_id=args.session_id or source.stem,
        source_uri=source.resolve().as_uri(),
        source_sha256=source_sha,
        session_start_ns=time.time_ns() if args.start_ns is None else args.start_ns,
        fps=reader.info.fps,
        total_frames_hint=reader.info.frame_count,
        task_whitelist=whitelist,
        git_commit=git_commit(),
    )
    audit = JsonlAuditSink(out / "frame_audit_log.jsonl")
    try:
        result = SessionPipeline(config, backend, audit).run(prefetch(reader.frames()), info, out / "session.mcap")
        if config.output.write_mp4:
            from physical_ai_pipeline.output.mp4_export import export_mp4

            export_mp4(
                out / "session.mcap", out / "session.mp4", fps=reader.info.fps, timing=config.output.mp4_timing
            )
        imu = Path(args.imu) if args.imu else source.with_suffix(".txt")
        audio = Path(args.audio) if args.audio else source.with_suffix(".aac")
        imu_path = imu if imu.exists() else None
        audio_path = audio if audio.exists() else source  # falls back to the video's own audio track
        if imu_path or audio_path:
            from physical_ai_pipeline.output.sensor_export import export_sensors

            export_sensors(
                out / "session.mcap", out, reader.info.fps, config.output.mp4_timing,
                imu_path, audio_path, config.output.imu_offset_s, out / "session.mp4",
            )  # fmt: skip
    finally:
        audit.close()
        backend.close()
    lineage = build_lineage(
        session_id=info.session_id,
        source_uri=info.source_uri,
        source_sha256=source_sha,
        source_generation=None,
        mcap_uri=(out / "session.mcap").resolve().as_uri(),
        mcap_sha256=sha256_path(out / "session.mcap"),
        config=config,
        models=result.models,
        git_commit=info.git_commit,
        stats={**result.stats, "decode_errors": reader.decode_errors},
        node_info=collect_node_info(query_metadata_server=False),
        started_at=started,
        finished_at=utcnow(),
    )
    write_lineage(out / "lineage.json", lineage)
    print(json.dumps({"mcap": str(out / "session.mcap"), "stats": result.stats}, indent=2))
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    from physical_ai_pipeline.verify import verify_mcap

    ok = True
    for path in args.mcap:
        report = verify_mcap(path)
        print(report.to_json())
        ok &= report.ok
    return 0 if ok else 1


def demo_script(frames: int, width: int, height: int) -> dict[str, Any]:
    """Fake-model script for ``pap synth``: a worker grasping a cup, a child
    entering mid-session (with a person-count change), and a phone call."""
    w, h = width, height
    person = [0.15 * w, 0.1 * h, 0.5 * w, 0.95 * h]
    cup = [0.6 * w, 0.55 * h, 0.7 * w, 0.7 * h]
    worker_pose = {"bbox": person, "keypoints": {
        "left_wrist": [0.63 * w, 0.6 * h, 0.9], "right_wrist": [0.35 * w, 0.6 * h, 0.9],
        "left_ear": [0.3 * w, 0.2 * h, 0.9], "right_ear": [0.36 * w, 0.2 * h, 0.9]}}  # fmt: skip
    third = frames // 3
    child_at = third + 7  # not on the 15-frame grid: found by the person-change override
    segments: list[dict[str, Any]] = [
        {"start": 0, "end": frames - 1, "objects": [{"class": "person", "bbox": person, "conf": 0.95},
                                                    {"class": "cup", "bbox": cup, "conf": 0.9}],
         "poses": [worker_pose], "hands": [{"bbox": [0.6 * w, 0.56 * h, 0.67 * w, 0.66 * h], "conf": 0.85}]},
        {"start": child_at, "end": child_at + 20,
         "objects": [{"class": "person", "bbox": [0.75 * w, 0.3 * h, 0.9 * w, 0.8 * h], "conf": 0.9}],
         "faces": [{"bbox": [0.8 * w, 0.32 * h, 0.85 * w, 0.4 * h], "score": 0.93, "age": 7}]},
        {"start": 2 * third, "end": 2 * third + 40,
         "objects": [{"class": "cell phone", "bbox": [0.28 * w, 0.17 * h, 0.38 * w, 0.24 * h], "conf": 0.8}]},
    ]  # fmt: skip
    return {"segments": segments}


def _cmd_synth(args: argparse.Namespace) -> int:
    from physical_ai_pipeline.synthetic import SceneGenerator, blurred, dark, write_video

    gen = SceneGenerator(args.width, args.height, seed=args.seed)

    def frames() -> Any:
        for i in range(args.frames):
            image = gen.frame(i)
            if args.frames // 6 <= i < args.frames // 6 + 4:
                image = dark(image)
            elif i == args.frames // 2:
                image = blurred(image)
            yield image

    path = write_video(args.output, frames(), fps=args.fps)
    script_path = path.with_suffix(".script.json")
    script_path.write_text(json.dumps(demo_script(args.frames, args.width, args.height), indent=2))
    print(json.dumps({"video": str(path), "fake_script": str(script_path)}))
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    from physical_ai_pipeline.decode.video_reader import iter_arrays
    from physical_ai_pipeline.filters.tier1 import Tier1Precompute, Tier1Stage
    from physical_ai_pipeline.synthetic import SceneGenerator
    from physical_ai_pipeline.timeline import FrameWindow

    config = PipelineConfig.load(args.config).tier1
    images = list(SceneGenerator(args.width, args.height).frames(args.frames))
    results: dict[str, float] = {}
    for label, workers in (("inline", 0), (f"threads_{config.metric_workers}", config.metric_workers)):
        stage, window = Tier1Stage(config), FrameWindow()
        records = iter_arrays(images)
        if workers:
            records = Tier1Precompute(config, workers=workers)(records)
        t0 = time.perf_counter()
        for record in records:
            window.add(record)
            stage.process(record, window)
            window.pop_until(record.index - 1)
        results[f"{label}_ms_per_frame"] = round((time.perf_counter() - t0) / args.frames * 1000, 3)
    print(json.dumps({"width": args.width, "height": args.height, "frames": args.frames, **results}, indent=2))
    return 0


def _cmd_worker(args: argparse.Namespace) -> int:
    from physical_ai_pipeline.cloud.worker import run

    return run(PipelineConfig.load(args.config))


def _cmd_clean(args: argparse.Namespace) -> int:
    from physical_ai_pipeline.laptop import LaptopOptions, clean_video

    folder = clean_video(args.input, args.out, LaptopOptions(
        max_width=args.max_width or None, blur_threshold=args.blur_threshold,
        remove_duplicates=args.remove_duplicates, remove_idle=args.remove_idle,
        preserve_audio=not args.no_audio,
    ))
    print((folder / "report.json").read_text())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pap", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--log-level", default="WARNING")
    sub = parser.add_subparsers(dest="command", required=True)

    clean = sub.add_parser("clean", help="laptop CPU quality cleaning into MP4 + audit (no AI models)")
    clean.add_argument("input")
    clean.add_argument("--out", default="output")
    clean.add_argument("--max-width", type=int, default=1280, help="0 preserves source resolution")
    clean.add_argument("--blur-threshold", type=float, default=100)
    clean.add_argument("--remove-duplicates", action="store_true")
    clean.add_argument("--remove-idle", action="store_true")
    clean.add_argument("--no-audio", action="store_true")
    clean.set_defaults(func=_cmd_clean)

    run = sub.add_parser("run", help="process one local video into session.mcap + lineage.json")
    run.add_argument("input")
    run.add_argument("--out", required=True)
    run.add_argument("--config")
    run.add_argument("--backend", choices=("fake", "tensorrt"), default="fake")
    run.add_argument("--fake-script", help="ScriptedBackend JSON (fake backend only)")
    run.add_argument("--whitelist", help="comma-separated task whitelist classes")
    run.add_argument("--imu", help="IMU .txt (JSON lines); default: <input>.txt next to the video")
    run.add_argument("--audio", help="audio .aac; default: <input>.aac, else the video's own audio track")
    run.add_argument("--session-id")
    run.add_argument("--start-ns", type=int)
    run.set_defaults(func=_cmd_run)

    verify = sub.add_parser("verify", help="verify .mcap files load as Foxglove expects")
    verify.add_argument("mcap", nargs="+")
    verify.set_defaults(func=_cmd_verify)

    synth = sub.add_parser("synth", help="write a synthetic test video and fake-model script")
    synth.add_argument("output")
    synth.add_argument("--frames", type=int, default=600)
    synth.add_argument("--width", type=int, default=640)
    synth.add_argument("--height", type=int, default=360)
    synth.add_argument("--fps", type=float, default=30.0)
    synth.add_argument("--seed", type=int, default=0)
    synth.set_defaults(func=_cmd_synth)

    bench = sub.add_parser("bench", help="Tier 1 CPU cost per frame")
    bench.add_argument("--frames", type=int, default=120)
    bench.add_argument("--width", type=int, default=1920)
    bench.add_argument("--height", type=int, default=1080)
    bench.add_argument("--config")
    bench.set_defaults(func=_cmd_bench)

    worker = sub.add_parser("worker", help="run the Pub/Sub worker")
    worker.add_argument("--config")
    worker.set_defaults(func=_cmd_worker)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level, stream=sys.stderr)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
