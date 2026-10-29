"""Bounded-memory CPU quality cleaning, independent of models and protobufs."""

from __future__ import annotations

import json
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from tempfile import mkdtemp
from typing import Any

import av
import cv2

from physical_ai_pipeline.config import BlurConfig, DuplicateConfig, IdleConfig, Tier1Config
from physical_ai_pipeline.decode.video_reader import VideoReader
from physical_ai_pipeline.filters.tier1 import Tier1Stage
from physical_ai_pipeline.frame import FrameRecord
from physical_ai_pipeline.timeline import FrameWindow


class CleaningCancelled(RuntimeError):
    """The user stopped a local run."""


@dataclass(frozen=True)
class LaptopOptions:
    max_width: int | None = 1280
    threads: int = 2
    blur_threshold: float = 100.0
    remove_duplicates: bool = False
    remove_idle: bool = False
    preserve_audio: bool = True

    def __post_init__(self) -> None:
        if self.max_width is not None and self.max_width < 2:
            raise ValueError("Maximum width must be at least 2")
        if not 1 <= self.threads <= 4:
            raise ValueError("Use between 1 and 4 threads")
        if not 0 <= self.blur_threshold <= 10000:
            raise ValueError("Blur threshold must be between 0 and 10000")


def clean_video(
    source: str | Path,
    output_dir: str | Path,
    options: LaptopOptions | None = None,
    *,
    progress: Callable[[dict[str, Any]], None] | None = None,
    cancel: threading.Event | None = None,
) -> Path:
    """Create a unique run folder. Failed/cancelled runs retain a status report only.

    Writes CFR H.264, a streamed frame audit, and report.json. CPU mode does
    not perform semantic inference or claim the full pipeline's compliance checks.
    """
    options = options or LaptopOptions()
    source = Path(source).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not source.is_file():
        raise ValueError("Choose an existing input video")
    reader = VideoReader(source, max_width=options.max_width, threads=options.threads)
    if min(reader.info.width, reader.info.height) < 2:
        raise ValueError("Video dimensions must be at least 2 × 2")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(mkdtemp(prefix=f"{source.stem[:60]}_{datetime.now():%Y%m%d_%H%M%S}_", dir=output_dir))
    part = run_dir / "cleaned.partial.mp4"
    audit_path = run_dir / "frames.jsonl"
    cfg = Tier1Config(
        metric_workers=1,
        blur=BlurConfig(enabled=options.blur_threshold > 0, min_laplacian_variance=options.blur_threshold),
        duplicate=DuplicateConfig(enabled=options.remove_duplicates),
        idle=IdleConfig(enabled=options.remove_idle, flow_width=240),
    )
    stage, window = Tier1Stage(cfg), FrameWindow()
    lag = cfg.idle.min_consecutive_frames if options.remove_idle else 0
    total, kept = 0, 0
    reasons: Counter[str] = Counter()
    started = time.monotonic()
    last_update = 0.0
    report: dict[str, Any] = {
        "schema_version": 1,
        "application": "FrameWise",
        "started_at": datetime.now(UTC).isoformat(),
        "mode": "cpu_quality_only",
        "source": str(source),
        "options": asdict(options),
        "semantic_checks_performed": False,
        "audio_preserved": False,
        "source_width": reader.info.width,
        "source_height": reader.info.height,
        "fps": reader.info.fps,
        "output_folder": str(run_dir),
    }

    def check_cancel() -> None:
        if cancel is not None and cancel.is_set():
            raise CleaningCancelled("Processing cancelled")

    # OpenCV otherwise starts its own large pool inside each metric operation.
    previous_threads = cv2.getNumThreads()
    cv2.setNumThreads(options.threads)
    try:
        with av.open(str(part), "w") as container, audit_path.open("w", encoding="utf-8") as audit:
            stream = container.add_stream("libx264", rate=Fraction(reader.info.fps).limit_denominator(100000))
            width = min(reader.info.width, options.max_width or reader.info.width) & ~1
            height = max(2, round(reader.info.height * width / reader.info.width) & ~1)
            stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
            stream.codec_context.thread_count = options.threads
            stream.options = {"crf": "23", "preset": "veryfast"}
            report.update(output_width=width, output_height=height)

            def emit(record: FrameRecord) -> None:
                nonlocal kept
                check_cancel()
                dropped = record.dropped or record.image is None
                why = sorted({v.reason.value for v in record.verdicts}) if dropped else []
                reasons.update(why)
                audit.write(
                    json.dumps({"frame": record.index, "pts_ns": record.pts_ns, "kept": not dropped, "reasons": why})
                    + "\n"
                )
                if not dropped:
                    assert record.image is not None
                    frame = av.VideoFrame.from_ndarray(record.image, format="bgr24")
                    frame = frame.reformat(width=width, height=height, format="yuv420p")
                    frame.pts = kept
                    frame.time_base = Fraction(1, 1) / stream.average_rate
                    for packet in stream.encode(frame):
                        container.mux(packet)
                    kept += 1

            frames = reader.frames()
            try:
                for record in frames:
                    check_cancel()
                    total += 1
                    window.add(record)
                    stage.process(record, window)
                    for ready in window.pop_until(record.index - lag):
                        emit(ready)
                    now = time.monotonic()
                    if progress and now - last_update >= 0.2:
                        progress(
                            {
                                "phase": "Cleaning video",
                                "processed": total,
                                "kept": kept,
                                "total": reader.info.frame_count,
                                "elapsed": now - started,
                            }
                        )
                        last_update = now
                for ready in window.pop_until(total):
                    emit(ready)
                for packet in stream.encode():
                    container.mux(packet)
            finally:
                frames.close()
        check_cancel()
        if not total:
            raise ValueError("The input contains no decodable video frames")
        if kept:
            if options.preserve_audio:
                if progress:
                    progress(
                        {
                            "phase": "Saving audio",
                            "processed": 0,
                            "kept": kept,
                            "total": kept,
                            "elapsed": time.monotonic() - started,
                        }
                    )
                report["audio_preserved"] = _add_audio(
                    source,
                    part,
                    audit_path,
                    reader.info.fps,
                    options,
                    check_cancel,
                    progress=progress,
                    started=started,
                    kept=kept,
                )
            check_cancel()
            part.replace(run_dir / "cleaned.mp4")
        else:
            part.unlink(missing_ok=True)
        report["status"] = "completed" if kept else "no_frames_kept"
    except BaseException as exc:
        report.update(status="cancelled" if isinstance(exc, CleaningCancelled) else "failed", error=str(exc))
        part.unlink(missing_ok=True)
        (run_dir / "audio.partial.mp4").unlink(missing_ok=True)
        raise
    finally:
        cv2.setNumThreads(previous_threads)
        report.update(
            total_frames=total,
            kept_frames=kept,
            dropped_frames=total - kept,
            dropped_by_reason=dict(reasons),
            decode_errors=reader.decode_errors,
            elapsed_seconds=round(time.monotonic() - started, 3),
        )
        report["finished_at"] = datetime.now(UTC).isoformat()
        report_part = run_dir / "report.partial.json"
        report_part.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report_part.replace(run_dir / "report.json")
    return run_dir


def _add_audio(
    source: Path,
    video: Path,
    audit_path: Path,
    fps: float,
    options: LaptopOptions,
    check_cancel: Callable[[], None],
    *,
    progress: Callable[[dict[str, Any]], None] | None = None,
    started: float = 0,
    kept: int = 0,
) -> bool:
    """Cut the first audio track to kept frames, streaming PCM and audit from disk.

    Output audio is mono 48 kHz AAC. Silence pads missing samples. Each kept
    video's frame interval contributes exactly one CFR frame of audio.
    """
    import numpy as np

    rate = 48000
    tmp = video.parent / "audio.partial.mp4"
    with av.open(str(source)) as src:
        if not src.streams.audio:
            return False
        audio = src.streams.audio[0]
        audio.codec_context.thread_count = options.threads
        vs = src.streams.video[0]
        origin = float((vs.start_time or 0) * vs.time_base)
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=rate)

        def chunks() -> Iterator[tuple[int, Any]]:
            next_start = 0
            for decoded in src.decode(audio=0):
                check_cancel()
                for frame in resampler.resample(decoded):
                    start = (
                        round((float(frame.pts * frame.time_base) - origin) * rate)
                        if frame.pts is not None
                        else next_start
                    )
                    samples = frame.to_ndarray().reshape(-1)
                    next_start = start + len(samples)
                    yield start, samples
            for frame in resampler.resample(None):
                samples = frame.to_ndarray().reshape(-1)
                yield next_start, samples
                next_start += len(samples)

        decoded_chunks = iter(chunks())
        chunk = next(decoded_chunks, None)
        with av.open(str(video)) as vin, av.open(str(tmp), "w") as dst, audit_path.open() as audit:
            vout = dst.add_stream_from_template(vin.streams.video[0])
            aout = dst.add_stream("aac", rate=rate, layout="mono")
            aout.codec_context.thread_count = options.threads
            for packet in vin.demux(video=0):
                check_cancel()
                if packet.dts is not None:
                    packet.stream = vout
                    dst.mux(packet)
            position, frame_count = 0, 0
            last_update = time.monotonic()
            for line in audit:
                check_cancel()
                row = json.loads(line)
                if not row["kept"]:
                    continue
                start = round(row["pts_ns"] * rate / 1e9)
                frame_count += 1
                size = round(frame_count * rate / fps) - position
                end = start + size
                samples = np.zeros((1, size), dtype=np.float32)
                while chunk is not None:
                    offset, data = chunk
                    if offset >= end:
                        break
                    left, right = max(start, offset), min(end, offset + len(data))
                    if right > left:
                        samples[0, left - start : right - start] = data[left - offset : right - offset]
                    if offset + len(data) >= end:
                        break
                    chunk = next(decoded_chunks, None)
                frame = av.AudioFrame.from_ndarray(samples, format="fltp", layout="mono")
                frame.sample_rate, frame.pts, frame.time_base = rate, position, Fraction(1, rate)
                position += size
                for packet in aout.encode(frame):
                    dst.mux(packet)
                now = time.monotonic()
                if progress and now - last_update > 0.2:
                    progress(
                        {
                            "phase": "Saving audio",
                            "processed": frame_count,
                            "total": kept,
                            "kept": kept,
                            "elapsed": now - started,
                        }
                    )
                    last_update = now
            for packet in aout.encode():
                dst.mux(packet)
    tmp.replace(video)
    return True
