"""Streaming PyAV decoder producing contiguous FrameRecords.

Frames are indexed sequentially in decode order (0, 1, 2, ...). Timestamps
come from each frame's PTS. A packet that raises during decode yields a
placeholder record with ``image=None`` so the Tier 1 decode check can log it.
Decoding runs on a background thread with a bounded queue so it overlaps
filtering and inference.
"""

from __future__ import annotations

import math
import queue
import threading
from collections.abc import Generator, Iterable, Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import TypeVar, cast

import numpy as np

from physical_ai_pipeline.frame import FrameRecord, ImageBGR
from physical_ai_pipeline.logging_utils import get_logger, log_event

_log = get_logger(__name__)
T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int | None
    duration_s: float | None
    codec: str


def probe(path: str | Path, fallback_fps: float = 30.0) -> VideoInfo:
    import av

    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError("The selected file has no video stream. Choose a video file.")
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.guessed_rate
        fps = float(rate) if rate else fallback_fps
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("The video frame rate is invalid.")
        duration = float(stream.duration * stream.time_base) if stream.duration and stream.time_base else None
        if duration is None and container.duration:
            duration = container.duration / 1_000_000
        frames = int(stream.frames) if stream.frames else (round(duration * fps) if duration else None)
        return VideoInfo(
            stream.codec_context.width, stream.codec_context.height, fps, frames, duration, stream.codec_context.name
        )


class VideoReader:
    def __init__(
        self, path: str | Path, fallback_fps: float = 30.0, *, max_width: int | None = None, threads: int = 0
    ) -> None:
        self.path = Path(path)
        self.info = probe(self.path, fallback_fps)
        self.decode_errors = 0
        if max_width is not None and max_width < 2:
            raise ValueError("max_width must be at least 2")
        if threads < 0:
            raise ValueError("threads must be non-negative")
        self.max_width = max_width
        self.threads = threads

    def frames(self) -> Generator[FrameRecord, None, None]:
        import av

        index = 0
        frame_ns = int(1e9 / self.info.fps)
        last_pts_ns = -frame_ns
        start_pts: int | None = None
        with av.open(str(self.path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            stream.codec_context.thread_count = self.threads
            tb = stream.time_base
            time_base = Fraction(tb.numerator, tb.denominator) if tb else Fraction(1, 90000)
            for packet in container.demux(stream):
                try:
                    decoded = packet.decode()
                except av.error.FFmpegError as exc:
                    self.decode_errors += 1
                    last_pts_ns += frame_ns
                    log_event(_log, "decode_error", level=30, frame_index=index, error=str(exc))
                    yield FrameRecord(index=index, pts_ns=last_pts_ns, image=None)
                    index += 1
                    continue
                for frame in decoded:
                    if frame.pts is not None:
                        if start_pts is None:
                            start_pts = frame.pts
                        pts_ns = int((frame.pts - start_pts) * time_base * 1_000_000_000)
                        pts_ns = max(pts_ns, last_pts_ns + 1)
                    else:
                        pts_ns = last_pts_ns + frame_ns
                    last_pts_ns = pts_ns
                    if self.max_width and frame.width > self.max_width:
                        width = self.max_width & ~1
                        height = max(2, round(frame.height * width / frame.width) & ~1)
                        frame = frame.reformat(width=width, height=height)
                    image = cast(ImageBGR, frame.to_ndarray(format="bgr24"))
                    yield FrameRecord(index=index, pts_ns=pts_ns, image=image)
                    index += 1


def iter_arrays(images: Iterable[np.ndarray | None], fps: float = 30.0) -> Iterator[FrameRecord]:
    """FrameRecords from in-memory frames (tests, synthetic data)."""
    step = 1e9 / fps
    for index, image in enumerate(images):
        yield FrameRecord(index=index, pts_ns=round(index * step), image=image)


_SENTINEL = object()


def prefetch(source: Iterator[T], maxsize: int = 64) -> Iterator[T]:
    """Run ``source`` on a daemon thread, buffering up to ``maxsize`` items."""
    buffer: queue.Queue[object] = queue.Queue(maxsize=maxsize)
    errors: list[BaseException] = []
    stop = threading.Event()

    def worker() -> None:
        try:
            for item in source:
                while not stop.is_set():
                    try:
                        buffer.put(item, timeout=0.5)
                        break
                    except queue.Full:
                        continue
                if stop.is_set():
                    return
        except BaseException as exc:  # noqa: BLE001 - re-raised in consumer
            errors.append(exc)
        finally:
            while not stop.is_set():
                try:
                    buffer.put(_SENTINEL, timeout=0.1)
                    break
                except queue.Full:
                    continue
            close = getattr(source, "close", None)
            if close is not None:
                close()

    thread = threading.Thread(target=worker, name="decode-prefetch", daemon=True)
    thread.start()
    try:
        while True:
            item = buffer.get()
            if item is _SENTINEL:
                break
            yield item  # type: ignore[misc]
        if errors:
            raise errors[0]
    finally:
        stop.set()
        thread.join(timeout=1)
