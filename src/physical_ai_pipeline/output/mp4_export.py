"""Export the kept frames of a session MCAP as an MP4 (H.264).

Reads ``/camera/image_compressed`` back out of the finished ``.mcap`` so the
pipeline itself is untouched. ``timing="continuous"`` (default) plays the kept
frames back to back at the source frame rate, so dropped stretches are cut out.
``timing="original"`` keeps each frame's original presentation timestamp
(variable frame rate): dropped stretches hold the previous frame and the video
stays time-aligned with every MCAP channel.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

from physical_ai_pipeline.output.mcap_writer import TOPIC_IMAGE

_TIME_BASE = 90_000  # ticks per second


def export_mp4(
    mcap_path: str | Path, mp4_path: str | Path, fps: float = 30.0, crf: int = 20, timing: str = "continuous"
) -> int:
    """Write ``mp4_path`` from the kept frames in ``mcap_path``; returns frames written."""
    mp4_path = Path(mp4_path)
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(mp4_path), mode="w")
    stream: av.video.stream.VideoStream | None = None
    first_ns: int | None = None
    last_pts = -1
    count = 0
    try:
        with open(mcap_path, "rb") as fh:
            for _schema, _channel, message, proto in make_reader(
                fh, decoder_factories=_factories()
            ).iter_decoded_messages(topics=[TOPIC_IMAGE]):
                image = cv2.imdecode(np.frombuffer(proto.data, np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError("cannot decode JPEG from MCAP")
                if stream is None:
                    h, w = image.shape[:2]
                    stream = container.add_stream("libx264", rate=round(fps))
                    stream.width, stream.height = w & ~1, h & ~1  # H.264 needs even dimensions
                    stream.pix_fmt = "yuv420p"
                    stream.codec_context.time_base = Fraction(1, _TIME_BASE)
                    stream.options = {"crf": str(crf), "preset": "medium"}
                    first_ns = message.log_time
                assert first_ns is not None
                frame = av.VideoFrame.from_ndarray(
                    np.ascontiguousarray(image[: stream.height, : stream.width]).astype(np.uint8), format="bgr24"
                )
                if timing == "continuous":
                    pts = round(count * _TIME_BASE / fps)
                else:
                    pts = max((message.log_time - first_ns) * _TIME_BASE // 1_000_000_000, last_pts + 1)
                frame.pts, last_pts = pts, pts
                frame.time_base = stream.codec_context.time_base
                for packet in stream.encode(frame):
                    container.mux(packet)
                count += 1
        if stream is not None:
            for packet in stream.encode():
                container.mux(packet)
    finally:
        container.close()
    if count == 0:
        mp4_path.unlink(missing_ok=True)
    return count


def _factories() -> list[DecoderFactory]:
    return [DecoderFactory()]
