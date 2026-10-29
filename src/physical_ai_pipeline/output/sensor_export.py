"""Cut IMU samples and audio to match the frames kept in a session MCAP.

Kept frames are read back from ``/camera/image_compressed``; consecutive kept
frames merge into source-time segments. Anything outside those segments
(dropped frames, child-exclusion blocks included) is removed from the IMU log
and the audio, so every output stream lines up with the output video.

``timing="continuous"`` concatenates the segments (matches the continuous MP4).
``timing="original"`` keeps source timestamps; dropped audio becomes silence.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import av
import numpy as np
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

from physical_ai_pipeline.output.mcap_writer import TOPIC_IMAGE, TOPIC_METADATA

Segment = tuple[int, int]  # [start_ns, end_ns) in source time (pts from video start)
AUDIO_RATE = 16_000


def kept_segments(mcap_path: str | Path, fps: float) -> list[Segment]:
    """Merged source-time intervals covered by kept frames."""
    frame_ns = int(1e9 / fps)
    base: int | None = None
    stamps: list[int] = []
    with open(mcap_path, "rb") as fh:
        reader = make_reader(fh, decoder_factories=[DecoderFactory()])
        for _schema, channel, message, _proto in reader.iter_decoded_messages(topics=[TOPIC_METADATA, TOPIC_IMAGE]):
            if channel.topic == TOPIC_METADATA:
                base = message.log_time
            else:
                stamps.append(message.log_time)
    if base is None:
        raise ValueError("MCAP has no session metadata")
    segments: list[Segment] = []
    for ns in sorted(stamps):
        start = ns - base
        if segments and start - segments[-1][1] <= frame_ns // 2:
            segments[-1] = (segments[-1][0], start + frame_ns)
        else:
            segments.append((start, start + frame_ns))
    return segments


def output_offsets(segments: list[Segment], timing: str) -> list[int]:
    """Output-time start of each segment."""
    if timing == "original":
        return [s for s, _ in segments]
    offsets, acc = [], 0
    for s, e in segments:
        offsets.append(acc)
        acc += e - s
    return offsets


def export_imu(
    imu_path: str | Path,
    out_path: str | Path,
    segments: list[Segment],
    timing: str = "continuous",
    offset_s: float = 0.0,
) -> int:
    """Write the IMU samples that fall inside kept segments as JSON lines.

    Sample time is taken relative to the first sample (``t_us`` is a device
    clock) plus ``offset_s``. Each output row keeps its original fields and
    gains ``t_out_us`` (position in the output video).
    """
    offsets = output_offsets(segments, timing)
    rows = [json.loads(line) for line in Path(imu_path).read_text().splitlines() if line.strip()]
    if not rows:
        Path(out_path).write_text("")
        return 0
    t0 = rows[0]["t_us"]
    written, i = 0, 0
    with open(out_path, "w") as out:
        for row in rows:
            ns = int((row["t_us"] - t0) * 1000 + offset_s * 1e9)
            while i < len(segments) and ns >= segments[i][1]:
                i += 1
            if i == len(segments):
                break
            if ns < segments[i][0]:
                continue
            row["t_out_us"] = (offsets[i] + ns - segments[i][0]) // 1000
            out.write(json.dumps(row, separators=(",", ":")) + "\n")
            written += 1
    return written


def read_pcm(path: str | Path, rate: int = AUDIO_RATE) -> np.ndarray:
    """Decode the first audio stream of ``path`` to mono int16 PCM."""
    resampler = av.AudioResampler(format="s16", layout="mono", rate=rate)
    chunks: list[np.ndarray] = []
    with av.open(str(path)) as container:
        if not container.streams.audio:
            return np.zeros(0, dtype=np.int16)
        for frame in container.decode(audio=0):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().reshape(-1))
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)


def cut_pcm(pcm: np.ndarray, segments: list[Segment], timing: str = "continuous", rate: int = AUDIO_RATE) -> np.ndarray:
    def idx(ns: int) -> int:
        return min(max(round(ns * rate / 1e9), 0), len(pcm))

    if timing == "original":
        out = np.zeros_like(pcm)
        for s, e in segments:
            out[idx(s) : idx(e)] = pcm[idx(s) : idx(e)]
        return out
    parts = [pcm[idx(s) : idx(e)] for s, e in segments]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int16)


def _encode(pcm: np.ndarray, stream: av.audio.stream.AudioStream, container: av.container.OutputContainer) -> None:
    rate = stream.codec_context.sample_rate
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=rate)
    step, pos = 1024, 0
    for start in range(0, len(pcm), step):
        chunk = np.ascontiguousarray(pcm[start : start + step]).reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(chunk, format="s16", layout="mono")
        frame.sample_rate = rate
        frame.pts = pos
        pos += chunk.shape[1]
        for out in resampler.resample(frame):
            for packet in stream.encode(out):
                container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)


def write_aac(pcm: np.ndarray, out_path: str | Path, rate: int = AUDIO_RATE) -> None:
    with av.open(str(out_path), mode="w", format="adts") as container:
        stream = container.add_stream("aac", rate=rate, layout="mono")
        _encode(pcm, stream, container)


def add_audio_to_mp4(mp4_path: str | Path, pcm: np.ndarray, rate: int = AUDIO_RATE) -> None:
    """Re-mux ``mp4_path`` (video stream copied untouched) with an AAC track."""
    mp4_path = Path(mp4_path)
    tmp = mp4_path.with_suffix(".tmp.mp4")
    with av.open(str(mp4_path)) as src, av.open(str(tmp), mode="w") as dst:
        vin = src.streams.video[0]
        vout = dst.add_stream_from_template(vin)
        aout = dst.add_stream("aac", rate=rate, layout="mono")
        for packet in src.demux(vin):
            if packet.dts is None:
                continue
            packet.stream = vout
            dst.mux(packet)
        _encode(pcm, aout, dst)
    shutil.move(str(tmp), mp4_path)


def export_sensors(
    mcap_path: str | Path,
    out_dir: str | Path,
    fps: float,
    timing: str = "continuous",
    imu_path: str | Path | None = None,
    audio_path: str | Path | None = None,
    imu_offset_s: float = 0.0,
    mp4_path: str | Path | None = None,
) -> dict[str, int]:
    """Cut the IMU log and audio to the kept frames; returns counts written."""
    out_dir = Path(out_dir)
    segments = kept_segments(mcap_path, fps)
    result: dict[str, int] = {"segments": len(segments)}
    if imu_path is not None:
        result["imu_samples"] = export_imu(imu_path, out_dir / "session.imu.jsonl", segments, timing, imu_offset_s)
    if audio_path is not None:
        source_pcm = read_pcm(audio_path)
        if len(source_pcm) == 0:  # video without an audio track
            return result
        pcm = cut_pcm(source_pcm, segments, timing)
        write_aac(pcm, out_dir / "session.aac")
        result["audio_samples"] = len(pcm)
        if mp4_path is not None and Path(mp4_path).exists():
            add_audio_to_mp4(mp4_path, pcm)
    return result
