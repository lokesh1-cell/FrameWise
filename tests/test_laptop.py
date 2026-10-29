from __future__ import annotations

import json
import threading
from pathlib import Path

import av
import numpy as np
import pytest

from physical_ai_pipeline.decode.video_reader import VideoReader, prefetch
from physical_ai_pipeline.laptop import CleaningCancelled, LaptopOptions, clean_video
from physical_ai_pipeline.synthetic import SceneGenerator, dark, write_video


def test_clean_mp4_matches_audit_and_never_overwrites_input(tmp_path: Path) -> None:
    scene = SceneGenerator(320, 180)
    source = write_video(tmp_path / "input.mp4", iter([scene.frame(0), dark(scene.frame(1)), scene.frame(2)]), fps=30)
    original = source.read_bytes()
    folder = clean_video(source, tmp_path / "results", LaptopOptions(max_width=160))
    report = json.loads((folder / "report.json").read_text())
    rows = [json.loads(line) for line in (folder / "frames.jsonl").read_text().splitlines()]
    with av.open(str(folder / "cleaned.mp4")) as video:
        frames = list(video.decode(video=0))
    assert len(frames) == report["kept_frames"] == sum(row["kept"] for row in rows) == 2
    assert frames[0].width == 160 and frames[0].height == 90
    assert report["dropped_by_reason"] == {"UNDEREXPOSED": 1}
    assert report["semantic_checks_performed"] is False
    assert source.read_bytes() == original
    second = clean_video(source, tmp_path / "results", LaptopOptions(max_width=160))
    assert second != folder and (folder / "cleaned.mp4").is_file()


def test_all_filtered_produces_report_without_invalid_mp4(tmp_path: Path) -> None:
    source = write_video(tmp_path / "dark.mp4", iter([np.zeros((90, 160, 3), np.uint8)] * 3))
    folder = clean_video(source, tmp_path / "results")
    assert json.loads((folder / "report.json").read_text())["status"] == "no_frames_kept"
    assert not list(folder.glob("*.mp4"))


def test_cancellation_cleans_partial_video(tmp_path: Path) -> None:
    scene = SceneGenerator(160, 90)
    source = write_video(tmp_path / "input.mp4", scene.frames(60))
    event = threading.Event()

    def stop(_progress):
        event.set()

    with pytest.raises(CleaningCancelled):
        clean_video(source, tmp_path / "results", progress=stop, cancel=event)
    folder = next((tmp_path / "results").iterdir())
    assert json.loads((folder / "report.json").read_text())["status"] == "cancelled"
    assert not list(folder.glob("*.mp4"))


def test_decoder_caps_resolution(tmp_path: Path) -> None:
    source = write_video(tmp_path / "input.mp4", SceneGenerator(320, 180).frames(2))
    assert next(VideoReader(source, max_width=160, threads=2).frames()).image.shape == (90, 160, 3)


def test_prefetch_stops_and_closes_source_when_consumer_exits() -> None:
    closed = threading.Event()

    def source():
        try:
            yield from range(10000)
        finally:
            closed.set()

    stream = prefetch(source(), maxsize=1)
    next(stream)
    stream.close()
    assert closed.wait(2)


def test_disabled_idle_does_not_compute_optical_flow(monkeypatch) -> None:
    from physical_ai_pipeline.config import Tier1Config
    from physical_ai_pipeline.filters.tier1 import metrics

    def unexpected(*_args):
        pytest.fail("Optical flow should be skipped when idle filtering is disabled")

    monkeypatch.setattr(metrics, "mean_flow_magnitude", unexpected)
    image = SceneGenerator(160, 90).frame(0)
    assert metrics.compute_metrics(image, image, Tier1Config(idle={"enabled": False})).flow_magnitude == -1


def test_audio_is_cut_to_output_duration(tmp_path: Path) -> None:
    from fractions import Fraction

    source = tmp_path / "audio.mp4"
    scene = SceneGenerator(160, 90)
    with av.open(str(source), "w") as container:
        video = container.add_stream("libx264", rate=30)
        video.width, video.height, video.pix_fmt = 160, 90, "yuv420p"
        audio = container.add_stream("aac", rate=48000, layout="mono")
        for i in range(30):
            image = scene.frame(i) if i < 10 or i >= 20 else dark(scene.frame(i))
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            frame.pts, frame.time_base = i, Fraction(1, 30)
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)
        times = np.arange(48000) / 48000
        frequency = np.where(times < 1 / 3, 440, np.where(times < 2 / 3, 880, 1320))
        samples = np.sin(np.cumsum(frequency) * 2 * np.pi / 48000).astype(np.float32).reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(samples, format="fltp", layout="mono")
        frame.sample_rate, frame.pts = 48000, 0
        for packet in audio.encode(frame):
            container.mux(packet)
        for packet in audio.encode():
            container.mux(packet)
    folder = clean_video(source, tmp_path / "results", LaptopOptions(blur_threshold=0))
    report = json.loads((folder / "report.json").read_text())
    assert report["audio_preserved"] and report["kept_frames"] == 20
    with av.open(str(folder / "cleaned.mp4")) as output:
        track = output.streams.audio[0]
        assert abs(float(track.duration * track.time_base) - 20 / 30) < 0.05
        pcm = np.concatenate([frame.to_ndarray().reshape(-1) for frame in output.decode(audio=0)])

        def dominant(start: float) -> float:
            chunk = pcm[round(start * 48000) : round((start + 0.1) * 48000)]
            return float(np.fft.rfftfreq(len(chunk), 1 / 48000)[np.argmax(abs(np.fft.rfft(chunk)))])

        assert abs(dominant(0.1) - 440) < 15
        assert abs(dominant(0.5) - 1320) < 15


def test_no_video_stream_has_clear_error(tmp_path: Path) -> None:
    import wave

    source = tmp_path / "audio.wav"
    with wave.open(str(source), "wb") as audio:
        audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        audio.writeframes(bytes(3200))
    with pytest.raises(ValueError, match="no video stream"):
        clean_video(source, tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("duplicates,idle,expected", [(True, False, 1), (False, True, 1), (False, False, 45)])
def test_static_and_duplicate_filters_have_expected_boundaries(
    tmp_path: Path, duplicates: bool, idle: bool, expected: int
) -> None:
    image = SceneGenerator(160, 90).frame(0)
    source = write_video(tmp_path / "static.mp4", iter([image] * 45))
    folder = clean_video(
        source, tmp_path / "out", LaptopOptions(remove_duplicates=duplicates, remove_idle=idle, blur_threshold=0)
    )
    report = json.loads((folder / "report.json").read_text())
    assert report["kept_frames"] == expected
    assert report["total_frames"] == 45


def test_encoder_error_restores_opencv_threads_and_removes_partial_output(tmp_path: Path, monkeypatch) -> None:
    import cv2

    import physical_ai_pipeline.laptop as laptop

    source = write_video(tmp_path / "input.mp4", SceneGenerator(160, 90).frames(3))
    original = laptop.av.open
    previous = cv2.getNumThreads()

    def fail_output(path, *args, **kwargs):
        if str(path).endswith("cleaned.partial.mp4"):
            raise OSError("Simulated output write failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(laptop.av, "open", fail_output)
    with pytest.raises(OSError, match="write failure"):
        clean_video(source, tmp_path / "out")
    folder = next((tmp_path / "out").iterdir())
    assert json.loads((folder / "report.json").read_text())["status"] == "failed"
    assert not list(folder.glob("*.mp4")) and cv2.getNumThreads() == previous


def test_cancel_during_audio_removes_both_partial_files(tmp_path: Path, monkeypatch) -> None:
    source = write_video(tmp_path / "input.mp4", SceneGenerator(160, 90).frames(3))

    def cancel_audio(_source, video, *_args, **_kwargs):
        (video.parent / "audio.partial.mp4").write_bytes(b"partial")
        raise CleaningCancelled("Cancelled during audio")

    monkeypatch.setattr("physical_ai_pipeline.laptop._add_audio", cancel_audio)
    with pytest.raises(CleaningCancelled):
        clean_video(source, tmp_path / "out")
    folder = next((tmp_path / "out").iterdir())
    assert json.loads((folder / "report.json").read_text())["status"] == "cancelled"
    assert not list(folder.glob("*.mp4"))


def test_small_video_is_not_upscaled_and_fractional_rate_is_retained(tmp_path: Path) -> None:
    source = write_video(tmp_path / "fractional.mp4", SceneGenerator(160, 90).frames(8), fps=29.97)
    folder = clean_video(source, tmp_path / "out", LaptopOptions(blur_threshold=0))
    with av.open(str(folder / "cleaned.mp4")) as video:
        stream = video.streams.video[0]
        assert (stream.width, stream.height) == (160, 90)
        assert abs(float(stream.average_rate) - 29.97) < 0.02
