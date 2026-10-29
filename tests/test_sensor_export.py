import json
from pathlib import Path

import numpy as np

from physical_ai_pipeline.output.sensor_export import cut_pcm, export_imu, output_offsets

S = 1_000_000_000


def _imu(path: Path, n: int = 100, hz: int = 10) -> Path:
    rows = [{"t_us": 7_000_000 + i * 1_000_000 // hz, "acc": [0, 0, i], "gyro": [0, 0, 0]} for i in range(n)]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def test_imu_continuous_cuts_dropped_and_retimes(tmp_path: Path) -> None:
    segments = [(0, 2 * S), (5 * S, 6 * S)]  # keep 0-2 s and 5-6 s of a 10 s clip (10 Hz)
    out = tmp_path / "imu.jsonl"
    assert export_imu(_imu(tmp_path / "imu.txt"), out, segments) == 30
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["acc"][2] for r in rows[:2]] == [0, 1]
    assert rows[19]["acc"][2] == 19 and rows[20]["acc"][2] == 50  # 2-5 s removed
    assert rows[20]["t_out_us"] == 2_000_000  # second segment starts right after the first
    assert all(b["t_out_us"] > a["t_out_us"] for a, b in zip(rows, rows[1:], strict=False))


def test_imu_original_keeps_source_time(tmp_path: Path) -> None:
    segments = [(0, S), (5 * S, 6 * S)]
    out = tmp_path / "imu.jsonl"
    export_imu(_imu(tmp_path / "imu.txt"), out, segments, timing="original")
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows[10]["t_out_us"] == 5_000_000


def test_imu_offset_shifts_alignment(tmp_path: Path) -> None:
    out = tmp_path / "imu.jsonl"
    n = export_imu(_imu(tmp_path / "imu.txt"), out, [(0, S)], offset_s=0.5)
    assert n == 5  # samples now at 0.5..0.9 s fall inside the first second


def test_pcm_continuous_and_original() -> None:
    rate = 16_000
    pcm = np.arange(10 * rate, dtype=np.int16)
    segs = [(0, S), (5 * S, 6 * S)]
    cont = cut_pcm(pcm, segs, "continuous", rate)
    assert len(cont) == 2 * rate and cont[rate] == pcm[5 * rate]
    orig = cut_pcm(pcm, segs, "original", rate)
    assert len(orig) == len(pcm) and orig[2 * rate] == 0 and orig[5 * rate] == pcm[5 * rate]


def test_output_offsets() -> None:
    assert output_offsets([(0, S), (5 * S, 6 * S)], "continuous") == [0, S]
    assert output_offsets([(0, S), (5 * S, 6 * S)], "original") == [0, 5 * S]
