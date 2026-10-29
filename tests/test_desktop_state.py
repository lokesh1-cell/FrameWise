from __future__ import annotations

import json
from pathlib import Path

import pytest

from physical_ai_pipeline.desktop_state import DesktopState, duration_label, parse_options, size_label


@pytest.mark.parametrize(
    "value", ["not json", "[]", '{"resolution":"invalid"}', '{"threads":"99"}', '{"blur":"NaN"}', '{"blur":"-1"}']
)
def test_invalid_preferences_fall_back_safely(tmp_path: Path, value: str) -> None:
    path = tmp_path / "settings.json"
    path.write_text(value)
    state = DesktopState.load(path, tmp_path / "output")
    assert state.output == str(tmp_path / "output")
    assert state.blur == "100" and state.threads == "2"


def test_preferences_roundtrip_and_bounded_history(tmp_path: Path) -> None:
    path = tmp_path / "state" / "settings.json"
    state = DesktopState(str(tmp_path / "exports"), blur="60", threads="3", duplicates=True)
    for i in range(25):
        state.add_export(
            tmp_path / str(i),
            {"source": f"{i}.mp4", "status": "completed", "kept_frames": 3, "total_frames": 4, "elapsed_seconds": 1.2},
        )
    state.save(path)
    loaded = DesktopState.load(path, tmp_path / "fallback")
    assert loaded == state
    assert len(loaded.history) == 20 and loaded.history[0]["source"] == "5.mp4"
    assert not path.with_suffix(".tmp").exists()


def test_malformed_history_rows_are_ignored(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "history": [
                    None,
                    {"folder": "/tmp", "source": "x.mp4"},
                    {
                        "folder": "/tmp",
                        "source": "x",
                        "status": "completed",
                        "kept": "broken",
                        "total": 1,
                        "elapsed": 0,
                    },
                ]
            }
        )
    )
    assert DesktopState.load(path, tmp_path).history == []


@pytest.mark.parametrize("blur,threads", [("abc", "2"), ("100", "0"), ("100", "5"), ("inf", "2"), ("-3", "2")])
def test_ui_options_reject_invalid_numbers(blur: str, threads: str) -> None:
    with pytest.raises(ValueError):
        parse_options("Original size", blur, False, False, True, threads)


def test_presentation_values() -> None:
    assert duration_label(65) == "1:05"
    assert duration_label(3661) == "1:01:01"
    assert duration_label(None) == "Unknown duration"
    assert duration_label(float("nan")) == "Unknown duration"
    assert size_label(1024 * 1024) == "1.0 MB"
    assert parse_options("Original size", "0", False, False, False, "1").max_width is None
