"""Validated desktop preferences, export history and presentation helpers."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from physical_ai_pipeline.laptop import LaptopOptions

RESOLUTIONS: dict[str, int | None] = {"1280 px wide": 1280, "1920 px wide": 1920, "Original size": None}


def duration_label(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "Unknown duration"
    value = round(seconds)
    hours, remainder = divmod(value, 3600)
    minutes, seconds_int = divmod(remainder, 60)
    return f"{hours}:{minutes:02}:{seconds_int:02}" if hours else f"{minutes}:{seconds_int:02}"


def size_label(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{size} B"
        value /= 1024
    raise AssertionError("unreachable")


def parse_options(resolution: str, blur: str, duplicates: bool, idle: bool, audio: bool, threads: str) -> LaptopOptions:
    if resolution not in RESOLUTIONS:
        raise ValueError("Choose an output size from the list.")
    try:
        threshold = float(blur)
        thread_count = int(threads)
    except ValueError as exc:
        raise ValueError("Blur threshold and CPU threads must be numbers.") from exc
    return LaptopOptions(
        max_width=RESOLUTIONS[resolution],
        blur_threshold=threshold,
        threads=thread_count,
        remove_duplicates=duplicates,
        remove_idle=idle,
        preserve_audio=audio,
    )


@dataclass
class DesktopState:
    output: str
    resolution: str = "1280 px wide"
    blur: str = "100"
    duplicates: bool = False
    idle: bool = False
    audio: bool = True
    threads: str = "2"
    history: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path, default_output: Path) -> DesktopState:
        state = cls(output=str(default_output))
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return state
            for key in ("output", "resolution", "blur", "threads"):
                if isinstance(data.get(key), str) and data[key].strip():
                    setattr(state, key, data[key])
            for key in ("duplicates", "idle", "audio"):
                if isinstance(data.get(key), bool):
                    setattr(state, key, data[key])
            parse_options(state.resolution, state.blur, state.duplicates, state.idle, state.audio, state.threads)
            history = data.get("history", [])
            if isinstance(history, list):
                state.history = [
                    row
                    for row in history
                    if isinstance(row, dict)
                    and isinstance(row.get("folder"), str)
                    and isinstance(row.get("source"), str)
                    and row.get("status") in {"completed", "no_frames_kept"}
                    and all(isinstance(row.get(key), int) and row[key] >= 0 for key in ("kept", "total"))
                    and isinstance(row.get("elapsed"), (int, float))
                    and math.isfinite(row["elapsed"])
                    and row["elapsed"] >= 0
                ][-20:]
        except (OSError, ValueError, TypeError):
            return cls(output=str(default_output))
        return state

    def save(self, path: Path) -> None:
        from dataclasses import asdict

        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        temp.replace(path)

    def add_export(self, folder: Path, report: dict[str, Any]) -> None:
        self.history = [row for row in self.history if row["folder"] != str(folder)]
        self.history.append(
            {
                "folder": str(folder),
                "source": str(report["source"]),
                "status": report["status"],
                "kept": report["kept_frames"],
                "total": report["total_frames"],
                "elapsed": report["elapsed_seconds"],
            }
        )
        self.history = self.history[-20:]
