"""Native interaction tests; run with xvfb-run on CI or an available display."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from physical_ai_pipeline.synthetic import SceneGenerator, write_video


@pytest.fixture
def app(tmp_path: Path) -> Iterator[Any]:
    tk = pytest.importorskip("tkinter")
    from physical_ai_pipeline.desktop import FrameWiseApp

    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("A native desktop display is not available")
    errors = []
    root.report_callback_exception = lambda *args: errors.append(args)
    instance = FrameWiseApp(root, tmp_path / "settings.json")
    instance.output.set(str(tmp_path / "exports"))
    yield instance
    instance.cancel.set()
    if instance.worker:
        instance.worker.join(timeout=10)
    try:
        root.destroy()
    except tk.TclError:
        pass
    assert not errors, errors


def pump(app: Any, condition: Callable[[], bool], timeout: float = 12) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.root.update()
        if condition():
            return
        time.sleep(0.02)
    pytest.fail(f"Desktop operation timed out: {app.status.get()} / {app.error.get()}")


def load(app: Any, tmp_path: Path, *, dark: bool = False) -> Path:
    import numpy as np

    source = write_video(
        tmp_path / "source.mp4",
        iter([np.zeros((90, 160, 3), np.uint8)] * 3) if dark else SceneGenerator(160, 90).frames(12),
    )
    app.select_source(source)
    pump(app, lambda: app.source_info is not None)
    return source


def test_native_preview_export_and_history(app: Any, tmp_path: Path) -> None:
    source = load(app, tmp_path)
    assert app.preview_image is not None
    assert app.source == source
    app.start()
    pump(app, lambda: app.last_output is not None and not app.running)
    report = json.loads((app.last_output / "report.json").read_text())
    assert report["kept_frames"] > 0
    assert app.status.get() == "Export complete"
    assert len(app.history.get_children()) == 1
    assert "disabled" not in app.play_button.state()
    # The action area must remain visible in the smallest supported window.
    app.root.geometry("980x740")
    app.root.update()
    assert (
        app.folder_button.winfo_rooty() + app.folder_button.winfo_height()
        <= app.root.winfo_rooty() + app.root.winfo_height()
    )


def test_native_inline_validation_prevents_job(app: Any, tmp_path: Path) -> None:
    load(app, tmp_path)
    app.blur.set("not a number")
    app.start()
    assert app.error.get() and not app.running
    app.blur.set("100")
    output_file = tmp_path / "not-a-folder"
    output_file.write_text("preserve me")
    app.output.set(str(output_file))
    app.start()
    assert "file" in app.error.get() and output_file.read_text() == "preserve me"


def test_native_no_frames_kept_has_report_but_no_play_action(app: Any, tmp_path: Path) -> None:
    load(app, tmp_path, dark=True)
    app.start()
    pump(app, lambda: app.last_output is not None and not app.running)
    assert app.status.get() == "No frames passed the filters"
    assert "disabled" in app.play_button.state()
    assert "disabled" not in app.report_button.state()


def test_native_ignores_stale_previews(app: Any) -> None:
    app.preview_token = 2
    app.loading = True
    app._handle("load_error", (1, "stale error"))
    assert app.loading and not app.error.get()


def test_native_export_error_is_recoverable(app: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    load(app, tmp_path)

    def fail(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("Disk is full")

    monkeypatch.setattr("physical_ai_pipeline.desktop.clean_video", fail)
    app.start()
    pump(app, lambda: app.status.get() == "Export failed")
    assert "Disk is full" in app.error.get()
    assert "disabled" not in app.start_button.state()


def test_native_cancellation_and_close_wait_for_worker(
    app: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from physical_ai_pipeline.laptop import CleaningCancelled

    load(app, tmp_path)

    def wait_for_cancel(*_args: Any, **kwargs: Any) -> None:
        assert kwargs["cancel"].wait(5)
        raise CleaningCancelled()

    monkeypatch.setattr("physical_ai_pipeline.desktop.clean_video", wait_for_cancel)
    app.start()
    app.stop()
    pump(app, lambda: app.status.get() == "Export cancelled")
    assert not app.running
    app.start()
    app.close()
    assert app.cancel.is_set()
    app.worker.join(timeout=5)
    assert not app.running
