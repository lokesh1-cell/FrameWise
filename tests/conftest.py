from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if not (_ROOT / "src/physical_ai_pipeline/proto_gen/physical_ai_pb2.py").is_file():
    # Fresh clone: generated protobuf modules are not committed.
    subprocess.run(["bash", str(_ROOT / "scripts/compile_proto.sh")], check=True)

import numpy as np
import pytest

from physical_ai_pipeline.config import PipelineConfig
from physical_ai_pipeline.synthetic import SceneGenerator


@pytest.fixture(scope="session")
def scene() -> SceneGenerator:
    return SceneGenerator(320, 180, seed=1)


@pytest.fixture()
def config() -> PipelineConfig:
    return PipelineConfig()


@pytest.fixture(autouse=True)
def _no_pap_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    import os

    for key in list(os.environ):
        if key.startswith("PAP_"):
            monkeypatch.delenv(key)
    yield


def frames(scene: SceneGenerator, count: int) -> list[np.ndarray]:
    return [scene.frame(i) for i in range(count)]


from dataclasses import dataclass  # noqa: E402
from typing import Any  # noqa: E402

from physical_ai_pipeline.audit import MemoryAuditSink  # noqa: E402
from physical_ai_pipeline.decode.video_reader import iter_arrays  # noqa: E402
from physical_ai_pipeline.inference.fake_backend import ScriptedBackend  # noqa: E402
from physical_ai_pipeline.pipeline import SessionInfo, SessionPipeline, SessionResult  # noqa: E402


@dataclass
class SessionRun:
    result: SessionResult
    kept: list[int]
    backend: ScriptedBackend
    audit: MemoryAuditSink
    mcap: Path


def read_kept_indices(path: Path) -> list[int]:
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory

    with open(path, "rb") as fh:
        reader = make_reader(fh, decoder_factories=[DecoderFactory()])
        return [int(p.frame_index) for _, _, _, p in reader.iter_decoded_messages(topics=["/detections"])]


def run_session(
    tmp_path: Path,
    images: list[Any],
    segments: list[dict[str, Any]] | None = None,
    config: PipelineConfig | None = None,
    whitelist: tuple[str, ...] | None = None,
) -> SessionRun:
    backend = ScriptedBackend.from_mapping({"segments": segments or []})
    audit = MemoryAuditSink()
    mcap = tmp_path / "session.mcap"
    info = SessionInfo("test-session", "mem://test", total_frames_hint=len(images), task_whitelist=whitelist)
    result = SessionPipeline(config or PipelineConfig(), backend, audit).run(iter_arrays(images), info, mcap)
    return SessionRun(result, read_kept_indices(mcap), backend, audit, mcap)


PERSON = {"class": "person", "bbox": [40, 20, 160, 175], "conf": 0.95}


def child_face(age: float = 8.0, score: float = 0.95) -> dict[str, Any]:
    return {"bbox": [200, 30, 230, 70], "score": score, "age": age}
