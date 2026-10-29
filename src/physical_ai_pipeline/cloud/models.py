"""Sync TensorRT engines from GCS and pin their SHA256 from manifest.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from physical_ai_pipeline.cloud.gcs import GcsRef, ObjectStore, sha256_path
from physical_ai_pipeline.config import InferenceConfig, ModelConfig
from physical_ai_pipeline.logging_utils import get_logger, log_event

_log = get_logger(__name__)
_ROLES = ("detector", "pose", "hands", "face_detector", "face_attributes", "depth")


def sync_engines(store: ObjectStore, models_uri: str, engine_dir: str | Path) -> dict[str, str]:
    """Download ``manifest.json`` and every engine it lists into ``engine_dir``
    (skipping files whose hash already matches). Returns {engine filename: sha256}."""
    base = GcsRef.parse(models_uri.rstrip("/") + "/")
    target = Path(engine_dir)
    target.mkdir(parents=True, exist_ok=True)
    manifest_path = target / "manifest.json"
    store.download(GcsRef(base.bucket, base.name + "manifest.json"), manifest_path)
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    hashes: dict[str, str] = {}
    for entry in manifest.get("engines", []):
        name, expected = str(entry["engine"]), str(entry["sha256"])
        local = target / name
        if not (local.is_file() and sha256_path(local) == expected):
            actual = store.download(GcsRef(base.bucket, base.name + name), local)
            if actual != expected:
                local.unlink(missing_ok=True)
                raise ValueError(f"engine {name}: downloaded sha256 {actual} != manifest {expected}")
        hashes[name] = expected
        log_event(_log, "engine_synced", engine=name, sha256=expected)
    return hashes


def pin_hashes(config: InferenceConfig, hashes: dict[str, str]) -> InferenceConfig:
    """Return a copy of ``config`` whose ModelConfig.sha256 values come from the manifest."""
    updates: dict[str, ModelConfig] = {}
    for role in _ROLES:
        model: ModelConfig = getattr(config, role)
        name = Path(model.engine_path).name
        if model.enabled and name in hashes:
            updates[role] = model.model_copy(update={"sha256": hashes[name]})
    return config.model_copy(update=updates)
