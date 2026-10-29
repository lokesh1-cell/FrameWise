"""lineage.json: full provenance for one session MCAP."""

from __future__ import annotations

import json
import os
import platform
import socket
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from physical_ai_pipeline.config import PipelineConfig
from physical_ai_pipeline.inference.backend import ModelInfo

LINEAGE_SCHEMA_VERSION = "1"
_METADATA_URL = "http://metadata.google.internal/computeMetadata/v1/instance/"


def _gce_metadata(path: str, timeout: float = 0.5) -> str | None:
    request = urllib.request.Request(_METADATA_URL + path, headers={"Metadata-Flavor": "Google"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:  # noqa: S310 - fixed internal URL
            return str(resp.read().decode("utf-8"))
    except (OSError, ValueError):
        return None


def collect_node_info(query_metadata_server: bool = True) -> dict[str, Any]:
    """GKE node / pod identity from the downward API env vars and, when
    reachable, the GCE metadata server."""
    info: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "node_name": os.environ.get("NODE_NAME"),
        "pod_name": os.environ.get("POD_NAME"),
        "pod_namespace": os.environ.get("POD_NAMESPACE"),
        "pod_uid": os.environ.get("POD_UID"),
        "container_image": os.environ.get("CONTAINER_IMAGE"),
    }
    if query_metadata_server:
        for key, path in (("machine_type", "machine-type"), ("zone", "zone"), ("instance_id", "id")):
            value = _gce_metadata(path)
            info[key] = value.rsplit("/", 1)[-1] if value else None
        preemptible = _gce_metadata("scheduling/preemptible")
        info["spot"] = preemptible.strip().upper() == "TRUE" if preemptible else None
    try:
        import tensorrt

        info["tensorrt_version"] = tensorrt.__version__
    except ImportError:
        info["tensorrt_version"] = None
    return info


def build_lineage(
    *,
    session_id: str,
    source_uri: str,
    source_sha256: str,
    source_generation: str | None,
    mcap_uri: str,
    mcap_sha256: str,
    config: PipelineConfig,
    models: Sequence[ModelInfo],
    git_commit: str,
    stats: Mapping[str, Any],
    node_info: Mapping[str, Any],
    started_at: datetime,
    finished_at: datetime,
) -> dict[str, Any]:
    return {
        "schema_version": LINEAGE_SCHEMA_VERSION,
        "session_id": session_id,
        "pipeline": {"version": config.pipeline_version, "git_commit": git_commit},
        "source": {"uri": source_uri, "sha256": source_sha256, "gcs_generation": source_generation},
        "output": {"mcap_uri": mcap_uri, "mcap_sha256": mcap_sha256},
        "models": [asdict(m) for m in models],
        "filter_thresholds": json.loads(config.thresholds_json()),
        "processing": {
            "started_at": started_at.astimezone(UTC).isoformat(),
            "finished_at": finished_at.astimezone(UTC).isoformat(),
            "stats": dict(stats),
        },
        "compute": dict(node_info),
        "privacy": {
            "child_policy": "drop [F_k-30, F_k+30] around any sampled frame with age<18 & conf>0.85",
            "face_data_persisted": False,
        },
    }


def write_lineage(path: str | Path, lineage: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(lineage, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(target)
    return target


def git_commit() -> str:
    return os.environ.get("GIT_COMMIT", "unknown")


def utcnow() -> datetime:
    return datetime.now(tz=UTC)
