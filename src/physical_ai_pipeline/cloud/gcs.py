"""GCS access behind a small protocol so the worker is testable offline."""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class GcsRef:
    bucket: str
    name: str
    generation: str | None = None

    @property
    def uri(self) -> str:
        return f"gs://{self.bucket}/{self.name}"

    @classmethod
    def parse(cls, uri: str) -> GcsRef:
        if not uri.startswith("gs://"):
            raise ValueError(f"not a gs:// URI: {uri}")
        bucket, _, name = uri[5:].partition("/")
        if not bucket:
            raise ValueError(f"missing bucket in {uri}")
        return cls(bucket, name)


class ObjectStore(Protocol):
    def download(self, ref: GcsRef, dest: Path) -> str:
        """Download to ``dest`` and return the SHA256 hex digest."""
        ...

    def upload(self, src: Path, ref: GcsRef, content_type: str) -> None: ...

    def exists(self, ref: GcsRef) -> bool: ...

    def metadata(self, ref: GcsRef) -> dict[str, Any]:
        """Object metadata: ``custom`` (x-goog-meta-*), ``time_created`` (RFC3339), ``size``."""
        ...

    def list(self, prefix: GcsRef) -> list[GcsRef]: ...


_CHUNK = 8 * 1024 * 1024


class GoogleCloudStorage:
    def __init__(self, project: str | None = None) -> None:
        from google.cloud import storage

        self._client = storage.Client(project=project)

    def _blob(self, ref: GcsRef) -> Any:
        generation = int(ref.generation) if ref.generation else None
        return self._client.bucket(ref.bucket).blob(ref.name, generation=generation)

    def download(self, ref: GcsRef, dest: Path) -> str:
        dest.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        with self._blob(ref).open("rb", chunk_size=_CHUNK) as src, open(dest, "wb") as out:
            while block := src.read(_CHUNK):
                digest.update(block)
                out.write(block)
        return digest.hexdigest()

    def upload(self, src: Path, ref: GcsRef, content_type: str) -> None:
        blob = self._client.bucket(ref.bucket).blob(ref.name)
        blob.upload_from_filename(str(src), content_type=content_type, timeout=600)

    def exists(self, ref: GcsRef) -> bool:
        return bool(self._client.bucket(ref.bucket).blob(ref.name).exists())

    def metadata(self, ref: GcsRef) -> dict[str, Any]:
        blob = self._client.bucket(ref.bucket).get_blob(ref.name)
        if blob is None:
            raise FileNotFoundError(ref.uri)
        return {
            "custom": dict(blob.metadata or {}),
            "time_created": blob.time_created.isoformat() if blob.time_created else None,
            "size": blob.size,
            "generation": str(blob.generation),
        }

    def list(self, prefix: GcsRef) -> list[GcsRef]:
        return [GcsRef(prefix.bucket, b.name) for b in self._client.list_blobs(prefix.bucket, prefix=prefix.name)]


class LocalObjectStore:
    """Maps gs://bucket/name to ``root/bucket/name`` (tests and local runs).
    Custom metadata lives in ``<file>.meta.json`` when present."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path(self, ref: GcsRef) -> Path:
        return self.root / ref.bucket / ref.name

    def download(self, ref: GcsRef, dest: Path) -> str:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.path(ref), dest)
        return sha256_path(dest)

    def upload(self, src: Path, ref: GcsRef, content_type: str) -> None:
        target = self.path(ref)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)

    def exists(self, ref: GcsRef) -> bool:
        return self.path(ref).is_file()

    def metadata(self, ref: GcsRef) -> dict[str, Any]:
        import json
        from datetime import UTC, datetime

        path = self.path(ref)
        meta_path = path.with_name(path.name + ".meta.json")
        custom = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
        stat = path.stat()
        return {
            "custom": custom,
            "time_created": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
            "size": stat.st_size,
            "generation": "1",
        }

    def list(self, prefix: GcsRef) -> list[GcsRef]:
        base = self.root / prefix.bucket
        return [
            GcsRef(prefix.bucket, str(p.relative_to(base)))
            for p in sorted(base.rglob("*"))
            if p.is_file() and str(p.relative_to(base)).startswith(prefix.name)
        ]


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(_CHUNK):
            digest.update(block)
    return digest.hexdigest()
