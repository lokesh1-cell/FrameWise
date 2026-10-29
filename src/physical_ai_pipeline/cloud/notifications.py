"""Parse GCS Pub/Sub notifications (JSON_API_V1 payload format)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

OBJECT_FINALIZE = "OBJECT_FINALIZE"


@dataclass(frozen=True, slots=True)
class GcsEvent:
    event_type: str
    bucket: str
    name: str
    generation: str | None
    size: int | None
    content_type: str | None
    time_created: str | None
    metadata: Mapping[str, str]

    @property
    def uri(self) -> str:
        return f"gs://{self.bucket}/{self.name}"


def parse_notification(attributes: Mapping[str, str], data: bytes) -> GcsEvent:
    """Build a GcsEvent from a Pub/Sub message's attributes and body.

    Raises ValueError for messages that are not GCS notifications.
    """
    event_type = attributes.get("eventType")
    bucket = attributes.get("bucketId")
    name = attributes.get("objectId")
    if not (event_type and bucket and name):
        raise ValueError("message is not a GCS notification (missing eventType/bucketId/objectId)")
    payload: dict[str, Any] = {}
    if data:
        try:
            loaded = json.loads(data.decode("utf-8"))
            payload = loaded if isinstance(loaded, dict) else {}
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = {}
    size = payload.get("size")
    return GcsEvent(
        event_type=event_type,
        bucket=bucket,
        name=name,
        generation=attributes.get("objectGeneration") or payload.get("generation"),
        size=int(size) if size is not None else None,
        content_type=payload.get("contentType"),
        time_created=payload.get("timeCreated"),
        metadata=dict(payload.get("metadata") or {}),
    )
