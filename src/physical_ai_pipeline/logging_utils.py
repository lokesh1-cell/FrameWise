"""Structured JSON logging compatible with Google Cloud Logging.

Each record is one JSON object per line with ``severity``, ``message``,
``time`` and ``logger`` keys, plus any fields passed via ``extra={"fields": {...}}``
or :func:`log_event`.
"""

from __future__ import annotations

import json
import logging
import sys
import traceback
from datetime import UTC, datetime
from typing import Any

_RESERVED = "fields"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "time": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "severity": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        fields = getattr(record, _RESERVED, None)
        if isinstance(fields, dict):
            payload.update(fields)
        if record.exc_info:
            payload["exception"] = "".join(traceback.format_exception(*record.exc_info))
        return json.dumps(payload, default=_json_default, separators=(",", ":"))


def _json_default(value: object) -> Any:
    if hasattr(value, "item"):
        return value.item()  # numpy scalar
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    return str(value)


def configure_logging(level: str | int = "INFO", stream: Any = None) -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields: Any) -> None:
    fields["event"] = event
    logger.log(level, event, extra={_RESERVED: fields})
