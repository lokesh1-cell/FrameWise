"""Streaming inserts into the BigQuery ``frame_audit_log`` table."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from physical_ai_pipeline.audit import AuditRow
from physical_ai_pipeline.logging_utils import get_logger, log_event

_log = get_logger(__name__)


class BigQueryInsertError(RuntimeError):
    pass


class BigQueryAuditSink:
    """Buffers rows and streams them with ``insert_rows_json``.

    ``insert_id`` becomes the BigQuery insertId (best-effort dedup when a
    redelivered Pub/Sub message reprocesses a session) and is not stored as a
    column.
    """

    def __init__(self, table_id: str, batch_rows: int = 500, client: Any | None = None) -> None:
        if client is None:
            from google.cloud import bigquery

            client = bigquery.Client()
        self._client = client
        self.table_id = table_id
        self.batch_rows = batch_rows
        self._buffer: list[AuditRow] = []
        self.rows_written = 0

    def write_rows(self, rows: Sequence[AuditRow]) -> None:
        self._buffer.extend(rows)
        while len(self._buffer) >= self.batch_rows:
            self._send(self._buffer[: self.batch_rows])
            del self._buffer[: self.batch_rows]

    def flush(self) -> None:
        if self._buffer:
            self._send(self._buffer)
            self._buffer = []

    def _send(self, rows: Sequence[AuditRow]) -> None:
        row_ids = [str(r.get("insert_id", "")) or None for r in rows]
        payload = [{k: v for k, v in r.items() if k != "insert_id"} for r in rows]
        errors = self._client.insert_rows_json(self.table_id, payload, row_ids=row_ids)
        if errors:
            log_event(_log, "bigquery_insert_failed", level=40, table=self.table_id, errors=errors[:5])
            raise BigQueryInsertError(f"{len(errors)} rows rejected by {self.table_id}")
        self.rows_written += len(rows)
        log_event(_log, "bigquery_rows_written", level=10, table=self.table_id, rows=len(rows))
