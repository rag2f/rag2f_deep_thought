"""DuckDB task store for FluxCapacitor."""

# ruff: noqa: S608

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from threading import Lock
from typing import Any

import duckdb
from rag2f.core.flux_capacitor.store import BaseTaskStore
from rag2f.core.flux_capacitor.task_models import PayloadRef, Task, TaskStatusView

logger = logging.getLogger(__name__)

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TERMINAL_STATUSES = {"completed", "failed"}


def _safe_identifier(value: str) -> str:
    """Return a validated DuckDB identifier."""
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"Invalid DuckDB identifier: {value!r}")
    return value


def _payload_to_json(payload_ref: PayloadRef | dict[str, Any] | None) -> str | None:
    """Serialize a Flux payload reference for storage."""
    if payload_ref is None:
        return None
    payload = payload_ref.to_dict() if isinstance(payload_ref, PayloadRef) else dict(payload_ref)
    return json.dumps(payload, sort_keys=True)


def _payload_from_json(value: str | None) -> PayloadRef | None:
    """Deserialize a Flux payload reference from storage."""
    if not value:
        return None
    payload = json.loads(value)
    if not isinstance(payload, dict):
        return None
    return PayloadRef.from_mapping(payload)


class DuckDBTaskStore(BaseTaskStore):
    """DuckDB-backed Flux task store.

    The default database path is ``:memory:`` because task state is currently
    intended for local orchestration and testable workflow tracking.
    """

    def __init__(self, db_path: str = ":memory:", table_name: str = "flux_tasks") -> None:
        """Initialize the task store."""
        self._db_path = db_path
        self._table_name = _safe_identifier(table_name)
        self._lock = Lock()
        self._conn = duckdb.connect(db_path)
        self._setup_schema()

    def _setup_schema(self) -> None:
        """Create the task table when missing."""
        with self._lock:
            self._conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._table_name} (
                    id VARCHAR PRIMARY KEY,
                    root_id VARCHAR,
                    parent_id VARCHAR,
                    plugin_id VARCHAR NOT NULL,
                    hook VARCHAR NOT NULL,
                    payload_ref JSON,
                    status VARCHAR NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    started_at TIMESTAMP,
                    finished_at TIMESTAMP,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    worker_id VARCHAR,
                    last_error VARCHAR,
                    queue_ref VARCHAR,
                    reservation_ref VARCHAR
                )
            """)
            self._conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self._table_name}_parent_id
                ON {self._table_name}(parent_id)
            """)
            self._conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self._table_name}_status
                ON {self._table_name}(status)
            """)
        logger.info(
            "DuckDB Flux task store initialized (path=%s, table=%s)",
            self._db_path,
            self._table_name,
        )

    def create_task(self, task: Task) -> Task:
        """Persist a task and return it."""
        if task.parent_id:
            parent = self.get_task(task.parent_id)
            if parent is not None:
                task.root_id = parent.root_id
        else:
            task.root_id = task.id

        with self._lock:
            self._conn.execute(
                f"""
                INSERT INTO {self._table_name} (
                    id, root_id, parent_id, plugin_id, hook, payload_ref, status,
                    created_at, started_at, finished_at, attempts, worker_id,
                    last_error, queue_ref, reservation_ref
                ) VALUES (?, ?, ?, ?, ?, ?::JSON, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    task.id,
                    task.root_id,
                    task.parent_id,
                    task.plugin_id,
                    task.hook,
                    _payload_to_json(task.payload_ref),
                    task.status,
                    task.created_at,
                    task.started_at,
                    task.finished_at,
                    task.attempts,
                    task.worker_id,
                    task.last_error,
                    task.queue_ref,
                    task.reservation_ref,
                ],
            )
        return task

    def get_task(self, task_id: str) -> Task | None:
        """Fetch a task by id."""
        with self._lock:
            row = self._conn.execute(
                f"""
                SELECT id, root_id, parent_id, plugin_id, hook, payload_ref, status,
                       created_at, started_at, finished_at, attempts, worker_id,
                       last_error, queue_ref, reservation_ref
                FROM {self._table_name}
                WHERE id = ?
                """,
                [task_id],
            ).fetchone()
        if row is None:
            return None
        return self._row_to_task(row)

    def list_children(self, parent_id: str) -> list[Task]:
        """Return tasks that have ``parent_id`` as parent."""
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT id, root_id, parent_id, plugin_id, hook, payload_ref, status,
                       created_at, started_at, finished_at, attempts, worker_id,
                       last_error, queue_ref, reservation_ref
                FROM {self._table_name}
                WHERE parent_id = ?
                ORDER BY created_at ASC, id ASC
                """,
                [parent_id],
            ).fetchall()
        return [self._row_to_task(row) for row in rows]

    def list_unfinished_tasks(self) -> list[Task]:
        """Return tasks that are not in a terminal state."""
        placeholders = ", ".join("?" for _ in _TERMINAL_STATUSES)
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT id, root_id, parent_id, plugin_id, hook, payload_ref, status,
                       created_at, started_at, finished_at, attempts, worker_id,
                       last_error, queue_ref, reservation_ref
                FROM {self._table_name}
                WHERE status NOT IN ({placeholders})
                ORDER BY created_at ASC, id ASC
                """,
                list(_TERMINAL_STATUSES),
            ).fetchall()
        return [self._row_to_task(row) for row in rows]

    def mark_reserved(self, task_id: str, *, worker_id: str, reservation_ref: str | None) -> None:
        """Mark a task as reserved for execution."""
        with self._lock:
            self._conn.execute(
                f"""
                UPDATE {self._table_name}
                SET status = 'reserved', started_at = ?, finished_at = NULL,
                    worker_id = ?, reservation_ref = ?, last_error = NULL,
                    attempts = attempts + 1
                WHERE id = ?
                """,
                [datetime.now(UTC), worker_id, reservation_ref, task_id],
            )

    def mark_completed(self, task_id: str) -> None:
        """Mark task as finished successfully."""
        with self._lock:
            self._conn.execute(
                f"""
                UPDATE {self._table_name}
                SET status = 'completed', finished_at = ?, worker_id = NULL,
                    reservation_ref = NULL, last_error = NULL
                WHERE id = ?
                """,
                [datetime.now(UTC), task_id],
            )

    def mark_failed(self, task_id: str, *, error_msg: str) -> None:
        """Mark task as finished with error."""
        with self._lock:
            self._conn.execute(
                f"""
                UPDATE {self._table_name}
                SET status = 'failed', finished_at = ?, worker_id = NULL,
                    reservation_ref = NULL, last_error = ?
                WHERE id = ?
                """,
                [datetime.now(UTC), error_msg, task_id],
            )

    def mark_retry(self, task_id: str, *, error_msg: str) -> None:
        """Mark task for retry after a failed execution attempt."""
        with self._lock:
            self._conn.execute(
                f"""
                UPDATE {self._table_name}
                SET status = 'retry_scheduled', finished_at = NULL, worker_id = NULL,
                    reservation_ref = NULL, last_error = ?
                WHERE id = ?
                """,
                [error_msg, task_id],
            )

    def get_status(self, task_id: str, *, include_descendants: bool = False) -> TaskStatusView:
        """Return the status projection for a task."""
        task = self.get_task(task_id)
        if task is None:
            return TaskStatusView(
                exists=False,
                task_id=task_id,
                root_id=None,
                parent_id=None,
                plugin_id=None,
                hook=None,
                status="missing",
            )

        descendants = self._list_descendants(task.id)
        counts = {
            "pending": 0,
            "reserved": 0,
            "completed": 0,
            "failed": 0,
            "retry_scheduled": 0,
        }
        for descendant in descendants:
            if descendant.status in counts:
                counts[descendant.status] += 1

        return TaskStatusView(
            exists=True,
            task_id=task.id,
            root_id=task.root_id,
            parent_id=task.parent_id,
            plugin_id=task.plugin_id,
            hook=task.hook,
            status=task.status,
            attempts=task.attempts,
            worker_id=task.worker_id,
            last_error=task.last_error,
            has_children=bool(self.list_children(task.id)),
            descendant_count=len(descendants),
            pending_count=counts["pending"],
            reserved_count=counts["reserved"],
            completed_count=counts["completed"],
            failed_count=counts["failed"],
            retry_scheduled_count=counts["retry_scheduled"],
            tree_completed=(
                task.status == "completed"
                and all(descendant.status == "completed" for descendant in descendants)
            ),
            descendants=descendants if include_descendants else [],
        )

    def _list_descendants(self, task_id: str) -> list[Task]:
        """Return all descendants for a task."""
        descendants: list[Task] = []
        for child in self.list_children(task_id):
            descendants.append(child)
            descendants.extend(self._list_descendants(child.id))
        return descendants

    def _row_to_task(self, row: tuple[Any, ...]) -> Task:
        """Convert a DuckDB row to a Flux task."""
        task = Task(
            id=row[0],
            root_id=row[1],
            parent_id=row[2],
            plugin_id=row[3],
            hook=row[4],
            payload_ref=_payload_from_json(row[5]),
            status=row[6],
            created_at=row[7],
            started_at=row[8],
            finished_at=row[9],
            attempts=row[10],
            worker_id=row[11],
            last_error=row[12],
            queue_ref=row[13],
            reservation_ref=row[14],
        )
        return task

    def close(self) -> None:
        """Close the DuckDB connection."""
        self._conn.close()
