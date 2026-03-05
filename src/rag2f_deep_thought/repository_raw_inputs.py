"""DuckDB Repository Implementation for RAG2F Deep Thought.

Implements BaseRepository and QueryableRepository protocols using DuckDB
for local, embedded storage of text documents with deduplication support.
"""

import logging
from datetime import datetime
from threading import Lock
from typing import Any

import duckdb
from rag2f.core.xfiles.capabilities import (
    Capabilities,
    FeatureSupport,
    FilterCapability,
    NativeCapability,
    PaginationCapability,
    QueryCapability,
)
from rag2f.core.xfiles.exceptions import (
    AlreadyExists,
    BackendError,
    NotFound,
    NotSupported,
)
from rag2f.core.xfiles.repository import RepositoryNativeMixin
from rag2f.core.xfiles.types import (
    Document,
    DocumentId,
    Patch,
    QuerySpec,
    WhereNode,
)

from rag2f_deep_thought.bootstrap import TABLE_RAW_INPUTS

logger = logging.getLogger(__name__)


# Supported filter operators
SUPPORTED_OPS = ("eq", "ne", "gt", "gte", "lt", "lte", "in", "and", "or", "not")


class repository_raw_inputs(RepositoryNativeMixin):
    """DuckDB-based repository for text storage.

    This repository stores text documents with:
    - id: BLOB (16 bytes blake2b hash)
    - text: VARCHAR (original text)
    - created: TIMESTAMP (creation timestamp)

    Implements BaseRepository and QueryableRepository protocols.
    Thread-safe for concurrent access.
    """

    def __init__(self, db_path: str = ":memory:", table_name: str = TABLE_RAW_INPUTS):
        """Initialize DuckDB repository.

        Args:
            db_path: Path to DuckDB database file, or ":memory:" for in-memory.
            table_name: Name of the table to use for storage.
            repo_name: Human-readable name for this repository.
        """
        self._db_path = db_path
        self._table_name = table_name
        self._repo_name = self._table_name
        self._conn: duckdb.DuckDBPyConnection | None = None
        self._lock = Lock()

        self._setup_connection()

    def _setup_connection(self) -> None:
        """Initialize DuckDB connection and schema."""
        self._conn = duckdb.connect(self._db_path)

        logger.info("Initializing DuckDB repository (path=%s)", self._db_path)

        # Create table for texts
        # id: BLOB (16 bytes blake2b), text: VARCHAR, created: TIMESTAMP
        self._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._table_name} (
                id BLOB PRIMARY KEY,
                text VARCHAR NOT NULL,
                created TIMESTAMP NOT NULL
            )
        """)

        # Index for performance on id lookups
        self._conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self._table_name}_id 
            ON {self._table_name}(id)
        """)

        logger.info("DuckDB repository schema initialized (table=%s)", self._table_name)

    # =========================================================================
    # BaseRepository Protocol
    # =========================================================================

    @property
    def name(self) -> str:
        """Human-readable name for this repository instance."""
        return self._repo_name

    def capabilities(self) -> Capabilities:
        """Declare the capabilities supported by this repository."""
        return Capabilities(
            crud=True,
            query=QueryCapability(supported=True),
            projection=FeatureSupport(supported=True, pushdown=True),
            filter=FilterCapability(
                supported=True,
                pushdown=True,
                ops=SUPPORTED_OPS,
            ),
            order_by=FeatureSupport(supported=True, pushdown=True),
            pagination=PaginationCapability(
                supported=True,
                pushdown=True,
                mode="offset",
                max_limit=10000,
            ),
            native=NativeCapability(
                supported=True,
                kinds=("primary", "connection"),
            ),
        )

    def get(
        self,
        id: DocumentId,
        select: list[str] | None = None,
    ) -> Document:
        """Retrieve a document by its identifier.

        Args:
            id: The document identifier (bytes or hex string).
            select: Optional projection - list of field paths to return.

        Returns:
            The document as a dict.

        Raises:
            NotFound: If the document doesn't exist.
        """
        id_bytes = self._normalize_id(id)

        # Build projection
        columns = self._build_select(select)

        with self._lock:
            result = self._conn.execute(
                f"SELECT {columns} FROM {self._table_name} WHERE id = ?",  # noqa: S608
                [id_bytes],
            ).fetchone()

        if result is None:
            raise NotFound(id=id_bytes.hex(), repository=self._repo_name)

        return self._row_to_document(result, select)

    def insert(self, id: DocumentId, item: Document) -> None:
        """Insert a new document.

        Args:
            id: The document identifier (bytes or hex string).
            item: The document data (must contain 'text', optionally 'created').

        Raises:
            AlreadyExists: If a document with this id already exists.
            BackendError: If the backend operation fails.
        """
        id_bytes = self._normalize_id(id)
        text = item.get("text", "")
        created = item.get("created", datetime.now())

        if isinstance(created, str):
            created = datetime.fromisoformat(created)

        try:
            with self._lock:
                self._conn.execute(
                    f"INSERT INTO {self._table_name} (id, text, created) VALUES (?, ?, ?)",  # noqa: S608
                    [id_bytes, text, created],
                )
            logger.debug("Inserted document id=%s", id_bytes.hex())
        except duckdb.ConstraintException as e:
            raise AlreadyExists(id=id_bytes.hex(), repository=self._repo_name) from e
        except Exception as e:
            raise BackendError(f"Failed to insert document: {e}", cause=e) from e

    def update(self, id: DocumentId, patch: Patch) -> None:
        """Update an existing document with a partial patch.

        Args:
            id: The document identifier.
            patch: Partial update data (keys to merge).

        Raises:
            NotFound: If the document doesn't exist.
            BackendError: If the backend operation fails.
        """
        id_bytes = self._normalize_id(id)

        # Check if document exists
        with self._lock:
            existing = self._conn.execute(
                f"SELECT 1 FROM {self._table_name} WHERE id = ?",  # noqa: S608
                [id_bytes],
            ).fetchone()

        if existing is None:
            raise NotFound(id=id_bytes.hex(), repository=self._repo_name)

        # Build SET clause for allowed fields
        allowed_fields = {"text", "created"}
        set_clauses = []
        values = []

        for key, value in patch.items():
            if key in allowed_fields:
                set_clauses.append(f"{key} = ?")
                if key == "created" and isinstance(value, str):
                    value = datetime.fromisoformat(value)
                values.append(value)

        if not set_clauses:
            return  # Nothing to update

        values.append(id_bytes)  # For WHERE clause

        try:
            with self._lock:
                self._conn.execute(
                    f"UPDATE {self._table_name} SET {', '.join(set_clauses)} WHERE id = ?",  # noqa: S608
                    values,
                )
            logger.debug("Updated document id=%s", id_bytes.hex())
        except Exception as e:
            raise BackendError(f"Failed to update document: {e}", cause=e) from e

    def delete(self, id: DocumentId) -> None:
        """Delete a document by its identifier.

        Args:
            id: The document identifier.

        Raises:
            NotFound: If the document doesn't exist.
            BackendError: If the backend operation fails.
        """
        id_bytes = self._normalize_id(id)

        try:
            with self._lock:
                result = self._conn.execute(
                    f"DELETE FROM {self._table_name} WHERE id = ? RETURNING id",  # noqa: S608
                    [id_bytes],
                ).fetchone()

            if result is None:
                raise NotFound(id=id_bytes.hex(), repository=self._repo_name)

            logger.debug("Deleted document id=%s", id_bytes.hex())
        except NotFound:
            raise
        except Exception as e:
            raise BackendError(f"Failed to delete document: {e}", cause=e) from e

    def _get_native_handle(self, kind: str) -> object:
        """Get native DuckDB handle.

        Args:
            kind: "primary" or "connection" for the DuckDB connection.

        Returns:
            The DuckDB connection object.
        """
        if kind in ("primary", "connection"):
            return self._conn
        raise NotSupported(
            f"native:{kind}",
            repository=self._repo_name,
            details=f"Kind '{kind}' not available",
        )

    # =========================================================================
    # QueryableRepository Protocol
    # =========================================================================

    def find(self, query: QuerySpec) -> list[Document]:
        """Find documents matching the query specification.

        Args:
            query: Query specification with select, where, order_by,
                limit, and offset.

        Returns:
            List of matching documents.
        """
        # Build SQL query
        columns = self._build_select(query.select)
        sql = f"SELECT {columns} FROM {self._table_name}"  # noqa: S608
        params: list[Any] = []

        # WHERE clause
        if query.where is not None:
            where_sql, where_params = self._build_where(query.where)
            sql += f" WHERE {where_sql}"
            params.extend(where_params)

        # ORDER BY clause
        if query.order_by:
            order_sql = self._build_order_by(query.order_by)
            sql += f" ORDER BY {order_sql}"

        # LIMIT and OFFSET
        if query.limit is not None:
            sql += f" LIMIT {query.limit}"
        if query.offset > 0:
            sql += f" OFFSET {query.offset}"

        logger.debug("Executing query: %s with params: %s", sql, params)

        with self._lock:
            results = self._conn.execute(sql, params).fetchall()

        return [self._row_to_document(row, query.select) for row in results]

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _normalize_id(self, id: DocumentId) -> bytes:
        """Convert id to bytes if needed."""
        if isinstance(id, bytes):
            return id
        if isinstance(id, str):
            # Assume hex string
            return bytes.fromhex(id)
        raise ValueError(f"Invalid id type: {type(id)}")

    def _build_select(self, select: list[str] | None) -> str:
        """Build SELECT columns clause."""
        if select is None or not select:
            return "id, text, created"

        # Map to actual column names, validate
        valid_columns = {"id", "text", "created"}
        columns = [col for col in select if col in valid_columns]

        if not columns:
            return "id, text, created"

        return ", ".join(columns)

    def _build_where(self, node: WhereNode) -> tuple[str, list[Any]]:
        """Build WHERE clause from WhereNode AST.

        Returns:
            Tuple of (sql_fragment, params_list).
        """
        if not node:
            return "1=1", []

        op = node[0]

        if op == "eq":
            field, value = node[1], node[2]
            return f"{self._safe_column(field)} = ?", [self._convert_value(field, value)]

        elif op == "ne":
            field, value = node[1], node[2]
            return f"{self._safe_column(field)} != ?", [self._convert_value(field, value)]

        elif op == "gt":
            field, value = node[1], node[2]
            return f"{self._safe_column(field)} > ?", [self._convert_value(field, value)]

        elif op == "gte":
            field, value = node[1], node[2]
            return f"{self._safe_column(field)} >= ?", [self._convert_value(field, value)]

        elif op == "lt":
            field, value = node[1], node[2]
            return f"{self._safe_column(field)} < ?", [self._convert_value(field, value)]

        elif op == "lte":
            field, value = node[1], node[2]
            return f"{self._safe_column(field)} <= ?", [self._convert_value(field, value)]

        elif op == "in":
            field, values = node[1], node[2]
            if not values:
                return "1=0", []  # Empty IN = no match
            placeholders = ", ".join("?" for _ in values)
            converted = [self._convert_value(field, v) for v in values]
            return f"{self._safe_column(field)} IN ({placeholders})", converted

        elif op == "and":
            left_sql, left_params = self._build_where(node[1])
            right_sql, right_params = self._build_where(node[2])
            return f"({left_sql} AND {right_sql})", left_params + right_params

        elif op == "or":
            left_sql, left_params = self._build_where(node[1])
            right_sql, right_params = self._build_where(node[2])
            return f"({left_sql} OR {right_sql})", left_params + right_params

        elif op == "not":
            inner_sql, inner_params = self._build_where(node[1])
            return f"NOT ({inner_sql})", inner_params

        else:
            raise NotSupported(
                f"filter:{op}",
                repository=self._repo_name,
                details=f"Operator '{op}' not supported",
            )

    def _safe_column(self, field: str) -> str:
        """Validate and return safe column name."""
        valid_columns = {"id", "text", "created"}
        if field not in valid_columns:
            raise NotSupported(
                f"field:{field}",
                repository=self._repo_name,
                details=f"Field '{field}' not available. Valid: {valid_columns}",
            )
        return field

    def _convert_value(self, field: str, value: Any) -> Any:
        """Convert value for the given field type."""
        if field == "id":
            if isinstance(value, str):
                return bytes.fromhex(value)
            return value
        if field == "created" and isinstance(value, str):
            return datetime.fromisoformat(value)
        return value

    def _build_order_by(self, order_by: list[str]) -> str:
        """Build ORDER BY clause."""
        clauses = []
        for field in order_by:
            if field.startswith("-"):
                col = self._safe_column(field[1:])
                clauses.append(f"{col} DESC")
            else:
                col = self._safe_column(field)
                clauses.append(f"{col} ASC")
        return ", ".join(clauses) if clauses else "id ASC"

    def _row_to_document(
        self,
        row: tuple,
        select: list[str] | None = None,
    ) -> Document:
        """Convert database row to Document dict."""
        if select is None or not select:
            # Default: id, text, created
            return {
                "id": row[0].hex() if isinstance(row[0], bytes) else row[0],
                "text": row[1],
                "created": row[2].isoformat() if hasattr(row[2], "isoformat") else row[2],
            }

        # Map based on select order
        valid_columns = ["id", "text", "created"]
        selected = [col for col in select if col in valid_columns]

        doc = {}
        for i, col in enumerate(selected):
            value = row[i]
            if col == "id" and isinstance(value, bytes):
                value = value.hex()
            elif col == "created" and hasattr(value, "isoformat"):
                value = value.isoformat()
            doc[col] = value

        return doc

    def close(self) -> None:
        """Close the DuckDB connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info("DuckDB repository connection closed")


__all__ = ["repository_raw_inputs"]
