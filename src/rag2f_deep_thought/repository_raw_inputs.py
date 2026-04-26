"""DuckDB repository for raw inputs and optional embeddings."""

# ruff: noqa: S608

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
    VectorSearchCapability,
)
from rag2f.core.xfiles.exceptions import (
    AlreadyExists,
    BackendError,
    NotFound,
    NotSupported,
    ValidationError,
)
from rag2f.core.xfiles.repository import RepositoryNativeMixin
from rag2f.core.xfiles.types import Document, DocumentId, Patch, QuerySpec, WhereNode

from rag2f_deep_thought.bootstrap import TABLE_RAW_INPUTS

logger = logging.getLogger(__name__)


SUPPORTED_OPS = ("eq", "ne", "gt", "gte", "lt", "lte", "in", "and", "or", "not")


def _normalize_embedding_size(value: Any) -> int | None:
    """Normalize configured embedding size to an integer when present."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("Embedding size must be an integer, not a boolean")

    try:
        size = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid embedding size: {value!r}") from exc

    if size <= 0:
        raise ValueError(f"Embedding size must be positive, got {size}")
    return size


def _normalize_embedding(embedding: Any, expected_size: int | None = None) -> list[float] | None:
    """Normalize an embedding vector to a list of floats or None."""
    if embedding is None:
        return None
    if not isinstance(embedding, list | tuple):
        raise ValidationError("Embedding must be a list or tuple of floats")

    normalized = [float(value) for value in embedding]
    if expected_size is not None and len(normalized) != expected_size:
        raise ValidationError(
            f"Embedding size mismatch: expected {expected_size}, got {len(normalized)}"
        )
    return normalized


class RawInputsRepository(RepositoryNativeMixin):
    """DuckDB-based repository for raw text storage and vector retrieval."""

    def __init__(
        self,
        db_path: str = ":memory:",
        table_name: str = TABLE_RAW_INPUTS,
        embedding_size: int | None = None,
        enable_hnsw: bool = True,
    ):
        """Initialize DuckDB repository.

        Args:
            db_path: Path to DuckDB database file, or ":memory:" for in-memory.
            table_name: Name of the table to use for storage.
            embedding_size: Optional fixed embedding dimension used for validation
                and HNSW indexing.
            enable_hnsw: Whether HNSW/VSS should be attempted.
        """
        self._db_path = db_path
        self._table_name = table_name
        self._repo_name = self._table_name
        self._embedding_size = _normalize_embedding_size(embedding_size)
        self._enable_hnsw = enable_hnsw
        self._conn: duckdb.DuckDBPyConnection | None = None
        self._lock = Lock()
        self._hnsw_index_name = f"idx_{self._table_name}_embedding_hnsw"
        self._hnsw_index_enabled = False

        self._setup_connection()

    def _setup_connection(self) -> None:
        """Initialize DuckDB connection and schema."""
        self._conn = duckdb.connect(self._db_path)

        logger.info("Initializing DuckDB repository (path=%s)", self._db_path)

        embedding_type = self._embedding_column_type()
        self._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._table_name} (
                id BLOB PRIMARY KEY,
                text VARCHAR NOT NULL,
                created TIMESTAMP NOT NULL,
                embedding {embedding_type},
                flux_task_id VARCHAR
            )
        """)

        self._conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self._table_name}_id
            ON {self._table_name}(id)
        """)

        self._ensure_embedding_column_shape()
        self._ensure_flux_task_id_column()
        self._try_enable_hnsw_index()

        logger.info("DuckDB repository schema initialized (table=%s)", self._table_name)

    def _embedding_column_type(self) -> str:
        """Return the DuckDB column type for embeddings."""
        if self._embedding_size is None:
            return "FLOAT[]"
        return f"FLOAT[{self._embedding_size}]"

    def _ensure_embedding_column_shape(self) -> None:
        """Validate the existing embedding column matches the configured shape."""
        column_info = self._conn.execute(f"DESCRIBE {self._table_name}").fetchall()  # noqa: S608
        embedding_info = next((row for row in column_info if row[0] == "embedding"), None)
        if embedding_info is None:
            raise BackendError(f"Table {self._table_name} is missing embedding column")

        actual_type = str(embedding_info[1]).upper()
        expected_type = self._embedding_column_type().upper()
        if actual_type != expected_type:
            raise BackendError(
                "Embedding column type mismatch: "
                f"expected {expected_type}, found {actual_type}. "
                "Recreate the database or align the configured embedding size."
            )

    def _ensure_flux_task_id_column(self) -> None:
        """Add the async Flux task id column to existing tables."""
        column_info = self._conn.execute(f"DESCRIBE {self._table_name}").fetchall()  # noqa: S608
        if any(row[0] == "flux_task_id" for row in column_info):
            return
        self._conn.execute(f"ALTER TABLE {self._table_name} ADD COLUMN flux_task_id VARCHAR")

    def _try_enable_hnsw_index(self) -> None:
        """Enable DuckDB VSS/HNSW indexing when available."""
        if self._embedding_size is None:
            logger.info("HNSW index skipped because embedding size is not configured")
            return

        if not self._enable_hnsw:
            logger.info("HNSW index disabled; repository will use exact vector search")
            return

        try:
            self._conn.execute("LOAD vss")
            if self._db_path != ":memory:":
                self._conn.execute("SET hnsw_enable_experimental_persistence = true")
            self._conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {self._hnsw_index_name}
                ON {self._table_name}
                USING HNSW (embedding)
                WITH (metric = 'cosine')
                """
            )
            self._hnsw_index_enabled = True
            logger.info(
                "DuckDB HNSW index enabled (table=%s, index=%s)",
                self._table_name,
                self._hnsw_index_name,
            )
        except Exception as exc:
            logger.warning("DuckDB HNSW unavailable after bootstrap check: %s", exc)

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
            vector_search=VectorSearchCapability(
                supported=True,
                dimensions=self._embedding_size,
                distance_metrics=("cosine",),
            ),
        )

    def get(
        self,
        id: DocumentId,
        select: list[str] | None = None,
    ) -> Document:
        """Retrieve a document by its identifier."""
        id_bytes = self._normalize_id(id)
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
        """Insert a new document."""
        id_bytes = self._normalize_id(id)
        text = item.get("text", "")
        created = item.get("created", datetime.now())
        embedding = _normalize_embedding(item.get("embedding"), self._embedding_size)
        flux_task_id = item.get("flux_task_id")

        if isinstance(created, str):
            created = datetime.fromisoformat(created)

        try:
            with self._lock:
                self._conn.execute(
                    f"INSERT INTO {self._table_name} (id, text, created, embedding, flux_task_id) VALUES (?, ?, ?, ?, ?)",  # noqa: S608
                    [id_bytes, text, created, embedding, flux_task_id],
                )
            logger.debug("Inserted document id=%s", id_bytes.hex())
        except duckdb.ConstraintException as exc:
            raise AlreadyExists(id=id_bytes.hex(), repository=self._repo_name) from exc
        except Exception as exc:
            raise BackendError(f"Failed to insert document: {exc}", cause=exc) from exc

    def update(self, id: DocumentId, patch: Patch) -> None:
        """Update an existing document with a partial patch."""
        id_bytes = self._normalize_id(id)

        with self._lock:
            existing = self._conn.execute(
                f"SELECT 1 FROM {self._table_name} WHERE id = ?",  # noqa: S608
                [id_bytes],
            ).fetchone()

        if existing is None:
            raise NotFound(id=id_bytes.hex(), repository=self._repo_name)

        allowed_fields = {"text", "created", "embedding", "flux_task_id"}
        set_clauses = []
        values = []

        for key, value in patch.items():
            if key in allowed_fields:
                set_clauses.append(f"{key} = ?")
                if key == "created" and isinstance(value, str):
                    value = datetime.fromisoformat(value)
                elif key == "embedding":
                    value = _normalize_embedding(value, self._embedding_size)
                values.append(value)

        if not set_clauses:
            return

        values.append(id_bytes)

        try:
            with self._lock:
                self._conn.execute(
                    f"UPDATE {self._table_name} SET {', '.join(set_clauses)} WHERE id = ?",  # noqa: S608
                    values,
                )
            logger.debug("Updated document id=%s", id_bytes.hex())
        except Exception as exc:
            raise BackendError(f"Failed to update document: {exc}", cause=exc) from exc

    def delete(self, id: DocumentId) -> None:
        """Delete a document by its identifier."""
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
        except Exception as exc:
            raise BackendError(f"Failed to delete document: {exc}", cause=exc) from exc

    def _get_native_handle(self, kind: str) -> object:
        """Get native DuckDB handle."""
        if kind in ("primary", "connection"):
            return self._conn
        raise NotSupported(
            f"native:{kind}",
            repository=self._repo_name,
            details=f"Kind '{kind}' not available",
        )

    def find(self, query: QuerySpec) -> list[Document]:
        """Find documents matching the query specification."""
        columns = self._build_select(query.select)
        sql = f"SELECT {columns} FROM {self._table_name}"  # noqa: S608
        params: list[Any] = []

        if query.where is not None:
            where_sql, where_params = self._build_where(query.where)
            sql += f" WHERE {where_sql}"
            params.extend(where_params)

        if query.order_by:
            order_sql = self._build_order_by(query.order_by)
            sql += f" ORDER BY {order_sql}"

        if query.limit is not None:
            sql += f" LIMIT {query.limit}"
        if query.offset > 0:
            sql += f" OFFSET {query.offset}"

        logger.debug("Executing query: %s with params: %s", sql, params)

        with self._lock:
            results = self._conn.execute(sql, params).fetchall()

        return [self._row_to_document(row, query.select) for row in results]

    def vector_search(
        self,
        embedding: list[float],
        top_k: int = 10,
        where: WhereNode | None = None,
        select: list[str] | None = None,
    ) -> list[Document]:
        """Search documents by cosine similarity."""
        query_embedding = _normalize_embedding(embedding, self._embedding_size)
        if query_embedding is None:
            raise ValidationError("Embedding must not be None for vector_search")

        if top_k <= 0:
            return []

        columns = self._build_select(select)
        filters = ["embedding IS NOT NULL"]
        params: list[Any] = []

        if where is not None:
            where_sql, where_params = self._build_where(where)
            filters.append(where_sql)
            params.extend(where_params)

        if self._embedding_size is None:
            sql = (  # noqa: S608
                f"SELECT {columns}, list_cosine_similarity(embedding, ?) AS _score "
                f"FROM {self._table_name} WHERE {' AND '.join(filters)} "
                f"ORDER BY _score DESC LIMIT {int(top_k)}"
            )
            params.append(query_embedding)
        else:
            cast = f"FLOAT[{self._embedding_size}]"
            sql = (  # noqa: S608
                f"SELECT {columns}, 1.0 - array_cosine_distance(embedding, ?::{cast}) AS _score "
                f"FROM {self._table_name} WHERE {' AND '.join(filters)} "
                f"ORDER BY array_cosine_distance(embedding, ?::{cast}) ASC LIMIT {int(top_k)}"
            )
            params.extend([query_embedding, query_embedding])

        logger.debug("Executing vector search: %s", sql)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()

        return [self._row_to_document_with_score(row, select) for row in rows]

    def _normalize_id(self, id: DocumentId) -> bytes:
        """Convert id to bytes if needed."""
        if isinstance(id, bytes):
            return id
        if isinstance(id, str):
            return bytes.fromhex(id)
        raise ValueError(f"Invalid id type: {type(id)}")

    def _build_select(self, select: list[str] | None) -> str:
        """Build SELECT columns clause."""
        if select is None or not select:
            return "id, text, created, embedding, flux_task_id"

        valid_columns = {"id", "text", "created", "embedding", "flux_task_id"}
        columns = [col for col in select if col in valid_columns]

        if not columns:
            return "id, text, created, embedding, flux_task_id"

        return ", ".join(columns)

    def _build_where(self, node: WhereNode) -> tuple[str, list[Any]]:
        """Build WHERE clause from WhereNode AST."""
        if not node:
            return "1=1", []

        op = node[0]

        if op == "eq":
            field, value = node[1], node[2]
            return f"{self._safe_column(field)} = ?", [self._convert_value(field, value)]
        if op == "ne":
            field, value = node[1], node[2]
            return f"{self._safe_column(field)} != ?", [self._convert_value(field, value)]
        if op == "gt":
            field, value = node[1], node[2]
            return f"{self._safe_column(field)} > ?", [self._convert_value(field, value)]
        if op == "gte":
            field, value = node[1], node[2]
            return f"{self._safe_column(field)} >= ?", [self._convert_value(field, value)]
        if op == "lt":
            field, value = node[1], node[2]
            return f"{self._safe_column(field)} < ?", [self._convert_value(field, value)]
        if op == "lte":
            field, value = node[1], node[2]
            return f"{self._safe_column(field)} <= ?", [self._convert_value(field, value)]
        if op == "in":
            field, values = node[1], node[2]
            if not values:
                return "1=0", []
            placeholders = ", ".join("?" for _ in values)
            converted = [self._convert_value(field, value) for value in values]
            return f"{self._safe_column(field)} IN ({placeholders})", converted
        if op == "and":
            left_sql, left_params = self._build_where(node[1])
            right_sql, right_params = self._build_where(node[2])
            return f"({left_sql} AND {right_sql})", left_params + right_params
        if op == "or":
            left_sql, left_params = self._build_where(node[1])
            right_sql, right_params = self._build_where(node[2])
            return f"({left_sql} OR {right_sql})", left_params + right_params
        if op == "not":
            inner_sql, inner_params = self._build_where(node[1])
            return f"NOT ({inner_sql})", inner_params

        raise NotSupported(
            f"filter:{op}",
            repository=self._repo_name,
            details=f"Operator '{op}' not supported",
        )

    def _safe_column(self, field: str) -> str:
        """Validate and return safe column name."""
        valid_columns = {"id", "text", "created", "embedding", "flux_task_id"}
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
        if field == "embedding":
            return _normalize_embedding(value, self._embedding_size)
        return value

    def _build_order_by(self, order_by: list[str]) -> str:
        """Build ORDER BY clause."""
        clauses = []
        for field in order_by:
            if field.startswith("-"):
                clauses.append(f"{self._safe_column(field[1:])} DESC")
            else:
                clauses.append(f"{self._safe_column(field)} ASC")
        return ", ".join(clauses) if clauses else "id ASC"

    def _row_to_document(
        self,
        row: tuple,
        select: list[str] | None = None,
    ) -> Document:
        """Convert database row to Document dict."""
        if select is None or not select:
            return {
                "id": row[0].hex() if isinstance(row[0], bytes) else row[0],
                "text": row[1],
                "created": row[2].isoformat() if hasattr(row[2], "isoformat") else row[2],
                "embedding": list(row[3]) if row[3] is not None else None,
                "flux_task_id": row[4],
            }

        valid_columns = ["id", "text", "created", "embedding", "flux_task_id"]
        selected = [col for col in select if col in valid_columns]

        doc = {}
        for i, col in enumerate(selected):
            value = row[i]
            if col == "id" and isinstance(value, bytes):
                value = value.hex()
            elif col == "created" and hasattr(value, "isoformat"):
                value = value.isoformat()
            elif col == "embedding" and value is not None:
                value = list(value)
            doc[col] = value

        return doc

    def _row_to_document_with_score(
        self,
        row: tuple,
        select: list[str] | None = None,
    ) -> Document:
        """Convert a vector-search row to a Document including score."""
        doc = self._row_to_document(row[:-1], select)
        doc["_score"] = float(row[-1]) if row[-1] is not None else None
        return doc

    def close(self) -> None:
        """Close the DuckDB connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info("DuckDB repository connection closed")


__all__ = ["RawInputsRepository"]
