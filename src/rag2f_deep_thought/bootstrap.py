import logging
import os
from contextlib import suppress

import duckdb
from rag2f.core.flux_capacitor import InMemoryTaskQueue
from rag2f.core.morpheus.decorators.plugin_decorator import plugin
from rag2f.core.morpheus.plugin import Plugin
from rag2f.core.rag2f import RAG2F

from .plugin_context import get_plugin_id, set_plugin_id

logger = logging.getLogger(__name__)


# Repository ID constant for consistent access
TABLE_RAW_INPUTS = "raw_inputs"
OPENAI_EMBEDDER_PLUGIN_ID = "rag2f_openai_embedder"
FLUX_TASK_STORE_TABLE = "flux_tasks"
FLUX_QUEUE_HOOK_RAW_INPUT_EMBEDDER = "raw_input_embedder"


def _config_bool(value, *, default: bool = False) -> bool:
    """Normalize bool-like configuration values."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _resolve_embedding_size(rag2f_instance: RAG2F) -> int | None:
    """Resolve embedding size from Spock configuration or the default embedder."""
    configured_size = rag2f_instance.config_manager.get_plugin_config(
        OPENAI_EMBEDDER_PLUGIN_ID, "size"
    )
    if configured_size is not None:
        return int(configured_size)

    try:
        return int(rag2f_instance.optimus_prime.get_default().size)
    except Exception:
        return None


def _check_vss_availability(db_path: str, embedding_size: int | None) -> bool:
    """Check whether DuckDB VSS can be enabled for this repository."""
    if embedding_size is None:
        return False

    connection = duckdb.connect(db_path)
    try:
        with suppress(Exception):
            connection.execute("INSTALL vss")
        connection.execute("LOAD vss")
    except Exception:
        logger.warning("VSS unavailable: using DuckDB exact vector search fallback.")
        return False
    finally:
        connection.close()

    return True


def _register_flux_backends(plugin_id: str, config: dict, rag2f_instance: RAG2F) -> None:
    """Register FluxCapacitor store and queue backends for async raw input embedding."""
    from .flux.duckdb_store import DuckDBTaskStore

    store_name = config.get("flux_store_name", f"{plugin_id}_duckdb_tasks")
    store_db_path = config.get("flux_store_db_path", ":memory:")
    store_table = config.get("flux_store_table", FLUX_TASK_STORE_TABLE)
    store = DuckDBTaskStore(db_path=store_db_path, table_name=store_table)
    rag2f_instance.flux_capacitor.register_store(store_name, store)
    rag2f_instance.flux_capacitor.set_default_store(store_name)

    queue_backend = str(config.get("flux_queue_backend", "redis")).strip().lower()
    queue_name = config.get("flux_queue_name", f"{plugin_id}_flux_queue")
    queue = None

    if queue_backend == "memory":
        queue = InMemoryTaskQueue()
    else:
        try:
            from .flux.redis_stream_queue import RedisStreamQueueConfig, RedisStreamTaskQueue

            redis_url = config.get("redis_url") or os.getenv("RAG2F_DEEP_THOUGHT_REDIS_URL")
            redis_url = redis_url or os.getenv("REDIS_URL")
            redis_config = RedisStreamQueueConfig(
                url=redis_url,
                stream_name=config.get("flux_stream_name", f"{plugin_id}:flux:tasks"),
                group_name=config.get("flux_consumer_group", f"{plugin_id}:flux:workers"),
                max_len=config.get("flux_stream_max_len", 100_000),
                block_ms=config.get("flux_stream_block_ms"),
            )
            queue = RedisStreamTaskQueue(redis_config)
        except Exception as exc:
            if _config_bool(config.get("flux_queue_require_redis")):
                raise
            logger.warning("Redis Flux queue unavailable; using in-memory queue: %s", exc)
            queue = InMemoryTaskQueue()

    rag2f_instance.flux_capacitor.register_queue(queue_name, queue)
    rag2f_instance.flux_capacitor.set_default_queue(queue_name)
    logger.info(
        "Flux backends registered (store=%s, queue=%s, queue_type=%s)",
        store_name,
        queue_name,
        type(queue).__name__,
    )


@plugin
def activated(plugin: Plugin, rag2f_instance: RAG2F):
    """Bootstrap DuckDB text repository from plugin configuration.

    This hook initializes and registers the RawInputsRepository
    for text storage and deduplication.

    Configuration is retrieved using the plugin ID: 'rag2f_deep_thought'

    Optional configuration:
    - db_path: Path to DuckDB database file (default: ":memory:")

    Example JSON configuration:
    {
      "plugins": {
        "rag2f_deep_thought": {
          "db_path": "/path/to/data.duckdb",
        }
      }
    }

    Example environment variables:
    RAG2F__PLUGINS__RAG2F_DEEP_THOUGHT__DB_PATH=/path/to/data.duckdb

    Args:
        repositories_registry: Dictionary to populate with repositories
        rag2f: RAG2F instance providing access to Spock configuration

    Returns:
        Updated repositories_registry with DuckDB repository
    """

    plugin_id = plugin.id
    set_plugin_id(plugin_id)

    logger.debug(f"🔍 Plugin '{plugin_id}' ovveride activate execution: {activated}")

    config = rag2f_instance.spock.get_plugin_config(plugin_id)

    if config is None:
        config = {}

    try:
        _register_flux_backends(plugin_id, config, rag2f_instance)

        # Import repository (lazy import)
        from .repository_raw_inputs import RawInputsRepository

        # Extract configuration with defaults
        db_path = config.get("db_path", ":memory:")
        table_name = TABLE_RAW_INPUTS
        embedding_size = _resolve_embedding_size(rag2f_instance)
        vss_available = _check_vss_availability(db_path, embedding_size)

        # Create repository instance
        repository = RawInputsRepository(
            db_path=db_path,
            table_name=table_name,  # Use a fixed table name for consistency; can be made configurable if needed
            embedding_size=embedding_size,
            enable_hnsw=vss_available,
        )

        # Register with metadata for searchability (Result Pattern)
        register_result = rag2f_instance.xfiles.execute_register(
            get_repository_id(rag2f_instance, table_name),
            repository,
            meta={
                "type": "duckdb",
                "domain": "texts",
                "plugin": plugin_id,
                "table": table_name,
                "embedding_size": embedding_size,
                "vector_search": repository.capabilities().vector_search.supported,
            },
        )

        if register_result.is_ok():
            if register_result.created:
                logger.info(
                    "DuckDB repository registered as '%s' (db_path=%s, table=%s)",
                    get_repository_id(rag2f_instance, table_name),
                    db_path,
                    table_name,
                )
            else:
                logger.warning(
                    "DuckDB repository '%s' already registered (skipped)",
                    get_repository_id(rag2f_instance, table_name),
                )
        else:
            logger.error(
                "Failed to register DuckDB repository '%s': %s",
                get_repository_id(rag2f_instance, table_name),
                register_result.detail.message if register_result.detail else "Unknown error",
            )

    except ImportError as e:
        logger.error(
            "Failed to import RawInputsRepository. Ensure 'duckdb' package is installed: %s", e
        )
    except Exception as e:
        logger.error("Unexpected error bootstrapping DuckDB repository: %s", e)

    return


def get_repository_id(rag2f: RAG2F, table_name: str) -> str:
    """Get the repository ID for the DuckDB text repository.

    Args:
        rag2f: The RAG2F instance, must not be None.
        table_name: The name of the table, must not be None.

    Returns:
        The repository ID string used for registration.

    Raises:
        ValueError: If rag2f is None or table_name is None.
    """
    if rag2f is None:
        raise ValueError("The 'rag2f' instance must not be None.")
    if table_name is None:
        raise ValueError("The 'table_name' must not be None.")

    # Get plugin_id from RAG2F instance; it may not be necessary to use the rag2f instance if the plugin_id is already set in the context
    plugin_id = get_plugin_id(rag2f)

    return f"{plugin_id}_{table_name}"
