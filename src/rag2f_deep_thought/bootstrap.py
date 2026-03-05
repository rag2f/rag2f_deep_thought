import logging

from rag2f.core.morpheus.decorators.plugin_decorator import plugin
from rag2f.core.morpheus.plugin import Plugin
from rag2f.core.rag2f import RAG2F

from .plugin_context import get_plugin_id, set_plugin_id

logger = logging.getLogger(__name__)


# Repository ID constant for consistent access
POSTFIX_REPOSITORY_ID = "duckdb_raw_inputs"


@plugin
def activated(plugin: Plugin, rag2f_instance: RAG2F):
    """Bootstrap DuckDB text repository from plugin configuration.

    This hook initializes and registers the DuckDBTextRepository
    for text storage and deduplication.

    Configuration is retrieved using the plugin ID: 'rag2f_deep_thought'

    Optional configuration:
    - db_path: Path to DuckDB database file (default: ":memory:")
    - table_name: Table name for texts (default: "texts")
    - repo_name: Repository display name (default: "deep_thought_texts")

    Example JSON configuration:
    {
      "plugins": {
        "rag2f_deep_thought": {
          "db_path": "/path/to/data.duckdb",
          "table_name": "texts",
          "repo_name": "deep_thought_texts"
        }
      }
    }

    Example environment variables:
    RAG2F__PLUGINS__RAG2F_DEEP_THOUGHT__DB_PATH=/path/to/data.duckdb
    RAG2F__PLUGINS__RAG2F_DEEP_THOUGHT__TABLE_NAME=texts

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
        # Import repository (lazy import)
        from .duckdb_repository import DuckDBTextRepository

        # Extract configuration with defaults
        db_path = config.get("db_path", ":memory:")
        table_name = config.get("table_name", "texts")
        repo_name = config.get("repo_name", get_repository_id(rag2f_instance))

        # Create repository instance
        repository = DuckDBTextRepository(
            db_path=db_path,
            table_name=table_name,
            repo_name=repo_name,
        )

        # Register with metadata for searchability (Result Pattern)
        register_result = rag2f_instance.xfiles.execute_register(
            get_repository_id(rag2f_instance),
            repository,
            meta={
                "type": "duckdb",
                "domain": "texts",
                "plugin": plugin_id,
                "table": table_name,
            },
        )

        if register_result.is_ok():
            if register_result.created:
                logger.info(
                    "DuckDB repository registered as '%s' (db_path=%s, table=%s)",
                    get_repository_id(rag2f_instance),
                    db_path,
                    table_name,
                )
            else:
                logger.warning(
                    "DuckDB repository '%s' already registered (skipped)",
                    get_repository_id(rag2f_instance),
                )
        else:
            logger.error(
                "Failed to register DuckDB repository '%s': %s",
                get_repository_id(rag2f_instance),
                register_result.detail.message if register_result.detail else "Unknown error",
            )

    except ImportError as e:
        logger.error(
            "Failed to import DuckDBTextRepository. Ensure 'duckdb' package is installed: %s", e
        )
    except Exception as e:
        logger.error("Unexpected error bootstrapping DuckDB repository: %s", e)

    return


def get_repository_id(rag2f=None) -> str:
    """Get the repository ID for the DuckDB text repository.

    Returns:
        The repository ID string used for registration.
    """
    # Get plugin_id from RAG2F instance; it may not be necessary to use the rag2f instance if the plugin_id is already set in the context
    plugin_id = get_plugin_id(rag2f)

    return f"{plugin_id}_{POSTFIX_REPOSITORY_ID}"
