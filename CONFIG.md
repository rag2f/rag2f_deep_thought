# Plugin Configuration (Spock)


This plugin now reads its configuration through the centralized **Spock** system of RAG2F.
The plugin configuration must be placed in the main configuration file (or via environment variables) under the `plugins.<plugin_id>` node.

Note: The APIs in this repository expect the plugin to retrieve the configuration using the `plugin_id` (e.g. `rag2f_deep_thought`) via `rag2f.spock.get_plugin_config(plugin_id)`.

## Where to put the configuration

In the main configuration file (e.g. `config.json`), the plugin section should have this structure:

```json
{
  "plugins": {
    "rag2f_deep_thought": {
      "db_path": ":memory:",
      "flux_store_db_path": ":memory:",
      "flux_queue_backend": "redis",
      "redis_url": "redis://:rag2f-devcontainer-redis@redis:6379/0",
      "flux_stream_name": "rag2f_deep_thought:flux:tasks",
      "flux_consumer_group": "rag2f_deep_thought:flux:workers"
    }
  }
}
```

In this example, the `plugin_id` is `rag2f_deep_thought` and Spock will load the configuration when the plugin requests it.

## Environment variables (Spock)

Spock also supports environment variables. The format is based on double underscore prefixes to represent the hierarchy.

Examples to set the plugin configuration via ENV:

```bash
export RAG2F__PLUGINS__RAG2F_DEEP_THOUGHT__DB_PATH=":memory:"
export RAG2F__PLUGINS__RAG2F_DEEP_THOUGHT__FLUX_STORE_DB_PATH=":memory:"
export RAG2F__PLUGINS__RAG2F_DEEP_THOUGHT__FLUX_QUEUE_BACKEND="redis"
export RAG2F__PLUGINS__RAG2F_DEEP_THOUGHT__REDIS_URL="redis://:rag2f-devcontainer-redis@redis:6379/0"
```

Spock will parse types (int, float, bool, JSON) whenever possible.

## Source priorities

1. **Environment Variables** (highest priority)
2. **JSON files** (config.json passed to RAG2F)
3. **Default values in code** (lowest priority)

## Example: how the plugin accesses its configuration

In the code, the plugin retrieves its configuration like this:

```python
plugin_cfg = rag2f.spock.get_plugin_config("rag2f_deep_thought")
```

After obtaining `plugin_cfg`, the plugin can validate required fields and raise a clear error if any are missing.

### Optional parameters

- `db_path`: DuckDB database path for raw inputs. Defaults to `:memory:`.
- `flux_store_db_path`: DuckDB database path for Flux task state. Defaults to `:memory:`.
- `flux_store_name`: Registry name for the Flux store.
- `flux_store_table`: DuckDB table used by the Flux store. Defaults to `flux_tasks`.
- `flux_queue_backend`: `redis` or `memory`. Defaults to `redis`.
- `redis_url`: Redis connection URL used by the Redis Stream queue.
- `flux_stream_name`: Redis Stream key used for Flux task delivery.
- `flux_consumer_group`: Redis consumer group used by Flux workers.
- `flux_stream_max_len`: Approximate max stream length. Defaults to `100000`.
- `flux_queue_require_redis`: When true, activation fails if Redis cannot be configured instead of falling back to memory.
- `dedup_key`: Hex string, bytes-like value, or JSON list of integers used to derive deterministic input ids.


