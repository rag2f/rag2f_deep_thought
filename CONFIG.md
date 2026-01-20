# Plugin Configuration (Spock)


This plugin now reads its configuration through the centralized **Spock** system of RAG2F.
The plugin configuration must be placed in the main configuration file (or via environment variables) under the `plugins.<plugin_id>` node.

Note: The APIs in this repository expect the plugin to retrieve the configuration using the `plugin_id` (e.g. `rag2f_plugin_template`) via `rag2f.spock.get_plugin_config(plugin_id)`.

## Where to put the configuration

In the main configuration file (e.g. `config.json`), the plugin section should have this structure:

```json
{
  "plugins": {
    "rag2f_plugin_template": {
      "size": 8,
      "seed": "optional-seed"
    }
  }
}
```

In this example, the `plugin_id` is `rag2f_plugin_template` and Spock will load the configuration when the plugin requests it.

## Environment variables (Spock)

Spock also supports environment variables. The format is based on double underscore prefixes to represent the hierarchy.

Examples to set the plugin configuration via ENV:

```bash
export RAG2F__PLUGINS__RAG2F_PLUGIN_TEMPLATE__SIZE="8"
export RAG2F__PLUGINS__RAG2F_PLUGIN_TEMPLATE__SEED="optional-seed"
```

Spock will parse types (int, float, bool, JSON) whenever possible.

## Source priorities

1. **Environment Variables** (highest priority)
2. **JSON files** (config.json passed to RAG2F)
3. **Default values in code** (lowest priority)

## Example: how the plugin accesses its configuration

In the code, the plugin retrieves its configuration like this:

```python
plugin_cfg = rag2f.spock.get_plugin_config("rag2f_plugin_template")
```

After obtaining `plugin_cfg`, the plugin can validate required fields and raise a clear error if any are missing.

### Required parameters

- `size`: Output vector length (int > 0)

### Optional parameters

- `seed`: String mixed into deterministic generation


