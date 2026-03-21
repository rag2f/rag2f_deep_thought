# rag2f_deep_thought

DuckDB-backed RAG2F plugin for raw text storage, embeddings, and vector retrieval.

## What it does

- stores ingested raw inputs in DuckDB
- generates and persists embeddings during foreground ingest
- retrieves results with vector similarity search through Indiana Jones hooks
- falls back to exact DuckDB vector search when HNSW/VSS is unavailable

## Test configuration

The project uses `pytest` with tests discovered from the [tests](tests) folder.

Current configuration lives in [pyproject.toml](pyproject.toml):

- `testpaths = ["tests"]`
- `python_files = ["test_*.py"]`
- `asyncio_mode = "strict"`

Run tests with:

```bash
pytest
```

## VSS and fallback

`vss` is a DuckDB extension, not a Python package dependency to add to `pyproject.toml`.

At plugin bootstrap:

- DuckDB checks whether `vss` can be installed and loaded
- if available, the repository enables an HNSW index on embeddings
- if unavailable, the plugin logs a warning and falls back automatically

Bootstrap warning:

```text
VSS unavailable: using DuckDB exact vector search fallback.
```

Fallback behavior:

- no HNSW index is created
- embeddings are still stored normally
- retrieval still uses vector similarity
- DuckDB executes exact similarity ordering with `array_cosine_distance` or `list_cosine_similarity`

## CI/CD Workflows

- `.github/workflows/ci-dev-testpypi.yml`: validates structure, builds, and publishes dev versions to TestPyPI
- `.github/workflows/release-tags.yml`: builds from tags, publishes to PyPI, creates GitHub Releases