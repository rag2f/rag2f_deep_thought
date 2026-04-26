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

## Devcontainer services

The devcontainer starts two services through Docker Compose:

- `workspace`: the existing Python 3.12 development container with the same bootstrap scripts and `.venv` volume
- `redis`: a Redis 8 sidecar pinned to `redis:8.6.2`, which is the latest stable Redis 8 tag at the time of writing

At startup, `initializeCommand` runs `scripts/write-devcontainer-compose-env.sh` on the host and generates `.devcontainer/.env` from the current workspace path and basename. This keeps the Docker Compose pattern reusable across repositories while preserving the original lifecycle commands in `devcontainer.json`.

Inside the devcontainer, Redis is already available at:

```text
redis://:rag2f-devcontainer-redis@redis:6379/0
```

The shared Redis settings live in `.devcontainer/devcontainer.env`. The generated workspace-specific Compose variables live in `.devcontainer/.env`. After pulling changes to the devcontainer configuration, rebuild the container so Compose recreates the sidecar and refreshes the generated workspace settings.