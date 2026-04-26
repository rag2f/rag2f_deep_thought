from __future__ import annotations

import os
import sys
from urllib.parse import urlsplit, urlunsplit

import pytest
from rich.traceback import install

from rag2f_deep_thought.plugin_context import reset_plugin_id

TEST_REDIS_DB = 10

# Enable readable tracebacks in development / test environments.
# Can be disabled with PYTEST_RICH=0
if os.getenv("PYTEST_RICH", "1") == "1":
    install(
        show_locals=True,  # show local variables for each frame
        width=None,  # use terminal width
        word_wrap=True,  # wrap long lines
        extra_lines=1,  # some context around lines
        suppress=["/usr/lib/python3", "site-packages"],  # hide "noisy" third-party frames
    )


# Ensure `src` is on sys.path so imports like `from rag2f.core...` resolve during tests.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if os.path.isdir(SRC):
    sys.path.insert(0, SRC)
sys.path.insert(0, ROOT)


class StaticEmbedder:
    """Simple deterministic embedder for tests."""

    def __init__(self, mapping: dict[str, list[float]], size: int):
        self._mapping = mapping
        self._size = size

    @property
    def size(self) -> int:
        """Return the embedding dimension."""
        return self._size

    def getEmbedding(self, text: str, *, normalize: bool = False) -> list[float]:
        """Return the predefined embedding for the given text."""
        del normalize
        return self._mapping[text]


def _redis_url_for_test_db(url: str, *, db: int = TEST_REDIS_DB) -> str:
    """Return a Redis URL rewritten to use the dedicated test database."""
    parsed = urlsplit(url)
    return urlunsplit((parsed.scheme, parsed.netloc, f"/{db}", parsed.query, parsed.fragment))


@pytest.fixture(scope="session")
def redis_test_url() -> str:
    """Return the Redis URL pinned to the dedicated test database."""
    base_url = os.getenv("REDIS_URL", "redis://:rag2f-devcontainer-redis@redis:6379/0")
    return _redis_url_for_test_db(base_url)


@pytest.fixture(scope="session", autouse=True)
def configure_test_redis_environment(redis_test_url: str):
    """Point test-time Redis consumers to the dedicated test database."""
    previous_redis_url = os.environ.get("REDIS_URL")
    previous_plugin_redis_url = os.environ.get("RAG2F__PLUGINS__RAG2F_DEEP_THOUGHT__REDIS_URL")

    os.environ["REDIS_URL"] = redis_test_url
    os.environ["RAG2F__PLUGINS__RAG2F_DEEP_THOUGHT__REDIS_URL"] = redis_test_url

    try:
        yield
    finally:
        if previous_redis_url is None:
            os.environ.pop("REDIS_URL", None)
        else:
            os.environ["REDIS_URL"] = previous_redis_url

        if previous_plugin_redis_url is None:
            os.environ.pop("RAG2F__PLUGINS__RAG2F_DEEP_THOUGHT__REDIS_URL", None)
        else:
            os.environ["RAG2F__PLUGINS__RAG2F_DEEP_THOUGHT__REDIS_URL"] = previous_plugin_redis_url


@pytest.fixture(scope="session")
def redis_test_client(redis_test_url: str):
    """Return a Redis client bound to DB 10 and keep it clean for the suite."""
    redis = pytest.importorskip("redis")
    client = redis.Redis.from_url(redis_test_url, decode_responses=True)
    try:
        client.ping()
    except redis.RedisError as exc:
        pytest.skip(f"Redis sidecar unavailable for tests: {exc}")

    client.flushdb()
    try:
        yield client
    finally:
        client.flushdb()
        client.close()


@pytest.fixture(autouse=True)
def reset_deep_thought_plugin_context():
    """Isolate plugin context between tests."""
    reset_plugin_id()
    yield
    reset_plugin_id()


@pytest.fixture
def static_embedder_factory():
    """Build deterministic embedders for tests."""

    def _factory(mapping: dict[str, list[float]], size: int) -> StaticEmbedder:
        return StaticEmbedder(mapping, size)

    return _factory
