from __future__ import annotations

import os
import sys

import pytest
from rich.traceback import install

from rag2f_deep_thought.plugin_context import reset_plugin_id

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
