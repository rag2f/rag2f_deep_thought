from __future__ import annotations

import pytest
from rag2f.core.dto.indiana_jones_dto import RetrieveResult, ReturnMode

from rag2f_deep_thought.indiana_jones_hooks import indiana_jones_retrieve
from rag2f_deep_thought.plugin_context import set_plugin_id
from rag2f_deep_thought.repository_raw_inputs import RawInputsRepository


class FakeGetResult:
    def __init__(self, repository):
        self.repository = repository
        self.detail = None

    def is_ok(self) -> bool:
        return True


class FakeXFiles:
    def __init__(self, repository):
        self._repository = repository

    def execute_get(self, repository_id: str):
        del repository_id
        return FakeGetResult(self._repository)


class FakeConfigManager:
    def __init__(self, size: int):
        self._size = size

    def get_plugin_config(self, plugin_id: str, key: str | None = None, default=None):
        if plugin_id == "rag2f_openai_embedder" and key == "size":
            return self._size
        return default


class FakeOptimusPrime:
    def __init__(self, embedder):
        self._embedder = embedder

    def get_default(self):
        return self._embedder


class FakeRag2f:
    def __init__(self, repository, embedder, size: int):
        self.xfiles = FakeXFiles(repository)
        self.optimus_prime = FakeOptimusPrime(embedder)
        self.config_manager = FakeConfigManager(size)


def test_repository_allows_nullable_embedding_then_update():
    """Documents can be inserted with null embeddings and updated later."""
    repository = RawInputsRepository(embedding_size=3)

    repository.insert("01", {"text": "hello", "embedding": None, "flux_task_id": "task-1"})
    first = repository.get("01")
    assert first["embedding"] is None
    assert first["flux_task_id"] == "task-1"

    repository.update("01", {"embedding": [0.1, 0.2, 0.3]})
    updated = repository.get("01")
    assert updated["embedding"] == pytest.approx([0.1, 0.2, 0.3])
    assert updated["flux_task_id"] == "task-1"


def test_repository_vector_search_orders_by_similarity():
    """Vector search returns the closest vectors first."""
    repository = RawInputsRepository(embedding_size=3)
    repository.insert("01", {"text": "first", "embedding": [1.0, 0.0, 0.0]})
    repository.insert("02", {"text": "second", "embedding": [0.0, 1.0, 0.0]})

    results = repository.vector_search([0.9, 0.1, 0.0], top_k=2, select=["id", "text"])

    assert [item["id"] for item in results] == ["01", "02"]
    assert results[0]["_score"] > results[1]["_score"]


def test_indiana_jones_retrieve_uses_vector_search(static_embedder_factory):
    """Indiana Jones retrieval ranks documents using the query embedding."""
    repository = RawInputsRepository(embedding_size=3)
    repository.insert("01", {"text": "vector match", "embedding": [1.0, 0.0, 0.0]})
    repository.insert("02", {"text": "other", "embedding": [0.0, 1.0, 0.0]})

    embedder = static_embedder_factory({"needle": [1.0, 0.0, 0.0]}, size=3)
    rag2f = FakeRag2f(repository, embedder, size=3)
    result = RetrieveResult.success(query="needle")
    set_plugin_id("rag2f_deep_thought")

    output = indiana_jones_retrieve.function(
        result,
        "needle",
        2,
        ReturnMode.WITH_ITEMS,
        False,
        rag2f,
    )

    assert output.is_ok()
    assert [item.id for item in output.items] == ["01", "02"]
    assert output.items[0].score > output.items[1].score
