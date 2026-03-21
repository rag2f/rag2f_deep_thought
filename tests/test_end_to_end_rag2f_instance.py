from __future__ import annotations

import pytest
from rag2f.core.dto.indiana_jones_dto import ReturnMode
from rag2f.core.rag2f import RAG2F
from rag2f.core.xfiles import QuerySpec


@pytest.mark.asyncio
async def test_end_to_end_instance_persists_and_searches_embeddings(
    monkeypatch,
    tmp_path,
    static_embedder_factory,
):
    """A real RAG2F instance stores embeddings during ingest and uses them in retrieval."""
    db_path = tmp_path / "end_to_end.duckdb"
    monkeypatch.setenv("RAG2F__RAG2F__EMBEDDER_DEFAULT", "fake_end_to_end_embedder")
    monkeypatch.setenv("RAG2F__PLUGINS__RAG2F_OPENAI_EMBEDDER__API_KEY", "test-key")
    monkeypatch.setenv(
        "RAG2F__PLUGINS__RAG2F_OPENAI_EMBEDDER__BASE_URL", "http://localhost:9999/v1"
    )
    monkeypatch.setenv("RAG2F__PLUGINS__RAG2F_OPENAI_EMBEDDER__MODEL", "test-model")
    monkeypatch.setenv("RAG2F__PLUGINS__RAG2F_OPENAI_EMBEDDER__SIZE", "3")

    config = {
        "rag2f": {
            "embedder_default": "fake_end_to_end_embedder",
        },
        "plugins": {
            "rag2f_deep_thought": {
                "db_path": str(db_path),
            },
            "rag2f_openai_embedder": {
                "api_key": "test-key",
                "base_url": "http://localhost:9999/v1",
                "model": "test-model",
                "size": 3,
            },
        },
    }

    rag2f = await RAG2F.create(config=config)
    assert rag2f.spock.get_rag2f_config("embedder_default") == "fake_end_to_end_embedder"
    rag2f.optimus_prime.register(
        "fake_end_to_end_embedder",
        static_embedder_factory(
            {
                "alpha document": [1.0, 0.0, 0.0],
                "beta document": [0.0, 1.0, 0.0],
                "alpha query": [0.9, 0.1, 0.0],
            },
            size=3,
        ),
    )

    first_insert = rag2f.johnny5.execute_handle_text_foreground("alpha document")
    second_insert = rag2f.johnny5.execute_handle_text_foreground("beta document")

    assert first_insert.is_ok()
    assert second_insert.is_ok()

    repository_result = rag2f.xfiles.execute_get("rag2f_deep_thought_raw_inputs")
    assert repository_result.is_ok()
    repository = repository_result.repository
    assert repository is not None

    documents = repository.find(
        QuerySpec(select=["text", "embedding"], order_by=["text"], limit=10)
    )
    documents_by_text = {document["text"]: document for document in documents}

    assert documents_by_text["alpha document"]["embedding"] == pytest.approx([1.0, 0.0, 0.0])
    assert documents_by_text["beta document"]["embedding"] == pytest.approx([0.0, 1.0, 0.0])

    retrieve = rag2f.indiana_jones.execute_retrieve("alpha query", k=2)
    assert retrieve.is_ok()
    assert [item.text for item in retrieve.items] == ["alpha document", "beta document"]
    assert retrieve.items[0].score > retrieve.items[1].score

    search = rag2f.indiana_jones.execute_search(
        "alpha query",
        k=2,
        return_mode=ReturnMode.WITH_ITEMS,
    )
    assert search.is_ok()
    assert search.response == "alpha document"
    assert [item.text for item in search.items or []] == ["alpha document", "beta document"]
