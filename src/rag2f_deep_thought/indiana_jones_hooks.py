import logging

from rag2f.core.dto.indiana_jones_dto import RetrievedItem, ReturnMode
from rag2f.core.dto.result_dto import StatusCode, StatusDetail
from rag2f.core.morpheus.decorators import hook

from .bootstrap import TABLE_RAW_INPUTS, get_repository_id

logger = logging.getLogger(__name__)
OPENAI_EMBEDDER_PLUGIN_ID = "rag2f_openai_embedder"


def _resolve_query_embedding(query: str, rag2f) -> list[float] | None:
    """Build the query embedding using the default embedder."""
    try:
        embedder = rag2f.optimus_prime.get_default()
    except Exception as exc:
        logger.error("Default embedder unavailable for retrieval: %s", exc)
        return None

    try:
        embedding = list(embedder.getEmbedding(query))
    except Exception as exc:
        logger.error("Failed to generate query embedding: %s", exc)
        return None

    configured_size = rag2f.config_manager.get_plugin_config(OPENAI_EMBEDDER_PLUGIN_ID, "size")
    if configured_size is not None and len(embedding) != int(configured_size):
        logger.error(
            "Query embedding size mismatch (expected=%s, got=%d)",
            configured_size,
            len(embedding),
        )
        return None
    return embedding


@hook("indiana_jones_retrieve", priority=10)
def indiana_jones_retrieve(
    result, query: str, k: int, return_mode: ReturnMode, for_synthesize: bool, rag2f
):
    """Retrieve matching items using vector similarity search."""
    del return_mode, for_synthesize
    repository_id = get_repository_id(rag2f, TABLE_RAW_INPUTS)
    get_result = rag2f.xfiles.execute_get(repository_id)
    if not get_result.is_ok() or get_result.repository is None:
        msg = get_result.detail.message if get_result.detail else "Unknown error"
        logger.error("Failed to get repository '%s': %s", repository_id, msg)
        return result

    repository = get_result.repository
    query_embedding = _resolve_query_embedding(query, rag2f)
    if query_embedding is None:
        result.status = "error"
        result.detail = StatusDetail(
            code=StatusCode.INVALID,
            message="Default embedder is not available or returned an invalid vector",
        )
        return result

    try:
        docs = repository.vector_search(
            query_embedding,
            top_k=max(int(k), 0),
            select=["id", "text", "created"],
        )
    except Exception as e:
        logger.error("Repository vector_search() failed: %s", e)
        return result

    result.items = [
        RetrievedItem(
            id=str(doc.get("id", "")),
            text=str(doc.get("text", "")),
            metadata={"repository_id": repository_id, "created": doc.get("created")},
            score=float(doc.get("_score")) if doc.get("_score") is not None else None,
        )
        for doc in docs
    ]

    if not result.items:
        result.status = "error"
        result.detail = StatusDetail(code=StatusCode.NO_RESULTS, message="No results")

    return result


@hook("indiana_jones_synthesize", priority=10)
def indiana_jones_synthesize(result, retrieve_result, return_mode: ReturnMode, kwargs, rag2f):
    """Synthesize a response from retrieved items.

    Without a generator/LLM backend, we "synthesize" by returning the best
    matching item text. The retrieval is already done by the framework.
    """
    del kwargs, rag2f  # Unused in this simple implementation

    items = retrieve_result.items
    if not items:
        result.status = "error"
        result.detail = StatusDetail(code=StatusCode.NO_RESULTS, message="No results")
        return result

    result.response = items[0].text
    result.used_source_ids = [item.id for item in items]

    if return_mode == ReturnMode.WITH_ITEMS:
        result.items = items

    return result
