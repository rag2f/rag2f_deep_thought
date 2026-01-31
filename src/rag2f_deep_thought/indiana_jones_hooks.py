import logging

from rag2f.core.dto.indiana_jones_dto import RetrievedItem, ReturnMode
from rag2f.core.dto.result_dto import StatusCode, StatusDetail
from rag2f.core.morpheus.decorators import hook
from rag2f.core.xfiles import QuerySpec

from .bootstrap_repository import get_repository_id

logger = logging.getLogger(__name__)


def _simple_lexical_score(query: str, text: str) -> float:
    """Compute a simple lexical relevance score.

    This plugin stores raw text only (no embeddings). For the tutorial example,
    we implement a minimal, deterministic scoring:
    - case-insensitive substring match
    - score = number of occurrences of query in text

    Args:
        query: Query string.
        text: Candidate document text.

    Returns:
        A relevance score (0 means no match).
    """
    needle = (query or "").casefold().strip()
    if not needle:
        return 0.0

    haystack = (text or "").casefold()
    if needle not in haystack:
        return 0.0

    return float(haystack.count(needle))


@hook("indiana_jones_retrieve", priority=10)
def indiana_jones_retrieve(
    result, query: str, k: int, return_mode: ReturnMode, for_synthesize: bool, rag2f
):
    """Retrieve matching items using a lexical scan.

    This is intentionally simple: it scans stored texts in DuckDB and returns
    the top-k matches by a naive lexical score.
    """
    del return_mode, for_synthesize  # Unused in this simple lexical implementation
    repository_id = get_repository_id(rag2f)
    get_result = rag2f.xfiles.execute_get(repository_id)
    if not get_result.is_ok() or get_result.repository is None:
        msg = get_result.detail.message if get_result.detail else "Unknown error"
        logger.error("Failed to get repository '%s': %s", repository_id, msg)
        return result

    repository = get_result.repository

    # Pull candidate documents. For a demo plugin this is fine.
    # (No server-side LIKE filter is exposed via QuerySpec in this repository.)
    query_all = QuerySpec(select=["id", "text", "created"], limit=10000)
    try:
        docs = repository.find(query_all)
    except Exception as e:
        logger.error("Repository find() failed: %s", e)
        return result

    scored: list[tuple[float, dict]] = []
    for doc in docs:
        score = _simple_lexical_score(query, doc.get("text", ""))
        if score > 0:
            scored.append((score, doc))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    top = scored[: max(int(k), 0)] if k is not None else scored

    result.items = [
        RetrievedItem(
            id=str(doc.get("id", "")),
            text=str(doc.get("text", "")),
            metadata={"repository_id": repository_id, "created": doc.get("created")},
            score=float(score),
        )
        for score, doc in top
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
