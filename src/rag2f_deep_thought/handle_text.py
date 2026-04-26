import hashlib
import logging
import re
import unicodedata
from datetime import datetime
from typing import Any

from rag2f.core.morpheus.decorators import hook
from rag2f.core.rag2f import RAG2F
from rag2f.core.xfiles import QuerySpec, eq

from .bootstrap import FLUX_QUEUE_HOOK_RAW_INPUT_EMBEDDER, TABLE_RAW_INPUTS, get_repository_id
from .plugin_context import get_plugin_id

logger = logging.getLogger(__name__)
OPENAI_EMBEDDER_PLUGIN_ID = "rag2f_openai_embedder"


# Collassa *qualsiasi* whitespace Unicode (spazi, tab, newline, righe vuote, ecc.)
_WS_ALL = re.compile(r"\s+", flags=re.UNICODE)
_DEDUP_KEY = bytes.fromhex("8c323a051fc680456c2409727cee0f4edd0609062600a51fbdca7edc766a2f9f")


def _resolve_embedding_size(rag2f: RAG2F) -> int | None:
    """Resolve the configured embedding size from Spock or the active embedder."""
    configured_size = rag2f.config_manager.get_plugin_config(OPENAI_EMBEDDER_PLUGIN_ID, "size")
    if configured_size is not None:
        return int(configured_size)

    try:
        return int(rag2f.optimus_prime.get_default().size)
    except Exception:
        return None


def _generate_embedding(text: str, rag2f: RAG2F) -> list[float] | None:
    """Generate an embedding for text using the default embedder when available."""
    try:
        embedder = rag2f.optimus_prime.get_default()
    except Exception as exc:
        logger.info("Default embedder unavailable, storing null embedding: %s", exc)
        return None

    embedding = list(embedder.getEmbedding(text))
    expected_size = _resolve_embedding_size(rag2f)
    if expected_size is not None and len(embedding) != expected_size:
        logger.warning(
            "Embedding discarded because size mismatch (expected=%d, got=%d)",
            expected_size,
            len(embedding),
        )
        return None
    return embedding


def _generate_embedding_or_raise(text: str, rag2f: RAG2F) -> list[float]:
    """Generate an embedding or raise so Flux marks the task as failed."""
    embedding = _generate_embedding(text, rag2f)
    if embedding is None:
        raise RuntimeError("Embedding generation returned no vector")
    return embedding


def _backfill_embedding(repository: Any, id: str, embedding: list[float] | None) -> None:
    """Update an existing document embedding when it was previously null."""
    if embedding is None:
        return

    try:
        existing = repository.get(id, select=["embedding"])
        if existing.get("embedding") is None:
            repository.update(id, {"embedding": embedding})
    except Exception as exc:
        logger.warning("Failed to backfill embedding for id=%s: %s", id, exc)


def _coerce_dedup_key(value) -> bytes:
    """Coerce `dedup_key` from Spock config into bytes.

    Supported:
    - bytes/bytearray/memoryview
    - hex string (even-length)
    - list[int] (0-255) from JSON
    """
    if value is None:
        return _DEDUP_KEY
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, str):
        s = value.strip()
        # accept hex strings (common for env/config)
        try:
            return bytes.fromhex(s)
        except ValueError as e:
            raise ValueError("Invalid dedup_key: expected hex string or bytes-like value") from e
    if isinstance(value, list) and all(isinstance(x, int) for x in value):
        return bytes(value)
    raise TypeError(
        f"Invalid dedup_key type: {type(value).__name__}. Expected bytes, hex str, or list[int]."
    )


def dedup_id(text: str, dedup_key: bytes, digest_size: int = 16) -> bytes:
    """
    Normalizza testo per dedup "flat":
    - normalizza newline (CRLF/CR -> LF)
    - Unicode normalization NFC (canonical composition)
    - case-insensitive robusto con casefold()
    - collassa ogni sequenza di whitespace in un singolo spazio
    Ritorna hash con protezione DoS blake2b.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text)
    text = text.casefold()
    text = _WS_ALL.sub(" ", text).strip()
    data = text.encode("utf-8", "surrogatepass")
    # use blake2b for DoS protection and speed
    return hashlib.blake2b(data, digest_size=digest_size, key=dedup_key).digest()


@hook("get_id_input_text", priority=10)
def get_id_input_text(id, text, rag2f):
    logger.debug(f"Hook object: {get_id_input_text}")
    plugin_id = get_plugin_id(rag2f)
    config = rag2f.spock.get_plugin_config(plugin_id)
    dedup_key = _coerce_dedup_key(config.get("dedup_key"))
    if dedup_key == _DEDUP_KEY:
        logger.warning(
            "Using default dedup_key, consider setting a custom one in plugin config for better security es: RAG2F__PLUGINS__RAG2F_DEEP_THOUGHT__DEDUP_KEY="
        )
    id = dedup_id(text, dedup_key).hex()
    return id


def _enqueue_raw_input_embedding(rag2f: RAG2F, repository_id: str, track_id: str) -> str:
    """Enqueue an asynchronous embedding task for a raw input."""
    plugin_id = get_plugin_id(rag2f)
    return rag2f.flux_capacitor.enqueue(
        plugin_id=plugin_id,
        hook=FLUX_QUEUE_HOOK_RAW_INPUT_EMBEDDER,
        payload_ref={
            "repository": repository_id,
            "id": track_id,
            "meta": {"track_id": track_id},
        },
    )


@hook("check_duplicated_input_text", priority=10)
def check_duplicated_input_text(duplicated, id, text, rag2f):
    logger.debug(f"Hook object: {check_duplicated_input_text}")
    repository_id = get_repository_id(rag2f, TABLE_RAW_INPUTS)
    # Get repository from XFile using Result Pattern
    get_result = rag2f.xfiles.execute_get(repository_id)
    if not get_result.is_ok() or get_result.repository is None:
        msg = get_result.detail.message if get_result.detail else "Unknown error"
        logger.error("Failed to get repository '%s': %s", repository_id, msg)
        return False

    repository = get_result.repository

    # Use QuerySpec with select=["id"] to check existence efficiently using index
    # Filter by id (hex string) using eq operator
    query = QuerySpec(
        select=["id"],
        where=eq("id", id),
        limit=1,
    )

    results = repository.find(query)
    duplicated = len(results) > 0

    logger.debug("Duplicate check for id=%s: %s", id, duplicated)
    return duplicated


@hook("handle_text_foreground", priority=10)
def handle_text_foreground(done, id, text, rag2f: RAG2F):
    logger.debug(f"Hook object: {handle_text_foreground}")
    repository_id = get_repository_id(rag2f, TABLE_RAW_INPUTS)
    # Get repository from XFile using Result Pattern
    get_result = rag2f.xfiles.execute_get(repository_id)
    if not get_result.is_ok() or get_result.repository is None:
        msg = get_result.detail.message if get_result.detail else "Unknown error"
        logger.error("Failed to get repository '%s': %s", repository_id, msg)
        return False

    repository = get_result.repository

    # Timestamp corrente
    created = datetime.now()

    # Prepare document
    document = {
        "text": text,
        "created": created,
    }
    # Insert using repository CRUD (id is hex string)
    try:
        repository.insert(id, document)
        flux_task_id = _enqueue_raw_input_embedding(rag2f, repository_id, id)
        repository.update(id, {"flux_task_id": flux_task_id})
        logger.debug("Stored text with id: %s", id)
        done = True
    except Exception as e:
        # AlreadyExists is handled gracefully - text already stored
        from rag2f.core.xfiles import AlreadyExists

        if isinstance(e, AlreadyExists):
            logger.debug("Text already exists: %s", id)
            _backfill_embedding(repository, id, document.get("embedding"))
            done = True
        else:
            logger.error("Error storing text: %s", e)
            done = False
    return done


@hook(FLUX_QUEUE_HOOK_RAW_INPUT_EMBEDDER, priority=10)
def raw_input_embedder(context=None, payload_ref=None, rag2f: RAG2F | None = None, **kwargs):
    """Consume a Flux task and backfill the raw input embedding."""
    del kwargs
    if rag2f is None and context is not None:
        rag2f = context.rag2f
    if rag2f is None:
        raise RuntimeError("RAG2F instance is required to embed raw input")
    if payload_ref is None and context is not None:
        payload_ref = context.task.payload_mapping()
    if not payload_ref:
        raise RuntimeError("Flux payload_ref is required to embed raw input")

    repository_id = payload_ref.get("repository")
    track_id = payload_ref.get("id")
    if not repository_id or not track_id:
        raise RuntimeError("Flux payload_ref must include repository and id")

    get_result = rag2f.xfiles.execute_get(repository_id)
    if not get_result.is_ok() or get_result.repository is None:
        msg = get_result.detail.message if get_result.detail else "Unknown error"
        raise RuntimeError(f"Failed to get repository '{repository_id}': {msg}")

    repository = get_result.repository
    document = repository.get(track_id, select=["text", "embedding"])
    if document.get("embedding") is not None:
        logger.debug("Raw input already has embedding (id=%s)", track_id)
        return {"embedded": False, "reason": "already_present", "id": track_id}

    embedding = _generate_embedding_or_raise(str(document["text"]), rag2f)
    repository.update(track_id, {"embedding": embedding})
    logger.debug("Raw input embedding stored (id=%s)", track_id)
    return {"embedded": True, "id": track_id}
