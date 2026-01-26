import hashlib
import logging
import re
import unicodedata
from datetime import datetime

from rag2f.core.morpheus.decorators import hook
from rag2f.core.rag2f import RAG2F
from rag2f.core.xfiles import QuerySpec, eq

from .bootstrap_repository import get_repository_id
from .plugin_context import get_plugin_id

logger = logging.getLogger(__name__)


# Collassa *qualsiasi* whitespace Unicode (spazi, tab, newline, righe vuote, ecc.)
_WS_ALL = re.compile(r"\s+", flags=re.UNICODE)
_DEDUP_KEY = bytes.fromhex("8c323a051fc680456c2409727cee0f4edd0609062600a51fbdca7edc766a2f9f")


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


@hook("check_duplicated_input_text", priority=10)
def check_duplicated_input_text(duplicated, id, text, rag2f):
    logger.debug(f"Hook object: {check_duplicated_input_text}")
    repository_id = get_repository_id(rag2f)
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
    repository_id = get_repository_id(rag2f)
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
        logger.debug("Stored text with id: %s", id)
        done = True
    except Exception as e:
        # AlreadyExists is handled gracefully - text already stored
        from rag2f.core.xfiles import AlreadyExists

        if isinstance(e, AlreadyExists):
            logger.debug("Text already exists: %s", id)
            done = True
        else:
            logger.error("Error storing text: %s", e)
            done = False

    # default_embedder = rag2f.optimus_prime.get_default()
    # emb = default_embedder.getEmbedding(text)
    return done
