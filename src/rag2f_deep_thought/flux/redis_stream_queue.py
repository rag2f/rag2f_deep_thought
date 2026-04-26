"""Redis Stream queue backend for FluxCapacitor."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any

from rag2f.core.flux_capacitor.queue import BaseTaskQueue
from rag2f.core.flux_capacitor.task_models import (
    PayloadRef,
    TaskBackendCapabilities,
    TaskEnvelope,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RedisStreamQueueConfig:
    """Configuration for a Redis Stream task queue."""

    url: str | None = None
    stream_name: str = "rag2f:flux:tasks"
    group_name: str = "rag2f-flux-workers"
    max_len: int | None = 100_000
    block_ms: int | None = None


def _payload_to_json(payload_ref: PayloadRef | dict[str, Any] | None) -> str:
    """Serialize a payload reference into a Redis field."""
    if payload_ref is None:
        return ""
    payload = payload_ref.to_dict() if isinstance(payload_ref, PayloadRef) else dict(payload_ref)
    return json.dumps(payload, sort_keys=True)


def _payload_from_json(value: str | bytes | None) -> PayloadRef | None:
    """Deserialize a payload reference from a Redis field."""
    if not value:
        return None
    decoded = _decode(value)
    payload = json.loads(decoded)
    if not isinstance(payload, dict):
        return None
    return PayloadRef.from_mapping(payload)


def _decode(value: Any) -> str:
    """Decode Redis bytes when decode_responses is disabled."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _datetime_to_field(value: datetime | None) -> str:
    """Serialize a datetime to a Redis field."""
    return value.isoformat() if value is not None else ""


def _datetime_from_field(value: str | bytes | None) -> datetime | None:
    """Deserialize a datetime from a Redis field."""
    if not value:
        return None
    return datetime.fromisoformat(_decode(value))


class RedisStreamTaskQueue(BaseTaskQueue):
    """Flux task queue implemented with Redis Streams and consumer groups."""

    def __init__(self, config: RedisStreamQueueConfig, client: Any | None = None) -> None:
        """Initialize the queue and ensure its consumer group exists."""
        self._config = config
        self._client = client or self._client_from_config(config)
        self._reservations: dict[str, TaskEnvelope] = {}
        self._ensure_consumer_group()

    @property
    def capabilities(self) -> TaskBackendCapabilities:
        """Return queue backend capabilities."""
        return TaskBackendCapabilities(
            supports_ack=True,
            supports_delay=True,
            supports_reclaim=True,
            supports_ordering=True,
            supports_visibility_timeout=False,
        )

    def publish(self, envelope: TaskEnvelope) -> str | None:
        """Publish a task envelope to the Redis stream."""
        available_at = envelope.available_at or datetime.now(UTC)
        fields = self._envelope_to_fields(replace(envelope, available_at=available_at))
        kwargs: dict[str, Any] = {}
        if self._config.max_len is not None:
            kwargs["maxlen"] = self._config.max_len
            kwargs["approximate"] = True
        queue_ref = self._client.xadd(self._config.stream_name, fields, **kwargs)
        return _decode(queue_ref)

    def reserve(self, *, worker_id: str) -> TaskEnvelope | None:
        """Reserve the next available task for a worker."""
        response = self._client.xreadgroup(
            groupname=self._config.group_name,
            consumername=worker_id,
            streams={self._config.stream_name: ">"},
            count=1,
            block=self._config.block_ms,
        )
        if not response:
            return None

        _stream_name, messages = response[0]
        if not messages:
            return None

        message_id, fields = messages[0]
        reservation_ref = _decode(message_id)
        envelope = self._fields_to_envelope(fields, reservation_ref=reservation_ref)
        now = datetime.now(UTC)
        if envelope.available_at is not None and envelope.available_at > now:
            self._reservations[reservation_ref] = envelope
            self.release(reservation_ref, retry_at=envelope.available_at)
            return None

        reserved = replace(
            envelope,
            queue_ref=envelope.queue_ref or reservation_ref,
            reservation_ref=reservation_ref,
            available_at=None,
        )
        self._reservations[reservation_ref] = reserved
        return reserved

    def ack(self, reservation_ref: str) -> None:
        """Acknowledge successful processing of a reservation."""
        if not reservation_ref:
            return
        self._reservations.pop(reservation_ref, None)
        self._client.xack(self._config.stream_name, self._config.group_name, reservation_ref)

    def release(self, reservation_ref: str, *, retry_at: datetime | None = None) -> None:
        """Release a reservation by re-publishing the same logical task."""
        if not reservation_ref:
            return
        envelope = self._reservations.pop(reservation_ref, None)
        if envelope is not None:
            self.publish(
                replace(
                    envelope,
                    reservation_ref=None,
                    available_at=retry_at or datetime.now(UTC),
                )
            )
        self._client.xack(self._config.stream_name, self._config.group_name, reservation_ref)

    def _ensure_consumer_group(self) -> None:
        """Create the stream consumer group when it does not exist yet."""
        try:
            self._client.xgroup_create(
                self._config.stream_name,
                self._config.group_name,
                id="0",
                mkstream=True,
            )
        except Exception as exc:
            if "BUSYGROUP" not in str(exc):
                raise
            logger.debug(
                "Redis stream consumer group already exists (stream=%s, group=%s)",
                self._config.stream_name,
                self._config.group_name,
            )

    def _client_from_config(self, config: RedisStreamQueueConfig) -> Any:
        """Build a Redis client from queue configuration."""
        if not config.url:
            raise ValueError("Redis URL is required for RedisStreamTaskQueue")
        try:
            import redis
        except ImportError as exc:
            raise RuntimeError("Install the 'redis' package to use RedisStreamTaskQueue") from exc
        return redis.Redis.from_url(config.url, decode_responses=True)

    def _envelope_to_fields(self, envelope: TaskEnvelope) -> dict[str, str]:
        """Convert an envelope to Redis stream fields."""
        return {
            "task_id": envelope.task_id,
            "root_id": envelope.root_id,
            "parent_id": envelope.parent_id or "",
            "plugin_id": envelope.plugin_id,
            "hook": envelope.hook,
            "payload_ref": _payload_to_json(envelope.payload_ref),
            "queue_ref": envelope.queue_ref or "",
            "available_at": _datetime_to_field(envelope.available_at),
        }

    def _fields_to_envelope(self, fields: dict[Any, Any], *, reservation_ref: str) -> TaskEnvelope:
        """Convert Redis stream fields to a task envelope."""
        normalized = {_decode(key): value for key, value in fields.items()}
        queue_ref = _decode(normalized.get("queue_ref")) if normalized.get("queue_ref") else None
        return TaskEnvelope(
            task_id=_decode(normalized["task_id"]),
            root_id=_decode(normalized["root_id"]),
            parent_id=_decode(normalized["parent_id"]) if normalized.get("parent_id") else None,
            plugin_id=_decode(normalized["plugin_id"]),
            hook=_decode(normalized["hook"]),
            payload_ref=_payload_from_json(normalized.get("payload_ref")),
            reservation_ref=reservation_ref,
            queue_ref=queue_ref or reservation_ref,
            available_at=_datetime_from_field(normalized.get("available_at")),
        )
