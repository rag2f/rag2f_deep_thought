from __future__ import annotations

import pytest
from rag2f.core.flux_capacitor.task_models import TaskEnvelope

from rag2f_deep_thought.flux.redis_stream_queue import (
    RedisStreamQueueConfig,
    RedisStreamTaskQueue,
)


class FakeRedisStreamClient:
    """Minimal Redis Stream client for queue contract testing."""

    def __init__(self) -> None:
        self.messages = []
        self.next_index = 0
        self.acked = []

    def xgroup_create(self, stream_name, group_name, *, id, mkstream):
        del stream_name, group_name, id, mkstream

    def xadd(self, stream_name, fields, **kwargs):
        del stream_name, kwargs
        message_id = f"{len(self.messages) + 1}-0"
        self.messages.append((message_id, fields))
        return message_id

    def xreadgroup(self, *, groupname, consumername, streams, count, block):
        del groupname, consumername, count, block
        if self.next_index >= len(self.messages):
            return []
        stream_name = next(iter(streams.keys()))
        message = self.messages[self.next_index]
        self.next_index += 1
        return [(stream_name, [message])]

    def xack(self, stream_name, group_name, reservation_ref):
        del stream_name, group_name
        self.acked.append(reservation_ref)


def test_redis_stream_queue_round_trips_task_envelope():
    """Redis Stream queue preserves task routing and payload metadata."""
    client = FakeRedisStreamClient()
    queue = RedisStreamTaskQueue(
        RedisStreamQueueConfig(
            stream_name="test:flux:tasks",
            group_name="test:flux:workers",
            max_len=None,
        ),
        client=client,
    )

    queue_ref = queue.publish(
        TaskEnvelope(
            task_id="task-1",
            root_id="task-1",
            parent_id=None,
            plugin_id="rag2f_deep_thought",
            hook="raw_input_embedder",
            payload_ref={
                "repository": "rag2f_deep_thought_raw_inputs",
                "id": "doc-1",
                "meta": {"track_id": "doc-1"},
            },
        )
    )

    reserved = queue.reserve(worker_id="worker-1")

    assert reserved is not None
    assert reserved.task_id == "task-1"
    assert reserved.queue_ref == queue_ref
    assert reserved.reservation_ref == queue_ref
    assert reserved.plugin_id == "rag2f_deep_thought"
    assert reserved.hook == "raw_input_embedder"
    assert reserved.payload_ref.repository == "rag2f_deep_thought_raw_inputs"
    assert reserved.payload_ref.id == "doc-1"

    queue.ack(reserved.reservation_ref)

    assert client.acked == [queue_ref]


@pytest.mark.usefixtures("redis_test_client")
def test_redis_stream_queue_round_trips_task_envelope_with_sidecar(
    redis_test_client,
    redis_test_url,
):
    """Redis Stream queue works against the real sidecar using DB 10."""
    queue = RedisStreamTaskQueue(
        RedisStreamQueueConfig(
            url=redis_test_url,
            stream_name="test:flux:tasks:db10",
            group_name="test:flux:workers:db10",
            max_len=None,
        )
    )

    queue_ref = queue.publish(
        TaskEnvelope(
            task_id="task-real-1",
            root_id="task-real-1",
            parent_id=None,
            plugin_id="rag2f_deep_thought",
            hook="raw_input_embedder",
            payload_ref={
                "repository": "rag2f_deep_thought_raw_inputs",
                "id": "doc-real-1",
                "meta": {"track_id": "doc-real-1"},
            },
        )
    )

    reserved = queue.reserve(worker_id="worker-sidecar")

    assert reserved is not None
    assert reserved.task_id == "task-real-1"
    assert reserved.queue_ref == queue_ref
    assert reserved.reservation_ref == queue_ref
    assert reserved.payload_ref is not None
    assert reserved.payload_ref.repository == "rag2f_deep_thought_raw_inputs"
    assert reserved.payload_ref.id == "doc-real-1"

    queue.ack(reserved.reservation_ref)

    assert redis_test_client.exists("test:flux:tasks:db10") == 1
