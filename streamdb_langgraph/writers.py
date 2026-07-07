"""Ready-made :class:`StateWriterProtocol` implementations.

The translator writes through the ``upsert`` / ``delete`` / ``close`` surface of
:class:`~streamdb_langgraph.state_protocol.StateWriterProtocol`. A production
consumer implements that surface against a durable stream; these two writers
cover the common non-production cases:

* :class:`CollectingStateWriter` — buffer every emitted State-Protocol frame in a
  list (tests, snapshots, "run the graph and inspect the rows").
* :class:`QueueStateWriter` — hand each frame to an ``asyncio.Queue`` so a
  consumer can iterate rows as they stream (see
  :func:`streamdb_langgraph.driver.iter_state_rows`).
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from streamdb_langgraph.state_protocol import change_event


class CollectingStateWriter:
    """In-memory :class:`StateWriterProtocol` that records every wire frame.

    Each ``upsert`` / ``delete`` is turned into the same State-Protocol frame a
    durable-stream writer would put on the wire (see ``change_event``) and pushed
    onto :attr:`frames`. ``close()`` flips :attr:`closed`.
    """

    def __init__(self) -> None:
        self.frames: list[dict[str, Any]] = []
        self.closed = False

    async def upsert(
        self,
        type: str,
        key: str,
        value: Mapping[str, Any],
        *,
        txid: str | None = None,
    ) -> None:
        self.frames.append(
            change_event(
                type=type, key=key, operation="upsert", value=value, txid=txid
            )
        )

    async def delete(self, type: str, key: str) -> None:
        self.frames.append(change_event(type=type, key=key, operation="delete"))

    async def close(self) -> None:
        self.closed = True


class QueueStateWriter:
    """A :class:`StateWriterProtocol` that publishes each frame to a queue.

    ``close()`` enqueues the sentinel so a consumer draining the queue knows the
    run is done. Used by :func:`streamdb_langgraph.driver.iter_state_rows`.
    """

    CLOSED: object = object()

    def __init__(self, queue: asyncio.Queue[Any]) -> None:
        self._queue = queue

    async def upsert(
        self,
        type: str,
        key: str,
        value: Mapping[str, Any],
        *,
        txid: str | None = None,
    ) -> None:
        await self._queue.put(
            change_event(
                type=type, key=key, operation="upsert", value=value, txid=txid
            )
        )

    async def delete(self, type: str, key: str) -> None:
        await self._queue.put(
            change_event(type=type, key=key, operation="delete")
        )

    async def close(self) -> None:
        await self._queue.put(self.CLOSED)
