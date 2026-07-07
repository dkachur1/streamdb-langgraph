"""One-call helpers to drive a compiled LangGraph graph into State-Protocol rows.

Both wrap the same three steps — open a v3 run stream
(:func:`~streamdb_langgraph.stream.make_run_stream`), construct a
:class:`~streamdb_langgraph.state_translator.StateTranslator`, and drain the run —
so a caller goes from "I have a compiled graph" to "rows are flowing" in one line.

* :func:`translate_run` writes into any :class:`StateWriterProtocol` you supply
  (your durable-stream writer, or :class:`CollectingStateWriter` for a buffer).
* :func:`iter_state_rows` yields each frame as it is produced — no writer needed.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from streamdb_langgraph.state_protocol import StateWriterProtocol
from streamdb_langgraph.state_translator import StateTranslator
from streamdb_langgraph.stream import make_run_stream
from streamdb_langgraph.writers import QueueStateWriter


async def translate_run(
    graph: CompiledStateGraph,
    graph_input: Any,
    *,
    writer: StateWriterProtocol,
    thread_id: str,
    run_id: str | None = None,
    config: RunnableConfig | None = None,
    initial_state: dict[str, Any] | None = None,
) -> StateTranslator:
    """Drive one run of ``graph`` to completion, writing rows into ``writer``.

    Emits the ``running`` run row, drains every channel of the v3 stream into
    keyed State-Protocol upserts, then marks the run complete and closes the
    writer. Returns the :class:`StateTranslator` (its ``emitted_turn()`` and
    counters are useful for resume / cancel-salvage flows).

    ``graph_input`` is a fresh state dict or a ``langgraph.types.Command``
    (e.g. ``Command(resume=...)``); ``config`` must carry ``thread_id`` under
    ``configurable`` for interrupt-resume to work.
    """
    translator = StateTranslator(writer, thread_id=thread_id, run_id=run_id)
    stream = await make_run_stream(graph, graph_input, config=config)
    await translator.run(stream, initial_state=initial_state)
    return translator


async def iter_state_rows(
    graph: CompiledStateGraph,
    graph_input: Any,
    *,
    thread_id: str,
    run_id: str | None = None,
    config: RunnableConfig | None = None,
    initial_state: dict[str, Any] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Yield each State-Protocol frame as the run produces it.

    A no-writer convenience: drives ``graph`` on a background task and yields the
    frames a durable-stream writer would put on the wire, in emission order,
    until the run closes. Any exception raised inside the run propagates out of
    the iterator once its buffered frames have been yielded.
    """
    queue: asyncio.Queue[Any] = asyncio.Queue()
    writer = QueueStateWriter(queue)
    translator = StateTranslator(writer, thread_id=thread_id, run_id=run_id)
    stream = await make_run_stream(graph, graph_input, config=config)

    async def _drive() -> None:
        # The translator closes the writer (enqueuing the sentinel) on both the
        # success and error paths, so the drain loop below always terminates.
        await translator.run(stream, initial_state=initial_state)

    task: asyncio.Task[None] = asyncio.create_task(_drive())
    try:
        while True:
            frame = await queue.get()
            if frame is QueueStateWriter.CLOSED:
                break
            yield frame
    finally:
        # Surface run failures / cancellation to the caller; a clean run already
        # finished by the time the sentinel arrived, so this just joins it.
        await task
