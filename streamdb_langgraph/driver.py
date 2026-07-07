"""One-call helpers to drive a compiled LangGraph graph into State-Protocol rows.

Both wrap the same three steps — open a v3 run stream
(:func:`~streamdb_langgraph.stream.make_run_stream`), construct a
:class:`~streamdb_langgraph.state_translator.StateTranslator`, and drain the run —
so a caller goes from "I have a compiled graph" to "rows are flowing" in one line.

* :func:`translate_run` writes into any :class:`StateWriterProtocol` you supply
  (your durable-stream writer, or :class:`CollectingStateWriter` for a buffer).
* :func:`translate_turn` is the sole-writer convenience: it appends the USER
  message row AND drives the agent's reply on ONE shared, monotonically-seeded
  ``order`` / ``ordinal`` sequence — the canonical "one backend writes the whole
  conversation" pattern — without dropping down to :class:`StateTranslator`.
* :func:`iter_state_rows` yields each frame as it is produced — no writer needed.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Sequence
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from streamdb_langgraph.history_replay import next_live_order, next_live_ordinal
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


async def translate_turn(
    graph: CompiledStateGraph,
    user_message: str | HumanMessage,
    *,
    writer: StateWriterProtocol,
    thread_id: str,
    prior_messages: Sequence[BaseMessage] = (),
    run_id: str | None = None,
    user_message_id: str | None = None,
    attachments: list[dict[str, Any]] | None = None,
    ordinal_start: int | None = None,
    order_start: int | None = None,
    config: RunnableConfig | None = None,
    initial_state: dict[str, Any] | None = None,
) -> StateTranslator:
    """Drive one conversational TURN as the stream's sole writer: append the user
    message row AND stream the agent's reply on ONE monotonic ``order`` / ``ordinal``
    sequence seeded past ``prior_messages``.

    This is the canonical "one backend writes the whole conversation" pattern. The
    browser never writes rows; the user's text reaches this call (e.g. via an
    ``onSend`` POST), and a single :class:`StateTranslator` emits the user row and
    every assistant/tool row off shared counters — no client-guessed ordinal, no
    second writer. Across turns the counters continue past all prior history so a
    new row never collides with, or sorts before, an earlier one.

    ``user_message`` is the new turn's text (or a ``HumanMessage`` whose id/content
    are reused). ``prior_messages`` is the folded conversation history (LangChain
    messages, in ``order``); it both feeds the graph and, by default, sizes the
    continuation counters via :func:`next_live_ordinal` / :func:`next_live_order`.

    ``ordinal_start`` / ``order_start`` override that derivation for callers whose
    folded history does not carry tool-call rows on its ``AIMessage``s (so
    ``next_live_order`` would undercount): pass the true next-free ``order`` — e.g.
    one past the highest ``order`` the stream has already carried. Returns the
    :class:`StateTranslator` (its ``emitted_turn()`` / counters aid resume/salvage).
    """
    if isinstance(user_message, HumanMessage):
        text = _message_text(user_message.content)
        user_id = user_message_id or _coerce_id(user_message.id)
    else:
        text = user_message
        user_id = user_message_id or f"msg-{uuid.uuid4()}"

    translator = StateTranslator(writer, thread_id=thread_id, run_id=run_id)
    # Continue past all prior history so the new user/assistant/tool rows share the
    # conversation's single monotonic sequence instead of restarting at the top.
    translator.set_ordinal_start(
        ordinal_start if ordinal_start is not None else next_live_ordinal(prior_messages)
    )
    translator.set_order_start(
        order_start if order_start is not None else next_live_order(prior_messages)
    )

    await translator.emit_user_message(user_id, text, attachments=attachments)

    graph_input = {"messages": [*prior_messages, HumanMessage(content=text, id=user_id)]}
    stream = await make_run_stream(graph, graph_input, config=config)
    await translator.run(stream, initial_state=initial_state)
    return translator


def _message_text(content: Any) -> str:
    """Plain text of a LangChain message ``content`` (a string, or content blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return ""


def _coerce_id(message_id: str | None) -> str:
    return message_id or f"msg-{uuid.uuid4()}"


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
