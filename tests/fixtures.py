"""Test fixtures for driving :class:`StateTranslator` over synthetic runs.

These stand-ins mirror the channel shapes ``StateTranslator.run`` drains from a
real ``langgraph.stream.AsyncGraphRunStream`` (``messages`` / ``tool_calls`` /
``custom`` / ``subgraphs`` / ``values`` projections plus the interrupt hooks), so
a test can exercise the translator over scripted LLM output without a live graph.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable, Sequence
from types import SimpleNamespace
from typing import Any


async def _aiter(items: Iterable[Any]) -> AsyncIterator[Any]:
    for item in items:
        yield item


async def raising_channel(exc: BaseException) -> AsyncIterator[Any]:
    """A channel that raises ``exc`` the first time it is iterated — drives the
    translator's exception / cancellation branches (pass a ``ValueError`` for a
    graph exception, a bare ``asyncio.CancelledError`` for a channel cancel)."""
    raise exc
    yield  # pragma: no cover - makes this an async generator


async def blocking_channel() -> AsyncIterator[Any]:
    """A channel that blocks forever — lets a test cancel the run mid-flight to
    exercise the parent-cancellation path."""
    await asyncio.Event().wait()
    yield  # pragma: no cover - never reached


def _as_aiter(items: Iterable[Any] | AsyncIterator[Any]) -> AsyncIterator[Any]:
    if hasattr(items, "__anext__"):
        return items  # type: ignore[return-value]
    return _aiter(items)


class FakeMessage:
    """One LLM message as consumed by ``StateTranslator._consume_message``.

    ``blocks`` is an ordered list of ``(kind, index, token)`` where ``kind``
    is ``"reasoning"`` or ``"text"``. Yields the same ``content-block-delta``
    envelope shape the translator reads from a real LangGraph message.

    ``tool_calls`` declares the tool-call *requests* this message carries —
    each a ``(id, name, args)`` tuple or a finalized ``ToolCall`` dict, read off
    ``output_message.tool_calls`` once the message stream is exhausted, just like
    a real ``ChatModelStream``.
    """

    def __init__(
        self,
        message_id: str | None,
        blocks: Sequence[tuple[str, int, str]],
        node: str = "model",
        tool_calls: Iterable[Any] = (),
    ) -> None:
        self.message_id = message_id
        self._blocks = blocks
        # The translator only projects messages from the main agent ``"model"``
        # node; peripheral middleware nodes are skipped. Default to the model
        # node so a plain ``FakeMessage`` represents real assistant output.
        self.node = node
        normalized: list[dict[str, Any]] = [
            tc
            if isinstance(tc, dict)
            else {
                "id": tc[0],
                "name": tc[1],
                "args": tc[2] if len(tc) > 2 else {},
                "type": "tool_call",
            }
            for tc in tool_calls
        ]
        self.output_message = SimpleNamespace(tool_calls=normalized)

    def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for kind, index, token in self._blocks:
            if kind == "reasoning":
                delta = {"type": "reasoning-delta", "reasoning": token}
            else:
                delta = {"type": "text-delta", "text": token}
            events.append(
                {"event": "content-block-delta", "index": index, "delta": delta}
            )
        return _aiter(events)


class FakeToolCall:
    """One tool-call handle as consumed by
    ``StateTranslator._consume_tool_call_execution``."""

    def __init__(
        self,
        tool_call_id: str,
        tool_name: str,
        tool_input: Any,
        output: Any = None,
        error: Any = None,
        output_delay: float = 0.0,
    ) -> None:
        self.tool_call_id = tool_call_id
        self.tool_name = tool_name
        self.input = tool_input
        self.output = output
        self.error = error
        self._output_delay = output_delay

    @property
    def output_deltas(self) -> AsyncIterator[Any]:
        if self._output_delay:
            return self._delayed_output_deltas()
        return _aiter([])

    async def _delayed_output_deltas(self) -> AsyncIterator[Any]:
        await asyncio.sleep(self._output_delay)
        yield None


class FakeSubgraph:
    """A ``stream.subgraphs`` handle as consumed by the subagent translator.

    Subagent children run under the ``subagent:<agent_id>`` namespace, so the
    real handle carries ``graph_name="subagent"`` and ``trigger_call_id`` =
    the subagent's ``agent_id``, with child-scoped ``messages`` / ``custom`` /
    ``subgraphs`` projections.
    """

    def __init__(
        self,
        trigger_call_id: str,
        *,
        graph_name: str = "subagent",
        messages: Iterable[Any] = (),
        custom: Iterable[Any] = (),
        subgraphs: Iterable[Any] = (),
        tool_calls: Iterable[Any] = (),
        output_state: Any | None = None,
    ) -> None:
        self.trigger_call_id = trigger_call_id
        self.graph_name = graph_name
        self.extensions = {
            "messages": _as_aiter(messages),
            "custom": _as_aiter(custom),
            "subgraphs": _as_aiter(subgraphs),
            "tool_calls": _as_aiter(tool_calls),
        }
        self._output_state = output_state or {"messages": []}

    async def output(self) -> Any:
        return self._output_state


class FakeInterrupt:
    def __init__(self, interrupt_id: str, value: Any) -> None:
        self.id = interrupt_id
        self.value = value


class _FakeStream:
    def __init__(
        self,
        messages: Iterable[Any],
        tool_calls: Iterable[Any],
        custom: Iterable[Any],
        subgraphs: Iterable[Any] = (),
        values: Iterable[Any] = (),
        interrupted: bool = False,
        interrupts: Iterable[Any] = (),
    ) -> None:
        self.extensions = {
            "messages": _as_aiter(messages),
            "tool_calls": _as_aiter(tool_calls),
            "custom": _as_aiter(custom),
            "subgraphs": _as_aiter(subgraphs),
            "values": _as_aiter(values),
        }
        self._interrupted = interrupted
        self._interrupts = list(interrupts)

    async def __aenter__(self) -> _FakeStream:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def interrupted(self) -> bool:
        return self._interrupted

    async def interrupts(self) -> list[Any]:
        return self._interrupts


def fake_run_stream(
    *,
    messages: Iterable[Any] = (),
    tool_calls: Iterable[Any] = (),
    custom: Iterable[Any] = (),
    subgraphs: Iterable[Any] = (),
    values: Iterable[Any] = (),
    interrupted: bool = False,
    interrupts: Iterable[Any] = (),
) -> _FakeStream:
    """Build a minimal stand-in for ``AsyncGraphRunStream`` exposing the
    channels ``StateTranslator.run`` drains."""
    return _FakeStream(
        messages,
        tool_calls,
        custom,
        subgraphs,
        values=values,
        interrupted=interrupted,
        interrupts=interrupts,
    )
