"""The one-call convenience layer: ``translate_run`` / ``iter_state_rows``.

Both are driven over a scripted stream (``make_run_stream`` monkeypatched to a
``fake_run_stream``) so the wiring is tested without a live LangGraph graph.
"""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from streamdb_langgraph import (
    CollectingStateWriter,
    iter_state_rows,
    translate_run,
    translate_turn,
)
from streamdb_langgraph import driver as driver_module
from tests.fixtures import (
    FakeMessage,
    FakeToolCall,
    blocking_channel,
    fake_run_stream,
)


@pytest.fixture
def scripted_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch ``make_run_stream`` (as imported into driver) to a one-message run."""

    async def _fake_make_run_stream(graph, graph_input, **kwargs):  # noqa: ANN001, ANN003
        return fake_run_stream(
            messages=[FakeMessage("msg-1", [("text", 0, "hello")])]
        )

    monkeypatch.setattr(driver_module, "make_run_stream", _fake_make_run_stream)


def _types(frames: list[dict[str, object]]) -> list[str]:
    return [str(f["type"]) for f in frames]


async def test_translate_run_writes_run_and_message_rows(
    scripted_stream: None,
) -> None:
    writer = CollectingStateWriter()

    translator = await translate_run(
        object(),  # graph unused; make_run_stream is patched
        {"messages": []},
        writer=writer,
        thread_id="conv-1",
        run_id="run-1",
    )

    assert writer.closed is True
    assert translator.thread_id == "conv-1"
    types = _types(writer.frames)
    # lifecycle brackets the transcript: running run row first, complete last.
    assert types[0] == "run"
    assert types[-1] == "run"
    assert "message" in types
    assert "messageChunk" in types
    run_rows = [f for f in writer.frames if f["type"] == "run"]
    assert run_rows[0]["value"]["status"] == "running"
    assert run_rows[-1]["value"]["status"] == "complete"


async def test_iter_state_rows_yields_frames_in_order(
    scripted_stream: None,
) -> None:
    frames = [
        frame
        async for frame in iter_state_rows(
            object(),
            {"messages": []},
            thread_id="conv-1",
        )
    ]

    types = _types(frames)
    assert types[0] == "run"
    assert types[-1] == "run"
    assert "message" in types
    assert frames[-1]["value"]["status"] == "complete"


async def test_iter_state_rows_early_break_cancels_background_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Breaking out of the iterator early must cancel the background run instead
    of orphaning it — a run that never closes (blocked channel) would otherwise
    hang the ``finally`` join forever. The ``wait_for`` guards against that."""

    async def _fake_make_run_stream(graph, graph_input, **kwargs):  # noqa: ANN001, ANN003
        # A channel that never ends → the run never reaches its CLOSED sentinel.
        return fake_run_stream(values=blocking_channel())

    monkeypatch.setattr(driver_module, "make_run_stream", _fake_make_run_stream)

    agen = iter_state_rows(object(), {"messages": []}, thread_id="conv-1")
    first = await asyncio.wait_for(agen.__anext__(), timeout=1)
    assert first["type"] == "run"
    # Early break: closing the generator runs its finally, which must cancel the
    # still-running drive task (not await it) — so aclose returns, doesn't hang.
    await asyncio.wait_for(agen.aclose(), timeout=1)


def _messages(frames: list[dict[str, object]]) -> list[dict[str, object]]:
    return [f["value"] for f in frames if f["type"] == "message"]  # type: ignore[misc]


def _user_row(frames: list[dict[str, object]]) -> dict[str, object]:
    return next(m for m in _messages(frames) if m["role"] == "user")


def _max_order(frames: list[dict[str, object]]) -> int:
    orders = [
        f["value"]["order"]  # type: ignore[index]
        for f in frames
        if isinstance(f.get("value"), dict) and "order" in f["value"]  # type: ignore[operator]
    ]
    return max(orders)


def _patch_stream(
    monkeypatch: pytest.MonkeyPatch,
    *,
    message_id: str,
    tool_call_id: str,
) -> None:
    async def _fake_make_run_stream(graph, graph_input, **kwargs):  # noqa: ANN001, ANN003
        return fake_run_stream(
            messages=[FakeMessage(message_id, [("text", 0, "reply")])],
            tool_calls=[
                FakeToolCall(tool_call_id, "calculator", {"expression": "2+2"}, output="4")
            ],
        )

    monkeypatch.setattr(driver_module, "make_run_stream", _fake_make_run_stream)


async def test_translate_turn_emits_user_row_then_reply_on_one_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One turn: the user row leads, seeded at ordinal/order 0, and the assistant
    reply continues the SAME counters (no second writer, no guessed ordinal)."""
    _patch_stream(monkeypatch, message_id="asst-1", tool_call_id="tc-1")
    writer = CollectingStateWriter()

    await translate_turn(
        object(),
        "hello",
        writer=writer,
        thread_id="conv-1",
        user_message_id="u1",
    )

    assert writer.closed is True
    user = _user_row(writer.frames)
    assert user["id"] == "u1"
    assert user["ordinal"] == 0
    assert user["order"] == 0
    assert user["text"] == "hello"
    # The assistant message shares the sequence: next free ordinal after the user.
    assistant = next(m for m in _messages(writer.frames) if m["role"] == "assistant")
    assert assistant["ordinal"] == 1
    # Lifecycle still brackets the turn.
    types = _types(writer.frames)
    assert types[-1] == "run" and writer.frames[-1]["value"]["status"] == "complete"


async def test_translate_turn_two_turns_share_one_monotonic_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two turns on one conversation: turn 1 lays down user 0, asst 1, tool 2; turn
    2, seeded past that history, continues at user order 3 — no collision, no reset."""
    writer = CollectingStateWriter()

    _patch_stream(monkeypatch, message_id="asst-1", tool_call_id="tc-1")
    await translate_turn(
        object(),
        "first",
        writer=writer,
        thread_id="conv-1",
        user_message_id="u1",
    )

    turn1 = list(writer.frames)
    assert _user_row(turn1)["order"] == 0
    # user(0) + asst(1) + tool(2): three rows share the conversation-global order.
    assert _max_order(turn1) == 2

    # Fold turn 1 into history the way a sole-writer server would, then continue the
    # sequence past the highest order already on the stream.
    prior = [
        HumanMessage(content="first", id="u1"),
        AIMessage(content="reply", id="asst-1"),
    ]
    _patch_stream(monkeypatch, message_id="asst-2", tool_call_id="tc-2")
    await translate_turn(
        object(),
        "second",
        writer=writer,
        thread_id="conv-1",
        prior_messages=prior,
        user_message_id="u2",
        order_start=_max_order(turn1) + 1,
    )

    turn2_frames = writer.frames[len(turn1) :]
    user2 = _user_row(turn2_frames)
    # Continues past turn 1: order 3 (never re-uses 0), ordinal 2 (2 prior messages).
    assert user2["id"] == "u2"
    assert user2["order"] == 3
    assert user2["ordinal"] == 2
