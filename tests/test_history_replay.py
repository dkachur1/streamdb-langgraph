"""replay_history tests — checkpoint messages → state-protocol rows.

Guards the seed that lets a fresh StreamDB render the whole transcript from one
offset-0 read: prior turns become message/tool rows keyed by the REAL
tool_call_id (no message_to_ui synthesis), stamped with the conversation-global
``order`` the client renders by, and the returned ordinal lets the live run
continue after history.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from streamdb_langgraph.history_replay import next_live_ordinal, replay_history
from streamdb_langgraph.state_protocol import TYPE_MESSAGE, TYPE_TOOL


class RecordingStateWriter:
    def __init__(self) -> None:
        self.frames: list[tuple[str, str, dict[str, Any]]] = []

    async def upsert(self, type: str, key: str, value: Mapping[str, Any]) -> None:
        self.frames.append((type, key, dict(value)))

    async def delete(self, type: str, key: str) -> None: ...

    async def close(self) -> None: ...

    def of(self, type_name: str) -> list[dict[str, Any]]:
        return [v for (t, _k, v) in self.frames if t == type_name]

    def by_key(self, type_name: str, key: str) -> dict[str, Any] | None:
        found = None
        for t, k, v in self.frames:
            if t == type_name and k == key:
                found = v
        return found


async def _run(messages: list[Any]) -> tuple[RecordingStateWriter, int]:
    w = RecordingStateWriter()
    next_ordinal = await replay_history(w, "t", messages)
    return w, next_ordinal


class TestMessages:
    async def test_user_and_assistant_become_ordered_rows(self):
        w, nxt = await _run(
            [
                HumanMessage(id="u1", content="hi"),
                AIMessage(id="a1", content="hello there"),
            ]
        )
        assert w.by_key(TYPE_MESSAGE, "u1") == {
            "id": "u1",
            "threadId": "t",
            "role": "user",
            "ordinal": 0,
            "order": 0,
            "text": "hi",
        }
        assert w.by_key(TYPE_MESSAGE, "a1")["ordinal"] == 1
        assert w.by_key(TYPE_MESSAGE, "a1")["order"] == 1
        assert w.by_key(TYPE_MESSAGE, "a1")["text"] == "hello there"
        # next free ordinal for the live run
        assert nxt == 2

    async def test_list_content_text_is_concatenated(self):
        w, _ = await _run(
            [
                AIMessage(
                    id="a1",
                    content=[
                        {"type": "text", "text": "ab"},
                        {"type": "text", "text": "cd"},
                    ],
                )
            ]
        )
        assert w.by_key(TYPE_MESSAGE, "a1")["text"] == "abcd"

    async def test_reasoning_blocks_extracted(self):
        w, _ = await _run(
            [
                AIMessage(
                    id="a1",
                    content=[
                        {"type": "thinking", "thinking": "hmm"},
                        {"type": "text", "text": "answer"},
                    ],
                )
            ]
        )
        row = w.by_key(TYPE_MESSAGE, "a1")
        assert row["text"] == "answer"
        assert row["reasoning"] == "hmm"

    async def test_summarization_messages_skipped(self):
        w, nxt = await _run(
            [
                AIMessage(id="summarization-1", content="compacted"),
                AIMessage(id="a2", content="real"),
            ]
        )
        assert w.by_key(TYPE_MESSAGE, "summarization-1") is None
        assert w.by_key(TYPE_MESSAGE, "a2")["ordinal"] == 0
        assert nxt == 1

    async def test_summarization_human_note_skipped(self):
        # The hard-compaction "note to self" is a HumanMessage; it must not seed
        # a fake user bubble and must not consume an ordinal, or the live run's
        # ordinals (via next_live_ordinal) drift out of the seeded sequence.
        msgs = [
            HumanMessage(id="u1", content="q1"),
            AIMessage(id="a1", content="a1"),
            HumanMessage(id="summarization-note", content="notes to self"),
            HumanMessage(id="u2", content="q2"),
        ]
        w, nxt = await _run(msgs)
        assert w.by_key(TYPE_MESSAGE, "summarization-note") is None
        assert w.by_key(TYPE_MESSAGE, "u1")["ordinal"] == 0
        assert w.by_key(TYPE_MESSAGE, "a1")["ordinal"] == 1
        assert w.by_key(TYPE_MESSAGE, "u2")["ordinal"] == 2
        assert nxt == 3
        assert next_live_ordinal(msgs) == 3


class TestToolCalls:
    async def test_tool_call_and_result_keyed_by_real_id(self):
        w, _ = await _run(
            [
                AIMessage(
                    id="a1",
                    content="",
                    tool_calls=[
                        {"id": "call_real", "name": "search", "args": {"q": "x"}}
                    ],
                ),
                ToolMessage(id="tm1", tool_call_id="call_real", content="found it"),
            ]
        )
        tool = w.by_key(TYPE_TOOL, "call_real")
        assert tool["id"] == "call_real"  # real id, not synthesized
        assert "messageId" not in tool  # dead field; nothing reads it
        assert tool["name"] == "search"
        assert tool["argsText"]
        assert tool["result"]
        assert "isError" not in tool

    async def test_tool_order_follows_its_requesting_message(self):
        # The tool's `order` continues the same conversation-global counter as
        # its requesting message's — assemble.ts has no `messageId` anchor, so a
        # missing/stale `order` would strand the card on the last message.
        w, _ = await _run(
            [
                AIMessage(
                    id="a1",
                    content="",
                    tool_calls=[{"id": "call_real", "name": "search", "args": {}}],
                ),
                ToolMessage(id="tm1", tool_call_id="call_real", content="found it"),
            ]
        )
        assert w.by_key(TYPE_MESSAGE, "a1")["order"] == 0
        assert w.by_key(TYPE_TOOL, "call_real")["order"] == 1

    async def test_errored_tool_result(self):
        w, _ = await _run(
            [
                AIMessage(
                    id="a1",
                    content="",
                    tool_calls=[{"id": "c1", "name": "search", "args": {}}],
                ),
                ToolMessage(
                    id="tm1", tool_call_id="c1", content="boom", status="error"
                ),
            ]
        )
        assert w.by_key(TYPE_TOOL, "c1")["isError"] is True

    async def test_tool_result_without_matching_call_ignored(self):
        w, _ = await _run([ToolMessage(id="tm1", tool_call_id="orphan", content="x")])
        assert w.of(TYPE_TOOL) == []

    async def test_interrupt_tool_projects_artifact_result(self):
        # ask_question / request_review results unwrap to their structured
        # artifact — the identical shape StateTranslator projects live — not the
        # ToolMessage's prose content string.
        w, _ = await _run(
            [
                AIMessage(
                    id="a1",
                    content="",
                    tool_calls=[
                        {"id": "call_1", "name": "ask_question", "args": {"q": "x"}}
                    ],
                ),
                ToolMessage(
                    id="tm1",
                    tool_call_id="call_1",
                    content="The user answered.",
                    artifact={"answers": [["yes"]]},
                ),
            ]
        )
        assert w.by_key(TYPE_TOOL, "call_1")["result"] == {"answers": [["yes"]]}

    async def test_non_interrupt_tool_keeps_content_string_result(self):
        # The artifact-unwrap is scoped to INTERRUPT_TOOL_NAMES only — any other
        # tool's result stays the plain content string, even if it happens to
        # carry an `.artifact`.
        w, _ = await _run(
            [
                AIMessage(
                    id="a1",
                    content="",
                    tool_calls=[{"id": "call_1", "name": "search", "args": {}}],
                ),
                ToolMessage(
                    id="tm1",
                    tool_call_id="call_1",
                    content="found it",
                    artifact={"leak": True},
                ),
            ]
        )
        assert w.by_key(TYPE_TOOL, "call_1")["result"] == "found it"


class TestCreatedAt:
    async def test_human_message_created_at_from_sent_at(self):
        # sent_at is stamped by the API boundary (TimestampMiddleware) on
        # HumanMessages persisted by the legacy AI-SDK path — exactly the
        # pre-cutover checkpoints this replay backfills.
        w, _ = await _run(
            [
                HumanMessage(
                    id="u1",
                    content="hi",
                    additional_kwargs={"sent_at": "2024-01-02T03:04:05+00:00"},
                )
            ]
        )
        assert w.by_key(TYPE_MESSAGE, "u1")["createdAt"] == 1704164645000

    async def test_human_message_without_sent_at_omits_created_at(self):
        w, _ = await _run([HumanMessage(id="u1", content="hi")])
        assert "createdAt" not in w.by_key(TYPE_MESSAGE, "u1")

    async def test_human_message_unparseable_sent_at_omits_created_at(self):
        w, _ = await _run(
            [
                HumanMessage(
                    id="u1", content="hi", additional_kwargs={"sent_at": "not-a-date"}
                )
            ]
        )
        assert "createdAt" not in w.by_key(TYPE_MESSAGE, "u1")

    async def test_assistant_message_never_gets_created_at(self):
        # No LangChain or provider field records when an AIMessage was
        # produced — the client's arrival-order fallback applies instead.
        w, _ = await _run([AIMessage(id="a1", content="hello")])
        assert "createdAt" not in w.by_key(TYPE_MESSAGE, "a1")


class TestOrder:
    async def test_order_stamped_in_emission_sequence_across_turns(self):
        # One counter, incremented per row (message then its own tool-calls) in
        # emission order — matching `next_live_order`'s count of the same
        # messages, so a live run resumes exactly where the replay left off.
        w, _ = await _run(
            [
                HumanMessage(id="u1", content="q1"),
                AIMessage(
                    id="a1",
                    content="",
                    tool_calls=[{"id": "c1", "name": "search", "args": {}}],
                ),
                ToolMessage(id="tm1", tool_call_id="c1", content="r1"),
                HumanMessage(id="u2", content="q2"),
                AIMessage(id="a2", content="done"),
            ]
        )
        assert w.by_key(TYPE_MESSAGE, "u1")["order"] == 0
        assert w.by_key(TYPE_MESSAGE, "a1")["order"] == 1
        assert w.by_key(TYPE_TOOL, "c1")["order"] == 2
        assert w.by_key(TYPE_MESSAGE, "u2")["order"] == 3
        assert w.by_key(TYPE_MESSAGE, "a2")["order"] == 4


async def test_empty_history_returns_zero():
    _w, nxt = await _run([])
    assert nxt == 0


class TestWindowing:
    @staticmethod
    def _conversation(turns: int) -> list[Any]:
        msgs: list[Any] = []
        for i in range(turns):
            msgs.append(HumanMessage(id=f"u{i}", content=f"q{i}"))
            msgs.append(AIMessage(id=f"a{i}", content=f"r{i}"))
        return msgs

    async def test_windows_to_last_n_turns(self):
        w = RecordingStateWriter()
        nxt = await replay_history(w, "t", self._conversation(5), max_turns=2)
        rows = w.of(TYPE_MESSAGE)
        # Only the last 2 turns (4 messages) are seeded; ordinals restart at 0.
        assert {r["id"] for r in rows} == {"u3", "a3", "u4", "a4"}
        assert w.by_key(TYPE_MESSAGE, "u3")["ordinal"] == 0
        assert nxt == 4

    async def test_window_larger_than_history_keeps_all(self):
        w = RecordingStateWriter()
        nxt = await replay_history(w, "t", self._conversation(2), max_turns=10)
        assert {r["id"] for r in w.of(TYPE_MESSAGE)} == {"u0", "a0", "u1", "a1"}
        assert nxt == 4

    async def test_window_keeps_tool_pair_intact(self):
        # The window boundary lands on a turn start, so a tool result is never
        # split from its requesting assistant message.
        msgs: list[Any] = [
            HumanMessage(id="u0", content="old"),
            AIMessage(id="a0", content="old"),
            HumanMessage(id="u1", content="new"),
            AIMessage(
                id="a1",
                content="",
                tool_calls=[{"id": "c1", "name": "search", "args": {}}],
            ),
            ToolMessage(id="tm1", tool_call_id="c1", content="found"),
        ]
        w = RecordingStateWriter()
        await replay_history(w, "t", msgs, max_turns=1)
        assert w.by_key(TYPE_MESSAGE, "u0") is None
        assert w.by_key(TYPE_TOOL, "c1")["result"]
