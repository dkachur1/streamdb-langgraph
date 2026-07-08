"""StateTranslator emission tests.

Drives the real ``StateTranslator`` over the synthetic LLM output the
the translator drives, and asserts it materialises the right rows. The
state protocol's invariant is simpler than AG-UI's lifecycle: every frame is a
full-row upsert (or delete), so the assertions check row contents + last-writer-
wins, not open/close framing.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import pytest

from streamdb_langgraph.state_protocol import (
    TYPE_AGENT_STATE,
    TYPE_EFFECT,
    TYPE_INTERRUPT,
    TYPE_MESSAGE,
    TYPE_MESSAGE_CHUNK,
    TYPE_RUN,
    TYPE_TOOL,
    TYPE_TOOL_ACTION,
    TYPE_TOOL_SUMMARY,
)
from streamdb_langgraph.state_translator import StateTranslator
from tests.fixtures import (
    FakeInterrupt,
    FakeMessage,
    FakeSubgraph,
    FakeToolCall,
    blocking_channel,
    fake_run_stream,
    raising_channel,
)


class RecordingStateWriter:
    """In-memory ``StateWriterProtocol`` — records frames instead of persisting."""

    def __init__(self) -> None:
        self.frames: list[tuple[str, str, str, dict[str, Any] | None]] = []
        self.closed = False

    async def upsert(
        self,
        type: str,
        key: str,
        value: Mapping[str, Any],
        *,
        txid: str | None = None,
    ) -> None:
        self.frames.append(("upsert", type, key, dict(value)))

    async def delete(self, type: str, key: str) -> None:
        self.frames.append(("delete", type, key, None))

    async def close(self) -> None:
        self.closed = True

    # helpers -----------------------------------------------------------------
    def upserts(self, type_name: str) -> list[dict[str, Any]]:
        return [
            v
            for (op, t, _k, v) in self.frames
            if op == "upsert" and t == type_name and v
        ]

    def last(self, type_name: str, key: str) -> dict[str, Any] | None:
        found = None
        for op, t, k, v in self.frames:
            if op == "upsert" and t == type_name and k == key:
                found = v
        return found

    def _channel_chunks(
        self, message_id: str, channel: str | None
    ) -> list[dict[str, Any]]:
        rows = [
            v
            for v in self.upserts(TYPE_MESSAGE_CHUNK)
            if v.get("messageId") == message_id and v.get("channel") == channel
        ]
        return sorted(rows, key=lambda v: v["seq"])

    def chunks(self, message_id: str) -> list[dict[str, Any]]:
        """The TEXT ``messageChunk`` rows (channel absent), in ``seq`` order."""
        return self._channel_chunks(message_id, None)

    def chunk_text(self, message_id: str) -> str:
        return "".join(c["text"] for c in self.chunks(message_id))

    def reasoning_chunks(self, message_id: str) -> list[dict[str, Any]]:
        """The reasoning ``messageChunk`` rows (channel="reasoning"), seq order."""
        return self._channel_chunks(message_id, "reasoning")

    def reasoning_text(self, message_id: str) -> str:
        return "".join(c["text"] for c in self.reasoning_chunks(message_id))


async def _run(**channels: Any) -> RecordingStateWriter:
    writer = RecordingStateWriter()
    translator = StateTranslator(writer, thread_id="t", run_id="r")
    interrupted = channels.pop("interrupted", False)
    interrupts = channels.pop("interrupts", ())
    stream = fake_run_stream(interrupted=interrupted, interrupts=interrupts, **channels)
    await translator.run(stream)
    return writer


class TestRunLifecycle:
    async def test_empty_run_is_running_then_complete_then_closed(self):
        w = await _run()
        runs = [(k, v["status"]) for (op, t, k, v) in w.frames if t == TYPE_RUN and v]
        # The run row is keyed by the stable RUN_ROW_ID ("run"), not the thread id,
        # so the frontend runtime's fixed-key lookup binds it.
        assert runs == [("run", "running"), ("run", "complete")]
        # Every run row carries the enqueued run id (feedback attribution).
        assert all(
            v["runId"] == "r" for (op, t, k, v) in w.frames if t == TYPE_RUN and v
        )
        assert w.closed


class TestMessages:
    async def test_text_streams_as_chunks_with_anchor(self):
        # The anchor row carries no streamed text (text="") — the text rides
        # append-only `messageChunk` deltas, and concatenating them by `seq`
        # reproduces the full message.
        msg = FakeMessage("m1", [("text", 0, "Hel"), ("text", 0, "lo")])
        w = await _run(messages=[msg])
        anchor = w.last(TYPE_MESSAGE, "m1")
        assert anchor is not None
        assert isinstance(anchor.pop("createdAt"), int)
        assert anchor == {
            "id": "m1",
            "threadId": "t",
            "role": "assistant",
            "ordinal": 0,
            "order": 0,
            "text": "",
        }
        assert w.chunk_text("m1") == "Hello"

    async def test_chunks_are_deltas_not_growing_prefixes(self):
        # Each chunk carries ONLY the new characters since the prior flush — the
        # whole point: no chunk re-ships an accumulated prefix. seqs increment 0,1…
        tokens = [("text", 0, "abcdefghij") for _ in range(50)]  # 500 chars
        msg = FakeMessage("m1", tokens)
        w = await _run(messages=[msg])
        chunks = w.chunks("m1")
        # Flushed at ~80 chars → several chunks, far fewer than the 50 tokens.
        assert 1 < len(chunks) < 50
        assert [c["seq"] for c in chunks] == list(range(len(chunks)))
        # No single chunk holds the whole text; their concatenation does.
        assert all(len(c["text"]) < 500 for c in chunks)
        assert w.chunk_text("m1") == "abcdefghij" * 50
        # The anchor never re-upserts a growing text prefix (text stays "").
        assert all(row["text"] == "" for row in w.upserts(TYPE_MESSAGE))

    async def test_reasoning_and_text_ride_separate_chunk_channels(self):
        # Reasoning now streams as its own `messageChunk` channel (channel=
        # "reasoning", keyed msg_id:r:seq) exactly like text — the anchor carries
        # neither live; concatenating each channel's chunks by seq restores both.
        msg = FakeMessage("m1", [("reasoning", 0, "think"), ("text", 1, "answer")])
        w = await _run(messages=[msg])
        anchor = w.last(TYPE_MESSAGE, "m1")
        assert anchor is not None
        assert anchor["text"] == ""
        assert "reasoning" not in anchor
        assert w.reasoning_text("m1") == "think"
        assert w.chunk_text("m1") == "answer"

    async def test_reasoning_chunks_are_deltas_not_growing_prefixes(self):
        # Each reasoning chunk carries ONLY the new characters since the prior
        # flush (mirrors the text-chunk delta invariant); the anchor never grows a
        # reasoning prefix — the O(L²) cost this replaces.
        tokens = [("reasoning", 0, "abcdefghij") for _ in range(50)]  # 500 chars
        msg = FakeMessage("m1", tokens)
        w = await _run(messages=[msg])
        chunks = w.reasoning_chunks("m1")
        assert 1 < len(chunks) < 50
        assert [c["seq"] for c in chunks] == list(range(len(chunks)))
        assert all(len(c["text"]) < 500 for c in chunks)
        assert all(c["channel"] == "reasoning" for c in chunks)
        assert w.reasoning_text("m1") == "abcdefghij" * 50
        # The anchor never carries a reasoning prefix on any upsert.
        assert all("reasoning" not in row for row in w.upserts(TYPE_MESSAGE))

    async def test_reasoning_and_text_chunk_keys_do_not_collide(self):
        # Interleaved reasoning + text over one message must not overwrite each
        # other: text keys msg_id:seq, reasoning keys msg_id:r:seq (disjoint).
        msg = FakeMessage("m1", [("reasoning", 0, "R" * 30), ("text", 1, "T" * 100)])
        w = await _run(messages=[msg])
        text_ids = {c["id"] for c in w.chunks("m1")}
        reasoning_ids = {c["id"] for c in w.reasoning_chunks("m1")}
        assert text_ids and reasoning_ids
        assert text_ids.isdisjoint(reasoning_ids)
        assert all(cid.startswith("m1:r:") for cid in reasoning_ids)
        assert w.chunk_text("m1") == "T" * 100
        assert w.reasoning_text("m1") == "R" * 30

    async def test_subagent_reasoning_chunks_carry_agent_id(self):
        # A subagent shares `_stream_message_text`; its reasoning chunks must be
        # tagged with the dispatch tool_call_id (agentId) exactly like its text
        # chunks, and keyed in the reasoning space (msg_id:r:seq).
        sub_msg = FakeMessage(
            "sm1", [("reasoning", 0, "R" * 30), ("text", 1, "hi there")]
        )
        sg = FakeSubgraph("sa1", messages=[sub_msg])
        w = await _run(
            custom=[
                {
                    "content_type": "subagent_start",
                    "subagent_id": "sa1",
                    "tool_call_id": "call_dispatch",
                }
            ],
            subgraphs=[sg],
        )
        reasoning = w.reasoning_chunks("sm1")
        text = w.chunks("sm1")
        assert reasoning and text
        assert all(c["agentId"] == "call_dispatch" for c in reasoning)
        assert all(c["agentId"] == "call_dispatch" for c in text)
        assert all(c["id"].startswith("sm1:r:") for c in reasoning)
        assert w.reasoning_text("sm1") == "R" * 30
        assert w.chunk_text("sm1") == "hi there"
        # The subagent anchor carries no live reasoning either.
        assert "reasoning" not in w.last(TYPE_MESSAGE, "sm1")

    async def test_content_free_message_anchors_with_no_chunks(self):
        # A tool-only dispatch turn (no text) must still materialise its anchor
        # row up-front, so its tool-calls anchor to that turn instead of falling
        # back to the last message — and emits no chunks.
        msg = FakeMessage("m1", blocks=[])
        w = await _run(messages=[msg])
        row = w.last(TYPE_MESSAGE, "m1")
        assert row is not None
        assert row["role"] == "assistant"
        assert row["text"] == ""
        assert w.chunks("m1") == []

    async def test_message_and_tool_share_one_global_order(self):
        # The single sequence the client renders by: a message and a tool emitted
        # in the same run draw distinct positions from one conversation-global
        # counter, so assemble can slot the tool after its message by `order`.
        msg = FakeMessage("m1", [("text", 0, "hi")])
        tc = FakeToolCall("call_1", "search", {"q": "x"}, output="found")
        w = await _run(messages=[msg], tool_calls=[tc])
        msg_order = w.last(TYPE_MESSAGE, "m1")["order"]
        tool_order = w.last(TYPE_TOOL, "call_1")["order"]
        assert isinstance(msg_order, int)
        assert isinstance(tool_order, int)
        assert msg_order != tool_order


class TestToolCalls:
    async def test_tool_call_args_then_result_are_two_upserts_same_key(self):
        tc = FakeToolCall("call_1", "search", {"q": "x"}, output="found")
        w = await _run(tool_calls=[tc])
        tool_frames = [
            v for (op, t, k, v) in w.frames if t == TYPE_TOOL and k == "call_1"
        ]
        assert len(tool_frames) == 2
        assert "result" not in tool_frames[0]
        assert tool_frames[0]["argsText"]
        assert tool_frames[1]["result"]

    async def test_errored_tool_marks_is_error(self):
        tc = FakeToolCall("call_1", "search", {"q": "x"}, error=RuntimeError("boom"))
        w = await _run(tool_calls=[tc])
        assert w.last(TYPE_TOOL, "call_1")["isError"] is True

    async def test_interrupt_tool_projects_clean_artifact_object(self):
        # ask_question / request_review project the ToolMessage's structured
        # artifact (the user's decision) verbatim as `result` — a clean object the
        # receipt reads directly, not the serialised ToolMessage. The output stands
        # in for the ToolMessage: a value carrying an `.artifact` attribute.
        output = SimpleNamespace(artifact={"answers": [["yes"]]})
        tc = FakeToolCall("call_1", "ask_question", {"q": "x"}, output=output)
        w = await _run(tool_calls=[tc])
        assert w.last(TYPE_TOOL, "call_1")["result"] == {"answers": [["yes"]]}

    async def test_interrupt_tool_with_real_toolmessage_projects_artifact(self):
        # The production shape: tc.output is a langchain ToolMessage carrying the
        # structured answer in `.artifact`. The result row must be that clean
        # object, NOT the serialized ToolMessage string the client can't parse.
        from langchain_core.messages import ToolMessage

        output = ToolMessage(
            content="The user answered.",
            artifact={"answers": [["yes"]]},
            tool_call_id="call_1",
        )
        tc = FakeToolCall("call_1", "ask_question", {"q": "x"}, output=output)
        w = await _run(tool_calls=[tc])
        assert w.last(TYPE_TOOL, "call_1")["result"] == {"answers": [["yes"]]}

    async def test_non_interrupt_tool_keeps_content_string_result(self):
        # A regular tool's result stays its content string — the artifact path is
        # scoped to the interrupt tools only.
        output = SimpleNamespace(artifact={"answers": [["leak"]]})
        tc = FakeToolCall("call_1", "search", {"q": "x"}, output=output)
        w = await _run(tool_calls=[tc])
        assert w.last(TYPE_TOOL, "call_1")["result"] != {"answers": [["leak"]]}

    async def test_tool_rows_carry_stable_first_seen_order(self):
        a = FakeToolCall("call_a", "search", {"q": "x"}, output="ra")
        b = FakeToolCall("call_b", "lookup", {"q": "y"}, output="rb")
        w = await _run(tool_calls=[a, b])
        per_key: dict[str, list[int]] = {"call_a": [], "call_b": []}
        for op, t, k, v in w.frames:
            if op == "upsert" and t == TYPE_TOOL and v:
                per_key[k].append(v["order"])
        # A row's args + result upserts share one order — the late result must
        # not move the card.
        assert len(set(per_key["call_a"])) == 1
        assert len(set(per_key["call_b"])) == 1
        # Distinct tools get distinct first-seen positions for the client to sort by.
        assert per_key["call_a"][0] != per_key["call_b"][0]


class TestState:
    async def test_values_upsert_one_agent_state_row(self):
        w = await _run(values=[{"plan_mode": False}, {"plan_mode": True}])
        states = w.upserts(TYPE_AGENT_STATE)
        assert states  # at least one
        assert states[-1]["id"] == "t"

    async def test_unchanged_state_is_not_re_upserted(self):
        w = await _run(values=[{"plan_mode": True}, {"plan_mode": True}])
        assert len(w.upserts(TYPE_AGENT_STATE)) == 1


class TestCustomCollections:
    async def test_tool_summary_becomes_its_own_row(self):
        w = await _run(
            custom=[
                {
                    "content_type": "tool_summary",
                    "text": "ran search",
                    "tool_call_id": "call_1",
                }
            ]
        )
        assert w.last(TYPE_TOOL_SUMMARY, "call_1") == {
            "id": "call_1",
            "toolCallId": "call_1",
            "text": "ran search",
        }

    async def test_tool_action_becomes_its_own_row(self):
        w = await _run(
            custom=[
                {
                    "content_type": "tool_action",
                    "tool_call_id": "call_1",
                    "added": ["Cutting process at 3s cycle time"],
                    "removed": ["Cutting process at 5s cycle time"],
                    "summary": "Updated the cutting station",
                }
            ]
        )
        assert w.last(TYPE_TOOL_ACTION, "call_1") == {
            "id": "call_1",
            "threadId": "t",
            "added": ["Cutting process at 3s cycle time"],
            "removed": ["Cutting process at 5s cycle time"],
            "summary": "Updated the cutting station",
        }

    async def test_tool_action_without_summary_omits_field(self):
        w = await _run(
            custom=[
                {
                    "content_type": "tool_action",
                    "tool_call_id": "call_1",
                    "added": [],
                    "removed": [],
                    "summary": "",
                }
            ]
        )
        row = w.last(TYPE_TOOL_ACTION, "call_1")
        assert row is not None
        assert "summary" not in row

    async def test_tool_action_missing_tool_call_id_dropped(self):
        w = await _run(custom=[{"content_type": "tool_action", "added": ["x"]}])
        assert w.upserts(TYPE_TOOL_ACTION) == []

    async def test_tool_action_does_not_also_emit_effect(self):
        w = await _run(
            custom=[
                {
                    "content_type": "tool_action",
                    "tool_call_id": "call_1",
                    "added": ["x"],
                    "removed": [],
                }
            ]
        )
        assert w.upserts(TYPE_EFFECT) == []

    async def test_batman_started_becomes_thread_scoped_effect(self):
        # Background knowledge-base consolidation → a one-shot effect the client
        # translates into its (timer-cleared) `batman` flag.
        w = await _run(
            custom=[{"content_type": "batman_started", "trigger": "turn_end"}]
        )
        effects = [e for e in w.upserts(TYPE_EFFECT) if e.get("kind") == "batman"]
        assert len(effects) == 1
        assert effects[0]["threadId"] == "t"

    async def test_compaction_start_and_end_toggle_effect_active(self):
        # Compaction's start/finish edges each ride their own effect, toggling the
        # client's `compacting` flag; both carry the thread id for routing.
        w = await _run(
            custom=[
                {"content_type": "compaction_start", "trigger": "hard"},
                {"content_type": "compaction_end", "trigger": "hard"},
            ]
        )
        effects = [e for e in w.upserts(TYPE_EFFECT) if e.get("kind") == "compaction"]
        assert [e["active"] for e in effects] == [True, False]
        assert all(e["threadId"] == "t" for e in effects)


class TestInterrupts:
    async def test_interrupt_upserts_row_and_marks_run_interrupt(self):
        intr = FakeInterrupt("i5", {"type": "ask_question", "tool_call_id": "call_1"})
        w = await _run(interrupted=True, interrupts=[intr])
        row = w.last(TYPE_INTERRUPT, "i5")
        assert row is not None
        # threadId stamps the row so the client filters the shared interrupts
        # collection by thread.
        assert row["threadId"] == "t"
        assert w.last(TYPE_RUN, "run")["status"] == "interrupt"
        assert w.closed

    async def test_clear_interrupts_deletes_addressed_rows(self):
        writer = RecordingStateWriter()
        translator = StateTranslator(writer, thread_id="t", run_id="r")
        await translator.clear_interrupts(["i5", "i6"])
        assert ("delete", TYPE_INTERRUPT, "i5", None) in writer.frames
        assert ("delete", TYPE_INTERRUPT, "i6", None) in writer.frames


class TestEmittedTurn:
    async def test_captures_partial_turn_for_cancel_salvage(self):
        writer = RecordingStateWriter()
        translator = StateTranslator(writer, thread_id="t", run_id="r")
        await translator.emit_user_message("u1", "hello")
        stream = fake_run_stream(
            messages=[FakeMessage("m1", [("text", 0, "par"), ("text", 0, "tial")])],
            tool_calls=[FakeToolCall("tc1", "search", {"q": "x"}, output="ok")],
        )
        await translator.run(stream)

        turn = translator.emitted_turn()
        assert turn.texts == {"u1": "hello", "m1": "partial"}
        assert turn.user_message_ids == frozenset({"u1"})
        assert turn.tool_call_ids == frozenset({"tc1"})
        # Ordinals preserve transcript order so salvage archives in sequence.
        assert turn.ordinals["u1"] < turn.ordinals["m1"]

    async def test_content_free_anchor_is_captured_with_empty_text(self):
        writer = RecordingStateWriter()
        translator = StateTranslator(writer, thread_id="t", run_id="r")
        stream = fake_run_stream(messages=[FakeMessage("m1", [])])
        await translator.run(stream)
        assert translator.emitted_turn().texts == {"m1": ""}


class TestRunAbort:
    """The exception / cancellation branches of ``run()``: a broken or cancelled
    run must still leave the run row terminal and close the writer, or the
    frontend runtime binds ``isRunning`` forever."""

    async def test_graph_exception_marks_run_error_and_closes(self):
        writer = RecordingStateWriter()
        translator = StateTranslator(writer, thread_id="t", run_id="r")
        stream = fake_run_stream(messages=raising_channel(ValueError("boom")))
        with pytest.raises(BaseExceptionGroup):
            await translator.run(stream)
        assert writer.closed
        assert writer.last(TYPE_RUN, "run")["status"] == "error"

    async def test_channel_cancel_marks_run_terminal_closes_and_propagates(self):
        writer = RecordingStateWriter()
        translator = StateTranslator(writer, thread_id="t", run_id="r")
        stream = fake_run_stream(messages=raising_channel(asyncio.CancelledError()))
        with pytest.raises(asyncio.CancelledError):
            await translator.run(stream)
        assert writer.closed
        assert writer.last(TYPE_RUN, "run")["status"] == "error"

    async def test_parent_cancel_marks_run_terminal_closes_and_propagates(self):
        writer = RecordingStateWriter()
        translator = StateTranslator(writer, thread_id="t", run_id="r")
        stream = fake_run_stream(messages=blocking_channel())
        task = asyncio.create_task(translator.run(stream))
        # Let the run open (running row + start blocking) before cancelling it.
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert writer.closed
        assert writer.last(TYPE_RUN, "run")["status"] == "error"

    async def test_writer_closed_exactly_once_on_abort(self):
        class CountingWriter(RecordingStateWriter):
            def __init__(self) -> None:
                super().__init__()
                self.close_count = 0

            async def close(self) -> None:
                self.close_count += 1
                await super().close()

        writer = CountingWriter()
        translator = StateTranslator(writer, thread_id="t", run_id="r")
        stream = fake_run_stream(messages=raising_channel(ValueError("boom")))
        with pytest.raises(BaseExceptionGroup):
            await translator.run(stream)
        assert writer.close_count == 1
