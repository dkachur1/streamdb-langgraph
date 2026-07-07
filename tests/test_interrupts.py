"""Tests for LangGraph → AG-UI interrupt adaptation."""

from __future__ import annotations

from langgraph.types import Interrupt

from streamdb_langgraph.interrupts import (
    ag_ui_interrupt_reason,
    ag_ui_interrupt_rows,
    langgraph_interrupt_to_ag_ui_interrupt,
)


class TestInterruptReason:
    def test_known_types(self):
        assert ag_ui_interrupt_reason("ask_question") == "input_required"
        assert ag_ui_interrupt_reason("request_review") == "confirmation"

    def test_unknown_type_gets_namespaced_fallback(self):
        assert ag_ui_interrupt_reason("something_new") == "langgraph:something_new"


class TestConvertRequestReview:
    def test_payload_passes_through_metadata(self):
        interrupt = Interrupt(
            value={
                "type": "request_review",
                "mode": "approval",
                "document": "## Review",
                "tool_call_id": "call-9",
            },
            id="int-1",
        )

        item = langgraph_interrupt_to_ag_ui_interrupt(interrupt)

        assert item is not None
        assert item["reason"] == "confirmation"
        assert item["toolCallId"] == "call-9"
        assert item["metadata"]["document"] == "## Review"
        assert item["metadata"]["mode"] == "approval"


class TestAgUiInterruptRowsCollision:
    """LangGraph hashes an interrupt id from the checkpoint namespace alone, so
    parallel tool calls in one ToolNode can legitimately share a raw id
    (langchain-ai/langgraph#6626). Rows must stay disambiguated so upserting
    them doesn't drop one interrupt via last-writer-wins key collision."""

    def test_distinct_raw_ids_pass_through_unchanged(self):
        interrupts = [
            Interrupt(value={"type": "ask_question", "tool_call_id": "call-1"}, id="int-1"),
            Interrupt(value={"type": "ask_question", "tool_call_id": "call-2"}, id="int-2"),
        ]

        rows = ag_ui_interrupt_rows(interrupts, "thread-1")

        assert [row["id"] for row in rows] == ["int-1", "int-2"]

    def test_colliding_raw_ids_are_disambiguated_by_tool_call_id(self):
        interrupts = [
            Interrupt(value={"type": "ask_question", "tool_call_id": "call-1"}, id="dup"),
            Interrupt(value={"type": "ask_question", "tool_call_id": "call-2"}, id="dup"),
        ]

        rows = ag_ui_interrupt_rows(interrupts, "thread-1")

        ids = [row["id"] for row in rows]
        assert ids[0] == "dup"
        assert ids[1] == "dup:call-2"
        assert len(set(ids)) == 2

    def test_colliding_raw_ids_without_tool_call_id_get_ordinal_suffix(self):
        interrupts = [
            Interrupt(value={"type": "ask_question"}, id="dup"),
            Interrupt(value={"type": "ask_question"}, id="dup"),
        ]

        rows = ag_ui_interrupt_rows(interrupts, "thread-1")

        ids = [row["id"] for row in rows]
        assert ids == ["dup", "dup:1"]
