"""Tests for LangGraph → AG-UI interrupt adaptation."""

from __future__ import annotations

from langgraph.types import Interrupt

from streamdb_langgraph.interrupts import (
    ag_ui_interrupt_reason,
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
