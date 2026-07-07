"""State-protocol producer types — StreamDB-style state sync for LangGraph runs.

Replaces AG-UI-style event-replay emission (``TEXT_MESSAGE_CONTENT`` deltas, the
``TOOL_CALL_START/ARGS/RESULT`` envelope, ``STATE_DELTA`` RFC-6902 jsonpatch, and a
custom-event bus) with one idea: emit the FULL current row, keyed, with an
operation. The client materialises last-writer-wins, so:

  * resume = re-read the stream (idempotent re-apply) — no
    ``already_streamed_tool_call_ids`` suppression;
  * a tool result is just another ``upsert`` of the same tool row — no
    RESULT-suppression / receipt store;
  * shared state is ``upsert`` of one ``agentState`` row — no
    ``jsonpatch.from_diff`` and no STATE_SNAPSHOT re-baseline;
  * app signals (suggestions, tool summaries) are their own collections — no
    CUSTOM-event bridge on the frontend.

Wire shape mirrors packages/state/src/types.ts (``@durable-streams/state``) so the
frontend ``createStreamDB`` materialises frames directly.

This module carries only the wire-frame builder and the
:class:`StateWriterProtocol` the translator depends on. Supplying a concrete
durable-stream-backed writer that implements ``StateWriterProtocol`` (upsert /
delete / close) is the consumer's responsibility.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, Protocol

Operation = Literal["insert", "update", "delete", "upsert"]

# Collection discriminators — must match the frontend state schema (schema.ts).
TYPE_MESSAGE = "message"
TYPE_MESSAGE_CHUNK = "messageChunk"
TYPE_TOOL = "tool"
TYPE_RUN = "run"
TYPE_AGENT_STATE = "agentState"
TYPE_SUGGESTION = "suggestion"
TYPE_TOOL_SUMMARY = "toolSummary"
TYPE_TOOL_ACTION = "toolAction"
TYPE_INTERRUPT = "interrupt"
TYPE_EFFECT = "effect"
TYPE_REPORT = "report"

# The stream is per-conversation, so there is exactly one logical current-run row.
# It is keyed by this stable id (last-writer-wins) rather than the thread id, so the
# frontend runtime — which looks the run row up by a fixed key — always binds it
# (``isRunning`` / the streaming indicator). Must match the frontend's ``RUN_ROW_ID``
# (@assistant-ui/react-durable-streams schema.ts).
RUN_ROW_ID = "run"


def change_event(
    *,
    type: str,
    key: str,
    operation: Operation,
    value: Mapping[str, Any] | None = None,
    old_value: Mapping[str, Any] | None = None,
    txid: str | None = None,
) -> dict[str, Any]:
    """Build one wire frame. ``value`` is the whole row, never a delta. ``txid``,
    when set, rides the frame headers so the client's ``db.utils.awaitTxId``
    resolves once this frame syncs back — confirming an optimistic write."""
    headers: dict[str, str] = {"operation": operation}
    if txid is not None:
        headers["txid"] = txid
    frame: dict[str, Any] = {
        "type": type,
        "key": key,
        "headers": headers,
    }
    if value is not None:
        frame["value"] = dict(value)
    if old_value is not None:
        frame["old_value"] = dict(old_value)
    return frame


class StateWriterProtocol(Protocol):
    """The surface :class:`~streamdb_langgraph.state_translator.StateTranslator`
    depends on, so tests (and the backend's DS writer) can inject any sink."""

    async def upsert(
        self,
        type: str,
        key: str,
        value: Mapping[str, Any],
        *,
        txid: str | None = None,
    ) -> None: ...

    async def delete(self, type: str, key: str) -> None: ...

    async def close(self) -> None: ...
