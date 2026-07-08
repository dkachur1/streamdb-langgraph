"""Replay LangGraph checkpoint messages as state-protocol rows.

So one durable stream — and one frontend ``createStreamDB`` — carries the whole
transcript: a run stream is seeded with prior turns (as ``message`` / ``tool``
rows) before the live run's frames, so the client materialises history + live
from one offset-0 read.

The source is the LangGraph checkpoint (``state.values["messages"]``) — the
durable source of truth — mapped the SAME way :class:`StateTranslator` maps live
messages, keyed by the REAL ``tool_call_id``. This is deliberately NOT routed
through ``message_to_ui`` / ``conversation_messages``: that mapper synthesizes
ids (``{msg_id}-{name}-{idx}``) to dodge an AI-SDK reducer collision that does
not exist under last-writer-wins upserts, and would corrupt ids that must match
the live tool rows (which the resume path re-upserts by real id, idempotently).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)

from streamdb_langgraph.interrupts import unwrap_tool_result
from streamdb_langgraph.serialization import to_ag_ui_json, to_ag_ui_tool_name
from streamdb_langgraph.state_protocol import (
    TYPE_MESSAGE,
    TYPE_TOOL,
    StateWriterProtocol,
)

logger = logging.getLogger(__name__)

# Hard-compaction summary messages must not render in chat (mirrors
# StateTranslator / the live messages-channel filter). The soft-path summary is
# an AIMessage; the hard-path "note to self" is a HumanMessage — both carry this
# id prefix, so both branches below filter it (else the note renders as a fake
# user bubble on reconnect).
SUMMARIZATION_MESSAGE_ID_PREFIX = "summarization-"


def _carries_ordinal(
    message: BaseMessage,
    summarization_message_id_prefix: str = SUMMARIZATION_MESSAGE_ID_PREFIX,
) -> bool:
    """True for messages that occupy an ordinal slot: a non-summary
    ``HumanMessage`` or ``AIMessage``, both with an id. Tool results attach to
    their requesting assistant message and never take a slot; the compaction
    "note to self" (a summarization-prefixed ``HumanMessage``) is never rendered,
    so it takes no slot either. This is the single rule shared by
    ``replay_history`` (seeding) and ``next_live_ordinal`` (the live run's
    starting ordinal), so seeded and live ordinals stay on one sequence.
    """
    msg_id = getattr(message, "id", None)
    if isinstance(message, (HumanMessage, AIMessage)):
        return bool(msg_id) and not (
            isinstance(msg_id, str)
            and msg_id.startswith(summarization_message_id_prefix)
        )
    return False


def next_live_ordinal(
    messages: Sequence[BaseMessage],
    summarization_message_id_prefix: str = SUMMARIZATION_MESSAGE_ID_PREFIX,
) -> int:
    """The next free ordinal after ``messages`` — the starting ordinal for a live
    run's first new message, equal to the count ``replay_history`` would assign to
    the same history. Keyed on the checkpoint message list (the source of truth),
    so the live run's user/assistant rows continue the same monotonic sequence the
    cold-seed lays down from ``sequence_number``."""
    return sum(
        1
        for message in messages
        if _carries_ordinal(message, summarization_message_id_prefix)
    )


def next_live_order(
    messages: Sequence[BaseMessage],
    summarization_message_id_prefix: str = SUMMARIZATION_MESSAGE_ID_PREFIX,
) -> int:
    """The next free conversation-global ``order`` after ``messages`` — the start
    for a live run so resumed rows sort after history instead of jumping to the
    top. ``order`` is stamped on EVERY row (a message AND each of its tool-calls),
    so this is the message count plus the tool-calls — matching
    ``replay_history``'s per-row counter (one ``order`` per message then per
    tool). Only tool-calls carrying an ``id`` are counted: ``replay_history``
    skips id-less calls (they cannot be keyed), so counting them here would
    overshoot the seed and leave a gap the resumed run never fills."""
    tool_calls = sum(
        1
        for m in messages
        if isinstance(m, AIMessage)
        for call in (getattr(m, "tool_calls", None) or ())
        if call.get("id")
    )
    return next_live_ordinal(messages, summarization_message_id_prefix) + tool_calls


def _content_text(content: Any) -> str:
    """Concatenate the plain-text of a LangChain message ``content`` (string or
    a list of content blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: list[str] = []
        for block in content:
            if isinstance(block, str):
                out.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                out.append(str(block.get("text", "")))
        return "".join(out)
    return ""


def _human_created_at(message: HumanMessage) -> int | None:
    """Epoch-ms ``createdAt`` from the frozen ``additional_kwargs["sent_at"]``
    (ISO-8601, stamped at the API boundary — see ``TimestampMiddleware``).

    Only ``HumanMessage`` carries a per-message wall-clock stamp; no LangChain
    or provider field records when an ``AIMessage`` was produced, so assistant
    rows are seeded without ``createdAt`` (the client falls back to arrival
    order). ``None`` for historical checkpoints that predate the stamp.
    """
    additional_kwargs = getattr(message, "additional_kwargs", None) or {}
    sent_at_iso = additional_kwargs.get("sent_at")
    if not isinstance(sent_at_iso, str) or not sent_at_iso:
        return None
    try:
        return int(datetime.fromisoformat(sent_at_iso).timestamp() * 1000)
    except (TypeError, ValueError):
        logger.warning("replay_history: could not parse sent_at %r", sent_at_iso)
        return None


def _content_reasoning(message: AIMessage) -> str:
    """Best-effort reasoning text from an assistant message's ``thinking`` /
    ``reasoning`` content blocks (Anthropic-style)."""
    content = message.content
    if not isinstance(content, list):
        return ""
    out: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype in {"thinking", "reasoning"}:
            out.append(str(block.get(btype, "") or block.get("text", "")))
    return "".join(out)


def _window_to_last_turns(
    messages: Sequence[BaseMessage], max_turns: int
) -> Sequence[BaseMessage]:
    """Keep only the last ``max_turns`` user turns — a ``HumanMessage`` and every
    message after it — so a windowed snapshot stays whole-turned (a tool result
    is never split from its requesting assistant message). Older history loads
    lazily rather than riding the conversation stream's bounded log."""
    if max_turns <= 0:
        return messages
    human_positions = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    if len(human_positions) <= max_turns:
        return messages
    return messages[human_positions[-max_turns] :]


async def replay_history(
    writer: StateWriterProtocol,
    thread_id: str,
    messages: Sequence[BaseMessage],
    *,
    max_turns: int | None = None,
    summarization_message_id_prefix: str = SUMMARIZATION_MESSAGE_ID_PREFIX,
) -> int:
    """Seed prior turns as rows. Returns the next free ``ordinal`` so the live
    run's messages continue after history (the caller passes it to the
    StateTranslator as ``ordinal_start``).

    ``max_turns`` windows the replay to the most recent turns (the conversation
    stream's log is capped, so a full transcript can't always fit at offset 0);
    ordinals still start at 0 for the windowed set.

    Every message and tool row also gets the conversation-global ``order`` the
    live path stamps (see ``StateTranslator._render_order``) — one counter,
    incremented in emission order across messages and their tool calls, so
    ``next_live_order`` (counting the same messages) picks up exactly where
    this leaves off and a tool never falls back to the client's "attach to the
    last message" default (the stranded-card bug the live path fixed).

    Tool rows are collected with their args from the requesting ``AIMessage`` and
    completed with the result from the matching ``ToolMessage`` (keyed by real
    ``tool_call_id``), then upserted whole — the same row shape the live path
    produces, so a resume run's live result upsert merges idempotently.

    A ``HumanMessage`` row gets ``createdAt`` when its checkpoint carries a
    ``sent_at`` stamp (see ``_human_created_at``); an ``AIMessage`` row never
    does — nothing in the checkpoint records when it was produced — so the
    client's arrival-order fallback applies to replayed assistant rows.
    """
    if max_turns is not None:
        messages = _window_to_last_turns(messages, max_turns)
    ordinal = 0
    order = 0
    tool_rows: dict[str, dict[str, Any]] = {}

    for message in messages:
        msg_id = getattr(message, "id", None)
        if isinstance(message, HumanMessage):
            if not msg_id or (
                isinstance(msg_id, str)
                and msg_id.startswith(summarization_message_id_prefix)
            ):
                continue
            value: dict[str, Any] = {
                "id": msg_id,
                "threadId": thread_id,
                "role": "user",
                "ordinal": ordinal,
                "order": order,
                "text": _content_text(message.content),
            }
            created_at = _human_created_at(message)
            if created_at is not None:
                value["createdAt"] = created_at
            await writer.upsert(TYPE_MESSAGE, msg_id, value)
            ordinal += 1
            order += 1
        elif isinstance(message, AIMessage):
            if not msg_id or (
                isinstance(msg_id, str)
                and msg_id.startswith(summarization_message_id_prefix)
            ):
                continue
            value = {
                "id": msg_id,
                "threadId": thread_id,
                "role": "assistant",
                "ordinal": ordinal,
                "order": order,
                "text": _content_text(message.content),
            }
            reasoning = _content_reasoning(message)
            if reasoning:
                value["reasoning"] = reasoning
            await writer.upsert(TYPE_MESSAGE, msg_id, value)
            ordinal += 1
            order += 1
            for call in message.tool_calls:
                call_id = call.get("id")
                if not call_id:
                    continue
                tool_rows[call_id] = {
                    "id": call_id,
                    "threadId": thread_id,
                    "name": to_ag_ui_tool_name(call.get("name", "")),
                    "argsText": to_ag_ui_json(call.get("args", {})),
                    "order": order,
                }
                order += 1
        elif isinstance(message, ToolMessage):
            row = tool_rows.get(message.tool_call_id)
            if row is None:
                continue
            if getattr(message, "status", None) == "error":
                row["isError"] = True
                row["result"] = to_ag_ui_json({"error": _content_text(message.content)})
            else:
                row["result"] = unwrap_tool_result(row["name"], message)

    for row in tool_rows.values():
        await writer.upsert(TYPE_TOOL, row["id"], row)

    return ordinal
