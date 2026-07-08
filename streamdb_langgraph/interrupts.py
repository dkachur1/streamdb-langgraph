"""Adapters from LangGraph interrupts to AG-UI interrupt payloads."""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal, TypedDict

from langgraph.errors import GraphInterrupt
from langgraph.types import Interrupt

from streamdb_langgraph.serialization import to_ag_ui_content

# Channel for pending LangGraph interrupts. Mirrors the (private) LangGraph constant.
INTERRUPT_WRITE_CHANNEL = "__interrupt__"


def is_interrupt_error(error: object) -> bool:
    """True if ``error`` is (or wraps) a LangGraph interrupt.

    A tool that raises ``GraphInterrupt`` (ask_question / request_review) is
    *paused*, not failed. Classify strictly on the exception TYPE — a bare
    ``GraphInterrupt`` / ``Interrupt``, a tuple of ``Interrupt`` (the exception's
    ``args``), or a ``BaseExceptionGroup`` wrapping one. A stringified error is
    deliberately NOT matched: substring-matching ``"Interrupt(value="`` on a
    message string misclassifies a genuine tool failure that merely quotes that
    text as a paused interrupt, stranding the tool card with no result/error.
    """
    if isinstance(error, (GraphInterrupt, Interrupt)):
        return True
    if isinstance(error, BaseExceptionGroup):
        return any(is_interrupt_error(exc) for exc in error.exceptions)
    if isinstance(error, (tuple, list)):
        return any(is_interrupt_error(item) for item in error)
    return False


def interrupts_from_pending_writes(
    pending_writes: Sequence[tuple[str, str, Any]] | None,
) -> list[Interrupt]:
    """Every interrupt staged on a checkpoint's ``__interrupt__`` channel.

    The authoritative, non-racy interrupt source shared by the live-run
    finaliser and ``/history`` restore: a paused run commits its pending
    ``__interrupt__`` writes to the checkpoint as it unwinds, so this read sees
    committed truth regardless of stream teardown timing. Collects across ALL
    tasks (parallel tool calls each stage their own write) — unlike a
    first-match read. A write value is either one ``Interrupt`` or a sequence of
    them (mirrors LangGraph's ``tasks_w_writes``)."""
    found: list[Interrupt] = []
    for _task_id, channel, value in pending_writes or ():
        if channel != INTERRUPT_WRITE_CHANNEL:
            continue
        found.extend(_coerce_interrupts(value))
    return found


def interrupts_from_channel_values(
    channel_values: Mapping[str, Any] | None,
) -> list[Interrupt]:
    """Interrupts committed to a checkpoint's ``__interrupt__`` channel values.

    The committed-values companion to :func:`interrupts_from_pending_writes`.
    Once a paused run's ``__interrupt__`` writes are folded into the checkpoint's
    channel values (rather than left as pending writes), a pending-writes read
    alone misses them — so an "is a question still open?" test must consult both
    sources to stay non-racy across the fold."""
    if not channel_values:
        return []
    return _coerce_interrupts(channel_values.get(INTERRUPT_WRITE_CHANNEL))


def _coerce_interrupts(value: Any) -> list[Interrupt]:
    """One ``__interrupt__`` write value (a single ``Interrupt`` or a sequence of
    them, mirroring LangGraph's ``tasks_w_writes``) as a flat interrupt list."""
    if value is None:
        return []
    items = (
        value
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes))
        else [value]
    )
    return [item for item in items if isinstance(item, Interrupt)]


class AgUiInterrupt(TypedDict, total=False):
    """AG-UI Interrupt item.

    Kept local until the Python ``ag_ui.core`` package ships typed interrupt
    models for ``RunFinishedEvent.outcome``.

    Keys are wire-format (camelCase for multi-word names): ``outcome`` rides
    ``RunFinishedEvent`` as a pydantic *extra* field, so its dict contents are
    dumped verbatim — ``by_alias`` only renames declared model fields.
    """

    id: str
    reason: str
    message: str
    toolCallId: str
    metadata: dict[str, Any]


class AgUiInterruptOutcome(TypedDict):
    """``RunFinishedEvent.outcome`` for paused AG-UI runs."""

    type: Literal["interrupt"]
    interrupts: list[AgUiInterrupt]


AG_UI_INTERRUPT_REASON_BY_LANGGRAPH_TYPE: dict[str, str] = {
    "ask_question": "input_required",
    "request_review": "confirmation",
}

# Tools whose result IS a structured decision artifact (the user's answer /
# review decision). Their durable ``tool`` row carries that artifact object
# verbatim as ``result`` — not the serialised ToolMessage — so the receipt UI
# reads a clean object identically live and on replay. See
# ``StateTranslator._tool_result`` and ``history_replay.replay_history``.
INTERRUPT_TOOL_NAMES: frozenset[str] = frozenset({"ask_question", "request_review"})


def unwrap_tool_result(tool_name: str, output: Any) -> Any:
    """The clean, non-error ``result`` to project for a settled tool call.

    An interrupt tool (ask_question / request_review) projects its structured
    decision artifact verbatim — read off ``output.artifact`` (a real
    ``ToolMessage``) or ``output["artifact"]`` (a dict-shaped output) — so the
    receipt UI gets a clean object, never the serialised ToolMessage string it
    can't parse. Every other tool keeps its plain content string: ``output`` is
    either the tool's raw return value (the live path's ``tc.output``) or a
    ``ToolMessage`` wrapping it (the replayed checkpoint's shape), so unwrap
    ``.content`` when present and fall back to ``output`` itself otherwise.
    Shared by ``StateTranslator._tool_result`` (live) and
    ``history_replay.replay_history`` (replayed) so the two produce
    byte-identical rows for the same tool call.
    """
    if tool_name in INTERRUPT_TOOL_NAMES:
        artifact = getattr(output, "artifact", None)
        if artifact is None and isinstance(output, dict):
            artifact = output.get("artifact")
        if artifact is not None:
            return artifact
    return to_ag_ui_content(getattr(output, "content", output))


def ag_ui_interrupt_reason(interrupt_type: str) -> str:
    return AG_UI_INTERRUPT_REASON_BY_LANGGRAPH_TYPE.get(
        interrupt_type, f"langgraph:{interrupt_type}"
    )


def langgraph_interrupt_to_ag_ui_interrupt(
    interrupt: Interrupt,
) -> AgUiInterrupt | None:
    """Convert one LangGraph interrupt object into AG-UI interrupt shape.

    ``stream.interrupts()`` is typed ``list[Any]``, but the runtime objects are
    genuine ``langgraph.types.Interrupt`` instances; annotating the parameter is
    the narrowing point. ``Interrupt.value`` stays ``Any``, so the payload guard
    below is still load-bearing.
    """
    payload = interrupt.value
    if not isinstance(payload, dict):
        return None

    interrupt_type = str(payload.get("type") or "unknown")
    item: AgUiInterrupt = {
        "id": interrupt.id or uuid.uuid4().hex,
        "reason": ag_ui_interrupt_reason(interrupt_type),
        "metadata": payload,
    }
    prompt = payload.get("message")
    if isinstance(prompt, str) and prompt:
        item["message"] = prompt
    tool_call_id = payload.get("tool_call_id")
    if isinstance(tool_call_id, str) and tool_call_id:
        item["toolCallId"] = tool_call_id
    return item


def ag_ui_interrupt_rows(
    interrupts: Iterable[Interrupt], thread_id: str
) -> list[dict[str, Any]]:
    """Open interrupts as AG-UI ``interrupt`` STATE rows for one thread.

    The single shape the durable state stream carries in its ``interrupt``
    collection. Shared by the live-run finaliser (``StateTranslator``), the
    replay bridge (``history_replay``), and ``/history`` restore so all three
    project an open interrupt identically — ``threadId`` is the row's only
    routing key; the id falls back to a fresh token only when LangGraph supplied
    none.

    Disambiguates on a raw-id collision: LangGraph's ``interrupt()`` id is a hash
    of the checkpoint namespace alone (no per-call index folded in), so multiple
    ``interrupt()`` calls from parallel tool calls in the same ``ToolNode`` can
    legitimately share one raw id
    (see https://github.com/langchain-ai/langgraph/issues/6626). Upserting two
    rows under the same key would silently drop one interrupt from the
    collection (last-writer-wins), so a colliding row is re-keyed on its
    ``toolCallId`` (unique per parallel call) — or an ordinal suffix, if even
    that is missing — before it lands on the stream. This keeps every pending
    interrupt visible; it does NOT by itself fix resuming them correctly, since
    LangGraph's own resume-value routing keys off the same colliding raw id
    upstream of this translator.
    """
    rows: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    for interrupt in interrupts:
        item = langgraph_interrupt_to_ag_ui_interrupt(interrupt)
        if item is None:
            continue
        row: dict[str, Any] = dict(item)
        row.setdefault("id", f"interrupt_{uuid.uuid4().hex}")
        raw_id = row["id"]
        count = seen.get(raw_id, 0)
        seen[raw_id] = count + 1
        if count > 0:
            disambiguator = row.get("toolCallId") or str(count)
            row["id"] = f"{raw_id}:{disambiguator}"
        row["threadId"] = thread_id
        rows.append(row)
    return rows
