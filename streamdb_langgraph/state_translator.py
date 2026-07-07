"""Translate LangGraph v3 typed projections into state-protocol row upserts.

Consumes the typed projections from :func:`streamdb_langgraph.stream.make_run_stream`
and, instead of emitting AG-UI lifecycle events, materialises rows the frontend
``createStreamDB`` keeps last-writer-wins:

  * messages channel → one ``message`` ANCHOR row (id, role, ordinal, order)
    plus append-only ``messageChunk`` rows that each carry only the NEW
    characters since the last flush. Text rides the default channel (key
    ``msg_id:seq``); reasoning rides ``channel="reasoning"`` chunks on their own
    non-colliding key space (``msg_id:r:seq``) with a separate seq counter. The
    client reassembles each channel by concatenating that message's chunks in
    ``seq`` order (assemble.ts); the anchor's own ``text`` / ``reasoning`` fields
    stay empty for live streams and are only a fallback for replayed / backfilled
    rows that have no chunks. Streaming deltas instead of re-upserting the whole
    growing prefix turns the per-message wire cost from ~L²/48 into ~L.
  * tool_calls channel → ``tool`` row: args, then the result, are successive
    upserts of the same key. No envelope, no resume suppression.
  * values channel → one ``agentState`` row (the ``project_client_state``
    allowlist), re-upserted whole on change. No jsonpatch, no re-baseline.
  * custom channel → ``suggestion`` / ``toolSummary`` / ``toolAction`` rows +
    ``effect`` rows for frontend updates. No custom-event bus.
  * lifecycle → one ``run`` row (running → complete/error/interrupt).
  * interrupts → ``interrupt`` rows; the run row's status carries the outcome.
  * subgraphs channel → a subagent's child model text + tool calls become
    ``message`` / ``tool`` rows tagged with the dispatch ``agentId`` (the parent
    tool_call_id), so the SubagentCard live-queries the nested thread. The
    main transcript query filters those out (``agentId is null``).

Cancellation aborts on CancelledError. Graph exceptions
set the run row to ``error`` before re-raising. ``close()`` (snapshot-end) is the
caller's responsibility via :meth:`_finalise`.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from langgraph.stream import AsyncGraphRunStream

from streamdb_langgraph.interrupts import (
    INTERRUPT_TOOL_NAMES,
    ag_ui_interrupt_rows,
    is_interrupt_error,
    unwrap_tool_result,
)
from streamdb_langgraph.serialization import to_ag_ui_json, to_ag_ui_tool_name
from streamdb_langgraph.state_projection import project_client_state
from streamdb_langgraph.state_protocol import (
    RUN_ROW_ID,
    TYPE_AGENT_STATE,
    TYPE_EFFECT,
    TYPE_INTERRUPT,
    TYPE_MESSAGE,
    TYPE_MESSAGE_CHUNK,
    TYPE_REPORT,
    TYPE_RUN,
    TYPE_SUGGESTION,
    TYPE_TOOL,
    TYPE_TOOL_ACTION,
    TYPE_TOOL_SUMMARY,
    StateWriterProtocol,
)

logger = logging.getLogger(__name__)

_MODEL_NODE = "model"

# Message ids with this prefix are compaction/summary artifacts the paired agent
# mints (e.g. a summarization middleware) and must not render as chat. The
# translator filters them off the messages channel by this prefix. It is
# duplicated by design from whatever component mints those ids — the two packages
# do not depend on each other — so it is exposed as a constructor override
# (``summarization_message_id_prefix``) for consumers whose middleware uses a
# different prefix. See ``tests/test_summarization_prefix_contract.py``.
SUMMARIZATION_MESSAGE_ID_PREFIX = "summarization-"

# Emit a reasoning ``messageChunk`` (channel="reasoning") carrying only the new
# reasoning every ~this many chars — finer than the text cadence so the smaller
# secondary stream renders smoothly without one row per token.
_REASONING_FLUSH_CHARS = 24

# Emit a text ``messageChunk`` row carrying only the new text every ~this many
# chars. Larger than the reasoning cadence to keep the chunk-row count reasonable;
# the client can restore finer-grained rendering with its own smoothing. Tunable.
_CHUNK_FLUSH_CHARS = 80


def _now_ms() -> int:
    """Epoch-millis wall-clock for a message row's ``createdAt`` (display order /
    timestamp). Stamped once per message so re-flush upserts keep it stable."""
    return int(time.time() * 1000)


# Custom content types that map to dedicated collections or are dropped because
# the same data rides the ``agentState`` row.
_STATE_CHANNEL_CUSTOM: frozenset[str] = frozenset(
    {
        "tool_start",
        "tool_end",
        "tool_error",
        "plan_mode_entered",
        "plan_mode_exited",
        "context_window",
        "task_snapshot",
    }
)


class _ChannelCancelled(Exception):
    """A stream projection raised CancelledError without parent cancellation."""


def _exception_message(exc: BaseException) -> str:
    if isinstance(exc, BaseExceptionGroup):
        for child in exc.exceptions:
            if not isinstance(child, asyncio.CancelledError):
                return _exception_message(child)
    return str(exc)


def _has_channel_cancelled(exc: BaseException) -> bool:
    if isinstance(exc, _ChannelCancelled):
        return True
    if isinstance(exc, BaseExceptionGroup):
        return any(_has_channel_cancelled(child) for child in exc.exceptions)
    return False


@dataclass(frozen=True)
class EmittedTurn:
    """What a run has put on the stream so far — the cancel-salvage view.

    ``texts`` maps every top-level message id the run emitted to its streamed
    text (``""`` for a content-free anchor); ``ordinals`` gives each id's live
    ordinal so the salvage can archive in transcript order. Subagent rows are
    excluded — they live on nested threads and never take top-level ordinals.
    """

    texts: dict[str, str]
    ordinals: dict[str, int]
    user_message_ids: frozenset[str]
    tool_call_ids: frozenset[str]


class StateTranslator:
    """Translate an :class:`AsyncGraphRunStream` to state-protocol row upserts."""

    def __init__(
        self,
        writer: StateWriterProtocol,
        *,
        thread_id: str,
        run_id: str | None = None,
        run_row_id: str = RUN_ROW_ID,
        summarization_message_id_prefix: str = SUMMARIZATION_MESSAGE_ID_PREFIX,
    ) -> None:
        self.writer = writer
        self.thread_id = thread_id
        self.run_id = run_id or f"run_{uuid.uuid4().hex}"
        # The stream is per-conversation, so there is one logical current-run row;
        # it is keyed by this stable id (LWW) so the frontend runtime — which reads
        # the run row by a fixed key — binds ``isRunning``. Defaults to the shared
        # ``RUN_ROW_ID`` ("run"); overridable only if a consumer keys it otherwise.
        self.run_row_id = run_row_id
        self._summarization_prefix = summarization_message_id_prefix
        # Live messages continue after any history seeded ahead of the run
        # (set_ordinal_start), so message rows share one monotonic ordinal.
        self._ordinal = 0
        self._ordinals: dict[str, int] = {}
        # The single sequence the client renders by. EVERY row (message + tool)
        # gets a conversation-global ``order`` the first time it's upserted, reused
        # on re-upsert so a late result never moves it. assemble sorts messages by
        # it and slots each tool after the message it followed — there is no
        # ``messageId`` anchor and no "current message" pointer (both produced the
        # empty-anchor bug that stranded cards at the bottom). Seeded across runs
        # by ``set_order_start`` so a resumed tool sorts after prior history.
        self._order_seq = 0
        self._order_by_key: dict[str, int] = {}
        self._errored = False
        self._run_started = False
        # Subagents: each nested handle (keyed by trigger_call_id) tags its
        # message/tool rows with the dispatch ``tool_call_id`` as ``agentId`` so
        # the SubagentCard live-queries them. The link arrives on subagent_start;
        # the drain awaits it (an event per subagent) before emitting, so a fast
        # first token can't leak an untagged row into the parent transcript.
        self._subagent_tool_call: dict[str, str] = {}
        self._subagent_ready: dict[str, asyncio.Event] = {}
        self._subagent_ordinal: dict[str, int] = {}
        self._subagent_msg_ordinal: dict[str, dict[str, int]] = {}
        # Partial-turn capture for cancel salvage: every top-level message this
        # run emitted (id → streamed text so far), which of those are user rows,
        # and the top-level tool calls. Read via ``emitted_turn()`` when a user
        # cancel needs to freeze the partial output into the checkpoint.
        self._emitted_texts: dict[str, str] = {}
        self._user_message_ids: set[str] = set()
        self._emitted_tool_call_ids: set[str] = set()

    def emitted_turn(self) -> EmittedTurn:
        """Snapshot of this run's emitted top-level rows for cancel salvage."""
        return EmittedTurn(
            texts=dict(self._emitted_texts),
            ordinals=dict(self._ordinals),
            user_message_ids=frozenset(self._user_message_ids),
            tool_call_ids=frozenset(self._emitted_tool_call_ids),
        )

    def set_ordinal_start(self, ordinal: int) -> None:
        """Start live message ordinals at ``ordinal`` — called after history is
        seeded ahead of the run, before the stream runs so no live message has
        been assigned an ordinal yet."""
        self._ordinal = ordinal

    def set_order_start(self, order: int) -> None:
        """Start the conversation-global ``order`` counter past all history rows,
        so a resumed tool/message sorts after prior turns instead of jumping to
        the top. Mirrors ``set_ordinal_start``; fed the cold-seed's total row
        count (messages + tool-calls)."""
        self._order_seq = order

    def _next_ordinal(self, message_id: str) -> int:
        existing = self._ordinals.get(message_id)
        if existing is not None:
            return existing
        ordinal = self._ordinal
        self._ordinal += 1
        self._ordinals[message_id] = ordinal
        return ordinal

    def _render_order(self, key: str) -> int:
        """First-seen conversation-global position for a row key, stable across
        re-upserts (see ``_order_by_key``). Synchronous — safe to call from the
        concurrent channel tasks without a lock (no await between read and write)."""
        existing = self._order_by_key.get(key)
        if existing is not None:
            return existing
        order = self._order_seq
        self._order_seq += 1
        self._order_by_key[key] = order
        return order

    async def _set_run(self, status: str, *, interrupt: Any = None) -> None:
        value: dict[str, Any] = {
            "id": self.run_row_id,
            "threadId": self.thread_id,
            # The enqueued run id; the client reads it off this row to attribute
            # message feedback. Wired in at construction — never the fallback id.
            "runId": self.run_id,
            "status": status,
        }
        if interrupt is not None:
            value["interrupt"] = interrupt
        await self.writer.upsert(TYPE_RUN, self.run_row_id, value)

    async def emit_run_started(self) -> None:
        if self._run_started:
            return
        self._run_started = True
        await self._set_run("running")
        # Drop the prior turn's predicted response so the suggestion row reflects
        # only the current turn; the client reads the collection directly.
        await self.writer.delete(TYPE_SUGGESTION, self.thread_id)

    async def emit_run_finished(self) -> None:
        """Terminal: mark the run complete and close the stream. Interface
        so a plan-mode-only
        branch is translator-agnostic."""
        await self._set_run("complete")
        await self.writer.close()

    async def emit_empty_run(self) -> None:
        await self.emit_run_started()
        await self.emit_run_finished()

    async def run(
        self,
        stream: AsyncGraphRunStream,
        *,
        initial_state: dict[str, Any] | None = None,
        emit_run_started: bool = True,
        pre_finish: Awaitable[None] | None = None,
    ) -> None:
        """``pre_finish`` is awaited right before the terminal run row so callers
        can settle out-of-band work whose result the client reads on run-finish
        (e.g. the background conversation-name DB write). Skipped on the error
        path, which has already closed the stream. Interface parity with
        ``run``."""
        if emit_run_started:
            await self.emit_run_started()
        if initial_state is not None:
            # Seed the agentState row so the client reflects prior-turn state
            # (tasks, plan-mode) the moment the run opens — the values channel
            # re-upserts the whole row once the graph emits its first snapshot.
            await self.writer.upsert(
                TYPE_AGENT_STATE,
                self.thread_id,
                {"id": self.thread_id, **initial_state},
            )
        try:
            async with stream:
                async with asyncio.TaskGroup() as tg:
                    tg.create_task(self._run_channel(self._stream_messages(stream)))
                    tg.create_task(self._run_channel(self._stream_custom(stream)))
                    tg.create_task(self._run_channel(self._stream_tool_calls(stream)))
                    tg.create_task(self._run_channel(self._stream_subgraphs(stream)))
                    tg.create_task(self._run_channel(self._stream_values(stream)))
        except asyncio.CancelledError:
            self._errored = True
            raise
        except Exception as exc:
            if _has_channel_cancelled(exc):
                self._errored = True
                raise asyncio.CancelledError from exc
            logger.exception("state translator caught graph exception")
            self._errored = True
            await self._set_run("error", interrupt={"message": _exception_message(exc)})
            await self.writer.close()
            raise
        finally:
            await self._finalise(stream, pre_finish=pre_finish)

    async def _run_channel(self, awaitable: Any) -> None:
        try:
            await awaitable
        except asyncio.CancelledError as exc:
            task = asyncio.current_task()
            if task is not None and task.cancelling():
                raise
            raise _ChannelCancelled from exc

    # ── messages ────────────────────────────────────────────────────────────
    async def _stream_messages(self, stream: AsyncGraphRunStream) -> None:
        async for message in stream.extensions["messages"]:
            if getattr(message, "node", None) != _MODEL_NODE:
                continue
            message_id = getattr(message, "message_id", None) or ""
            if message_id.startswith(self._summarization_prefix):
                continue
            await self._consume_message(message)

    async def _consume_message(self, message: Any) -> None:
        msg_id = message.message_id or f"msg_{uuid.uuid4().hex}"
        ordinal = self._next_ordinal(msg_id)
        created_at = _now_ms()
        self._emitted_texts.setdefault(msg_id, "")
        # Emit the anchor row up-front (even before any text) so a tool-only turn
        # still has a bubble: its tools slot after this message's ``order``, not
        # the previous turn's. The first upsert stamps the message's ``order``.
        # Text and reasoning ride ``messageChunk`` rows (below), not this row.
        await self._upsert_message(msg_id, ordinal, created_at)
        logger.debug("state: message msg_id=%s ordinal=%s", msg_id, ordinal)
        await self._stream_message_text(message, msg_id, ordinal, created_at)

    async def _stream_message_text(
        self,
        message: Any,
        msg_id: str,
        ordinal: int,
        created_at: int,
        *,
        agent_id: str | None = None,
    ) -> None:
        """Drive one model message's deltas: emit a ``messageChunk`` INSERT per
        text flush AND per reasoning flush, each carrying only the new characters
        since the last flush of its own kind. Text rides the default channel
        (key ``msg_id:seq``); reasoning rides ``channel="reasoning"`` chunks on a
        separate key space (``msg_id:r:seq``) with its own seq counter, so the two
        streams never collide. The anchor's ``reasoning`` field is NOT re-shipped
        live — it stays a replay/backfill fallback. Shared by the top-level and
        subagent paths (``agent_id`` tags every chunk)."""
        text = ""
        reasoning = ""
        # Chars already flushed on each channel. Coalescing bounds the append log —
        # a per-token write floods the conversation stream, trimming the head
        # snapshot a mid-run joiner reads.
        text_flushed = 0
        reasoning_flushed = 0
        seq = 0
        reasoning_seq = 0
        async for event in message:
            if event.get("event") != "content-block-delta":
                continue
            delta = event.get("delta")
            if not isinstance(delta, dict):
                continue
            dtype = delta.get("type")
            if dtype == "reasoning-delta":
                token = delta.get("reasoning", "")
                if not token:
                    continue
                reasoning += token
                if len(reasoning) - reasoning_flushed >= _REASONING_FLUSH_CHARS:
                    await self._emit_message_chunk(
                        msg_id,
                        reasoning_seq,
                        reasoning[reasoning_flushed:],
                        agent_id=agent_id,
                        channel="reasoning",
                    )
                    reasoning_flushed = len(reasoning)
                    reasoning_seq += 1
            elif dtype == "text-delta":
                token = delta.get("text", "")
                if not token:
                    continue
                text += token
                if agent_id is None:
                    self._emitted_texts[msg_id] = text
                if len(text) - text_flushed >= _CHUNK_FLUSH_CHARS:
                    await self._emit_message_chunk(
                        msg_id, seq, text[text_flushed:], agent_id=agent_id
                    )
                    text_flushed = len(text)
                    seq += 1
        # Clean drain: land the remaining tails. A content-free turn keeps just the
        # up-front anchor row (its tool-calls render on it); only write when there
        # is a delta to add.
        if len(reasoning) > reasoning_flushed:
            await self._emit_message_chunk(
                msg_id,
                reasoning_seq,
                reasoning[reasoning_flushed:],
                agent_id=agent_id,
                channel="reasoning",
            )
        if len(text) > text_flushed:
            await self._emit_message_chunk(
                msg_id, seq, text[text_flushed:], agent_id=agent_id
            )

    async def _emit_message_chunk(
        self,
        msg_id: str,
        seq: int,
        text: str,
        *,
        agent_id: str | None = None,
        channel: str | None = None,
    ) -> None:
        """Append-only ``messageChunk`` row carrying ONLY the new text. Keyed
        ``msg_id:seq`` for the default (text) channel and ``msg_id:r:seq`` for
        ``channel="reasoning"`` so the two channels never share a key; a resume
        re-applies each idempotently (last-writer-wins). The client concatenates a
        message's chunks per channel by ``seq``. ``channel`` absent = text."""
        chunk_id = f"{msg_id}:r:{seq}" if channel == "reasoning" else f"{msg_id}:{seq}"
        value: dict[str, Any] = {
            "id": chunk_id,
            "messageId": msg_id,
            "threadId": self.thread_id,
            "seq": seq,
            "text": text,
        }
        if channel is not None:
            value["channel"] = channel
        if agent_id:
            value["agentId"] = agent_id
        await self.writer.upsert(TYPE_MESSAGE_CHUNK, chunk_id, value)

    async def _upsert_message(
        self,
        msg_id: str,
        ordinal: int,
        created_at: int,
        *,
        agent_id: str | None = None,
    ) -> None:
        """Upsert the message ANCHOR row — id/role/ordinal/order/createdAt only.
        Streamed text AND reasoning ride ``messageChunk`` rows (separate
        channels), so ``text`` stays ``""`` and no ``reasoning`` is written live;
        both anchor fields are only the fallback the client reads for replayed /
        backfilled rows that carry no chunks."""
        value: dict[str, Any] = {
            "id": msg_id,
            "threadId": self.thread_id,
            "role": "assistant",
            "ordinal": ordinal,
            "order": self._render_order(msg_id),
            "createdAt": created_at,
            "text": "",
        }
        if agent_id:
            value["agentId"] = agent_id
        await self.writer.upsert(TYPE_MESSAGE, msg_id, value)

    async def emit_user_message(
        self,
        msg_id: str,
        text: str,
        attachments: list[dict[str, Any]] | None = None,
    ) -> None:
        """Put the new user turn on the stream as a ``message`` row.

        The graph's message stream only carries model-node output, so the human
        turn never flows through ``_consume_message`` — it is emitted here. But it
        goes through the SAME ``_next_ordinal`` counter as the assistant rows, so
        the whole conversation shares one monotonic sequence with no second writer
        or guessed ordinal. Same id as the client's optimistic row → last-writer-
        wins reconciles them; re-emit on a recovery re-claim is idempotent.
        ``attachments`` persist the file chips so they survive the echo + reload."""
        self._emitted_texts[msg_id] = text
        self._user_message_ids.add(msg_id)
        value: dict[str, Any] = {
            "id": msg_id,
            "threadId": self.thread_id,
            "role": "user",
            "ordinal": self._next_ordinal(msg_id),
            "order": self._render_order(msg_id),
            "createdAt": _now_ms(),
            "text": text,
        }
        if attachments:
            value["attachments"] = attachments
        await self.writer.upsert(
            TYPE_MESSAGE,
            msg_id,
            value,
            # The frame's txid is the message id, so the client's optimistic send
            # (db.utils.awaitTxId(id)) confirms the moment this row syncs back.
            txid=msg_id,
        )

    # ── tool calls ──────────────────────────────────────────────────────────
    async def _stream_tool_calls(self, stream: AsyncGraphRunStream) -> None:
        async with asyncio.TaskGroup() as tg:
            async for tc in stream.extensions["tool_calls"]:
                tg.create_task(self._consume_tool_call_execution(tc))

    def _tool_result(self, tc: Any, tool_name: str) -> Any:
        """The ``result`` to project for a settled tool call.

        An errored call projects the wrapped error; otherwise defers to
        ``unwrap_tool_result`` (shared with ``history_replay.replay_history``) so
        a live and a replayed interrupt-tool row carry the identical artifact
        shape.
        """
        if tc.error:
            return to_ag_ui_json({"error": str(tc.error)})
        return unwrap_tool_result(tool_name, tc.output)

    def _result_txid(self, tool_name: str, tool_call_id: str) -> str | None:
        """The reconciliation txid for a tool's result frame.

        Interrupt-tool results stamp ``txid=tool_call_id`` so the client's
        optimistic resume (``db.utils.awaitTxId``) holds its overlay until the real
        result syncs back — the same trick the user message uses with its id. Other
        tools need no reconciliation handle (nothing writes them optimistically)."""
        return tool_call_id if tool_name in INTERRUPT_TOOL_NAMES else None

    async def _consume_tool_call_execution(self, tc: Any) -> None:
        tool_name = to_ag_ui_tool_name(tc.tool_name)
        args_text = to_ag_ui_json(tc.input) if tc.input else ""
        self._emitted_tool_call_ids.add(tc.tool_call_id)
        order = self._render_order(tc.tool_call_id)
        logger.debug(
            "state: tool call id=%s order=%s name=%s",
            tc.tool_call_id,
            order,
            tool_name,
        )
        # First upsert: the call with its args. A resume replays the same id —
        # re-upserting is a no-op (last-writer-wins), so no suppression needed.
        await self.writer.upsert(
            TYPE_TOOL,
            tc.tool_call_id,
            {
                "id": tc.tool_call_id,
                "threadId": self.thread_id,
                "name": tool_name,
                "argsText": args_text,
                "order": order,
            },
        )
        async for _ in tc.output_deltas:
            pass
        # An interrupted tool (ask_question raising GraphInterrupt) is paused, not
        # failed — the run row's interrupt status carries it; the resumed run
        # upserts the real result. The error arrives wrapped/stringified, so use
        # the same robust detector is_interrupt_error provides.
        if is_interrupt_error(tc.error):
            return
        is_error = bool(tc.error)
        result = self._tool_result(tc, tool_name)
        # Second upsert: the same row, now with the result. Same ``order`` (the
        # result must not move the card). No separate RESULT event, so no
        # duplicate-card synthesis on the client.
        value: dict[str, Any] = {
            "id": tc.tool_call_id,
            "threadId": self.thread_id,
            "name": tool_name,
            "argsText": args_text,
            "result": result,
            "order": order,
        }
        if is_error:
            value["isError"] = True
        await self.writer.upsert(
            TYPE_TOOL,
            tc.tool_call_id,
            value,
            txid=self._result_txid(tool_name, tc.tool_call_id),
        )

    # ── custom ──────────────────────────────────────────────────────────────
    async def _stream_custom(self, stream: AsyncGraphRunStream) -> None:
        async for payload in stream.extensions["custom"]:
            await self._handle_custom(payload)

    async def _handle_custom(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        content_type = payload.get("content_type")
        if content_type == "tool_summary":
            text = payload.get("text")
            tool_call_id = payload.get("tool_call_id")
            if (
                isinstance(text, str)
                and text
                and isinstance(tool_call_id, str)
                and tool_call_id
            ):
                await self.writer.upsert(
                    TYPE_TOOL_SUMMARY,
                    tool_call_id,
                    {"id": tool_call_id, "toolCallId": tool_call_id, "text": text},
                )
            return
        if content_type == "tool_action":
            # ToolSummaryMiddleware's per-edit plain-language before/after,
            # one row per tool call (last-writer-wins). A dedicated branch (not
            # ``_STATE_CHANNEL_CUSTOM``, which drops its members) so it doesn't
            # ALSO fall through to the generic effect below.
            tool_call_id = payload.get("tool_call_id")
            if isinstance(tool_call_id, str) and tool_call_id:
                added = payload.get("added")
                removed = payload.get("removed")
                value: dict[str, Any] = {
                    "id": tool_call_id,
                    "threadId": self.thread_id,
                    "added": added if isinstance(added, list) else [],
                    "removed": removed if isinstance(removed, list) else [],
                }
                summary = payload.get("summary")
                if isinstance(summary, str) and summary:
                    value["summary"] = summary
                await self.writer.upsert(TYPE_TOOL_ACTION, tool_call_id, value)
            return
        if content_type == "tool_end":
            ops = payload.get("frontend_updates")
            if isinstance(ops, list) and ops:
                raw_tcid = payload.get("tool_call_id")
                tool_call_id = raw_tcid if isinstance(raw_tcid, str) else None
                for op in ops:
                    # A generated report is STATE (it persists across the turn), so
                    # it rides its own collection — not the one-shot effect log.
                    if isinstance(op, dict) and op.get("type") == "report_generated":
                        await self._emit_report(op, tool_call_id)
                    else:
                        await self._emit_effect(op)
            return
        if content_type in _STATE_CHANNEL_CUSTOM:
            # Rides the agentState row (or intentionally dropped).
            return
        if content_type == "subagent_start":
            # Links a subagent to its dispatch tool_call_id; the subgraph drain
            # tags the nested rows with it (see _stream_subgraphs).
            self._record_subagent_link(payload)
            return
        if content_type == "subagent_end":
            # Status rides the dispatch tool's result (the card reads
            # props.status); the nested rows are state and need no terminal frame.
            return
        if content_type == "batman_started":
            # Background knowledge-base consolidation: a live-only indicator with
            # no end signal (it runs out-of-process). Rides a one-shot effect —
            # transient, thread-scoped, never replayed — so the client's flag
            # self-clears on a timer (see runEffect). NOT the agentState row: the
            # values channel re-upserts that whole row and would clobber the flag.
            await self._emit_effect({"kind": "batman", "threadId": self.thread_id})
            return
        if content_type in ("compaction_start", "compaction_end"):
            # Compaction has clean start/finish edges; each rides its own one-shot
            # effect toggling the client's `compacting` flag (same rationale as
            # batman for staying off the agentState row).
            await self._emit_effect(
                {
                    "kind": "compaction",
                    "threadId": self.thread_id,
                    "active": content_type == "compaction_start",
                }
            )
            return
        # Anything else is a one-shot effect the frontend acts on.
        await self._emit_effect(
            {"kind": str(content_type or "unknown"), "payload": payload}
        )

    async def _emit_effect(self, op: Mapping[str, Any]) -> None:
        effect_id = f"effect_{uuid.uuid4().hex}"
        await self.writer.upsert(TYPE_EFFECT, effect_id, {"id": effect_id, **dict(op)})

    async def _emit_report(
        self, op: Mapping[str, Any], tool_call_id: str | None
    ) -> None:
        """A generated report as a ``report`` row keyed by file id. The client
        reads the collection (chips); a cold reload re-materialises it from the
        transcript (``history_replay``), so it survives without replaying an
        effect."""
        file_id = op.get("file_id")
        if not isinstance(file_id, str) or not file_id:
            return
        value: dict[str, Any] = {
            "id": file_id,
            "threadId": self.thread_id,
            "fileId": file_id,
            "filename": str(op.get("filename") or ""),
            "createdAt": int(time.time() * 1000),
        }
        if tool_call_id:
            value["toolCallId"] = tool_call_id
        await self.writer.upsert(TYPE_REPORT, file_id, value)

    # ── subgraphs (subagents) ─────────────────────────────────────────────────
    def _record_subagent_link(self, payload: Mapping[str, Any]) -> None:
        subagent_id = payload.get("subagent_id")
        tool_call_id = payload.get("tool_call_id")
        if (
            isinstance(subagent_id, str)
            and isinstance(tool_call_id, str)
            and tool_call_id
        ):
            self._subagent_tool_call[subagent_id] = tool_call_id
            self._subagent_event(subagent_id).set()

    def _subagent_event(self, subagent_id: str) -> asyncio.Event:
        return self._subagent_ready.setdefault(subagent_id, asyncio.Event())

    async def _subagent_agent_id(self, subagent_id: str) -> str | None:
        """The dispatch tool_call_id to tag this subagent's rows with. Awaits the
        subagent_start link so a row is never emitted before it's known (an
        untagged row would leak into the parent transcript). ``None`` if the link
        never lands — the nested rows are then skipped, not mistagged."""
        try:
            await asyncio.wait_for(self._subagent_event(subagent_id).wait(), timeout=30)
        except TimeoutError:
            logger.warning("subagent %s: no subagent_start link", subagent_id)
            return None
        return self._subagent_tool_call.get(subagent_id)

    def _next_subagent_ordinal(self, subagent_id: str, msg_id: str) -> int:
        by_msg = self._subagent_msg_ordinal.setdefault(subagent_id, {})
        existing = by_msg.get(msg_id)
        if existing is not None:
            return existing
        ordinal = self._subagent_ordinal.get(subagent_id, 0)
        self._subagent_ordinal[subagent_id] = ordinal + 1
        by_msg[msg_id] = ordinal
        return ordinal

    async def _stream_subgraphs(self, stream: AsyncGraphRunStream) -> None:
        async with asyncio.TaskGroup() as tg:
            await self._drain_subgraphs(stream.extensions["subgraphs"], tg)

    async def _drain_subgraphs(self, channel: Any, tg: asyncio.TaskGroup) -> None:
        """Emit each subagent's child model text + tool calls as ``message`` /
        ``tool`` rows tagged with the dispatch ``agentId`` (recursing into nested
        subagents). Projections are read inside the loop body per the v3 run
        stream's lazy-subscribe constraint. Materialises state rows instead of
        custom subagent events."""
        async for sg in channel:
            if getattr(sg, "graph_name", None) != "subagent":
                continue
            subagent_id = sg.trigger_call_id or f"sa_{uuid.uuid4().hex}"
            extensions = getattr(sg, "extensions", {})
            messages = extensions.get("messages")
            tool_calls = extensions.get("tool_calls")
            custom = extensions.get("custom")
            nested = extensions.get("subgraphs")
            if messages is not None:
                tg.create_task(self._drain_subagent_messages(subagent_id, messages))
            if tool_calls is not None:
                tg.create_task(self._drain_subagent_tools(subagent_id, tool_calls, tg))
            if custom is not None:
                tg.create_task(self._drain_subagent_lifecycle(custom))
            if nested is not None:
                tg.create_task(self._drain_subgraphs(nested, tg))

    async def _drain_subagent_messages(self, subagent_id: str, channel: Any) -> None:
        agent_id = await self._subagent_agent_id(subagent_id)
        if agent_id is None:
            return
        async for message in channel:
            if getattr(message, "node", None) != _MODEL_NODE:
                continue
            msg_id = message.message_id or f"submsg_{uuid.uuid4().hex}"
            ordinal = self._next_subagent_ordinal(subagent_id, msg_id)
            created_at = _now_ms()
            # Emit the anchor up-front (even before any text) so a tool-only nested
            # turn has a bubble for its tools to slot after by ``order``. Text and
            # reasoning ride ``messageChunk`` rows tagged with ``agentId``.
            await self._upsert_message(msg_id, ordinal, created_at, agent_id=agent_id)
            logger.debug(
                "state: subagent message msg_id=%s agentId=%s ordinal=%s",
                msg_id,
                agent_id,
                ordinal,
            )
            await self._stream_message_text(
                message, msg_id, ordinal, created_at, agent_id=agent_id
            )

    async def _drain_subagent_tools(
        self, subagent_id: str, channel: Any, tg: asyncio.TaskGroup
    ) -> None:
        async for tc in channel:
            tg.create_task(self._consume_subagent_tool(subagent_id, tc))

    async def _consume_subagent_tool(self, subagent_id: str, tc: Any) -> None:
        agent_id = await self._subagent_agent_id(subagent_id)
        if agent_id is None:
            return
        tool_name = to_ag_ui_tool_name(tc.tool_name)
        args_text = to_ag_ui_json(tc.input) if tc.input else ""
        order = self._render_order(tc.tool_call_id)
        logger.debug(
            "state: subagent tool id=%s agentId=%s order=%s name=%s",
            tc.tool_call_id,
            agent_id,
            order,
            tool_name,
        )
        await self.writer.upsert(
            TYPE_TOOL,
            tc.tool_call_id,
            {
                "id": tc.tool_call_id,
                "threadId": self.thread_id,
                "agentId": agent_id,
                "name": tool_name,
                "argsText": args_text,
                "order": order,
            },
        )
        async for _ in tc.output_deltas:
            pass
        if is_interrupt_error(tc.error):
            return
        is_error = bool(tc.error)
        result = self._tool_result(tc, tool_name)
        value: dict[str, Any] = {
            "id": tc.tool_call_id,
            "threadId": self.thread_id,
            "agentId": agent_id,
            "name": tool_name,
            "argsText": args_text,
            "result": result,
            "order": order,
        }
        if is_error:
            value["isError"] = True
        await self.writer.upsert(
            TYPE_TOOL,
            tc.tool_call_id,
            value,
            txid=self._result_txid(tool_name, tc.tool_call_id),
        )

    async def _drain_subagent_lifecycle(self, channel: Any) -> None:
        """A nested subagent's start lands on its parent's custom projection;
        record the link so grandchild rows tag correctly. Everything else here
        (sandbox tool instrumentation, …) is ignored — nested tools come from the
        native ``tool_calls`` projection."""
        async for payload in channel:
            if (
                isinstance(payload, dict)
                and payload.get("content_type") == "subagent_start"
            ):
                self._record_subagent_link(payload)

    # ── values → agentState ───────────────────────────────────────────────────
    async def _stream_values(self, stream: AsyncGraphRunStream) -> None:
        prev: dict[str, Any] | None = None
        async for raw_state in stream.extensions["values"]:
            if not isinstance(raw_state, dict):
                continue
            current = project_client_state(raw_state)
            if current != prev:
                await self.writer.upsert(
                    TYPE_AGENT_STATE,
                    self.thread_id,
                    {"id": self.thread_id, **current},
                )
            prev = current

    # ── finalise ───────────────────────────────────────────────────────────────
    async def clear_interrupts(self, interrupt_ids: Iterable[str]) -> None:
        """Delete addressed interrupt rows on resume. The run row self-heals (one
        row per thread, overwritten to ``running``), but the per-interrupt rows
        are keyed by interrupt id and would otherwise linger in the client's
        shared collection — the frontend reads open interrupts straight off it."""
        for interrupt_id in interrupt_ids:
            await self.writer.delete(TYPE_INTERRUPT, interrupt_id)

    async def _finalise(
        self,
        stream: AsyncGraphRunStream,
        *,
        pre_finish: Awaitable[None] | None = None,
    ) -> None:
        if self._errored:
            return
        # Flush any out-of-band producer (the conversation-name DB write) before
        # the terminal run row, so the client's run-finish refetch reads it. A
        # failure there must not abort the run.
        if pre_finish is not None:
            try:
                await pre_finish
            except Exception:
                logger.exception("state translator pre_finish hook failed")
        interrupts = await self._collect_interrupts(stream)
        if interrupts:
            for item in interrupts:
                await self.writer.upsert(TYPE_INTERRUPT, item["id"], item)
            await self._set_run("interrupt", interrupt=interrupts[0])
        else:
            await self._set_run("complete")
        await self.writer.close()

    async def _collect_interrupts(
        self, stream: AsyncGraphRunStream
    ) -> list[dict[str, Any]]:
        if not await stream.interrupted():
            return []
        return ag_ui_interrupt_rows(await stream.interrupts(), self.thread_id)
