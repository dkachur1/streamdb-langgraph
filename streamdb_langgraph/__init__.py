"""streamdb-langgraph — translate LangGraph runs into State-Protocol row upserts.

The LangGraph-side counterpart to ``@assistant-ui/react-durable-streams`` and
rheoDS: drive :class:`StateTranslator` over a v3 ``AsyncGraphRunStream`` and it
materialises keyed, last-writer-wins rows (messages, tool calls, agent state,
interrupts, suggestions, subagent transcripts) that a StreamDB frontend renders
directly — no AG-UI lifecycle-event replay, no custom-event bus.
"""

from __future__ import annotations

from streamdb_langgraph.driver import iter_state_rows, translate_run
from streamdb_langgraph.history_replay import (
    SUMMARIZATION_MESSAGE_ID_PREFIX,
    next_live_order,
    next_live_ordinal,
    replay_history,
)
from streamdb_langgraph.state_projection import project_client_state
from streamdb_langgraph.state_protocol import (
    Operation,
    StateWriterProtocol,
    change_event,
)
from streamdb_langgraph.state_translator import StateTranslator
from streamdb_langgraph.stream import make_run_stream
from streamdb_langgraph.writers import CollectingStateWriter, QueueStateWriter

__all__ = [
    "SUMMARIZATION_MESSAGE_ID_PREFIX",
    "CollectingStateWriter",
    "Operation",
    "QueueStateWriter",
    "StateTranslator",
    "StateWriterProtocol",
    "change_event",
    "iter_state_rows",
    "make_run_stream",
    "next_live_order",
    "next_live_ordinal",
    "project_client_state",
    "replay_history",
    "translate_run",
]
