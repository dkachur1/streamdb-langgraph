"""Project ``AgentState`` onto the client-facing state mirrored over AG-UI.

This module is the single allowlist that decides what crosses the wire on
the AG-UI ``state`` channel (``STATE_SNAPSHOT`` + ``STATE_DELTA``). The
shape returned here *is* the contract the frontend couples to — not the
internal ``AgentState`` layout.

Two rules the allowlist exists to enforce:

* ``messages`` is NEVER mirrored here. It is streamed on the dedicated
  ``messages`` channel; echoing it as state would re-emit every token as
  state churn and double up the conversation.
* Internal bookkeeping (tool-call history, system-prompt snapshots,
  archival scratch, slug ledgers) stays server-side. Only slices a client
  renders belong here.

Adding a slice is a one-liner in :func:`project_client_state`; that is the
whole point of routing state through one projection instead of hand-built
patches scattered across tools and middleware.
"""

from __future__ import annotations

from typing import Any


def _tasks_slice(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Client view of the agent task list (``TodoPanel``).

    Emits a list (not the id-keyed dict) so creation order survives on the
    wire: a JS object hoists integer-like string keys ahead of insertion
    order, and some task ids are all-decimal ``uuid4().hex[:8]``, so an
    id-keyed object would let ``Object.values`` reorder the list.
    """
    tasks = state.get("tasks") or {}
    if not isinstance(tasks, dict):
        return []
    return list(tasks.values())


def _context_window_slice(state: dict[str, Any]) -> dict[str, Any]:
    """Live token usage written per model call by ``LlmUsageTrackingMiddleware``.

    ``{}`` until the first model call of a conversation; the client renders
    its zero state.
    """
    cw = state.get("context_window")
    return dict(cw) if isinstance(cw, dict) else {}


def _subagent_threads_slice(
    state: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Subagent transcripts keyed by spawn tool-call id (``SubagentMiddleware``),
    so the nested-thread card survives a STATE snapshot on reconnect. Card-format
    records; the client converts to AG-UI wire for the SubagentCard. Written once
    per subagent at finalize, so it rides ``STATE_DELTA`` as a single-key add, not
    per-token churn."""
    threads = state.get("subagent_threads")
    return dict(threads) if isinstance(threads, dict) else {}


def _active_project_slice(state: dict[str, Any]) -> str | None:
    """Active project slug (``ProjectsMiddleware``), independent of the active
    workflow. ``None`` when no project is selected (unset or cleared to ``""``)."""
    slug = state.get("active_project")
    return slug if isinstance(slug, str) and slug else None


def _active_workflow_slice(state: dict[str, Any]) -> str | None:
    """Active workflow dir_name (``WorkflowMiddleware``), independent of the
    active project. ``None`` when no workflow is selected (unset or cleared)."""
    slug = state.get("active_workflow")
    return slug if isinstance(slug, str) and slug else None


def project_client_state(state: dict[str, Any]) -> dict[str, Any]:
    """Reduce a raw ``AgentState`` dict to the client-facing state object.

    Returns a fresh dict each call so callers can diff successive
    projections without aliasing.
    """
    return {
        "active_project": _active_project_slice(state),
        "active_workflow": _active_workflow_slice(state),
        "context_window": _context_window_slice(state),
        "subagent_threads": _subagent_threads_slice(state),
        "tasks": _tasks_slice(state),
    }
