"""Helpers for constructing v3 event-streaming runs we'll translate to AG-UI.

LangGraph 1.2 introduced ``graph.astream_events(version="v3")`` which
returns an :class:`AsyncGraphRunStream` exposing typed projections
(messages, values, subgraphs, custom, tool_calls, lifecycle,
interrupts, ...) over one underlying event flow. See
``https://docs.langchain.com/oss/python/langgraph/event-streaming``.

This module is the single place that bakes in the transformer list we
require — every caller building a v3 stream goes through
:func:`make_run_stream` so the orchestrator and tests can't drift on
which transformers are registered.

Transformers we always want:

* :class:`langgraph.stream.transformers.CustomTransformer` — exposes
  ``get_stream_writer()`` events on ``stream.custom``. Without this,
  our ~22 in-graph emit sites are invisible.
* Tool-call projection — exposes ``stream.tool_calls`` with the
  correlated ``tool-started`` / ``tool-output-delta`` /
  ``tool-finished`` / ``tool-error`` lifecycle. LangGraph wires this
  automatically for graphs compiled through ``langchain.agents.create_agent``;
  non-agent graphs must compile/register the tool-call transformer themselves.

The v3 protocol is marked experimental in 1.2.1; callers will see
``LangChainBetaWarning`` on each call. Acceptable for the parallel
AG-UI migration path; we re-evaluate when v3 sheds the beta flag.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.stream import AsyncGraphRunStream
from langgraph.stream.transformers import CustomTransformer, StreamTransformer

# v3 ``astream_events`` registers base projections (messages, values,
# lifecycle, subgraphs). ``CustomTransformer`` is NOT registered by default
# and is required to surface ~22 in-graph ``get_stream_writer`` emissions on
# ``stream.custom``.
DEFAULT_TRANSFORMERS: tuple[type[StreamTransformer], ...] = (CustomTransformer,)
AG_UI_TRANSLATOR_REQUIRED_CHANNELS = frozenset(
    ("messages", "custom", "subgraphs", "tool_calls", "values")
)


def _assert_ag_ui_translator_channels(stream: AsyncGraphRunStream) -> None:
    """Fail at the stream-construction seam if translator channels are absent."""
    missing = AG_UI_TRANSLATOR_REQUIRED_CHANNELS.difference(stream.extensions)
    if not missing:
        return

    missing_list = ", ".join(sorted(missing))
    raise RuntimeError(
        "AG-UI translator requires v3 stream channel(s) that are missing: "
        f"{missing_list}. Compile the graph through langchain.agents.create_agent "
        "or register the matching LangGraph stream transformer before calling "
        "make_run_stream()."
    )


async def make_run_stream(
    graph: CompiledStateGraph,
    graph_input: Any,
    *,
    config: RunnableConfig | None = None,
    extra_transformers: tuple[type[StreamTransformer], ...] = (),
) -> AsyncGraphRunStream:
    """Open a v3 event-streaming run with the standard translator transformers.

    Parameters
    ----------
    graph
        Compiled LangGraph state graph to drive.
    graph_input
        Run input — either a state dict for a fresh run or a
        ``langgraph.types.Command`` (e.g. ``Command(resume=...)``) to
        continue from an interrupt.
    config
        Optional ``RunnableConfig``. Must carry ``thread_id`` under
        ``configurable`` for interrupt resume to work.
    extra_transformers
        Additional :class:`StreamTransformer` subclasses to register
        alongside :data:`DEFAULT_TRANSFORMERS`. Use for one-off custom
        projections (telemetry, stats, etc.) — graph-wide projections
        should be added to :data:`DEFAULT_TRANSFORMERS` itself.

    Returns
    -------
    AsyncGraphRunStream
        The run handle. Read typed projections concurrently
        (``stream.messages``, ``stream.custom``, ``stream.subgraphs``,
        ``stream.tool_calls``, etc.) or iterate ``stream`` for the raw
        ordered protocol event stream.
    """
    # Dedup so a caller re-passing a default transformer via ``extra_transformers``
    # can't register the same projection twice — LangGraph raises a "projection
    # keys that conflict" error on duplicate transformers.
    seen: set[type[StreamTransformer]] = set()
    transformers: list[type[StreamTransformer]] = []
    for transformer in (*DEFAULT_TRANSFORMERS, *extra_transformers):
        if transformer not in seen:
            seen.add(transformer)
            transformers.append(transformer)
    stream = await graph.astream_events(
        graph_input,
        config=config,
        version="v3",
        transformers=transformers,
    )
    _assert_ag_ui_translator_channels(stream)
    return stream
