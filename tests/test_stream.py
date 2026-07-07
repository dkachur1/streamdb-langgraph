from __future__ import annotations

import pytest

from streamdb_langgraph.stream import CustomTransformer, make_run_stream


class _FakeStream:
    def __init__(self, extensions: set[str]) -> None:
        self.extensions = {name: object() for name in extensions}


class _FakeGraph:
    def __init__(self, extensions: set[str]) -> None:
        self.extensions = extensions
        self.calls: list[dict[str, object]] = []

    async def astream_events(self, graph_input, **kwargs):
        self.calls.append({"graph_input": graph_input, **kwargs})
        return _FakeStream(self.extensions)


@pytest.mark.asyncio
async def test_make_run_stream_registers_custom_transformer_and_returns_stream():
    graph = _FakeGraph({"messages", "custom", "subgraphs", "tool_calls", "values"})

    stream = await make_run_stream(graph, {"messages": []})  # type: ignore[arg-type]

    assert stream.extensions.keys() == {
        "messages",
        "custom",
        "subgraphs",
        "tool_calls",
        "values",
    }
    assert graph.calls == [
        {
            "graph_input": {"messages": []},
            "config": None,
            "version": "v3",
            "transformers": [CustomTransformer],
        }
    ]


@pytest.mark.asyncio
async def test_make_run_stream_fails_fast_without_required_projection():
    graph = _FakeGraph({"messages", "custom", "subgraphs"})

    with pytest.raises(RuntimeError, match="tool_calls"):
        await make_run_stream(graph, {"messages": []})  # type: ignore[arg-type]
