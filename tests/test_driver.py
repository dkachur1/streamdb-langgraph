"""The one-call convenience layer: ``translate_run`` / ``iter_state_rows``.

Both are driven over a scripted stream (``make_run_stream`` monkeypatched to a
``fake_run_stream``) so the wiring is tested without a live LangGraph graph.
"""

from __future__ import annotations

import pytest

from streamdb_langgraph import (
    CollectingStateWriter,
    iter_state_rows,
    translate_run,
)
from streamdb_langgraph import driver as driver_module
from tests.fixtures import FakeMessage, fake_run_stream


@pytest.fixture
def scripted_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch ``make_run_stream`` (as imported into driver) to a one-message run."""

    async def _fake_make_run_stream(graph, graph_input, **kwargs):  # noqa: ANN001, ANN003
        return fake_run_stream(
            messages=[FakeMessage("msg-1", [("text", 0, "hello")])]
        )

    monkeypatch.setattr(driver_module, "make_run_stream", _fake_make_run_stream)


def _types(frames: list[dict[str, object]]) -> list[str]:
    return [str(f["type"]) for f in frames]


async def test_translate_run_writes_run_and_message_rows(
    scripted_stream: None,
) -> None:
    writer = CollectingStateWriter()

    translator = await translate_run(
        object(),  # graph unused; make_run_stream is patched
        {"messages": []},
        writer=writer,
        thread_id="conv-1",
        run_id="run-1",
    )

    assert writer.closed is True
    assert translator.thread_id == "conv-1"
    types = _types(writer.frames)
    # lifecycle brackets the transcript: running run row first, complete last.
    assert types[0] == "run"
    assert types[-1] == "run"
    assert "message" in types
    assert "messageChunk" in types
    run_rows = [f for f in writer.frames if f["type"] == "run"]
    assert run_rows[0]["value"]["status"] == "running"
    assert run_rows[-1]["value"]["status"] == "complete"


async def test_iter_state_rows_yields_frames_in_order(
    scripted_stream: None,
) -> None:
    frames = [
        frame
        async for frame in iter_state_rows(
            object(),
            {"messages": []},
            thread_id="conv-1",
        )
    ]

    types = _types(frames)
    assert types[0] == "run"
    assert types[-1] == "run"
    assert "message" in types
    assert frames[-1]["value"]["status"] == "complete"
