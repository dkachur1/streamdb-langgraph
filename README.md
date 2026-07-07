# streamdb-langgraph

Translate a running [LangGraph](https://langchain-ai.github.io/langgraph/) agent
into **State-Protocol row upserts** — the keyed, last-writer-wins frames a
StreamDB / durable-streams frontend materialises directly.

It is the LangGraph-side counterpart to the frontend runtime in
[`@assistant-ui/react-durable-streams`](https://github.com/dkachur1/react-durable-streams)
and [rheoDS](https://github.com/dkachur1/rheods), structurally analogous to how
[`ag-ui-langgraph`](https://github.com/ag-ui-protocol/ag-ui) pairs LangGraph with
the AG-UI client libraries — except the wire format here is the State Protocol
(rows), not AG-UI lifecycle events over SSE.

## Features

- **Rows, not event replay.** Instead of a `TEXT_MESSAGE_CONTENT` / `TOOL_CALL_*`
  lifecycle envelope plus RFC-6902 state patches, every change is the *full
  current row*, keyed, with an operation. The client keeps last-writer-wins, so
  resume is just an idempotent re-read — no `already_streamed` suppression, no
  receipt store, no snapshot re-baseline.
- **Streaming text as deltas.** Message text and reasoning ride append-only
  `messageChunk` rows carrying only the new characters, turning per-message wire
  cost from ~O(L²) to ~O(L).
- **Full agent surface.** Messages, tool calls (args then result), agent state,
  interrupts (human-in-the-loop), suggestions, tool summaries/actions, and nested
  **subagent** transcripts tagged by dispatch id.
- **History seeding.** `replay_history` maps a LangGraph checkpoint's message
  list into the same rows the live path produces, so one durable stream carries
  history + live from a single offset-0 read.
- **Pluggable sink.** The translator writes through a small
  `StateWriterProtocol` (`upsert` / `delete` / `close`). Bring your own
  durable-stream writer, or use the bundled in-memory / queue writers.

## Installation

```bash
pip install streamdb-langgraph
```

Requires Python 3.12+. Depends on `langgraph>=1.2`, `langchain-core>=1.4`,
`ag-ui-protocol`, `jsonpatch`, and `pydantic-core`.

## Usage

Given a compiled LangGraph graph, go from graph to rows in one call:

```python
from streamdb_langgraph import CollectingStateWriter, translate_run

writer = CollectingStateWriter()
await translate_run(
    graph,                                        # a compiled LangGraph graph
    {"messages": [{"role": "user", "content": "hi"}]},
    writer=writer,
    thread_id="conversation-1",
)

for frame in writer.frames:
    print(frame["type"], frame["key"], frame["headers"]["operation"])
    # message  msg_abc  upsert
    # run      conversation-1  upsert
    # ...
```

Or iterate frames as the run produces them — no writer to implement:

```python
from streamdb_langgraph import iter_state_rows

async for frame in iter_state_rows(
    graph,
    {"messages": [{"role": "user", "content": "hi"}]},
    thread_id="conversation-1",
):
    await my_durable_stream.append(frame)   # write to your durable stream
```

### Writing to a durable stream

In production you implement `StateWriterProtocol` against your durable stream so
the translator writes straight through:

```python
from collections.abc import Mapping
from typing import Any

from streamdb_langgraph import StateTranslator, change_event, make_run_stream


class DurableStreamWriter:
    """Append State-Protocol frames to one durable stream (illustrative)."""

    def __init__(self, stream_key: str) -> None:
        self.stream_key = stream_key

    async def upsert(
        self, type: str, key: str, value: Mapping[str, Any], *, txid: str | None = None
    ) -> None:
        frame = change_event(type=type, key=key, operation="upsert", value=value, txid=txid)
        await http_post(f"/streams/{self.stream_key}", json=frame)   # your transport

    async def delete(self, type: str, key: str) -> None:
        await http_post(f"/streams/{self.stream_key}", json=change_event(
            type=type, key=key, operation="delete"))

    async def close(self) -> None:
        ...   # the per-conversation stream stays open; nothing to do


writer = DurableStreamWriter(stream_key="conversation-1")
translator = StateTranslator(writer, thread_id="conversation-1", run_id="run-42")
stream = await make_run_stream(graph, graph_input, config=run_config)
await translator.run(stream)
```

`translate_run` / `iter_state_rows` are just this three-step sequence wrapped up.

### Seeding history

To render a whole prior transcript from one read, seed the checkpoint before the
live run and continue the shared ordinal counter:

```python
from streamdb_langgraph import next_live_order, next_live_ordinal, replay_history

prior_messages = checkpoint["channel_values"]["messages"]
await replay_history(writer, thread_id, prior_messages)

translator = StateTranslator(writer, thread_id=thread_id)
translator.set_ordinal_start(next_live_ordinal(prior_messages))
translator.set_order_start(next_live_order(prior_messages))
await translator.run(await make_run_stream(graph, graph_input, config=run_config))
```

## How it works

`make_run_stream` opens a LangGraph v3 event-streaming run
(`graph.astream_events(version="v3")`) with the transformers needed to expose the
typed `messages` / `tool_calls` / `custom` / `subgraphs` / `values` projections.
`StateTranslator.run` drains those channels concurrently and, for each, upserts a
keyed row:

| Channel      | Rows produced                                               |
| ------------ | ----------------------------------------------------------- |
| `messages`   | one `message` anchor + append-only `messageChunk` deltas    |
| `tool_calls` | one `tool` row (args, then result — successive upserts)     |
| `values`     | one `agentState` row (a client-facing allowlist projection) |
| `custom`     | `suggestion` / `toolSummary` / `toolAction` / `effect` rows |
| `subgraphs`  | subagent `message` / `tool` rows tagged with `agentId`      |
| lifecycle    | one `run` row (`running` → `complete` / `error`)            |
| interrupts   | `interrupt` rows; the run row's status carries the outcome  |

Every row is the whole current value, keyed by a stable id, so the frontend's
`createStreamDB` materialises it last-writer-wins and a resumed run re-upserting
the same ids merges idempotently.

## Configuration

- **Summary-message filtering.** Agents that run compaction mint summary messages
  with an id prefix that must not render in chat. The default prefix is
  `"summarization-"`; override it per call via
  `StateTranslator(..., summarization_message_id_prefix=...)`,
  `replay_history(..., summarization_message_id_prefix=...)`, and the
  `next_live_*` helpers if your agent uses a different one.
- **Client-state allowlist.** `project_client_state` is the single allowlist
  deciding which slices of agent state cross the wire on the `agentState` row.
- **Extra stream transformers.** `make_run_stream(..., extra_transformers=(...))`
  registers additional LangGraph projections alongside the defaults.

## Limitations

- Targets the LangGraph v3 event-streaming API (`astream_events(version="v3")`),
  which is marked experimental in LangGraph 1.2; callers see a
  `LangChainBetaWarning` per run.
- Non-agent graphs must register the tool-call projection themselves;
  `make_run_stream` fails fast if a required channel is missing. Graphs compiled
  through `langchain.agents.create_agent` get it automatically.
- The package produces frames; transporting them to a durable stream (and cold
  reseeding, offset management, retention) is the consumer's responsibility.
- **Parallel-interrupt resume is bounded by a LangGraph limitation, not this
  package.** LangGraph hashes an `interrupt()` id from the checkpoint namespace
  alone, so multiple `interrupt()` calls from parallel tool calls in one
  `ToolNode` can share a raw id
  ([langchain-ai/langgraph#6626](https://github.com/langchain-ai/langgraph/issues/6626)).
  `ag_ui_interrupt_rows` disambiguates the *rows* on a collision (keyed on
  `toolCallId`) so no pending interrupt is silently dropped from the stream —
  but routing a resume *answer* back to the correct one is a LangGraph-side fix,
  since resume-value routing keys off the same colliding id upstream of this
  translator.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
