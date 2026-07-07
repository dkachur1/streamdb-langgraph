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

Given a compiled LangGraph graph, go from graph to State-Protocol rows in one call.
`translate_run` opens a LangGraph **v3 event-streaming run** (`astream_events(version="v3")`),
drains each channel into keyed, last-writer-wins upserts, and writes them through a
`StateWriterProtocol` you provide — you never touch the event stream yourself:

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

The one piece of glue you write is a `StateWriterProtocol` for your transport; hand
it to `translate_run` and it drives the v3 stream, writing rows straight through:

```python
from collections.abc import Mapping
from typing import Any

from streamdb_langgraph import change_event, translate_run


class DurableStreamWriter:  # implements StateWriterProtocol
    def __init__(self, stream_key: str) -> None:
        self.stream_key = stream_key

    async def upsert(self, type, key, value: Mapping[str, Any], *, txid=None) -> None:
        await http_post(  # your transport
            f"/streams/{self.stream_key}",
            json=change_event(type=type, key=key, operation="upsert", value=value, txid=txid),
        )

    async def delete(self, type, key) -> None:
        await http_post(
            f"/streams/{self.stream_key}",
            json=change_event(type=type, key=key, operation="delete"),
        )

    async def close(self) -> None:
        ...  # the per-conversation stream stays open; nothing to do


await translate_run(
    graph,
    {"messages": [{"role": "user", "content": "hi"}]},
    writer=DurableStreamWriter(stream_key="conversation-1"),
    thread_id="conversation-1",
)
```

### A chat backend: one call per turn

For a chat where the **backend is the sole writer** — the browser never writes rows,
it POSTs the text and reads the stream back — use `translate_turn`. It appends the
user message *and* streams the agent's reply on one monotonic `order`/`ordinal`
sequence, seeded past the prior transcript so turns never collide or reorder:

```python
from streamdb_langgraph import translate_turn

await translate_turn(
    graph,
    "what's 17 times 23?",          # the new turn's text (e.g. from an onSend POST)
    writer=DurableStreamWriter(stream_key=conversation_id),
    thread_id=conversation_id,
    prior_messages=history,         # folded conversation so far (LangChain messages)
)
```

This is what the [react-durable-streams example](https://github.com/dkachur1/react-durable-streams/tree/main/examples/langgraph-chat) does — its agent is a FastAPI `/message` endpoint that calls `translate_turn` once per turn.

### Lower-level control

`translate_run` / `translate_turn` / `iter_state_rows` wrap a small public core: a
`StateTranslator` driven over `make_run_stream(graph, input, config=...)` — the raw
LangGraph v3 event stream. Reach for it directly only when you need finer control
(resume via `Command(resume=...)`, cancel-salvage); see `StateTranslator.run`.

### Seeding a prior transcript

`translate_turn` already continues the counters past `prior_messages`, so a normal
multi-turn chat needs nothing extra. To *materialize* a whole prior transcript that
isn't on the stream yet (e.g. from a LangGraph checkpoint) so the client renders it
from a single read, `replay_history` writes those rows first:

```python
from streamdb_langgraph import replay_history, translate_turn

prior_messages = checkpoint["channel_values"]["messages"]
await replay_history(writer, thread_id, prior_messages)   # backfill the transcript
await translate_turn(
    graph, new_text, writer=writer, thread_id=thread_id, prior_messages=prior_messages
)
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
