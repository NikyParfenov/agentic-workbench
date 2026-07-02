# Runtime Validator — LangGraph Integration

## Purpose

Run the validator inside a LangGraph graph: build a trace from graph state and
drop a validation node into your control flow. For the core concepts see the
[Overview](overview.md).

## Install

The integration needs the optional extra:

```bash
uv pip install -e ".[langgraph]"
```

Importing the integration without LangGraph installed raises a clear
`ImportError` telling you to install the extra.

## Collecting a trace from state

`state_to_trace` builds an `ExecutionTrace` from a LangGraph state dict. It reads
a set of `_trace_*` keys that your nodes populate as the run progresses.

```python
from agent_runtime_validator.integrations.langgraph.adapter import state_to_trace

trace = state_to_trace(state, run_id="run-123")
```

State keys it reads:

| State key | Maps to | Default |
|-----------|---------|---------|
| `run_id` | `ExecutionTrace.run_id` | `"langgraph-run"` |
| `started_at` | `started_at` | now (UTC) |
| `_trace_messages` | `messages` | `[]` |
| `_trace_agent_calls` | `agent_calls` | `[]` |
| `_trace_tool_calls` | `tool_calls` | `[]` |
| `_trace_tool_results` | `tool_results` | `[]` |
| `_trace_routing_events` | `routing_events` | `[]` |
| `_trace_artifacts` | `artifacts` | `[]` |
| `_trace_errors` | `errors` | `[]` |
| `_trace_metadata` | `metadata` | `{}` |
| `_trace_token_usage` | `token_usage` | `None` |

Populate these keys from your nodes (e.g. append a `ToolCall` to
`_trace_tool_calls` whenever a tool runs) so the validator sees an up-to-date
snapshot.

## Adding a validation node

`ValidationNode` wraps a `RuntimeValidator` as a callable graph node. It reads a
trace from state, validates it, and writes the decision back.

```python
from agent_runtime_validator.integrations.langgraph.nodes import ValidationNode
from agent_runtime_validator.triggers import MaxRoutesTrigger, SameToolLoopTrigger

node = ValidationNode(
    triggers=[
        MaxRoutesTrigger(max_routes=10),
        SameToolLoopTrigger(max_repeats=3),
    ],
)

builder.add_node("validation", node)
builder.add_edge("researcher", "validation")
builder.add_edge("validation", "supervisor")
```

> Complete runnable example: [`examples/langgraph_supervisor.py`](../../examples/langgraph_supervisor.py).

### Constructor options

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `triggers` | — | Triggers to run (required) |
| `validator` | `None` | Optional deep-check validator |
| `policy` | `None` | Optional custom policy |
| `trace_key` | `"trace"` | State key to read/write the trace |
| `decision_key` | `"decision"` | State key to write the decision to |
| `max_validator_calls_per_run` | `None` | Max validator invocations per run; `None` = unlimited |
| `on_validator_budget_exhausted` | `"skip"` | What to do when budget is exhausted — see [Validators](validators.md#validator-call-budget) |
| `validator_mode` | `"on_trigger"` | When to invoke the validator — `"on_trigger"`: only when triggers fire; `"always"`: on every call |
| `trace_builder` | `None` | Custom callable `(state) -> ExecutionTrace`; replaces default trace resolution |

### Validator mode

`validator_mode` controls when the validator is invoked:

- **`"on_trigger"` (default)** — the validator runs only when at least one trigger fires. Use this for inline mid-run monitoring; the common "all-clear" path never calls the validator.
- **`"always"`** — the validator always runs, regardless of trigger results. Use this when the validator is a post-run quality check that should inspect every completed trace.

```python
node = ValidationNode(
    triggers=[...],
    validator=judge,
    validator_mode="always",
)
```

### How the node resolves the trace

When called, the node resolves the trace and returns
`{trace_key: trace, decision_key: decision}` — a partial state update, which is
what LangGraph expects from a node. Returning only the updated keys matters:
echoing untouched keys would re-apply their reducers (an `operator.add`
messages channel would duplicate its entries on every validation step).
Writing the resolved trace back into state ensures `trace.metadata` (including
validator call budget) persists across subsequent node invocations, even when
the trace was originally passed as a serialized dict.

**With `trace_builder=None` (default)** the node applies this three-tier lookup:

1. `state[trace_key]` is missing → build via `state_to_trace(state)`
2. `state[trace_key]` is a `dict` → parse with `ExecutionTrace(**trace)`
3. `state[trace_key]` is an `ExecutionTrace` → use as-is

**With a custom `trace_builder`** the callable is called with `state` and its
return value is used directly. The `trace_key` lookup is skipped entirely,
giving you full control over how the trace is assembled.

### Custom trace builder

Supply `trace_builder` when the default `state_to_trace` mapping does not fit
your state schema — for example, if your graph stores messages under a
different key, derives tool calls from a custom structure, or needs to combine
multiple state slices.

```python
from agent_runtime_validator.integrations.langgraph import (
    ValidationNode,
    TraceBuilderFn,
)
from agent_runtime_validator import TraceBuilder, ExecutionTrace

def build_trace(state: dict) -> ExecutionTrace:
    builder = TraceBuilder(run_id=state.get("run_id", "run"))
    for msg in state.get("chat_history", []):
        builder.record_message(msg["role"], msg["content"])
    for call in state.get("tool_history", []):
        builder.record_tool_call(call["name"], call_id=call["id"], args=call.get("args", {}))
    return builder.build()

node = ValidationNode(
    triggers=[...],
    trace_builder=build_trace,
)
```

`TraceBuilderFn` is a type alias for `Callable[[dict[str, Any]], ExecutionTrace]`
that you can use to annotate your builder function.

## Routing on the decision

Use `create_validation_router` to generate a conditional routing function:

```python
from agent_runtime_validator.integrations.langgraph import (
    ValidationNode,
    create_validation_router,
)

builder.add_conditional_edges(
    "validation",
    create_validation_router(
        continue_to="supervisor",
        retry_to="researcher",
        reroute_to="fallback_agent",
        interrupt_to="human_review",
        # abort_to defaults to END
    ),
)
```

The router maps each `decision.action` to a node name. For `reroute`, it returns
`reroute_to` by default. If `allowed_reroutes` is set and
`decision.validator_result.suggested_next_agent` is in that allowlist, it routes
to the suggested node instead.

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `continue_to` | — | Node for `continue` (required) |
| `retry_to` | `continue_to` | Node for `retry_last_step` |
| `reroute_to` | `continue_to` | Fallback node for `reroute` |
| `interrupt_to` | `END` | Node for `interrupt` (map to a human-review node to keep the graph running) |
| `abort_to` | `END` | Node for `abort` |
| `allowed_reroutes` | `None` | Allowlist for `suggested_next_agent`; `None` = ignore suggested, use `reroute_to` |

Stop decisions are safe by default: an unmapped `interrupt` or `abort` ends the
run rather than silently routing back to `continue_to`.

By default `suggested_next_agent` is ignored — reroute always goes to
`reroute_to`. Pass an explicit set of node names to opt in to dynamic rerouting.
This prevents an LLM judge from routing to arbitrary or non-existent nodes.

The router handles both Pydantic `ValidationDecision` objects and plain dicts
(common after LangGraph state serialization via checkpointer).

For a fully manual router instead:

```python
def route(state):
    decision = state["decision"]
    if decision.action == "abort":
        return "end"
    if decision.action == "retry_last_step":
        return "researcher"
    return "supervisor"

builder.add_conditional_edges("validation", route)
```

## Async graphs

For async LangGraph graphs, register `node.async_call` instead of `node` as the
graph node. `async_call` has the same signature as `__call__` but awaits the
validator and uses `validate_async` internally.

```python
from agent_runtime_validator.integrations.langgraph import ValidationNode
from agent_runtime_validator.validators import LLMJudgeValidator

async def call_model_async(prompt: str) -> str:
    return await my_llm_client.generate(prompt)

node = ValidationNode(
    triggers=[...],
    validator=LLMJudgeValidator(model=call_model_async),
)

# Register the async entry point — NOT node itself
builder.add_node("validation", node.async_call)
builder.add_conditional_edges("validation", create_validation_router(...))
```

`node.__call__` is the sync entry point; `node.async_call` is the async one.
They share all state (triggers, validator, budget) — only the execution path
differs. Do **not** mix them on the same `ValidationNode` within the same graph.

If your validator is sync (e.g. `TriggerScoreValidator`, `JsonSchemaValidator`),
you can still register `node.async_call` — it will work correctly, calling the
sync validator inside an async context.

Calling sync `validate()` with an async model raises `RuntimeError` with a clear
message pointing to `validate_async`. The async registration sidesteps this
entirely.

> For a `validator_mode="always"` LangGraph flow, see [`examples/langgraph_always_validator.py`](../../examples/langgraph_always_validator.py).

## Trace persistence and archiving

`ValidationNode` writes the resolved `ExecutionTrace` back into state on every
call (returning the `{trace_key: trace, decision_key: decision}` update). This
is important for two reasons:

1. **Budget continuity** — `trace.metadata["_runtime_validator_call_count"]`
   persists across node invocations. Without writing the trace back, the budget
   counter resets on every call when the trace was passed as a serialized dict.

2. **Archiving** — the trace in state at the end of a run contains the full
   history of events. Saving it gives you an offline record for debugging and
   replay.

### Extracting the trace at end of run

Use `get_trace_from_state` to read the trace regardless of how LangGraph stored
it (in-memory `ExecutionTrace`, serialized `dict`, or absent):

```python
from agent_runtime_validator.integrations.langgraph import get_trace_from_state
from agent_runtime_validator import save_trace

def archive_trace(state: dict) -> dict:
    trace = get_trace_from_state(state)   # trace_key defaults to "trace"
    if trace is not None:
        save_trace(trace, f"traces/{trace.run_id}.json")
    return state

builder.add_node("archive", archive_trace)
builder.add_edge("validation", "archive")
builder.add_edge("archive", END)
```

`get_trace_from_state(state, trace_key="trace")` handles all three storage forms:
- `ExecutionTrace` object (before checkpointing)
- `dict` (after LangGraph deserializes a checkpoint)
- absent key → returns `None`

### Offline replay

Once archived, replay the trace against an updated validator config without
re-running the graph:

```python
from agent_runtime_validator import load_trace, replay, RuntimeValidator
from agent_runtime_validator.triggers import SameToolLoopTrigger

tuned_validator = RuntimeValidator(
    triggers=[SameToolLoopTrigger(max_repeats=2)],  # tightened threshold
)
trace = load_trace("traces/run-abc123.json")
decision = replay(trace, tuned_validator)
print(decision.action, decision.triggered_by)
```

See [Quickstart — Saving and replaying traces](quickstart.md#saving-and-replaying-traces)
for the full offline replay API.

### Budget state across checkpoints

LangGraph checkpointers (e.g. `MemorySaver`, `SqliteSaver`) serialize state to
JSON between steps. After deserialization the `ExecutionTrace` arrives as a
plain `dict`. `ValidationNode` detects this and re-parses it before validating,
so `trace.metadata["_runtime_validator_call_count"]` is never lost.

If you build a custom `trace_builder`, make sure your builder preserves
`state[trace_key]["metadata"]` (if the key already holds a serialized trace)
so the budget counter survives checkpoints.

## LangChain messages adapter

Many LangGraph projects keep their execution history in `state["messages"]` as a
list of LangChain `BaseMessage` objects. `from_langchain_messages` turns that
list directly into an `ExecutionTrace` — no LangChain import required.

```python
from agent_runtime_validator.integrations.langgraph import from_langchain_messages
```

### Mapping rules

| LangChain type | Produces |
|---|---|
| `HumanMessage` (`type="human"`) | `MessageEvent(role="user")` |
| `AIMessage` (`type="ai"`) | `MessageEvent(role="assistant")` + `ToolCall` per `tool_calls` entry |
| `SystemMessage` (`type="system"`) | `MessageEvent(role="system")` |
| `ToolMessage` (`type="tool"`) | `MessageEvent(role="tool")` + `ToolResult` |
| `ToolMessage` with `status="error"` | As above, but content maps to `ToolResult.error` (feeds `ToolErrorRateTrigger`) |
| Any other type | `MessageEvent(role="assistant")` |

### Usage as a `trace_builder` callback

Pass it as `trace_builder` to `ValidationNode` so the node converts
`state["messages"]` on every call:

```python
from agent_runtime_validator.integrations.langgraph import (
    ValidationNode,
    from_langchain_messages,
)
from agent_runtime_validator.schema.trace import ExecutionTrace

def build_trace(state: dict) -> ExecutionTrace:
    return from_langchain_messages(
        state["messages"],
        run_id=state.get("run_id", "run"),
        agent_name="analyst",
        include_subgraph_thoughts=True,   # default: lift _subgraph_thoughts from message metadata
    )

node = ValidationNode(triggers=[...], trace_builder=build_trace)
```

When `include_subgraph_thoughts=True` (the default), the adapter inspects each
message's `additional_kwargs`, `response_metadata`, and `metadata` dicts for a
key named `_subgraph_thoughts`. If the value is a list of strings it is parsed
via `from_subgraph_thoughts` and the resulting tool calls and results are merged
into the main trace.

- Default key is `_subgraph_thoughts`; pass `subgraph_thoughts_key=` to use a
  different key.
- The parser is best-effort (via `from_subgraph_thoughts`) — unrecognised lines
  produce a message only, never an error.
- Routing events and semantic agent calls are never inferred from thought text;
  those must be recorded explicitly.

### Limitations

`from_langchain_messages` only maps messages, tool calls, and tool results.
Routing events and agent calls must be recorded explicitly (e.g. via
`TraceBuilder`) because no reliable convention exists for expressing them
inside a plain message list.

## Lifting structured subgraph messages

If an inner graph keeps its full tool conversation local, do not copy those
`ToolMessage` objects into the outer graph's `messages`. Instead, convert the
inner messages into an `ExecutionTrace` delta and merge that delta into the
outer state's `trace` with `lift_subgraph_messages`:

```python
from agent_runtime_validator.integrations.langgraph import (
    get_trace_from_state,
    lift_subgraph_messages,
)

async def run_worker_subgraph(state: dict) -> dict:
    result_state = await worker_graph.ainvoke({"messages": []})
    subgraph_messages = result_state["messages"]

    trace = lift_subgraph_messages(
        parent_trace=get_trace_from_state(state),
        subgraph_messages=subgraph_messages,
        run_id=state.get("run_id", "run"),
        agent_name="worker",
        include_subgraph_thoughts=False,  # default
    )

    final_message = build_worker_summary(subgraph_messages)

    return {
        "messages": [final_message],
        "trace": trace,
    }
```

This is the preferred production path when structured `AIMessage.tool_calls`
and `ToolMessage` objects are available. `from_subgraph_thoughts` remains a
best-effort fallback for textual debug logs, not the primary machine-readable
trace source — which is why `include_subgraph_thoughts` defaults to `False`
here.

What you get:

- **Outer messages stay compact.** Only the worker's summary enters the outer
  chat history — no context bloat, no tool-call pairing breakage after
  message trimming.
- **The validator still sees everything.** The lifted `ToolCall` /
  `ToolResult` events land in the trace, so loop and error-rate triggers keep
  working.
- **Parent data is preserved.** Existing events and metadata on the parent
  trace are never overwritten; the helper records its own marker under
  `trace.metadata["_last_lifted_source"]`.

Behavior details:

- `parent_trace` accepts an `ExecutionTrace`, a serialized `dict` (after
  checkpointing), or `None` (starts a fresh trace).
- `run_id` priority: explicit argument → parent trace's `run_id` →
  `"subgraph-run"`.
- Inputs are never mutated; a new trace is returned.
- Routing events and `AgentCall` events are never inferred from messages —
  record them explicitly (`TraceBuilder.record_routing` /
  `record_agent_call`) at the point where routing or delegation happens.

## Subgraph thoughts adapter

Some LangGraph subgraphs surface their internal reasoning as a list of plain
strings (thought logs, print-style traces). `from_subgraph_thoughts` turns that
list directly into an `ExecutionTrace` — no LangChain import required.

```python
from agent_runtime_validator.integrations.langgraph import from_subgraph_thoughts

trace = from_subgraph_thoughts(
    state.get("subgraph_thoughts", []),
    run_id=state.get("run_id", "subgraph-run"),
    agent_name="analyst",
)
```

### What it does

- Every line is preserved as a `MessageEvent(role="assistant")`.
- Lines that match known tool-call patterns additionally produce a `ToolCall`.
- Lines that match known tool-result patterns additionally produce a `ToolResult`.
- `ToolResult` entries are matched to `ToolCall` entries by bracket id (`[c1]`)
  when present, or by the latest unmatched call when exactly one is pending.

### Supported line patterns

```
Tool call [c1] analyze_item with arguments: {"item_id": "demo-item"}
Tool call analyze_item with arguments: {"item_id": "demo-item"}
Calling tool analyze_item with args {"item_id": "demo-item"}
Tool result [c1]: {"status": "ok"}
Tool response [c1]: done
Tool output [c1]: done
```

### Limitations

- Best-effort parser — unrecognised lines produce a message only, never an error.
- Original lines are always preserved as messages; nothing is discarded.
- Routing events and agent calls are never inferred from thought text. Use
  `TraceBuilder.record_routing` / `record_agent_call` to record those events
  explicitly.

## High-level state helper

`build_trace_from_state` is a convenience function that assembles an
`ExecutionTrace` in a single call by merging an existing trace (if any),
LangChain messages, and artifacts from state.

Use it when your graph already stores messages under `state["messages"]`
and optionally has an existing trace under `state["trace"]`. It handles all
three storage forms for the existing trace (``ExecutionTrace`` object, serialized
``dict``, or absent key) and delegates message conversion to
`from_langchain_messages`.

```python
from agent_runtime_validator.integrations.langgraph import build_trace_from_state

validation_node = ValidationNode(
    triggers=[...],
    validator=judge,
    validator_mode="always",
    trace_builder=lambda state: build_trace_from_state(
        state,
        run_id=state.get("run_id", "run"),
        messages_key="messages",
        artifacts_key="artifacts",
        agent_name="analyst",
    ),
)
```

Key notes:

- Repeated calls are safe. The helper records how many messages and artifacts
  it has already merged in `trace.metadata["_merged_message_count"]` /
  `trace.metadata["_merged_artifact_count"]`, so when `ValidationNode` writes
  the trace back into state and runs again on the next step, only entries added
  since the previous call are merged — events are never duplicated.

- Use this function when state has LangChain messages and/or an existing trace.
  For graphs that build events manually via `TraceBuilder`, use
  `TraceBuilder.record_routing` / `record_agent_call` for routing and semantic
  agent-call events — `build_trace_from_state` never infers those from messages.

- Message trimming (e.g. LangGraph's `messages` reducer) can remove older
  context from `state["messages"]`. If your graph runs for many steps, storing a
  durable `ExecutionTrace` in state (the `trace_key` slot) is safer than relying
  solely on messages — the helper merges both, so older events in the trace are
  preserved even after the messages list is trimmed.

## Related

- [Overview](overview.md)
- [Architecture](architecture.md)
- [Quickstart](quickstart.md)
- [Validators](validators.md)
- [Project README](../../README.md)
