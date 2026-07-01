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
| `trace_builder` | `None` | Custom callable `(state) -> ExecutionTrace`; replaces default trace resolution |

### How the node resolves the trace

When called, the node resolves the trace and then returns
`{**state, trace_key: trace, decision_key: decision}`, leaving all other
state fields untouched. Writing the resolved trace back into state ensures
`trace.metadata` (including validator call budget) persists across subsequent
node invocations, even when the trace was originally passed as a serialized dict.

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
| `interrupt_to` | `continue_to` | Node for `interrupt` |
| `abort_to` | `END` | Node for `abort` |
| `allowed_reroutes` | `None` | Allowlist for `suggested_next_agent`; `None` = ignore suggested, use `reroute_to` |

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

For async graphs, call `node.async_call(state)`; it runs the same logic through
`validate_async`, awaiting async validators. Use it when your validator calls an
async model — see [Validators](validators.md#sync-vs-async).

## Related

- [Overview](overview.md)
- [Architecture](architecture.md)
- [Quickstart](quickstart.md)
- [Validators](validators.md)
- [Project README](../../README.md)
