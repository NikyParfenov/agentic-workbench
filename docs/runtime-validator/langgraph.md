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
| `trace_key` | `"trace"` | State key to read the trace from |
| `decision_key` | `"decision"` | State key to write the decision to |

### How the node resolves the trace

When called, the node reads `state[trace_key]` and:

- if it is missing, builds one via `state_to_trace(state)`;
- if it is a dict, parses it with `ExecutionTrace(**trace)`;
- if it is already an `ExecutionTrace`, uses it as-is.

It then returns `{**state, decision_key: decision}`, leaving the rest of the
state untouched.

## Routing on the decision

Use a conditional edge to branch on the decision the node wrote:

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
