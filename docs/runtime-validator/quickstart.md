# Runtime Validator — Quickstart

## Purpose

Take you from zero to a working validation loop: install, build a trace, run the
validator, and act on the decision. For the bigger picture see the
[Overview](overview.md).

## Install

See the [Project README](../../README.md#installation). In short:

```bash
uv pip install -e .
# optional LangGraph integration
uv pip install -e ".[langgraph]"
```

## 1. Configure the validator

Start with triggers only — this is the recommended baseline and never calls an
LLM.

```python
from agent_runtime_validator import RuntimeValidator
from agent_runtime_validator.triggers import (
    MaxRoutesTrigger, SameToolLoopTrigger, NoProgressTrigger,
)

validator = RuntimeValidator(
    triggers=[
        MaxRoutesTrigger(max_routes=10),
        SameToolLoopTrigger(max_repeats=3),
        NoProgressTrigger(min_tool_calls=5),
    ],
)
```

Browse the full set in [Triggers](triggers.md). Add a validator later when you
need deeper checks ([Validators](validators.md)).

> Runnable version: [`examples/basic_runtime.py`](../../examples/basic_runtime.py).

## 2. Collect a trace

A trace is a snapshot of what the agent has done so far. Use `TraceBuilder` to
construct one incrementally — all `record_*` methods return `self` for fluent
chaining, and timestamps default to `now()` if omitted.

```python
from agent_runtime_validator import TraceBuilder

builder = TraceBuilder(run_id="run-123")

# As the agent acts, record events:
builder.record_routing("supervisor", "researcher", reason="delegate research")
builder.record_tool_call("search", call_id="c1", args={"query": "demo-topic"})
builder.record_tool_result("c1", "search", output="found result")
builder.record_message("assistant", "I found the result.")

trace = builder.build()
```

Fluent chaining is supported:

```python
trace = (
    TraceBuilder(run_id="run-123")
    .record_routing("supervisor", "researcher")
    .record_tool_call("search", call_id="c1", args={"query": "demo-topic"})
    .record_tool_result("c1", "search", output="found result")
    .build()
)
```

Build from an existing trace (adds more events on top):

```python
builder = TraceBuilder.from_trace(existing_trace)
builder.record_error("TimeoutError", "search timed out", agent_name="researcher")
updated_trace = builder.build()
```

Merge a child subgraph's trace into a parent:

```python
merged_trace = TraceBuilder.merge(parent_trace, child_trace).build()
```

`build()` is non-destructive — the builder is not consumed and can be called
multiple times to produce independent snapshots.

What the trace can hold:

| Field | Records |
|-------|---------|
| `messages` | Conversation turns |
| `tool_calls` / `tool_results` | Tool invocations and their outcomes |
| `agent_calls` | One agent invoking another |
| `routing_events` | Hand-offs between agents |
| `artifacts` | Concrete outputs produced |
| `errors` | Raised errors |
| `token_usage` | Total tokens, if you track them |

If you prefer to construct the Pydantic model directly, import `ExecutionTrace`
and append to its lists yourself — `TraceBuilder` is a convenience wrapper.

Using LangGraph? Build the trace automatically from graph state — see
[LangGraph](langgraph.md).

## 3. Get a decision

```python
decision = validator.validate(trace)

if not decision.should_continue:
    print(f"Stop: {decision.action} — {decision.reason}")
    print(f"Triggered by: {decision.triggered_by}")
```

## 4. Act on the decision

Map each action to behavior in your own runtime:

| `decision.action` | Typical handling |
|-------------------|------------------|
| `continue` | Proceed to the next step |
| `retry_last_step` | Re-run the previous node |
| `reroute` | Send to a different agent |
| `interrupt` | Pause and escalate to a human |
| `abort` | Stop the run immediately |

The library only *decides*; honoring the action is up to your application.

## 5. Add a deterministic validator (optional)

When triggered patterns are enough to decide on an action and you don't need an
LLM, use `TriggerScoreValidator`. It assigns a weight to each trigger, sums the
fired weights, and recommends an action when the score crosses a threshold.

```python
from agent_runtime_validator.validators import TriggerScoreValidator

validator = RuntimeValidator(
    triggers=[
        NoToolUsageTrigger(watched_agents={"research_agent"}),
        NoProgressTrigger(min_tool_calls=3),
        AgentPingPongTrigger(max_cycles=2),
    ],
    validator=TriggerScoreValidator(
        weights={
            "NoToolUsageTrigger": 2.0,
            "NoProgressTrigger": 2.0,
            "AgentPingPongTrigger": 3.0,
        },
        threshold=3.0,
        recommendation="reroute",
        max_attempts=1,
    ),
)
```

`max_attempts` prevents infinite loops — after the limit is reached, the
validator switches to `"interrupt"` regardless.

For an LLM-based judge, cap repeated validator calls in retry/reroute loops:

```python
from agent_runtime_validator.validators import LLMJudgeValidator

validator = RuntimeValidator(
    triggers=[...],
    validator=LLMJudgeValidator(model=call_model),
    max_validator_calls_per_run=1,
    on_validator_budget_exhausted="skip",  # default: triggers + policy decide
)
```

For full LLM judge and schema validation options, see [Validators](validators.md).

> Runnable version: [`examples/trigger_score.py`](../../examples/trigger_score.py).

## Saving and replaying traces

Save any trace to a JSON file — useful for debugging, auditing, and building offline test fixtures:

```python
from agent_runtime_validator import save_trace, load_trace, replay

# Production: save after each run
save_trace(trace, f"traces/{trace.run_id}.json")

# CI / development: load and replay against an updated validator config
loaded = load_trace("traces/run-abc123.json")
decision = replay(loaded, validator)
print(decision.action, decision.reason)
```

The file format is standard Pydantic JSON (all timestamps are ISO 8601 with timezone). For raw JSON strings without touching the filesystem:

```python
from agent_runtime_validator import trace_to_json, trace_from_json

json_str = trace_to_json(trace)         # -> str (pretty-printed)
trace2   = trace_from_json(json_str)    # -> ExecutionTrace
```

`replay` and `replay_async` are thin wrappers over `validator.validate()` that
signal the intent — an existing trace is being re-validated offline, not being
produced live.

## Async runs

If a validator calls an async model or does async I/O, use `validate_async`.
Triggers are always synchronous, so the trigger-only setup above works with
either method.

```python
decision = await validator.validate_async(trace)
```

Calling sync `validate()` with an async validator raises `RuntimeError`; the
message tells you to switch to `validate_async()`.

## Tuning the policy

Soften the default reactions by passing a configured policy:

```python
from agent_runtime_validator.policies import DefaultPolicy

validator = RuntimeValidator(
    triggers=[...],
    policy=DefaultPolicy(
        retry_on_medium=True,    # medium → retry_last_step (else continue)
        interrupt_on_high=True,  # high → interrupt (else continue)
        abort_on_critical=True,  # critical → abort (else interrupt)
    ),
)
```

## Logging

The library uses standard `logging` and does not configure it globally. Enable
it from your application:

```python
import logging

logging.basicConfig(level=logging.INFO)

# For full trigger-by-trigger detail:
logging.getLogger("agent_runtime_validator").setLevel(logging.DEBUG)
```

| Level | What is logged |
|-------|---------------|
| `DEBUG` | Each trigger result, validator details, continue decisions |
| `INFO` | Fired trigger names, validator invocation |
| `WARNING` | Non-continue decisions, malformed LLM judge responses |
| `ERROR` | LLM judge retries exhausted |

No prompts, tool arguments, or trace content are logged — only counts, names,
severities, and recommendations. The one exception: when the LLM judge returns
an unparseable response, the first 200 characters of that raw response are
logged at `WARNING` to aid debugging.

## Related

- [Overview](overview.md)
- [Architecture](architecture.md)
- [Triggers](triggers.md)
- [Validators](validators.md)
- [LangGraph integration](langgraph.md)
- [Project README](../../README.md)
