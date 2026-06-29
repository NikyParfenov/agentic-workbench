# Runtime Validator — Triggers

## Purpose

Reference for the built-in triggers: what each one detects, its parameters, and
its default severity. Triggers are the deterministic first stage of validation —
fast, side-effect free, and never call an LLM.

## How triggers behave

You pass a list of triggers to `RuntimeValidator`. On every `validate` call each
trigger inspects the trace and returns a `TriggerResult` (fired or not, plus a
severity and supporting evidence). All triggers run on every call — the pipeline
does not stop at the first one that fires, so the policy sees the full picture.

Import them from `agent_runtime_validator.triggers`.

## Built-in triggers

| Trigger | Fires when | Default severity | Key parameters |
|---------|------------|------------------|----------------|
| `MaxToolCallsTrigger` | Total tool calls ≥ limit | `high` | `max_calls` |
| `MaxContextTokensTrigger` | Token usage ≥ limit | `high` | `max_tokens` |
| `MaxExecutionTimeTrigger` | Elapsed seconds ≥ limit | `high` | `max_seconds` |
| `MaxRoutesTrigger` | Routing events ≥ limit | `high` | `max_routes` |
| `NoProgressTrigger` | ≥ N tool calls and **zero** artifacts | `medium` | `min_tool_calls` (5) |
| `AgentPingPongTrigger` | An agent pair alternates ≥ N cycles | `high` | `max_cycles` |
| `SameToolLoopTrigger` | One tool called ≥ N times | `medium` | `max_repeats` |
| `SameToolSameArgsLoopTrigger` | Same tool + identical args ≥ N times | `high` | `max_repeats` |
| `ToolErrorRateTrigger` | Error rate ≥ threshold | `high` | `max_error_rate`, `min_results` (1) |
| `NoToolUsageTrigger` | A watched agent made fewer calls than expected | `medium` | `watched_agents`, `min_expected_calls` (1) |

Every trigger also accepts `severity=` to override its default.

## Details and tuning

### Budget triggers

`MaxToolCallsTrigger`, `MaxRoutesTrigger`, `MaxExecutionTimeTrigger`, and
`MaxContextTokensTrigger` are simple ceilings. They fire when the count or value
**reaches or exceeds** the limit.

`MaxContextTokensTrigger` uses `trace.token_usage` when you set it; otherwise it
estimates from message, argument, and result text. `MaxExecutionTimeTrigger`
measures `finished_at - started_at`, falling back to "now" while the run is
still in progress.

### Loop triggers

`SameToolLoopTrigger` counts calls per tool name. `SameToolSameArgsLoopTrigger`
counts calls per (tool name + argument) pair, catching an agent that re-issues
the exact same call — a common hallucination pattern. Both fire on the most
frequent entry reaching `max_repeats`.

### Routing triggers

`MaxRoutesTrigger` caps total hand-offs. `AgentPingPongTrigger` looks for a pair
of agents alternating `A → B → A → B`; `max_cycles` is how many round trips are
allowed before it fires. It needs at least two routing events.

### Progress and quality triggers

`NoProgressTrigger` fires when the agent has made at least `min_tool_calls` calls
but produced **no** artifacts — useful for catching busy-but-unproductive runs.

`ToolErrorRateTrigger` fires when the share of tool results carrying an error
reaches `max_error_rate` (a float, e.g. `0.5` for 50%). `min_results` guards
against firing on a tiny sample.

### Agent-coverage trigger

`NoToolUsageTrigger` takes a set of agent names you expect to use tools and fires
if any of them made fewer than `min_expected_calls` tool calls. Use it when a
specialist agent is supposed to act but might be skipped. Orchestrator/router
agents that legitimately never call tools should **not** be in `watched_agents`.

```python
from agent_runtime_validator.triggers import NoToolUsageTrigger

NoToolUsageTrigger(watched_agents={"researcher", "coder"}, min_expected_calls=1)
```

## Choosing severities

Severity drives the default action (see [Architecture](architecture.md#how-actions-are-chosen)).
Raise a trigger's severity when firing should escalate; lower it when it is only
advisory.

```python
from agent_runtime_validator.triggers import SameToolLoopTrigger

SameToolLoopTrigger(max_repeats=3, severity="high")  # interrupt instead of retry
```

## Writing a custom trigger

Subclass `BaseTrigger` and return a `TriggerResult`.

```python
from agent_runtime_validator import BaseTrigger, TriggerResult, ExecutionTrace

class NoUserMessageTrigger(BaseTrigger):
    def evaluate(self, trace: ExecutionTrace) -> TriggerResult:
        has_user = any(m.role == "user" for m in trace.messages)
        return TriggerResult(
            triggered=not has_user,
            trigger_name="NoUserMessageTrigger",
            severity="low",
            reason="No user message in trace" if not has_user else "User message present",
            evidence={"message_count": len(trace.messages)},
        )
```

> Runnable version: [`examples/custom_trigger.py`](../../examples/custom_trigger.py).

Keep custom triggers deterministic — put LLM logic in a validator instead
([Validators](validators.md)).

## Related

- [Overview](overview.md)
- [Architecture](architecture.md)
- [Quickstart](quickstart.md)
- [Validators](validators.md)
- [Design decisions](design-decisions.md)
