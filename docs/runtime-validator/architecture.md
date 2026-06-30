# Runtime Validator — Architecture

## Purpose

Document the data model, control flow, and extension points of the runtime
validator for contributors and advanced users. For a conceptual introduction see
the [Overview](overview.md).

## Pipeline

```
ExecutionTrace
      |
      v
[ Triggers ]      evaluate(trace) -> TriggerResult        (always run, deterministic)
      |
      v
  any fired? --no--> [ Policy ]
      |
     yes (and validator is not NoOp, and call budget allows)
      |
      v
[ Validator ]     validate(trace, results) -> ValidatorResult   (optional, may call LLM)
      |
      | skipped when budget exhausted with on_validator_budget_exhausted="skip"
      |
      v
[ Policy ]        decide(...) -> ValidationDecision
      |
      v
ValidationDecision
```

## ExecutionTrace

A framework-neutral Pydantic snapshot of a run. Only `run_id` and `started_at`
are required.

| Field | Type | Notes |
|-------|------|-------|
| `run_id` | `str` | Identifier for the run |
| `started_at` | `AwareDatetime` | Timezone-aware start time |
| `finished_at` | `AwareDatetime \| None` | Set when the run ends |
| `messages` | `list[MessageEvent]` | Conversation turns |
| `agent_calls` | `list[AgentCall]` | One agent invoking another |
| `tool_calls` | `list[ToolCall]` | Tool invocations |
| `tool_results` | `list[ToolResult]` | Tool outcomes |
| `routing_events` | `list[RoutingEvent]` | Hand-offs between agents |
| `artifacts` | `list[ArtifactEvent]` | Concrete outputs |
| `errors` | `list[ErrorEvent]` | Raised errors |
| `metadata` | `dict` | Free-form |
| `token_usage` | `int \| None` | Total tokens, if tracked |

### Event types

| Event | Key fields |
|-------|-----------|
| `MessageEvent` | `role`, `content`, `agent_name`, `timestamp` |
| `AgentCall` | `caller`, `callee`, `input`, `output`, `timestamp` |
| `ToolCall` | `tool_name`, `args`, `agent_name`, `call_id`, `timestamp` |
| `ToolResult` | `call_id`, `tool_name`, `output`, `error`, `timestamp` |
| `RoutingEvent` | `from_agent`, `to_agent`, `reason`, `timestamp` |
| `ArtifactEvent` | `artifact_id`, `artifact_type`, `content`, `timestamp` |
| `ErrorEvent` | `error_type`, `message`, `agent_name`, `timestamp` |

## Result types

### TriggerResult

| Field | Type | Notes |
|-------|------|-------|
| `triggered` | `bool` | Whether the condition was met |
| `trigger_name` | `str` | Identifies the trigger |
| `severity` | `Severity` | `low` / `medium` / `high` / `critical` |
| `reason` | `str` | Explanation |
| `evidence` | `dict` | Supporting numbers |

### ValidatorResult

| Field | Type | Notes |
|-------|------|-------|
| `valid` | `bool` | Passed validation |
| `confidence` | `float` 0–1 | Validator confidence |
| `issues` | `list[str]` | Problems found |
| `findings` | `list[JudgeFinding]` | Structured findings (category, severity, evidence) |
| `recommendation` | `Action` | Overrides policy default when present |
| `reason` | `str` | Explanation |
| `suggested_next_agent` | `str \| None` | Hint for `reroute` |
| `suggested_message` | `str \| None` | Message to inject next |

### ValidationDecision

| Field | Type | Notes |
|-------|------|-------|
| `should_continue` | `bool` | `True` only when `action == "continue"` |
| `action` | `Action` | Final action |
| `severity` | `Severity` | Highest fired severity |
| `reason` | `str` | Explanation |
| `triggered_by` | `list[str]` | Names of fired triggers |
| `validator_result` | `ValidatorResult \| None` | Present if a validator ran |

## RuntimeValidator internals

`RuntimeValidator(triggers, validator=None, policy=None, ...)` defaults to
`NoOpValidator`, `DefaultPolicy`, unlimited validator invocations
(`max_validator_calls_per_run=None`), and skip-on-budget-exhaustion
(`on_validator_budget_exhausted="skip"`). Both `validate` and `validate_async`:

1. Run every trigger to build `trigger_results`.
2. Compute `fired = [r for r in trigger_results if r.triggered]`.
3. Invoke the validator **only** if `fired`, the validator is not a
   `NoOpValidator`, and `max_validator_calls_per_run` has not been exhausted.
4. If the validator budget is exhausted, either pass `validator_result=None`
   (`on_validator_budget_exhausted="skip"`) or pass a synthetic
   `ValidatorResult` with the configured recommendation.
5. Call `policy.decide(trace, trigger_results, validator_result)`.

The sync `validate` raises `RuntimeError` if a validator returns an awaitable;
`validate_async` awaits it instead. The `NoOpValidator` is special-cased so the
healthy path performs no extra work. Validator call budget state is kept in
`trace.metadata["_runtime_validator_call_count"]`.

## How actions are chosen

`DefaultPolicy` picks the highest fired severity, then maps it to an action. A
validator's `recommendation`, when present, can escalate this default. Downgrades
are blocked unless `allow_validator_downgrade=True` and the validator confidence
meets `min_confidence_for_override`; critical severity cannot be downgraded.

| Severity | Default action | Disabled-toggle fallback |
|----------|----------------|--------------------------|
| `low` | `continue` | — |
| `medium` | `retry_last_step` | `continue` (`retry_on_medium=False`) |
| `high` | `interrupt` | `continue` (`interrupt_on_high=False`) |
| `critical` | `abort` | `interrupt` (`abort_on_critical=False`) |

When no trigger fires, the decision is always `continue`.

## Package structure

```
src/agent_runtime_validator/
├── runtime.py            # RuntimeValidator
├── schema/               # ExecutionTrace, events, decisions
├── triggers/             # BaseTrigger + built-in triggers
├── validators/           # BaseValidator + built-in validators
├── policies/             # BasePolicy + DefaultPolicy
├── utils/                # token counting, hashing, redaction
└── integrations/
    └── langgraph/        # adapter + ValidationNode
```

## Extension points

| Base class | Implement | Returns |
|------------|-----------|---------|
| `BaseTrigger` | `evaluate(trace)` | `TriggerResult` |
| `BaseValidator` | `validate(trace, trigger_results)` | `ValidatorResult` or awaitable |
| `BasePolicy` | `decide(trace, triggered, validator_result)` | `ValidationDecision` |

See [`examples/custom_trigger.py`](../../examples/custom_trigger.py) for a
working custom trigger, and [Validators](validators.md) for custom validators.

## Related

- [Overview](overview.md)
- [Quickstart](quickstart.md)
- [Triggers](triggers.md)
- [Validators](validators.md)
- [Design decisions](design-decisions.md)
