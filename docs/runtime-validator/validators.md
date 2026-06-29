# Runtime Validator — Validators

## Purpose

Reference for validators: the optional deep-check stage that runs only after a
trigger fires. Validators can confirm a problem, add detail, and recommend an
action that overrides the policy default.

## When a validator runs

A validator is invoked only when **both** are true:

1. At least one trigger fired, and
2. The configured validator is not the default `NoOpValidator`.

This keeps the healthy path free of extra work. If you pass no validator, the
runtime uses `NoOpValidator` and skips the stage entirely.

Import validators from `agent_runtime_validator.validators`.

## Built-in validators

| Validator | Checks | Recommends on failure |
|-----------|--------|-----------------------|
| `NoOpValidator` | Nothing — default pass-through | n/a |
| `JsonSchemaValidator` | Tool call **args** and tool **results** against JSON Schemas | `interrupt` |
| `ToolArgumentValidator` | Args of **completed** tool calls against JSON Schemas | `interrupt` |
| `LLMJudgeValidator` | Sends the trace summary + fired triggers to an LLM | from the model |

### JsonSchemaValidator

Validates two things against schemas you register by tool name: the arguments of
every tool call, and the output of every tool result (parsed as JSON, falling
back to the raw string). Any mismatch is collected as an issue.

```python
from agent_runtime_validator.validators import JsonSchemaValidator

validator = JsonSchemaValidator(
    arg_schemas={
        "search_gene": {
            "type": "object",
            "properties": {"gene": {"type": "string"}},
            "required": ["gene"],
            "additionalProperties": False,
        },
    },
    result_schemas={ "search_gene": {"type": "object"} },
)
```

### ToolArgumentValidator

Like the args half of `JsonSchemaValidator`, but only checks calls that already
have a matching `ToolResult` — i.e. calls that actually completed. Use it when
you only care about validating calls after the fact.

```python
from agent_runtime_validator.validators import ToolArgumentValidator

validator = ToolArgumentValidator(arg_schemas={"search_gene": {...}})
```

> Runnable version: [`examples/tool_argument_validation.py`](../../examples/tool_argument_validation.py).

### LLMJudgeValidator

Formats a prompt from the trace summary and the fired triggers, sends it to a
model callable, and parses the JSON response into a `ValidatorResult`. The model
is any `Callable[[str], str]` (sync) or `Callable[[str], Awaitable[str]]`
(async). If the response cannot be parsed, the result is `valid=False`,
`confidence=0.0`, recommending `interrupt`.

```python
from agent_runtime_validator.validators import LLMJudgeValidator

judge = LLMJudgeValidator(model=lambda prompt: my_client.generate(prompt))
```

Override the prompt with `prompt_template=`; the default is exported as
`DEFAULT_JUDGE_PROMPT`. The checklist it uses is below.

> Runnable version: [`examples/llm_judge.py`](../../examples/llm_judge.py).

## What the default LLM judge looks for

The default prompt asks the model to detect:

- Repeated tool calls
- Identical tool calls with identical arguments
- Excessive routing between agents
- Tool failures and retries
- Lack of progress despite many actions
- Contradictory tool outputs
- Hallucinated execution claims
- Hallucinated tool arguments (invented IDs, datasets, file paths, thresholds, filters, sample names)
- Unnecessary iterations
- Opportunities to reroute execution to a more suitable agent

Because the judge only runs after a deterministic trigger fires, this deep —
and potentially expensive — analysis happens only when a run already looks
suspect.

## The ValidatorResult

A validator returns a `ValidatorResult`. Its `recommendation` overrides the
policy's severity-based action when present.

| Field | Type | Meaning |
|-------|------|---------|
| `valid` | `bool` | Whether the run passed validation |
| `confidence` | `float` 0–1 | Validator's confidence |
| `issues` | `list[str]` | Specific problems found |
| `recommendation` | action | `continue`, `retry_last_step`, `reroute`, `interrupt`, `abort` |
| `reason` | `str` | Explanation |
| `suggested_next_agent` | `str \| None` | Hint for `reroute` |
| `suggested_message` | `str \| None` | Message to inject on the next step |

The decision exposes this object as `decision.validator_result`.

## Sync vs async

`validate(trace)` runs validators synchronously. If a validator returns an
awaitable (e.g. an async LLM call), sync `validate` raises `RuntimeError` — use
`validate_async(trace)` instead, which awaits the result.

```python
decision = await validator.validate_async(trace)
```

`LLMJudgeValidator` detects an async model and enforces this rule with a clear
error message.

### TriggerScoreValidator

A deterministic alternative to the LLM judge for cases where triggers alone
provide enough signal. Aggregates fired trigger results using configurable
weights, computes a risk score, and returns a recommendation when the score
crosses a threshold.

```python
from agent_runtime_validator.validators import TriggerScoreValidator

validator = TriggerScoreValidator(
    weights={
        "NoToolUsageTrigger": 2.0,
        "NoProgressTrigger": 2.0,
        "AgentPingPongTrigger": 3.0,
    },
    threshold=3.0,
    recommendation="reroute",
    max_attempts=1,
)
```

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `weights` | — | Score per trigger name (unweighted triggers contribute 0) |
| `threshold` | — | Score at or above which the recommendation fires |
| `recommendation` | `"reroute"` | Action to recommend when threshold is crossed |
| `max_attempts` | `1` | Loop guard — after this many non-continue recommendations, falls back to `interrupt` |

Below threshold the validator returns `recommendation="continue"`. At or above
it, the configured recommendation is returned and the counter incremented. Once
the counter reaches `max_attempts`, the validator returns
`"interrupt"` regardless — preventing infinite reroute/retry loops.

Use cases: no tool usage + no progress → reroute; agent ping-pong → interrupt;
same-tool-same-args loop → retry or reroute — anywhere the trigger evidence is
sufficient and an LLM call would add latency and cost without value.

> Runnable version: [`examples/trigger_score.py`](../../examples/trigger_score.py).

## Writing a custom validator

Subclass `BaseValidator` and implement `validate`. Return a `ValidatorResult`
for sync work, or an awaitable for async work.

```python
from agent_runtime_validator import BaseValidator, ValidatorResult, ExecutionTrace
from agent_runtime_validator import TriggerResult

class NonEmptyArtifactValidator(BaseValidator):
    def validate(self, trace: ExecutionTrace, trigger_results: list[TriggerResult]) -> ValidatorResult:
        empty = [a.artifact_id for a in trace.artifacts if not a.content.strip()]
        ok = not empty
        return ValidatorResult(
            valid=ok,
            confidence=1.0,
            issues=[] if ok else [f"Empty artifact: {i}" for i in empty],
            recommendation="continue" if ok else "interrupt",
            reason="All artifacts non-empty" if ok else f"{len(empty)} empty artifact(s)",
        )
```

## Related

- [Overview](overview.md)
- [Architecture](architecture.md)
- [Quickstart](quickstart.md)
- [Triggers](triggers.md)
- [Design decisions](design-decisions.md)
