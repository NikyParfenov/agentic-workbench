# Runtime Validator — Validators

## Purpose

Reference for validators: the optional deep-check stage that runs only after a
trigger fires and the validator call budget allows it. Validators can confirm a
problem, add detail, and recommend an action; policy combines that recommendation
with trigger severity safeguards.

## When a validator runs

The default `"on_trigger"` mode invokes the validator only when all three are
true:

1. At least one trigger fired,
2. The configured validator is not the default `NoOpValidator`, and
3. `max_validator_calls_per_run` is unset or has not been exhausted.

This keeps the healthy path free of extra work. If you pass no validator, the
runtime uses `NoOpValidator` and skips the stage entirely.

Use `validator_mode="always"` to always invoke the validator (see below).

Import validators from `agent_runtime_validator.validators`.

## Validator mode

`RuntimeValidator.validator_mode` controls when the validator is invoked.

| Mode | Validator called when… |
|------|------------------------|
| `"on_trigger"` (default) | At least one trigger fired |
| `"always"` | Always — regardless of trigger results |

```python
from agent_runtime_validator import RuntimeValidator, ValidatorMode
from agent_runtime_validator.validators import LLMJudgeValidator

# Post-run quality gate: validator runs on every completed trace
runtime = RuntimeValidator(
    triggers=[...],               # triggers still run as usual
    validator=LLMJudgeValidator(model=call_model),
    validator_mode="always",
)
decision = runtime.validate(completed_trace)
```

**`"on_trigger"` (default)** — the validator is the optional deep-check that
runs only when something already looks wrong. The common "all-clear" path never
calls the validator. Use this for inline mid-run monitoring.

**`"always"`** — the validator always runs, making it a mandatory inspection
step. Use this when you want an LLM judge (or schema validator) to review every
completed trace, not just ones that tripped a trigger.

### always + DefaultPolicy

When `validator_mode="always"` and the validator recommends stopping
(anything other than `"continue"`), `DefaultPolicy` honors that recommendation
even if no triggers fired. If triggers also fired, the higher severity of the
two wins.

Escalations are honored regardless of validator `confidence` (fail-closed);
the `min_confidence_for_override` gate applies only to downgrades. Keep this
in mind when choosing `fallback_recommendation` for `LLMJudgeValidator` in
`"always"` mode — a malformed judge response escalates with `confidence=0.0`.

### always + budget

`max_validator_calls_per_run` still applies in `"always"` mode. Once the
budget is exhausted, `on_validator_budget_exhausted` controls the fallback —
exactly the same as in `"on_trigger"` mode.

## Built-in validators

| Validator | Checks | Recommends on failure |
|-----------|--------|-----------------------|
| `NoOpValidator` | Nothing — default pass-through | n/a |
| `JsonSchemaValidator` | Tool call **args** and tool **results** against JSON Schemas | `interrupt` |
| `ToolArgumentValidator` | Args of **completed** tool calls against JSON Schemas | `interrupt` |
| `LLMJudgeValidator` | Sends the trace summary + fired triggers to an LLM | from the model |
| `TriggerScoreValidator` | Weighted risk score from fired triggers | configurable |

### JsonSchemaValidator

Validates two things against schemas you register by tool name: the arguments of
every tool call, and the output of every tool result (parsed as JSON, falling
back to the raw string). Any mismatch is collected as an issue.

```python
from agent_runtime_validator.validators import JsonSchemaValidator

validator = JsonSchemaValidator(
    arg_schemas={
        "lookup_record": {
            "type": "object",
            "properties": {"record_id": {"type": "string"}},
            "required": ["record_id"],
            "additionalProperties": False,
        },
    },
    result_schemas={ "lookup_record": {"type": "object"} },
)
```

### ToolArgumentValidator

Like the args half of `JsonSchemaValidator`, but only checks calls that already
have a matching `ToolResult` — i.e. calls that actually completed. Use it when
you only care about validating calls after the fact.

```python
from agent_runtime_validator.validators import ToolArgumentValidator

validator = ToolArgumentValidator(arg_schemas={"lookup_record": {...}})
```

> Runnable version: [`examples/tool_argument_validation.py`](../../examples/tool_argument_validation.py).

### LLMJudgeValidator

Formats a prompt from the trace summary and the fired triggers, sends it to a
model callable, and parses the JSON response into a `ValidatorResult`. The model
is any `Callable[[str], str]` (sync) or `Callable[[str], Awaitable[str]]`
(async). If the response cannot be parsed, the result is `valid=False`,
`confidence=0.0`, recommending `interrupt`.

The trace is always formatted with bounded limits — no unlimited trace dumping.
Use `TraceFormatConfig` to control those limits.

```python
from agent_runtime_validator.validators import LLMJudgeValidator, TraceFormatConfig

judge = LLMJudgeValidator(
    model=lambda prompt: model_backend.complete(prompt),
    trace_format=TraceFormatConfig(
        max_events_per_section=100,
        max_chars_per_field=1000,
        max_chars_artifact_preview=500,
        truncation="middle_ellipsis",
    ),
)
```

Override the prompt with `prompt_template=`; the default lives in
`agent_runtime_validator/validators/prompts.py` and is exported as
`DEFAULT_JUDGE_PROMPT`. The checklist it uses is below.

#### Reference cases (few-shot precedents)

Pass `reference_examples` to give the judge historical precedents — past runs
that were reviewed as healthy or problematic — so it can calibrate its verdict.
Each `JudgeExample` is a structured **`ExecutionTrace`** (not prose), plus:

- `label` — a **retrospective, curated assessment** of that historical case:
  `"good"` (reviewed as healthy) or `"bad"` (reviewed as problematic).
- `note` — an optional short explanation of *why* the case was judged that way.

The `label` is reference metadata describing the past case only. It is **not**
an expected verdict, recommendation, or routing rule for the candidate: the
judge is instructed to treat cases as non-binding context and evaluate the
candidate trace on its own evidence. There are deliberately no `expected_valid`
/ `expected_recommendation` fields — strict, known routing rules are a separate,
deterministic use case, not this one.

```python
from agent_runtime_validator import TraceBuilder
from agent_runtime_validator.validators import LLMJudgeValidator, JudgeExample

good_case = (
    TraceBuilder(run_id="hist-good")
    .record_message("user", "Find recent evidence on topic X and cite sources")
    .record_routing("supervisor", "literature_researcher", reason="needs sourced evidence")
    .record_artifact("a1", "report", "summary with citations")
    .build()
)
bad_case = (
    TraceBuilder(run_id="hist-bad")
    .record_message("user", "Find recent evidence on topic Y and cite sources")
    .record_routing("supervisor", "document_analyst", reason="no retrieval capability")
    .build()   # no grounded artifact was produced
)

judge = LLMJudgeValidator(
    model=lambda prompt: model_backend.complete(prompt),
    reference_examples=[
        JudgeExample(label="good", trace=good_case,
                     note="similar request; source-backed artifact produced"),
        JudgeExample(label="bad", trace=bad_case,
                     note="similar request routed to an agent that produced no grounded evidence"),
    ],
)
```

These are especially useful for **semantic routing evaluation**: a similar
request may have a known-healthy or known-problematic trajectory, and the judge
can weigh the candidate against it — without being forced to copy the past
verdict.

Reference traces are rendered **before** the candidate trace as a labeled
`<reference_cases>` block, using the same `TraceFormatConfig` limits
(truncation, redaction) as the candidate. They are additionally bounded by a
reference budget so a large list cannot blow up the prompt:

- `max_reference_examples` (default `4`) — at most this many cases, in caller order.
- `max_total_reference_chars` (default `12_000`) — combined rendered size; whole
  cases that do not fit are dropped (never truncated mid-case), with a short note
  that cases were omitted. Setting either to `0` disables reference cases.

Trace payloads (messages, tool results, artifacts, errors) are **untrusted
execution data**: the default prompt tells the model not to follow any
instructions found inside reference cases or the candidate trace, and to use
their content only as evidence.

#### TraceFormatConfig

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `max_events_per_section` | `50` | Max items from each trace list (messages, tool calls, etc.) |
| `max_chars_per_field` | `500` | Max chars per text field (args, outputs, errors) |
| `max_chars_artifact_preview` | `200` | Max chars for artifact content preview |
| `max_chars_trigger_evidence` | `300` | Max chars for each fired trigger's evidence JSON |
| `include_trace_details` | `True` | Include per-event trace sections in prompt |
| `truncation` | `"tail"` | Truncation strategy: `"tail"`, `"head"`, or `"middle_ellipsis"` |
| `max_reference_examples` | `4` | Max reference cases rendered, in caller order (`0` disables) |
| `max_total_reference_chars` | `12_000` | Combined size budget for the reference block (`0` disables) |

Truncation strategies:
- `"tail"` — keep the beginning, drop the end (default; preserves context).
- `"head"` — drop the beginning, keep the end (preserves recent content).
- `"middle_ellipsis"` — keep both ends, replace the middle with a marker.

Legacy parameters `max_trace_events` and `include_trace_details` are still
accepted for backward compatibility and map to `trace_format.max_events_per_section`
and `trace_format.include_trace_details` respectively. `trace_format` takes
precedence if both are provided.

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

In the default `"on_trigger"` mode the judge only runs after a deterministic
trigger fires, so this deep — and potentially expensive — analysis happens
only when a run already looks suspect. In `"always"` mode it inspects every
validation call; pair it with `max_validator_calls_per_run` to bound cost.

## The ValidatorResult

A validator returns a `ValidatorResult`. Its `recommendation` can escalate the
policy's severity-based action; downgrades are only accepted when the policy is
configured to allow them and the validator confidence is high enough.

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

## Validator call budget

`RuntimeValidator.max_validator_calls_per_run` limits how many times the
optional validator is invoked for the same trace/run. This is useful for
expensive validators such as `LLMJudgeValidator` in retry/reroute loops.

```python
from agent_runtime_validator import RuntimeValidator
from agent_runtime_validator.validators import LLMJudgeValidator

runtime = RuntimeValidator(
    triggers=[...],
    validator=LLMJudgeValidator(model=call_model),
    max_validator_calls_per_run=1,
    on_validator_budget_exhausted="skip",  # default
)
```

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `max_validator_calls_per_run` | `None` | Max invocations per run; `None` = unlimited, `0` = never call |
| `on_validator_budget_exhausted` | `"skip"` | What to do when budget is exhausted |

Budget state is stored in `trace.metadata["_arv_validator_call_count"]`
and persists across repeated calls to `validate()` on the same `ExecutionTrace`
object. In LangGraph, `ValidationNode` writes the trace back into state after
each call so budget state survives across serialized/checkpointed traces.

### Exhausted behavior options

| `on_validator_budget_exhausted` | Behavior when budget is exhausted |
|---|---|
| `"skip"` | Do not call validator; pass `validator_result=None` to policy. Triggers + policy decide. **Default.** |
| `"continue"` | Return synthetic validator result recommending `continue`; trigger severity may still produce retry/interrupt/abort. |
| `"retry_last_step"` | Return synthetic validator result recommending retry. Use only with bounded retry loops. |
| `"reroute"` | Return synthetic validator result recommending reroute. Useful with fallback/human-review branches. |
| `"interrupt"` | Return synthetic validator result recommending interrupt. Safe fail-closed behavior. |
| `"abort"` | Return synthetic validator result recommending abort. Strict fail-closed behavior. |

> **Note:** `"skip"` does not force `decision.action == "continue"`. If a fired
> trigger has high severity, the default policy still returns `"interrupt"`.
> Similarly, `"continue"` does not guarantee a final continue action — the policy
> also considers fired trigger severity.

**`max_retries` vs `max_validator_calls_per_run`**

`LLMJudgeValidator.max_retries` retries malformed LLM responses inside a
single validator invocation.

`RuntimeValidator.max_validator_calls_per_run` limits how many validator
invocations can happen across the entire run — it guards against repeated
expensive calls during retry/reroute orchestration loops.

## Validator errors

If the validator itself raises — a network timeout, provider rate limit, or a
bug in a custom validator — `RuntimeValidator` contains the exception instead
of letting it crash the host run. `on_validator_error` controls the fallback:

```python
runtime = RuntimeValidator(
    triggers=[...],
    validator=LLMJudgeValidator(model=call_model),
    on_validator_error="skip",   # default
)
```

| `on_validator_error` | Behavior when the validator raises |
|---|---|
| `"skip"` | Treat the validator as unavailable; pass `validator_result=None` to the policy. Triggers + policy decide. **Default.** |
| `"continue"` / `"retry_last_step"` / `"reroute"` / `"interrupt"` / `"abort"` | Return a synthetic validator result with that recommendation, same as the budget-exhausted options above. |

The raw exception is logged (with traceback) but never propagated and never
placed into the decision — only the exception *type* appears in the synthetic
result's reason, so provider error details cannot leak into downstream prompts.

This is deliberately separate from malformed-**output** handling:
`LLMJudgeValidator.max_retries` / `fallback_recommendation` cover a response
that arrived but could not be parsed; `on_validator_error` covers the call not
completing at all. A validator call that raises still consumes one unit of the
validator call budget.

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
