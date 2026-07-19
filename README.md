# agentic-workbench

![CI](https://github.com/NikyParfenov/agentic-workbench/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
![License](https://img.shields.io/github/license/NikyParfenov/agentic-workbench)

A toolkit for building, validating, evaluating, and benchmarking agentic systems.

## agent-runtime-validator

Runtime validation and recovery for agentic systems.

> **Status: v0.1-alpha.** The core API is usable, but config-driven validation and broader framework integrations are still evolving.

> **Examples disclaimer:** All examples use fictional agents, tools, datasets, and workflows for demonstration purposes only.

## Problem Statement

Traditional guardrails validate inputs and outputs. Evaluation frameworks validate completed traces offline.

Neither addresses **runtime execution validation** — detecting failure patterns while an agentic system is actively running:

- Repeated tool loops
- Ping-pong routing between agents
- Excessive routing depth
- No progress despite multiple tool calls
- Repeated failures
- Exploding token/cost usage

## How this differs from guardrails and observability tools

| Dimension | Guardrails | Observability (LangSmith, Phoenix, LangFuse) | **agent-runtime-validator** |
|---|---|---|---|
| **What it checks** | Individual inputs, outputs, tool safety | Completed traces, offline evals | Execution trajectory *while running* |
| **When it runs** | Before/after a single LLM call | After the run finishes | During the run, at each checkpoint |
| **What it returns** | Pass/fail on content | Metrics, scores, dashboards | Actionable decision: continue, retry, reroute, interrupt, abort |
| **Loop/routing detection** | No | Post-hoc analysis | Real-time, deterministic triggers |

This is not a safety moderation library. It does not replace input/output guardrails or tracing platforms. It fills the gap between them — a lightweight in-process decision layer over execution trajectories.

## When not to use this

- You only need input/output content moderation — use a guardrails library instead.
- You need a hosted tracing dashboard — use LangSmith, Phoenix, or LangFuse.
- You expect automatic trace collection without instrumenting your nodes — this library requires you to populate the trace.
- You need production-stable APIs today — this is alpha software.

## Runtime Validation Concept

`agent-runtime-validator` is a **middleware/validation layer**. It sits alongside your agent graph and inspects the execution trace at runtime. It decides whether execution should:

- **continue** — everything looks fine
- **retry_last_step** — recoverable issue, retry
- **reroute** — redirect to a different agent
- **interrupt** — pause and report to user
- **abort** — stop immediately

## Architecture

```
ExecutionTrace
     ↓
Triggers  →  TriggerResult[]
     ↓  (per validator_mode: when triggers fired, or always — budget permitting)
Validator  →  ValidatorResult | skipped
     ↓
Policy  →  ValidationDecision
```

**Triggers** are deterministic and fast. They fire on observable patterns (loop counts, routing depth, error rates).

**Validators** are optional. In the default `validator_mode="on_trigger"`, `LLMJudgeValidator` invokes an LLM only when triggers fire — not on every step. For retry/reroute loops, set `max_validator_calls_per_run=1` to cap expensive validator calls per trace/run.

**Policies** map trigger severity + validator recommendations to decisions.

### Validator mode

`validator_mode` controls when the validator stage runs:

- `"on_trigger"` (default) — the validator runs only when at least one trigger fires. The common "all-clear" path never calls it. Use for inline mid-run monitoring.
- `"always"` — the validator runs on every validation call, regardless of trigger results. Use when the validator is a quality check that should inspect every completed trace.

```python
validator = RuntimeValidator(
    triggers=[...],
    validator=judge,
    validator_mode="always",
)
```

See [docs/runtime-validator/validators.md](docs/runtime-validator/validators.md#validator-mode) for details.

## Installation

### From source

```bash
git clone https://github.com/NikyParfenov/agentic-workbench.git
cd agentic-workbench
```

### Development setup (uv)

Install all dependencies from `uv.lock`:

```bash
uv sync
```

### Editable installation

Using uv:

```bash
uv pip install -e .
```

With LangGraph integration:

```bash
uv pip install -e ".[langgraph]"
```

Using pip:

```bash
pip install -e .
```

With LangGraph integration:

```bash
pip install -e ".[langgraph]"
```

## Quickstart

```python
from agent_runtime_validator import RuntimeValidator
from agent_runtime_validator.triggers import MaxRoutesTrigger, SameToolLoopTrigger, NoProgressTrigger

validator = RuntimeValidator(
    triggers=[
        MaxRoutesTrigger(max_routes=10),
        SameToolLoopTrigger(max_repeats=3),
        NoProgressTrigger(min_tool_calls=5),
    ],
)

decision = validator.validate(trace)
if decision.is_terminal:                 # interrupt or abort — stop the run
    print(f"Stopping: {decision.action} — {decision.reason}")
elif not decision.should_continue:       # retry_last_step or reroute — recover
    print(f"Recovery requested: {decision.action} — {decision.reason}")
```

`should_continue` means "no intervention requested" (`action == "continue"`),
not "the run must halt" — recovery actions keep the run alive. Use
`decision.is_terminal` to check for a stop.

For a runnable loop-detection demo:

```bash
uv run python examples/basic_loop_detection.py
```

```
=== Validation decision ===
Action:      interrupt
Severity:    high
Triggered:   ['SameToolSameArgsLoopTrigger', 'NoProgressTrigger', 'ToolErrorRateTrigger']
```

## TraceBuilder

`TraceBuilder` is a fluent API for constructing an `ExecutionTrace` step by step, without having to assemble Pydantic objects manually:

```python
from agent_runtime_validator import TraceBuilder

builder = TraceBuilder(run_id="run-42")
builder.record_message("user", "analyze dataset X")
builder.record_tool_call("load_data", call_id="c1", args={"path": "data.csv"})
builder.record_tool_result("c1", "load_data", output="loaded 1000 rows")
trace = builder.build()

decision = validator.validate(trace)
```

Merge a finished subagent trace back into the supervisor builder:

```python
parent_builder = TraceBuilder.from_trace(parent_trace)
parent_builder.merge_trace(subagent_trace)
merged = parent_builder.build()
```

## LangGraph Example

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
builder.add_edge("research_agent", "validation")
builder.add_edge("validation", "supervisor")
```

The node reads `state["trace"]` (an `ExecutionTrace`) and writes `state["decision"]` (a `ValidationDecision`).

For nested graphs, keep inner tool messages out of the outer chat history and
lift them into `ExecutionTrace` with `lift_subgraph_messages(...)` — see
[docs/runtime-validator/langgraph.md](docs/runtime-validator/langgraph.md#lifting-structured-subgraph-messages).

## Trigger Reference

Triggers are grouped by the trace event type they inspect.

### Tool and artifact triggers

```python
from agent_runtime_validator.triggers import (
    MaxToolCallsTrigger,          # too many ToolCall entries
    SameToolLoopTrigger,          # same tool repeated N times
    SameToolSameArgsLoopTrigger,  # exact same ToolCall repeated
    NoProgressTrigger,            # many ToolCall entries, no ArtifactEvent entries
    ToolErrorRateTrigger,         # ToolResult error rate above threshold
    NoToolUsageTrigger,           # watched agents have too few ToolCall entries
)
```

### Routing triggers

```python
from agent_runtime_validator.triggers import (
    MaxRoutesTrigger,             # too many RoutingEvent entries
    AgentPingPongTrigger,         # alternating RoutingEvent A→B→A→B
)
```

### Agent-call triggers

Use these when your trace records semantic subagent calls as `AgentCall` events.
If you only record routing and tool events, you can ignore this group.

```python
from agent_runtime_validator.triggers import (
    MaxAgentCallsTrigger,         # too many AgentCall entries
    AgentDelegationLoopTrigger,   # repeated AgentCall caller→callee pair
    SubagentNoOutputTrigger,      # AgentCall recorded with output=None
)
```

### Budget triggers

```python
from agent_runtime_validator.triggers import (
    MaxContextTokensTrigger,      # token budget exceeded
    MaxExecutionTimeTrigger,      # wall-clock time limit
)
```

## ToolArgumentValidator Example

Deterministically validates tool call arguments against a schema registry:

```python
from agent_runtime_validator.validators import ToolArgumentValidator

validator = RuntimeValidator(
    triggers=[SameToolLoopTrigger(max_repeats=3)],
    validator=ToolArgumentValidator(
        arg_schemas={
            "lookup_record": {
                "type": "object",
                "properties": {"record_id": {"type": "string"}},
                "required": ["record_id"],
                "additionalProperties": False,
            }
        },
    ),
)
```

## Trigger Score Validator

Deterministic weighted scoring — no LLM needed. Aggregates fired triggers into a
risk score and recommends an action when the threshold is crossed:

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

`max_attempts` prevents infinite reroute/retry loops — after the limit, the
validator switches to `"interrupt"`.

## LLM Judge

```python
from agent_runtime_validator import RuntimeValidator
from agent_runtime_validator.validators import LLMJudgeValidator

# Sync model (any callable: str → str)
judge = LLMJudgeValidator(
    model=lambda prompt: model_backend.complete(prompt)
)

validator = RuntimeValidator(
    triggers=[...],
    validator=judge,
    max_validator_calls_per_run=1,          # recommended for LLM judges
    on_validator_budget_exhausted="skip",  # default: let triggers + policy decide
)

# Async model (callable: str → Awaitable[str])
async_judge = LLMJudgeValidator(
    model=lambda prompt: async_model_backend.complete(prompt)
)
decision = await validator.validate_async(trace)
```

The LLM judge checks for: repeated tool calls, repeated failures, ping-pong routing, lack of progress, hallucinated tool arguments.

Pass `reference_examples=[JudgeExample(label="good"|"bad", trace=...)]` to give the judge historical `ExecutionTrace` precedents as non-binding few-shot calibration — see [Validators — Reference cases](docs/runtime-validator/validators.md#reference-cases-few-shot-precedents).

## Recovery Policy

```python
from agent_runtime_validator.policies import DefaultPolicy

policy = DefaultPolicy(
    retry_on_medium=True,    # medium severity → retry_last_step
    interrupt_on_high=True,  # high severity → interrupt
    abort_on_critical=True,  # critical severity → abort
    max_retries_per_run=3,   # retry budget — exhausted retries escalate to interrupt
)

validator = RuntimeValidator(triggers=[...], policy=policy)
```

## Saving and Replaying Traces

Save a trace to disk and replay it offline against an updated validator config without re-running the graph:

```python
from agent_runtime_validator import save_trace, load_trace, replay, RuntimeValidator
from agent_runtime_validator.triggers import SameToolLoopTrigger

# Save after the run
save_trace(trace, "traces/run-abc.json")

# Later: reload and replay with tightened thresholds
tuned = RuntimeValidator(triggers=[SameToolLoopTrigger(max_repeats=2)])
trace = load_trace("traces/run-abc.json")
decision = replay(trace, tuned)
print(decision.action, decision.triggered_by)
```

`save_trace` / `load_trace` read and write UTF-8 JSON. `replay_async` is available for async validators.

## Logging

```python
import logging

logging.basicConfig(level=logging.INFO)

# Full trigger-by-trigger detail:
logging.getLogger("agent_runtime_validator").setLevel(logging.DEBUG)
```

## Roadmap

- **v0.1-alpha** — LangGraph, deterministic + supervisor triggers, trigger score validator, LLM judge, `TraceBuilder`, trace import/export, offline replay, `validator_mode`, policy safety controls, decision routing, logging
- **v0.2** — Production trace ergonomics: config-driven validation, artifact mapping/validation, agent-scoped/windowed triggers, trace-emitter patterns, safer redaction defaults
- **v0.3** — CrewAI, LlamaIndex, OpenAI Agents SDK, PydanticAI, provider examples, packaging/integration polish
- **v0.4** — CompositeValidator, ExecutionInvariantValidator, trigger composition, cost tracking, offline observability trace importers
- **v1.0** — Incremental runtime API, distributed traces, OpenTelemetry/Prometheus, stable observability integrations
