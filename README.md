# agentic-workbench

A toolkit for building, validating, evaluating, and benchmarking agentic systems.

## agent-runtime-validator

Runtime validation and recovery for agentic systems.

## Problem Statement

Traditional guardrails validate inputs and outputs. Evaluation frameworks validate completed traces offline.

Neither addresses **runtime execution validation** — detecting failure patterns while an agentic system is actively running:

- Repeated tool loops
- Ping-pong routing between agents
- Excessive routing depth
- No progress despite multiple tool calls
- Repeated failures
- Exploding token/cost usage

## Why Guardrails Are Insufficient

Input/output guardrails validate individual requests. They cannot observe the **trajectory** — whether the agent is making progress, stuck in a loop, or bouncing between agents unproductively.

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
     ↓  (only if triggers fired)
Validator  →  ValidatorResult
     ↓
Policy  →  ValidationDecision
```

**Triggers** are deterministic and fast. They fire on observable patterns (loop counts, routing depth, error rates).

**Validators** are optional. `LLMJudgeValidator` invokes an LLM only when triggers fire — not on every step.

**Policies** map trigger severity + validator recommendations to decisions.

## Installation

### From source

```bash
git clone https://github.com/nikitaparfenov/agentic-workbench.git
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
if not decision.should_continue:
    print(f"Stopping: {decision.action} — {decision.reason}")
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
builder.add_edge("bio_agent", "validation")
builder.add_edge("validation", "supervisor")
```

The node reads `state["trace"]` (an `ExecutionTrace`) and writes `state["decision"]` (a `ValidationDecision`).

## Trigger Reference

```python
from agent_runtime_validator.triggers import (
    MaxToolCallsTrigger,          # too many tool calls total
    MaxRoutesTrigger,             # too many agent hops
    MaxContextTokensTrigger,      # token budget exceeded
    MaxExecutionTimeTrigger,      # wall-clock time limit
    SameToolLoopTrigger,          # same tool repeated N times
    SameToolSameArgsLoopTrigger,  # exact same call repeated
    AgentPingPongTrigger,         # A→B→A→B routing pattern
    NoProgressTrigger,            # many tool calls, no artifacts
    ToolErrorRateTrigger,         # error rate above threshold
    NoToolUsageTrigger,           # watched agents made no tool calls
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
            "search_gene": {
                "type": "object",
                "properties": {"gene": {"type": "string"}},
                "required": ["gene"],
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
        NoToolUsageTrigger(watched_agents={"bio_agent"}),
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
from agent_runtime_validator.validators import LLMJudgeValidator

# Sync model (any callable: str → str)
judge = LLMJudgeValidator(
    model=lambda prompt: my_client.generate(prompt)
)

# Async model (callable: str → Awaitable[str])
async_judge = LLMJudgeValidator(
    model=lambda prompt: my_async_client.generate(prompt)
)
decision = await validator.validate_async(trace)
```

The LLM judge checks for: repeated tool calls, repeated failures, ping-pong routing, lack of progress, hallucinated tool arguments.

## Recovery Policy

```python
from agent_runtime_validator.policies import DefaultPolicy

policy = DefaultPolicy(
    retry_on_medium=True,    # medium severity → retry_last_step
    interrupt_on_high=True,  # high severity → interrupt
    abort_on_critical=True,  # critical severity → abort
)

validator = RuntimeValidator(triggers=[...], policy=policy)
```

## Logging

```python
import logging

logging.basicConfig(level=logging.INFO)

# Full trigger-by-trigger detail:
logging.getLogger("agent_runtime_validator").setLevel(logging.DEBUG)
```

## Roadmap

- **v0.1** — LangGraph, deterministic triggers, trigger score validator, LLM Judge, recovery decisions
- **v0.2** — Offline replay, trace import/export, artifact validation, pre-execution hooks
- **v0.3** — CrewAI, LlamaIndex, OpenAI Agents SDK, PydanticAI, agent-scoped triggers
- **v0.4** — CompositeValidator, ExecutionInvariantValidator, trigger composition, cost tracking
- **v1.0** — Decentralized multi-agent support, graph cycle/deadlock detection
