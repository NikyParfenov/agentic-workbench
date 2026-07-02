# Runtime Validator — Overview

## Purpose

`agent-runtime-validator` watches an agentic system **while it runs** and decides
whether execution should keep going, retry, reroute, pause, or stop. This page
explains what it does and which problems it solves. For hands-on setup see the
[Quickstart](quickstart.md); for the data model and internals see
[Architecture](architecture.md).

## What problem it solves

Input/output guardrails check individual requests. Offline evaluation checks a
run after it finishes. Neither catches a run that is *currently* going wrong —
looping, stalling, or burning budget. This library inspects the live execution
trace and reacts before the run wastes more time or money.

## Use cases

| Symptom in a run | Trigger that catches it |
|------------------|-------------------------|
| Agent calls the same tool over and over | `SameToolLoopTrigger` |
| Agent repeats the *identical* call (hallucinated retry) | `SameToolSameArgsLoopTrigger` |
| Two agents hand off back and forth forever | `AgentPingPongTrigger` |
| Routing bounces through too many agents | `MaxRoutesTrigger` |
| Many tool calls but nothing produced | `NoProgressTrigger` |
| Tools keep failing | `ToolErrorRateTrigger` |
| Token budget exceeded | `MaxContextTokensTrigger` |
| Wall-clock budget exceeded | `MaxExecutionTimeTrigger` |
| Too many tool calls overall | `MaxToolCallsTrigger` |
| A specialist agent never used its tools | `NoToolUsageTrigger` |
| Too many agent-to-agent delegations overall | `MaxAgentCallsTrigger` |
| Supervisor keeps delegating to the same subagent | `AgentDelegationLoopTrigger` |
| A subagent was called but returned no output | `SubagentNoOutputTrigger` |

See [Triggers](triggers.md) for parameters and tuning.

## Pipeline overview

You hand a snapshot of the run (an `ExecutionTrace`) to a `RuntimeValidator`.
It runs three stages and returns a `ValidationDecision`:

1. **Triggers** — fast, deterministic checks over the trace. Each returns
   whether it fired, at what severity, and supporting evidence.
2. **Validator** *(optional)* — a deeper check. In the default
   `validator_mode="on_trigger"` it runs **only if a trigger fired** and the
   validator call budget allows it; with `validator_mode="always"` it runs on
   every validation call. It can be a deterministic scorer
   (`TriggerScoreValidator`), schema-based, or an LLM judge. Skipped by
   default when no validator is configured, and skipped after budget exhaustion
   when `on_validator_budget_exhausted="skip"`.
3. **Policy** — turns the fired triggers (and any validator recommendation) into
   a single action.

Because the validator (in the default mode) only runs when something already
looks wrong — and can be capped with `max_validator_calls_per_run` — the common
"everything is fine" path stays cheap and never calls an LLM. Field-level
details of every stage live in [Architecture](architecture.md).

> Runnable starting point: [`examples/basic_runtime.py`](../../examples/basic_runtime.py).

## Actions

The decision's `action` tells your application what to do next. Honoring it is up
to you.

| Action | Meaning |
|--------|---------|
| `continue` | Everything looks fine; proceed |
| `retry_last_step` | Recoverable issue; re-run the previous step |
| `reroute` | Redirect to a different agent |
| `interrupt` | Pause and escalate (e.g. to a human) |
| `abort` | Stop the run immediately |

How severities map to these actions is covered in
[Architecture](architecture.md#how-actions-are-chosen).

## Related

- [Quickstart](quickstart.md)
- [Architecture](architecture.md)
- [Triggers](triggers.md)
- [Validators](validators.md)
- [LangGraph integration](langgraph.md)
- [Design decisions](design-decisions.md)
- [Roadmap](roadmap.md)
- [Project README](../../README.md)
