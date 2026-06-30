# Changelog

## 0.1.0a1 — 2026-06-30

Initial alpha release.

### Added

- `RuntimeValidator` pipeline: triggers → validator → policy → decision
- 10 deterministic triggers: `MaxToolCallsTrigger`, `MaxRoutesTrigger`, `MaxContextTokensTrigger`, `MaxExecutionTimeTrigger`, `SameToolLoopTrigger`, `SameToolSameArgsLoopTrigger`, `AgentPingPongTrigger`, `NoProgressTrigger`, `ToolErrorRateTrigger`, `NoToolUsageTrigger`
- `DefaultPolicy` with severity-based escalation and opt-in downgrade
- `LLMJudgeValidator`: sync/async LLM-based deep analysis with structured `JudgeFinding` output
- `TriggerScoreValidator`: deterministic weighted risk scoring with loop guard
- `ToolArgumentValidator`: JSON Schema-based argument validation
- LangGraph integration: `ValidationNode`, `create_validation_router`, `state_to_trace`
- `ExecutionTrace`, `ValidationDecision`, `ValidatorResult`, `JudgeFinding` Pydantic v2 models
- Standard library logging under `agent_runtime_validator` logger (opt-in, never globally configured)
- GitHub Actions CI on Python 3.11, 3.12, 3.13
- Examples: `examples/basic_loop_detection.py`, `examples/langgraph_supervisor.py`
- Documentation: overview, quickstart, triggers, validators, policy configuration, LangGraph integration, architecture, design decisions
- Apache-2.0 license, PEP 561 `py.typed` marker
