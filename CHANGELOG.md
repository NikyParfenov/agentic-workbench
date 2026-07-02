# Changelog

## 0.1.0a1 — 2026-06-30

Initial alpha release.

### Added

- `RuntimeValidator` pipeline: triggers → validator → policy → decision, with
  sync (`validate`) and async (`validate_async`) paths
- `validator_mode` (`"on_trigger"` / `"always"`) separating mid-run monitoring
  from run-every-time quality checks
- Per-run validator call budget: `max_validator_calls_per_run` and
  `on_validator_budget_exhausted`
- 13 deterministic triggers: `MaxToolCallsTrigger`, `MaxRoutesTrigger`,
  `MaxContextTokensTrigger`, `MaxExecutionTimeTrigger`, `SameToolLoopTrigger`,
  `SameToolSameArgsLoopTrigger`, `AgentPingPongTrigger`, `NoProgressTrigger`,
  `ToolErrorRateTrigger`, `NoToolUsageTrigger`, `MaxAgentCallsTrigger`,
  `AgentDelegationLoopTrigger`, `SubagentNoOutputTrigger`
- `DefaultPolicy` with severity-based escalation and opt-in downgrade
  (`allow_validator_downgrade`, `min_confidence_for_override`)
- `LLMJudgeValidator`: sync/async LLM-based deep analysis with structured
  `JudgeFinding` output, `TraceFormatConfig` prompt shaping, truncation
  strategies, redaction hook, retries, confidence clamping, and robust JSON
  extraction (fenced blocks, preamble, trailing prose)
- `TriggerScoreValidator`: deterministic weighted risk scoring with loop guard
- `ToolArgumentValidator` and `JsonSchemaValidator`: JSON Schema-based
  argument/result validation
- `TraceBuilder`: fluent incremental trace construction, `from_trace`, and
  trace merging
- Trace I/O and offline replay: `trace_to_json`, `trace_from_json`,
  `save_trace`, `load_trace`, `replay`, `replay_async`
- LangGraph integration: `ValidationNode` (sync/async, custom `trace_builder`),
  `create_validation_router` (dynamic reroute gated by explicit allowlist),
  `state_to_trace`, `get_trace_from_state`, `build_trace_from_state`,
  `from_langchain_messages`, `from_subgraph_thoughts`,
  `lift_subgraph_messages`, `TraceBuilderFn`
- `ExecutionTrace`, `ValidationDecision`, `ValidatorResult`, `JudgeFinding`
  Pydantic v2 models
- Standard library logging under `agent_runtime_validator` logger (opt-in,
  never globally configured)
- GitHub Actions CI on Python 3.11, 3.12, 3.13
- Examples: `basic_runtime.py`, `basic_loop_detection.py`, `custom_trigger.py`,
  `tool_argument_validation.py`, `trigger_score.py`, `llm_judge.py`,
  `redacted_judge.py`, `langgraph_supervisor.py`,
  `langgraph_always_validator.py`
- Documentation: overview, quickstart, triggers, validators, policy
  configuration, LangGraph integration, architecture, design decisions, roadmap
- Apache-2.0 license, PEP 561 `py.typed` marker
