# Runtime Validator — Roadmap

## Purpose

Outline what exists today and what is planned, so users can see where the module
is headed and which integrations to expect. Items beyond v0.1 are aspirational
and may change.

## v0.1-alpha — current

| Area | Included |
|------|----------|
| Triggers | Max calls, tokens, time, routes; same-tool loop; same-tool-same-args loop; agent ping-pong; no progress; tool error rate; no tool usage; max agent delegations; agent delegation loop; subagent no output |
| Validators | NoOp, JSON Schema, tool argument, LLM judge (with trace details, agent delegations section, truncation notices, confidence clamping, retries, redaction, robust JSON extraction), trigger score; `TraceFormatConfig` for prompt shaping |
| Policy | `DefaultPolicy` with severity-based action mapping, validator escalation/downgrade safety controls |
| Runtime | Sync and async `RuntimeValidator` with standard-library logging and optional per-run validator call budget; `validator_mode` (`on_trigger` / `always`) |
| Trace API | `TraceBuilder` fluent API; `trace_to_json`, `trace_from_json`, `save_trace`, `load_trace`; `replay`, `replay_async` for offline re-validation |
| Integrations | LangGraph (`ValidationNode`, `state_to_trace`, `create_validation_router`, `get_trace_from_state`, `build_trace_from_state`, `from_langchain_messages`, `from_subgraph_thoughts`, `lift_subgraph_messages`, `TraceBuilderFn`, async `async_call`) |
| OSS | GitHub Actions CI, CONTRIBUTING.md, Apache-2.0, typed, examples |

## v0.2 — production trace ergonomics and config

- Config-driven validation: YAML/TOML configs and reusable profiles (strict,
  cost_saver, supervisor, final_output_gate)
- Artifact mapping and validation: helpers/patterns for converting project
  artifacts into `ArtifactEvent` plus schema/content checks on produced artifacts
- Agent-scoped and windowed trigger filtering (`agent_names`, `window_size`,
  `consecutive`, `ignore_tools`) for large nested traces
- Trace-emitter patterns for nested graphs: documenting `state["trace"]` as the
  machine-readable telemetry channel separate from chat `messages`
- Safer default trace formatting/redaction profiles for traces passed to LLM
  judges
- Pre-execution tool argument validation hooks

## v0.3 — more frameworks and integration polish

- Integrations: CrewAI, LlamaIndex, OpenAI Agents SDK, PydanticAI
- Optional `tiktoken` extra for token counting (keep `len//4` fallback)
- LLM judge provider examples: OpenAI, Anthropic, LiteLLM, local models
- Packaging and integration examples for non-LangGraph runtimes

## v0.4 — composition and recovery

- `CompositeValidator` — chain validators with configurable aggregation
- `ExecutionInvariantValidator` — user-defined invariants over trace state
- Configurable trigger composition (AND/OR logic between triggers)
- Cost-tracking trigger (dollar budget, not just tokens)
- Webhook/callback on decision
- Offline observability trace importers, starting with one provider before
  broadening support

## v1.0 — scale

- Incremental runtime API (`on_tool_call`, `on_tool_result`, `checkpoint`)
- Decentralized multi-agent support
- Graph cycle and deadlock detection
- Distributed trace support (span_id, parent_span_id, service spans)
- LangSmith / LangFuse / Phoenix trace importers and replay workflows
- OpenTelemetry / Prometheus integration
- Real-time dashboard / observability integration
- Stable extension API for custom triggers, validators, and policies

## Related

- [Overview](overview.md)
- [Architecture](architecture.md)
- [Design decisions](design-decisions.md)
- [Project README](../../README.md)
