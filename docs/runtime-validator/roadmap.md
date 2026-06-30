# Runtime Validator — Roadmap

## Purpose

Outline what exists today and what is planned, so users can see where the module
is headed and which integrations to expect. Items beyond v0.1 are aspirational
and may change.

## v0.1-alpha — current

| Area | Included |
|------|----------|
| Triggers | Max calls, tokens, time, routes; same-tool loop; same-tool-same-args loop; agent ping-pong; no progress; tool error rate; no tool usage |
| Validators | NoOp, JSON Schema, tool argument, LLM judge (with trace details, retries, redaction, robust JSON extraction), trigger score |
| Policy | `DefaultPolicy` with severity-based action mapping, validator escalation/downgrade safety controls |
| Runtime | Sync and async `RuntimeValidator` with standard-library logging and optional per-run validator call budget |
| Integrations | LangGraph (`ValidationNode`, `state_to_trace`, `create_validation_router`) |
| OSS | GitHub Actions CI, CONTRIBUTING.md, Apache-2.0, typed, examples |

## v0.2 — offline, import, and config

- Trace import/export: JSONL, `ExecutionTrace.model_dump_json`, offline replay helper
- Trace import from LangSmith, LangFuse, Arize Phoenix
- Config-driven validation: YAML/TOML configs, presets (strict, cost_saver, research_agent, supervisor)
- Artifact content validation (schema checks on produced artifacts)
- Pre-execution tool argument validation hooks

## v0.3 — more frameworks and trigger improvements

- Integrations: CrewAI, LlamaIndex, OpenAI Agents SDK, PydanticAI
- Agent-scoped trigger filtering (`agent_names`, `window_size`, `consecutive`, `ignore_tools`)
- Optional `tiktoken` extra for token counting (keep `len//4` fallback)
- LLM judge provider examples: OpenAI, Anthropic, LiteLLM, local models
- Default redaction utilities (emails, phone numbers, API keys, bearer tokens)

## v0.4 — composition and recovery

- `CompositeValidator` — chain validators with configurable aggregation
- `ExecutionInvariantValidator` — user-defined invariants over trace state
- Configurable trigger composition (AND/OR logic between triggers)
- Cost-tracking trigger (dollar budget, not just tokens)
- Webhook/callback on decision

## v1.0 — scale

- Incremental runtime API (`on_tool_call`, `on_tool_result`, `checkpoint`)
- Decentralized multi-agent support
- Graph cycle and deadlock detection
- Distributed trace support (span_id, parent_span_id, service spans)
- LangSmith / LangFuse / Phoenix trace importers
- OpenTelemetry / Prometheus integration
- Real-time dashboard / observability integration
- Stable extension API for custom triggers, validators, and policies

## Related

- [Overview](overview.md)
- [Architecture](architecture.md)
- [Design decisions](design-decisions.md)
- [Project README](../../README.md)
