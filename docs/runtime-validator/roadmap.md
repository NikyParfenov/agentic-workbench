# Runtime Validator — Roadmap

## Purpose

Outline what exists today and what is planned, so users can see where the module
is headed and which integrations to expect. Items beyond v0.1 are aspirational
and may change.

## v0.1 — current

| Area | Included |
|------|----------|
| Triggers | Max calls, tokens, time, routes; same-tool loop; same-tool-same-args loop; agent ping-pong; no progress; tool error rate; no tool usage |
| Validators | NoOp, JSON Schema, tool argument, LLM judge, trigger score |
| Policy | `DefaultPolicy` with severity-based action mapping |
| Runtime | Sync and async `RuntimeValidator` |
| Integrations | LangGraph (`ValidationNode`, `state_to_trace`) |

## v0.2 — offline and import

- Offline replay: validate completed traces from logs
- Trace import from LangSmith, LangFuse, Arize Phoenix
- Artifact content validation (schema checks on produced artifacts)
- Trace export/serialization for debugging
- Pre-execution tool argument validation hooks

## v0.3 — more frameworks

- Integrations: CrewAI, LlamaIndex, OpenAI Agents SDK, PydanticAI
- Agent-scoped trigger filtering (`agent_names` parameter for tool-oriented triggers)

## v0.4 — composition and recovery

- `CompositeValidator` — chain validators with configurable aggregation
- `ExecutionInvariantValidator` — user-defined invariants over trace state
- Configurable trigger composition (AND/OR logic between triggers)
- Cost-tracking trigger (dollar budget, not just tokens)
- Webhook/callback on decision

## v1.0 — scale

- Decentralized multi-agent support
- Graph cycle and deadlock detection
- Distributed trace support (spans across services)
- Real-time dashboard / observability integration
- Stable extension API for custom triggers, validators, and policies

## Related

- [Overview](overview.md)
- [Architecture](architecture.md)
- [Design decisions](design-decisions.md)
- [Project README](../../README.md)
