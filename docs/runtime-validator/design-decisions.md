# Runtime Validator — Design Decisions

## Purpose

Record the key design choices behind the runtime validator and the reasoning
that justifies them, so future changes can be weighed against the original
intent. Entries are append-only; do not rewrite history.

## 2026-06-28 — Triggers are deterministic; LLM logic lives in validators

**Decision:** Triggers must be pure, deterministic checks over the trace. Any
LLM or I/O-based reasoning belongs in a validator.

**Reason:** Triggers run on every validation call. Keeping them cheap and
predictable makes the common "healthy run" path fast and testable.

**Consequences:** Behavioral judgment (e.g. "is this argument sensible?") cannot
live in a trigger; it must be expressed as a validator that runs after a trigger
fires.

## 2026-06-28 — Validators run only when a trigger fires

**Decision:** The validator stage executes only if at least one trigger fired
and the configured validator is not `NoOpValidator`.

**Reason:** Deep checks — especially LLM judges — are expensive. There is no
value in running them when nothing looks wrong.

**Consequences:** A validator never sees a fully healthy run. Checks that must
run unconditionally should be modeled as a trigger plus validator pair, or as a
trigger alone.

## 2026-06-28 — The pipeline does not short-circuit

**Decision:** All triggers evaluate on every call, even after one fires.

**Reason:** The policy needs the complete picture. A run can hit a loop and a
timeout at once; surfacing both yields better decisions and diagnostics.

**Consequences:** Trigger cost is additive. With many expensive custom triggers
this matters, which reinforces the "triggers stay cheap" decision above.

## 2026-06-28 — Explicit sync/async split

**Decision:** `validate` is synchronous and raises `RuntimeError` if a validator
returns an awaitable; `validate_async` awaits validators.

**Reason:** Silently running an event loop inside a sync call is surprising and
error-prone. An explicit error points the caller to the right method.

**Consequences:** Callers using async validators must choose `validate_async`
deliberately. Triggers remain synchronous in both paths.

## 2026-06-28 — Validator recommendation overrides policy default

**Decision:** When a validator returns a `recommendation`, the policy uses it
instead of the severity-to-action mapping.

**Reason:** A validator has looked deeper than the triggers; its judgment should
win over the coarse severity heuristic.

**Consequences:** A misbehaving validator can steer the final action. Validators
should return conservative recommendations and set `confidence` honestly.

## 2026-06-28 — Trace is a framework-neutral model

**Decision:** All inputs are normalized into a single `ExecutionTrace` Pydantic
model; framework specifics live in adapters (e.g. LangGraph `state_to_trace`).

**Reason:** Triggers, validators, and policies should not depend on any one agent
framework, so new integrations are additive.

**Consequences:** Each new framework needs an adapter that maps its state onto
`ExecutionTrace`. The trace schema becomes a stable contract to evolve carefully.

## 2026-06-28 — Policies are separate from validation

**Decision:** Validators produce recommendations; policies decide actions.

**Reason:** Different applications have different operational requirements. One
system may interrupt on loops, another may retry automatically.

**Consequences:** Validation logic remains reusable across projects.

## 2026-06-29 — Not every validator needs an LLM

**Decision:** `TriggerScoreValidator` maps fired triggers to a recommendation
purely through weighted scoring, without calling a model.

**Reason:** For many failure patterns (no progress, ping-pong routing, no tool
usage) the triggers already carry enough signal. An LLM call adds latency, cost,
and a parse-failure mode with no added insight. A deterministic scorer is faster,
cheaper, and fully testable.

**Consequences:** The validator interface stays the same (`BaseValidator`), so
`TriggerScoreValidator` and `LLMJudgeValidator` are interchangeable. A
`max_attempts` guard prevents infinite retry/reroute loops by tracking counts
in `trace.metadata`.

## 2026-06-30 — Validator calls can be budgeted per run

**Decision:** `RuntimeValidator` exposes `max_validator_calls_per_run` to limit
how many times the optional validator can be invoked for the same trace/run.
When the budget is exhausted, the default behavior is
`on_validator_budget_exhausted="skip"`: do not call the validator and let fired
triggers plus policy decide.

**Reason:** LLM judges and other deep validators may be expensive. In a
retry/reroute loop, repeatedly calling the same validator can burn latency and
cost without adding useful signal. Skipping the optional validator after the
budget is exhausted preserves deterministic trigger handling without forcing an
automatic interrupt or abort.

**Consequences:** Users can set `max_validator_calls_per_run=1` for production
LLM judge usage. If they want fail-closed behavior when the budget is exhausted,
they can set `on_validator_budget_exhausted` to `"interrupt"` or `"abort"`.
Budget state lives in `trace.metadata`, so integrations that parse serialized
traces must write the updated trace back into state.

## 2026-06-30 — Policy blocks validator downgrades by default

**Decision:** Validator recommendations can escalate the severity-derived action
by default. Downgrades require explicit opt-in with
`allow_validator_downgrade=True` and sufficient validator confidence. Critical
trigger severity cannot be downgraded.

**Reason:** A validator may have deeper context, but deterministic trigger
evidence should not be silently weakened by a low-confidence or overly optimistic
validator result.

**Consequences:** Users who want validator recommendations to soften trigger
responses must opt in deliberately. When a downgrade is rejected, the decision
reason explains that policy kept the safer action due to severity/confidence
safeguards.

## Related

- [Overview](overview.md)
- [Architecture](architecture.md)
- [Triggers](triggers.md)
- [Validators](validators.md)
- [Roadmap](roadmap.md)
