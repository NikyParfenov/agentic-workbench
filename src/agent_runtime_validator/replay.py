"""Offline replay — re-validate a previously captured trace.

Typical workflow::

    # production: save the trace after each run
    from agent_runtime_validator import save_trace
    save_trace(trace, f"traces/{trace.run_id}.json")

    # CI / development: load and replay with an updated validator config
    from agent_runtime_validator import load_trace, replay, RuntimeValidator
    from agent_runtime_validator.triggers import SameToolLoopTrigger

    validator = RuntimeValidator(triggers=[SameToolLoopTrigger(max_repeats=2)])
    trace = load_trace("traces/run-123.json")
    decision = replay(trace, validator)
    print(decision.action, decision.reason)

Replay isolates the trace from runtime side effects so results are
reproducible:

- The input trace is deep-copied; replay never mutates it.
- Internal runtime state persisted in ``trace.metadata`` (validator budget,
  attempt counters — every ``_arv_*`` key, plus legacy pre-prefix names) is
  stripped from the copy, so an archived trace whose budget was consumed
  during the original run still gets a real validation on replay.
- If ``finished_at`` is missing, it is pinned to the latest event timestamp
  (falling back to ``started_at``), so time-based triggers measure the
  *observed trace duration* — the span from ``started_at`` to the last
  recorded event — instead of the wall-clock age of the archive. Idle time
  after the last event is not part of that span.

Replaying the same trace against the same validator config therefore yields
the same decision, no matter how often or when it is replayed.
"""

from datetime import datetime

from .schema.trace import ExecutionTrace
from .schema.decisions import ValidationDecision
from .runtime import RuntimeValidator, INTERNAL_STATE_PREFIX

# Key names used before the _arv_ prefix was introduced; stripped for traces
# archived by older versions.
_LEGACY_INTERNAL_KEYS = frozenset(
    {"_runtime_validator_call_count", "_trigger_score_attempts"}
)


def _latest_event_timestamp(trace: ExecutionTrace) -> datetime | None:
    timestamps = [
        event.timestamp
        for events in (
            trace.messages,
            trace.agent_calls,
            trace.tool_calls,
            trace.tool_results,
            trace.routing_events,
            trace.artifacts,
            trace.errors,
        )
        for event in events
    ]
    return max(timestamps) if timestamps else None


def _prepare_replay_trace(trace: ExecutionTrace) -> ExecutionTrace:
    clone = trace.model_copy(deep=True)
    clone.metadata = {
        key: value
        for key, value in clone.metadata.items()
        if not key.startswith(INTERNAL_STATE_PREFIX)
        and key not in _LEGACY_INTERNAL_KEYS
    }
    if clone.finished_at is None:
        clone.finished_at = _latest_event_timestamp(clone) or clone.started_at
    return clone


def replay(trace: ExecutionTrace, validator: RuntimeValidator) -> ValidationDecision:
    """Re-run *validator* against a captured *trace* and return the decision.

    The trace is deep-copied and normalized first (internal runtime counters
    stripped, ``finished_at`` pinned for time determinism), so the input is
    never mutated and repeated replays produce identical decisions.
    """
    return validator.validate(_prepare_replay_trace(trace))


async def replay_async(
    trace: ExecutionTrace, validator: RuntimeValidator
) -> ValidationDecision:
    """Async version of :func:`replay`. Same isolation guarantees."""
    return await validator.validate_async(_prepare_replay_trace(trace))
