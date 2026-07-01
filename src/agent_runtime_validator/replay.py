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
"""

from .schema.trace import ExecutionTrace
from .schema.decisions import ValidationDecision
from .runtime import RuntimeValidator


def replay(trace: ExecutionTrace, validator: RuntimeValidator) -> ValidationDecision:
    """Re-run *validator* against *trace* and return the decision.

    Equivalent to ``validator.validate(trace)`` but signals intent — the trace
    was captured earlier and is being replayed offline.
    """
    return validator.validate(trace)


async def replay_async(
    trace: ExecutionTrace, validator: RuntimeValidator
) -> ValidationDecision:
    """Async version of :func:`replay`."""
    return await validator.validate_async(trace)
