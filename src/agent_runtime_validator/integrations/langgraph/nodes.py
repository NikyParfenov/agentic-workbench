try:
    from langgraph.graph import END  # noqa: F401
except ImportError:
    raise ImportError(
        "LangGraph integration requires langgraph. "
        "Install it with: pip install agent-runtime-validator[langgraph]"
    ) from None

from typing import Any, Callable

from ...schema.trace import ExecutionTrace
from ...schema.decisions import ValidationDecision
from ...triggers.base import BaseTrigger
from ...validators.base import BaseValidator
from ...policies.base import BasePolicy
from ...runtime import RuntimeValidator, OnValidatorBudgetExhausted, ValidatorMode
from .adapter import state_to_trace

TraceBuilderFn = Callable[[dict[str, Any]], ExecutionTrace]
"""Type alias for a callable that builds an ``ExecutionTrace`` from LangGraph state.

Signature: ``(state: dict[str, Any]) -> ExecutionTrace``

Pass an instance as ``trace_builder`` to ``ValidationNode`` to replace the
default trace-resolution logic with your own.
"""


def create_validation_router(
    continue_to: str,
    retry_to: str | None = None,
    reroute_to: str | None = None,
    interrupt_to: str | None = None,
    abort_to: str | None = None,
    decision_key: str = "decision",
    allowed_reroutes: set[str] | None = None,
):
    """Create a conditional routing function for LangGraph based on ValidationDecision.

    Returns a callable suitable for ``builder.add_conditional_edges(node, router)``.
    Each action maps to a node name. ``continue_to`` is required; ``retry_to``
    and ``reroute_to`` default to ``continue_to``. ``interrupt_to`` and
    ``abort_to`` default to ``END`` — a stop decision must never silently
    continue, so map ``interrupt_to`` explicitly (e.g. to a human-review node)
    if the graph should keep running.

    ``allowed_reroutes`` is an explicit allowlist of node names that
    ``suggested_next_agent`` can resolve to. By default (``None``),
    ``suggested_next_agent`` is ignored and reroute always goes to
    ``reroute_to``. Pass a set to opt in to dynamic rerouting.
    """
    abort_target = abort_to if abort_to is not None else END
    interrupt_target = interrupt_to if interrupt_to is not None else END

    def router(state: dict) -> str:
        raw = state.get(decision_key)
        if raw is None:
            return continue_to
        if isinstance(raw, dict):
            decision = ValidationDecision(**raw)
        else:
            decision = raw

        match decision.action:
            case "continue":
                return continue_to
            case "retry_last_step":
                return retry_to if retry_to is not None else continue_to
            case "reroute":
                if (
                    allowed_reroutes is not None
                    and decision.validator_result
                    and decision.validator_result.suggested_next_agent
                    and decision.validator_result.suggested_next_agent in allowed_reroutes
                ):
                    return decision.validator_result.suggested_next_agent
                return reroute_to if reroute_to is not None else continue_to
            case "interrupt":
                return interrupt_target
            case "abort":
                return abort_target
            case _:
                return continue_to

    return router


class ValidationNode:
    def __init__(
        self,
        triggers: list[BaseTrigger],
        validator: BaseValidator | None = None,
        policy: BasePolicy | None = None,
        trace_key: str = "trace",
        decision_key: str = "decision",
        max_validator_calls_per_run: int | None = None,
        on_validator_budget_exhausted: OnValidatorBudgetExhausted = "skip",
        validator_mode: ValidatorMode = "on_trigger",
        trace_builder: TraceBuilderFn | None = None,
    ):
        self._runtime = RuntimeValidator(
            triggers=triggers,
            validator=validator,
            policy=policy,
            max_validator_calls_per_run=max_validator_calls_per_run,
            on_validator_budget_exhausted=on_validator_budget_exhausted,
            validator_mode=validator_mode,
        )
        self.trace_key = trace_key
        self.decision_key = decision_key
        self._trace_builder = trace_builder

    def _resolve_trace(self, state: dict[str, Any]) -> ExecutionTrace:
        if self._trace_builder is not None:
            return self._trace_builder(state)
        raw = state.get(self.trace_key)
        if raw is None:
            return state_to_trace(state)
        if isinstance(raw, dict):
            return ExecutionTrace(**raw)
        if isinstance(raw, ExecutionTrace):
            return raw
        return state_to_trace(state)

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        trace = self._resolve_trace(state)
        decision = self._runtime.validate(trace)
        return {**state, self.trace_key: trace, self.decision_key: decision}

    async def async_call(self, state: dict[str, Any]) -> dict[str, Any]:
        trace = self._resolve_trace(state)
        decision = await self._runtime.validate_async(trace)
        return {**state, self.trace_key: trace, self.decision_key: decision}
