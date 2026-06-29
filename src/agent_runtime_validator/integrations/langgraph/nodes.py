try:
    from langgraph.graph import END  # noqa: F401
except ImportError:
    raise ImportError(
        "LangGraph integration requires langgraph. "
        "Install it with: pip install agent-runtime-validator[langgraph]"
    ) from None

from ...schema.trace import ExecutionTrace
from ...triggers.base import BaseTrigger
from ...validators.base import BaseValidator
from ...policies.base import BasePolicy
from ...runtime import RuntimeValidator
from .adapter import state_to_trace


class ValidationNode:
    def __init__(
        self,
        triggers: list[BaseTrigger],
        validator: BaseValidator | None = None,
        policy: BasePolicy | None = None,
        trace_key: str = "trace",
        decision_key: str = "decision",
    ):
        self._runtime = RuntimeValidator(triggers=triggers, validator=validator, policy=policy)
        self.trace_key = trace_key
        self.decision_key = decision_key

    def __call__(self, state: dict) -> dict:
        trace = state.get(self.trace_key)
        if trace is None:
            trace = state_to_trace(state)
        elif isinstance(trace, dict):
            trace = ExecutionTrace(**trace)
        decision = self._runtime.validate(trace)
        return {**state, self.decision_key: decision}

    async def async_call(self, state: dict) -> dict:
        trace = state.get(self.trace_key)
        if trace is None:
            trace = state_to_trace(state)
        elif isinstance(trace, dict):
            trace = ExecutionTrace(**trace)
        decision = await self._runtime.validate_async(trace)
        return {**state, self.decision_key: decision}
