from ..schema.trace import ExecutionTrace
from ..schema.decisions import TriggerResult, Severity
from .base import BaseTrigger


class MaxAgentCallsTrigger(BaseTrigger):
    """Fires when the total number of agent-to-agent delegations reaches a limit.

    Use in supervisor patterns where excessive sub-agent invocations indicate
    delegation explosion or unbounded recursion.
    """

    def __init__(self, max_calls: int, severity: Severity = "high"):
        if max_calls < 1:
            raise ValueError("max_calls must be >= 1")
        self.max_calls = max_calls
        self.severity: Severity = severity

    def evaluate(self, trace: ExecutionTrace) -> TriggerResult:
        count = len(trace.agent_calls)
        triggered = count >= self.max_calls
        return TriggerResult(
            triggered=triggered,
            trigger_name="MaxAgentCallsTrigger",
            severity=self.severity,
            reason=(
                f"Agent calls ({count}) reached limit ({self.max_calls})"
                if triggered
                else f"Agent calls ({count}) within limit ({self.max_calls})"
            ),
            evidence={"count": count, "max_calls": self.max_calls},
        )
