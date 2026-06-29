from ..schema.trace import ExecutionTrace
from ..schema.decisions import TriggerResult, Severity
from .base import BaseTrigger


class MaxToolCallsTrigger(BaseTrigger):
    def __init__(self, max_calls: int, severity: Severity = "high"):
        self.max_calls = max_calls
        self.severity: Severity = severity

    def evaluate(self, trace: ExecutionTrace) -> TriggerResult:
        count = len(trace.tool_calls)
        triggered = count >= self.max_calls
        return TriggerResult(
            triggered=triggered,
            trigger_name="MaxToolCallsTrigger",
            severity=self.severity,
            reason=(
                f"Tool calls ({count}) reached limit ({self.max_calls})"
                if triggered
                else f"Tool calls ({count}) within limit ({self.max_calls})"
            ),
            evidence={"count": count, "max_calls": self.max_calls},
        )
