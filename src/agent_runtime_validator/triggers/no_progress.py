from ..schema.trace import ExecutionTrace
from ..schema.decisions import TriggerResult, Severity
from .base import BaseTrigger


class NoProgressTrigger(BaseTrigger):
    """Fires when many tool calls have occurred but no artifacts were produced."""

    def __init__(self, min_tool_calls: int = 5, severity: Severity = "medium"):
        if min_tool_calls < 1:
            raise ValueError("min_tool_calls must be >= 1")
        self.min_tool_calls = min_tool_calls
        self.severity: Severity = severity

    def evaluate(self, trace: ExecutionTrace) -> TriggerResult:
        call_count = len(trace.tool_calls)
        artifact_count = len(trace.artifacts)
        triggered = call_count >= self.min_tool_calls and artifact_count == 0
        return TriggerResult(
            triggered=triggered,
            trigger_name="NoProgressTrigger",
            severity=self.severity,
            reason=(
                f"No artifacts produced after {call_count} tool calls"
                if triggered
                else f"Progress detected: {artifact_count} artifact(s) or only {call_count} tool call(s)"
            ),
            evidence={
                "tool_call_count": call_count,
                "artifact_count": artifact_count,
                "min_tool_calls": self.min_tool_calls,
            },
        )
