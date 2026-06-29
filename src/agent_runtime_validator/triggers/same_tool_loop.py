from collections import Counter
from ..schema.trace import ExecutionTrace
from ..schema.decisions import TriggerResult, Severity
from .base import BaseTrigger


class SameToolLoopTrigger(BaseTrigger):
    def __init__(self, max_repeats: int, severity: Severity = "medium"):
        self.max_repeats = max_repeats
        self.severity: Severity = severity

    def evaluate(self, trace: ExecutionTrace) -> TriggerResult:
        counts = Counter(call.tool_name for call in trace.tool_calls)
        worst_tool, worst_count = max(counts.items(), key=lambda x: x[1], default=(None, 0))
        triggered = worst_count >= self.max_repeats
        return TriggerResult(
            triggered=triggered,
            trigger_name="SameToolLoopTrigger",
            severity=self.severity,
            reason=(
                f"Tool '{worst_tool}' called {worst_count} times (limit {self.max_repeats})"
                if triggered
                else f"No tool called {self.max_repeats}+ times"
            ),
            evidence={
                "tool": worst_tool,
                "count": worst_count,
                "max_repeats": self.max_repeats,
            },
        )
