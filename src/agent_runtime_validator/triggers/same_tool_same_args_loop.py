from collections import Counter
from ..schema.trace import ExecutionTrace
from ..schema.decisions import TriggerResult, Severity
from ..utils.hashing import hash_args
from .base import BaseTrigger


class SameToolSameArgsLoopTrigger(BaseTrigger):
    def __init__(self, max_repeats: int, severity: Severity = "high"):
        self.max_repeats = max_repeats
        self.severity: Severity = severity

    def evaluate(self, trace: ExecutionTrace) -> TriggerResult:
        keys = [(call.tool_name, hash_args(call.args)) for call in trace.tool_calls]
        counts = Counter(keys)
        worst_key, worst_count = max(counts.items(), key=lambda x: x[1], default=(None, 0))
        triggered = worst_count >= self.max_repeats

        if worst_key is not None:
            tool_name, _ = worst_key
        else:
            tool_name = None

        return TriggerResult(
            triggered=triggered,
            trigger_name="SameToolSameArgsLoopTrigger",
            severity=self.severity,
            reason=(
                f"Tool '{tool_name}' called with identical args {worst_count} times (limit {self.max_repeats})"
                if triggered
                else f"No tool+args pair repeated {self.max_repeats}+ times"
            ),
            evidence={
                "tool": tool_name,
                "count": worst_count,
                "max_repeats": self.max_repeats,
            },
        )
