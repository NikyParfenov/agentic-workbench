from ..schema.trace import ExecutionTrace
from ..schema.decisions import TriggerResult, Severity
from .base import BaseTrigger


class ToolErrorRateTrigger(BaseTrigger):
    def __init__(self, max_error_rate: float, min_results: int = 1, severity: Severity = "high"):
        if not 0 < max_error_rate <= 1:
            raise ValueError("max_error_rate must be in (0, 1]")
        if min_results < 1:
            raise ValueError("min_results must be >= 1")
        self.max_error_rate = max_error_rate
        self.min_results = min_results
        self.severity: Severity = severity

    def evaluate(self, trace: ExecutionTrace) -> TriggerResult:
        total = len(trace.tool_results)
        if total < self.min_results:
            return TriggerResult(
                triggered=False,
                trigger_name="ToolErrorRateTrigger",
                severity=self.severity,
                reason=f"Not enough tool results ({total}) to evaluate error rate",
                evidence={"total": total, "errors": 0, "error_rate": 0.0, "min_results": self.min_results},
            )

        errors = sum(1 for r in trace.tool_results if r.error is not None)
        rate = errors / total
        triggered = rate >= self.max_error_rate
        return TriggerResult(
            triggered=triggered,
            trigger_name="ToolErrorRateTrigger",
            severity=self.severity,
            reason=(
                f"Tool error rate {rate:.0%} ({errors}/{total}) reached limit {self.max_error_rate:.0%}"
                if triggered
                else f"Tool error rate {rate:.0%} ({errors}/{total}) within limit {self.max_error_rate:.0%}"
            ),
            evidence={
                "total": total,
                "errors": errors,
                "error_rate": rate,
                "max_error_rate": self.max_error_rate,
            },
        )
