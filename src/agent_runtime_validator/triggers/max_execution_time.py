from datetime import datetime, timezone
from ..schema.trace import ExecutionTrace
from ..schema.decisions import TriggerResult, Severity
from .base import BaseTrigger


class MaxExecutionTimeTrigger(BaseTrigger):
    def __init__(self, max_seconds: float, severity: Severity = "high"):
        self.max_seconds = max_seconds
        self.severity: Severity = severity

    def evaluate(self, trace: ExecutionTrace) -> TriggerResult:
        end = trace.finished_at or datetime.now(timezone.utc)
        elapsed = (end - trace.started_at).total_seconds()
        triggered = elapsed >= self.max_seconds
        return TriggerResult(
            triggered=triggered,
            trigger_name="MaxExecutionTimeTrigger",
            severity=self.severity,
            reason=(
                f"Execution time ({elapsed:.1f}s) reached limit ({self.max_seconds}s)"
                if triggered
                else f"Execution time ({elapsed:.1f}s) within limit ({self.max_seconds}s)"
            ),
            evidence={"elapsed_seconds": elapsed, "max_seconds": self.max_seconds},
        )
