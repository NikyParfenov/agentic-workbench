from ..schema.trace import ExecutionTrace
from ..schema.decisions import TriggerResult, Severity
from .base import BaseTrigger


class MaxRoutesTrigger(BaseTrigger):
    def __init__(self, max_routes: int, severity: Severity = "high"):
        if max_routes < 1:
            raise ValueError("max_routes must be >= 1")
        self.max_routes = max_routes
        self.severity: Severity = severity

    def evaluate(self, trace: ExecutionTrace) -> TriggerResult:
        count = len(trace.routing_events)
        triggered = count >= self.max_routes
        return TriggerResult(
            triggered=triggered,
            trigger_name="MaxRoutesTrigger",
            severity=self.severity,
            reason=(
                f"Routing events ({count}) reached limit ({self.max_routes})"
                if triggered
                else f"Routing events ({count}) within limit ({self.max_routes})"
            ),
            evidence={"count": count, "max_routes": self.max_routes},
        )
