from ..schema.trace import ExecutionTrace
from ..schema.decisions import (
    TriggerResult, ValidatorResult, ValidationDecision, Severity, Action,
)
from .base import BasePolicy

_SEVERITY_ORDER: dict[Severity, int] = {"low": 0, "medium": 1, "high": 2, "critical": 3}


class DefaultPolicy(BasePolicy):
    def __init__(
        self,
        retry_on_medium: bool = True,
        interrupt_on_high: bool = True,
        abort_on_critical: bool = True,
    ):
        self.retry_on_medium = retry_on_medium
        self.interrupt_on_high = interrupt_on_high
        self.abort_on_critical = abort_on_critical

    def _highest_severity(self, fired: list[TriggerResult]) -> Severity:
        if not fired:
            return "low"
        return max(fired, key=lambda t: _SEVERITY_ORDER[t.severity]).severity

    def _action_for_severity(self, severity: Severity) -> Action:
        if severity == "low":
            return "continue"
        if severity == "medium":
            return "retry_last_step" if self.retry_on_medium else "continue"
        if severity == "high":
            return "interrupt" if self.interrupt_on_high else "continue"
        return "abort" if self.abort_on_critical else "interrupt"

    def decide(
        self,
        trace: ExecutionTrace,
        triggered: list[TriggerResult],
        validator_result: ValidatorResult | None,
    ) -> ValidationDecision:
        _ = trace
        fired = [t for t in triggered if t.triggered]
        triggered_by = [t.trigger_name for t in fired]

        if not fired:
            return ValidationDecision(
                should_continue=True,
                action="continue",
                severity="low",
                reason="No triggers fired",
                triggered_by=[],
                validator_result=validator_result,
            )

        severity = self._highest_severity(fired)
        action: Action = (
            validator_result.recommendation
            if validator_result is not None
            else self._action_for_severity(severity)
        )
        reason = (
            validator_result.reason
            if validator_result is not None
            else f"Trigger(s) fired: {', '.join(triggered_by)}"
        )

        return ValidationDecision(
            should_continue=(action == "continue"),
            action=action,
            severity=severity,
            reason=reason,
            triggered_by=triggered_by,
            validator_result=validator_result,
        )
