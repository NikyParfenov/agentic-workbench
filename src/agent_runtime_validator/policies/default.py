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
        allow_validator_downgrade: bool = False,
        min_confidence_for_override: float = 0.7,
    ):
        self.retry_on_medium = retry_on_medium
        self.interrupt_on_high = interrupt_on_high
        self.abort_on_critical = abort_on_critical
        self.allow_validator_downgrade = allow_validator_downgrade
        self.min_confidence_for_override = min_confidence_for_override

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

    def _resolve_action(
        self, severity: Severity, validator_result: ValidatorResult | None,
    ) -> Action:
        severity_action = self._action_for_severity(severity)
        if validator_result is None:
            return severity_action

        rec = validator_result.recommendation
        severity_rank = _SEVERITY_ORDER.get(severity, 0)
        sev_action_rank = ["continue", "retry_last_step", "reroute", "interrupt", "abort"]
        rec_idx = sev_action_rank.index(rec) if rec in sev_action_rank else 0
        sev_idx = sev_action_rank.index(severity_action) if severity_action in sev_action_rank else 0

        is_escalation = rec_idx > sev_idx
        if is_escalation:
            return rec

        if severity_rank >= _SEVERITY_ORDER["critical"]:
            return severity_action

        if not self.allow_validator_downgrade:
            return severity_action
        if validator_result.confidence < self.min_confidence_for_override:
            return severity_action
        return rec

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
            # When the validator runs without trigger signal (always mode),
            # honor escalations — a validator recommending stop should not be
            # silently ignored just because triggers were quiet.
            if validator_result is not None and validator_result.recommendation != "continue":
                return ValidationDecision(
                    should_continue=False,
                    action=validator_result.recommendation,
                    severity="low",
                    reason=validator_result.reason,
                    triggered_by=[],
                    validator_result=validator_result,
                )
            return ValidationDecision(
                should_continue=True,
                action="continue",
                severity="low",
                reason="No triggers fired",
                triggered_by=[],
                validator_result=validator_result,
            )

        severity = self._highest_severity(fired)
        action = self._resolve_action(severity, validator_result)

        if validator_result is None:
            reason = f"Trigger(s) fired: {', '.join(triggered_by)}"
        elif action == validator_result.recommendation:
            reason = validator_result.reason
        else:
            reason = (
                f"Validator recommended {validator_result.recommendation!r}, "
                f"but policy kept {action!r} due to severity/confidence safeguards. "
                f"Validator reason: {validator_result.reason}"
            )

        return ValidationDecision(
            should_continue=(action == "continue"),
            action=action,
            severity=severity,
            reason=reason,
            triggered_by=triggered_by,
            validator_result=validator_result,
        )
