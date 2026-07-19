from ..schema.trace import ExecutionTrace
from ..schema.decisions import (
    TriggerResult, ValidatorResult, ValidationDecision, Severity, Action,
)
from .base import BasePolicy

_SEVERITY_ORDER: dict[Severity, int] = {"low": 0, "medium": 1, "high": 2, "critical": 3}

# Severity implied by an action, used when no trigger fired and the validator
# alone drove the decision — so an abort never ships with severity="low".
_ACTION_SEVERITY: dict[Action, Severity] = {
    "continue": "low",
    "retry_last_step": "medium",
    "reroute": "medium",
    "interrupt": "high",
    "abort": "critical",
}

# Per-run retry counter, persisted in trace.metadata under the _arv_ prefix so
# it survives checkpoints during the run and is stripped by replay().
_RETRY_COUNT_KEY = "_arv_policy_retry_count"


class DefaultPolicy(BasePolicy):
    def __init__(
        self,
        retry_on_medium: bool = True,
        interrupt_on_high: bool = True,
        abort_on_critical: bool = True,
        allow_validator_downgrade: bool = False,
        min_confidence_for_override: float = 0.7,
        max_retries_per_run: int | None = 3,
    ):
        if max_retries_per_run is not None and max_retries_per_run < 0:
            raise ValueError("max_retries_per_run must be >= 0 or None")
        self.retry_on_medium = retry_on_medium
        self.interrupt_on_high = interrupt_on_high
        self.abort_on_critical = abort_on_critical
        self.allow_validator_downgrade = allow_validator_downgrade
        self.min_confidence_for_override = min_confidence_for_override
        self.max_retries_per_run = max_retries_per_run

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

    def _apply_retry_budget(
        self, trace: ExecutionTrace, action: Action
    ) -> tuple[Action, str | None]:
        """Bound retry_last_step decisions per run.

        Triggers evaluate cumulatively over a trace that never shrinks, so a
        fired trigger keeps firing at every subsequent checkpoint. Without a
        bound, a medium-severity trigger plus ``retry_on_medium=True`` retries
        forever. Returns the (possibly escalated) action and, when escalated,
        the budget explanation to prepend to the reason.
        """
        if action != "retry_last_step" or self.max_retries_per_run is None:
            return action, None
        count = trace.metadata.get(_RETRY_COUNT_KEY, 0)
        if count >= self.max_retries_per_run:
            return "interrupt", (
                f"Retry budget exhausted ({count}/{self.max_retries_per_run} "
                f"retries used this run); escalating to interrupt"
            )
        trace.metadata[_RETRY_COUNT_KEY] = count + 1
        return action, None

    def decide(
        self,
        trace: ExecutionTrace,
        triggered: list[TriggerResult],
        validator_result: ValidatorResult | None,
    ) -> ValidationDecision:
        fired = [t for t in triggered if t.triggered]
        triggered_by = [t.trigger_name for t in fired]

        if not fired:
            # When the validator runs without trigger signal (always mode),
            # honor escalations — a validator recommending stop should not be
            # silently ignored just because triggers were quiet.
            if validator_result is not None and validator_result.recommendation != "continue":
                action = validator_result.recommendation
                action, budget_reason = self._apply_retry_budget(trace, action)
                reason = (
                    f"{budget_reason}. Validator reason: {validator_result.reason}"
                    if budget_reason else validator_result.reason
                )
                return ValidationDecision(
                    should_continue=False,
                    action=action,
                    severity=_ACTION_SEVERITY[action],
                    reason=reason,
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
        action, budget_reason = self._apply_retry_budget(trace, action)

        if budget_reason is not None:
            reason = f"{budget_reason}. Trigger(s) fired: {', '.join(triggered_by)}"
        elif validator_result is None:
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
