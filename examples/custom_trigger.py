"""Define and use a custom trigger.

Subclass `BaseTrigger`, implement `evaluate`, and pass it to `RuntimeValidator`
like any built-in trigger. Keep custom triggers deterministic — put LLM logic in
a validator instead.

Run:
    uv run python examples/custom_trigger.py
"""
from datetime import datetime, timezone

from agent_runtime_validator import (
    BaseTrigger,
    ExecutionTrace,
    RuntimeValidator,
    TriggerResult,
)
from agent_runtime_validator.schema.decisions import Severity
from agent_runtime_validator.schema.events import ErrorEvent


def now() -> datetime:
    return datetime.now(timezone.utc)


class TooManyErrorsTrigger(BaseTrigger):
    """Fires when the trace accumulates more than `max_errors` errors."""

    def __init__(self, max_errors: int, severity: Severity = "high"):
        self.max_errors = max_errors
        # Annotate explicitly so the Literal type is not widened to `str`.
        self.severity: Severity = severity

    def evaluate(self, trace: ExecutionTrace) -> TriggerResult:
        count = len(trace.errors)
        triggered = count > self.max_errors
        return TriggerResult(
            triggered=triggered,
            trigger_name="TooManyErrorsTrigger",
            severity=self.severity,
            reason=f"{count} error(s), limit {self.max_errors}",
            evidence={"count": count, "max_errors": self.max_errors},
        )


def build_trace() -> ExecutionTrace:
    trace = ExecutionTrace(run_id="custom-trigger-demo", started_at=now())
    for i in range(3):
        trace.errors.append(
            ErrorEvent(error_type="ToolError", message=f"boom {i}", timestamp=now())
        )
    return trace


def main() -> None:
    validator = RuntimeValidator(triggers=[TooManyErrorsTrigger(max_errors=2)])

    decision = validator.validate(build_trace())

    print(f"action:       {decision.action}")
    print(f"severity:     {decision.severity}")
    print(f"triggered_by: {decision.triggered_by}")
    print(f"reason:       {decision.reason}")


if __name__ == "__main__":
    main()
