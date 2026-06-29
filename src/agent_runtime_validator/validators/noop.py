from ..schema.trace import ExecutionTrace
from ..schema.decisions import TriggerResult, ValidatorResult
from .base import BaseValidator


class NoOpValidator(BaseValidator):
    def validate(  # noqa: ARG002
        self, trace: ExecutionTrace, trigger_results: list[TriggerResult]
    ) -> ValidatorResult:
        _ = trace, trigger_results
        return ValidatorResult(
            valid=True,
            confidence=1.0,
            issues=[],
            recommendation="continue",
            reason="NoOpValidator: no validation performed",
        )
