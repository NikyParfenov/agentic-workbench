from ..schema.trace import ExecutionTrace
from ..schema.decisions import TriggerResult, ValidatorResult, Recommendation
from .base import BaseValidator


class TriggerScoreValidator(BaseValidator):
    """Deterministic validator that aggregates fired triggers into a risk score."""

    _COUNTER_KEY = "_arv_trigger_score_attempts"

    def __init__(
        self,
        weights: dict[str, float],
        threshold: float,
        recommendation: Recommendation = "reroute",
        max_attempts: int = 1,
    ):
        self.weights = weights
        self.threshold = threshold
        self.recommendation: Recommendation = recommendation
        self.max_attempts = max_attempts

    def validate(
        self, trace: ExecutionTrace, trigger_results: list[TriggerResult]
    ) -> ValidatorResult:
        fired = [t for t in trigger_results if t.triggered]
        score = sum(self.weights.get(t.trigger_name, 0.0) for t in fired)

        if score < self.threshold:
            return ValidatorResult(
                valid=True,
                confidence=1.0,
                recommendation="continue",
                reason=f"Trigger score {score:.1f} below threshold {self.threshold:.1f}",
            )

        count = trace.metadata.get(self._COUNTER_KEY, 0)
        if count >= self.max_attempts:
            return ValidatorResult(
                valid=False,
                confidence=1.0,
                issues=["Maximum trigger-score attempts reached"],
                recommendation="interrupt",
                reason=f"Reroute/retry limit reached ({count}/{self.max_attempts})",
            )

        trace.metadata[self._COUNTER_KEY] = count + 1

        issues = [
            f"{t.trigger_name} (weight {self.weights.get(t.trigger_name, 0.0):.1f}): {t.reason}"
            for t in fired
            if t.trigger_name in self.weights
        ]

        return ValidatorResult(
            valid=False,
            confidence=1.0,
            issues=issues,
            recommendation=self.recommendation,
            reason=f"Trigger score {score:.1f} exceeded threshold {self.threshold:.1f}",
        )
