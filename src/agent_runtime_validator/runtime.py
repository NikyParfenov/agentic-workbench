import logging
from inspect import isawaitable
from typing import Literal, get_args

from .schema.trace import ExecutionTrace
from .schema.decisions import ValidationDecision, ValidatorResult, Recommendation
from .triggers.base import BaseTrigger
from .validators.base import BaseValidator
from .validators.noop import NoOpValidator
from .policies.base import BasePolicy
from .policies.default import DefaultPolicy

logger = logging.getLogger("agent_runtime_validator")

_BUDGET_KEY = "_runtime_validator_call_count"

OnValidatorBudgetExhausted = Literal[
    "skip",
    "continue",
    "retry_last_step",
    "reroute",
    "interrupt",
    "abort",
]

_VALID_ON_EXHAUSTED = frozenset(get_args(OnValidatorBudgetExhausted))

OnValidatorError = Literal[
    "skip",
    "continue",
    "retry_last_step",
    "reroute",
    "interrupt",
    "abort",
]
"""What to do when the validator itself raises (network timeout, rate limit,
provider outage, or a bug in a custom validator).

``"skip"`` (default): treat the validator as unavailable — pass
``validator_result=None`` to the policy and let fired triggers decide. Any
other value produces a synthetic ``ValidatorResult`` with that recommendation.

This is separate from malformed *output* handling: ``LLMJudgeValidator`` keeps
its own ``max_retries``/``fallback_recommendation`` path for responses that
arrive but cannot be parsed. ``on_validator_error`` covers the call not
completing at all. The raw exception is logged, never propagated to the host.
"""

_VALID_ON_ERROR = frozenset(get_args(OnValidatorError))

ValidatorMode = Literal["on_trigger", "always"]
"""Controls when the validator is invoked.

``"on_trigger"`` (default): validator only runs when at least one trigger fires.
Use this for inline mid-run monitoring — the common "all-clear" path never calls
the validator.

``"always"``: validator always runs, regardless of trigger results.
Use this when the validator is a post-run quality check that should inspect
every completed trace, even clean ones.
"""

_VALID_VALIDATOR_MODES = frozenset(get_args(ValidatorMode))


class RuntimeValidator:
    def __init__(
        self,
        triggers: list[BaseTrigger],
        validator: BaseValidator | None = None,
        policy: BasePolicy | None = None,
        max_validator_calls_per_run: int | None = None,
        on_validator_budget_exhausted: OnValidatorBudgetExhausted = "skip",
        validator_mode: ValidatorMode = "on_trigger",
        on_validator_error: OnValidatorError = "skip",
    ) -> None:
        if max_validator_calls_per_run is not None and max_validator_calls_per_run < 0:
            raise ValueError("max_validator_calls_per_run must be >= 0")
        if on_validator_budget_exhausted not in _VALID_ON_EXHAUSTED:
            raise ValueError(
                "on_validator_budget_exhausted must be one of "
                f"{sorted(_VALID_ON_EXHAUSTED)}"
            )
        if validator_mode not in _VALID_VALIDATOR_MODES:
            raise ValueError(
                f"validator_mode must be one of {sorted(_VALID_VALIDATOR_MODES)}"
            )
        if on_validator_error not in _VALID_ON_ERROR:
            raise ValueError(
                f"on_validator_error must be one of {sorted(_VALID_ON_ERROR)}"
            )
        self.triggers = triggers
        self.validator = validator or NoOpValidator()
        self.policy = policy or DefaultPolicy()
        self.max_validator_calls_per_run = max_validator_calls_per_run
        self.on_validator_budget_exhausted: OnValidatorBudgetExhausted = on_validator_budget_exhausted
        self.validator_mode: ValidatorMode = validator_mode
        self.on_validator_error: OnValidatorError = on_validator_error

    def _evaluate_triggers(self, trace: ExecutionTrace):
        logger.debug("run=%s evaluating %d trigger(s)", trace.run_id, len(self.triggers))
        trigger_results = [t.evaluate(trace) for t in self.triggers]
        for tr in trigger_results:
            logger.debug(
                "run=%s trigger=%s triggered=%s severity=%s reason=%s",
                trace.run_id, tr.trigger_name, tr.triggered, tr.severity, tr.reason,
            )
        fired = [r for r in trigger_results if r.triggered]
        if fired:
            logger.info(
                "run=%s %d trigger(s) fired: %s",
                trace.run_id, len(fired), ", ".join(r.trigger_name for r in fired),
            )
        else:
            logger.debug("run=%s no triggers fired", trace.run_id)
        return trigger_results, fired

    def _log_decision(self, trace: ExecutionTrace, decision: ValidationDecision):
        if decision.action != "continue":
            logger.warning(
                "run=%s action=%s severity=%s triggered_by=%s reason=%s",
                trace.run_id, decision.action, decision.severity,
                decision.triggered_by, decision.reason,
            )
        else:
            logger.debug("run=%s action=continue", trace.run_id)

    def _budget_remaining(self, trace: ExecutionTrace) -> bool:
        if self.max_validator_calls_per_run is None:
            return True
        return trace.metadata.get(_BUDGET_KEY, 0) < self.max_validator_calls_per_run

    def _increment_budget(self, trace: ExecutionTrace) -> None:
        trace.metadata[_BUDGET_KEY] = trace.metadata.get(_BUDGET_KEY, 0) + 1

    def _exhausted_validator_result(self) -> ValidatorResult | None:
        exhausted = self.on_validator_budget_exhausted
        if exhausted == "skip":
            return None
        rec: Recommendation = exhausted
        return ValidatorResult(
            valid=rec == "continue",
            confidence=1.0,
            recommendation=rec,
            reason=f"Validator budget exhausted (max_validator_calls_per_run={self.max_validator_calls_per_run})",
            issues=["Validator budget exhausted"],
        )

    def _error_validator_result(self, exc: Exception) -> ValidatorResult | None:
        """Synthetic result for a validator that raised, per ``on_validator_error``.

        Only the exception *type* is surfaced in the reason — the full message
        and traceback go to the log, so provider error details never leak into
        decisions or downstream prompts.
        """
        behavior = self.on_validator_error
        if behavior == "skip":
            return None
        rec: Recommendation = behavior
        return ValidatorResult(
            valid=rec == "continue",
            confidence=1.0,
            recommendation=rec,
            reason=(
                f"Validator raised {type(exc).__name__}; "
                f"applying on_validator_error={behavior!r}"
            ),
            issues=["Validator infrastructure error"],
        )

    def _should_invoke_validator(self, fired: list) -> bool:
        if isinstance(self.validator, NoOpValidator):
            return False
        if self.validator_mode == "always":
            return True
        return bool(fired)

    def validate(self, trace: ExecutionTrace) -> ValidationDecision:
        trigger_results, fired = self._evaluate_triggers(trace)

        validator_result: ValidatorResult | None = None
        if self._should_invoke_validator(fired):
            if not self._budget_remaining(trace):
                logger.info(
                    "run=%s validator budget exhausted (max=%s on_exhausted=%s), skipping validator",
                    trace.run_id, self.max_validator_calls_per_run,
                    self.on_validator_budget_exhausted,
                )
                validator_result = self._exhausted_validator_result()
            else:
                validator_cls = type(self.validator).__name__
                logger.info("run=%s invoking validator=%s", trace.run_id, validator_cls)
                self._increment_budget(trace)
                try:
                    result = self.validator.validate(trace, trigger_results)
                except Exception as exc:
                    logger.exception(
                        "run=%s validator=%s raised %s; applying on_validator_error=%r",
                        trace.run_id, validator_cls, type(exc).__name__,
                        self.on_validator_error,
                    )
                    validator_result = self._error_validator_result(exc)
                else:
                    if isawaitable(result):
                        close = getattr(result, "close", None)
                        if callable(close):
                            close()
                        raise RuntimeError(
                            "Validator returned an awaitable from sync validate(). "
                            "Use validate_async() instead."
                        )
                    validator_result = result
                    logger.debug(
                        "run=%s validator result: valid=%s confidence=%.2f recommendation=%s issues=%d",
                        trace.run_id, validator_result.valid, validator_result.confidence,
                        validator_result.recommendation, len(validator_result.issues),
                    )

        decision = self.policy.decide(trace, trigger_results, validator_result)
        self._log_decision(trace, decision)
        return decision

    async def validate_async(self, trace: ExecutionTrace) -> ValidationDecision:
        trigger_results, fired = self._evaluate_triggers(trace)

        validator_result: ValidatorResult | None = None
        if self._should_invoke_validator(fired):
            if not self._budget_remaining(trace):
                logger.info(
                    "run=%s validator budget exhausted (max=%s on_exhausted=%s), skipping validator",
                    trace.run_id, self.max_validator_calls_per_run,
                    self.on_validator_budget_exhausted,
                )
                validator_result = self._exhausted_validator_result()
            else:
                validator_cls = type(self.validator).__name__
                logger.info("run=%s invoking validator=%s", trace.run_id, validator_cls)
                self._increment_budget(trace)
                try:
                    validator_result = await self.validator.validate_async(trace, trigger_results)
                except Exception as exc:
                    logger.exception(
                        "run=%s validator=%s raised %s; applying on_validator_error=%r",
                        trace.run_id, validator_cls, type(exc).__name__,
                        self.on_validator_error,
                    )
                    validator_result = self._error_validator_result(exc)
                else:
                    logger.debug(
                        "run=%s validator result: valid=%s confidence=%.2f recommendation=%s issues=%d",
                        trace.run_id, validator_result.valid, validator_result.confidence,
                        validator_result.recommendation, len(validator_result.issues),
                    )

        decision = self.policy.decide(trace, trigger_results, validator_result)
        self._log_decision(trace, decision)
        return decision
