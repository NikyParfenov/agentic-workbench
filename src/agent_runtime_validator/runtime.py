import logging
from inspect import isawaitable

from .schema.trace import ExecutionTrace
from .schema.decisions import ValidationDecision
from .triggers.base import BaseTrigger
from .validators.base import BaseValidator
from .validators.noop import NoOpValidator
from .policies.base import BasePolicy
from .policies.default import DefaultPolicy

logger = logging.getLogger("agent_runtime_validator")


class RuntimeValidator:
    def __init__(
        self,
        triggers: list[BaseTrigger],
        validator: BaseValidator | None = None,
        policy: BasePolicy | None = None,
    ):
        self.triggers = triggers
        self.validator = validator or NoOpValidator()
        self.policy = policy or DefaultPolicy()

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

    def validate(self, trace: ExecutionTrace) -> ValidationDecision:
        trigger_results, fired = self._evaluate_triggers(trace)

        validator_result = None
        if fired and not isinstance(self.validator, NoOpValidator):
            validator_cls = type(self.validator).__name__
            logger.info("run=%s invoking validator=%s", trace.run_id, validator_cls)
            result = self.validator.validate(trace, trigger_results)
            if isawaitable(result):
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

        validator_result = None
        if fired and not isinstance(self.validator, NoOpValidator):
            validator_cls = type(self.validator).__name__
            logger.info("run=%s invoking validator=%s", trace.run_id, validator_cls)
            validator_result = await self.validator.validate_async(trace, trigger_results)
            logger.debug(
                "run=%s validator result: valid=%s confidence=%.2f recommendation=%s issues=%d",
                trace.run_id, validator_result.valid, validator_result.confidence,
                validator_result.recommendation, len(validator_result.issues),
            )

        decision = self.policy.decide(trace, trigger_results, validator_result)
        self._log_decision(trace, decision)
        return decision
