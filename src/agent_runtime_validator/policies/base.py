from abc import ABC, abstractmethod
from ..schema.trace import ExecutionTrace
from ..schema.decisions import TriggerResult, ValidatorResult, ValidationDecision


class BasePolicy(ABC):
    @abstractmethod
    def decide(
        self,
        trace: ExecutionTrace,
        triggered: list[TriggerResult],
        validator_result: ValidatorResult | None,
    ) -> ValidationDecision: ...
