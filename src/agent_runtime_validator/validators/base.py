from abc import ABC, abstractmethod
from typing import Awaitable
from ..schema.trace import ExecutionTrace
from ..schema.decisions import TriggerResult, ValidatorResult


class BaseValidator(ABC):
    @abstractmethod
    def validate(
        self, trace: ExecutionTrace, trigger_results: list[TriggerResult]
    ) -> ValidatorResult | Awaitable[ValidatorResult]: ...

    async def validate_async(
        self, trace: ExecutionTrace, trigger_results: list[TriggerResult]
    ) -> ValidatorResult:
        result = self.validate(trace, trigger_results)
        if isinstance(result, ValidatorResult):
            return result
        return await result  # type: ignore[misc]
