from abc import ABC, abstractmethod
from ..schema.trace import ExecutionTrace
from ..schema.decisions import TriggerResult


class BaseTrigger(ABC):
    @abstractmethod
    def evaluate(self, trace: ExecutionTrace) -> TriggerResult: ...
