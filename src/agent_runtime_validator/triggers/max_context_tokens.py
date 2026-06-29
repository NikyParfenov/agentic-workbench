import json
from ..schema.trace import ExecutionTrace
from ..schema.decisions import TriggerResult, Severity
from ..utils.token_counting import estimate_tokens
from .base import BaseTrigger


class MaxContextTokensTrigger(BaseTrigger):
    def __init__(self, max_tokens: int, severity: Severity = "high"):
        self.max_tokens = max_tokens
        self.severity: Severity = severity

    def evaluate(self, trace: ExecutionTrace) -> TriggerResult:
        total = 0
        if trace.token_usage is not None:
            total = trace.token_usage
        else:
            for msg in trace.messages:
                total += estimate_tokens(msg.content)
            for call in trace.tool_calls:
                total += estimate_tokens(json.dumps(call.args))
            for result in trace.tool_results:
                if result.output:
                    total += estimate_tokens(result.output)

        triggered = total >= self.max_tokens
        return TriggerResult(
            triggered=triggered,
            trigger_name="MaxContextTokensTrigger",
            severity=self.severity,
            reason=(
                f"Estimated context tokens ({total}) reached limit ({self.max_tokens})"
                if triggered
                else f"Estimated context tokens ({total}) within limit ({self.max_tokens})"
            ),
            evidence={"estimated_tokens": total, "max_tokens": self.max_tokens},
        )
