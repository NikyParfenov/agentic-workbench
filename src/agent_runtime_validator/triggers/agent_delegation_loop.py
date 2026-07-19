from collections import Counter

from ..schema.trace import ExecutionTrace
from ..schema.decisions import TriggerResult, Severity
from .base import BaseTrigger


class AgentDelegationLoopTrigger(BaseTrigger):
    """Fires when a supervisor re-delegates to the same subagent too many times.

    Counts how often each ``(caller, callee)`` pair appears in ``agent_calls``
    and fires when any pair reaches ``max_repeats``. A supervisor that keeps
    dispatching to the same subagent without progress is the primary target.
    """

    def __init__(self, max_repeats: int, severity: Severity = "high"):
        if max_repeats < 1:
            raise ValueError("max_repeats must be >= 1")
        self.max_repeats = max_repeats
        self.severity: Severity = severity

    def evaluate(self, trace: ExecutionTrace) -> TriggerResult:
        counts: Counter[tuple[str, str]] = Counter(
            (ac.caller, ac.callee) for ac in trace.agent_calls
        )
        if not counts:
            return TriggerResult(
                triggered=False,
                trigger_name="AgentDelegationLoopTrigger",
                severity=self.severity,
                reason="No agent calls recorded",
                evidence={"max_repeats": self.max_repeats},
            )

        (top_caller, top_callee), top_count = counts.most_common(1)[0]
        triggered = top_count >= self.max_repeats
        return TriggerResult(
            triggered=triggered,
            trigger_name="AgentDelegationLoopTrigger",
            severity=self.severity,
            reason=(
                f"{top_caller!r} → {top_callee!r} repeated {top_count} time(s) "
                f"(limit {self.max_repeats})"
                if triggered
                else f"No delegation pair repeated {self.max_repeats} or more times "
                f"(max seen: {top_count})"
            ),
            evidence={
                "top_caller": top_caller,
                "top_callee": top_callee,
                "top_count": top_count,
                "max_repeats": self.max_repeats,
            },
        )
