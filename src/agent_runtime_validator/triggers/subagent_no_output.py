from ..schema.trace import ExecutionTrace
from ..schema.decisions import TriggerResult, Severity
from .base import BaseTrigger


class SubagentNoOutputTrigger(BaseTrigger):
    """Fires when too many agent calls completed without returning output.

    An ``AgentCall`` with ``output=None`` means the callee was invoked but never
    wrote a result back to the supervisor. When several subagents stall this way,
    it typically signals a graph deadlock, an unhandled exception in a subgraph,
    or an agent that silently timed out.
    """

    def __init__(self, min_stalled: int = 1, severity: Severity = "medium"):
        self.min_stalled = min_stalled
        self.severity: Severity = severity

    def evaluate(self, trace: ExecutionTrace) -> TriggerResult:
        stalled = [ac for ac in trace.agent_calls if ac.output is None]
        count = len(stalled)
        triggered = count >= self.min_stalled
        stalled_callees = sorted({ac.callee for ac in stalled})
        return TriggerResult(
            triggered=triggered,
            trigger_name="SubagentNoOutputTrigger",
            severity=self.severity,
            reason=(
                f"{count} agent call(s) with no output (stalled): "
                f"{', '.join(stalled_callees)}"
                if triggered
                else f"All agent calls returned output ({count} stalled < {self.min_stalled})"
            ),
            evidence={
                "stalled_count": count,
                "stalled_callees": stalled_callees,
                "min_stalled": self.min_stalled,
            },
        )
