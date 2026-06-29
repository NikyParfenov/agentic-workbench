from ..schema.trace import ExecutionTrace
from ..schema.decisions import TriggerResult, Severity
from .base import BaseTrigger


class NoToolUsageTrigger(BaseTrigger):
    """Fires when watched agents made no tool calls."""

    def __init__(
        self,
        watched_agents: set[str],
        min_expected_calls: int = 1,
        severity: Severity = "medium",
    ):
        self.watched_agents = watched_agents
        self.min_expected_calls = min_expected_calls
        self.severity: Severity = severity

    def evaluate(self, trace: ExecutionTrace) -> TriggerResult:
        silent_agents: list[str] = []
        for agent in self.watched_agents:
            count = sum(
                1 for c in trace.tool_calls if c.agent_name == agent
            )
            if count < self.min_expected_calls:
                silent_agents.append(agent)

        triggered = len(silent_agents) > 0
        return TriggerResult(
            triggered=triggered,
            trigger_name="NoToolUsageTrigger",
            severity=self.severity,
            reason=(
                f"Agents with no/insufficient tool calls: {', '.join(sorted(silent_agents))}"
                if triggered
                else "All watched agents made expected tool calls"
            ),
            evidence={
                "silent_agents": sorted(silent_agents),
                "watched_agents": sorted(self.watched_agents),
                "min_expected_calls": self.min_expected_calls,
            },
        )
