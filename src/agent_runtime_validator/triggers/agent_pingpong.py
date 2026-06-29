from ..schema.trace import ExecutionTrace
from ..schema.decisions import TriggerResult, Severity
from .base import BaseTrigger


class AgentPingPongTrigger(BaseTrigger):
    """Fires when the same pair of agents routes back and forth repeatedly."""

    def __init__(self, max_cycles: int, severity: Severity = "high"):
        self.max_cycles = max_cycles
        self.severity: Severity = severity

    def evaluate(self, trace: ExecutionTrace) -> TriggerResult:
        events = trace.routing_events
        if len(events) < 2:
            return TriggerResult(
                triggered=False,
                trigger_name="AgentPingPongTrigger",
                severity=self.severity,
                reason="Not enough routing events to detect ping-pong",
                evidence={"max_cycles": self.max_cycles},
            )

        # Count consecutive back-and-forth cycles between any pair
        max_cycles_seen = 0
        worst_pair: tuple[str, str] | None = None

        # For each ordered pair, track how many alternating round trips occurred
        i = 0
        while i < len(events) - 1:
            a_to_b = (events[i].from_agent, events[i].to_agent)
            b_to_a = (events[i].to_agent, events[i].from_agent)
            cycles = 1
            j = i + 1
            while j < len(events) - 1:
                if (events[j].from_agent, events[j].to_agent) == b_to_a:
                    if (events[j + 1].from_agent, events[j + 1].to_agent) == a_to_b:
                        cycles += 1
                        j += 2
                        continue
                break
            if cycles > max_cycles_seen:
                max_cycles_seen = cycles
                worst_pair = a_to_b
            i += 1

        triggered = max_cycles_seen >= self.max_cycles
        pair_str = f"{worst_pair[0]} ↔ {worst_pair[1]}" if worst_pair else "unknown"
        return TriggerResult(
            triggered=triggered,
            trigger_name="AgentPingPongTrigger",
            severity=self.severity,
            reason=(
                f"Agents {pair_str} ping-ponged {max_cycles_seen} times (limit {self.max_cycles})"
                if triggered
                else f"No agent pair ping-ponged {self.max_cycles}+ times"
            ),
            evidence={
                "pair": list(worst_pair) if worst_pair else None,
                "cycles": max_cycles_seen,
                "max_cycles": self.max_cycles,
            },
        )
