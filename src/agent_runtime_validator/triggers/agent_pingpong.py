from ..schema.trace import ExecutionTrace
from ..schema.decisions import TriggerResult, Severity
from .base import BaseTrigger


class AgentPingPongTrigger(BaseTrigger):
    """Fires when the same pair of agents routes back and forth repeatedly.

    A *cycle* is one completed round trip: ``A → B`` followed by ``B → A``.
    ``max_cycles`` is the number of consecutive round trips at which the
    trigger fires — an outbound leg without the return does not count.
    """

    def __init__(self, max_cycles: int, severity: Severity = "high"):
        if max_cycles < 1:
            raise ValueError("max_cycles must be >= 1")
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

        # Scan for runs of consecutive alternating events (each event reverses
        # the previous one); a run of length L contains L // 2 round trips.
        max_round_trips = 0
        worst_pair: tuple[str, str] | None = None
        run_start = 0

        for k in range(1, len(events) + 1):
            alternates = (
                k < len(events)
                and events[k].from_agent == events[k - 1].to_agent
                and events[k].to_agent == events[k - 1].from_agent
            )
            if not alternates:
                round_trips = (k - run_start) // 2
                if round_trips > max_round_trips:
                    max_round_trips = round_trips
                    worst_pair = (events[run_start].from_agent, events[run_start].to_agent)
                run_start = k

        triggered = max_round_trips >= self.max_cycles
        pair_str = f"{worst_pair[0]} ↔ {worst_pair[1]}" if worst_pair else "unknown"
        return TriggerResult(
            triggered=triggered,
            trigger_name="AgentPingPongTrigger",
            severity=self.severity,
            reason=(
                f"Agents {pair_str} ping-ponged {max_round_trips} round trip(s) (limit {self.max_cycles})"
                if triggered
                else f"No agent pair completed {self.max_cycles}+ ping-pong round trips"
            ),
            evidence={
                "pair": list(worst_pair) if worst_pair else None,
                "cycles": max_round_trips,
                "max_cycles": self.max_cycles,
            },
        )
