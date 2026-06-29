"""Deterministic trigger-score validation without an LLM.

Weighted triggers are aggregated into a risk score. If the score exceeds the
threshold, the validator recommends rerouting. A loop guard prevents infinite
reroutes by switching to "interrupt" after max_attempts.

Run:
    uv run python examples/trigger_score.py
"""
from datetime import datetime, timezone

from agent_runtime_validator import ExecutionTrace, RuntimeValidator
from agent_runtime_validator.schema.events import ToolCall
from agent_runtime_validator.triggers import (
    NoToolUsageTrigger,
    NoProgressTrigger,
    AgentPingPongTrigger,
)
from agent_runtime_validator.validators import TriggerScoreValidator


def now() -> datetime:
    return datetime.now(timezone.utc)


def build_trace() -> ExecutionTrace:
    trace = ExecutionTrace(run_id="score-demo", started_at=now())
    for i in range(4):
        trace.tool_calls.append(
            ToolCall(tool_name="search", call_id=f"c{i+1}", args={"q": "acme"}, timestamp=now())
        )
    return trace


def main() -> None:
    validator = RuntimeValidator(
        triggers=[
            NoToolUsageTrigger(watched_agents={"bio_agent"}),
            NoProgressTrigger(min_tool_calls=3),
            AgentPingPongTrigger(max_cycles=2),
        ],
        validator=TriggerScoreValidator(
            weights={
                "NoToolUsageTrigger": 2.0,
                "NoProgressTrigger": 2.0,
                "AgentPingPongTrigger": 3.0,
            },
            threshold=3.0,
            recommendation="reroute",
            max_attempts=1,
        ),
    )

    trace = build_trace()
    decision = validator.validate(trace)

    print(f"action:       {decision.action}")
    print(f"severity:     {decision.severity}")
    print(f"triggered_by: {decision.triggered_by}")
    print(f"reason:       {decision.reason}")

    if decision.validator_result:
        print(f"issues:       {decision.validator_result.issues}")

    print("\n--- Second validation (loop guard) ---")
    decision2 = validator.validate(trace)
    print(f"action:       {decision2.action}")
    print(f"reason:       {decision2.reason}")


if __name__ == "__main__":
    main()
