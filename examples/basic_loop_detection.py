"""Detect an agent stuck in a tool loop and interrupt it.

Simulates an agent calling the same tool with the same arguments repeatedly.
The validator catches the loop after 3 repetitions and the decision escalates
to "interrupt". No LLM, no framework — pure triggers + policy.

Run:
    uv run python examples/basic_loop_detection.py
"""
from datetime import datetime, timezone

from agent_runtime_validator import ExecutionTrace, RuntimeValidator
from agent_runtime_validator.schema.events import ToolCall, ToolResult
from agent_runtime_validator.triggers import (
    SameToolSameArgsLoopTrigger,
    NoProgressTrigger,
    ToolErrorRateTrigger,
)


def now() -> datetime:
    return datetime.now(timezone.utc)


def simulate_stuck_agent() -> ExecutionTrace:
    """Agent searches for 'missing-dataset' 5 times, always gets an error."""
    trace = ExecutionTrace(run_id="stuck-agent-demo", started_at=now())
    for i in range(5):
        cid = f"c{i + 1}"
        trace.tool_calls.append(
            ToolCall(
                tool_name="query_dataset",
                call_id=cid,
                args={"dataset": "missing-dataset", "limit": 100},
                agent_name="data_agent",
                timestamp=now(),
            )
        )
        trace.tool_results.append(
            ToolResult(
                call_id=cid,
                tool_name="query_dataset",
                error="DatasetNotFoundError: 'missing-dataset' does not exist",
                timestamp=now(),
            )
        )
    return trace


def main() -> None:
    validator = RuntimeValidator(
        triggers=[
            SameToolSameArgsLoopTrigger(max_repeats=3, severity="high"),
            NoProgressTrigger(min_tool_calls=3),
            ToolErrorRateTrigger(max_error_rate=0.5, severity="high"),
        ],
    )

    trace = simulate_stuck_agent()

    print("=== Before validation ===")
    print(f"Tool calls:  {len(trace.tool_calls)}")
    print(f"Errors:      {sum(1 for r in trace.tool_results if r.error)}")
    print(f"Artifacts:   {len(trace.artifacts)}")
    print()

    decision = validator.validate(trace)

    print("=== Validation decision ===")
    print(f"Action:      {decision.action}")
    print(f"Severity:    {decision.severity}")
    print(f"Continue?    {decision.should_continue}")
    print(f"Triggered:   {decision.triggered_by}")
    print(f"Reason:      {decision.reason}")


if __name__ == "__main__":
    main()
