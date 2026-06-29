"""Basic runtime validation with triggers only — no LLM, no framework.

Simulates an agent that calls the same tool repeatedly without producing any
artifact, then validates the trace.

Run:
    uv run python examples/basic_runtime.py
"""
from datetime import datetime, timezone

from agent_runtime_validator import ExecutionTrace, RuntimeValidator
from agent_runtime_validator.schema.events import ToolCall, ToolResult
from agent_runtime_validator.triggers import (
    MaxToolCallsTrigger,
    SameToolLoopTrigger,
    NoProgressTrigger,
)


def now() -> datetime:
    return datetime.now(timezone.utc)


def build_trace() -> ExecutionTrace:
    trace = ExecutionTrace(run_id="basic-demo", started_at=now())
    for i in range(4):
        call_id = f"c{i + 1}"
        trace.tool_calls.append(
            ToolCall(tool_name="search", call_id=call_id, args={"q": "acme"}, timestamp=now())
        )
        trace.tool_results.append(
            ToolResult(call_id=call_id, tool_name="search", output="no results", timestamp=now())
        )
    return trace


def main() -> None:
    validator = RuntimeValidator(
        triggers=[
            MaxToolCallsTrigger(max_calls=10),
            SameToolLoopTrigger(max_repeats=3),
            NoProgressTrigger(min_tool_calls=3),
        ],
    )

    decision = validator.validate(build_trace())

    print(f"should_continue: {decision.should_continue}")
    print(f"action:          {decision.action}")
    print(f"severity:        {decision.severity}")
    print(f"triggered_by:    {decision.triggered_by}")
    print(f"reason:          {decision.reason}")


if __name__ == "__main__":
    main()
