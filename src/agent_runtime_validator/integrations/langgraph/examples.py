"""
Supervisor-based LangGraph example graph demonstrating ValidationNode.
Run directly: python -m agent_runtime_validator.integrations.langgraph.examples
"""
from datetime import datetime, timezone
from agent_runtime_validator.schema.trace import ExecutionTrace
from agent_runtime_validator.schema.events import RoutingEvent, ToolCall, ToolResult
from agent_runtime_validator.triggers import MaxRoutesTrigger, SameToolLoopTrigger, NoProgressTrigger
from agent_runtime_validator.integrations.langgraph.nodes import ValidationNode


def run_example() -> None:
    node = ValidationNode(
        triggers=[
            MaxRoutesTrigger(max_routes=10),
            SameToolLoopTrigger(max_repeats=3),
            NoProgressTrigger(min_tool_calls=3),
        ],
    )

    ts = datetime.now(timezone.utc)
    trace = ExecutionTrace(
        run_id="demo-001",
        started_at=ts,
        routing_events=[
            RoutingEvent(from_agent="Supervisor", to_agent="ResearchAgent", timestamp=ts),
        ],
        tool_calls=[
            ToolCall(tool_name="lookup_record", call_id="c1", args={"record_id": "demo-record"}, timestamp=ts),
            ToolCall(tool_name="lookup_record", call_id="c2", args={"record_id": "demo-record"}, timestamp=ts),
            ToolCall(tool_name="lookup_record", call_id="c3", args={"record_id": "demo-record"}, timestamp=ts),
        ],
        tool_results=[
            ToolResult(call_id="c1", tool_name="lookup_record", output="not found", timestamp=ts),
            ToolResult(call_id="c2", tool_name="lookup_record", output="not found", timestamp=ts),
            ToolResult(call_id="c3", tool_name="lookup_record", output="not found", timestamp=ts),
        ],
    )

    state = {"trace": trace}
    result = node(state)
    decision = result["decision"]

    print(f"Action:      {decision.action}")
    print(f"Severity:    {decision.severity}")
    print(f"Triggered:   {decision.triggered_by}")
    print(f"Reason:      {decision.reason}")


if __name__ == "__main__":
    run_example()
