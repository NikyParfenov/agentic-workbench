from datetime import datetime, timezone, timedelta
from agent_runtime_validator.schema.events import (
    AgentCall, ToolCall, ToolResult, RoutingEvent, ArtifactEvent, ErrorEvent,
)
from agent_runtime_validator.schema.trace import ExecutionTrace


def make_trace(**kwargs) -> ExecutionTrace:
    defaults = {
        "run_id": "test-run",
        "started_at": datetime(2024, 1, 1, tzinfo=timezone.utc),
    }
    defaults.update(kwargs)
    return ExecutionTrace(**defaults)


def make_tool_call(
    tool_name: str = "search",
    call_id: str = "c1",
    args: dict | None = None,
    agent_name: str | None = None,
) -> ToolCall:
    return ToolCall(
        tool_name=tool_name,
        call_id=call_id,
        args=args or {},
        agent_name=agent_name,
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )


def make_tool_result(
    call_id: str = "c1",
    tool_name: str = "search",
    output: str | None = "result",
    error: str | None = None,
) -> ToolResult:
    return ToolResult(
        call_id=call_id,
        tool_name=tool_name,
        output=output,
        error=error,
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )


def make_routing_event(
    from_agent: str = "Supervisor",
    to_agent: str = "BioAgent",
) -> RoutingEvent:
    return RoutingEvent(
        from_agent=from_agent,
        to_agent=to_agent,
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )


def make_artifact(
    artifact_id: str = "a1",
    artifact_type: str = "report",
    ts_offset_days: int = 1,
) -> ArtifactEvent:
    return ArtifactEvent(
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        content="result content",
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=ts_offset_days),
    )


def make_agent_call(
    caller: str = "supervisor",
    callee: str = "researcher",
    input: str = "do this",
    output: str | None = "done",
) -> AgentCall:
    return AgentCall(
        caller=caller,
        callee=callee,
        input=input,
        output=output,
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )


def make_old_trace(seconds_ago: int) -> ExecutionTrace:
    return make_trace(
        started_at=datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)
    )
