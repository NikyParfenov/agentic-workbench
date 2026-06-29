from datetime import datetime, timezone

try:
    from langgraph.graph import StateGraph  # noqa: F401
except ImportError:
    raise ImportError(
        "LangGraph integration requires langgraph. "
        "Install it with: pip install agent-runtime-validator[langgraph]"
    ) from None

from ...schema.trace import ExecutionTrace


def state_to_trace(state: dict, run_id: str | None = None) -> ExecutionTrace:
    return ExecutionTrace(
        run_id=run_id or state.get("run_id", "langgraph-run"),
        started_at=state.get("started_at", datetime.now(timezone.utc)),
        messages=state.get("_trace_messages", []),
        agent_calls=state.get("_trace_agent_calls", []),
        tool_calls=state.get("_trace_tool_calls", []),
        tool_results=state.get("_trace_tool_results", []),
        routing_events=state.get("_trace_routing_events", []),
        artifacts=state.get("_trace_artifacts", []),
        errors=state.get("_trace_errors", []),
        metadata=state.get("_trace_metadata", {}),
        token_usage=state.get("_trace_token_usage"),
    )
