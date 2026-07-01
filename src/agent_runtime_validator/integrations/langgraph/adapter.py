from datetime import datetime, timezone
from typing import Any

try:
    from langgraph.graph import StateGraph  # noqa: F401
except ImportError:
    raise ImportError(
        "LangGraph integration requires langgraph. "
        "Install it with: pip install agent-runtime-validator[langgraph]"
    ) from None

from ...schema.trace import ExecutionTrace


def state_to_trace(state: dict[str, Any], run_id: str | None = None) -> ExecutionTrace:
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


def get_trace_from_state(
    state: dict[str, Any], trace_key: str = "trace"
) -> ExecutionTrace | None:
    """Extract an ``ExecutionTrace`` from LangGraph state, handling all storage forms.

    Returns ``None`` when *trace_key* is not present in *state*.
    Handles three forms that appear in practice:

    - ``ExecutionTrace`` object (in-memory, before checkpointing)
    - ``dict`` (after LangGraph deserializes a checkpointed state)
    - absent key (graph never ran through a ``ValidationNode``)

    Useful at the end of a graph run to retrieve the trace for archiving::

        def save_run_trace(state: dict) -> dict:
            trace = get_trace_from_state(state)
            if trace is not None:
                save_trace(trace, f"traces/{trace.run_id}.json")
            return state
    """
    raw = state.get(trace_key)
    if raw is None:
        return None
    if isinstance(raw, ExecutionTrace):
        return raw
    if isinstance(raw, dict):
        return ExecutionTrace(**raw)
    return None
