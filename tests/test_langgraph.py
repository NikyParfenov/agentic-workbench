import sys
import pytest

langgraph = pytest.importorskip("langgraph", reason="langgraph not installed")

import os
sys.path.insert(0, os.path.dirname(__file__))

from datetime import datetime, timezone
from conftest import make_trace, make_routing_event
from agent_runtime_validator.integrations.langgraph.nodes import ValidationNode
from agent_runtime_validator.integrations.langgraph.adapter import state_to_trace
from agent_runtime_validator.triggers import MaxRoutesTrigger


def test_validation_node_returns_continue_when_no_triggers_fire():
    node = ValidationNode(triggers=[MaxRoutesTrigger(max_routes=100)])
    trace = make_trace(routing_events=[make_routing_event("A", "B")])
    state = {"trace": trace, "messages": []}
    result = node(state)
    assert result["decision"].action == "continue"
    assert result["messages"] == []


def test_validation_node_does_not_mutate_state():
    node = ValidationNode(triggers=[MaxRoutesTrigger(max_routes=1)])
    trace = make_trace(routing_events=[make_routing_event("A", "B")])
    state = {"trace": trace}
    original_id = id(state)
    result = node(state)
    assert id(result) != original_id
    assert "decision" not in state


def test_validation_node_fires_trigger():
    node = ValidationNode(triggers=[MaxRoutesTrigger(max_routes=1)])
    trace = make_trace(routing_events=[make_routing_event("A", "B")])
    state = {"trace": trace}
    result = node(state)
    assert result["decision"].triggered_by == ["MaxRoutesTrigger"]


def test_validation_node_custom_keys():
    node = ValidationNode(
        triggers=[MaxRoutesTrigger(max_routes=1)],
        trace_key="execution_trace",
        decision_key="validation_result",
    )
    trace = make_trace(routing_events=[make_routing_event("A", "B")])
    state = {"execution_trace": trace}
    result = node(state)
    assert "validation_result" in result
    assert result["validation_result"].triggered_by == ["MaxRoutesTrigger"]


def test_state_to_trace_minimal():
    state = {
        "run_id": "test-123",
        "started_at": datetime(2024, 1, 1, tzinfo=timezone.utc),
    }
    trace = state_to_trace(state)
    assert trace.run_id == "test-123"
    assert trace.tool_calls == []


def test_state_to_trace_default_run_id():
    state = {}
    trace = state_to_trace(state)
    assert trace.run_id == "langgraph-run"


async def test_validation_node_async():
    node = ValidationNode(triggers=[MaxRoutesTrigger(max_routes=1)])
    trace = make_trace(routing_events=[make_routing_event("A", "B")])
    state = {"trace": trace}
    result = await node.async_call(state)
    assert result["decision"].triggered_by == ["MaxRoutesTrigger"]
