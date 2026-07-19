"""Tests for ingestion/config hardening: trigger param validation, global
fallback tool-call ids, strict trace-key resolution."""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import pytest

from conftest import make_trace, make_routing_event
from agent_runtime_validator.triggers import (
    MaxToolCallsTrigger,
    MaxRoutesTrigger,
    MaxContextTokensTrigger,
    MaxExecutionTimeTrigger,
    SameToolLoopTrigger,
    SameToolSameArgsLoopTrigger,
    AgentPingPongTrigger,
    NoProgressTrigger,
    ToolErrorRateTrigger,
    NoToolUsageTrigger,
    MaxAgentCallsTrigger,
    AgentDelegationLoopTrigger,
    SubagentNoOutputTrigger,
)


# ---------------------------------------------------------------------------
# Trigger constructor validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "factory",
    [
        lambda: MaxToolCallsTrigger(max_calls=0),
        lambda: MaxToolCallsTrigger(max_calls=-3),
        lambda: MaxRoutesTrigger(max_routes=0),
        lambda: MaxContextTokensTrigger(max_tokens=0),
        lambda: MaxExecutionTimeTrigger(max_seconds=0),
        lambda: MaxExecutionTimeTrigger(max_seconds=-1.5),
        lambda: SameToolLoopTrigger(max_repeats=0),
        lambda: SameToolSameArgsLoopTrigger(max_repeats=0),
        lambda: AgentPingPongTrigger(max_cycles=0),
        lambda: NoProgressTrigger(min_tool_calls=0),
        lambda: ToolErrorRateTrigger(max_error_rate=0.0),
        lambda: ToolErrorRateTrigger(max_error_rate=1.5),
        lambda: ToolErrorRateTrigger(max_error_rate=-0.2),
        lambda: ToolErrorRateTrigger(max_error_rate=0.5, min_results=0),
        lambda: NoToolUsageTrigger(watched_agents=set()),
        lambda: NoToolUsageTrigger(watched_agents={"a"}, min_expected_calls=0),
        lambda: MaxAgentCallsTrigger(max_calls=0),
        lambda: AgentDelegationLoopTrigger(max_repeats=0),
        lambda: SubagentNoOutputTrigger(min_stalled=0),
    ],
)
def test_invalid_trigger_params_raise(factory):
    with pytest.raises(ValueError):
        factory()


def test_boundary_values_accepted():
    """The minimum valid value must not raise."""
    MaxToolCallsTrigger(max_calls=1)
    SameToolLoopTrigger(max_repeats=1)
    ToolErrorRateTrigger(max_error_rate=1.0, min_results=1)
    NoToolUsageTrigger(watched_agents={"a"}, min_expected_calls=1)
    MaxExecutionTimeTrigger(max_seconds=0.1)


def test_same_tool_loop_zero_no_longer_fires_on_empty_trace():
    """Regression: max_repeats=0 used to fire on an empty trace with tool=None."""
    with pytest.raises(ValueError):
        SameToolLoopTrigger(max_repeats=0)


# ---------------------------------------------------------------------------
# Global fallback tool-call ids in from_langchain_messages
# ---------------------------------------------------------------------------

def test_fallback_call_ids_unique_across_messages():
    pytest.importorskip("langgraph", reason="langgraph not installed")
    from agent_runtime_validator.integrations.langgraph import from_langchain_messages

    class IdLessAIMessage:
        type = "ai"
        content = "working"
        tool_calls = [{"name": "analyze_item", "args": {"item_id": "demo-item"}}]

    trace = from_langchain_messages([IdLessAIMessage(), IdLessAIMessage()])
    ids = [c.call_id for c in trace.tool_calls]
    assert len(ids) == 2
    assert len(set(ids)) == 2, f"colliding fallback call ids: {ids}"
    assert ids == ["tool-call-0", "tool-call-1"]


def test_fallback_counter_skips_provider_ids():
    pytest.importorskip("langgraph", reason="langgraph not installed")
    from agent_runtime_validator.integrations.langgraph import from_langchain_messages

    class MixedAIMessage:
        type = "ai"
        content = "working"
        tool_calls = [
            {"id": "provider-1", "name": "analyze_item", "args": {}},
            {"name": "analyze_item", "args": {}},  # id-less
        ]

    trace = from_langchain_messages([MixedAIMessage(), MixedAIMessage()])
    ids = [c.call_id for c in trace.tool_calls]
    assert ids == ["provider-1", "tool-call-0", "provider-1", "tool-call-1"]


# ---------------------------------------------------------------------------
# strict_trace_key resolution in ValidationNode
# ---------------------------------------------------------------------------

def _node(**kwargs):
    pytest.importorskip("langgraph", reason="langgraph not installed")
    from agent_runtime_validator.integrations.langgraph.nodes import ValidationNode
    return ValidationNode(triggers=[MaxRoutesTrigger(max_routes=1)], **kwargs)


def test_strict_missing_trace_key_raises():
    node = _node(strict_trace_key=True)
    with pytest.raises(ValueError, match="strict_trace_key"):
        node({"messages": []})


def test_strict_wrong_type_raises():
    node = _node(strict_trace_key=True)
    with pytest.raises(ValueError, match="unusable type"):
        node({"trace": 42})


def test_strict_accepts_valid_trace():
    node = _node(strict_trace_key=True)
    trace = make_trace(routing_events=[make_routing_event("a", "b")])
    result = node({"trace": trace})
    assert result["decision"].triggered_by == ["MaxRoutesTrigger"]


def test_non_strict_missing_default_key_falls_back_silently(caplog):
    """Default key missing is normal (state_to_trace pattern) — no warning."""
    import logging
    node = _node()
    with caplog.at_level(logging.WARNING, logger="agent_runtime_validator"):
        node({"messages": []})
    assert not any("trace_key" in r.message or "falling back" in r.message
                   for r in caplog.records)


def test_non_strict_missing_custom_key_warns(caplog):
    import logging
    node = _node(trace_key="my_trace")
    with caplog.at_level(logging.WARNING, logger="agent_runtime_validator"):
        node({"messages": []})
    assert any("my_trace" in r.message for r in caplog.records)


def test_non_strict_wrong_type_warns_and_falls_back(caplog):
    import logging
    node = _node()
    with caplog.at_level(logging.WARNING, logger="agent_runtime_validator"):
        result = node({"trace": 42})
    assert any("unusable type" in r.message for r in caplog.records)
    assert result["decision"].action == "continue"  # empty fallback trace
