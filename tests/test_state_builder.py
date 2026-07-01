"""Tests for build_trace_from_state high-level state helper.

No LangChain installation required — fake message classes are used throughout.
"""

from __future__ import annotations

import pytest

langgraph = pytest.importorskip("langgraph", reason="langgraph not installed")

from datetime import datetime, timezone

from agent_runtime_validator.integrations.langgraph import build_trace_from_state
from agent_runtime_validator.schema.events import ArtifactEvent
from agent_runtime_validator.schema.trace import ExecutionTrace


# ---------------------------------------------------------------------------
# Fake message classes (no LangChain needed)
# ---------------------------------------------------------------------------

_TS = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


class FakeAIMessage:
    type = "ai"
    content = "Working"
    tool_calls = [{"id": "c1", "name": "analyze_item", "args": {"item_id": "demo-item"}}]


class FakeHumanMessage:
    type = "human"
    content = "Do the analysis"
    tool_calls: list = []


class FakeAIMessageWithThoughts:
    type = "ai"
    content = "Thinking..."
    tool_calls: list = []
    additional_kwargs = {
        "_subgraph_thoughts": [
            'Tool call [s1] analyze_item with arguments: {"item_id": "demo-item"}',
            'Tool result [s1]: {"status": "ok"}',
        ]
    }


# ---------------------------------------------------------------------------
# 1. Empty state returns ExecutionTrace with fallback run_id="langgraph-run"
# ---------------------------------------------------------------------------

def test_empty_state_fallback_run_id():
    trace = build_trace_from_state({})
    assert isinstance(trace, ExecutionTrace)
    assert trace.run_id == "langgraph-run"


# ---------------------------------------------------------------------------
# 2. run_id argument takes precedence over state["run_id"]
# ---------------------------------------------------------------------------

def test_run_id_arg_overrides_state():
    trace = build_trace_from_state({"run_id": "state-run"}, run_id="arg-run")
    assert trace.run_id == "arg-run"


# ---------------------------------------------------------------------------
# 3. state["run_id"] used when run_id argument is None
# ---------------------------------------------------------------------------

def test_state_run_id_used_when_no_arg():
    trace = build_trace_from_state({"run_id": "from-state"})
    assert trace.run_id == "from-state"


# ---------------------------------------------------------------------------
# 4. Existing ExecutionTrace in state[trace_key] is preserved (events not dropped)
# ---------------------------------------------------------------------------

def test_existing_trace_events_preserved():
    artifact = ArtifactEvent(
        artifact_id="item_id",
        artifact_type="report",
        content="existing result",
        timestamp=_TS,
    )
    existing = ExecutionTrace(
        run_id="base-run",
        started_at=_TS,
        artifacts=[artifact],
    )
    trace = build_trace_from_state({"trace": existing})
    assert any(a.artifact_id == "item_id" for a in trace.artifacts)


# ---------------------------------------------------------------------------
# 5. Existing trace as dict is parsed correctly
# ---------------------------------------------------------------------------

def test_existing_trace_dict_parsed():
    existing = ExecutionTrace(run_id="dict-run", started_at=_TS)
    state = {"trace": existing.model_dump()}
    trace = build_trace_from_state(state)
    assert trace.run_id == "dict-run"


# ---------------------------------------------------------------------------
# 6. Messages list is converted via from_langchain_messages and merged
# ---------------------------------------------------------------------------

def test_messages_converted_and_merged():
    trace = build_trace_from_state(
        {"messages": [FakeAIMessage()]},
        run_id="msg-run",
    )
    assert len(trace.messages) == 1
    assert len(trace.tool_calls) == 1
    assert trace.tool_calls[0].tool_name == "analyze_item"


# ---------------------------------------------------------------------------
# 7. Artifacts as ArtifactEvent objects are included
# ---------------------------------------------------------------------------

def test_artifacts_as_artifact_event_included():
    artifact = ArtifactEvent(
        artifact_id="item_id",
        artifact_type="report",
        content="content",
        timestamp=_TS,
    )
    trace = build_trace_from_state({"artifacts": [artifact]})
    assert len(trace.artifacts) == 1
    assert trace.artifacts[0].artifact_id == "item_id"


# ---------------------------------------------------------------------------
# 8. Artifacts as dicts are parsed into ArtifactEvent
# ---------------------------------------------------------------------------

def test_artifacts_as_dicts_parsed():
    artifact_dict = {
        "artifact_id": "item_id",
        "artifact_type": "report",
        "content": "text",
        "timestamp": _TS,
    }
    trace = build_trace_from_state({"artifacts": [artifact_dict]})
    assert len(trace.artifacts) == 1
    assert trace.artifacts[0].artifact_id == "item_id"


# ---------------------------------------------------------------------------
# 9. Invalid artifact dict is skipped silently (no raise)
# ---------------------------------------------------------------------------

def test_invalid_artifact_dict_skipped_silently():
    bad = {"not_a_valid": "artifact"}
    trace = build_trace_from_state({"artifacts": [bad]})
    assert trace.artifacts == []


# ---------------------------------------------------------------------------
# 10. Existing trace + messages are merged without dropping prior tool calls
# ---------------------------------------------------------------------------

def test_existing_trace_and_messages_merged_no_drop():
    from agent_runtime_validator.schema.events import ToolCall

    prior_call = ToolCall(
        tool_name="analyze_item",
        call_id="prior-c1",
        args={"item_id": "demo-item"},
        timestamp=_TS,
    )
    existing = ExecutionTrace(
        run_id="merge-run",
        started_at=_TS,
        tool_calls=[prior_call],
    )
    trace = build_trace_from_state(
        {"trace": existing, "messages": [FakeAIMessage()]},
    )
    call_ids = [tc.call_id for tc in trace.tool_calls]
    assert "prior-c1" in call_ids
    assert "c1" in call_ids


# ---------------------------------------------------------------------------
# 11. Provided metadata is merged without deleting existing metadata keys
# ---------------------------------------------------------------------------

def test_metadata_merged_no_overwrite():
    existing = ExecutionTrace(
        run_id="meta-run",
        started_at=_TS,
        metadata={"existing_key": "original"},
    )
    trace = build_trace_from_state(
        {"trace": existing},
        metadata={"new_key": "new_value", "existing_key": "should_not_replace"},
    )
    assert trace.metadata["existing_key"] == "original"
    assert trace.metadata["new_key"] == "new_value"


# ---------------------------------------------------------------------------
# 12. include_subgraph_thoughts=True lifts _subgraph_thoughts from message metadata
# ---------------------------------------------------------------------------

def test_include_subgraph_thoughts_true_lifts_thoughts():
    trace = build_trace_from_state(
        {"messages": [FakeAIMessageWithThoughts()]},
        include_subgraph_thoughts=True,
    )
    assert any(tc.tool_name == "analyze_item" for tc in trace.tool_calls)
    assert len(trace.tool_results) >= 1


# ---------------------------------------------------------------------------
# 13. include_subgraph_thoughts=False ignores thought metadata
# ---------------------------------------------------------------------------

def test_include_subgraph_thoughts_false_ignores():
    trace = build_trace_from_state(
        {"messages": [FakeAIMessageWithThoughts()]},
        include_subgraph_thoughts=False,
    )
    # No tool calls should come from subgraph thoughts
    assert trace.tool_calls == []
    assert trace.tool_results == []


# ---------------------------------------------------------------------------
# 14. Input state dict is not mutated after the call
# ---------------------------------------------------------------------------

def test_input_state_not_mutated():
    state: dict = {"run_id": "no-mut", "messages": [FakeAIMessage()]}
    original_keys = set(state.keys())
    original_messages_id = id(state["messages"])
    build_trace_from_state(state)
    assert set(state.keys()) == original_keys
    assert id(state["messages"]) == original_messages_id


# ---------------------------------------------------------------------------
# 15. Works as trace_builder lambda: ValidationNode(trace_builder=...)
# ---------------------------------------------------------------------------

def test_as_validation_node_trace_builder_lambda():
    from agent_runtime_validator.integrations.langgraph import ValidationNode
    from agent_runtime_validator.triggers import MaxRoutesTrigger

    class FakeMsg:
        type = "ai"
        content = "Working"
        tool_calls = [{"id": "c1", "name": "analyze_item", "args": {"item_id": "demo-item"}}]

    node = ValidationNode(
        triggers=[MaxRoutesTrigger(max_routes=1000)],
        validator_mode="always",
        trace_builder=lambda s: build_trace_from_state(s, agent_name="analyst"),
    )
    state = {"run_id": "lambda-run", "messages": [FakeMsg()]}
    result = node(state)
    assert result["decision"] is not None
