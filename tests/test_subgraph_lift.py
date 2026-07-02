"""Tests for lift_subgraph_messages — structured subgraph → parent trace lifting.

No LangChain installation required — fake message classes are used throughout.
"""

from __future__ import annotations

import pytest

langgraph = pytest.importorskip("langgraph", reason="langgraph not installed")

from datetime import datetime, timezone

from agent_runtime_validator.integrations.langgraph import lift_subgraph_messages
from agent_runtime_validator.schema.events import MessageEvent, ToolCall
from agent_runtime_validator.schema.trace import ExecutionTrace


_TS = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Fake message classes (no LangChain needed)
# ---------------------------------------------------------------------------

class FakeAIMessageWithToolCall:
    type = "ai"
    content = "calling tool"
    tool_calls = [{"id": "c1", "name": "analyze_item", "args": {"item_id": "demo-item"}}]


class FakeToolMessage:
    type = "tool"
    content = '{"status": "ok"}'
    tool_call_id = "c1"
    name = "analyze_item"
    tool_calls: list = []


class FakeToolMessageWithErrorStatus:
    type = "tool"
    content = "ValueError: item not found"
    tool_call_id = "c2"
    name = "analyze_item"
    status = "error"
    tool_calls: list = []


class FakeAIMessageWithThoughtsOnly:
    type = "ai"
    content = "Thinking..."
    tool_calls: list = []
    additional_kwargs = {
        "_subgraph_thoughts": [
            'Tool call [s1] analyze_item with arguments: {"item_id": "demo-item"}',
            'Tool result [s1]: {"status": "ok"}',
        ]
    }


def _parent_trace(**kwargs) -> ExecutionTrace:
    defaults: dict = {
        "run_id": "parent-run",
        "started_at": _TS,
        "messages": [
            MessageEvent(role="user", content="do the task", timestamp=_TS),
        ],
        "tool_calls": [
            ToolCall(tool_name="plan", call_id="p1", args={}, timestamp=_TS),
        ],
    }
    defaults.update(kwargs)
    return ExecutionTrace(**defaults)


# ---------------------------------------------------------------------------
# 1. Missing parent: a fresh trace is created from the subgraph messages
# ---------------------------------------------------------------------------

def test_lift_subgraph_messages_creates_trace_when_parent_missing():
    trace = lift_subgraph_messages(
        parent_trace=None,
        subgraph_messages=[FakeAIMessageWithToolCall(), FakeToolMessage()],
        run_id="fresh-run",
        agent_name="worker",
    )
    assert isinstance(trace, ExecutionTrace)
    assert trace.run_id == "fresh-run"
    assert len(trace.tool_calls) == 1
    assert trace.tool_calls[0].tool_name == "analyze_item"
    assert trace.tool_calls[0].agent_name == "worker"
    assert len(trace.tool_results) == 1
    assert trace.tool_results[0].call_id == "c1"


def test_lift_subgraph_messages_default_run_id_when_no_parent():
    trace = lift_subgraph_messages(
        parent_trace=None,
        subgraph_messages=[FakeAIMessageWithToolCall()],
    )
    assert trace.run_id == "subgraph-run"


def test_lift_subgraph_messages_parent_run_id_wins_over_default():
    parent = _parent_trace()
    trace = lift_subgraph_messages(
        parent_trace=parent,
        subgraph_messages=[FakeAIMessageWithToolCall()],
    )
    assert trace.run_id == "parent-run"


# ---------------------------------------------------------------------------
# 2. Existing parent: old and new events are both preserved
# ---------------------------------------------------------------------------

def test_lift_subgraph_messages_merges_into_existing_trace():
    parent = _parent_trace()
    trace = lift_subgraph_messages(
        parent_trace=parent,
        subgraph_messages=[FakeAIMessageWithToolCall(), FakeToolMessage()],
    )
    call_ids = [tc.call_id for tc in trace.tool_calls]
    assert "p1" in call_ids  # pre-existing parent event
    assert "c1" in call_ids  # lifted subgraph event
    assert len(trace.messages) == 3  # 1 parent + 2 subgraph
    assert trace.started_at == _TS  # parent started_at preserved


# ---------------------------------------------------------------------------
# 3. Serialized parent trace (dict) is parsed and merged
# ---------------------------------------------------------------------------

def test_lift_subgraph_messages_accepts_serialized_parent_trace():
    parent = _parent_trace()
    trace = lift_subgraph_messages(
        parent_trace=parent.model_dump(),
        subgraph_messages=[FakeAIMessageWithToolCall()],
    )
    assert trace.run_id == "parent-run"
    call_ids = [tc.call_id for tc in trace.tool_calls]
    assert "p1" in call_ids
    assert "c1" in call_ids


# ---------------------------------------------------------------------------
# 4. Input parent trace is never mutated
# ---------------------------------------------------------------------------

def test_lift_subgraph_messages_does_not_mutate_parent_trace():
    parent = _parent_trace(metadata={"key": "original"})
    lift_subgraph_messages(
        parent_trace=parent,
        subgraph_messages=[FakeAIMessageWithToolCall(), FakeToolMessage()],
        metadata={"extra": "value"},
    )
    assert len(parent.messages) == 1
    assert len(parent.tool_calls) == 1
    assert len(parent.tool_results) == 0
    assert parent.metadata == {"key": "original"}
    assert parent.run_id == "parent-run"


# ---------------------------------------------------------------------------
# 5. Textual thought parsing is OFF by default
# ---------------------------------------------------------------------------

def test_lift_subgraph_messages_defaults_to_no_textual_thought_parsing():
    trace = lift_subgraph_messages(
        parent_trace=None,
        subgraph_messages=[FakeAIMessageWithThoughtsOnly()],
    )
    assert trace.tool_calls == []
    assert trace.tool_results == []
    # The message itself is still preserved.
    assert len(trace.messages) == 1


# ---------------------------------------------------------------------------
# 6. Textual thought parsing is available as explicit opt-in
# ---------------------------------------------------------------------------

def test_lift_subgraph_messages_can_opt_into_subgraph_thoughts():
    trace = lift_subgraph_messages(
        parent_trace=None,
        subgraph_messages=[FakeAIMessageWithThoughtsOnly()],
        include_subgraph_thoughts=True,
    )
    assert any(tc.tool_name == "analyze_item" for tc in trace.tool_calls)
    assert len(trace.tool_results) >= 1


# ---------------------------------------------------------------------------
# 7. Parent metadata is preserved; helper adds its own non-conflicting key
# ---------------------------------------------------------------------------

def test_lift_subgraph_messages_preserves_parent_metadata():
    parent = _parent_trace(metadata={"_source": "parent", "custom": "keep-me"})
    trace = lift_subgraph_messages(
        parent_trace=parent,
        subgraph_messages=[FakeAIMessageWithToolCall()],
    )
    assert trace.metadata["_source"] == "parent"
    assert trace.metadata["custom"] == "keep-me"
    assert trace.metadata["_last_lifted_source"] == "subgraph_messages"


def test_lift_subgraph_messages_caller_metadata_does_not_override_parent():
    parent = _parent_trace(metadata={"custom": "parent-value"})
    trace = lift_subgraph_messages(
        parent_trace=parent,
        subgraph_messages=[FakeAIMessageWithToolCall()],
        metadata={"custom": "caller-value", "new_key": "added"},
    )
    assert trace.metadata["custom"] == "parent-value"
    assert trace.metadata["new_key"] == "added"


# ---------------------------------------------------------------------------
# 8. ToolMessage error status is mapped onto ToolResult.error
# ---------------------------------------------------------------------------

def test_lift_subgraph_messages_tool_error_status():
    trace = lift_subgraph_messages(
        parent_trace=None,
        subgraph_messages=[FakeToolMessageWithErrorStatus()],
    )
    assert len(trace.tool_results) == 1
    assert trace.tool_results[0].error == "ValueError: item not found"
    assert trace.tool_results[0].output is None


# ---------------------------------------------------------------------------
# 9. No routing events or agent calls are inferred
# ---------------------------------------------------------------------------

def test_lift_subgraph_messages_does_not_infer_routing_or_agent_calls():
    trace = lift_subgraph_messages(
        parent_trace=None,
        subgraph_messages=[FakeAIMessageWithToolCall(), FakeToolMessage()],
    )
    assert trace.routing_events == []
    assert trace.agent_calls == []


def test_lift_subgraph_messages_preserves_parent_routing_and_agent_calls():
    from agent_runtime_validator.schema.events import AgentCall, RoutingEvent

    parent = _parent_trace(
        routing_events=[RoutingEvent(from_agent="a", to_agent="b", timestamp=_TS)],
        agent_calls=[AgentCall(caller="a", callee="b", input="task", timestamp=_TS)],
    )
    trace = lift_subgraph_messages(
        parent_trace=parent,
        subgraph_messages=[FakeAIMessageWithToolCall()],
    )
    assert len(trace.routing_events) == 1
    assert len(trace.agent_calls) == 1
