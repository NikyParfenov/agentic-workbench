"""Tests for from_langchain_messages adapter.

No LangChain installation required — fake message classes are used throughout.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from agent_runtime_validator.integrations.langgraph import from_langchain_messages
from agent_runtime_validator.schema.events import ArtifactEvent
from agent_runtime_validator.schema.trace import ExecutionTrace


# ---------------------------------------------------------------------------
# Fake message classes (no LangChain needed)
# ---------------------------------------------------------------------------

class FakeHumanMessage:
    type = "human"
    content = "hello"
    tool_calls: list = []


class FakeAIMessage:
    type = "ai"
    content = "I will call a tool"
    tool_calls = [{"id": "c1", "name": "lookup_record", "args": {"record_id": "demo-record"}}]


class FakeToolMessage:
    type = "tool"
    content = "not found"
    tool_call_id = "c1"
    name = "lookup_record"


class FakeSystemMessage:
    type = "system"
    content = "You are a helpful assistant."
    tool_calls: list = []


class FakeAIMessageListContent:
    """AIMessage whose content is a list of blocks (multi-modal style)."""
    type = "ai"
    content = [
        {"type": "text", "text": "Here is what I found:"},
        {"type": "text", "text": "Result A"},
    ]
    tool_calls: list = []


class FakeAIMessageNoId:
    """AIMessage whose tool_calls entry has no id (fallback required)."""
    type = "ai"
    content = "calling a tool"
    tool_calls = [{"name": "some_tool", "args": {}}]  # no "id" key


class FakeToolMessageNoName:
    """ToolMessage without a name attribute."""
    type = "tool"
    content = "output"
    tool_call_id = "c2"
    # intentionally no `.name`


# ---------------------------------------------------------------------------
# 1. Role mapping for human / ai / system / tool → MessageEvent
# ---------------------------------------------------------------------------

def test_human_message_maps_to_user_role():
    trace = from_langchain_messages([FakeHumanMessage()])
    assert len(trace.messages) == 1
    assert trace.messages[0].role == "user"
    assert trace.messages[0].content == "hello"


def test_ai_message_maps_to_assistant_role():
    trace = from_langchain_messages([FakeAIMessage()])
    # AI message produces a MessageEvent
    msg = next(m for m in trace.messages if m.role == "assistant")
    assert msg.content == "I will call a tool"


def test_system_message_maps_to_system_role():
    trace = from_langchain_messages([FakeSystemMessage()])
    assert trace.messages[0].role == "system"
    assert trace.messages[0].content == "You are a helpful assistant."


def test_tool_message_maps_to_tool_role():
    trace = from_langchain_messages([FakeToolMessage()])
    msg = next(m for m in trace.messages if m.role == "tool")
    assert msg.content == "not found"


def test_unknown_type_falls_back_to_assistant():
    class FakeUnknown:
        type = "mystery"
        content = "???"
        tool_calls: list = []

    trace = from_langchain_messages([FakeUnknown()])
    assert trace.messages[0].role == "assistant"


def test_role_detection_via_class_name_fallback():
    """When .type is absent, class name lower-cased is used for role detection."""
    class HumanMessage:
        content = "hi"
        tool_calls: list = []
        # no .type attribute

    trace = from_langchain_messages([HumanMessage()])
    assert trace.messages[0].role == "user"


# ---------------------------------------------------------------------------
# 2. AIMessage tool_calls → ToolCall entries
# ---------------------------------------------------------------------------

def test_ai_tool_calls_produce_tool_call_entries():
    trace = from_langchain_messages([FakeAIMessage()])
    assert len(trace.tool_calls) == 1
    tc = trace.tool_calls[0]
    assert tc.tool_name == "lookup_record"
    assert tc.args == {"record_id": "demo-record"}
    assert tc.call_id == "c1"


def test_tool_call_agent_name_propagated():
    trace = from_langchain_messages([FakeAIMessage()], agent_name="my_agent")
    assert trace.tool_calls[0].agent_name == "my_agent"


# ---------------------------------------------------------------------------
# 3. ToolMessage → ToolResult
# ---------------------------------------------------------------------------

def test_tool_message_produces_tool_result():
    trace = from_langchain_messages([FakeToolMessage()])
    assert len(trace.tool_results) == 1
    tr = trace.tool_results[0]
    assert tr.call_id == "c1"
    assert tr.tool_name == "lookup_record"
    assert tr.output == "not found"
    assert tr.error is None


# ---------------------------------------------------------------------------
# 4. Missing tool call id → deterministic fallback f"tool-call-{index}"
# ---------------------------------------------------------------------------

def test_missing_tool_call_id_gets_deterministic_fallback():
    trace = from_langchain_messages([FakeAIMessageNoId()])
    assert trace.tool_calls[0].call_id == "tool-call-0"


def test_second_tool_call_no_id_gets_index_1():
    class FakeAITwoTools:
        type = "ai"
        content = "calling"
        tool_calls = [
            {"name": "tool_a", "args": {}},   # no id
            {"name": "tool_b", "args": {}},   # no id
        ]

    trace = from_langchain_messages([FakeAITwoTools()])
    assert trace.tool_calls[0].call_id == "tool-call-0"
    assert trace.tool_calls[1].call_id == "tool-call-1"


# ---------------------------------------------------------------------------
# 5. Missing tool name falls back to "unknown_tool"
# ---------------------------------------------------------------------------

def test_missing_tool_name_in_tool_message_falls_back():
    trace = from_langchain_messages([FakeToolMessageNoName()])
    assert trace.tool_results[0].tool_name == "unknown_tool"


# ---------------------------------------------------------------------------
# 6. All timestamps are timezone-aware
# ---------------------------------------------------------------------------

def test_all_timestamps_are_timezone_aware():
    messages = [FakeHumanMessage(), FakeAIMessage(), FakeToolMessage()]
    trace = from_langchain_messages(messages)
    for msg in trace.messages:
        assert msg.timestamp.tzinfo is not None, f"message timestamp missing tzinfo: {msg}"
    for tc in trace.tool_calls:
        assert tc.timestamp.tzinfo is not None
    for tr in trace.tool_results:
        assert tr.timestamp.tzinfo is not None


def test_started_at_propagated_and_timezone_aware():
    ts = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    trace = from_langchain_messages([FakeHumanMessage()], started_at=ts)
    assert trace.started_at == ts
    assert trace.started_at.tzinfo is not None


def test_default_started_at_is_timezone_aware():
    trace = from_langchain_messages([FakeHumanMessage()])
    assert trace.started_at.tzinfo is not None


# ---------------------------------------------------------------------------
# 7. artifacts argument is preserved in trace.artifacts
# ---------------------------------------------------------------------------

def test_artifacts_preserved():
    artifact = ArtifactEvent(
        artifact_id="a1",
        artifact_type="report",
        content="summary text",
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    trace = from_langchain_messages([], artifacts=[artifact])
    assert len(trace.artifacts) == 1
    assert trace.artifacts[0].artifact_id == "a1"


def test_artifacts_none_produces_empty_list():
    trace = from_langchain_messages([])
    assert trace.artifacts == []


# ---------------------------------------------------------------------------
# 8. metadata is preserved and merged with _source key
# ---------------------------------------------------------------------------

def test_metadata_merged_with_source():
    trace = from_langchain_messages([], metadata={"run_env": "prod"})
    assert trace.metadata["run_env"] == "prod"
    assert trace.metadata["_source"] == "langchain_messages"


def test_source_key_always_set_even_without_metadata():
    trace = from_langchain_messages([])
    assert trace.metadata["_source"] == "langchain_messages"


def test_metadata_source_overwritten_by_adapter():
    trace = from_langchain_messages([], metadata={"_source": "caller_set_this"})
    assert trace.metadata["_source"] == "langchain_messages"


# ---------------------------------------------------------------------------
# 9. Works as ValidationNode trace_builder callback
# ---------------------------------------------------------------------------

def test_adapter_as_validation_node_trace_builder():
    """Build a ValidationNode using from_langchain_messages as trace_builder."""
    # Import ValidationNode without requiring langgraph at module level
    langgraph = pytest.importorskip("langgraph", reason="langgraph not installed")  # noqa: F841

    from agent_runtime_validator.integrations.langgraph import ValidationNode
    from agent_runtime_validator.triggers import MaxRoutesTrigger

    def build_trace(state: dict) -> ExecutionTrace:
        return from_langchain_messages(
            state.get("messages", []),
            run_id=state.get("run_id", "run"),
            agent_name="research_agent",
        )

    node = ValidationNode(
        triggers=[MaxRoutesTrigger(max_routes=100)],
        trace_builder=build_trace,
    )
    state = {"messages": [FakeHumanMessage(), FakeAIMessage()], "run_id": "test-run"}
    result = node(state)
    decision = result["decision"]
    assert decision is not None
    assert decision.action == "continue"


# ---------------------------------------------------------------------------
# 10. Empty messages list → valid ExecutionTrace with empty lists
# ---------------------------------------------------------------------------

def test_empty_messages_produces_valid_trace():
    trace = from_langchain_messages([])
    assert isinstance(trace, ExecutionTrace)
    assert trace.messages == []
    assert trace.tool_calls == []
    assert trace.tool_results == []
    assert trace.run_id == "langchain-run"


def test_run_id_propagated():
    trace = from_langchain_messages([], run_id="my-run-42")
    assert trace.run_id == "my-run-42"


# ---------------------------------------------------------------------------
# Additional: list-content in AIMessage is joined as text
# ---------------------------------------------------------------------------

def test_list_content_ai_message_joined():
    trace = from_langchain_messages([FakeAIMessageListContent()])
    assert "Here is what I found:" in trace.messages[0].content
    assert "Result A" in trace.messages[0].content
