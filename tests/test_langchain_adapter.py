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


# ---------------------------------------------------------------------------
# 11. Subgraph thoughts lifted from message metadata
# ---------------------------------------------------------------------------

class FakeAIMessageWithThoughts:
    type = "ai"
    content = "Working..."
    tool_calls = []
    additional_kwargs = {
        "_subgraph_thoughts": [
            'Tool call [c1] analyze_item with arguments: {"item_id": "demo-item"}',
            'Tool result [c1]: {"status": "ok"}',
        ]
    }


class FakeAIMessageWithThoughtsInResponseMetadata:
    """Thoughts stored in response_metadata, not additional_kwargs."""
    type = "ai"
    content = "Working..."
    tool_calls = []
    additional_kwargs: dict = {}
    response_metadata = {
        "_subgraph_thoughts": [
            'Tool call [r1] fetch_data with arguments: {"id": "x"}',
            'Tool result [r1]: done',
        ]
    }


class FakeAIMessageWithWorkerLog:
    """Thoughts stored under a custom key."""
    type = "ai"
    content = "Processing..."
    tool_calls = []
    additional_kwargs = {
        "worker_log": [
            'Tool call [w1] process_item with arguments: {"item": "y"}',
            'Tool result [w1]: processed',
        ]
    }


class FakeAIMessageThoughtsNotList:
    """Subgraph thoughts key holds a non-list value — should be ignored."""
    type = "ai"
    content = "Just thinking..."
    tool_calls = []
    additional_kwargs = {"_subgraph_thoughts": "some string, not a list"}


class FakeAIMessageMalformedThought:
    """One thought line is malformed; adapter should not raise."""
    type = "ai"
    content = "Whatever"
    tool_calls = []
    additional_kwargs = {
        "_subgraph_thoughts": [
            "this is not a recognized pattern at all",
            'Tool call [m1] do_thing with arguments: {"k": "v"}',
            'Tool result [m1]: ok',
        ]
    }


def test_subgraph_thoughts_produce_tool_call_and_result():
    """Metadata _subgraph_thoughts produces ToolCall and ToolResult in returned trace."""
    trace = from_langchain_messages([FakeAIMessageWithThoughts()])
    assert len(trace.tool_calls) == 1
    assert trace.tool_calls[0].tool_name == "analyze_item"
    assert trace.tool_calls[0].call_id == "c1"
    assert len(trace.tool_results) == 1
    assert trace.tool_results[0].call_id == "c1"


def test_include_subgraph_thoughts_false_ignores_metadata():
    """include_subgraph_thoughts=False skips metadata inspection entirely."""
    trace = from_langchain_messages(
        [FakeAIMessageWithThoughts()],
        include_subgraph_thoughts=False,
    )
    assert len(trace.tool_calls) == 0
    assert len(trace.tool_results) == 0


def test_custom_subgraph_thoughts_key():
    """Custom subgraph_thoughts_key is respected."""
    trace = from_langchain_messages(
        [FakeAIMessageWithWorkerLog()],
        subgraph_thoughts_key="worker_log",
    )
    assert len(trace.tool_calls) == 1
    assert trace.tool_calls[0].tool_name == "process_item"
    assert len(trace.tool_results) == 1


def test_non_list_value_for_thoughts_key_ignored():
    """Non-list value for the thoughts key is ignored safely."""
    trace = from_langchain_messages([FakeAIMessageThoughtsNotList()])
    assert len(trace.tool_calls) == 0
    assert len(trace.tool_results) == 0


def test_malformed_thought_line_does_not_raise():
    """A malformed thought line does not raise; valid lines are still parsed."""
    trace = from_langchain_messages([FakeAIMessageMalformedThought()])
    # The malformed line does not raise; recognized lines produce events.
    assert len(trace.tool_calls) == 1
    assert len(trace.tool_results) == 1


def test_no_routing_or_agent_calls_inferred_from_thoughts():
    """No routing_events or agent_calls are inferred from subgraph thoughts."""
    trace = from_langchain_messages([FakeAIMessageWithThoughts()])
    assert trace.routing_events == []
    assert trace.agent_calls == []


def test_existing_message_mapping_still_works_with_subgraph_thoughts():
    """Top-level messages are still present when subgraph thoughts are lifted."""
    trace = from_langchain_messages([FakeHumanMessage(), FakeAIMessageWithThoughts()])
    # Human message + AI message from top-level + thought messages from subgraph
    top_level_roles = [m.role for m in trace.messages[:2]]
    assert "user" in top_level_roles
    assert "assistant" in top_level_roles
    # Tool call from subgraph metadata is merged in
    assert any(tc.tool_name == "analyze_item" for tc in trace.tool_calls)


def test_response_metadata_key_also_checked():
    """response_metadata is checked when additional_kwargs is absent or empty."""
    trace = from_langchain_messages([FakeAIMessageWithThoughtsInResponseMetadata()])
    assert len(trace.tool_calls) == 1
    assert trace.tool_calls[0].tool_name == "fetch_data"
    assert len(trace.tool_results) == 1


# ---------------------------------------------------------------------------
# 12. ToolMessage error status maps to ToolResult.error
# ---------------------------------------------------------------------------

class FakeToolMessageWithErrorStatus:
    type = "tool"
    content = "ValueError: item not found"
    tool_call_id = "c9"
    name = "analyze_item"
    status = "error"


class FakeToolMessageWithSuccessStatus:
    type = "tool"
    content = "all good"
    tool_call_id = "c10"
    name = "analyze_item"
    status = "success"


def test_tool_message_error_status_maps_to_error():
    trace = from_langchain_messages([FakeToolMessageWithErrorStatus()])
    tr = trace.tool_results[0]
    assert tr.error == "ValueError: item not found"
    assert tr.output is None


def test_tool_message_success_status_maps_to_output():
    trace = from_langchain_messages([FakeToolMessageWithSuccessStatus()])
    tr = trace.tool_results[0]
    assert tr.error is None
    assert tr.output == "all good"


def test_tool_message_error_status_feeds_error_rate_trigger():
    from agent_runtime_validator.triggers import ToolErrorRateTrigger

    trace = from_langchain_messages([FakeToolMessageWithErrorStatus()])
    result = ToolErrorRateTrigger(max_error_rate=0.5).evaluate(trace)
    assert result.triggered


# ---------------------------------------------------------------------------
# 13. _source metadata survives the subgraph-thoughts merge
# ---------------------------------------------------------------------------

def test_source_preserved_when_thoughts_merged():
    trace = from_langchain_messages([FakeAIMessageWithThoughts()])
    # Thought lines were lifted (tool call present) ...
    assert any(tc.tool_name == "analyze_item" for tc in trace.tool_calls)
    # ... but the trace-level _source still identifies the primary adapter.
    assert trace.metadata["_source"] == "langchain_messages"
