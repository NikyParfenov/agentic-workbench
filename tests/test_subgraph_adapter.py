"""Tests for from_subgraph_thoughts adapter.

No LangChain installation required.
"""

from __future__ import annotations

from datetime import datetime, timezone

from agent_runtime_validator.integrations.langgraph import from_subgraph_thoughts
from agent_runtime_validator.schema.trace import ExecutionTrace


# ---------------------------------------------------------------------------
# 1. Parses "Tool call [c1] analyze_item with arguments: {...}" correctly
# ---------------------------------------------------------------------------

def test_tool_call_with_bracket_id():
    trace = from_subgraph_thoughts([
        'Tool call [c1] analyze_item with arguments: {"item_id": "demo-item"}',
    ])
    assert len(trace.tool_calls) == 1
    tc = trace.tool_calls[0]
    assert tc.call_id == "c1"
    assert tc.tool_name == "analyze_item"
    assert tc.args == {"item_id": "demo-item"}


# ---------------------------------------------------------------------------
# 2. Tool call without explicit id → deterministic "subgraph-tool-call-0"
# ---------------------------------------------------------------------------

def test_tool_call_without_id_gets_deterministic_id():
    trace = from_subgraph_thoughts([
        'Tool call analyze_item with arguments: {"item_id": "demo-item"}',
    ])
    assert len(trace.tool_calls) == 1
    assert trace.tool_calls[0].call_id == "subgraph-tool-call-0"
    assert trace.tool_calls[0].tool_name == "analyze_item"


def test_second_tool_call_without_id_gets_index_1():
    trace = from_subgraph_thoughts([
        'Tool call analyze_item with arguments: {"item_id": "demo-item"}',
        'Calling tool analyze_item with args {"item_id": "demo-item-2"}',
    ])
    assert trace.tool_calls[0].call_id == "subgraph-tool-call-0"
    assert trace.tool_calls[1].call_id == "subgraph-tool-call-1"


# ---------------------------------------------------------------------------
# 3. Parses "Tool result [c1]: {...}" → ToolResult matched to call c1
# ---------------------------------------------------------------------------

def test_tool_result_with_bracket_id_matches_call():
    trace = from_subgraph_thoughts([
        'Tool call [c1] analyze_item with arguments: {"item_id": "demo-item"}',
        'Tool result [c1]: {"status": "ok"}',
    ])
    assert len(trace.tool_results) == 1
    tr = trace.tool_results[0]
    assert tr.call_id == "c1"
    assert tr.tool_name == "analyze_item"
    assert tr.output == '{"status": "ok"}'


# ---------------------------------------------------------------------------
# 4. Result without bracket id → matched to latest unmatched call
# ---------------------------------------------------------------------------

def test_result_without_bracket_id_matches_latest_unmatched():
    trace = from_subgraph_thoughts([
        'Tool call [c1] analyze_item with arguments: {"item_id": "demo-item"}',
        'Tool response: done',
    ])
    assert len(trace.tool_results) == 1
    tr = trace.tool_results[0]
    assert tr.call_id == "c1"
    assert tr.tool_name == "analyze_item"


# ---------------------------------------------------------------------------
# 5. Malformed JSON args → {} (no raise)
# ---------------------------------------------------------------------------

def test_malformed_json_args_falls_back_to_empty_dict():
    trace = from_subgraph_thoughts([
        'Tool call [c1] analyze_item with arguments: {this is not valid json}',
    ])
    assert len(trace.tool_calls) == 1
    assert trace.tool_calls[0].args == {}


# ---------------------------------------------------------------------------
# 6. Unknown/freeform line → MessageEvent only (no ToolCall/ToolResult added)
# ---------------------------------------------------------------------------

def test_freeform_line_produces_only_message_event():
    line = "Thinking about how to approach the demo-item analysis..."
    trace = from_subgraph_thoughts([line])
    assert len(trace.messages) == 1
    assert trace.messages[0].content == line
    assert trace.messages[0].role == "assistant"
    assert trace.tool_calls == []
    assert trace.tool_results == []


# ---------------------------------------------------------------------------
# 7. metadata includes _source="subgraph_thoughts"
# ---------------------------------------------------------------------------

def test_metadata_source_is_subgraph_thoughts():
    trace = from_subgraph_thoughts([], metadata={"env": "test"})
    assert trace.metadata["_source"] == "subgraph_thoughts"
    assert trace.metadata["env"] == "test"


def test_metadata_source_set_even_without_caller_metadata():
    trace = from_subgraph_thoughts([])
    assert trace.metadata["_source"] == "subgraph_thoughts"


# ---------------------------------------------------------------------------
# 8. All timestamps are timezone-aware
# ---------------------------------------------------------------------------

def test_all_timestamps_are_timezone_aware():
    trace = from_subgraph_thoughts([
        'Tool call [c1] analyze_item with arguments: {"item_id": "demo-item"}',
        'Tool result [c1]: {"status": "ok"}',
        "Some freeform thought",
    ])
    for msg in trace.messages:
        assert msg.timestamp.tzinfo is not None, f"message missing tzinfo: {msg}"
    for tc in trace.tool_calls:
        assert tc.timestamp.tzinfo is not None
    for tr in trace.tool_results:
        assert tr.timestamp.tzinfo is not None
    assert trace.started_at.tzinfo is not None


def test_provided_started_at_is_used_and_timezone_aware():
    ts = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    trace = from_subgraph_thoughts([], started_at=ts)
    assert trace.started_at == ts
    assert trace.started_at.tzinfo is not None


# ---------------------------------------------------------------------------
# 9. No routing_events or agent_calls inferred
# ---------------------------------------------------------------------------

def test_no_routing_events_or_agent_calls_inferred():
    trace = from_subgraph_thoughts([
        'Tool call [c1] analyze_item with arguments: {"item_id": "demo-item"}',
        'Tool result [c1]: {"status": "ok"}',
        "Routing to analyst node",
        "Calling sub-agent analyst",
    ])
    assert trace.routing_events == []
    assert trace.agent_calls == []


# ---------------------------------------------------------------------------
# 10. Empty input → valid ExecutionTrace with empty lists
# ---------------------------------------------------------------------------

def test_empty_input_produces_valid_trace():
    trace = from_subgraph_thoughts([])
    assert isinstance(trace, ExecutionTrace)
    assert trace.messages == []
    assert trace.tool_calls == []
    assert trace.tool_results == []
    assert trace.routing_events == []
    assert trace.agent_calls == []
    assert trace.run_id == "subgraph-run"


# ---------------------------------------------------------------------------
# Additional: every thought line is preserved as a message
# ---------------------------------------------------------------------------

def test_every_line_preserved_as_message():
    lines = [
        'Tool call [c1] analyze_item with arguments: {"item_id": "demo-item"}',
        'Tool result [c1]: {"status": "ok"}',
        "Some other thought",
    ]
    trace = from_subgraph_thoughts(lines)
    assert len(trace.messages) == 3
    for i, line in enumerate(lines):
        assert trace.messages[i].content == line


def test_agent_name_propagated_to_messages_and_tool_calls():
    trace = from_subgraph_thoughts(
        ['Tool call [c1] analyze_item with arguments: {"item_id": "demo-item"}'],
        agent_name="analyst",
    )
    assert trace.messages[0].agent_name == "analyst"
    assert trace.tool_calls[0].agent_name == "analyst"


def test_run_id_propagated():
    trace = from_subgraph_thoughts([], run_id="my-subgraph-run")
    assert trace.run_id == "my-subgraph-run"


def test_calling_tool_pattern_parsed():
    trace = from_subgraph_thoughts([
        'Calling tool analyze_item with args {"item_id": "demo-item"}',
    ])
    assert len(trace.tool_calls) == 1
    assert trace.tool_calls[0].tool_name == "analyze_item"
    assert trace.tool_calls[0].args == {"item_id": "demo-item"}


def test_tool_output_pattern_parsed():
    trace = from_subgraph_thoughts([
        'Tool call [c1] analyze_item with arguments: {"item_id": "demo-item"}',
        "Tool output [c1]: done",
    ])
    assert len(trace.tool_results) == 1
    assert trace.tool_results[0].call_id == "c1"
    assert trace.tool_results[0].output == "done"


def test_result_without_pending_call_uses_unknown_tool():
    """Result with a bracket id that has no matching call still produces a ToolResult."""
    trace = from_subgraph_thoughts([
        "Tool result [x99]: something",
    ])
    # result is created but call_id comes from bracket and tool_name falls back
    assert len(trace.tool_results) == 1
    assert trace.tool_results[0].call_id == "x99"
    assert trace.tool_results[0].tool_name == "unknown_tool"
