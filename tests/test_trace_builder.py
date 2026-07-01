"""Tests for TraceBuilder."""

from datetime import datetime, timezone

import pytest

from agent_runtime_validator import ExecutionTrace, TraceBuilder


def _ts() -> datetime:
    return datetime.now(timezone.utc)


def _base_trace(**kwargs) -> ExecutionTrace:
    return ExecutionTrace(run_id="run-1", started_at=_ts(), **kwargs)


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------

def test_build_returns_execution_trace():
    trace = TraceBuilder(run_id="r1").build()
    assert isinstance(trace, ExecutionTrace)
    assert trace.run_id == "r1"


def test_build_uses_explicit_started_at():
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    trace = TraceBuilder(run_id="r1", started_at=ts).build()
    assert trace.started_at == ts


def test_build_all_lists_empty_by_default():
    trace = TraceBuilder(run_id="r1").build()
    assert trace.messages == []
    assert trace.tool_calls == []
    assert trace.tool_results == []
    assert trace.routing_events == []
    assert trace.agent_calls == []
    assert trace.artifacts == []
    assert trace.errors == []


def test_build_is_idempotent():
    builder = TraceBuilder(run_id="r1")
    builder.record_message("user", "hello")
    t1 = builder.build()
    builder.record_message("assistant", "world")
    t2 = builder.build()
    # First snapshot is not mutated
    assert len(t1.messages) == 1
    assert len(t2.messages) == 2


# ---------------------------------------------------------------------------
# Individual record methods
# ---------------------------------------------------------------------------

def test_record_message_appends():
    trace = TraceBuilder(run_id="r1").record_message("user", "hi").build()
    assert len(trace.messages) == 1
    assert trace.messages[0].role == "user"
    assert trace.messages[0].content == "hi"


def test_record_tool_call_appends():
    trace = (
        TraceBuilder(run_id="r1")
        .record_tool_call("search", call_id="c1", args={"query": "demo-topic"})
        .build()
    )
    assert len(trace.tool_calls) == 1
    tc = trace.tool_calls[0]
    assert tc.tool_name == "search"
    assert tc.call_id == "c1"
    assert tc.args == {"query": "demo-topic"}


def test_record_tool_result_appends():
    trace = (
        TraceBuilder(run_id="r1")
        .record_tool_result("c1", "search", output="found result")
        .build()
    )
    assert len(trace.tool_results) == 1
    assert trace.tool_results[0].output == "found result"


def test_record_routing_appends():
    trace = (
        TraceBuilder(run_id="r1")
        .record_routing("supervisor", "researcher", reason="delegate")
        .build()
    )
    assert len(trace.routing_events) == 1
    r = trace.routing_events[0]
    assert r.from_agent == "supervisor"
    assert r.to_agent == "researcher"
    assert r.reason == "delegate"


def test_record_artifact_appends():
    trace = (
        TraceBuilder(run_id="r1")
        .record_artifact("a1", "report", content="results here", agent_name="writer")
        .build()
    )
    assert len(trace.artifacts) == 1
    a = trace.artifacts[0]
    assert a.artifact_id == "a1"
    assert a.artifact_type == "report"
    assert a.agent_name == "writer"


def test_record_error_appends():
    trace = (
        TraceBuilder(run_id="r1")
        .record_error("ValueError", "something went wrong", agent_name="analyzer")
        .build()
    )
    assert len(trace.errors) == 1
    e = trace.errors[0]
    assert e.error_type == "ValueError"
    assert e.message == "something went wrong"


def test_record_agent_call_appends():
    trace = (
        TraceBuilder(run_id="r1")
        .record_agent_call("sup", "sub", input="do this", output="done")
        .build()
    )
    assert len(trace.agent_calls) == 1
    ac = trace.agent_calls[0]
    assert ac.caller == "sup"
    assert ac.callee == "sub"


# ---------------------------------------------------------------------------
# Fluent chaining — all record_* methods return self
# ---------------------------------------------------------------------------

def test_fluent_chaining():
    trace = (
        TraceBuilder(run_id="r1")
        .record_message("user", "start")
        .record_tool_call("search", call_id="c1")
        .record_tool_result("c1", "search", output="ok")
        .record_routing("a", "b")
        .record_artifact("art1", "text", "content")
        .record_error("E", "oops")
        .record_agent_call("sup", "sub", input="go")
        .build()
    )
    assert len(trace.messages) == 1
    assert len(trace.tool_calls) == 1
    assert len(trace.tool_results) == 1
    assert len(trace.routing_events) == 1
    assert len(trace.artifacts) == 1
    assert len(trace.errors) == 1
    assert len(trace.agent_calls) == 1


# ---------------------------------------------------------------------------
# Auto-timestamp
# ---------------------------------------------------------------------------

def test_record_message_auto_timestamps():
    trace = TraceBuilder(run_id="r1").record_message("user", "hi").build()
    assert trace.messages[0].timestamp is not None
    assert trace.messages[0].timestamp.tzinfo is not None


# ---------------------------------------------------------------------------
# from_trace
# ---------------------------------------------------------------------------

def test_from_trace_roundtrip():
    original = (
        TraceBuilder(run_id="run-orig")
        .record_message("user", "original message")
        .record_tool_call("search", call_id="c1")
        .build()
    )
    rebuilt = TraceBuilder.from_trace(original).build()
    assert rebuilt.run_id == original.run_id
    assert rebuilt.started_at == original.started_at
    assert len(rebuilt.messages) == 1
    assert len(rebuilt.tool_calls) == 1


def test_from_trace_does_not_share_lists():
    original = (
        TraceBuilder(run_id="r1")
        .record_message("user", "hello")
        .build()
    )
    builder2 = TraceBuilder.from_trace(original)
    builder2.record_message("assistant", "world")
    # original trace is unaffected
    assert len(original.messages) == 1


# ---------------------------------------------------------------------------
# merge_trace
# ---------------------------------------------------------------------------

def test_merge_trace_appends_events():
    child = (
        TraceBuilder(run_id="child")
        .record_tool_call("db", call_id="c2")
        .build()
    )
    trace = (
        TraceBuilder(run_id="parent")
        .record_tool_call("search", call_id="c1")
        .merge_trace(child)
        .build()
    )
    assert len(trace.tool_calls) == 2
    assert trace.run_id == "parent"  # parent run_id preserved


def test_merge_trace_sums_token_usage():
    builder = TraceBuilder(run_id="r1")
    builder.set_token_usage(100)
    child = ExecutionTrace(run_id="c", started_at=_ts(), token_usage=50)
    builder.merge_trace(child)
    assert builder.build().token_usage == 150


def test_merge_trace_child_metadata_wins():
    builder = TraceBuilder(run_id="r1")
    builder.update_metadata(key="parent_value")
    child = ExecutionTrace(run_id="c", started_at=_ts(), metadata={"key": "child_value"})
    builder.merge_trace(child)
    assert builder.build().metadata["key"] == "child_value"


# ---------------------------------------------------------------------------
# merge classmethod
# ---------------------------------------------------------------------------

def test_merge_classmethod_preserves_parent_run_id():
    parent = TraceBuilder(run_id="parent").record_message("user", "p").build()
    child = TraceBuilder(run_id="child").record_tool_call("x", call_id="c1").build()
    merged = TraceBuilder.merge(parent, child).build()
    assert merged.run_id == "parent"
    assert len(merged.messages) == 1
    assert len(merged.tool_calls) == 1


def test_merge_classmethod_preserves_parent_started_at():
    ts_parent = datetime(2024, 1, 1, tzinfo=timezone.utc)
    ts_child = datetime(2024, 6, 1, tzinfo=timezone.utc)
    parent = TraceBuilder(run_id="p", started_at=ts_parent).build()
    child = TraceBuilder(run_id="c", started_at=ts_child).build()
    merged = TraceBuilder.merge(parent, child).build()
    assert merged.started_at == ts_parent


# ---------------------------------------------------------------------------
# Misc builder helpers
# ---------------------------------------------------------------------------

def test_set_finished_at():
    ts = datetime(2024, 12, 31, tzinfo=timezone.utc)
    trace = TraceBuilder(run_id="r1").set_finished_at(ts).build()
    assert trace.finished_at == ts


def test_set_token_usage():
    trace = TraceBuilder(run_id="r1").set_token_usage(999).build()
    assert trace.token_usage == 999


def test_update_metadata():
    trace = TraceBuilder(run_id="r1").update_metadata(foo="bar", num=42).build()
    assert trace.metadata["foo"] == "bar"
    assert trace.metadata["num"] == 42
