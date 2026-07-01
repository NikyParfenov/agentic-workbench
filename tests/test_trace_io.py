"""Tests for trace JSON export/import and offline replay."""

import json
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import pytest
from pathlib import Path

from conftest import make_trace, make_tool_call, make_routing_event, make_artifact
from agent_runtime_validator import (
    ExecutionTrace,
    TraceBuilder,
    trace_to_json,
    trace_from_json,
    save_trace,
    load_trace,
    replay,
    replay_async,
    RuntimeValidator,
)
from agent_runtime_validator.triggers import MaxRoutesTrigger


# ---------------------------------------------------------------------------
# trace_to_json / trace_from_json
# ---------------------------------------------------------------------------

def test_trace_to_json_returns_string():
    trace = make_trace()
    result = trace_to_json(trace)
    assert isinstance(result, str)


def test_trace_to_json_is_valid_json():
    trace = make_trace()
    result = trace_to_json(trace)
    parsed = json.loads(result)
    assert parsed["run_id"] == "test-run"


def test_trace_to_json_default_indent():
    trace = make_trace()
    result = trace_to_json(trace)
    assert "\n" in result  # pretty-printed


def test_trace_from_json_roundtrip():
    original = (
        TraceBuilder(run_id="round-trip")
        .record_tool_call("search", call_id="c1", args={"q": "acme"})
        .record_routing("a", "b", reason="delegate")
        .build()
    )
    json_str = trace_to_json(original)
    restored = trace_from_json(json_str)
    assert restored.run_id == original.run_id
    assert restored.started_at == original.started_at
    assert len(restored.tool_calls) == 1
    assert restored.tool_calls[0].tool_name == "search"
    assert len(restored.routing_events) == 1


def test_trace_from_json_returns_execution_trace():
    trace = make_trace()
    restored = trace_from_json(trace_to_json(trace))
    assert isinstance(restored, ExecutionTrace)


def test_trace_from_json_invalid_raises():
    with pytest.raises(Exception):
        trace_from_json("not valid json")


def test_trace_roundtrip_preserves_metadata():
    trace = make_trace()
    trace.metadata["_budget_key"] = 3
    restored = trace_from_json(trace_to_json(trace))
    assert restored.metadata["_budget_key"] == 3


def test_trace_roundtrip_preserves_token_usage():
    trace = TraceBuilder(run_id="r1").set_token_usage(1234).build()
    restored = trace_from_json(trace_to_json(trace))
    assert restored.token_usage == 1234


# ---------------------------------------------------------------------------
# save_trace / load_trace
# ---------------------------------------------------------------------------

def test_save_and_load_roundtrip(tmp_path: Path):
    trace = (
        TraceBuilder(run_id="file-test")
        .record_message("user", "hello")
        .record_tool_call("search", call_id="c1")
        .build()
    )
    dest = tmp_path / "trace.json"
    save_trace(trace, dest)
    assert dest.exists()
    loaded = load_trace(dest)
    assert loaded.run_id == "file-test"
    assert len(loaded.messages) == 1
    assert len(loaded.tool_calls) == 1


def test_save_trace_creates_parent_dirs(tmp_path: Path):
    trace = make_trace()
    dest = tmp_path / "nested" / "dir" / "trace.json"
    save_trace(trace, dest)
    assert dest.exists()


def test_save_trace_accepts_string_path(tmp_path: Path):
    trace = make_trace()
    dest = str(tmp_path / "trace.json")
    save_trace(trace, dest)
    assert Path(dest).exists()


def test_load_trace_accepts_string_path(tmp_path: Path):
    trace = make_trace()
    dest = tmp_path / "trace.json"
    save_trace(trace, dest)
    loaded = load_trace(str(dest))
    assert isinstance(loaded, ExecutionTrace)


def test_load_trace_file_not_found(tmp_path: Path):
    with pytest.raises(Exception):
        load_trace(tmp_path / "nonexistent.json")


def test_save_trace_writes_json(tmp_path: Path):
    trace = make_trace()
    dest = tmp_path / "trace.json"
    save_trace(trace, dest)
    raw = json.loads(dest.read_text())
    assert "run_id" in raw


# ---------------------------------------------------------------------------
# replay / replay_async
# ---------------------------------------------------------------------------

def test_replay_returns_validation_decision():
    from agent_runtime_validator.schema.decisions import ValidationDecision
    trace = make_trace(routing_events=[make_routing_event("A", "B")])
    validator = RuntimeValidator(triggers=[MaxRoutesTrigger(max_routes=1)])
    decision = replay(trace, validator)
    assert isinstance(decision, ValidationDecision)
    assert decision.action == "interrupt"


def test_replay_no_triggers_fires_continue():
    trace = make_trace()
    validator = RuntimeValidator(triggers=[MaxRoutesTrigger(max_routes=100)])
    decision = replay(trace, validator)
    assert decision.action == "continue"


def test_replay_on_loaded_trace(tmp_path: Path):
    original = (
        TraceBuilder(run_id="offline")
        .record_routing("A", "B")
        .build()
    )
    dest = tmp_path / "trace.json"
    save_trace(original, dest)

    loaded = load_trace(dest)
    validator = RuntimeValidator(triggers=[MaxRoutesTrigger(max_routes=1)])
    decision = replay(loaded, validator)
    assert decision.triggered_by == ["MaxRoutesTrigger"]


async def test_replay_async_returns_decision():
    trace = make_trace(routing_events=[make_routing_event("A", "B")])
    validator = RuntimeValidator(triggers=[MaxRoutesTrigger(max_routes=1)])
    decision = await replay_async(trace, validator)
    assert decision.action == "interrupt"


# ---------------------------------------------------------------------------
# Top-level imports
# ---------------------------------------------------------------------------

def test_all_symbols_importable():
    from agent_runtime_validator import (
        trace_to_json, trace_from_json,
        save_trace, load_trace,
        replay, replay_async,
    )
    assert all(callable(f) for f in [
        trace_to_json, trace_from_json,
        save_trace, load_trace,
        replay, replay_async,
    ])
