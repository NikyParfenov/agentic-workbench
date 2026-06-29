import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from datetime import datetime, timezone, timedelta
from conftest import make_trace, make_tool_call, make_routing_event, make_tool_result
from agent_runtime_validator.triggers.max_calls import MaxToolCallsTrigger
from agent_runtime_validator.triggers.max_routes import MaxRoutesTrigger
from agent_runtime_validator.triggers.max_context_tokens import MaxContextTokensTrigger
from agent_runtime_validator.triggers.max_execution_time import MaxExecutionTimeTrigger
from agent_runtime_validator.triggers.same_tool_loop import SameToolLoopTrigger
from agent_runtime_validator.triggers.same_tool_same_args_loop import SameToolSameArgsLoopTrigger
from agent_runtime_validator.triggers.agent_pingpong import AgentPingPongTrigger
from agent_runtime_validator.triggers.no_progress import NoProgressTrigger
from agent_runtime_validator.triggers.tool_error_rate import ToolErrorRateTrigger
from agent_runtime_validator.schema.events import MessageEvent


def test_max_tool_calls_not_triggered_below_limit():
    trace = make_trace(tool_calls=[make_tool_call("search", "c1"), make_tool_call("search", "c2")])
    trigger = MaxToolCallsTrigger(max_calls=3)
    result = trigger.evaluate(trace)
    assert result.triggered is False
    assert result.trigger_name == "MaxToolCallsTrigger"


def test_max_tool_calls_triggered_at_limit():
    trace = make_trace(tool_calls=[make_tool_call("t", f"c{i}") for i in range(3)])
    trigger = MaxToolCallsTrigger(max_calls=3)
    result = trigger.evaluate(trace)
    assert result.triggered is True
    assert result.severity == "high"


def test_max_tool_calls_triggered_above_limit():
    trace = make_trace(tool_calls=[make_tool_call("t", f"c{i}") for i in range(5)])
    trigger = MaxToolCallsTrigger(max_calls=3)
    result = trigger.evaluate(trace)
    assert result.triggered is True


def test_max_tool_calls_custom_severity():
    trace = make_trace(tool_calls=[make_tool_call("t", "c1")])
    trigger = MaxToolCallsTrigger(max_calls=1, severity="critical")
    result = trigger.evaluate(trace)
    assert result.triggered is True
    assert result.severity == "critical"


def test_max_tool_calls_evidence():
    trace = make_trace(tool_calls=[make_tool_call("t", "c1"), make_tool_call("t", "c2")])
    trigger = MaxToolCallsTrigger(max_calls=5)
    result = trigger.evaluate(trace)
    assert result.evidence["count"] == 2
    assert result.evidence["max_calls"] == 5


def test_max_routes_not_triggered():
    trace = make_trace(routing_events=[make_routing_event("A", "B")])
    trigger = MaxRoutesTrigger(max_routes=3)
    result = trigger.evaluate(trace)
    assert result.triggered is False


def test_max_routes_triggered_at_limit():
    trace = make_trace(routing_events=[make_routing_event("A", "B") for _ in range(3)])
    trigger = MaxRoutesTrigger(max_routes=3)
    result = trigger.evaluate(trace)
    assert result.triggered is True
    assert result.severity == "high"


def test_max_routes_triggered_above_limit():
    trace = make_trace(routing_events=[make_routing_event("A", "B") for _ in range(5)])
    trigger = MaxRoutesTrigger(max_routes=3)
    result = trigger.evaluate(trace)
    assert result.triggered is True


def test_max_routes_custom_severity():
    trace = make_trace(routing_events=[make_routing_event("A", "B")])
    trigger = MaxRoutesTrigger(max_routes=1, severity="medium")
    result = trigger.evaluate(trace)
    assert result.triggered is True
    assert result.severity == "medium"


def test_max_routes_trigger_name():
    trace = make_trace()
    trigger = MaxRoutesTrigger(max_routes=1)
    result = trigger.evaluate(trace)
    assert result.trigger_name == "MaxRoutesTrigger"


# MaxContextTokensTrigger

def test_max_context_tokens_uses_token_usage_field():
    trace = make_trace(token_usage=500)
    trigger = MaxContextTokensTrigger(max_tokens=1000)
    result = trigger.evaluate(trace)
    assert result.triggered is False
    assert result.evidence["estimated_tokens"] == 500


def test_max_context_tokens_triggered_at_limit():
    trace = make_trace(token_usage=1000)
    trigger = MaxContextTokensTrigger(max_tokens=1000)
    result = trigger.evaluate(trace)
    assert result.triggered is True
    assert result.severity == "high"


def test_max_context_tokens_estimates_from_messages():
    msg = MessageEvent(
        role="user",
        content="a" * 400,
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    trace = make_trace(messages=[msg])
    trigger = MaxContextTokensTrigger(max_tokens=50)
    result = trigger.evaluate(trace)
    # 400 chars // 4 = 100 tokens → triggered
    assert result.triggered is True


def test_max_context_tokens_not_triggered_below_limit():
    trace = make_trace(token_usage=99)
    trigger = MaxContextTokensTrigger(max_tokens=100)
    result = trigger.evaluate(trace)
    assert result.triggered is False


def test_max_context_tokens_trigger_name():
    trace = make_trace(token_usage=0)
    trigger = MaxContextTokensTrigger(max_tokens=100)
    result = trigger.evaluate(trace)
    assert result.trigger_name == "MaxContextTokensTrigger"


# MaxExecutionTimeTrigger

def test_max_execution_time_uses_finished_at():
    started = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    finished = datetime(2024, 1, 1, 0, 0, 10, tzinfo=timezone.utc)
    trace = make_trace(started_at=started, finished_at=finished)
    trigger = MaxExecutionTimeTrigger(max_seconds=5.0)
    result = trigger.evaluate(trace)
    assert result.triggered is True
    assert result.evidence["elapsed_seconds"] == 10.0


def test_max_execution_time_not_triggered():
    started = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    finished = datetime(2024, 1, 1, 0, 0, 3, tzinfo=timezone.utc)
    trace = make_trace(started_at=started, finished_at=finished)
    trigger = MaxExecutionTimeTrigger(max_seconds=5.0)
    result = trigger.evaluate(trace)
    assert result.triggered is False


def test_max_execution_time_triggered_at_limit():
    started = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    finished = datetime(2024, 1, 1, 0, 0, 5, tzinfo=timezone.utc)
    trace = make_trace(started_at=started, finished_at=finished)
    trigger = MaxExecutionTimeTrigger(max_seconds=5.0)
    result = trigger.evaluate(trace)
    assert result.triggered is True


def test_max_execution_time_falls_back_to_now():
    trace = make_trace(started_at=datetime.now(timezone.utc) - timedelta(seconds=60))
    trigger = MaxExecutionTimeTrigger(max_seconds=30.0)
    result = trigger.evaluate(trace)
    assert result.triggered is True


def test_max_execution_time_trigger_name():
    started = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    finished = datetime(2024, 1, 1, 0, 0, 1, tzinfo=timezone.utc)
    trace = make_trace(started_at=started, finished_at=finished)
    trigger = MaxExecutionTimeTrigger(max_seconds=10.0)
    result = trigger.evaluate(trace)
    assert result.trigger_name == "MaxExecutionTimeTrigger"


# SameToolLoopTrigger

def test_same_tool_loop_not_triggered():
    trace = make_trace(tool_calls=[
        make_tool_call("search", "c1"),
        make_tool_call("fetch", "c2"),
    ])
    trigger = SameToolLoopTrigger(max_repeats=3)
    result = trigger.evaluate(trace)
    assert result.triggered is False


def test_same_tool_loop_triggered_at_limit():
    trace = make_trace(tool_calls=[make_tool_call("search", f"c{i}") for i in range(3)])
    trigger = SameToolLoopTrigger(max_repeats=3)
    result = trigger.evaluate(trace)
    assert result.triggered is True
    assert result.severity == "medium"
    assert result.evidence["tool"] == "search"


def test_same_tool_loop_triggered_above_limit():
    trace = make_trace(tool_calls=[make_tool_call("search", f"c{i}") for i in range(5)])
    trigger = SameToolLoopTrigger(max_repeats=3)
    result = trigger.evaluate(trace)
    assert result.triggered is True


def test_same_tool_loop_picks_most_frequent():
    trace = make_trace(tool_calls=[
        make_tool_call("search", "c1"),
        make_tool_call("fetch", "c2"),
        make_tool_call("fetch", "c3"),
        make_tool_call("fetch", "c4"),
    ])
    trigger = SameToolLoopTrigger(max_repeats=3)
    result = trigger.evaluate(trace)
    assert result.triggered is True
    assert result.evidence["tool"] == "fetch"


def test_same_tool_loop_empty_trace():
    trace = make_trace()
    trigger = SameToolLoopTrigger(max_repeats=3)
    result = trigger.evaluate(trace)
    assert result.triggered is False


def test_same_tool_loop_trigger_name():
    trace = make_trace()
    trigger = SameToolLoopTrigger(max_repeats=3)
    result = trigger.evaluate(trace)
    assert result.trigger_name == "SameToolLoopTrigger"


# SameToolSameArgsLoopTrigger

def test_same_tool_same_args_not_triggered_different_args():
    trace = make_trace(tool_calls=[
        make_tool_call("search", "c1", args={"q": "cats"}),
        make_tool_call("search", "c2", args={"q": "dogs"}),
        make_tool_call("search", "c3", args={"q": "birds"}),
    ])
    trigger = SameToolSameArgsLoopTrigger(max_repeats=3)
    result = trigger.evaluate(trace)
    assert result.triggered is False


def test_same_tool_same_args_triggered():
    trace = make_trace(tool_calls=[
        make_tool_call("search", f"c{i}", args={"q": "cats"}) for i in range(3)
    ])
    trigger = SameToolSameArgsLoopTrigger(max_repeats=3)
    result = trigger.evaluate(trace)
    assert result.triggered is True
    assert result.severity == "high"
    assert result.evidence["tool"] == "search"


def test_same_tool_same_args_different_tool_not_triggered():
    trace = make_trace(tool_calls=[
        make_tool_call("search", "c1", args={"q": "cats"}),
        make_tool_call("fetch", "c2", args={"q": "cats"}),
        make_tool_call("lookup", "c3", args={"q": "cats"}),
    ])
    trigger = SameToolSameArgsLoopTrigger(max_repeats=3)
    result = trigger.evaluate(trace)
    assert result.triggered is False


def test_same_tool_same_args_trigger_name():
    trace = make_trace()
    trigger = SameToolSameArgsLoopTrigger(max_repeats=3)
    result = trigger.evaluate(trace)
    assert result.trigger_name == "SameToolSameArgsLoopTrigger"


# AgentPingPongTrigger

def test_pingpong_not_triggered_too_few_events():
    trace = make_trace(routing_events=[make_routing_event("A", "B")])
    trigger = AgentPingPongTrigger(max_cycles=2)
    result = trigger.evaluate(trace)
    assert result.triggered is False


def test_pingpong_not_triggered_no_alternation():
    trace = make_trace(routing_events=[
        make_routing_event("A", "B"),
        make_routing_event("B", "C"),
        make_routing_event("C", "D"),
    ])
    trigger = AgentPingPongTrigger(max_cycles=2)
    result = trigger.evaluate(trace)
    assert result.triggered is False


def test_pingpong_triggered_two_cycles():
    # A→B, B→A, A→B, B→A = 2 full A↔B cycles
    trace = make_trace(routing_events=[
        make_routing_event("A", "B"),
        make_routing_event("B", "A"),
        make_routing_event("A", "B"),
        make_routing_event("B", "A"),
    ])
    trigger = AgentPingPongTrigger(max_cycles=2)
    result = trigger.evaluate(trace)
    assert result.triggered is True
    assert result.severity == "high"


def test_pingpong_not_triggered_one_cycle():
    trace = make_trace(routing_events=[
        make_routing_event("A", "B"),
        make_routing_event("B", "A"),
    ])
    trigger = AgentPingPongTrigger(max_cycles=2)
    result = trigger.evaluate(trace)
    assert result.triggered is False


def test_pingpong_trigger_name():
    trace = make_trace()
    trigger = AgentPingPongTrigger(max_cycles=2)
    result = trigger.evaluate(trace)
    assert result.trigger_name == "AgentPingPongTrigger"


# NoProgressTrigger

def test_no_progress_not_triggered_few_calls():
    trace = make_trace(tool_calls=[make_tool_call("t", f"c{i}") for i in range(3)])
    trigger = NoProgressTrigger(min_tool_calls=5)
    result = trigger.evaluate(trace)
    assert result.triggered is False


def test_no_progress_not_triggered_has_artifacts():
    from conftest import make_artifact
    trace = make_trace(
        tool_calls=[make_tool_call("t", f"c{i}") for i in range(5)],
        artifacts=[make_artifact("a1", "report")],
    )
    trigger = NoProgressTrigger(min_tool_calls=5)
    result = trigger.evaluate(trace)
    assert result.triggered is False


def test_no_progress_triggered():
    trace = make_trace(tool_calls=[make_tool_call("t", f"c{i}") for i in range(5)])
    trigger = NoProgressTrigger(min_tool_calls=5)
    result = trigger.evaluate(trace)
    assert result.triggered is True
    assert result.severity == "medium"


def test_no_progress_triggered_above_threshold():
    trace = make_trace(tool_calls=[make_tool_call("t", f"c{i}") for i in range(10)])
    trigger = NoProgressTrigger(min_tool_calls=5)
    result = trigger.evaluate(trace)
    assert result.triggered is True
    assert result.evidence["tool_call_count"] == 10
    assert result.evidence["artifact_count"] == 0


def test_no_progress_trigger_name():
    trace = make_trace()
    trigger = NoProgressTrigger(min_tool_calls=5)
    result = trigger.evaluate(trace)
    assert result.trigger_name == "NoProgressTrigger"


# ToolErrorRateTrigger

def test_tool_error_rate_not_enough_results():
    trace = make_trace()
    trigger = ToolErrorRateTrigger(max_error_rate=0.5, min_results=1)
    result = trigger.evaluate(trace)
    assert result.triggered is False


def test_tool_error_rate_not_triggered_no_errors():
    trace = make_trace(tool_results=[
        make_tool_result("c1", "search", output="ok"),
        make_tool_result("c2", "search", output="ok"),
    ])
    trigger = ToolErrorRateTrigger(max_error_rate=0.5)
    result = trigger.evaluate(trace)
    assert result.triggered is False
    assert result.evidence["errors"] == 0


def test_tool_error_rate_triggered():
    trace = make_trace(tool_results=[
        make_tool_result("c1", "search", output=None, error="timeout"),
        make_tool_result("c2", "search", output=None, error="timeout"),
        make_tool_result("c3", "search", output="ok"),
        make_tool_result("c4", "search", output="ok"),
    ])
    trigger = ToolErrorRateTrigger(max_error_rate=0.5)
    result = trigger.evaluate(trace)
    assert result.triggered is True
    assert result.severity == "high"
    assert result.evidence["errors"] == 2
    assert result.evidence["error_rate"] == 0.5


def test_tool_error_rate_below_threshold():
    trace = make_trace(tool_results=[
        make_tool_result("c1", "search", output=None, error="err"),
        make_tool_result("c2", "search", output="ok"),
        make_tool_result("c3", "search", output="ok"),
        make_tool_result("c4", "search", output="ok"),
    ])
    trigger = ToolErrorRateTrigger(max_error_rate=0.5)
    result = trigger.evaluate(trace)
    assert result.triggered is False


def test_tool_error_rate_trigger_name():
    trace = make_trace(tool_results=[make_tool_result("c1", "t", output="ok")])
    trigger = ToolErrorRateTrigger(max_error_rate=1.0)
    result = trigger.evaluate(trace)
    assert result.trigger_name == "ToolErrorRateTrigger"


# Trigger exports

def test_triggers_package_exports():
    from agent_runtime_validator.triggers import (
        BaseTrigger, MaxToolCallsTrigger, MaxRoutesTrigger,
        MaxContextTokensTrigger, MaxExecutionTimeTrigger,
        SameToolLoopTrigger, SameToolSameArgsLoopTrigger,
        AgentPingPongTrigger, NoProgressTrigger, ToolErrorRateTrigger,
    )
    assert issubclass(MaxToolCallsTrigger, BaseTrigger)
    assert issubclass(ToolErrorRateTrigger, BaseTrigger)
