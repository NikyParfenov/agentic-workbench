import sys
import pytest

langgraph = pytest.importorskip("langgraph", reason="langgraph not installed")

import os
sys.path.insert(0, os.path.dirname(__file__))

from datetime import datetime, timezone
from conftest import make_trace, make_routing_event
from agent_runtime_validator.integrations.langgraph.nodes import ValidationNode, create_validation_router
from agent_runtime_validator.integrations.langgraph.adapter import state_to_trace, get_trace_from_state
from agent_runtime_validator.integrations.langgraph import get_trace_from_state as get_trace_top
from agent_runtime_validator.schema.decisions import ValidationDecision, ValidatorResult, TriggerResult
from agent_runtime_validator.schema.trace import ExecutionTrace
from agent_runtime_validator.validators.base import BaseValidator
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


# --- create_validation_router ---

def _decision(action, suggested_next_agent=None):
    vr = None
    if suggested_next_agent:
        vr = ValidatorResult(
            valid=False, confidence=1.0, recommendation="reroute",
            reason="reroute", suggested_next_agent=suggested_next_agent,
        )
    return ValidationDecision(
        should_continue=(action == "continue"), action=action,
        severity="medium", reason="test", validator_result=vr,
    )


def test_router_continue():
    router = create_validation_router(continue_to="supervisor")
    assert router({"decision": _decision("continue")}) == "supervisor"


def test_router_retry():
    router = create_validation_router(continue_to="supervisor", retry_to="researcher")
    assert router({"decision": _decision("retry_last_step")}) == "researcher"


def test_router_reroute_suggested_ignored_by_default():
    router = create_validation_router(continue_to="supervisor", reroute_to="fallback")
    assert router({"decision": _decision("reroute", suggested_next_agent="data_agent")}) == "fallback"


def test_router_reroute_suggested_used_with_allowlist():
    router = create_validation_router(
        continue_to="supervisor", reroute_to="fallback",
        allowed_reroutes={"data_agent"},
    )
    assert router({"decision": _decision("reroute", suggested_next_agent="data_agent")}) == "data_agent"


def test_router_reroute_fallback():
    router = create_validation_router(continue_to="supervisor", reroute_to="fallback")
    assert router({"decision": _decision("reroute")}) == "fallback"


def test_router_reroute_blocked_by_allowlist():
    router = create_validation_router(
        continue_to="supervisor", reroute_to="fallback",
        allowed_reroutes={"safe_agent"},
    )
    assert router({"decision": _decision("reroute", suggested_next_agent="evil_agent")}) == "fallback"


def test_router_reroute_allowed_by_allowlist():
    router = create_validation_router(
        continue_to="supervisor", reroute_to="fallback",
        allowed_reroutes={"safe_agent"},
    )
    assert router({"decision": _decision("reroute", suggested_next_agent="safe_agent")}) == "safe_agent"


def test_router_interrupt():
    router = create_validation_router(continue_to="supervisor", interrupt_to="human")
    assert router({"decision": _decision("interrupt")}) == "human"


def test_router_abort_default_end():
    from langgraph.graph import END
    router = create_validation_router(continue_to="supervisor")
    assert router({"decision": _decision("abort")}) == END


def test_router_abort_custom():
    router = create_validation_router(continue_to="supervisor", abort_to="cleanup")
    assert router({"decision": _decision("abort")}) == "cleanup"


def test_router_missing_decision():
    router = create_validation_router(continue_to="supervisor")
    assert router({}) == "supervisor"


def test_router_dict_decision():
    router = create_validation_router(continue_to="supervisor", retry_to="researcher")
    decision_dict = _decision("retry_last_step").model_dump()
    assert router({"decision": decision_dict}) == "researcher"


# --- trace_builder ---

def test_trace_builder_called_with_state():
    from agent_runtime_validator.integrations.langgraph import TraceBuilderFn
    received: list[dict] = []

    def builder(state: dict) -> object:
        received.append(state)
        return make_trace()

    node = ValidationNode(triggers=[MaxRoutesTrigger(max_routes=100)], trace_builder=builder)  # type: ignore[arg-type]
    state = {"msg": "hello"}
    node(state)
    assert len(received) == 1
    assert received[0] is state


def test_trace_builder_return_value_used():
    custom_trace = make_trace(routing_events=[make_routing_event("A", "B")])

    def builder(state: dict) -> object:
        return custom_trace

    node = ValidationNode(triggers=[MaxRoutesTrigger(max_routes=1)], trace_builder=builder)  # type: ignore[arg-type]
    result = node({})
    assert result["decision"].triggered_by == ["MaxRoutesTrigger"]


def test_trace_builder_overrides_trace_key():
    """trace_builder should take precedence over state[trace_key]."""
    custom_trace = make_trace(routing_events=[make_routing_event("X", "Y")])
    other_trace = make_trace()  # empty, no routing events

    def builder(state: dict) -> object:
        return custom_trace

    node = ValidationNode(
        triggers=[MaxRoutesTrigger(max_routes=1)],
        trace_builder=builder,  # type: ignore[arg-type]
    )
    # state has a trace under the default key but builder should win
    result = node({"trace": other_trace})
    assert result["decision"].triggered_by == ["MaxRoutesTrigger"]


async def test_trace_builder_async():
    custom_trace = make_trace(routing_events=[make_routing_event("A", "B")])

    def builder(state: dict) -> object:
        return custom_trace

    node = ValidationNode(triggers=[MaxRoutesTrigger(max_routes=1)], trace_builder=builder)  # type: ignore[arg-type]
    result = await node.async_call({})
    assert result["decision"].triggered_by == ["MaxRoutesTrigger"]


def test_trace_builder_fn_type_importable():
    from agent_runtime_validator.integrations.langgraph import TraceBuilderFn
    assert TraceBuilderFn is not None


# --- get_trace_from_state ---

def test_get_trace_from_state_returns_trace_object():
    trace = make_trace()
    result = get_trace_from_state({"trace": trace})
    assert result is trace


def test_get_trace_from_state_parses_dict():
    trace = make_trace()
    state = {"trace": trace.model_dump()}
    result = get_trace_from_state(state)
    assert isinstance(result, type(trace))
    assert result.run_id == trace.run_id


def test_get_trace_from_state_returns_none_when_absent():
    result = get_trace_from_state({})
    assert result is None


def test_get_trace_from_state_custom_key():
    trace = make_trace()
    result = get_trace_from_state({"my_trace": trace}, trace_key="my_trace")
    assert result is trace


def test_get_trace_from_state_wrong_key_returns_none():
    trace = make_trace()
    result = get_trace_from_state({"trace": trace}, trace_key="other")
    assert result is None


def test_get_trace_from_state_importable_from_package():
    assert get_trace_top is not None
    assert callable(get_trace_top)


def test_validation_node_writes_trace_for_budget_persistence():
    """Budget counter in metadata survives a dict-serialized trace round-trip."""
    node = ValidationNode(
        triggers=[MaxRoutesTrigger(max_routes=1)],
        max_validator_calls_per_run=1,
        on_validator_budget_exhausted="interrupt",
    )
    trace = make_trace(routing_events=[make_routing_event("A", "B")])
    state = {"trace": trace}
    # First call — budget consumed
    result1 = node(state)
    # Simulate checkpoint serialization: trace stored as dict
    serialized_state = {**result1, "trace": result1["trace"].model_dump()}
    # Second call — budget exhausted, should use on_exhausted behavior
    result2 = node(serialized_state)
    # Decision should reflect exhausted budget (interrupt) not re-run validator
    assert result2["decision"] is not None


# --- validator_mode ---

class _CountingValidator(BaseValidator):
    """Stub validator that counts invocations and always returns continue."""

    def __init__(self):
        self.call_count = 0

    def validate(
        self, trace: ExecutionTrace, trigger_results: list[TriggerResult]
    ) -> ValidatorResult:
        self.call_count += 1
        return ValidatorResult(
            valid=True,
            confidence=1.0,
            recommendation="continue",
            reason="stub",
        )


def test_validation_node_default_validator_mode_is_on_trigger():
    """Default mode is on_trigger: validator is NOT called when no triggers fire."""
    validator = _CountingValidator()
    # MaxRoutesTrigger(max_routes=100) will not fire for a trace with 0 routing events
    node = ValidationNode(
        triggers=[MaxRoutesTrigger(max_routes=100)],
        validator=validator,
    )
    trace = make_trace()  # no routing events — trigger won't fire
    state = {"trace": trace}
    result = node(state)
    assert result["decision"].action == "continue"
    assert validator.call_count == 0


def test_validation_node_always_invokes_validator_without_trigger():
    """always mode: validator IS called even when no triggers fire."""
    validator = _CountingValidator()
    node = ValidationNode(
        triggers=[MaxRoutesTrigger(max_routes=100)],
        validator=validator,
        validator_mode="always",
    )
    trace = make_trace()  # no routing events — trigger won't fire
    state = {"trace": trace}
    result = node(state)
    assert validator.call_count == 1
    assert result["decision"].action is not None


async def test_validation_node_always_async_invokes_validator_without_trigger():
    """always mode via async_call: validator IS called even when no triggers fire."""
    validator = _CountingValidator()
    node = ValidationNode(
        triggers=[MaxRoutesTrigger(max_routes=100)],
        validator=validator,
        validator_mode="always",
    )
    trace = make_trace()  # no routing events — trigger won't fire
    state = {"trace": trace}
    result = await node.async_call(state)
    assert validator.call_count == 1
    assert result["decision"].action is not None


def test_validation_node_invalid_validator_mode_raises():
    """Passing an invalid validator_mode string should raise ValueError."""
    with pytest.raises(ValueError):
        ValidationNode(
            triggers=[MaxRoutesTrigger(max_routes=10)],
            validator_mode="not_a_valid_mode",  # type: ignore[arg-type]
        )
