import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from conftest import make_trace, make_tool_call, make_routing_event
from agent_runtime_validator.runtime import RuntimeValidator
from agent_runtime_validator.triggers import MaxRoutesTrigger, SameToolLoopTrigger
from agent_runtime_validator.validators.noop import NoOpValidator
from agent_runtime_validator.validators.llm_judge import LLMJudgeValidator

_VALID_JSON = '{"valid": true, "confidence": 0.99, "issues": [], "recommendation": "continue", "reason": "ok"}'
_ABORT_JSON = '{"valid": false, "confidence": 0.95, "issues": ["loop"], "recommendation": "abort", "reason": "loop"}'


def test_runtime_no_triggers_fired_returns_continue():
    validator = RuntimeValidator(triggers=[MaxRoutesTrigger(max_routes=10)])
    trace = make_trace(routing_events=[make_routing_event("A", "B")])
    decision = validator.validate(trace)
    assert decision.action == "continue"
    assert decision.should_continue is True


def test_runtime_trigger_fires_no_validator_uses_severity():
    validator = RuntimeValidator(triggers=[MaxRoutesTrigger(max_routes=1)])
    trace = make_trace(routing_events=[make_routing_event("A", "B")])
    decision = validator.validate(trace)
    assert decision.triggered_by == ["MaxRoutesTrigger"]
    assert decision.action == "interrupt"  # high severity default


def test_runtime_trigger_fires_with_validator():
    llm = LLMJudgeValidator(model=lambda p: _ABORT_JSON)
    validator = RuntimeValidator(
        triggers=[MaxRoutesTrigger(max_routes=1)],
        validator=llm,
    )
    trace = make_trace(routing_events=[make_routing_event("A", "B")])
    decision = validator.validate(trace)
    assert decision.action == "abort"
    assert decision.validator_result is not None


def test_runtime_validator_not_called_when_no_triggers_fire():
    calls = []

    def model(prompt: str) -> str:
        calls.append(prompt)
        return _VALID_JSON

    llm = LLMJudgeValidator(model=model)
    validator = RuntimeValidator(
        triggers=[MaxRoutesTrigger(max_routes=100)],
        validator=llm,
    )
    trace = make_trace(routing_events=[make_routing_event("A", "B")])
    validator.validate(trace)
    assert calls == []


def test_runtime_noop_validator_not_invoked_even_when_triggers_fire():
    validator = RuntimeValidator(triggers=[MaxRoutesTrigger(max_routes=1)])
    trace = make_trace(routing_events=[make_routing_event("A", "B")])
    decision = validator.validate(trace)
    assert decision.validator_result is None


def test_runtime_defaults_to_noop_and_default_policy():
    validator = RuntimeValidator(triggers=[])
    assert isinstance(validator.validator, NoOpValidator)


def test_runtime_all_triggers_evaluated():
    validator = RuntimeValidator(triggers=[
        MaxRoutesTrigger(max_routes=1),
        SameToolLoopTrigger(max_repeats=3),
    ])
    trace = make_trace(routing_events=[make_routing_event("A", "B")])
    decision = validator.validate(trace)
    assert "MaxRoutesTrigger" in decision.triggered_by
    assert "SameToolLoopTrigger" not in decision.triggered_by


async def test_runtime_async_validate():
    async def async_model(prompt: str) -> str:
        return _ABORT_JSON

    llm = LLMJudgeValidator(model=async_model)
    validator = RuntimeValidator(
        triggers=[MaxRoutesTrigger(max_routes=1)],
        validator=llm,
    )
    trace = make_trace(routing_events=[make_routing_event("A", "B")])
    decision = await validator.validate_async(trace)
    assert decision.action == "abort"
