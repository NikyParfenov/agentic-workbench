"""Tests for validator_mode ("checkpoint" vs "final_gate")."""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from conftest import make_trace, make_routing_event
from agent_runtime_validator import RuntimeValidator, ValidatorMode
from agent_runtime_validator.validators.llm_judge import LLMJudgeValidator

_CONTINUE_JSON = '{"valid":true,"confidence":0.99,"issues":[],"recommendation":"continue","reason":"all clear"}'
_ABORT_JSON = '{"valid":false,"confidence":0.95,"issues":["bad"],"recommendation":"abort","reason":"something bad"}'


def _counting_model(responses: list[str]):
    calls: list[str] = []
    idx = [0]

    def model(prompt: str) -> str:
        calls.append(prompt)
        response = responses[idx[0] % len(responses)]
        idx[0] += 1
        return response

    return model, calls


# ---------------------------------------------------------------------------
# Default mode is "checkpoint"
# ---------------------------------------------------------------------------

def test_default_mode_is_checkpoint():
    rv = RuntimeValidator(triggers=[])
    assert rv.validator_mode == "checkpoint"


def test_checkpoint_validator_not_called_when_no_triggers_fire():
    model, calls = _counting_model([_ABORT_JSON])
    rv = RuntimeValidator(
        triggers=[],  # nothing can fire
        validator=LLMJudgeValidator(model=model),
        validator_mode="checkpoint",
    )
    trace = make_trace()
    decision = rv.validate(trace)
    assert calls == []
    assert decision.action == "continue"


def test_checkpoint_validator_called_when_trigger_fires():
    model, calls = _counting_model([_ABORT_JSON])
    rv = RuntimeValidator(
        triggers=[__import__("agent_runtime_validator.triggers", fromlist=["MaxRoutesTrigger"]).MaxRoutesTrigger(max_routes=1)],
        validator=LLMJudgeValidator(model=model),
        validator_mode="checkpoint",
    )
    trace = make_trace(routing_events=[make_routing_event("A", "B")])
    decision = rv.validate(trace)
    assert len(calls) == 1
    assert decision.action == "abort"


# ---------------------------------------------------------------------------
# final_gate mode
# ---------------------------------------------------------------------------

def test_final_gate_validator_called_even_when_no_triggers_fire():
    model, calls = _counting_model([_CONTINUE_JSON])
    rv = RuntimeValidator(
        triggers=[],  # no triggers
        validator=LLMJudgeValidator(model=model),
        validator_mode="final_gate",
    )
    trace = make_trace()
    rv.validate(trace)
    assert len(calls) == 1


def test_final_gate_validator_escalates_on_no_trigger_signal():
    model, calls = _counting_model([_ABORT_JSON])
    rv = RuntimeValidator(
        triggers=[],
        validator=LLMJudgeValidator(model=model),
        validator_mode="final_gate",
    )
    trace = make_trace()
    decision = rv.validate(trace)
    assert decision.action == "abort"
    assert decision.validator_result is not None
    assert decision.validator_result.recommendation == "abort"


def test_final_gate_validator_continue_on_clean_trace():
    model, calls = _counting_model([_CONTINUE_JSON])
    rv = RuntimeValidator(
        triggers=[],
        validator=LLMJudgeValidator(model=model),
        validator_mode="final_gate",
    )
    trace = make_trace()
    decision = rv.validate(trace)
    assert decision.action == "continue"
    assert decision.validator_result is not None


def test_final_gate_noop_validator_never_invoked():
    from agent_runtime_validator.validators.noop import NoOpValidator
    rv = RuntimeValidator(
        triggers=[],
        validator=None,  # defaults to NoOpValidator
        validator_mode="final_gate",
    )
    trace = make_trace()
    decision = rv.validate(trace)
    assert decision.validator_result is None
    assert decision.action == "continue"


def test_final_gate_budget_still_applies():
    model, calls = _counting_model([_CONTINUE_JSON])
    rv = RuntimeValidator(
        triggers=[],
        validator=LLMJudgeValidator(model=model),
        validator_mode="final_gate",
        max_validator_calls_per_run=1,
        on_validator_budget_exhausted="skip",
    )
    trace = make_trace()
    rv.validate(trace)
    rv.validate(trace)  # second call — budget exhausted
    assert len(calls) == 1  # validator only called once


def test_final_gate_budget_exhausted_skip():
    model, calls = _counting_model([_ABORT_JSON])
    rv = RuntimeValidator(
        triggers=[],
        validator=LLMJudgeValidator(model=model),
        validator_mode="final_gate",
        max_validator_calls_per_run=0,
        on_validator_budget_exhausted="skip",
    )
    trace = make_trace()
    decision = rv.validate(trace)
    assert calls == []
    assert decision.action == "continue"


# ---------------------------------------------------------------------------
# final_gate async path
# ---------------------------------------------------------------------------

async def test_final_gate_async_validator_always_called():
    calls: list[str] = []

    async def async_model(prompt: str) -> str:
        calls.append(prompt)
        return _ABORT_JSON

    rv = RuntimeValidator(
        triggers=[],
        validator=LLMJudgeValidator(model=async_model),
        validator_mode="final_gate",
    )
    trace = make_trace()
    decision = await rv.validate_async(trace)
    assert len(calls) == 1
    assert decision.action == "abort"


# ---------------------------------------------------------------------------
# Invalid mode raises
# ---------------------------------------------------------------------------

def test_invalid_validator_mode_raises():
    import pytest
    with pytest.raises(ValueError, match="validator_mode"):
        RuntimeValidator(
            triggers=[],
            validator_mode="unknown_mode",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# ValidatorMode type exported
# ---------------------------------------------------------------------------

def test_validator_mode_importable():
    from agent_runtime_validator import ValidatorMode
    assert ValidatorMode is not None
