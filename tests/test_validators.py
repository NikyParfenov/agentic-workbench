import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from conftest import make_trace, make_tool_call, make_tool_result
from agent_runtime_validator.validators.base import BaseValidator
from agent_runtime_validator.validators.noop import NoOpValidator
from agent_runtime_validator.validators.jsonschema_validator import JsonSchemaValidator
from agent_runtime_validator.validators.tool_argument_validator import ToolArgumentValidator
from agent_runtime_validator.validators.llm_judge import LLMJudgeValidator


def test_noop_validator_returns_valid():
    trace = make_trace()
    validator = NoOpValidator()
    result = validator.validate(trace, [])
    assert result.valid is True
    assert result.confidence == 1.0
    assert result.recommendation == "continue"
    assert result.issues == []


def test_noop_validator_is_base_validator():
    assert issubclass(NoOpValidator, BaseValidator)


def test_noop_validator_ignores_trigger_results():
    from agent_runtime_validator.schema.decisions import TriggerResult
    trace = make_trace()
    trigger = TriggerResult(
        triggered=True,
        trigger_name="MaxToolCallsTrigger",
        severity="critical",
        reason="too many calls",
    )
    validator = NoOpValidator()
    result = validator.validate(trace, [trigger])
    assert result.valid is True
    assert result.recommendation == "continue"


# JsonSchemaValidator

SEARCH_SCHEMA = {
    "type": "object",
    "properties": {"q": {"type": "string"}},
    "required": ["q"],
}


def test_jsonschema_validator_valid_args():
    trace = make_trace(tool_calls=[make_tool_call("search", "c1", args={"q": "cats"})])
    validator = JsonSchemaValidator(arg_schemas={"search": SEARCH_SCHEMA})
    result = validator.validate(trace, [])
    assert result.valid is True
    assert result.issues == []
    assert result.recommendation == "continue"


def test_jsonschema_validator_invalid_args():
    trace = make_trace(tool_calls=[make_tool_call("search", "c1", args={"query": "cats"})])
    validator = JsonSchemaValidator(arg_schemas={"search": SEARCH_SCHEMA})
    result = validator.validate(trace, [])
    assert result.valid is False
    assert len(result.issues) == 1
    assert "search" in result.issues[0]
    assert result.recommendation == "interrupt"


def test_jsonschema_validator_skips_unregistered_tools():
    trace = make_trace(tool_calls=[make_tool_call("fetch", "c1", args={"url": "x"})])
    validator = JsonSchemaValidator(arg_schemas={"search": SEARCH_SCHEMA})
    result = validator.validate(trace, [])
    assert result.valid is True


def test_jsonschema_validator_result_schema():
    import json
    output = json.dumps({"hits": 5})
    result_schema = {
        "type": "object",
        "properties": {"hits": {"type": "integer"}},
        "required": ["hits"],
    }
    trace = make_trace(tool_results=[make_tool_result("c1", "search", output=output)])
    validator = JsonSchemaValidator(result_schemas={"search": result_schema})
    result = validator.validate(trace, [])
    assert result.valid is True


def test_jsonschema_validator_invalid_result():
    import json
    output = json.dumps({"hits": "not-a-number"})
    result_schema = {
        "type": "object",
        "properties": {"hits": {"type": "integer"}},
        "required": ["hits"],
    }
    trace = make_trace(tool_results=[make_tool_result("c1", "search", output=output)])
    validator = JsonSchemaValidator(result_schemas={"search": result_schema})
    result = validator.validate(trace, [])
    assert result.valid is False
    assert len(result.issues) == 1


def test_jsonschema_validator_is_base_validator():
    assert issubclass(JsonSchemaValidator, BaseValidator)


# ToolArgumentValidator

TOOL_SCHEMA = {
    "type": "object",
    "properties": {"q": {"type": "string"}},
    "required": ["q"],
    "additionalProperties": False,
}


def test_tool_arg_validator_skips_pending_calls():
    # call_id "c1" has no result — should be skipped
    trace = make_trace(
        tool_calls=[make_tool_call("search", "c1", args={})],
        tool_results=[],
    )
    validator = ToolArgumentValidator(arg_schemas={"search": TOOL_SCHEMA})
    result = validator.validate(trace, [])
    assert result.valid is True


def test_tool_arg_validator_validates_completed_calls():
    trace = make_trace(
        tool_calls=[make_tool_call("search", "c1", args={"q": "cats"})],
        tool_results=[make_tool_result("c1", "search")],
    )
    validator = ToolArgumentValidator(arg_schemas={"search": TOOL_SCHEMA})
    result = validator.validate(trace, [])
    assert result.valid is True


def test_tool_arg_validator_invalid_completed_call():
    trace = make_trace(
        tool_calls=[make_tool_call("search", "c1", args={"wrong_key": "x"})],
        tool_results=[make_tool_result("c1", "search")],
    )
    validator = ToolArgumentValidator(arg_schemas={"search": TOOL_SCHEMA})
    result = validator.validate(trace, [])
    assert result.valid is False
    assert len(result.issues) == 1
    assert result.recommendation == "interrupt"


def test_tool_arg_validator_skips_unregistered_tools():
    trace = make_trace(
        tool_calls=[make_tool_call("fetch", "c1", args={})],
        tool_results=[make_tool_result("c1", "fetch")],
    )
    validator = ToolArgumentValidator(arg_schemas={"search": TOOL_SCHEMA})
    result = validator.validate(trace, [])
    assert result.valid is True


def test_tool_arg_validator_is_base_validator():
    assert issubclass(ToolArgumentValidator, BaseValidator)


# LLMJudgeValidator

import json as _json


def _sync_model(response: dict):
    raw = _json.dumps(response)
    return lambda prompt: raw


def test_llm_judge_sync_model_valid():
    trace = make_trace()
    response = {
        "valid": True,
        "confidence": 0.9,
        "recommendation": "continue",
        "issues": [],
        "reason": "all good",
    }
    validator = LLMJudgeValidator(model=_sync_model(response))
    result = validator.validate(trace, [])
    assert result.valid is True
    assert result.confidence == 0.9
    assert result.recommendation == "continue"


def test_llm_judge_sync_model_invalid():
    trace = make_trace()
    response = {
        "valid": False,
        "confidence": 0.8,
        "recommendation": "abort",
        "issues": ["looping detected"],
        "reason": "agent is stuck",
    }
    validator = LLMJudgeValidator(model=_sync_model(response))
    result = validator.validate(trace, [])
    assert result.valid is False
    assert result.recommendation == "abort"
    assert "looping detected" in result.issues


def test_llm_judge_raises_on_async_model_in_sync():
    import pytest
    trace = make_trace()

    async def async_model(prompt: str) -> str:
        return _json.dumps({"valid": True, "confidence": 1.0,
                             "recommendation": "continue", "issues": [], "reason": ""})

    validator = LLMJudgeValidator(model=async_model)
    with pytest.warns(RuntimeWarning), pytest.raises(RuntimeError, match="validate_async"):
        validator.validate(trace, [])


async def test_llm_judge_async_model():
    trace = make_trace()

    async def async_model(prompt: str) -> str:
        return _json.dumps({"valid": True, "confidence": 0.7,
                             "recommendation": "continue", "issues": [], "reason": "ok"})

    validator = LLMJudgeValidator(model=async_model)
    result = await validator.validate_async(trace, [])
    assert result.valid is True
    assert result.confidence == 0.7


def test_llm_judge_unparseable_response():
    trace = make_trace()
    validator = LLMJudgeValidator(model=lambda p: "not json at all")
    result = validator.validate(trace, [])
    assert result.valid is False
    assert result.recommendation == "interrupt"
    assert result.confidence == 0.0


def test_llm_judge_parses_fenced_json():
    trace = make_trace()
    raw = '```json\n{"valid": true, "confidence": 0.8, "recommendation": "continue", "reason": "ok"}\n```'
    validator = LLMJudgeValidator(model=lambda p: raw)
    result = validator.validate(trace, [])
    assert result.valid is True
    assert result.confidence == 0.8


def test_llm_judge_parses_preamble_json():
    trace = make_trace()
    raw = 'Here is my analysis:\n{"valid": false, "confidence": 0.7, "recommendation": "interrupt", "reason": "looping"}'
    validator = LLMJudgeValidator(model=lambda p: raw)
    result = validator.validate(trace, [])
    assert result.valid is False
    assert result.recommendation == "interrupt"


def test_llm_judge_parses_json_with_trailing_text():
    trace = make_trace()
    raw = 'Sure!\n{"valid": true, "confidence": 0.6, "recommendation": "continue", "reason": "ok"}\nHope that helps!'
    validator = LLMJudgeValidator(model=lambda p: raw)
    result = validator.validate(trace, [])
    assert result.valid is True
    assert result.confidence == 0.6


def test_llm_judge_is_base_validator():
    assert issubclass(LLMJudgeValidator, BaseValidator)


def test_validators_package_exports():
    from agent_runtime_validator.validators import (
        BaseValidator, NoOpValidator, JsonSchemaValidator,
        ToolArgumentValidator, LLMJudgeValidator, TriggerScoreValidator,
        DEFAULT_JUDGE_PROMPT,
    )
    assert isinstance(DEFAULT_JUDGE_PROMPT, str)
    assert issubclass(LLMJudgeValidator, BaseValidator)
    assert issubclass(TriggerScoreValidator, BaseValidator)


# --- TriggerScoreValidator ---

from agent_runtime_validator.validators.trigger_score import TriggerScoreValidator
from agent_runtime_validator.schema.decisions import TriggerResult, Severity


def _trigger(name: str, severity: Severity = "medium") -> TriggerResult:
    return TriggerResult(triggered=True, trigger_name=name, severity=severity, reason=f"{name} fired")


def test_trigger_score_below_threshold():
    trace = make_trace()
    validator = TriggerScoreValidator(
        weights={"A": 1.0, "B": 2.0},
        threshold=5.0,
    )
    results = [_trigger("A")]
    result = validator.validate(trace, results)
    assert result.valid is True
    assert result.recommendation == "continue"


def test_trigger_score_above_threshold():
    trace = make_trace()
    validator = TriggerScoreValidator(
        weights={"A": 3.0, "B": 2.0},
        threshold=4.0,
        recommendation="reroute",
    )
    results = [_trigger("A"), _trigger("B")]
    result = validator.validate(trace, results)
    assert result.valid is False
    assert result.recommendation == "reroute"
    assert len(result.issues) == 2


def test_trigger_score_unknown_trigger_ignored():
    trace = make_trace()
    validator = TriggerScoreValidator(
        weights={"A": 1.0},
        threshold=3.0,
    )
    results = [_trigger("A"), _trigger("UnknownTrigger")]
    result = validator.validate(trace, results)
    assert result.valid is True
    assert result.recommendation == "continue"


def test_trigger_score_max_attempts():
    trace = make_trace()
    validator = TriggerScoreValidator(
        weights={"A": 5.0},
        threshold=3.0,
        recommendation="reroute",
        max_attempts=2,
    )
    results = [_trigger("A")]

    r1 = validator.validate(trace, results)
    assert r1.recommendation == "reroute"
    assert trace.metadata["_trigger_score_attempts"] == 1

    r2 = validator.validate(trace, results)
    assert r2.recommendation == "reroute"
    assert trace.metadata["_trigger_score_attempts"] == 2

    r3 = validator.validate(trace, results)
    assert r3.recommendation == "interrupt"
    assert "Maximum trigger-score attempts reached" in r3.issues


def test_trigger_score_recommendation_variants():
    for rec in ("reroute", "retry_last_step", "interrupt", "abort"):
        trace = make_trace()
        validator = TriggerScoreValidator(
            weights={"X": 10.0},
            threshold=1.0,
            recommendation=rec,
        )
        result = validator.validate(trace, [_trigger("X")])
        assert result.recommendation == rec
