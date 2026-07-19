"""Tests for validator failure containment (on_validator_error)."""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import pytest

from conftest import make_trace, make_routing_event
from agent_runtime_validator import RuntimeValidator
from agent_runtime_validator.schema.trace import ExecutionTrace
from agent_runtime_validator.triggers import MaxRoutesTrigger
from agent_runtime_validator.validators.base import BaseValidator
from agent_runtime_validator.validators.llm_judge import LLMJudgeValidator


class _RaisingValidator(BaseValidator):
    """Sync validator whose infrastructure always fails."""

    def validate(self, trace, trigger_results):
        raise ConnectionError("provider unreachable: secret-endpoint-details")


class _AsyncRaisingValidator(BaseValidator):
    def validate(self, trace, trigger_results):
        raise ConnectionError("sync path should not be used in this test")

    async def validate_async(self, trace, trigger_results):
        raise TimeoutError("provider timed out")


def _fired_trace() -> ExecutionTrace:
    return make_trace(routing_events=[make_routing_event("a", "b")])


def _fired_trigger() -> MaxRoutesTrigger:
    return MaxRoutesTrigger(max_routes=1)


# ---------------------------------------------------------------------------
# Default: skip — exception contained, triggers + policy decide
# ---------------------------------------------------------------------------

def test_sync_validator_exception_is_contained():
    rv = RuntimeValidator(triggers=[_fired_trigger()], validator=_RaisingValidator())
    decision = rv.validate(_fired_trace())  # must not raise
    assert decision.validator_result is None
    assert decision.triggered_by == ["MaxRoutesTrigger"]


def test_llm_judge_model_exception_is_contained():
    """A raising model callable inside LLMJudgeValidator must not kill the run."""

    def broken_model(prompt: str) -> str:
        raise TimeoutError("request timed out")

    rv = RuntimeValidator(
        triggers=[_fired_trigger()],
        validator=LLMJudgeValidator(model=broken_model),
    )
    decision = rv.validate(_fired_trace())
    assert decision.validator_result is None
    assert decision.action == "interrupt"  # high-severity trigger decides


async def test_async_validator_exception_is_contained():
    rv = RuntimeValidator(triggers=[_fired_trigger()], validator=_AsyncRaisingValidator())
    decision = await rv.validate_async(_fired_trace())
    assert decision.validator_result is None
    assert decision.triggered_by == ["MaxRoutesTrigger"]


# ---------------------------------------------------------------------------
# Configured behaviors
# ---------------------------------------------------------------------------

def test_on_validator_error_interrupt_produces_synthetic_result():
    rv = RuntimeValidator(
        triggers=[_fired_trigger()],
        validator=_RaisingValidator(),
        on_validator_error="interrupt",
    )
    decision = rv.validate(_fired_trace())
    assert decision.action == "interrupt"
    assert decision.validator_result is not None
    assert "Validator infrastructure error" in decision.validator_result.issues
    assert "ConnectionError" in decision.validator_result.reason


def test_on_validator_error_abort_in_always_mode_without_triggers():
    rv = RuntimeValidator(
        triggers=[],
        validator=_RaisingValidator(),
        validator_mode="always",
        on_validator_error="abort",
    )
    decision = rv.validate(make_trace())
    assert decision.action == "abort"


def test_error_reason_does_not_leak_exception_message():
    """Only the exception type appears in the decision; the message stays in logs."""
    rv = RuntimeValidator(
        triggers=[_fired_trigger()],
        validator=_RaisingValidator(),
        on_validator_error="interrupt",
    )
    decision = rv.validate(_fired_trace())
    assert decision.validator_result is not None
    assert "secret-endpoint-details" not in decision.validator_result.reason


def test_invalid_on_validator_error_raises():
    with pytest.raises(ValueError, match="on_validator_error"):
        RuntimeValidator(
            triggers=[],
            on_validator_error="explode",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Sync-awaitable programming error: still raises, but closes the awaitable
# ---------------------------------------------------------------------------

class _FakeAwaitable:
    def __init__(self):
        self.closed = False

    def __await__(self):  # pragma: no cover - never actually awaited
        yield

    def close(self):
        self.closed = True


class _AwaitableReturningValidator(BaseValidator):
    def __init__(self, awaitable):
        self.awaitable = awaitable

    def validate(self, trace, trigger_results):
        return self.awaitable


def test_sync_awaitable_still_raises_and_closes_coroutine():
    awaitable = _FakeAwaitable()
    rv = RuntimeValidator(
        triggers=[_fired_trigger()],
        validator=_AwaitableReturningValidator(awaitable),
    )
    with pytest.raises(RuntimeError, match="validate_async"):
        rv.validate(_fired_trace())
    assert awaitable.closed


# ---------------------------------------------------------------------------
# ValidationNode passthrough
# ---------------------------------------------------------------------------

def test_validation_node_contains_validator_errors():
    pytest.importorskip("langgraph", reason="langgraph not installed")
    from agent_runtime_validator.integrations.langgraph.nodes import ValidationNode

    node = ValidationNode(
        triggers=[_fired_trigger()],
        validator=_RaisingValidator(),
        on_validator_error="interrupt",
    )
    result = node({"trace": _fired_trace()})
    assert result["decision"].action == "interrupt"
