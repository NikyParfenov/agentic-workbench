"""Tests for RuntimeValidator validator call budget (max_validator_calls_per_run)."""
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from datetime import datetime, timezone
from conftest import make_trace
from agent_runtime_validator.runtime import RuntimeValidator
from agent_runtime_validator.validators.base import BaseValidator
from agent_runtime_validator.schema.decisions import TriggerResult, ValidatorResult, Recommendation
from agent_runtime_validator.schema.trace import ExecutionTrace
from agent_runtime_validator.schema.events import ToolCall
from agent_runtime_validator.triggers import MaxToolCallsTrigger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeValidator(BaseValidator):
    def __init__(self, recommendation: Recommendation = "interrupt"):
        self.calls = 0
        self._recommendation: Recommendation = recommendation

    def validate(self, trace: ExecutionTrace, trigger_results: list[TriggerResult]) -> ValidatorResult:
        self.calls += 1
        return ValidatorResult(
            valid=False,
            confidence=1.0,
            recommendation=self._recommendation,
            reason="called",
        )


class AsyncFakeValidator(BaseValidator):
    def __init__(self):
        self.calls = 0

    def validate(self, trace: ExecutionTrace, trigger_results: list[TriggerResult]) -> ValidatorResult:
        self.calls += 1
        return ValidatorResult(
            valid=False, confidence=1.0, recommendation="interrupt", reason="called"
        )

    async def validate_async(self, trace: ExecutionTrace, trigger_results: list[TriggerResult]) -> ValidatorResult:
        self.calls += 1
        return ValidatorResult(
            valid=False, confidence=1.0, recommendation="interrupt", reason="called async"
        )


def _trace_with_tool_call() -> ExecutionTrace:
    return make_trace(
        tool_calls=[
            ToolCall(
                tool_name="search",
                call_id="c1",
                args={},
                timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            )
        ]
    )


def _runtime(
    fake: BaseValidator,
    max_calls: int | None = 1,
    on_exhausted: str = "skip",
) -> RuntimeValidator:
    return RuntimeValidator(
        triggers=[MaxToolCallsTrigger(max_calls=1)],
        validator=fake,
        max_validator_calls_per_run=max_calls,
        on_validator_budget_exhausted=on_exhausted,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# 1. Default exhausted behavior is "skip"
# ---------------------------------------------------------------------------

def test_default_skip_does_not_create_synthetic_validator_result():
    fake = FakeValidator()
    runtime = _runtime(fake, max_calls=1)
    trace = _trace_with_tool_call()

    d1 = runtime.validate(trace)
    d2 = runtime.validate(trace)

    assert fake.calls == 1
    assert d1.validator_result is not None
    assert d1.validator_result.reason == "called"
    # "skip" → no synthetic ValidatorResult; policy uses trigger severity only
    assert d2.validator_result is None
    # MaxToolCallsTrigger fires at high severity → interrupt (not forced continue)
    assert d2.action == "interrupt"


# ---------------------------------------------------------------------------
# 2. Explicit "continue" creates synthetic result with recommendation=continue
# ---------------------------------------------------------------------------

def test_explicit_continue_creates_synthetic_result():
    fake = FakeValidator()
    runtime = _runtime(fake, max_calls=1, on_exhausted="continue")
    trace = _trace_with_tool_call()

    runtime.validate(trace)
    d2 = runtime.validate(trace)

    assert fake.calls == 1
    assert d2.validator_result is not None
    assert d2.validator_result.recommendation == "continue"
    assert d2.validator_result.valid is True


# ---------------------------------------------------------------------------
# 3. Explicit "interrupt" creates synthetic result
# ---------------------------------------------------------------------------

def test_explicit_interrupt_creates_synthetic_result():
    fake = FakeValidator()
    runtime = _runtime(fake, max_calls=1, on_exhausted="interrupt")
    trace = _trace_with_tool_call()

    runtime.validate(trace)
    d2 = runtime.validate(trace)

    assert d2.validator_result is not None
    assert d2.validator_result.recommendation == "interrupt"
    assert d2.validator_result.valid is False
    assert "budget exhausted" in d2.validator_result.reason.lower()


# ---------------------------------------------------------------------------
# 4. Explicit "reroute" creates synthetic result
# ---------------------------------------------------------------------------

def test_explicit_reroute_creates_synthetic_result():
    fake = FakeValidator()
    runtime = _runtime(fake, max_calls=1, on_exhausted="reroute")
    trace = _trace_with_tool_call()

    runtime.validate(trace)
    d2 = runtime.validate(trace)

    assert d2.validator_result is not None
    assert d2.validator_result.recommendation == "reroute"
    assert d2.validator_result.valid is False


# ---------------------------------------------------------------------------
# 5. max_validator_calls_per_run=0 with default skip
# ---------------------------------------------------------------------------

def test_zero_budget_with_default_skip():
    fake = FakeValidator()
    runtime = _runtime(fake, max_calls=0)  # on_exhausted="skip" by default
    trace = _trace_with_tool_call()

    d = runtime.validate(trace)

    assert fake.calls == 0
    assert d.validator_result is None
    # Trigger still fires; high severity → interrupt from policy
    assert d.action == "interrupt"


# ---------------------------------------------------------------------------
# 6. max_validator_calls_per_run=None → unlimited
# ---------------------------------------------------------------------------

def test_unlimited_budget_calls_every_time():
    fake = FakeValidator()
    runtime = _runtime(fake, max_calls=None)
    trace = _trace_with_tool_call()

    runtime.validate(trace)
    runtime.validate(trace)
    runtime.validate(trace)

    assert fake.calls == 3


# ---------------------------------------------------------------------------
# 7. Async path respects default skip
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_async_default_skip():
    fake = AsyncFakeValidator()
    runtime = _runtime(fake, max_calls=1)
    trace = _trace_with_tool_call()

    d1 = await runtime.validate_async(trace)
    d2 = await runtime.validate_async(trace)

    assert fake.calls == 1
    assert d1.validator_result is not None
    assert d2.validator_result is None
    assert d2.action == "interrupt"  # high severity trigger


# ---------------------------------------------------------------------------
# 8. Invalid exhausted behavior raises ValueError
# ---------------------------------------------------------------------------

def test_invalid_on_exhausted_raises():
    fake = FakeValidator()
    with pytest.raises(ValueError):
        RuntimeValidator(
            triggers=[MaxToolCallsTrigger(max_calls=1)],
            validator=fake,
            on_validator_budget_exhausted="explode",  # type: ignore[arg-type]
        )


def test_negative_max_calls_raises():
    fake = FakeValidator()
    with pytest.raises(ValueError, match="max_validator_calls_per_run"):
        RuntimeValidator(
            triggers=[MaxToolCallsTrigger(max_calls=1)],
            validator=fake,
            max_validator_calls_per_run=-1,
        )


# ---------------------------------------------------------------------------
# 9. LangGraph: budget persists through ValidationNode with serialized trace
# ---------------------------------------------------------------------------

def test_budget_persists_through_validation_node_with_serialized_trace():
    pytest.importorskip("langgraph", reason="langgraph not installed")
    from agent_runtime_validator.integrations.langgraph.nodes import ValidationNode

    fake = FakeValidator()
    node = ValidationNode(
        triggers=[MaxToolCallsTrigger(max_calls=1)],
        validator=fake,
        max_validator_calls_per_run=1,
        # on_validator_budget_exhausted defaults to "skip"
    )

    trace = _trace_with_tool_call()
    state = {"trace": trace.model_dump(mode="json")}

    first = node(state)
    second = node(first)

    assert fake.calls == 1
    # First call: validator ran
    assert first["decision"].validator_result is not None
    assert first["decision"].validator_result.reason == "called"
    # Second call: budget exhausted with "skip" → no synthetic validator result
    assert second["decision"].validator_result is None
    # Budget metadata must be in returned trace
    returned_trace = first["trace"]
    assert isinstance(returned_trace, ExecutionTrace)
    assert returned_trace.metadata.get("_runtime_validator_call_count") == 1


# ---------------------------------------------------------------------------
# Extra: budget is per trace instance
# ---------------------------------------------------------------------------

def test_budget_is_per_trace():
    fake = FakeValidator()
    runtime = _runtime(fake, max_calls=1)
    trace_a = _trace_with_tool_call()
    trace_b = _trace_with_tool_call()

    runtime.validate(trace_a)
    runtime.validate(trace_b)

    assert fake.calls == 2


# ---------------------------------------------------------------------------
# Extra: metadata key is correct
# ---------------------------------------------------------------------------

def test_budget_metadata_key():
    fake = FakeValidator()
    runtime = _runtime(fake, max_calls=2)
    trace = _trace_with_tool_call()

    runtime.validate(trace)

    assert trace.metadata.get("_runtime_validator_call_count") == 1
