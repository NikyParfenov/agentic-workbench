from datetime import datetime, timezone
from agent_runtime_validator.schema.events import MessageEvent, ToolCall
from agent_runtime_validator.schema.trace import ExecutionTrace
from agent_runtime_validator.schema.decisions import (
    TriggerResult, ValidatorResult, ValidationDecision,
)


def test_tool_call_requires_fields():
    call = ToolCall(
        tool_name="search_gene",
        call_id="c1",
        args={"gene": "TP53"},
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    assert call.tool_name == "search_gene"
    assert call.args == {"gene": "TP53"}
    assert call.metadata == {}


def test_tool_call_metadata_default_is_independent():
    a = ToolCall(tool_name="x", call_id="1", timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc))
    b = ToolCall(tool_name="y", call_id="2", timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc))
    a.metadata["k"] = "v"
    assert b.metadata == {}


def test_execution_trace_defaults():
    trace = ExecutionTrace(
        run_id="r1",
        started_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    assert trace.tool_calls == []
    assert trace.routing_events == []
    assert trace.token_usage is None
    assert trace.metadata == {}


def test_trigger_result_fields():
    result = TriggerResult(
        triggered=True,
        trigger_name="MaxToolCallsTrigger",
        severity="high",
        reason="limit reached",
    )
    assert result.triggered is True
    assert result.evidence == {}


def test_validator_result_fields():
    result = ValidatorResult(
        valid=False,
        confidence=0.9,
        recommendation="retry_last_step",
        reason="repeated tool calls",
    )
    assert result.issues == []
    assert result.suggested_next_agent is None


def test_validation_decision_fields():
    decision = ValidationDecision(
        should_continue=False,
        action="abort",
        severity="critical",
        reason="runaway agent",
    )
    assert decision.triggered_by == []
    assert decision.validator_result is None


def test_message_event_role_literal():
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        MessageEvent(
            role="invalid_role",  # type: ignore[arg-type]  # intentional: tests runtime rejection
            content="hello",
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )


def test_rejects_naive_timestamp():
    import pytest
    from datetime import datetime
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        ToolCall(
            tool_name="x",
            call_id="c1",
            timestamp=datetime(2024, 1, 1),  # naive — no tzinfo
        )


def test_validation_decision_action_literal():
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        ValidationDecision(
            should_continue=False,
            action="invalid_action",  # type: ignore[arg-type]  # intentional: tests runtime rejection
            severity="high",
            reason="test",
        )


def test_validator_result_confidence_range():
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        ValidatorResult(
            valid=True,
            confidence=1.5,  # out of range
            recommendation="continue",
            reason="test",
        )
