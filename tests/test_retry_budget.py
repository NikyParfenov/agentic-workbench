"""Tests for the policy-level retry budget (retry-loop safety)."""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import pytest

from conftest import make_trace, make_tool_call
from agent_runtime_validator import RuntimeValidator, replay
from agent_runtime_validator.policies.default import DefaultPolicy
from agent_runtime_validator.schema.decisions import TriggerResult, ValidatorResult
from agent_runtime_validator.triggers import SameToolLoopTrigger


def _medium_fired() -> TriggerResult:
    return TriggerResult(
        triggered=True, trigger_name="T", severity="medium", reason="loop"
    )


# ---------------------------------------------------------------------------
# Bounded retries with the default policy
# ---------------------------------------------------------------------------

def test_default_retry_budget_escalates_to_interrupt():
    policy = DefaultPolicy()  # max_retries_per_run=3 by default
    trace = make_trace()
    actions = [policy.decide(trace, [_medium_fired()], None).action for _ in range(5)]
    assert actions == [
        "retry_last_step", "retry_last_step", "retry_last_step",
        "interrupt", "interrupt",
    ]


def test_retry_budget_zero_never_retries():
    policy = DefaultPolicy(max_retries_per_run=0)
    trace = make_trace()
    decision = policy.decide(trace, [_medium_fired()], None)
    assert decision.action == "interrupt"


def test_retry_budget_none_is_unlimited():
    policy = DefaultPolicy(max_retries_per_run=None)
    trace = make_trace()
    actions = [policy.decide(trace, [_medium_fired()], None).action for _ in range(6)]
    assert actions == ["retry_last_step"] * 6


def test_negative_retry_budget_raises():
    with pytest.raises(ValueError, match="max_retries_per_run"):
        DefaultPolicy(max_retries_per_run=-1)


def test_retry_counter_lives_under_arv_prefix():
    policy = DefaultPolicy()
    trace = make_trace()
    policy.decide(trace, [_medium_fired()], None)
    assert trace.metadata["_arv_policy_retry_count"] == 1


def test_escalated_reason_mentions_budget():
    policy = DefaultPolicy(max_retries_per_run=1)
    trace = make_trace()
    policy.decide(trace, [_medium_fired()], None)
    decision = policy.decide(trace, [_medium_fired()], None)
    assert decision.action == "interrupt"
    assert "Retry budget exhausted" in decision.reason


def test_validator_recommended_retry_also_counted():
    """Retries from the no-trigger validator branch share the same budget."""
    policy = DefaultPolicy(max_retries_per_run=2)
    trace = make_trace()
    vr = ValidatorResult(
        valid=False, confidence=0.9,
        recommendation="retry_last_step", reason="transient",
    )
    actions = [policy.decide(trace, [], vr).action for _ in range(4)]
    assert actions == ["retry_last_step", "retry_last_step", "interrupt", "interrupt"]
    # Escalated no-trigger decision carries action-derived severity.
    assert policy.decide(trace, [], vr).severity == "high"


def test_non_retry_actions_do_not_consume_budget():
    policy = DefaultPolicy(max_retries_per_run=1)
    trace = make_trace()
    high = TriggerResult(triggered=True, trigger_name="H", severity="high", reason="x")
    for _ in range(3):
        assert policy.decide(trace, [high], None).action == "interrupt"
    assert "_arv_policy_retry_count" not in trace.metadata


# ---------------------------------------------------------------------------
# Regression: cumulative trigger + retry cannot loop forever
# ---------------------------------------------------------------------------

def test_cumulative_trigger_retry_loop_terminates():
    """Simulates a host that honors retry_last_step: the trace only grows, the
    trigger fires on every checkpoint, and the loop must still terminate."""
    rv = RuntimeValidator(
        triggers=[SameToolLoopTrigger(max_repeats=3, severity="medium")],
    )
    trace = make_trace()
    for i in range(3):
        trace.tool_calls.append(make_tool_call(call_id=f"c{i}"))

    iterations = 0
    decision = rv.validate(trace)
    iterations = 1
    while iterations < 20 and decision.action == "retry_last_step":
        # host "retries": the same tool call is appended, trace grows
        trace.tool_calls.append(make_tool_call(call_id=f"retry{iterations}"))
        decision = rv.validate(trace)
        iterations += 1
    assert decision.action == "interrupt"
    assert "Retry budget exhausted" in decision.reason
    assert iterations == 4  # 3 retries (default budget) + the escalation


# ---------------------------------------------------------------------------
# Replay integration: budget resets on replay
# ---------------------------------------------------------------------------

def test_replay_strips_policy_retry_counter():
    rv = RuntimeValidator(
        triggers=[SameToolLoopTrigger(max_repeats=1, severity="medium")],
    )
    trace = make_trace(metadata={"_arv_policy_retry_count": 99})
    trace.tool_calls.append(make_tool_call())
    decision = replay(trace, rv)
    assert decision.action == "retry_last_step"  # not interrupt from stale counter
    assert trace.metadata["_arv_policy_retry_count"] == 99  # input untouched
