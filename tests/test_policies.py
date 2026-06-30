import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from conftest import make_trace
from agent_runtime_validator.schema.decisions import TriggerResult, ValidatorResult, Severity
from agent_runtime_validator.policies.base import BasePolicy
from agent_runtime_validator.policies.default import DefaultPolicy


def _fired(name: str, severity: Severity) -> TriggerResult:
    return TriggerResult(triggered=True, trigger_name=name, severity=severity, reason="test")


def _not_fired(name: str) -> TriggerResult:
    return TriggerResult(triggered=False, trigger_name=name, severity="low", reason="ok")


def test_no_triggers_returns_continue():
    policy = DefaultPolicy()
    decision = policy.decide(make_trace(), [], None)
    assert decision.action == "continue"
    assert decision.should_continue is True


def test_all_not_fired_returns_continue():
    policy = DefaultPolicy()
    triggered = [_not_fired("MaxToolCallsTrigger"), _not_fired("MaxRoutesTrigger")]
    decision = policy.decide(make_trace(), triggered, None)
    assert decision.action == "continue"
    assert decision.triggered_by == []


def test_low_severity_returns_continue():
    policy = DefaultPolicy()
    decision = policy.decide(make_trace(), [_fired("T", "low")], None)
    assert decision.action == "continue"
    assert decision.severity == "low"


def test_medium_severity_returns_retry():
    policy = DefaultPolicy()
    decision = policy.decide(make_trace(), [_fired("T", "medium")], None)
    assert decision.action == "retry_last_step"
    assert decision.should_continue is False


def test_high_severity_returns_interrupt():
    policy = DefaultPolicy()
    decision = policy.decide(make_trace(), [_fired("T", "high")], None)
    assert decision.action == "interrupt"


def test_critical_severity_returns_abort():
    policy = DefaultPolicy()
    decision = policy.decide(make_trace(), [_fired("T", "critical")], None)
    assert decision.action == "abort"
    assert decision.should_continue is False


def test_highest_severity_wins():
    policy = DefaultPolicy()
    triggered = [_fired("A", "low"), _fired("B", "critical"), _fired("C", "medium")]
    decision = policy.decide(make_trace(), triggered, None)
    assert decision.severity == "critical"
    assert decision.action == "abort"


def test_triggered_by_contains_fired_names():
    policy = DefaultPolicy()
    triggered = [_fired("TrigA", "high"), _not_fired("TrigB"), _fired("TrigC", "medium")]
    decision = policy.decide(make_trace(), triggered, None)
    assert "TrigA" in decision.triggered_by
    assert "TrigC" in decision.triggered_by
    assert "TrigB" not in decision.triggered_by


def test_validator_result_overrides_action():
    policy = DefaultPolicy()
    vr = ValidatorResult(
        valid=False, confidence=0.9, recommendation="reroute", reason="reroute needed"
    )
    decision = policy.decide(make_trace(), [_fired("T", "medium")], vr)
    assert decision.action == "reroute"
    assert decision.severity == "medium"


def test_validator_result_preserved_in_decision():
    policy = DefaultPolicy()
    vr = ValidatorResult(
        valid=True, confidence=0.99, recommendation="continue", reason="all good"
    )
    decision = policy.decide(make_trace(), [_fired("T", "low")], vr)
    assert decision.validator_result is vr


def test_custom_policy_flags():
    policy = DefaultPolicy(retry_on_medium=False, interrupt_on_high=False, abort_on_critical=False)
    assert policy.decide(make_trace(), [_fired("T", "medium")], None).action == "continue"
    assert policy.decide(make_trace(), [_fired("T", "high")], None).action == "continue"
    assert policy.decide(make_trace(), [_fired("T", "critical")], None).action == "interrupt"


def test_default_policy_is_base_policy():
    assert issubclass(DefaultPolicy, BasePolicy)


def test_validator_escalation_allowed():
    policy = DefaultPolicy()
    vr = ValidatorResult(
        valid=False, confidence=0.9, recommendation="abort", reason="critical finding"
    )
    decision = policy.decide(make_trace(), [_fired("T", "medium")], vr)
    assert decision.action == "abort"


def test_validator_downgrade_blocked_by_default():
    policy = DefaultPolicy()
    vr = ValidatorResult(
        valid=True, confidence=0.3, recommendation="continue", reason="looks fine"
    )
    decision = policy.decide(make_trace(), [_fired("T", "high")], vr)
    assert decision.action == "interrupt"


def test_validator_downgrade_allowed_with_high_confidence():
    policy = DefaultPolicy(allow_validator_downgrade=True, min_confidence_for_override=0.7)
    vr = ValidatorResult(
        valid=True, confidence=0.9, recommendation="continue", reason="looks fine"
    )
    decision = policy.decide(make_trace(), [_fired("T", "medium")], vr)
    assert decision.action == "continue"


def test_validator_downgrade_rejected_low_confidence():
    policy = DefaultPolicy(allow_validator_downgrade=True, min_confidence_for_override=0.7)
    vr = ValidatorResult(
        valid=True, confidence=0.5, recommendation="continue", reason="not sure"
    )
    decision = policy.decide(make_trace(), [_fired("T", "high")], vr)
    assert decision.action == "interrupt"


def test_critical_cannot_be_downgraded():
    policy = DefaultPolicy(allow_validator_downgrade=True, min_confidence_for_override=0.0)
    vr = ValidatorResult(
        valid=True, confidence=1.0, recommendation="continue", reason="override attempt"
    )
    decision = policy.decide(make_trace(), [_fired("T", "critical")], vr)
    assert decision.action == "abort"
