"""Tests for replay isolation: cloning, internal-state stripping, time determinism."""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from datetime import datetime, timezone, timedelta

from conftest import make_trace, make_routing_event, make_tool_call
from agent_runtime_validator import RuntimeValidator, replay, replay_async
from agent_runtime_validator.triggers import MaxRoutesTrigger, MaxExecutionTimeTrigger
from agent_runtime_validator.validators.llm_judge import LLMJudgeValidator
from agent_runtime_validator.validators.trigger_score import TriggerScoreValidator

_ABORT_JSON = '{"valid":false,"confidence":0.9,"issues":["bad"],"recommendation":"abort","reason":"bad"}'


def _counting_model(response: str):
    calls: list[str] = []

    def model(prompt: str) -> str:
        calls.append(prompt)
        return response

    return model, calls


# ---------------------------------------------------------------------------
# Input trace is never mutated
# ---------------------------------------------------------------------------

def test_replay_does_not_mutate_input_trace():
    model, _ = _counting_model(_ABORT_JSON)
    rv = RuntimeValidator(
        triggers=[MaxRoutesTrigger(max_routes=1)],
        validator=LLMJudgeValidator(model=model),
        max_validator_calls_per_run=1,
    )
    trace = make_trace(routing_events=[make_routing_event("a", "b")])
    before = dict(trace.metadata)
    replay(trace, rv)
    assert trace.metadata == before  # no budget key written to the input


def test_replay_same_trace_twice_same_decision():
    model, calls = _counting_model(_ABORT_JSON)
    rv = RuntimeValidator(
        triggers=[MaxRoutesTrigger(max_routes=1)],
        validator=LLMJudgeValidator(model=model),
        max_validator_calls_per_run=1,
    )
    trace = make_trace(routing_events=[make_routing_event("a", "b")])
    d1 = replay(trace, rv)
    d2 = replay(trace, rv)
    # Without isolation the second replay would hit an exhausted budget and
    # skip the validator, changing the decision.
    assert len(calls) == 2
    assert d1.action == d2.action == "abort"
    assert d1.reason == d2.reason


# ---------------------------------------------------------------------------
# Persisted internal counters are stripped
# ---------------------------------------------------------------------------

def test_replay_strips_persisted_budget_counter():
    """An archived trace with consumed budget still gets a real validation."""
    model, calls = _counting_model(_ABORT_JSON)
    rv = RuntimeValidator(
        triggers=[MaxRoutesTrigger(max_routes=1)],
        validator=LLMJudgeValidator(model=model),
        max_validator_calls_per_run=1,
    )
    trace = make_trace(
        routing_events=[make_routing_event("a", "b")],
        metadata={"_arv_validator_call_count": 1},  # consumed in original run
    )
    decision = replay(trace, rv)
    assert len(calls) == 1  # validator actually ran
    assert decision.action == "abort"


def test_replay_strips_legacy_key_names():
    model, calls = _counting_model(_ABORT_JSON)
    rv = RuntimeValidator(
        triggers=[MaxRoutesTrigger(max_routes=1)],
        validator=LLMJudgeValidator(model=model),
        max_validator_calls_per_run=1,
    )
    trace = make_trace(
        routing_events=[make_routing_event("a", "b")],
        metadata={"_runtime_validator_call_count": 1},  # pre-prefix archive
    )
    replay(trace, rv)
    assert len(calls) == 1


def test_replay_strips_trigger_score_attempts():
    """Persisted attempt counter must not flip reroute into interrupt."""
    rv = RuntimeValidator(
        # medium severity so the validator's reroute is an escalation the
        # policy accepts — isolates the attempt-counter behavior under test
        triggers=[MaxRoutesTrigger(max_routes=1, severity="medium")],
        validator=TriggerScoreValidator(
            weights={"MaxRoutesTrigger": 5.0}, threshold=1.0,
            recommendation="reroute", max_attempts=1,
        ),
    )
    trace = make_trace(
        routing_events=[make_routing_event("a", "b")],
        metadata={"_arv_trigger_score_attempts": 1},
    )
    decision = replay(trace, rv)
    assert decision.action == "reroute"  # not "interrupt" from a stale counter


def test_replay_preserves_user_metadata():
    rv = RuntimeValidator(triggers=[])
    trace = make_trace(metadata={"env": "prod", "_arv_validator_call_count": 3})
    replay(trace, rv)
    assert trace.metadata == {"env": "prod", "_arv_validator_call_count": 3}


# ---------------------------------------------------------------------------
# Time determinism
# ---------------------------------------------------------------------------

def test_replay_time_trigger_uses_event_span_not_archive_age():
    """A years-old archive without finished_at must not trip the time trigger."""
    started = datetime(2024, 1, 1, tzinfo=timezone.utc)
    trace = make_trace(started_at=started)
    trace.tool_calls.append(make_tool_call())  # timestamp 2024-01-01 (conftest)
    rv = RuntimeValidator(triggers=[MaxExecutionTimeTrigger(max_seconds=60)])
    decision = replay(trace, rv)
    assert decision.action == "continue"
    assert decision.triggered_by == []


def test_replay_time_trigger_fires_on_genuinely_long_run():
    started = datetime(2024, 1, 1, tzinfo=timezone.utc)
    trace = make_trace(started_at=started)
    late = make_tool_call()
    late.timestamp = started + timedelta(seconds=300)
    trace.tool_calls.append(late)
    rv = RuntimeValidator(triggers=[MaxExecutionTimeTrigger(max_seconds=60)])
    decision = replay(trace, rv)
    assert "MaxExecutionTimeTrigger" in decision.triggered_by


def test_replay_empty_trace_without_finished_at_is_deterministic():
    started = datetime(2024, 1, 1, tzinfo=timezone.utc)
    trace = make_trace(started_at=started)  # no events, no finished_at
    rv = RuntimeValidator(triggers=[MaxExecutionTimeTrigger(max_seconds=60)])
    d1 = replay(trace, rv)
    d2 = replay(trace, rv)
    assert d1.triggered_by == d2.triggered_by == []


def test_replay_respects_explicit_finished_at():
    started = datetime(2024, 1, 1, tzinfo=timezone.utc)
    trace = make_trace(started_at=started, finished_at=started + timedelta(seconds=500))
    rv = RuntimeValidator(triggers=[MaxExecutionTimeTrigger(max_seconds=60)])
    decision = replay(trace, rv)
    assert "MaxExecutionTimeTrigger" in decision.triggered_by


# ---------------------------------------------------------------------------
# Async parity
# ---------------------------------------------------------------------------

async def test_replay_async_same_isolation():
    calls: list[str] = []

    async def model(prompt: str) -> str:
        calls.append(prompt)
        return _ABORT_JSON

    rv = RuntimeValidator(
        triggers=[MaxRoutesTrigger(max_routes=1)],
        validator=LLMJudgeValidator(model=model),
        max_validator_calls_per_run=1,
    )
    trace = make_trace(routing_events=[make_routing_event("a", "b")])
    before = dict(trace.metadata)
    d1 = await replay_async(trace, rv)
    d2 = await replay_async(trace, rv)
    assert len(calls) == 2
    assert d1.action == d2.action
    assert trace.metadata == before
