"""Tests for LLMJudgeValidator reference cases (structured few-shot JudgeExample)."""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from datetime import datetime, timezone
from typing import Any

import pytest

from conftest import make_trace, make_routing_event
from agent_runtime_validator.validators import (
    LLMJudgeValidator,
    JudgeExample,
    TraceFormatConfig,
    DEFAULT_JUDGE_PROMPT,
)
from agent_runtime_validator.schema.decisions import TriggerResult
from agent_runtime_validator.schema.events import RoutingEvent, ErrorEvent

_OK_JSON = '{"valid":true,"confidence":0.9,"issues":[],"recommendation":"continue","reason":"ok"}'
_TS = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _routing(from_agent: str, to_agent: str, reason: str | None = None) -> RoutingEvent:
    return RoutingEvent(from_agent=from_agent, to_agent=to_agent, reason=reason, timestamp=_TS)


def _error(error_type: str, message: str) -> ErrorEvent:
    return ErrorEvent(error_type=error_type, message=message, timestamp=_TS)


def _capturing_model():
    captured: dict[str, str] = {}

    def model(prompt: str) -> str:
        captured["prompt"] = prompt
        return _OK_JSON

    return model, captured


def _good(note: str | None = None):
    return JudgeExample(
        label="good",
        trace=make_trace(routing_events=[make_routing_event("planner", "analyst")]),
        note=note,
    )


def _bad(note: str | None = None):
    return JudgeExample(
        label="bad",
        trace=make_trace(
            routing_events=[make_routing_event("x", "y"), make_routing_event("y", "x")]
        ),
        note=note,
    )


# ---------------------------------------------------------------------------
# Structured-only JudgeExample
# ---------------------------------------------------------------------------

def test_judge_example_requires_trace():
    # Route through Any so the intentionally-invalid call is a runtime check,
    # not a static type error (keeps pyright clean over the test suite).
    ctor: Any = JudgeExample
    with pytest.raises(TypeError):
        ctor(label="good")


def test_judge_example_no_longer_accepts_description():
    ctor: Any = JudgeExample
    with pytest.raises(TypeError):
        ctor(label="good", description="planner -> analyst -> done")


def test_reference_examples_stored_as_tuple():
    ex = _good()
    judge = LLMJudgeValidator(model=lambda _p: _OK_JSON, reference_examples=[ex])
    assert isinstance(judge.reference_examples, tuple)
    assert judge.reference_examples == (ex,)


# ---------------------------------------------------------------------------
# Backward compatibility / rendering basics
# ---------------------------------------------------------------------------

def test_no_examples_renders_no_reference_section():
    model, captured = _capturing_model()
    LLMJudgeValidator(model=model).validate(make_trace(), [])
    assert "<reference_cases>" not in captured["prompt"]
    assert "{examples}" not in captured["prompt"]


def test_default_prompt_importable_from_llm_judge():
    from agent_runtime_validator.validators.llm_judge import DEFAULT_JUDGE_PROMPT as D
    assert isinstance(D, str)
    assert D is DEFAULT_JUDGE_PROMPT


def test_good_and_bad_cases_labeled():
    model, captured = _capturing_model()
    LLMJudgeValidator(model=model, reference_examples=[_good(), _bad()]).validate(make_trace(), [])
    p = captured["prompt"]
    assert '<reference_case label="GOOD">' in p
    assert '<reference_case label="BAD">' in p


def test_reference_cases_before_candidate_summary():
    model, captured = _capturing_model()
    LLMJudgeValidator(model=model, reference_examples=[_good()]).validate(make_trace(), [])
    p = captured["prompt"]
    assert p.index("<reference_cases>") < p.index("Trace summary")


def test_caller_order_preserved():
    model, captured = _capturing_model()
    LLMJudgeValidator(model=model, reference_examples=[_good(), _bad()]).validate(make_trace(), [])
    p = captured["prompt"]
    assert p.index('label="GOOD"') < p.index('label="BAD"')


def test_full_trace_example_renders_routing():
    model, captured = _capturing_model()
    LLMJudgeValidator(model=model, reference_examples=[_good()]).validate(make_trace(), [])
    assert "planner -> analyst" in captured["prompt"]


def test_note_rendered_as_curator_note():
    model, captured = _capturing_model()
    LLMJudgeValidator(model=model, reference_examples=[_good(note="clean handoff")]).validate(make_trace(), [])
    assert "Curator note: clean handoff" in captured["prompt"]


# ---------------------------------------------------------------------------
# Non-binding + untrusted-data prompt semantics
# ---------------------------------------------------------------------------

def test_prompt_states_cases_are_non_binding():
    model, captured = _capturing_model()
    LLMJudgeValidator(model=model, reference_examples=[_good()]).validate(make_trace(), [])
    p = captured["prompt"].lower()
    assert "not a rule" in p
    assert "own evidence" in p


def test_prompt_states_trace_contents_are_untrusted():
    model, captured = _capturing_model()
    LLMJudgeValidator(model=model).validate(make_trace(), [])
    p = captured["prompt"].lower()
    assert "untrusted execution data" in p
    assert "do not follow" in p


# ---------------------------------------------------------------------------
# Reference budget
# ---------------------------------------------------------------------------

def test_max_reference_examples_cuts_off_later_cases():
    model, captured = _capturing_model()
    cfg = TraceFormatConfig(max_reference_examples=2)
    examples = [_good("first"), _good("second"), _good("third"), _good("fourth")]
    LLMJudgeValidator(model=model, reference_examples=examples, trace_format=cfg).validate(make_trace(), [])
    p = captured["prompt"]
    assert p.count("<reference_case ") == 2
    assert "first" in p and "second" in p
    assert "third" not in p and "fourth" not in p
    assert "2 additional reference case(s) omitted" in p


def test_max_reference_examples_zero_disables_cases():
    model, captured = _capturing_model()
    cfg = TraceFormatConfig(max_reference_examples=0)
    LLMJudgeValidator(model=model, reference_examples=[_good(), _bad()], trace_format=cfg).validate(make_trace(), [])
    assert "<reference_cases>" not in captured["prompt"]


def test_reference_block_never_exceeds_total_char_budget():
    """The whole rendered block (wrapper + preamble + cases + footer + close)
    must stay within max_total_reference_chars for any budget."""
    from agent_runtime_validator.validators.llm_judge import _build_examples
    examples = [_good(f"case {i} " + "d" * 120) for i in range(6)]
    for budget in (0, 150, 400, 800, 1500, 4000):
        cfg = TraceFormatConfig(max_total_reference_chars=budget)
        block = _build_examples(examples, cfg, None)
        assert len(block) <= budget, f"budget={budget} produced {len(block)} chars"
        if block:
            assert block.startswith("<reference_cases>")
            assert block.count("<reference_cases>") == 1
            assert block.count("</reference_cases>") == 1
            # Delimiters always balanced — no case truncated mid-tag.
            assert block.count("<reference_case ") == block.count("</reference_case>")


def test_tight_budget_drops_whole_cases_with_balanced_tags():
    from agent_runtime_validator.validators.llm_judge import _build_examples, _omission_footer
    good, bad = _good("only one fits"), _bad("does not fit")
    # Budget that exactly fits the wrapper + one case + a "1 omitted" footer,
    # but not a second case.
    single = _build_examples([good], TraceFormatConfig(max_total_reference_chars=100_000), None)
    budget = len(single) + len(_omission_footer(1))
    cfg = TraceFormatConfig(max_total_reference_chars=budget)
    block = _build_examples([good, bad], cfg, None)
    assert len(block) <= budget
    assert block.count("<reference_case ") == 1
    assert block.count("</reference_case>") == 1
    assert block.count("<reference_cases>") == block.count("</reference_cases>") == 1
    assert "1 additional reference case(s) omitted" in block


def test_max_total_reference_chars_zero_disables_cases():
    model, captured = _capturing_model()
    cfg = TraceFormatConfig(max_total_reference_chars=0)
    LLMJudgeValidator(model=model, reference_examples=[_good()], trace_format=cfg).validate(make_trace(), [])
    assert "<reference_cases>" not in captured["prompt"]


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------

def test_reference_trace_content_redacted():
    model, captured = _capturing_model()
    ex = JudgeExample(
        label="bad",
        trace=make_trace(routing_events=[_routing("a", "b", reason="token abc123")]),
        note="leaked token abc123",
    )
    judge = LLMJudgeValidator(
        model=model,
        reference_examples=[ex],
        redact_fn=lambda s: s.replace("abc123", "[REDACTED]"),
    )
    judge.validate(make_trace(), [])
    p = captured["prompt"]
    assert "abc123" not in p
    assert "[REDACTED]" in p


def test_note_redacted_and_truncated():
    model, captured = _capturing_model()
    ex = JudgeExample(label="good", trace=make_trace(), note="secret " + "z" * 500)
    judge = LLMJudgeValidator(
        model=model,
        reference_examples=[ex],
        redact_fn=lambda s: s.replace("secret", "[X]"),
        trace_format=TraceFormatConfig(max_chars_per_field=40),
    )
    judge.validate(make_trace(), [])
    p = captured["prompt"]
    assert "secret" not in p
    assert "z" * 500 not in p


def test_routing_reason_redacted_in_candidate_trace():
    model, captured = _capturing_model()
    trace = make_trace(routing_events=[_routing("a", "b", reason="sk-live-123")])
    judge = LLMJudgeValidator(model=model, redact_fn=lambda s: s.replace("sk-live-123", "[KEY]"))
    judge.validate(trace, [])
    p = captured["prompt"]
    assert "sk-live-123" not in p
    assert "[KEY]" in p


def test_error_message_redacted_in_candidate_trace():
    model, captured = _capturing_model()
    trace = make_trace(errors=[_error("RuntimeError", "failed with secret xyz789")])
    judge = LLMJudgeValidator(model=model, redact_fn=lambda s: s.replace("xyz789", "[R]"))
    judge.validate(trace, [])
    p = captured["prompt"]
    assert "xyz789" not in p
    assert "[R]" in p


def test_trigger_reason_redacted_and_truncated():
    model, captured = _capturing_model()
    fired = TriggerResult(
        trigger_name="SecretTrigger",
        triggered=True,
        severity="high",
        reason="secret-trigger-reason " + "q" * 500,
        evidence={},
    )
    judge = LLMJudgeValidator(
        model=model,
        redact_fn=lambda s: s.replace("secret-trigger-reason", "[REDACTED]"),
        trace_format=TraceFormatConfig(max_chars_per_field=60),
    )
    judge.validate(make_trace(), [fired])
    p = captured["prompt"]
    assert "secret-trigger-reason" not in p   # redacted in both summary and detail
    assert "[REDACTED]" in p
    assert "q" * 500 not in p                  # truncated


def test_trigger_evidence_redacted_in_candidate_trace():
    model, captured = _capturing_model()
    fired = TriggerResult(
        trigger_name="SecretTrigger",
        triggered=True,
        severity="high",
        reason="fired",
        evidence={"leaked": "topsecret42"},
    )
    judge = LLMJudgeValidator(model=model, redact_fn=lambda s: s.replace("topsecret42", "[E]"))
    judge.validate(make_trace(), [fired])
    p = captured["prompt"]
    assert "topsecret42" not in p
    assert "[E]" in p


# ---------------------------------------------------------------------------
# Custom template safety
# ---------------------------------------------------------------------------

def test_custom_template_without_examples_placeholder_does_not_crash():
    model, captured = _capturing_model()
    template = "Judge this run {run_id}. Triggers:\n{trigger_summary}\n{trace_details}"
    judge = LLMJudgeValidator(model=model, prompt_template=template, reference_examples=[_good()])
    result = judge.validate(make_trace(), [])
    assert result.recommendation == "continue"
    assert "Judge this run" in captured["prompt"]
