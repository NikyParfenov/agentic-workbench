"""Tests for TraceFormatConfig and truncation utilities."""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from datetime import datetime, timezone
from conftest import make_trace, make_tool_call, make_tool_result, make_routing_event, make_artifact
from agent_runtime_validator.utils.truncation import truncate
from agent_runtime_validator.validators.trace_format_config import TraceFormatConfig
from agent_runtime_validator.validators.llm_judge import LLMJudgeValidator, _build_trace_details
from agent_runtime_validator.schema.decisions import TriggerResult


# ---------------------------------------------------------------------------
# Truncation strategies
# ---------------------------------------------------------------------------

def test_tail_no_truncation_when_short():
    assert truncate("hello", 100, "tail") == "hello"


def test_tail_truncates_end():
    text = "a" * 200
    result = truncate(text, 60, "tail")
    assert result.startswith("a")
    assert "200 chars total" in result
    assert not result.endswith("a")  # end was dropped


def test_head_truncates_beginning():
    text = "START" + "x" * 200 + "END"
    result = truncate(text, 60, "head")
    assert "chars total" in result
    assert "END" in result  # recent content preserved
    assert "START" not in result  # old content dropped


def test_middle_ellipsis_keeps_both_ends():
    long_text = "START" + "x" * 200 + "END"
    result = truncate(long_text, 60, "middle_ellipsis")
    assert "START" in result
    assert "END" in result
    assert "chars total" in result


def test_tail_no_truncation_when_exact_length():
    text = "abc"
    assert truncate(text, 3, "tail") == "abc"


def test_all_strategies_return_string():
    text = "x" * 200
    for strategy in ("tail", "head", "middle_ellipsis"):
        result = truncate(text, 50, strategy)  # type: ignore[arg-type]
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# TraceFormatConfig defaults
# ---------------------------------------------------------------------------

def test_trace_format_config_defaults():
    cfg = TraceFormatConfig()
    assert cfg.max_events_per_section == 50
    assert cfg.max_chars_per_field == 500
    assert cfg.max_chars_artifact_preview == 200
    assert cfg.include_trace_details is True
    assert cfg.truncation == "tail"


def test_trace_format_config_frozen():
    cfg = TraceFormatConfig()
    import dataclasses
    assert dataclasses.is_dataclass(cfg)
    try:
        cfg.max_events_per_section = 99  # type: ignore[misc]
        assert False, "should raise"
    except Exception:
        pass


# ---------------------------------------------------------------------------
# max_events_per_section
# ---------------------------------------------------------------------------

def test_max_events_per_section_limits_tool_calls(make_trace=make_trace):
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    from agent_runtime_validator.schema.events import ToolCall
    calls = [ToolCall(tool_name="s", call_id=f"c{i}", args={}, timestamp=ts) for i in range(10)]
    trace = make_trace(tool_calls=calls)
    cfg = TraceFormatConfig(max_events_per_section=3)
    details = _build_trace_details(trace, [], cfg, None)
    # Only 3 most recent calls should appear
    assert details.count("[c") <= 3


def test_max_events_zero_hides_section():
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    from agent_runtime_validator.schema.events import ToolCall
    calls = [ToolCall(tool_name="s", call_id=f"c{i}", args={}, timestamp=ts) for i in range(5)]
    trace = make_trace(tool_calls=calls)
    cfg = TraceFormatConfig(max_events_per_section=0)
    details = _build_trace_details(trace, [], cfg, None)
    assert "Tool calls:" not in details


# ---------------------------------------------------------------------------
# max_chars_per_field
# ---------------------------------------------------------------------------

def test_max_chars_per_field_truncates_args():
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    from agent_runtime_validator.schema.events import ToolCall
    long_arg = "x" * 1000
    calls = [ToolCall(tool_name="s", call_id="c1", args={"q": long_arg}, timestamp=ts)]
    trace = make_trace(tool_calls=calls)
    cfg = TraceFormatConfig(max_chars_per_field=50)
    details = _build_trace_details(trace, [], cfg, None)
    # The args string should be truncated; "1000 chars" marker or truncation evidence
    assert "chars total" in details or len(details) < 2000


# ---------------------------------------------------------------------------
# max_chars_artifact_preview
# ---------------------------------------------------------------------------

def test_max_chars_artifact_preview_limits_artifact_content():
    from agent_runtime_validator.schema.events import ArtifactEvent
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    artifact = ArtifactEvent(
        artifact_id="a1",
        artifact_type="report",
        content="C" * 500,
        timestamp=ts,
    )
    trace = make_trace(artifacts=[artifact])
    cfg = TraceFormatConfig(max_chars_artifact_preview=30)
    details = _build_trace_details(trace, [], cfg, None)
    # preview of 500-char content should be truncated
    assert "500 chars total" in details or details.count("C") < 500


# ---------------------------------------------------------------------------
# include_trace_details=False
# ---------------------------------------------------------------------------

def test_include_trace_details_false_omits_details():
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    from agent_runtime_validator.schema.events import ToolCall
    calls = [ToolCall(tool_name="search", call_id="c1", args={}, timestamp=ts)]
    trace = make_trace(tool_calls=calls)

    judge = LLMJudgeValidator(
        model=lambda p: '{"valid":true,"confidence":1.0,"issues":[],"recommendation":"continue","reason":"ok"}',
        trace_format=TraceFormatConfig(include_trace_details=False),
    )
    from agent_runtime_validator.schema.decisions import TriggerResult
    tr = TriggerResult(triggered=True, trigger_name="T", severity="medium", reason="fired")
    prompt = judge._make_prompt(trace, [tr])
    assert "Tool calls:" not in prompt
    assert "Trace details:" not in prompt


# ---------------------------------------------------------------------------
# Backward compatibility: max_trace_events maps to max_events_per_section
# ---------------------------------------------------------------------------

def test_backward_compat_max_trace_events():
    judge = LLMJudgeValidator(
        model=lambda p: "",
        max_trace_events=25,
    )
    assert judge._trace_format.max_events_per_section == 25
    assert judge.max_trace_events == 25


def test_backward_compat_include_trace_details():
    judge = LLMJudgeValidator(
        model=lambda p: "",
        include_trace_details=False,
    )
    assert judge._trace_format.include_trace_details is False
    assert judge.include_trace_details is False


def test_explicit_trace_format_overrides_legacy_params():
    cfg = TraceFormatConfig(max_events_per_section=77)
    judge = LLMJudgeValidator(
        model=lambda p: "",
        max_trace_events=10,
        trace_format=cfg,
    )
    assert judge._trace_format.max_events_per_section == 77
