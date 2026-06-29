"""LLM-judge validation with a stub model (no API key needed).

A deterministic trigger fires first; only then is the LLM judge invoked. Replace
`stub_model` with a real client call to use a live model. For an async model,
swap `validate` for `await validator.validate_async(trace)`.

Run:
    uv run python examples/llm_judge.py
"""
import json
from datetime import datetime, timezone

from agent_runtime_validator import ExecutionTrace, RuntimeValidator
from agent_runtime_validator.schema.events import ToolCall
from agent_runtime_validator.triggers import SameToolSameArgsLoopTrigger
from agent_runtime_validator.validators import LLMJudgeValidator


def now() -> datetime:
    return datetime.now(timezone.utc)


def stub_model(prompt: str) -> str:
    """Stand-in for a real model. A live model would read `prompt` and reply."""
    _ = prompt
    return json.dumps(
        {
            "valid": False,
            "confidence": 0.9,
            "recommendation": "interrupt",
            "issues": ["search_gene called 3x with identical args"],
            "findings": [
                {
                    "category": "repeated_tool_call",
                    "severity": "high",
                    "confidence": 0.95,
                    "summary": "search_gene called 3 times with identical args {'gene': 'TP53'}.",
                    "evidence": ["call c1: search_gene(gene=TP53)", "call c2: same", "call c3: same"],
                    "tool_name": "search_gene",
                    "suggested_fix": "Try a different gene identifier or check upstream data.",
                },
                {
                    "category": "no_progress",
                    "severity": "medium",
                    "confidence": 0.8,
                    "summary": "No artifacts produced after 3 tool calls.",
                    "evidence": ["tool_calls=3", "artifacts=0"],
                },
            ],
            "reason": "Agent is repeating an identical tool call without progress.",
            "suggested_next_agent": None,
            "suggested_message": "Try a different gene identifier.",
        }
    )


def build_trace() -> ExecutionTrace:
    trace = ExecutionTrace(run_id="llm-judge-demo", started_at=now())
    for i in range(3):
        trace.tool_calls.append(
            ToolCall(tool_name="search_gene", call_id=f"c{i + 1}", args={"gene": "TP53"}, timestamp=now())
        )
    return trace


def main() -> None:
    validator = RuntimeValidator(
        triggers=[SameToolSameArgsLoopTrigger(max_repeats=3)],
        validator=LLMJudgeValidator(model=stub_model),
    )

    decision = validator.validate(build_trace())

    print(f"action: {decision.action}")
    print(f"reason: {decision.reason}")

    result = decision.validator_result
    if result is not None:
        print(f"valid:      {result.valid}")
        print(f"confidence: {result.confidence}")
        print(f"issues:     {result.issues}")
        print(f"suggested:  {result.suggested_message}")
        for f in result.findings:
            print(f"  [{f.severity}] {f.category}: {f.summary}")


if __name__ == "__main__":
    main()
