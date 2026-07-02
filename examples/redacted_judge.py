"""LLM judge with redaction — sensitive data is masked before reaching the model.

The trace contains an account ID and an email address in tool arguments and
results. `redact_fn` strips them before the prompt is built, so the LLM never
sees the raw values.

Run:
    uv run python examples/redacted_judge.py
"""
import json
import re
from datetime import datetime, timezone

from agent_runtime_validator import ExecutionTrace, RuntimeValidator
from agent_runtime_validator.schema.events import ToolCall, ToolResult
from agent_runtime_validator.triggers import SameToolLoopTrigger
from agent_runtime_validator.validators import LLMJudgeValidator


def now() -> datetime:
    return datetime.now(timezone.utc)


def redact(text: str) -> str:
    text = re.sub(r"A-\d{3,}", "A-***", text)
    text = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "***@***.***",
        text,
    )
    return text


def stub_model(prompt: str) -> str:
    """Print a snippet of the prompt so you can see redaction in action."""
    print("--- Prompt snippet (tool calls section) ---")
    for line in prompt.splitlines():
        if line.strip().startswith("[c"):
            print(line)
    print("---\n")

    return json.dumps({
        "valid": False,
        "confidence": 0.85,
        "recommendation": "interrupt",
        "issues": ["Repeated lookup with same account"],
        "findings": [
            {
                "category": "repeated_tool_call",
                "severity": "medium",
                "confidence": 0.85,
                "summary": "lookup_account called 3x with identical args.",
                "evidence": ["c1, c2, c3 all identical"],
            },
        ],
        "reason": "Agent is stuck retrying the same account lookup.",
    })


def build_trace() -> ExecutionTrace:
    trace = ExecutionTrace(run_id="redact-demo", started_at=now())
    for i in range(3):
        cid = f"c{i + 1}"
        trace.tool_calls.append(ToolCall(
            tool_name="lookup_account",
            call_id=cid,
            args={"account_id": "A-90210", "email": "jane.doe@example.org"},
            timestamp=now(),
        ))
        trace.tool_results.append(ToolResult(
            call_id=cid,
            tool_name="lookup_account",
            output='{"name": "Jane Doe", "email": "jane.doe@example.org", "balance": "..."}',
            timestamp=now(),
        ))
    return trace


def main() -> None:
    validator = RuntimeValidator(
        triggers=[SameToolLoopTrigger(max_repeats=3)],
        validator=LLMJudgeValidator(
            model=stub_model,
            redact_fn=redact,
        ),
    )

    decision = validator.validate(build_trace())

    print(f"action: {decision.action}")
    print(f"reason: {decision.reason}")


if __name__ == "__main__":
    main()
