"""Validate tool-call arguments against a JSON Schema.

The trace contains a completed call whose argument has the wrong type, so the
validator reports an issue and recommends interrupting.

Run:
    uv run python examples/tool_argument_validation.py
"""
from datetime import datetime, timezone

from agent_runtime_validator import ExecutionTrace, RuntimeValidator
from agent_runtime_validator.schema.events import ToolCall, ToolResult
from agent_runtime_validator.triggers import MaxToolCallsTrigger
from agent_runtime_validator.validators import ToolArgumentValidator


def now() -> datetime:
    return datetime.now(timezone.utc)


SCHEMA = {
    "type": "object",
    "properties": {"gene": {"type": "string"}},
    "required": ["gene"],
    "additionalProperties": False,
}


def build_trace() -> ExecutionTrace:
    trace = ExecutionTrace(run_id="arg-validation-demo", started_at=now())
    # `gene` should be a string; here it is an int, which violates the schema.
    trace.tool_calls.append(
        ToolCall(tool_name="search_gene", call_id="c1", args={"gene": 12345}, timestamp=now())
    )
    trace.tool_results.append(
        ToolResult(call_id="c1", tool_name="search_gene", output="...", timestamp=now())
    )
    return trace


def main() -> None:
    validator = RuntimeValidator(
        triggers=[MaxToolCallsTrigger(max_calls=1)],
        validator=ToolArgumentValidator(arg_schemas={"search_gene": SCHEMA}),
    )

    decision = validator.validate(build_trace())

    print(f"action: {decision.action}")
    result = decision.validator_result
    if result is not None:
        print(f"valid:  {result.valid}")
        print(f"issues: {result.issues}")


if __name__ == "__main__":
    main()
