import jsonschema
from ..schema.trace import ExecutionTrace
from ..schema.decisions import TriggerResult, ValidatorResult
from .base import BaseValidator


class ToolArgumentValidator(BaseValidator):
    """Validates args of completed tool calls (those with a ToolResult) against declared schemas."""

    def __init__(self, arg_schemas: dict[str, dict]):
        self.arg_schemas = arg_schemas

    def validate(
        self, trace: ExecutionTrace, trigger_results: list[TriggerResult]
    ) -> ValidatorResult:
        _ = trigger_results
        completed_ids = {r.call_id for r in trace.tool_results}
        issues: list[str] = []

        for call in trace.tool_calls:
            if call.call_id not in completed_ids:
                continue
            schema = self.arg_schemas.get(call.tool_name)
            if schema is None:
                continue
            try:
                jsonschema.validate(instance=call.args, schema=schema)
            except jsonschema.ValidationError as exc:
                issues.append(
                    f"Tool '{call.tool_name}' (call_id={call.call_id}): {exc.message}"
                )

        valid = len(issues) == 0
        return ValidatorResult(
            valid=valid,
            confidence=1.0,
            issues=issues,
            recommendation="continue" if valid else "interrupt",
            reason=(
                "All completed tool call arguments are valid"
                if valid
                else f"{len(issues)} argument violation(s) in completed calls"
            ),
        )
