import json
import jsonschema
from ..schema.trace import ExecutionTrace
from ..schema.decisions import TriggerResult, ValidatorResult
from .base import BaseValidator


class JsonSchemaValidator(BaseValidator):
    """Validates tool call args and results against user-provided JSON schemas."""

    def __init__(
        self,
        arg_schemas: dict[str, dict] | None = None,
        result_schemas: dict[str, dict] | None = None,
    ):
        self.arg_schemas = arg_schemas or {}
        self.result_schemas = result_schemas or {}

    def validate(
        self, trace: ExecutionTrace, trigger_results: list[TriggerResult]
    ) -> ValidatorResult:
        _ = trigger_results
        issues: list[str] = []

        for call in trace.tool_calls:
            schema = self.arg_schemas.get(call.tool_name)
            if schema is None:
                continue
            try:
                jsonschema.validate(instance=call.args, schema=schema)
            except jsonschema.ValidationError as exc:
                issues.append(
                    f"Tool '{call.tool_name}' (call_id={call.call_id}) args invalid: {exc.message}"
                )

        for result in trace.tool_results:
            schema = self.result_schemas.get(result.tool_name)
            if schema is None or result.output is None:
                continue
            try:
                parsed = json.loads(result.output)
            except (json.JSONDecodeError, TypeError):
                parsed = result.output
            try:
                jsonschema.validate(instance=parsed, schema=schema)
            except jsonschema.ValidationError as exc:
                issues.append(
                    f"Tool '{result.tool_name}' (call_id={result.call_id}) result invalid: {exc.message}"
                )

        valid = len(issues) == 0
        return ValidatorResult(
            valid=valid,
            confidence=1.0,
            issues=issues,
            recommendation="continue" if valid else "interrupt",
            reason="All schemas valid" if valid else f"{len(issues)} schema violation(s) detected",
        )
