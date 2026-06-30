import inspect
import json
import logging
from typing import Callable, Awaitable
from ..schema.trace import ExecutionTrace
from ..schema.decisions import TriggerResult, ValidatorResult, Recommendation
from ..utils.redaction import apply_redaction
from .base import BaseValidator

logger = logging.getLogger("agent_runtime_validator")

_RESPONSE_SCHEMA = json.dumps(ValidatorResult.model_json_schema(), indent=2)

_MAX_TEXT_LEN = 500

_JUDGE_TEMPLATE = """\
You are an agent execution validator. Analyze the following execution trace and triggered alerts, \
then decide whether execution should continue.

Trace summary:
- run_id: {run_id}
- tool calls: {tool_call_count}
- routing events: {routing_event_count}
- errors: {error_count}
- artifacts produced: {artifact_count}
- token usage: {token_usage}

Triggered alerts:
{trigger_summary}
{trace_details}
Check for the following issues:
1. Repeated tool calls
2. Identical tool calls with identical arguments (argument hallucination)
3. Excessive routing between agents
4. Tool failures and retries
5. Lack of progress despite many actions
6. Contradictory tool outputs
7. Hallucinated execution claims (work claimed but not present in the trace)
8. Hallucinated tool arguments (invented IDs, datasets, file paths, thresholds, filters, sample names)
9. Unnecessary iterations
10. Opportunities to reroute execution to a more suitable agent

Respond with a JSON object matching this schema (no markdown, no explanation outside JSON):

{response_schema}
"""

_ESCAPED_SCHEMA = _RESPONSE_SCHEMA.replace("{", "{{").replace("}", "}}")
DEFAULT_JUDGE_PROMPT = _JUDGE_TEMPLATE.replace("{response_schema}", _ESCAPED_SCHEMA)


def _truncate(text: str, max_len: int = _MAX_TEXT_LEN) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"... ({len(text)} chars total)"


def _build_trace_details(
    trace: ExecutionTrace,
    trigger_results: list[TriggerResult],
    max_events: int,
    redact_fn: Callable[[str], str] | None,
) -> str:
    sections: list[str] = []

    messages = trace.messages[-max_events:]
    if messages:
        lines = []
        for m in messages:
            content = _truncate(apply_redaction(m.content, redact_fn))
            agent = f" ({m.agent_name})" if m.agent_name else ""
            lines.append(f"  [{m.role}{agent}] {content}")
        sections.append("Messages:\n" + "\n".join(lines))

    routing = trace.routing_events[-max_events:]
    if routing:
        lines = [f"  {r.from_agent} -> {r.to_agent}" + (f" ({r.reason})" if r.reason else "") for r in routing]
        sections.append("Routing events:\n" + "\n".join(lines))

    calls = trace.tool_calls[-max_events:]
    if calls:
        lines = []
        for c in calls:
            args_str = _truncate(apply_redaction(json.dumps(c.args, default=str), redact_fn))
            agent = f" ({c.agent_name})" if c.agent_name else ""
            lines.append(f"  [{c.call_id}]{agent} {c.tool_name}({args_str})")
        sections.append("Tool calls:\n" + "\n".join(lines))

    results = trace.tool_results[-max_events:]
    if results:
        lines = []
        for r in results:
            if r.error:
                out = f"ERROR: {_truncate(apply_redaction(r.error, redact_fn))}"
            elif r.output:
                out = _truncate(apply_redaction(r.output, redact_fn))
            else:
                out = "(no output)"
            lines.append(f"  [{r.call_id}] {r.tool_name} -> {out}")
        sections.append("Tool results:\n" + "\n".join(lines))

    artifacts = trace.artifacts[-max_events:]
    if artifacts:
        lines = []
        for a in artifacts:
            preview = _truncate(apply_redaction(a.content, redact_fn), 200)
            agent = f" ({a.agent_name})" if a.agent_name else ""
            lines.append(f"  {a.artifact_id}{agent} [{a.artifact_type}]: {preview}")
        sections.append("Artifacts:\n" + "\n".join(lines))

    errors = trace.errors[-max_events:]
    if errors:
        lines = []
        for e in errors:
            agent = f" ({e.agent_name})" if e.agent_name else ""
            lines.append(f"  {e.error_type}{agent}: {e.message}")
        sections.append("Errors:\n" + "\n".join(lines))

    fired = [t for t in trigger_results if t.triggered]
    if fired:
        lines = []
        for t in fired:
            evidence_str = json.dumps(t.evidence, default=str) if t.evidence else ""
            lines.append(f"  [{t.severity.upper()}] {t.trigger_name}: {t.reason}")
            if evidence_str:
                lines.append(f"    evidence: {_truncate(evidence_str, 300)}")
        sections.append("Fired triggers (detail):\n" + "\n".join(lines))

    if not sections:
        return ""
    return "\nTrace details:\n" + "\n\n".join(sections) + "\n\n"


def _build_prompt(
    trace: ExecutionTrace,
    trigger_results: list[TriggerResult],
    prompt_template: str,
    include_trace_details: bool,
    max_trace_events: int,
    redact_fn: Callable[[str], str] | None,
) -> str:
    fired = [t for t in trigger_results if t.triggered]
    if fired:
        trigger_summary = "\n".join(
            f"- [{t.severity.upper()}] {t.trigger_name}: {t.reason}" for t in fired
        )
    else:
        trigger_summary = "No triggers fired."

    if include_trace_details:
        trace_details = _build_trace_details(trace, trigger_results, max_trace_events, redact_fn)
    else:
        trace_details = ""

    return prompt_template.format(
        run_id=trace.run_id,
        tool_call_count=len(trace.tool_calls),
        routing_event_count=len(trace.routing_events),
        error_count=len(trace.errors),
        artifact_count=len(trace.artifacts),
        token_usage=trace.token_usage if trace.token_usage is not None else "unknown",
        trigger_summary=trigger_summary,
        trace_details=trace_details,
    )


def _extract_json(raw: str) -> str:
    """Extract JSON from raw LLM output: fenced blocks, preamble, trailing text."""
    import re
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _ = decoder.raw_decode(text[i:])
                return json.dumps(obj)
            except json.JSONDecodeError:
                continue
    return text


def _parse_response(raw: str, fallback_recommendation: Recommendation) -> ValidatorResult:
    extracted = _extract_json(raw)
    try:
        result = ValidatorResult.model_validate_json(extracted)
        logger.debug("LLM judge response parsed successfully")
        return result
    except Exception:
        logger.warning("LLM judge returned malformed JSON: %s", raw[:200])
        return ValidatorResult(
            valid=False,
            confidence=0.0,
            issues=["LLM response could not be parsed"],
            recommendation=fallback_recommendation,
            reason=f"Unparseable LLM response: {raw[:200]}",
        )


class LLMJudgeValidator(BaseValidator):
    def __init__(
        self,
        model: Callable[[str], "str | Awaitable[str]"],
        prompt_template: str = DEFAULT_JUDGE_PROMPT,
        include_trace_details: bool = True,
        max_trace_events: int = 50,
        max_retries: int = 0,
        fallback_recommendation: Recommendation = "interrupt",
        redact_fn: Callable[[str], str] | None = None,
    ):
        self.model = model
        self.prompt_template = prompt_template
        self.include_trace_details = include_trace_details
        self.max_trace_events = max_trace_events
        self.max_retries = max_retries
        self.fallback_recommendation: Recommendation = fallback_recommendation
        self.redact_fn = redact_fn

    def _make_prompt(self, trace: ExecutionTrace, trigger_results: list[TriggerResult]) -> str:
        return _build_prompt(
            trace, trigger_results, self.prompt_template,
            self.include_trace_details, self.max_trace_events, self.redact_fn,
        )

    def validate(
        self, trace: ExecutionTrace, trigger_results: list[TriggerResult]
    ) -> ValidatorResult:
        logger.debug(
            "LLM judge: include_trace_details=%s max_trace_events=%d",
            self.include_trace_details, self.max_trace_events,
        )
        prompt = self._make_prompt(trace, trigger_results)
        last_raw = ""
        attempts = 1 + self.max_retries
        for attempt in range(attempts):
            raw = self.model(prompt)
            if inspect.isawaitable(raw):
                raise RuntimeError(
                    "LLMJudgeValidator.validate() received an async model. "
                    "Use validate_async() instead."
                )
            last_raw = raw  # type: ignore[assignment]
            result = _parse_response(last_raw, self.fallback_recommendation)
            if "LLM response could not be parsed" not in result.issues:
                return result
            if attempt < attempts - 1:
                logger.warning("LLM judge: retrying (%d/%d)", attempt + 1, self.max_retries)
        logger.error(
            "LLM judge: retries exhausted, using fallback recommendation=%s",
            self.fallback_recommendation,
        )
        return _parse_response(last_raw, self.fallback_recommendation)

    async def validate_async(
        self, trace: ExecutionTrace, trigger_results: list[TriggerResult]
    ) -> ValidatorResult:
        logger.debug(
            "LLM judge: include_trace_details=%s max_trace_events=%d",
            self.include_trace_details, self.max_trace_events,
        )
        prompt = self._make_prompt(trace, trigger_results)
        last_raw = ""
        attempts = 1 + self.max_retries
        for attempt in range(attempts):
            raw = self.model(prompt)
            if inspect.isawaitable(raw):
                raw = await raw
            last_raw = raw  # type: ignore[assignment]
            result = _parse_response(last_raw, self.fallback_recommendation)
            if "LLM response could not be parsed" not in result.issues:
                return result
            if attempt < attempts - 1:
                logger.warning("LLM judge: retrying (%d/%d)", attempt + 1, self.max_retries)
        logger.error(
            "LLM judge: retries exhausted, using fallback recommendation=%s",
            self.fallback_recommendation,
        )
        return _parse_response(last_raw, self.fallback_recommendation)
