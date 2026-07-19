import inspect
import json
import logging
from collections.abc import Sequence
from typing import Callable, Awaitable
from ..schema.trace import ExecutionTrace
from ..schema.decisions import TriggerResult, ValidatorResult, Recommendation
from ..utils.redaction import apply_redaction
from ..utils.truncation import truncate
from .base import BaseValidator
from .trace_format_config import TraceFormatConfig
from .prompts import DEFAULT_JUDGE_PROMPT, JudgeExample

logger = logging.getLogger("agent_runtime_validator")


def _trunc(text: str, config: TraceFormatConfig, field: str = "field") -> str:
    max_len = (
        config.max_chars_artifact_preview if field == "artifact"
        else config.max_chars_per_field
    )
    return truncate(text, max_len, config.truncation)


def _section_label(label: str, shown: int, total: int) -> str:
    if shown < total:
        return f"{label} [showing {shown} of {total}, most recent first]:"
    return f"{label}:"


def _build_trace_details(
    trace: ExecutionTrace,
    trigger_results: list[TriggerResult],
    config: TraceFormatConfig,
    redact_fn: Callable[[str], str] | None,
) -> str:
    n = config.max_events_per_section
    sections: list[str] = []

    all_messages = trace.messages
    messages = all_messages[-n:] if n else []
    if messages:
        lines = []
        for m in messages:
            content = _trunc(apply_redaction(m.content, redact_fn), config)
            agent = f" ({m.agent_name})" if m.agent_name else ""
            lines.append(f"  [{m.role}{agent}] {content}")
        header = _section_label("Messages", len(messages), len(all_messages))
        sections.append(header + "\n" + "\n".join(lines))

    all_routing = trace.routing_events
    routing = all_routing[-n:] if n else []
    if routing:
        lines = []
        for r in routing:
            reason = (
                f" ({_trunc(apply_redaction(r.reason, redact_fn), config)})"
                if r.reason else ""
            )
            lines.append(f"  {r.from_agent} -> {r.to_agent}{reason}")
        header = _section_label("Routing events", len(routing), len(all_routing))
        sections.append(header + "\n" + "\n".join(lines))

    all_agent_calls = trace.agent_calls
    agent_calls = all_agent_calls[-n:] if n else []
    if agent_calls:
        lines = []
        for ac in agent_calls:
            output_str = (
                _trunc(apply_redaction(ac.output, redact_fn), config)
                if ac.output else "(no output)"
            )
            lines.append(f"  {ac.caller} -> {ac.callee}: {output_str}")
        header = _section_label("Agent delegations", len(agent_calls), len(all_agent_calls))
        sections.append(header + "\n" + "\n".join(lines))

    all_calls = trace.tool_calls
    calls = all_calls[-n:] if n else []
    if calls:
        lines = []
        for c in calls:
            args_str = _trunc(apply_redaction(json.dumps(c.args, default=str), redact_fn), config)
            agent = f" ({c.agent_name})" if c.agent_name else ""
            lines.append(f"  [{c.call_id}]{agent} {c.tool_name}({args_str})")
        header = _section_label("Tool calls", len(calls), len(all_calls))
        sections.append(header + "\n" + "\n".join(lines))

    all_results = trace.tool_results
    results = all_results[-n:] if n else []
    if results:
        lines = []
        for r in results:
            if r.error:
                out = f"ERROR: {_trunc(apply_redaction(r.error, redact_fn), config)}"
            elif r.output:
                out = _trunc(apply_redaction(r.output, redact_fn), config)
            else:
                out = "(no output)"
            lines.append(f"  [{r.call_id}] {r.tool_name} -> {out}")
        header = _section_label("Tool results", len(results), len(all_results))
        sections.append(header + "\n" + "\n".join(lines))

    all_artifacts = trace.artifacts
    artifacts = all_artifacts[-n:] if n else []
    if artifacts:
        lines = []
        for a in artifacts:
            preview = _trunc(apply_redaction(a.content, redact_fn), config, field="artifact")
            agent = f" ({a.agent_name})" if a.agent_name else ""
            lines.append(f"  {a.artifact_id}{agent} [{a.artifact_type}]: {preview}")
        header = _section_label("Artifacts", len(artifacts), len(all_artifacts))
        sections.append(header + "\n" + "\n".join(lines))

    all_errors = trace.errors
    errors = all_errors[-n:] if n else []
    if errors:
        lines = []
        for e in errors:
            agent = f" ({e.agent_name})" if e.agent_name else ""
            message = _trunc(apply_redaction(e.message, redact_fn), config)
            lines.append(f"  {e.error_type}{agent}: {message}")
        header = _section_label("Errors", len(errors), len(all_errors))
        sections.append(header + "\n" + "\n".join(lines))

    fired = [t for t in trigger_results if t.triggered]
    if fired:
        lines = []
        for t in fired:
            reason = _trunc(apply_redaction(t.reason, redact_fn), config)
            lines.append(f"  [{t.severity.upper()}] {t.trigger_name}: {reason}")
            if t.evidence:
                # Serialize first, then redact, then truncate — so redaction sees
                # the full JSON text and truncation never splits mid-secret.
                evidence_str = apply_redaction(json.dumps(t.evidence, default=str), redact_fn)
                lines.append(
                    f"    evidence: "
                    f"{truncate(evidence_str, config.max_chars_trigger_evidence, config.truncation)}"
                )
        sections.append("Fired triggers (detail):\n" + "\n".join(lines))

    if not sections:
        return ""
    return "\nTrace details:\n" + "\n\n".join(sections) + "\n\n"


_REFERENCE_PREAMBLE = (
    "Reference cases are historical precedents included only for calibration.\n"
    "Each case is labeled GOOD or BAD based on prior review of that historical\n"
    "case. The label describes that past case only — it is not a rule requiring\n"
    "the same verdict or recommendation for the candidate trace. Use the cases\n"
    "as context, but evaluate the candidate trace on its own evidence."
)


def _render_reference_case(
    ex: JudgeExample,
    config: TraceFormatConfig,
    redact_fn: Callable[[str], str] | None,
) -> str:
    parts = [f'<reference_case label="{ex.label.upper()}">']
    if ex.note:
        parts.append(f"Curator note: {_trunc(apply_redaction(ex.note, redact_fn), config)}")
    details = _build_trace_details(ex.trace, [], config, redact_fn)
    parts.append(details.strip() if details else "Trace details: (empty trace)")
    parts.append("</reference_case>")
    return "\n".join(parts)


def _omission_footer(omitted: int) -> str:
    return (
        f"\n\n(Note: {omitted} additional reference case(s) omitted due to the "
        f"configured reference budget.)"
    )


def _build_examples(
    examples: Sequence[JudgeExample],
    config: TraceFormatConfig,
    redact_fn: Callable[[str], str] | None,
) -> str:
    """Render reference cases as a bounded, non-binding few-shot precedent block.

    Returns an empty string when there are no examples (or the reference budget
    is disabled), so the ``{examples}`` placeholder collapses to nothing and the
    rendered prompt is unchanged.

    Bounds: at most ``config.max_reference_examples`` cases in caller order, and
    the **entire** rendered block — wrapper tags, preamble, case separators, the
    omission footer, and the closing tag — stays within
    ``config.max_total_reference_chars``. Whole cases are dropped when they do
    not fit (never truncated mid-case), so the ``<reference_case>`` delimiters
    are always well-formed. If not even the wrapper plus one case fits, returns
    an empty string rather than an over-budget or case-less block.
    """
    if not examples:
        return ""
    max_n = config.max_reference_examples
    max_chars = config.max_total_reference_chars
    if max_n <= 0 or max_chars <= 0:
        return ""

    selected = list(examples)[:max_n]
    prefix = "<reference_cases>\n" + _REFERENCE_PREAMBLE + "\n\n"
    suffix = "\n</reference_cases>\n\n"
    # Budget left for the joined cases plus any omission footer.
    available = max_chars - len(prefix) - len(suffix)
    if available <= 0:
        return ""

    rendered: list[str] = []
    cases_len = 0
    for ex in selected:
        block = _render_reference_case(ex, config, redact_fn)
        sep = 2 if rendered else 0  # "\n\n" between cases
        new_cases_len = cases_len + sep + len(block)
        # Footer size assuming we stop right after including this case.
        omitted_if_stop = len(examples) - (len(rendered) + 1)
        footer_len = len(_omission_footer(omitted_if_stop)) if omitted_if_stop > 0 else 0
        if new_cases_len + footer_len > available:
            break
        rendered.append(block)
        cases_len = new_cases_len

    if not rendered:
        return ""

    omitted = len(examples) - len(rendered)
    footer = _omission_footer(omitted) if omitted > 0 else ""
    return prefix + "\n\n".join(rendered) + footer + suffix


def _build_prompt(
    trace: ExecutionTrace,
    trigger_results: list[TriggerResult],
    prompt_template: str,
    config: TraceFormatConfig,
    redact_fn: Callable[[str], str] | None,
    reference_examples: Sequence[JudgeExample] = (),
) -> str:
    fired = [t for t in trigger_results if t.triggered]
    trigger_summary = (
        "\n".join(
            f"- [{t.severity.upper()}] {t.trigger_name}: "
            f"{_trunc(apply_redaction(t.reason, redact_fn), config)}"
            for t in fired
        )
        if fired else "No triggers fired."
    )

    trace_details = (
        _build_trace_details(trace, trigger_results, config, redact_fn)
        if config.include_trace_details else ""
    )

    examples = _build_examples(reference_examples, config, redact_fn)

    return prompt_template.format(
        run_id=trace.run_id,
        tool_call_count=len(trace.tool_calls),
        agent_call_count=len(trace.agent_calls),
        routing_event_count=len(trace.routing_events),
        error_count=len(trace.errors),
        artifact_count=len(trace.artifacts),
        token_usage=trace.token_usage if trace.token_usage is not None else "unknown",
        trigger_summary=trigger_summary,
        trace_details=trace_details,
        examples=examples,
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
        data = json.loads(extracted)
        if isinstance(data.get("confidence"), (int, float)):
            orig = float(data["confidence"])
            clamped = max(0.0, min(1.0, orig))
            if clamped != orig:
                logger.warning(
                    "LLM judge returned out-of-range confidence=%.3f; clamping to %.3f",
                    orig, clamped,
                )
                data["confidence"] = clamped
        result = ValidatorResult.model_validate(data)
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
        trace_format: TraceFormatConfig | None = None,
        reference_examples: Sequence[JudgeExample] | None = None,
    ):
        self.model = model
        self.prompt_template = prompt_template
        self.max_retries = max_retries
        self.fallback_recommendation: Recommendation = fallback_recommendation
        self.redact_fn = redact_fn
        self.reference_examples: tuple[JudgeExample, ...] = tuple(reference_examples or ())

        # Resolve effective config: explicit trace_format wins; legacy params otherwise.
        if trace_format is not None:
            self._trace_format = trace_format
        else:
            self._trace_format = TraceFormatConfig(
                max_events_per_section=max_trace_events,
                include_trace_details=include_trace_details,
            )

        # Keep legacy attributes for backward-compatible attribute access.
        self.include_trace_details = self._trace_format.include_trace_details
        self.max_trace_events = self._trace_format.max_events_per_section

    def _make_prompt(self, trace: ExecutionTrace, trigger_results: list[TriggerResult]) -> str:
        return _build_prompt(
            trace, trigger_results, self.prompt_template,
            self._trace_format, self.redact_fn, self.reference_examples,
        )

    def validate(
        self, trace: ExecutionTrace, trigger_results: list[TriggerResult]
    ) -> ValidatorResult:
        logger.debug(
            "LLM judge: include_trace_details=%s max_events_per_section=%d",
            self._trace_format.include_trace_details,
            self._trace_format.max_events_per_section,
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
            "LLM judge: include_trace_details=%s max_events_per_section=%d",
            self._trace_format.include_trace_details,
            self._trace_format.max_events_per_section,
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
