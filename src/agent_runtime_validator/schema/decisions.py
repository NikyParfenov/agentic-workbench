from typing import Literal
from pydantic import BaseModel, Field

# Severity of a trigger firing
Severity = Literal["low", "medium", "high", "critical"]
# Final action decided by the policy
Action = Literal["continue", "retry_last_step", "reroute", "interrupt", "abort"]
# Recommendation returned by a validator — policy may override it
Recommendation = Literal["continue", "retry_last_step", "reroute", "interrupt", "abort"]

FindingCategory = Literal[
    "repeated_tool_call",
    "tool_argument_hallucination",
    "excessive_routing",
    "repeated_failure",
    "no_progress",
    "contradictory_output",
    "hallucinated_claim",
    "unnecessary_iteration",
    "misrouting",
    "missing_required_artifact",
    "other",
]


class TriggerResult(BaseModel):
    triggered: bool
    trigger_name: str
    severity: Severity
    reason: str
    evidence: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)


class JudgeFinding(BaseModel):
    category: FindingCategory
    severity: Severity
    confidence: float = Field(..., ge=0.0, le=1.0)
    summary: str
    evidence: list[str] = Field(default_factory=list)
    agent_name: str | None = None
    tool_name: str | None = None
    call_id: str | None = None
    expected: str | None = None
    actual: str | None = None
    suggested_fix: str | None = None


class ValidatorResult(BaseModel):
    valid: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list)
    findings: list[JudgeFinding] = Field(default_factory=list)
    recommendation: Recommendation
    reason: str
    suggested_next_agent: str | None = None
    suggested_message: str | None = None


class ValidationDecision(BaseModel):
    """Final verdict of a validation pass.

    ``should_continue`` is ``True`` only when ``action == "continue"`` — it
    means *no intervention requested*, not *the run must halt*. Recovery
    actions (``retry_last_step``, ``reroute``) have ``should_continue=False``
    even though execution normally keeps going after the recovery step. Use
    :attr:`is_terminal` to ask "should the run stop".

    ``severity`` is the highest severity among fired triggers; when no trigger
    fired and the validator drove the decision, it is derived from the decided
    action instead.
    """

    should_continue: bool
    action: Action
    severity: Severity
    reason: str
    triggered_by: list[str] = Field(default_factory=list)
    validator_result: ValidatorResult | None = None

    @property
    def is_terminal(self) -> bool:
        """``True`` when the run should stop (``interrupt`` or ``abort``)."""
        return self.action in ("interrupt", "abort")
