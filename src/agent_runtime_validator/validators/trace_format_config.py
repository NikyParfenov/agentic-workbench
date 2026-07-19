from dataclasses import dataclass
from typing import Literal

from ..utils.truncation import TruncationStrategy


@dataclass(frozen=True)
class TraceFormatConfig:
    """Controls how ExecutionTrace is formatted into the LLM judge prompt.

    All limits are applied before the trace text reaches the model. Setting a value
    to ``0`` disables that section entirely.
    """

    max_events_per_section: int = 50
    max_chars_per_field: int = 500
    max_chars_artifact_preview: int = 200
    max_chars_trigger_evidence: int = 300
    include_trace_details: bool = True
    truncation: TruncationStrategy = "tail"

    # Reference (few-shot) example budget for LLMJudgeValidator. These bound the
    # precedent context so a large list of reference cases cannot blow up the
    # prompt. Setting either to 0 disables reference cases entirely.
    max_reference_examples: int = 4
    max_total_reference_chars: int = 12_000
