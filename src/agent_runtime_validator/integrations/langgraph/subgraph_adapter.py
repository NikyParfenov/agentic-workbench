"""Best-effort parser for textual thought/log lines into ExecutionTrace events.

No external dependencies. No LangChain required.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Sequence

from ...schema.events import MessageEvent, ToolCall, ToolResult
from ...schema.trace import ExecutionTrace

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# "Tool call [c1] analyze_item with arguments: {...}"
_RE_CALL_WITH_ID = re.compile(
    r"^Tool call \[(?P<call_id>[^\]]+)\]\s+(?P<tool_name>\S+)\s+with arguments:\s*(?P<args>\{.*\})",
    re.IGNORECASE | re.DOTALL,
)

# "Tool call analyze_item with arguments: {...}"
_RE_CALL_NO_ID = re.compile(
    r"^Tool call (?P<tool_name>\S+)\s+with arguments:\s*(?P<args>\{.*\})",
    re.IGNORECASE | re.DOTALL,
)

# "Calling tool analyze_item with args {...}"
_RE_CALL_ALT = re.compile(
    r"^Calling tool (?P<tool_name>\S+)\s+with args\s*(?P<args>\{.*\})",
    re.IGNORECASE | re.DOTALL,
)

# "Tool result [c1]: ..."
_RE_RESULT_WITH_ID = re.compile(
    r"^Tool (?:result|response|output) \[(?P<call_id>[^\]]+)\]:\s*(?P<output>.*)",
    re.IGNORECASE | re.DOTALL,
)

# "Tool result: ..." (no bracket id)
_RE_RESULT_NO_ID = re.compile(
    r"^Tool (?:result|response|output):\s*(?P<output>.*)",
    re.IGNORECASE | re.DOTALL,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_json_args(raw: str) -> dict[str, Any]:
    """Parse a JSON object from *raw*; return {} on any failure."""
    try:
        parsed = json.loads(raw.strip())
        if isinstance(parsed, dict):
            return parsed
        return {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def from_subgraph_thoughts(
    thoughts: Sequence[str],
    *,
    run_id: str = "subgraph-run",
    started_at: datetime | None = None,
    agent_name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ExecutionTrace:
    """Parse textual thought/log lines into an :class:`ExecutionTrace`.

    Every line always produces a ``MessageEvent(role="assistant")``.
    Lines that match known tool-call or tool-result patterns additionally
    produce ``ToolCall`` / ``ToolResult`` entries.

    Routing events and agent calls are never inferred.

    Parameters
    ----------
    thoughts:
        Sequence of log/thought strings to parse.
    run_id:
        Identifier for the run. Defaults to ``"subgraph-run"``.
    started_at:
        Trace start time (timezone-aware). Defaults to now (UTC).
    agent_name:
        Agent name attached to ``MessageEvent`` and ``ToolCall`` entries.
    metadata:
        Extra metadata to merge into the trace. ``"_source"`` is always set
        to ``"subgraph_thoughts"`` (caller-supplied values are overwritten).

    Returns
    -------
    ExecutionTrace
    """
    ts: datetime = started_at if started_at is not None else datetime.now(timezone.utc)

    trace_messages: list[MessageEvent] = []
    trace_tool_calls: list[ToolCall] = []
    trace_tool_results: list[ToolResult] = []

    # Map bracket call_id → index in trace_tool_calls (for result matching)
    _call_id_to_index: dict[str, int] = {}
    # Track which call indices have been matched by a result
    _matched_call_indices: set[int] = set()

    for line in thoughts:
        # Every line is a message.
        trace_messages.append(
            MessageEvent(
                role="assistant",
                content=line,
                agent_name=agent_name,
                timestamp=ts,
            )
        )

        # --- Attempt to parse a tool call ---
        call_match = (
            _RE_CALL_WITH_ID.match(line)
            or _RE_CALL_NO_ID.match(line)
            or _RE_CALL_ALT.match(line)
        )
        if call_match:
            groups = call_match.groupdict()
            tool_name: str = groups["tool_name"]
            raw_args: str = groups.get("args", "{}")
            args: dict[str, Any] = _parse_json_args(raw_args)
            bracket_id: str | None = groups.get("call_id")

            # Determine call_id
            if bracket_id:
                call_id: str = bracket_id
            else:
                call_id = f"subgraph-tool-call-{len(trace_tool_calls)}"

            idx = len(trace_tool_calls)
            trace_tool_calls.append(
                ToolCall(
                    tool_name=tool_name,
                    args=args,
                    agent_name=agent_name,
                    call_id=call_id,
                    timestamp=ts,
                )
            )
            if bracket_id:
                _call_id_to_index[bracket_id] = idx
            continue  # already processed this line as a call

        # --- Attempt to parse a tool result ---
        result_match = _RE_RESULT_WITH_ID.match(line) or _RE_RESULT_NO_ID.match(line)
        if result_match:
            groups = result_match.groupdict()
            output_str: str = groups.get("output", "") or ""
            bracket_id = groups.get("call_id")

            # Determine matching call
            matched_call: ToolCall | None = None
            if bracket_id and bracket_id in _call_id_to_index:
                matched_idx = _call_id_to_index[bracket_id]
                matched_call = trace_tool_calls[matched_idx]
                _matched_call_indices.add(matched_idx)
            else:
                # Fallback: latest unmatched call, only if exactly one pending
                unmatched = [
                    i for i in range(len(trace_tool_calls))
                    if i not in _matched_call_indices
                ]
                if len(unmatched) == 1:
                    matched_idx = unmatched[0]
                    matched_call = trace_tool_calls[matched_idx]
                    _matched_call_indices.add(matched_idx)

            result_call_id: str = (
                matched_call.call_id if matched_call is not None
                else bracket_id or "unknown_tool"
            )
            result_tool_name: str = (
                matched_call.tool_name if matched_call is not None
                else "unknown_tool"
            )

            trace_tool_results.append(
                ToolResult(
                    call_id=result_call_id,
                    tool_name=result_tool_name,
                    output=output_str,
                    error=None,
                    timestamp=ts,
                )
            )

    merged_metadata: dict[str, Any] = dict(metadata) if metadata else {}
    merged_metadata["_source"] = "subgraph_thoughts"

    return ExecutionTrace(
        run_id=run_id,
        started_at=ts,
        messages=trace_messages,
        tool_calls=trace_tool_calls,
        tool_results=trace_tool_results,
        metadata=merged_metadata,
    )
