"""Adapter that converts LangChain-style message lists into ExecutionTrace.

No LangChain dependency is required. The adapter uses duck typing — any
sequence of objects that expose ``.type``, ``.content``, and the relevant
optional attributes is accepted.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Sequence, cast

from ...schema.events import MessageEvent, ToolCall, ToolResult
from ...schema.trace import ExecutionTrace

_MessageRole = Literal["user", "assistant", "system", "tool"]

# ---------------------------------------------------------------------------
# Role mapping
# ---------------------------------------------------------------------------

_ROLE_MAP: dict[str, str] = {
    # LangChain .type values
    "human": "user",
    "ai": "assistant",
    "system": "system",
    "tool": "tool",
    # Class-name fallbacks (lower-cased class names for common LangChain types)
    "humanmessage": "user",
    "aimessage": "assistant",
    "systemmessage": "system",
    "toolmessage": "tool",
    "chatmessage": "assistant",
    "functionmessage": "tool",
}


def _message_type(message: Any) -> str:
    """Return the canonical LangChain type string for *message*.

    Checks ``.type`` first; falls back to the lower-cased class name.
    """
    if hasattr(message, "type"):
        return str(message.type).lower()
    return type(message).__name__.lower()


def _map_role(msg_type: str) -> _MessageRole:
    """Map a LangChain message type to an ExecutionTrace role."""
    return cast(_MessageRole, _ROLE_MAP.get(msg_type, "assistant"))


# ---------------------------------------------------------------------------
# Content extraction
# ---------------------------------------------------------------------------

def _extract_content(message: Any) -> str:
    """Return the text content of a message.

    AIMessages sometimes carry a list of content blocks instead of a plain
    string (e.g. ``[{"type": "text", "text": "..."}]``). We join the text
    parts in that case.
    """
    raw = getattr(message, "content", "")
    if isinstance(raw, list):
        parts: list[str] = []
        for block in raw:
            if isinstance(block, dict):
                text = block.get("text", "")
                if text:
                    parts.append(str(text))
            else:
                parts.append(str(block))
        return " ".join(parts)
    return str(raw)


# ---------------------------------------------------------------------------
# Tool-call extraction helpers
# ---------------------------------------------------------------------------

def _get_attr_or_key(obj: Any, attr: str, default: Any = None) -> Any:
    """Return ``obj[attr]`` if obj is dict-like, else ``getattr(obj, attr, default)``."""
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def from_langchain_messages(
    messages: Sequence[Any],
    *,
    run_id: str = "langchain-run",
    started_at: datetime | None = None,
    agent_name: str | None = None,
    artifacts: Sequence[Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> ExecutionTrace:
    """Convert a LangChain message list into an :class:`ExecutionTrace`.

    Parameters
    ----------
    messages:
        Sequence of LangChain-style message objects. No import of LangChain is
        needed — the function relies on duck typing (``message.type``,
        ``message.content``, etc.).
    run_id:
        Identifier for the run. Defaults to ``"langchain-run"``.
    started_at:
        Trace start time (timezone-aware). Defaults to now (UTC).
    agent_name:
        Agent name attached to ``MessageEvent`` and ``ToolCall`` entries.
    artifacts:
        Pre-built :class:`~agent_runtime_validator.schema.events.ArtifactEvent`
        instances to append to the trace. Must already be ``ArtifactEvent``
        objects.
    metadata:
        Extra metadata to merge into the trace. The key ``"_source"`` is
        always set to ``"langchain_messages"`` (caller-supplied values for
        ``"_source"`` are overwritten).

    Returns
    -------
    ExecutionTrace
    """
    ts: datetime = started_at if started_at is not None else datetime.now(timezone.utc)

    trace_messages: list[MessageEvent] = []
    trace_tool_calls: list[ToolCall] = []
    trace_tool_results: list[ToolResult] = []

    for idx, msg in enumerate(messages):
        msg_type = _message_type(msg)
        role = _map_role(msg_type)
        content = _extract_content(msg)

        trace_messages.append(
            MessageEvent(
                role=role,
                content=content,
                agent_name=agent_name,
                timestamp=ts,
            )
        )

        # ToolMessage → ToolResult
        if msg_type == "tool":
            call_id: str = str(getattr(msg, "tool_call_id", None) or "unknown")
            tool_name: str = str(getattr(msg, "name", None) or "unknown_tool") or "unknown_tool"
            trace_tool_results.append(
                ToolResult(
                    call_id=call_id,
                    tool_name=tool_name,
                    output=content,
                    error=None,
                    timestamp=ts,
                )
            )

        # AIMessage tool_calls → ToolCall
        raw_tool_calls = getattr(msg, "tool_calls", None)
        if raw_tool_calls:
            for tc_idx, tc in enumerate(raw_tool_calls):
                tc_name: str = str(_get_attr_or_key(tc, "name") or "")
                tc_args: dict = _get_attr_or_key(tc, "args") or {}
                if not isinstance(tc_args, dict):
                    tc_args = {}
                raw_call_id = _get_attr_or_key(tc, "id")
                tc_call_id: str = (
                    str(raw_call_id) if raw_call_id is not None else f"tool-call-{tc_idx}"
                )
                trace_tool_calls.append(
                    ToolCall(
                        tool_name=tc_name,
                        args=tc_args,
                        agent_name=agent_name,
                        call_id=tc_call_id,
                        timestamp=ts,
                    )
                )

    merged_metadata: dict[str, Any] = dict(metadata) if metadata else {}
    merged_metadata["_source"] = "langchain_messages"

    return ExecutionTrace(
        run_id=run_id,
        started_at=ts,
        messages=trace_messages,
        tool_calls=trace_tool_calls,
        tool_results=trace_tool_results,
        artifacts=list(artifacts) if artifacts else [],
        metadata=merged_metadata,
    )
