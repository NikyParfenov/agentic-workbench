"""Lift structured subgraph messages into a parent ExecutionTrace.

Many applications intentionally keep an inner graph's tool conversation out of
the outer graph's ``messages`` channel: copying inner ``ToolMessage`` objects
upward bloats context, breaks tool-call pairing after trimming, and pollutes
the conversational state. The cost is that the runtime validator no longer
sees the inner tool activity.

:func:`lift_subgraph_messages` closes that gap. It converts the inner graph's
structured messages (``AIMessage.tool_calls`` / ``ToolMessage``) into an
``ExecutionTrace`` delta and merges the delta into the parent trace — the
outer ``messages`` stay compact while the validator still receives structured
``ToolCall`` / ``ToolResult`` events.

No LangChain dependency is required; the conversion relies on the same duck
typing as :func:`from_langchain_messages`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Sequence

try:
    from langgraph.graph import StateGraph  # noqa: F401
except ImportError:
    raise ImportError(
        "LangGraph integration requires langgraph. "
        "Install it with: pip install agent-runtime-validator[langgraph]"
    ) from None

from ...schema.trace import ExecutionTrace
from ...trace_builder import TraceBuilder
from .langchain_adapter import from_langchain_messages

_LIFTED_SOURCE_KEY = "_last_lifted_source"
_LIFTED_SOURCE_VALUE = "subgraph_messages"


def lift_subgraph_messages(
    parent_trace: ExecutionTrace | dict[str, Any] | None,
    subgraph_messages: Sequence[Any],
    *,
    run_id: str | None = None,
    started_at: datetime | None = None,
    agent_name: str | None = None,
    metadata: dict[str, Any] | None = None,
    include_subgraph_thoughts: bool = False,
    subgraph_thoughts_key: str = "_subgraph_thoughts",
) -> ExecutionTrace:
    """Merge an inner graph's structured messages into a parent ``ExecutionTrace``.

    This is the preferred production path for nested graphs: consume the inner
    graph's ``AIMessage.tool_calls`` / ``ToolMessage`` objects directly instead
    of parsing textual debug logs. :func:`from_subgraph_thoughts` remains a
    best-effort fallback for the textual case and is deliberately **not**
    applied here unless explicitly opted into.

    Parameters
    ----------
    parent_trace:
        Existing trace to merge into. Accepts an ``ExecutionTrace``, a
        serialized ``dict`` (as produced by LangGraph checkpointing), or
        ``None`` to start a fresh trace.
    subgraph_messages:
        Sequence of LangChain-style message objects from the inner graph. No
        LangChain import is needed — duck typing is used throughout.
    run_id:
        Identifier for the run. Priority: this argument, then the parent
        trace's ``run_id``, then ``"subgraph-run"``.
    started_at:
        Timestamp attached to events converted from *subgraph_messages*
        (timezone-aware). Defaults to now (UTC). When *parent_trace* is
        provided, its ``started_at`` is preserved on the result.
    agent_name:
        Agent name attached to lifted ``MessageEvent`` and ``ToolCall``
        entries — typically the name of the subgraph/worker node.
    metadata:
        Extra metadata to merge into the result. Keys already present on the
        parent trace always win; caller keys fill the rest.
    include_subgraph_thoughts:
        Default ``False``. When ``True``, message metadata dicts are also
        scanned for textual thought lines (*subgraph_thoughts_key*) and parsed
        via the best-effort :func:`from_subgraph_thoughts` fallback. Leave off
        when structured tool messages are available.
    subgraph_thoughts_key:
        Metadata key holding textual thought lines when the fallback is
        enabled. Defaults to ``"_subgraph_thoughts"``.

    Returns
    -------
    ExecutionTrace
        A new trace containing all parent events plus the lifted subgraph
        events. Metadata rules: parent metadata is preserved verbatim (the
        child conversion's ``_source`` never overwrites it), caller *metadata*
        fills non-conflicting keys, and ``"_last_lifted_source"`` is set to
        ``"subgraph_messages"``.

    Notes
    -----
    - Inputs are never mutated; a new ``ExecutionTrace`` is returned.
    - Routing events, ``AgentCall`` events, and artifacts are never inferred
      from messages. Record those explicitly (e.g. via
      ``TraceBuilder.record_routing`` / ``record_agent_call``) at the point
      where routing or delegation actually happens.
    - A ``dict`` *parent_trace* that is not a valid serialized
      ``ExecutionTrace`` raises a validation error rather than being silently
      dropped.
    """
    base: ExecutionTrace | None
    if isinstance(parent_trace, ExecutionTrace):
        base = parent_trace
    elif isinstance(parent_trace, dict):
        base = ExecutionTrace(**parent_trace)
    else:
        base = None

    if run_id is not None:
        resolved_run_id = run_id
    elif base is not None:
        resolved_run_id = base.run_id
    else:
        resolved_run_id = "subgraph-run"

    child = from_langchain_messages(
        subgraph_messages,
        run_id=resolved_run_id,
        started_at=started_at,
        agent_name=agent_name,
        include_subgraph_thoughts=include_subgraph_thoughts,
        subgraph_thoughts_key=subgraph_thoughts_key,
    )

    if base is not None:
        merged = TraceBuilder.from_trace(base).merge_trace(child).build()
    else:
        merged = child

    # Metadata: child conversion keys first (minus its _source marker), then
    # caller-supplied keys, then parent keys — parent always wins. The helper
    # records its own marker under a non-conflicting key.
    final_meta: dict[str, Any] = dict(child.metadata)
    final_meta.pop("_source", None)
    if metadata:
        final_meta.update(metadata)
    if base is not None:
        final_meta.update(base.metadata)
    final_meta[_LIFTED_SOURCE_KEY] = _LIFTED_SOURCE_VALUE

    return merged.model_copy(update={"run_id": resolved_run_id, "metadata": final_meta})
