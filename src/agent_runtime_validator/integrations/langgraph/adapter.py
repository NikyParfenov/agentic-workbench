from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

try:
    from langgraph.graph import StateGraph  # noqa: F401
except ImportError:
    raise ImportError(
        "LangGraph integration requires langgraph. "
        "Install it with: pip install agent-runtime-validator[langgraph]"
    ) from None

from ...schema.events import ArtifactEvent
from ...schema.trace import ExecutionTrace
from ...trace_builder import TraceBuilder

# Metadata keys tracking how many state entries were already merged into the
# trace, so repeated calls (e.g. a ValidationNode running on every graph step)
# never duplicate previously merged events.
_MERGED_MESSAGES_KEY = "_merged_message_count"
_MERGED_ARTIFACTS_KEY = "_merged_artifact_count"


def _merged_count(trace: ExecutionTrace, key: str) -> int:
    value = trace.metadata.get(key, 0)
    if isinstance(value, int) and value >= 0:
        return value
    return 0


def state_to_trace(state: dict[str, Any], run_id: str | None = None) -> ExecutionTrace:
    return ExecutionTrace(
        run_id=run_id or state.get("run_id", "langgraph-run"),
        started_at=state.get("started_at", datetime.now(timezone.utc)),
        messages=state.get("_trace_messages", []),
        agent_calls=state.get("_trace_agent_calls", []),
        tool_calls=state.get("_trace_tool_calls", []),
        tool_results=state.get("_trace_tool_results", []),
        routing_events=state.get("_trace_routing_events", []),
        artifacts=state.get("_trace_artifacts", []),
        errors=state.get("_trace_errors", []),
        metadata=state.get("_trace_metadata", {}),
        token_usage=state.get("_trace_token_usage"),
    )


def get_trace_from_state(
    state: dict[str, Any], trace_key: str = "trace"
) -> ExecutionTrace | None:
    """Extract an ``ExecutionTrace`` from LangGraph state, handling all storage forms.

    Returns ``None`` when *trace_key* is not present in *state*.
    Handles three forms that appear in practice:

    - ``ExecutionTrace`` object (in-memory, before checkpointing)
    - ``dict`` (after LangGraph deserializes a checkpointed state)
    - absent key (graph never ran through a ``ValidationNode``)

    Useful at the end of a graph run to retrieve the trace for archiving::

        def save_run_trace(state: dict) -> dict:
            trace = get_trace_from_state(state)
            if trace is not None:
                save_trace(trace, f"traces/{trace.run_id}.json")
            return state
    """
    raw = state.get(trace_key)
    if raw is None:
        return None
    if isinstance(raw, ExecutionTrace):
        return raw
    if isinstance(raw, dict):
        return ExecutionTrace(**raw)
    return None


def build_trace_from_state(
    state: Mapping[str, Any],
    *,
    run_id: str | None = None,
    started_at: datetime | None = None,
    trace_key: str = "trace",
    messages_key: str = "messages",
    artifacts_key: str | None = "artifacts",
    agent_name: str | None = None,
    metadata: dict[str, Any] | None = None,
    include_subgraph_thoughts: bool = True,
    subgraph_thoughts_key: str = "_subgraph_thoughts",
) -> ExecutionTrace:
    """Build an :class:`ExecutionTrace` from LangGraph state by merging existing
    trace, messages, artifacts, and optional subgraph thoughts.

    Parameters
    ----------
    state:
        LangGraph state mapping. Not mutated by this function.
    run_id:
        Identifier for the run. Overrides all other sources when provided.
        Falls back in priority order: existing trace's ``run_id``,
        ``state["run_id"]``, then ``"langgraph-run"``.
    started_at:
        Trace start time (timezone-aware). Defaults to now (UTC).
    trace_key:
        State key that may hold an existing ``ExecutionTrace`` (or serialized
        dict). Defaults to ``"trace"``.
    messages_key:
        State key that holds a list of LangChain-style messages. Defaults to
        ``"messages"``.
    artifacts_key:
        State key that holds a list of artifacts. Pass ``None`` to disable
        artifact collection. Defaults to ``"artifacts"``.
    agent_name:
        Agent name attached to ``MessageEvent`` and ``ToolCall`` entries
        produced from messages.
    metadata:
        Extra metadata to merge into the trace. Keys already present on the
        trace always win; only missing keys are added.
    include_subgraph_thoughts:
        When ``True`` (default), subgraph thought lines are lifted from each
        message's metadata dicts and merged into the trace.
    subgraph_thoughts_key:
        The metadata key to look for subgraph thought lines. Defaults to
        ``"_subgraph_thoughts"``.

    Returns
    -------
    ExecutionTrace

    Notes
    -----
    - Routing events and agent calls are never inferred.
    - LangChain is not required; duck typing is used throughout.
    - The input *state* is never mutated.
    - Repeated calls are safe: the number of already-merged messages and
      artifacts is tracked in ``trace.metadata`` (``"_merged_message_count"``,
      ``"_merged_artifact_count"``), so passing the returned trace back in via
      ``state[trace_key]`` — as ``ValidationNode`` does — only merges entries
      added since the previous call.
    """
    ts: datetime = started_at if started_at is not None else datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # Step 1: resolve base trace (may hold run_id for fallback)
    # ------------------------------------------------------------------
    raw_trace = state.get(trace_key)
    base: ExecutionTrace | None = None

    if isinstance(raw_trace, ExecutionTrace):
        base = raw_trace
    elif isinstance(raw_trace, dict):
        try:
            base = ExecutionTrace(**raw_trace)
        except Exception:
            base = None

    # ------------------------------------------------------------------
    # Step 2: resolve run_id
    # ------------------------------------------------------------------
    resolved_run_id: str
    if run_id is not None:
        resolved_run_id = run_id
    elif base is not None:
        resolved_run_id = base.run_id
    else:
        state_run_id = state.get("run_id")
        if state_run_id is not None:
            resolved_run_id = str(state_run_id)
        else:
            resolved_run_id = "langgraph-run"

    # ------------------------------------------------------------------
    # Step 3: build initial trace from base (or empty)
    # ------------------------------------------------------------------
    if base is not None:
        # Re-use the existing trace, preserving its run_id with the resolved one.
        if base.run_id != resolved_run_id:
            # Rebuild with the resolved run_id.
            working_trace = ExecutionTrace(
                run_id=resolved_run_id,
                started_at=base.started_at,
                finished_at=base.finished_at,
                messages=list(base.messages),
                agent_calls=list(base.agent_calls),
                tool_calls=list(base.tool_calls),
                tool_results=list(base.tool_results),
                routing_events=list(base.routing_events),
                artifacts=list(base.artifacts),
                errors=list(base.errors),
                metadata=dict(base.metadata),
                token_usage=base.token_usage,
            )
        else:
            working_trace = base
    else:
        working_trace = ExecutionTrace(run_id=resolved_run_id, started_at=ts)

    # ------------------------------------------------------------------
    # Step 4: merge messages if present (only entries not merged before)
    # ------------------------------------------------------------------
    raw_messages = state.get(messages_key)
    merged_message_mark: int | None = None
    if isinstance(raw_messages, list):
        already_merged = _merged_count(working_trace, _MERGED_MESSAGES_KEY)
        new_messages = raw_messages[min(already_merged, len(raw_messages)):]
        merged_message_mark = len(raw_messages)
        if new_messages:
            # Lazy import to avoid circular imports at module level.
            from .langchain_adapter import from_langchain_messages

            messages_trace = from_langchain_messages(
                new_messages,
                run_id=resolved_run_id,
                started_at=ts,
                agent_name=agent_name,
                include_subgraph_thoughts=include_subgraph_thoughts,
                subgraph_thoughts_key=subgraph_thoughts_key,
            )
            working_trace = TraceBuilder.from_trace(working_trace).merge_trace(messages_trace).build()

    # ------------------------------------------------------------------
    # Step 5: merge artifacts if present (only entries not merged before)
    # ------------------------------------------------------------------
    merged_artifact_mark: int | None = None
    if artifacts_key is not None:
        raw_artifacts = state.get(artifacts_key)
        if isinstance(raw_artifacts, list):
            already_merged = _merged_count(working_trace, _MERGED_ARTIFACTS_KEY)
            new_artifacts = raw_artifacts[min(already_merged, len(raw_artifacts)):]
            merged_artifact_mark = len(raw_artifacts)
            collected: list[ArtifactEvent] = []
            for item in new_artifacts:
                if isinstance(item, ArtifactEvent):
                    collected.append(item)
                elif isinstance(item, dict):
                    try:
                        collected.append(ArtifactEvent(**item))
                    except Exception:
                        pass
                # Other types are silently ignored.
            if collected:
                builder = TraceBuilder.from_trace(working_trace)
                for artifact in collected:
                    builder.record_artifact(
                        artifact_id=artifact.artifact_id,
                        artifact_type=artifact.artifact_type,
                        content=artifact.content,
                        agent_name=artifact.agent_name,
                        timestamp=artifact.timestamp,
                        metadata=artifact.metadata,
                    )
                working_trace = builder.build()

    # ------------------------------------------------------------------
    # Step 6: merge provided metadata (existing keys win) and record marks
    # ------------------------------------------------------------------
    final_meta = dict(working_trace.metadata)
    if metadata:
        for k, v in metadata.items():
            final_meta.setdefault(k, v)
    if merged_message_mark is not None:
        final_meta[_MERGED_MESSAGES_KEY] = merged_message_mark
    if merged_artifact_mark is not None:
        final_meta[_MERGED_ARTIFACTS_KEY] = merged_artifact_mark

    if final_meta != working_trace.metadata:
        working_trace = ExecutionTrace(
            run_id=working_trace.run_id,
            started_at=working_trace.started_at,
            finished_at=working_trace.finished_at,
            messages=list(working_trace.messages),
            agent_calls=list(working_trace.agent_calls),
            tool_calls=list(working_trace.tool_calls),
            tool_results=list(working_trace.tool_results),
            routing_events=list(working_trace.routing_events),
            artifacts=list(working_trace.artifacts),
            errors=list(working_trace.errors),
            metadata=final_meta,
            token_usage=working_trace.token_usage,
        )

    return working_trace
