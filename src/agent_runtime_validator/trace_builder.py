"""TraceBuilder — ergonomic incremental construction of ExecutionTrace.

Usage::

    builder = TraceBuilder(run_id="run-123")
    builder.record_tool_call("search", call_id="c1", args={"q": "acme"})
    builder.record_tool_result("c1", "search", output="found it")
    trace = builder.build()

Fluent chaining is supported::

    trace = (
        TraceBuilder(run_id="run-123")
        .record_routing("supervisor", "researcher")
        .record_tool_call("search", call_id="c1", args={"q": "acme"})
        .build()
    )
"""

from datetime import datetime, timezone
from typing import Literal

from .schema.events import (
    AgentCall,
    ArtifactEvent,
    ErrorEvent,
    MessageEvent,
    RoutingEvent,
    ToolCall,
    ToolResult,
)
from .schema.trace import ExecutionTrace


def _now() -> datetime:
    return datetime.now(timezone.utc)


class TraceBuilder:
    """Builds an :class:`ExecutionTrace` incrementally.

    All ``record_*`` methods return ``self`` for fluent chaining.
    Events are stored in internal lists; :meth:`build` returns a fresh
    ``ExecutionTrace`` without mutating the builder.
    """

    def __init__(self, run_id: str, started_at: datetime | None = None) -> None:
        self._run_id = run_id
        self._started_at: datetime = started_at if started_at is not None else _now()
        self._finished_at: datetime | None = None
        self._messages: list[MessageEvent] = []
        self._agent_calls: list[AgentCall] = []
        self._tool_calls: list[ToolCall] = []
        self._tool_results: list[ToolResult] = []
        self._routing_events: list[RoutingEvent] = []
        self._artifacts: list[ArtifactEvent] = []
        self._errors: list[ErrorEvent] = []
        self._metadata: dict = {}
        self._token_usage: int | None = None

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_trace(cls, trace: ExecutionTrace) -> "TraceBuilder":
        """Create a builder pre-populated with all events from *trace*."""
        builder = cls(run_id=trace.run_id, started_at=trace.started_at)
        builder._finished_at = trace.finished_at
        builder._messages = list(trace.messages)
        builder._agent_calls = list(trace.agent_calls)
        builder._tool_calls = list(trace.tool_calls)
        builder._tool_results = list(trace.tool_results)
        builder._routing_events = list(trace.routing_events)
        builder._artifacts = list(trace.artifacts)
        builder._errors = list(trace.errors)
        builder._metadata = dict(trace.metadata)
        builder._token_usage = trace.token_usage
        return builder

    @classmethod
    def merge(cls, parent: ExecutionTrace, child: ExecutionTrace) -> "TraceBuilder":
        """Create a builder that combines *parent* and *child* events.

        The builder starts from *parent* (preserving its ``run_id`` and
        ``started_at``), then appends *child* events on top. Child metadata
        overwrites parent metadata on key conflicts.
        """
        return cls.from_trace(parent).merge_trace(child)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def merge_trace(self, other: ExecutionTrace) -> "TraceBuilder":
        """Append all events from *other* to this builder."""
        self._messages.extend(other.messages)
        self._agent_calls.extend(other.agent_calls)
        self._tool_calls.extend(other.tool_calls)
        self._tool_results.extend(other.tool_results)
        self._routing_events.extend(other.routing_events)
        self._artifacts.extend(other.artifacts)
        self._errors.extend(other.errors)
        self._metadata.update(other.metadata)
        if other.token_usage is not None:
            self._token_usage = (self._token_usage or 0) + other.token_usage
        return self

    def set_finished_at(self, finished_at: datetime) -> "TraceBuilder":
        self._finished_at = finished_at
        return self

    def set_token_usage(self, tokens: int) -> "TraceBuilder":
        self._token_usage = tokens
        return self

    def update_metadata(self, **kwargs: object) -> "TraceBuilder":
        self._metadata.update(kwargs)
        return self

    # ------------------------------------------------------------------
    # Event recorders
    # ------------------------------------------------------------------

    def record_message(
        self,
        role: Literal["user", "assistant", "system", "tool"],
        content: str,
        agent_name: str | None = None,
        timestamp: datetime | None = None,
        metadata: dict | None = None,
    ) -> "TraceBuilder":
        self._messages.append(
            MessageEvent(
                role=role,
                content=content,
                agent_name=agent_name,
                timestamp=timestamp or _now(),
                metadata=metadata or {},
            )
        )
        return self

    def record_agent_call(
        self,
        caller: str,
        callee: str,
        input: str,
        output: str | None = None,
        timestamp: datetime | None = None,
        metadata: dict | None = None,
    ) -> "TraceBuilder":
        self._agent_calls.append(
            AgentCall(
                caller=caller,
                callee=callee,
                input=input,
                output=output,
                timestamp=timestamp or _now(),
                metadata=metadata or {},
            )
        )
        return self

    def record_tool_call(
        self,
        tool_name: str,
        call_id: str,
        args: dict | None = None,
        agent_name: str | None = None,
        timestamp: datetime | None = None,
        metadata: dict | None = None,
    ) -> "TraceBuilder":
        self._tool_calls.append(
            ToolCall(
                tool_name=tool_name,
                call_id=call_id,
                args=args or {},
                agent_name=agent_name,
                timestamp=timestamp or _now(),
                metadata=metadata or {},
            )
        )
        return self

    def record_tool_result(
        self,
        call_id: str,
        tool_name: str,
        output: str | None = None,
        error: str | None = None,
        timestamp: datetime | None = None,
        metadata: dict | None = None,
    ) -> "TraceBuilder":
        self._tool_results.append(
            ToolResult(
                call_id=call_id,
                tool_name=tool_name,
                output=output,
                error=error,
                timestamp=timestamp or _now(),
                metadata=metadata or {},
            )
        )
        return self

    def record_routing(
        self,
        from_agent: str,
        to_agent: str,
        reason: str | None = None,
        timestamp: datetime | None = None,
        metadata: dict | None = None,
    ) -> "TraceBuilder":
        self._routing_events.append(
            RoutingEvent(
                from_agent=from_agent,
                to_agent=to_agent,
                reason=reason,
                timestamp=timestamp or _now(),
                metadata=metadata or {},
            )
        )
        return self

    def record_artifact(
        self,
        artifact_id: str,
        artifact_type: str,
        content: str,
        agent_name: str | None = None,
        timestamp: datetime | None = None,
        metadata: dict | None = None,
    ) -> "TraceBuilder":
        self._artifacts.append(
            ArtifactEvent(
                artifact_id=artifact_id,
                artifact_type=artifact_type,
                content=content,
                agent_name=agent_name,
                timestamp=timestamp or _now(),
                metadata=metadata or {},
            )
        )
        return self

    def record_error(
        self,
        error_type: str,
        message: str,
        agent_name: str | None = None,
        timestamp: datetime | None = None,
        metadata: dict | None = None,
    ) -> "TraceBuilder":
        self._errors.append(
            ErrorEvent(
                error_type=error_type,
                message=message,
                agent_name=agent_name,
                timestamp=timestamp or _now(),
                metadata=metadata or {},
            )
        )
        return self

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> ExecutionTrace:
        """Return a new :class:`ExecutionTrace` from the current builder state.

        Calling :meth:`build` multiple times is safe — each call returns a
        fresh snapshot; the builder itself is not consumed.
        """
        return ExecutionTrace(
            run_id=self._run_id,
            started_at=self._started_at,
            finished_at=self._finished_at,
            messages=list(self._messages),
            agent_calls=list(self._agent_calls),
            tool_calls=list(self._tool_calls),
            tool_results=list(self._tool_results),
            routing_events=list(self._routing_events),
            artifacts=list(self._artifacts),
            errors=list(self._errors),
            metadata=dict(self._metadata),
            token_usage=self._token_usage,
        )
