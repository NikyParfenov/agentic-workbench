from pydantic import AwareDatetime, BaseModel, Field
from .events import (
    MessageEvent, AgentCall, ToolCall, ToolResult,
    RoutingEvent, ArtifactEvent, ErrorEvent,
)


class ExecutionTrace(BaseModel):
    run_id: str
    started_at: AwareDatetime
    finished_at: AwareDatetime | None = None
    messages: list[MessageEvent] = Field(default_factory=list)
    agent_calls: list[AgentCall] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    routing_events: list[RoutingEvent] = Field(default_factory=list)
    artifacts: list[ArtifactEvent] = Field(default_factory=list)
    errors: list[ErrorEvent] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    token_usage: int | None = None
