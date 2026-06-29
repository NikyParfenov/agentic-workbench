from typing import Literal
from pydantic import AwareDatetime, BaseModel, Field


class MessageEvent(BaseModel):
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    agent_name: str | None = None
    timestamp: AwareDatetime
    metadata: dict = Field(default_factory=dict)


class AgentCall(BaseModel):
    caller: str
    callee: str
    input: str
    output: str | None = None
    timestamp: AwareDatetime
    metadata: dict = Field(default_factory=dict)


class ToolCall(BaseModel):
    tool_name: str
    args: dict = Field(default_factory=dict)
    agent_name: str | None = None
    call_id: str
    timestamp: AwareDatetime
    metadata: dict = Field(default_factory=dict)


class ToolResult(BaseModel):
    call_id: str
    tool_name: str
    output: str | None = None
    error: str | None = None
    timestamp: AwareDatetime
    metadata: dict = Field(default_factory=dict)


class RoutingEvent(BaseModel):
    from_agent: str
    to_agent: str
    reason: str | None = None
    timestamp: AwareDatetime
    metadata: dict = Field(default_factory=dict)


class ArtifactEvent(BaseModel):
    artifact_id: str
    artifact_type: str
    content: str
    agent_name: str | None = None
    timestamp: AwareDatetime
    metadata: dict = Field(default_factory=dict)


class ErrorEvent(BaseModel):
    error_type: str
    message: str
    agent_name: str | None = None
    timestamp: AwareDatetime
    metadata: dict = Field(default_factory=dict)
