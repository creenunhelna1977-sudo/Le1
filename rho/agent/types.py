"""Types for the rho agent layer.

Agent events are Pydantic models so they follow the same discriminated-union
protocol as the AI layer. Tools stay dataclasses because they contain a Python
callable and are runtime configuration rather than transcript data.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from rho.ai.types import (
    AssistantMessage,
    AssistantMessageEvent,
    ImageContent,
    Message,
    TextContent,
    Tool,
    ToolCall,
    ToolResultMessage,
)


ToolResultContent = Annotated[
    Union[TextContent, ImageContent],
    Field(discriminator="type"),
]


class AgentToolResult(BaseModel):
    """Successful value returned by an agent tool."""

    model_config = ConfigDict(extra="forbid")

    content: list[ToolResultContent] = Field(default_factory=list)


@dataclass(frozen=True, slots=True)
class AgentTool:
    """A model-visible tool with a Python execution function."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    execute: Callable[..., Awaitable[AgentToolResult]] | None = None

    def to_ai_tool(self) -> Tool:
        """Convert the executable tool to the AI layer tool definition."""
        return Tool(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )


class AgentStart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["agent_start"] = "agent_start"


class AgentEnd(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["agent_end"] = "agent_end"
    messages: list[Message] = Field(default_factory=list)


class TurnStart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["turn_start"] = "turn_start"
    turn: int = 1


class TurnEnd(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["turn_end"] = "turn_end"
    message: AssistantMessage
    tool_results: list[ToolResultMessage] = Field(default_factory=list)


class MessageStart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["message_start"] = "message_start"
    message: Message


class MessageUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["message_update"] = "message_update"
    message: AssistantMessage
    ai_event: AssistantMessageEvent


class MessageEnd(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["message_end"] = "message_end"
    message: Message


class ToolExecutionStart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["tool_execution_start"] = "tool_execution_start"
    tool_call_id: str
    tool_name: str
    args: dict[str, Any] = Field(default_factory=dict)


class ToolExecutionEnd(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["tool_execution_end"] = "tool_execution_end"
    tool_call_id: str
    tool_name: str
    result: AgentToolResult
    is_error: bool = False


AgentEvent = Annotated[
    Union[
        AgentStart,
        AgentEnd,
        TurnStart,
        TurnEnd,
        MessageStart,
        MessageUpdate,
        MessageEnd,
        ToolExecutionStart,
        ToolExecutionEnd,
    ],
    Field(discriminator="type"),
]


__all__ = [
    "AgentEnd",
    "AgentEvent",
    "AgentStart",
    "AgentTool",
    "AgentToolResult",
    "MessageEnd",
    "MessageStart",
    "MessageUpdate",
    "ToolExecutionEnd",
    "ToolExecutionStart",
    "TurnEnd",
    "TurnStart",
]
