"""Agent layer types — events, tool definitions, results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal

from rho.ai.types import ImageContent, TextContent, ToolCall


# ═══════════════════════════════════════════════════════════════════
# AgentTool & AgentToolResult
# ═══════════════════════════════════════════════════════════════════

@dataclass
class AgentToolResult:
    """Result returned by tool execution."""
    content: list[TextContent | ImageContent] = field(default_factory=list)
    is_error: bool = False


@dataclass
class AgentTool:
    """A tool the agent can execute.

    Extends AI layer Tool: same name/description/parameters for the
    LLM to see, plus `execute` for the agent to call.
    """
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)   # JSON Schema
    execute: Callable[..., Awaitable[AgentToolResult]] | None = None
    label: str = ""


# ═══════════════════════════════════════════════════════════════════
# AgentEvent — what the agent yields during a run
# ═══════════════════════════════════════════════════════════════════

@dataclass
class AgentStart:
    type: Literal["agent_start"] = "agent_start"


@dataclass
class AgentEnd:
    type: Literal["agent_end"] = "agent_end"


@dataclass
class TurnStart:
    type: Literal["turn_start"] = "turn_start"


@dataclass
class TurnEnd:
    type: Literal["turn_end"] = "turn_end"
    message: Any = None           # AssistantMessage
    tool_results: list[Any] = field(default_factory=list)  # ToolResultMessage[]


@dataclass
class MessageStart:
    type: Literal["message_start"] = "message_start"
    message: Any = None           # Message


@dataclass
class MessageUpdate:
    type: Literal["message_update"] = "message_update"
    message: Any = None           # partial AssistantMessage
    ai_event: Any = None          # the AI layer event that triggered this update


@dataclass
class MessageEnd:
    type: Literal["message_end"] = "message_end"
    message: Any = None           # final Message


@dataclass
class ToolExecutionStart:
    type: Literal["tool_execution_start"] = "tool_execution_start"
    tool_call_id: str = ""
    tool_name: str = ""
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolExecutionEnd:
    type: Literal["tool_execution_end"] = "tool_execution_end"
    tool_call_id: str = ""
    tool_name: str = ""
    result: AgentToolResult = field(default_factory=AgentToolResult)
    is_error: bool = False
