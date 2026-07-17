"""Core type definitions for rho AI layer.

All types use pydantic v2 with discriminated unions, matching PI's
TypeScript discriminated union event/message protocol.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


# ═══════════════════════════════════════════════════════════════════
# API & Provider identity
# ═══════════════════════════════════════════════════════════════════

ApiLit = Literal["openai-completions", "anthropic-messages"]
ProviderId = str
ThinkingLevel = Literal["minimal", "low", "medium", "high"]
StopReason = Literal["stop", "length", "toolUse", "error", "aborted"]


# ═══════════════════════════════════════════════════════════════════
# Content blocks
# ═══════════════════════════════════════════════════════════════════

class TextContent(BaseModel):
    """Plain text content block."""
    model_config = ConfigDict(extra="forbid")

    type: Literal["text"] = "text"
    text: str
    text_signature: str | None = None


class ThinkingContent(BaseModel):
    """Reasoning / thinking content block."""
    model_config = ConfigDict(extra="forbid")

    type: Literal["thinking"] = "thinking"
    thinking: str
    thinking_signature: str | None = None
    redacted: bool = False


class ImageContent(BaseModel):
    """Base64-encoded image content block."""
    model_config = ConfigDict(extra="forbid")

    type: Literal["image"] = "image"
    data: str  # base64-encoded
    mime_type: str  # e.g. "image/png"


class ToolCall(BaseModel):
    """A tool/function call requested by the assistant."""
    model_config = ConfigDict(extra="forbid")

    type: Literal["toolCall"] = "toolCall"
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


ContentBlock = Annotated[
    Union[TextContent, ThinkingContent, ImageContent, ToolCall],
    Field(discriminator="type"),
]


# ═══════════════════════════════════════════════════════════════════
# Tool definitions
# ═══════════════════════════════════════════════════════════════════

class Tool(BaseModel):
    """Tool definition. `parameters` is a JSON Schema object."""
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════
# Messages
# ═══════════════════════════════════════════════════════════════════

class UserMessage(BaseModel):
    """A message from the user."""
    model_config = ConfigDict(extra="forbid")

    role: Literal["user"] = "user"
    content: str | list[Annotated[Union[TextContent, ImageContent], Field(discriminator="type")]]
    timestamp: float = Field(default_factory=lambda: __import__("time").time() * 1000)


class AssistantMessage(BaseModel):
    """A complete or partial assistant response."""
    model_config = ConfigDict(extra="forbid")

    role: Literal["assistant"] = "assistant"
    content: list[ContentBlock] = Field(default_factory=list)
    api: ApiLit | str = ""
    provider: ProviderId = ""
    model: str = ""
    response_model: str | None = None
    response_id: str | None = None
    usage: Usage = Field(default_factory=lambda: Usage())
    stop_reason: StopReason = "stop"
    error_message: str | None = None
    timestamp: float = Field(default_factory=lambda: __import__("time").time() * 1000)


class ToolResultMessage(BaseModel):
    """Result of a tool execution."""
    model_config = ConfigDict(extra="forbid")

    role: Literal["toolResult"] = "toolResult"
    tool_call_id: str
    tool_name: str
    content: list[Annotated[Union[TextContent, ImageContent], Field(discriminator="type")]] = Field(default_factory=list)
    added_tool_names: list[str] = Field(default_factory=list)
    is_error: bool = False
    timestamp: float = Field(default_factory=lambda: __import__("time").time() * 1000)


Message = Annotated[
    Union[UserMessage, AssistantMessage, ToolResultMessage],
    Field(discriminator="role"),
]


# ═══════════════════════════════════════════════════════════════════
# Usage & Cost
# ═══════════════════════════════════════════════════════════════════

class Cost(BaseModel):
    """Token cost breakdown in USD."""
    model_config = ConfigDict(extra="forbid")

    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0
    total: float = 0.0


class Usage(BaseModel):
    """Token usage for a response."""
    model_config = ConfigDict(extra="forbid")

    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    cache_write_1h: int | None = None
    reasoning: int | None = None
    total_tokens: int = 0
    cost: Cost = Field(default_factory=Cost)


class ModelCostRates(BaseModel):
    """Per-million-token pricing."""
    model_config = ConfigDict(extra="forbid")

    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0


# ═══════════════════════════════════════════════════════════════════
# Context
# ═══════════════════════════════════════════════════════════════════

class Context(BaseModel):
    """The full context sent to an LLM."""
    model_config = ConfigDict(extra="forbid")

    system_prompt: str | None = None
    messages: list[Message] = Field(default_factory=list)
    tools: list[Tool] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════
# Stream options
# ═══════════════════════════════════════════════════════════════════

class StreamOptions(BaseModel):
    """Base options for a streaming LLM request."""
    model_config = ConfigDict(extra="forbid")

    temperature: float | None = None
    max_tokens: int | None = None
    api_key: str | None = None
    headers: dict[str, str | None] = Field(default_factory=dict)
    timeout_ms: int | None = None
    max_retries: int = 0
    max_retry_delay_ms: int = 60_000
    cache_retention: Literal["none", "short", "long"] = "short"
    session_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)


class SimpleStreamOptions(StreamOptions):
    """Stream options with a simplified reasoning level."""
    reasoning: ThinkingLevel | None = None


# ═══════════════════════════════════════════════════════════════════
# Compat / compatibility overrides
# ═══════════════════════════════════════════════════════════════════

class OpenAICompletionsCompat(BaseModel):
    """Compat overrides for OpenAI-compatible completions APIs."""
    model_config = ConfigDict(extra="forbid")

    supports_store: bool = False
    supports_developer_role: bool = False
    supports_reasoning_effort: bool = True
    supports_usage_in_streaming: bool = True
    max_tokens_field: Literal["max_completion_tokens", "max_tokens"] = "max_completion_tokens"
    requires_tool_result_name: bool = False
    requires_assistant_after_tool_result: bool = False
    requires_thinking_as_text: bool = False
    requires_reasoning_content_on_assistant_messages: bool = False
    thinking_format: Literal[
        "openai", "openrouter", "deepseek", "together", "zai",
        "qwen", "qwen-chat-template", "chat-template", "string-thinking", "ant-ling",
    ] = "openai"
    supports_strict_mode: bool = True
    supports_long_cache_retention: bool = True
    session_affinity_format: Literal["openai", "openai-nosession", "openrouter"] = "openai"
    send_session_affinity_headers: bool = False


class AnthropicMessagesCompat(BaseModel):
    """Compat overrides for Anthropic-compatible Messages APIs."""
    model_config = ConfigDict(extra="forbid")

    supports_eager_tool_input_streaming: bool = True
    supports_long_cache_retention: bool = True
    send_session_affinity_headers: bool = False
    supports_cache_control_on_tools: bool = True
    supports_temperature: bool = True
    force_adaptive_thinking: bool = False
    allow_empty_signature: bool = False
    supports_tool_references: bool = False


# ═══════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════

class Model(BaseModel):
    """Descriptor for an LLM model."""
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str = ""
    api: ApiLit | str
    provider: ProviderId
    base_url: str
    reasoning: bool = False
    thinking_level_map: dict[str, str | None] = Field(default_factory=dict)
    input: list[Literal["text", "image"]] = Field(default_factory=lambda: ["text"])
    cost: ModelCostRates = Field(default_factory=ModelCostRates)
    context_window: int = 128_000
    max_tokens: int = 4096
    headers: dict[str, str] = Field(default_factory=dict)
    compat: dict[str, Any] = Field(default_factory=dict)

    @property
    def openai_compat(self) -> OpenAICompletionsCompat:
        """Resolve OpenAI completions compat settings."""
        return OpenAICompletionsCompat(**(self.compat if self.api == "openai-completions" else {}))

    @property
    def anthropic_compat(self) -> AnthropicMessagesCompat:
        """Resolve Anthropic messages compat settings."""
        return AnthropicMessagesCompat(**(self.compat if self.api == "anthropic-messages" else {}))


# ═══════════════════════════════════════════════════════════════════
# Streaming events
# ═══════════════════════════════════════════════════════════════════

class StartEvent(BaseModel):
    """Stream has started."""
    model_config = ConfigDict(extra="forbid")
    type: Literal["start"] = "start"
    partial: AssistantMessage


class TextStart(BaseModel):
    """A new text block is starting."""
    model_config = ConfigDict(extra="forbid")
    type: Literal["text_start"] = "text_start"
    content_index: int
    partial: AssistantMessage


class TextDelta(BaseModel):
    """Incremental text content."""
    model_config = ConfigDict(extra="forbid")
    type: Literal["text_delta"] = "text_delta"
    content_index: int
    delta: str
    partial: AssistantMessage


class TextEnd(BaseModel):
    """A text block is complete."""
    model_config = ConfigDict(extra="forbid")
    type: Literal["text_end"] = "text_end"
    content_index: int
    content: str
    partial: AssistantMessage


class ThinkingStart(BaseModel):
    """A new thinking block is starting."""
    model_config = ConfigDict(extra="forbid")
    type: Literal["thinking_start"] = "thinking_start"
    content_index: int
    partial: AssistantMessage


class ThinkingDelta(BaseModel):
    """Incremental thinking content."""
    model_config = ConfigDict(extra="forbid")
    type: Literal["thinking_delta"] = "thinking_delta"
    content_index: int
    delta: str
    partial: AssistantMessage


class ThinkingEnd(BaseModel):
    """A thinking block is complete."""
    model_config = ConfigDict(extra="forbid")
    type: Literal["thinking_end"] = "thinking_end"
    content_index: int
    content: str
    partial: AssistantMessage


class ToolCallStart(BaseModel):
    """A new tool call block is starting."""
    model_config = ConfigDict(extra="forbid")
    type: Literal["toolcall_start"] = "toolcall_start"
    content_index: int
    partial: AssistantMessage


class ToolCallDelta(BaseModel):
    """Incremental tool call argument JSON."""
    model_config = ConfigDict(extra="forbid")
    type: Literal["toolcall_delta"] = "toolcall_delta"
    content_index: int
    delta: str
    partial: AssistantMessage


class ToolCallEnd(BaseModel):
    """A tool call block is complete with parsed arguments."""
    model_config = ConfigDict(extra="forbid")
    type: Literal["toolcall_end"] = "toolcall_end"
    content_index: int
    tool_call: ToolCall
    partial: AssistantMessage


class DoneEvent(BaseModel):
    """Stream completed successfully."""
    model_config = ConfigDict(extra="forbid")
    type: Literal["done"] = "done"
    reason: Literal["stop", "length", "toolUse"]
    message: AssistantMessage


class ErrorEvent(BaseModel):
    """Stream terminated with an error."""
    model_config = ConfigDict(extra="forbid")
    type: Literal["error"] = "error"
    reason: Literal["aborted", "error"]
    error: AssistantMessage


AssistantMessageEvent = Annotated[
    Union[
        StartEvent,
        TextStart, TextDelta, TextEnd,
        ThinkingStart, ThinkingDelta, ThinkingEnd,
        ToolCallStart, ToolCallDelta, ToolCallEnd,
        DoneEvent,
        ErrorEvent,
    ],
    Field(discriminator="type"),
]
