"""Stateful agent loop built on top of rho's AI layer."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from copy import deepcopy
from typing import Any

from rho.ai.models import Models
from rho.ai.types import (
    AssistantMessage,
    Context,
    DoneEvent,
    ErrorEvent,
    Message,
    Model,
    SimpleStreamOptions,
    StartEvent,
    TextDelta,
    TextEnd,
    TextStart,
    ThinkingDelta,
    ThinkingEnd,
    ThinkingStart,
    ToolCall,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
    ToolResultMessage,
    UserMessage,
)
from rho.agent.types import (
    AgentEnd,
    AgentEvent,
    AgentStart,
    AgentTool,
    AgentToolResult,
    MessageEnd,
    MessageStart,
    MessageUpdate,
    ToolExecutionEnd,
    ToolExecutionStart,
    TurnEnd,
    TurnStart,
)


_ASSISTANT_CONTENT_EVENTS = (
    TextStart,
    TextDelta,
    TextEnd,
    ThinkingStart,
    ThinkingDelta,
    ThinkingEnd,
    ToolCallStart,
    ToolCallDelta,
    ToolCallEnd,
)


class Agent:
    """A stateful, sequential tool-calling agent."""

    def __init__(
        self,
        models: Models,
        model: Model,
        *,
        system_prompt: str = "",
        tools: Sequence[AgentTool] = (),
        messages: Sequence[Message] = (),
        options: SimpleStreamOptions | None = None,
    ) -> None:
        self.models = models
        self.model = model
        self.system_prompt = system_prompt
        self.options = options.model_copy(deep=True) if options else SimpleStreamOptions()
        self._messages = list(messages)
        self._tools = self._index_tools(tools)
        self._is_running = False
        self._current_assistant: AssistantMessage | None = None
        self._current_assistant_started = False

    @staticmethod
    def _index_tools(tools: Sequence[AgentTool]) -> dict[str, AgentTool]:
        indexed: dict[str, AgentTool] = {}
        for tool in tools:
            if not tool.name:
                raise ValueError("Agent tool name must not be empty")
            if tool.name in indexed:
                raise ValueError(f"Duplicate agent tool name: {tool.name}")
            indexed[tool.name] = tool
        return indexed

    @property
    def messages(self) -> list[Message]:
        """Return a top-level copy of the conversation transcript."""
        return list(self._messages)

    @property
    def tools(self) -> list[AgentTool]:
        """Return a top-level copy of the active tools."""
        return list(self._tools.values())

    @property
    def is_running(self) -> bool:
        return self._is_running

    def reset(self, messages: Sequence[Message] = ()) -> None:
        """Replace the transcript when the agent is idle."""
        if self._is_running:
            raise RuntimeError("Cannot reset a running agent")
        self._messages = list(messages)

    async def run(
        self,
        prompt: str | UserMessage,
    ) -> AsyncGenerator[AgentEvent, None]:
        """Run the agent until the model produces a non-tool response."""
        self._enter_run()
        try:
            user_message = self._normalize_prompt(prompt)
            self._messages.append(user_message)

            yield AgentStart()
            turn = 1
            yield TurnStart(turn=turn)
            yield MessageStart(message=self._snapshot_message(user_message))
            yield MessageEnd(message=self._snapshot_message(user_message))

            while True:
                context = Context(
                    system_prompt=self.system_prompt,
                    messages=list(self._messages),
                    tools=[tool.to_ai_tool() for tool in self._tools.values()],
                )
                self._current_assistant = None
                self._current_assistant_started = False
                async for agent_event in self._stream_assistant(context):
                    yield agent_event
                assistant = self._current_assistant
                started = self._current_assistant_started
                if assistant is None:
                    assistant = self._error_message(
                        RuntimeError("AI stream ended without a terminal event")
                    )

                if not started:
                    yield MessageStart(message=self._snapshot_message(assistant))
                self._messages.append(assistant)
                yield MessageEnd(message=self._snapshot_message(assistant))

                if assistant.stop_reason in {"error", "aborted"}:
                    yield TurnEnd(message=self._snapshot_message(assistant))
                    yield AgentEnd(messages=self._snapshot_messages())
                    return

                tool_calls = [block for block in assistant.content if isinstance(block, ToolCall)]
                if not tool_calls:
                    yield TurnEnd(message=self._snapshot_message(assistant))
                    yield AgentEnd(messages=self._snapshot_messages())
                    return

                tool_results: list[ToolResultMessage] = []
                for tool_call in tool_calls:
                    yield ToolExecutionStart(
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.name,
                        args=deepcopy(tool_call.arguments),
                    )
                    if assistant.stop_reason == "length":
                        result = AgentToolResult(content=[
                            {"type": "text", "text": (
                                f'Tool "{tool_call.name}" was not executed because the '
                                "response hit the output token limit and its arguments may be truncated."
                            )},
                        ])
                        is_error = True
                    else:
                        result, is_error = await self._execute_tool(tool_call)

                    yield ToolExecutionEnd(
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.name,
                        result=result.model_copy(deep=True),
                        is_error=is_error,
                    )
                    tool_result = ToolResultMessage(
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.name,
                        content=result.content,
                        is_error=is_error,
                    )
                    self._messages.append(tool_result)
                    tool_results.append(tool_result)
                    yield MessageStart(message=self._snapshot_message(tool_result))
                    yield MessageEnd(message=self._snapshot_message(tool_result))

                yield TurnEnd(
                    message=self._snapshot_message(assistant),
                    tool_results=[result.model_copy(deep=True) for result in tool_results],
                )
                turn += 1
                yield TurnStart(turn=turn)
        finally:
            self._is_running = False

    def _enter_run(self) -> None:
        if self._is_running:
            raise RuntimeError("Agent is already running")
        self._is_running = True

    def _normalize_prompt(self, prompt: str | UserMessage) -> UserMessage:
        if isinstance(prompt, UserMessage):
            return prompt
        return UserMessage(content=prompt)

    @staticmethod
    def _snapshot_message(message: Message) -> Message:
        return message.model_copy(deep=True)

    def _snapshot_messages(self) -> list[Message]:
        return [self._snapshot_message(message) for message in self._messages]

    async def _stream_assistant(self, context: Context) -> AsyncGenerator[AgentEvent, None]:
        """Consume one AI response and translate its streaming events."""
        try:
            async for event in self._model_stream(context):
                if isinstance(event, StartEvent):
                    self._current_assistant_started = True
                    yield MessageStart(message=self._snapshot_message(event.partial))
                elif isinstance(event, _ASSISTANT_CONTENT_EVENTS):
                    yield MessageUpdate(
                        message=self._snapshot_message(event.partial),
                        ai_event=event,
                    )
                elif isinstance(event, DoneEvent):
                    self._current_assistant = event.message
                    break
                elif isinstance(event, ErrorEvent):
                    self._current_assistant = event.error
                    break
        except Exception as exc:
            self._current_assistant = self._error_message(exc)

    async def _execute_tool(self, tool_call: ToolCall) -> tuple[AgentToolResult, bool]:
        tool = self._tools.get(tool_call.name)
        if tool is None:
            return AgentToolResult(content=[{
                "type": "text",
                "text": f"Tool '{tool_call.name}' not found",
            }]), True
        if tool.execute is None:
            return AgentToolResult(content=[{
                "type": "text",
                "text": f"Tool '{tool_call.name}' has no execute function",
            }]), True

        try:
            result = await tool.execute(**tool_call.arguments)
            if not isinstance(result, AgentToolResult):
                raise TypeError("Tool execute() must return AgentToolResult")
            return result, False
        except Exception as exc:
            return AgentToolResult(content=[{
                "type": "text",
                "text": str(exc),
            }]), True

    def _error_message(self, error: Exception) -> AssistantMessage:
        return AssistantMessage(
            api=self.model.api,
            provider=self.model.provider,
            model=self.model.id,
            stop_reason="error",
            error_message=str(error),
        )

    async def _model_stream(self, context: Context) -> AsyncGenerator[Any, None]:
        """Call the normal Models API, with a narrow fake compatibility path for tests."""
        if isinstance(self.models, Models):
            stream = self.models.stream_simple(self.model, context, self.options)
        else:
            stream_method = getattr(self.models, "stream", None)
            if stream_method is None:
                stream_method = getattr(self.models, "stream_simple")
            stream = stream_method(self.model, context, self.options)
        async for event in stream:
            yield event

__all__ = ["Agent"]
