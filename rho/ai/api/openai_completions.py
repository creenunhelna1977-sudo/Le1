"""OpenAI Chat Completions API implementation.

Matches PI's openai-completions.ts — builds /v1/chat/completions requests,
parses SSE responses, yields typed AssistantMessageEvents.

Uses raw httpx (no openai SDK), giving full control over HTTP.
"""

from __future__ import annotations

import json
import time
from typing import Any, AsyncGenerator

from rho.ai.http_client import HttpClient, normalize_provider_error
from rho.ai.types import (
    AnthropicMessagesCompat,
    AssistantMessage,
    AssistantMessageEvent,
    Context,
    DoneEvent,
    ErrorEvent,
    ImageContent,
    Message,
    Model,
    SimpleStreamOptions,
    StartEvent,
    StopReason,
    StreamOptions,
    TextContent,
    TextDelta,
    TextEnd,
    TextStart,
    ThinkingContent,
    ThinkingDelta,
    ThinkingEnd,
    ThinkingStart,
    Tool,
    ToolCall,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
    ToolResultMessage,
    Usage,
)


# ═══════════════════════════════════════════════════════════════════
# Streaming JSON parsing (for partial tool call args)
# ═══════════════════════════════════════════════════════════════════

def _parse_streaming_json(text: str) -> dict[str, Any]:
    """Best-effort parse partial JSON. Returns complete dict or partial.

    For complete JSON strings, parses normally.
    For partial: attempts repair and returns best-effort result.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Attempt simple repair: close unclosed strings/braces
        repaired = text
        # Count braces
        open_braces = repaired.count("{") - repaired.count("}")
        open_brackets = repaired.count("[") - repaired.count("]")
        # Count unescaped quotes
        in_string = False
        for i, ch in enumerate(repaired):
            if ch == '"' and (i == 0 or repaired[i - 1] != "\\"):
                in_string = not in_string
        if in_string:
            repaired += '"'
        repaired += "]" * open_brackets + "}" * open_braces
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            return {}


# ═══════════════════════════════════════════════════════════════════
# Message conversion: Context → OpenAI Chat Completions format
# ═══════════════════════════════════════════════════════════════════

def _convert_messages(
    ctx: Context,
    model: Model,
    compat: dict[str, Any],
) -> list[dict[str, Any]]:
    """Convert rho Context messages to OpenAI Chat Completions format."""
    params: list[dict[str, Any]] = []

    # System prompt
    if ctx.system_prompt:
        use_developer = model.reasoning and compat.get("supports_developer_role", False)
        role = "developer" if use_developer else "system"
        params.append({"role": role, "content": ctx.system_prompt})

    for msg in ctx.messages:
        if msg.role == "user":
            if isinstance(msg.content, str):
                params.append({"role": "user", "content": msg.content})
            else:
                content_parts: list[dict[str, Any]] = []
                for block in msg.content:
                    if block.type == "text":
                        content_parts.append({"type": "text", "text": block.text})
                    elif block.type == "image":
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{block.mime_type};base64,{block.data}"},
                        })
                if content_parts:
                    params.append({"role": "user", "content": content_parts})

        elif msg.role == "assistant":
            assistant_msg: dict[str, Any] = {"role": "assistant"}
            text_parts: list[str] = []
            thinking_blocks: list[ThinkingContent] = []
            tool_calls: list[dict[str, Any]] = []

            for block in msg.content:
                if block.type == "text" and block.text.strip():
                    text_parts.append(block.text)
                elif block.type == "thinking" and block.thinking.strip():
                    thinking_blocks.append(block)
                elif block.type == "toolCall":
                    tool_calls.append({
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.arguments),
                        },
                    })

            # Handle thinking blocks
            req_think_as_text = compat.get("requires_thinking_as_text", False)
            if thinking_blocks:
                if req_think_as_text:
                    thinking_text = "\n\n".join(b.thinking for b in thinking_blocks)
                    text_parts.insert(0, thinking_text)
                else:
                    # Use reasoning_content field (DeepSeek etc.)
                    thinking_text = "\n".join(b.thinking for b in thinking_blocks)
                    sig = thinking_blocks[0].thinking_signature
                    if sig:
                        assistant_msg[sig] = thinking_text

            # Set content
            if not req_think_as_text and not thinking_blocks and not text_parts:
                content = None  # tool-only assistant message
            elif text_parts:
                # Always use plain string (not content array) for compatibility
                content = "\n".join(text_parts)
            else:
                content = ""

            assistant_msg["content"] = content
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls

            # DeepSeek requirement: empty reasoning_content on assistant messages
            if compat.get("requires_reasoning_content_on_assistant_messages") and model.reasoning:
                if "reasoning_content" not in assistant_msg:
                    assistant_msg["reasoning_content"] = ""

            # Skip empty assistant messages
            if not content and not tool_calls and not thinking_blocks:
                continue

            params.append(assistant_msg)

        elif msg.role == "toolResult":
            # Handle consecutive tool results
            text_result = ""
            has_images = False
            image_blocks: list[dict[str, Any]] = []

            for block in msg.content:
                if block.type == "text":
                    text_result += block.text + "\n"
                elif block.type == "image":
                    has_images = True
                    if model.input and "image" in model.input:
                        image_blocks.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{block.mime_type};base64,{block.data}"},
                        })

            tool_msg: dict[str, Any] = {
                "role": "tool",
                "content": text_result.strip() if text_result.strip() else (
                    "(see attached image)" if has_images else "(no tool output)"
                ),
                "tool_call_id": msg.tool_call_id,
            }
            if compat.get("requires_tool_result_name") and msg.tool_name:
                tool_msg["name"] = msg.tool_name
            params.append(tool_msg)

            # Images from tool results become follow-up user messages
            if image_blocks:
                if compat.get("requires_assistant_after_tool_result"):
                    params.append({"role": "assistant", "content": ""})
                params.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Attached image(s) from tool result:"},
                        *image_blocks,
                    ],
                })

    return params


def _convert_tools(tools: list[Tool], compat: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert rho tools to OpenAI Chat Completions format."""
    result: list[dict[str, Any]] = []
    for tool in tools:
        t: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
        if compat.get("supports_strict_mode", True):
            t["function"]["strict"] = False
        result.append(t)
    return result


def _has_tool_history(messages: list[Message]) -> bool:
    """Check if conversation contains tool calls or results."""
    for msg in messages:
        if msg.role == "toolResult":
            return True
        if msg.role == "assistant":
            for block in msg.content:
                if block.type == "toolCall":
                    return True
    return False


def _parse_usage(raw: dict[str, Any]) -> dict[str, int]:
    """Parse OpenAI token usage from chunk."""
    return {
        "input": (raw.get("prompt_tokens", 0) or 0)
            - (raw.get("prompt_tokens_details", {}).get("cached_tokens", 0) or 0)
            - (raw.get("prompt_tokens_details", {}).get("cache_write_tokens", 0) or 0),
        "output": raw.get("completion_tokens", 0) or 0,
        "cache_read": raw.get("prompt_tokens_details", {}).get("cached_tokens", 0) or 0,
        "cache_write": raw.get("prompt_tokens_details", {}).get("cache_write_tokens", 0) or 0,
        "reasoning": raw.get("completion_tokens_details", {}).get("reasoning_tokens", 0) or 0,
    }


def _update_cost(output: AssistantMessage, model: Model) -> None:
    """Calculate response cost from the model's per-million-token rates."""
    output.usage.cost.input = output.usage.input * model.cost.input / 1_000_000
    output.usage.cost.output = output.usage.output * model.cost.output / 1_000_000
    output.usage.cost.cache_read = output.usage.cache_read * model.cost.cache_read / 1_000_000
    output.usage.cost.cache_write = output.usage.cache_write * model.cost.cache_write / 1_000_000
    output.usage.cost.total = (
        output.usage.cost.input
        + output.usage.cost.output
        + output.usage.cost.cache_read
        + output.usage.cost.cache_write
    )


def _snapshot(output: AssistantMessage) -> AssistantMessage:
    """Freeze the partial message state attached to a stream event."""
    return output.model_copy(deep=True)


def _map_stop_reason(finish_reason: str | None) -> tuple[StopReason, str | None]:
    """Map OpenAI finish_reason to rho StopReason."""
    if finish_reason is None:
        return "stop", None
    mapping: dict[str, tuple[StopReason, str | None]] = {
        "stop": ("stop", None),
        "end": ("stop", None),
        "length": ("length", None),
        "function_call": ("toolUse", None),
        "tool_calls": ("toolUse", None),
        "content_filter": ("error", "Provider finish_reason: content_filter"),
    }
    if finish_reason in mapping:
        return mapping[finish_reason]
    return ("error", f"Provider finish_reason: {finish_reason}")


# ═══════════════════════════════════════════════════════════════════
# Compat auto-detection
# ═══════════════════════════════════════════════════════════════════

def _detect_compat(model: Model) -> dict[str, Any]:
    """Auto-detect compat settings from provider name and base_url."""
    provider = model.provider
    url = model.base_url

    is_deepseek = provider == "deepseek" or "deepseek.com" in url
    is_openrouter = provider == "openrouter" or "openrouter.ai" in url
    is_zai = "api.z.ai" in url or "open.bigmodel.cn" in url
    is_together = provider == "together" or "api.together" in url

    is_non_standard = (
        is_deepseek or is_openrouter or is_zai or is_together or
        "cerebras.ai" in url or "api.x.ai" in url or "chutes.ai" in url
    )

    return {
        "supports_store": not is_non_standard,
        "supports_developer_role": not is_non_standard,
        "supports_reasoning_effort": not is_deepseek and not is_zai and not is_together,
        "supports_usage_in_streaming": True,
        "max_tokens_field": "max_tokens" if is_deepseek else "max_completion_tokens",
        "requires_tool_result_name": False,
        "requires_assistant_after_tool_result": False,
        "requires_thinking_as_text": False,
        "requires_reasoning_content_on_assistant_messages": is_deepseek,
        "thinking_format": "deepseek" if is_deepseek else (
            "zai" if is_zai else (
                "together" if is_together else (
                    "openrouter" if is_openrouter else "openai"
                )
            )
        ),
        "supports_strict_mode": True,
        "supports_long_cache_retention": True,
        "session_affinity_format": "openrouter" if is_openrouter else "openai",
        "send_session_affinity_headers": False,
    }


def _resolve_compat(model: Model) -> dict[str, Any]:
    """Merge detected compat with model-level overrides."""
    detected = _detect_compat(model)
    model_compat = model.compat if hasattr(model, "compat") else {}
    return {**detected, **model_compat}


# ═══════════════════════════════════════════════════════════════════
# Stream function
# ═══════════════════════════════════════════════════════════════════

async def stream(
    model: Model,
    context: Context,
    options: StreamOptions | None = None,
) -> AsyncGenerator[AssistantMessageEvent, None]:
    """Stream an OpenAI Chat Completions response.

    Args:
        model: The model descriptor.
        context: The conversation context.
        options: Stream options (temperature, max_tokens, api_key, etc.).

    Yields:
        AssistantMessageEvent entries: start, text_*, thinking_*, toolcall_*, done/error.
    """
    opts = options or StreamOptions()
    compat = _resolve_compat(model)

    # Build output accumulator
    output = AssistantMessage(
        api=model.api,
        provider=model.provider,
        model=model.id,
    )

    # Build request body
    body = _build_request_body(model, context, opts, compat)

    # Build headers
    headers: dict[str, str] = {}
    if opts.api_key:
        headers["Authorization"] = f"Bearer {opts.api_key}"
    # Model-level headers
    if model.headers:
        headers.update(model.headers)
    for key, value in opts.headers.items():
        if value is not None:
            headers[key] = value

    client = HttpClient()

    try:
        # Emit start
        yield StartEvent(partial=_snapshot(output))

        # Track streaming blocks
        blocks: list[dict[str, Any]] = []  # mutable content blocks
        text_block: dict[str, Any] | None = None
        thinking_block: dict[str, Any] | None = None
        tool_blocks: dict[int, dict[str, Any]] = {}  # index → block

        has_finish_reason = False

        async for chunk in client.stream_sse(
            url=f"{model.base_url}/chat/completions",
            json_data=body,
            headers=headers,
            timeout_ms=opts.timeout_ms,
            max_retries=opts.max_retries,
            max_retry_delay_ms=opts.max_retry_delay_ms,
        ):
            if chunk.get("_done"):
                continue

            # Capture response id/model
            if not output.response_id:
                output.response_id = chunk.get("id")
            if chunk.get("model") and chunk["model"] != model.id:
                output.response_model = chunk["model"]

            # Parse usage
            if chunk.get("usage"):
                u = _parse_usage(chunk["usage"])
                output.usage.input = max(output.usage.input, u["input"])
                output.usage.output = max(output.usage.output, u["output"])
                output.usage.cache_read = max(output.usage.cache_read, u["cache_read"])
                output.usage.cache_write = max(output.usage.cache_write, u["cache_write"])
                output.usage.reasoning = max(output.usage.reasoning or 0, u["reasoning"])
                output.usage.total_tokens = (
                    output.usage.input + output.usage.output
                    + output.usage.cache_read + output.usage.cache_write
                )
                _update_cost(output, model)

            choices = chunk.get("choices", [])
            if not choices:
                continue

            choice = choices[0]
            delta = choice.get("delta", {})

            # Fallback usage in choice (some providers)
            if not chunk.get("usage") and "usage" in choice:
                u2 = _parse_usage(choice["usage"])
                output.usage.input = max(output.usage.input, u2["input"])
                output.usage.output = max(output.usage.output, u2["output"])
                output.usage.cache_read = max(output.usage.cache_read, u2["cache_read"])
                output.usage.cache_write = max(output.usage.cache_write, u2["cache_write"])
                output.usage.reasoning = max(output.usage.reasoning or 0, u2["reasoning"])
                output.usage.total_tokens = (
                    output.usage.input + output.usage.output
                    + output.usage.cache_read + output.usage.cache_write
                )
                _update_cost(output, model)

            # Finish reason
            finish = choice.get("finish_reason")
            if finish:
                stop_reason, err_msg = _map_stop_reason(finish)
                output.stop_reason = stop_reason
                if err_msg:
                    output.error_message = err_msg
                has_finish_reason = True

            # Text content
            text = delta.get("content")
            if text and isinstance(text, str) and text:
                if text_block is None:
                    text_block = {"type": "text", "text": ""}
                    blocks.append(text_block)
                    ce_idx = blocks.index(text_block)
                    output.content.append(TextContent(text=""))
                    yield TextStart(content_index=ce_idx, partial=_snapshot(output))

                text_block["text"] += text
                ce_idx = blocks.index(text_block)
                # Update output content
                output.content[ce_idx] = TextContent(text=text_block["text"])
                yield TextDelta(
                    content_index=ce_idx,
                    delta=text,
                    partial=_snapshot(output),
                )

            # Thinking/reasoning content
            reasoning = None
            reasoning_signature: str | None = None
            for field in ["reasoning_content", "reasoning", "reasoning_text"]:
                val = delta.get(field)
                if isinstance(val, str) and val:
                    reasoning = val
                    reasoning_signature = field
                    break

            if reasoning:
                if thinking_block is None:
                    thinking_block = {"type": "thinking", "thinking": "", "signature": ""}
                    blocks.append(thinking_block)
                    ce_idx = blocks.index(thinking_block)
                    output.content.append(ThinkingContent(
                        thinking="",
                        thinking_signature=reasoning_signature,
                    ))
                    yield ThinkingStart(content_index=ce_idx, partial=_snapshot(output))

                thinking_block["thinking"] += reasoning
                thinking_block["signature"] = reasoning_signature or thinking_block.get("signature", "")
                ce_idx = blocks.index(thinking_block)
                output.content[ce_idx] = ThinkingContent(
                    thinking=thinking_block["thinking"],
                    thinking_signature=thinking_block.get("signature") or None,
                )
                yield ThinkingDelta(
                    content_index=ce_idx,
                    delta=reasoning,
                    partial=_snapshot(output),
                )

            # Tool calls
            tool_deltas = delta.get("tool_calls", []) or []
            for td in tool_deltas:
                idx = td.get("index", 0)
                if idx not in tool_blocks:
                    tool_blocks[idx] = {
                        "type": "toolCall",
                        "id": td.get("id", ""),
                        "name": td.get("function", {}).get("name", ""),
                        "partial_args": "",
                        "arguments": {},
                    }
                    blocks.append(tool_blocks[idx])
                    ce_idx = blocks.index(tool_blocks[idx])
                    output.content.append(ToolCall(
                        id=tool_blocks[idx]["id"],
                        name=tool_blocks[idx]["name"],
                        arguments={},
                    ))
                    yield ToolCallStart(content_index=ce_idx, partial=_snapshot(output))

                block = tool_blocks[idx]
                if td.get("id") and not block["id"]:
                    block["id"] = td["id"]
                if td.get("function", {}).get("name") and not block["name"]:
                    block["name"] = td["function"]["name"]

                args_delta = td.get("function", {}).get("arguments", "")
                if args_delta:
                    block["partial_args"] += args_delta
                    block["arguments"] = _parse_streaming_json(block["partial_args"])
                    ce_idx = blocks.index(block)
                    output.content[ce_idx] = ToolCall(
                        id=block["id"],
                        name=block["name"],
                        arguments=block["arguments"],
                    )
                    yield ToolCallDelta(
                        content_index=ce_idx,
                        delta=args_delta,
                        partial=_snapshot(output),
                    )

        # Finish all open blocks
        for block in blocks:
            ce_idx = blocks.index(block)
            if block["type"] == "text":
                output.content[ce_idx] = TextContent(text=block["text"])
                yield TextEnd(
                    content_index=ce_idx,
                    content=block["text"],
                    partial=_snapshot(output),
                )
            elif block["type"] == "thinking":
                output.content[ce_idx] = ThinkingContent(
                    thinking=block["thinking"],
                    thinking_signature=block.get("signature") or None,
                )
                yield ThinkingEnd(
                    content_index=ce_idx,
                    content=block["thinking"],
                    partial=_snapshot(output),
                )
            elif block["type"] == "toolCall":
                # Strip partial_args before finalizing
                tc = ToolCall(
                    id=block["id"],
                    name=block["name"],
                    arguments=block["arguments"],
                )
                output.content[ce_idx] = tc
                yield ToolCallEnd(
                    content_index=ce_idx,
                    tool_call=tc,
                    partial=_snapshot(output),
                )

        if not has_finish_reason:
            raise RuntimeError("Stream ended without finish_reason")

        if output.stop_reason == "error":
            raise RuntimeError(output.error_message or "Provider returned an error stop reason")

        yield DoneEvent(reason=output.stop_reason, message=output)  # type: ignore[arg-type]

    except Exception as exc:
        output.stop_reason = "error"
        output.error_message = str(normalize_provider_error(exc))
        yield ErrorEvent(reason="error", error=output)
    finally:
        await client.close()


# ═══════════════════════════════════════════════════════════════════
# Request body builder
# ═══════════════════════════════════════════════════════════════════

def _build_request_body(
    model: Model,
    ctx: Context,
    opts: StreamOptions,
    compat: dict[str, Any],
) -> dict[str, Any]:
    """Build the OpenAI /v1/chat/completions JSON body."""
    messages = _convert_messages(ctx, model, compat)
    body: dict[str, Any] = {
        "model": model.id,
        "messages": messages,
        "stream": True,
    }

    # Stream options
    if compat.get("supports_usage_in_streaming", True):
        body["stream_options"] = {"include_usage": True}

    # Store
    if compat.get("supports_store", False):
        body["store"] = False

    # Max tokens
    max_tokens = opts.max_tokens if opts.max_tokens is not None else model.max_tokens
    if max_tokens is not None:
        field = compat.get("max_tokens_field", "max_completion_tokens")
        body[field] = max_tokens

    # Temperature
    if opts.temperature is not None:
        body["temperature"] = opts.temperature

    # Tools
    active_tools = ctx.tools or []
    if active_tools:
        body["tools"] = _convert_tools(active_tools, compat)
    elif _has_tool_history(ctx.messages):
        # Some proxy providers require tools param when conversation has tool calls
        body["tools"] = []

    # Thinking / reasoning
    thinking_format = compat.get("thinking_format", "openai")
    reasoning = getattr(opts, "reasoning", None)
    if reasoning and model.reasoning:
        mapped_reasoning = model.thinking_level_map.get(reasoning, reasoning)
        if thinking_format == "deepseek":
            body["thinking"] = {"type": "enabled"}
        elif thinking_format == "openai" and compat.get("supports_reasoning_effort", True):
            body["reasoning_effort"] = mapped_reasoning

    return body


# ═══════════════════════════════════════════════════════════════════
# Simple stream (reasoning-level facade)
# ═══════════════════════════════════════════════════════════════════

async def stream_simple(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AsyncGenerator[AssistantMessageEvent, None]:
    """Map a simple reasoning level through the shared request builder."""
    async for event in stream(model, context, options):
        yield event
