"""Anthropic Messages API implementation.

Matches PI's anthropic-messages.ts — builds /v1/messages requests,
parses Anthropic-specific SSE events, yields typed AssistantMessageEvents.

Uses raw httpx (no anthropic SDK), giving full control over HTTP.
"""

from __future__ import annotations

import json
import time
from typing import Any, AsyncGenerator

from rho.ai.http_client import HttpClient, normalize_provider_error
from rho.ai.types import (
    AssistantMessage,
    AssistantMessageEvent,
    Context,
    ContentBlock,
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
    UserMessage,
    Usage,
)


# ═══════════════════════════════════════════════════════════════════
# Streaming JSON parsing
# ═══════════════════════════════════════════════════════════════════

def _parse_streaming_json(text: str) -> dict[str, Any]:
    """Best-effort parse partial JSON for tool call arguments."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        repaired = text
        open_braces = repaired.count("{") - repaired.count("}")
        open_brackets = repaired.count("[") - repaired.count("]")
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


# ═══════════════════════════════════════════════════════════════════
# Message conversion: Context → Anthropic Messages format
# ═══════════════════════════════════════════════════════════════════

def _convert_content_blocks(
    content: str | list[Any],
) -> str | list[dict[str, Any]]:
    """Convert rho user content to Anthropic format.

    Returns a string for text-only, or an array of content blocks for mixed.
    """
    if isinstance(content, str):
        return content

    has_images = any(c.type == "image" for c in content)
    if not has_images:
        return "\n".join(c.text for c in content if c.type == "text")

    blocks: list[dict[str, Any]] = []
    has_text = False
    for c in content:
        if c.type == "text":
            if c.text.strip():
                blocks.append({"type": "text", "text": c.text})
                has_text = True
        elif c.type == "image":
            blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": c.mime_type,
                    "data": c.data,
                },
            })
    if not has_text:
        blocks.insert(0, {"type": "text", "text": "(see attached image)"})
    return blocks


def _convert_messages(ctx: Context) -> list[dict[str, Any]]:
    """Convert rho Context messages to Anthropic Messages format."""
    params: list[dict[str, Any]] = []

    for msg in ctx.messages:
        if msg.role == "user":
            if isinstance(msg.content, str):
                if msg.content.strip():
                    params.append({"role": "user", "content": msg.content})
            else:
                content = _convert_content_blocks(msg.content)
                if isinstance(content, str) and not content.strip():
                    continue
                params.append({"role": "user", "content": content})

        elif msg.role == "assistant":
            blocks: list[dict[str, Any]] = []
            for block in msg.content:
                if block.type == "text":
                    if block.text.strip():
                        blocks.append({"type": "text", "text": block.text})
                elif block.type == "thinking":
                    sig = block.thinking_signature
                    if block.redacted:
                        blocks.append({
                            "type": "redacted_thinking",
                            "data": sig or "",
                        })
                    elif sig and sig.strip():
                        blocks.append({
                            "type": "thinking",
                            "thinking": block.thinking,
                            "signature": sig,
                        })
                    elif block.thinking.strip():
                        # No signature → convert to text
                        blocks.append({"type": "text", "text": block.thinking})
                elif block.type == "toolCall":
                    blocks.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.arguments,
                    })
            if blocks:
                params.append({"role": "assistant", "content": blocks})

        elif msg.role == "toolResult":
            # Collect tool results
            tool_results: list[dict[str, Any]] = []
            content = _convert_content_blocks(msg.content)

            tr: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": msg.tool_call_id,
                "is_error": msg.is_error,
            }
            if isinstance(content, str):
                tr["content"] = content
            else:
                tr["content"] = content
            tool_results.append(tr)

            # Anthropic requires tool results in user messages
            params.append({"role": "user", "content": tool_results})

    return params


def _convert_tools(
    tools: list[Tool],
    compat: dict[str, Any],
    cache_control: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Convert rho tools to Anthropic format."""
    result: list[dict[str, Any]] = []
    for i, tool in enumerate(tools):
        t: dict[str, Any] = {
            "name": tool.name,
            "description": tool.description,
            "input_schema": {
                "type": "object",
                "properties": tool.parameters.get("properties", {}),
                "required": tool.parameters.get("required", []),
            },
        }
        if compat.get("supports_eager_tool_input_streaming", True):
            t["eager_input_streaming"] = True
        # Cache control on last tool
        if cache_control and i == len(tools) - 1 and compat.get("supports_cache_control_on_tools", True):
            t["cache_control"] = cache_control
        result.append(t)
    return result


# ═══════════════════════════════════════════════════════════════════
# Stop reason mapping
# ═══════════════════════════════════════════════════════════════════

def _map_stop_reason(reason: str | None) -> tuple[StopReason, str | None]:
    """Map Anthropic stop_reason to rho StopReason."""
    mapping: dict[str, tuple[StopReason, str | None]] = {
        "end_turn": ("stop", None),
        "max_tokens": ("length", None),
        "tool_use": ("toolUse", None),
        "stop_sequence": ("stop", None),
        "pause_turn": ("stop", None),
    }
    if reason and reason in mapping:
        return mapping[reason]
    if reason == "refusal":
        return ("error", "The model refused to complete the request")
    if reason:
        return ("error", f"Unknown stop_reason: {reason}")
    return ("stop", None)


# ═══════════════════════════════════════════════════════════════════
# Compat
# ═══════════════════════════════════════════════════════════════════

def _resolve_compat(model: Model) -> dict[str, Any]:
    """Resolve Anthropic compat settings."""
    defaults = {
        "supports_eager_tool_input_streaming": True,
        "supports_long_cache_retention": True,
        "send_session_affinity_headers": False,
        "supports_cache_control_on_tools": True,
        "supports_temperature": True,
        "force_adaptive_thinking": False,
        "allow_empty_signature": False,
        "supports_tool_references": False,
    }
    model_compat = model.compat if hasattr(model, "compat") else {}
    return {**defaults, **model_compat}


# ═══════════════════════════════════════════════════════════════════
# Stream function
# ═══════════════════════════════════════════════════════════════════

async def stream(
    model: Model,
    context: Context,
    options: StreamOptions | None = None,
) -> AsyncGenerator[AssistantMessageEvent, None]:
    """Stream an Anthropic Messages response.

    Handles Anthropic-specific SSE events:
    message_start, content_block_start/delta/stop, message_delta, message_stop.
    """
    opts = options or StreamOptions()
    compat = _resolve_compat(model)

    output = AssistantMessage(
        api=model.api,
        provider=model.provider,
        model=model.id,
    )

    # Build request body
    body = _build_request_body(model, context, opts, compat)

    # Build headers
    headers: dict[str, str] = {
        "anthropic-version": "2023-06-01",
        "anthropic-dangerous-direct-browser-access": "true",
    }
    if opts.api_key:
        if opts.api_key.startswith("sk-ant-oat"):
            # OAuth token — use x-api-key header
            headers["x-api-key"] = opts.api_key
        else:
            headers["x-api-key"] = opts.api_key
    if model.headers:
        headers.update(model.headers)
    for key, value in opts.headers.items():
        if value is not None:
            headers[key] = value

    client = HttpClient()

    try:
        yield StartEvent(partial=_snapshot(output))

        # Block tracking by Anthropic index
        content_index_by_an_index: dict[int, int] = {}
        blocks: list[dict[str, Any]] = []

        saw_message_start = False
        saw_message_stop = False

        async for event_data in client.stream_sse(
            url=f"{model.base_url}/v1/messages",
            json_data=body,
            headers=headers,
            timeout_ms=opts.timeout_ms,
            max_retries=opts.max_retries,
            max_retry_delay_ms=opts.max_retry_delay_ms,
        ):
            event_type = event_data.pop("_sse_event", None)

            if event_type == "error":
                detail = event_data.get("error", event_data)
                if isinstance(detail, dict) and detail.get("message"):
                    raise RuntimeError(f"Anthropic stream error: {detail['message']}")
                raise RuntimeError(f"Anthropic stream error: {json.dumps(detail)}")

            if event_type == "message_start":
                saw_message_start = True
                msg = event_data.get("message", {})
                output.response_id = msg.get("id")
                usage_raw = msg.get("usage", {})
                if usage_raw:
                    output.usage.input = usage_raw.get("input_tokens", 0)
                    output.usage.output = usage_raw.get("output_tokens", 0)
                    output.usage.cache_read = usage_raw.get("cache_read_input_tokens", 0)
                    output.usage.cache_write = usage_raw.get("cache_creation_input_tokens", 0)
                    output.usage.total_tokens = (
                        output.usage.input + output.usage.output
                        + output.usage.cache_read + output.usage.cache_write
                    )
                    _update_cost(output, model)

            elif event_type == "content_block_start":
                block = event_data.get("content_block", {})
                an_idx = event_data.get("index", 0)
                btype = block.get("type")

                if btype == "text":
                    entry = {"type": "text", "text": "", "an_index": an_idx}
                    blocks.append(entry)
                    ci = len(blocks) - 1
                    content_index_by_an_index[an_idx] = ci
                    output.content.append(TextContent(text=""))
                    yield TextStart(content_index=ci, partial=_snapshot(output))

                elif btype == "thinking":
                    entry = {
                        "type": "thinking",
                        "thinking": "",
                        "signature": "",
                        "an_index": an_idx,
                    }
                    blocks.append(entry)
                    ci = len(blocks) - 1
                    content_index_by_an_index[an_idx] = ci
                    output.content.append(ThinkingContent(thinking=""))
                    yield ThinkingStart(content_index=ci, partial=_snapshot(output))

                elif btype == "redacted_thinking":
                    entry = {
                        "type": "thinking",
                        "thinking": "[Reasoning redacted]",
                        "signature": block.get("data", ""),
                        "redacted": True,
                        "an_index": an_idx,
                    }
                    blocks.append(entry)
                    ci = len(blocks) - 1
                    content_index_by_an_index[an_idx] = ci
                    output.content.append(ThinkingContent(
                        thinking="[Reasoning redacted]",
                        redacted=True,
                    ))
                    yield ThinkingStart(content_index=ci, partial=_snapshot(output))

                elif btype == "tool_use":
                    entry = {
                        "type": "toolCall",
                        "id": block.get("id", ""),
                        "name": block.get("name", ""),
                        "arguments": block.get("input", {}),
                        "partial_json": "",
                        "an_index": an_idx,
                    }
                    blocks.append(entry)
                    ci = len(blocks) - 1
                    content_index_by_an_index[an_idx] = ci
                    output.content.append(ToolCall(
                        id=entry["id"],
                        name=entry["name"],
                        arguments=entry["arguments"],
                    ))
                    yield ToolCallStart(content_index=ci, partial=_snapshot(output))

            elif event_type == "content_block_delta":
                delta = event_data.get("delta", {})
                an_idx = event_data.get("index", 0)
                ci = content_index_by_an_index.get(an_idx)
                if ci is None or ci >= len(blocks):
                    continue
                block = blocks[ci]
                dtype = delta.get("type")

                if dtype == "text_delta":
                    text = delta.get("text", "")
                    block["text"] += text
                    output.content[ci] = TextContent(text=block["text"])
                    yield TextDelta(content_index=ci, delta=text, partial=_snapshot(output))

                elif dtype == "thinking_delta":
                    thinking = delta.get("thinking", "")
                    block["thinking"] += thinking
                    output.content[ci] = ThinkingContent(thinking=block["thinking"])
                    yield ThinkingDelta(content_index=ci, delta=thinking, partial=_snapshot(output))

                elif dtype == "signature_delta":
                    sig = delta.get("signature", "")
                    block["signature"] = (block.get("signature", "") or "") + sig

                elif dtype == "input_json_delta":
                    partial = delta.get("partial_json", "")
                    block["partial_json"] += partial
                    block["arguments"] = _parse_streaming_json(block["partial_json"])
                    output.content[ci] = ToolCall(
                        id=block["id"],
                        name=block["name"],
                        arguments=block["arguments"],
                    )
                    yield ToolCallDelta(content_index=ci, delta=partial, partial=_snapshot(output))

            elif event_type == "content_block_stop":
                an_idx = event_data.get("index", 0)
                ci = content_index_by_an_index.get(an_idx)
                if ci is None or ci >= len(blocks):
                    continue
                block = blocks[ci]

                if block["type"] == "text":
                    output.content[ci] = TextContent(text=block["text"])
                    yield TextEnd(content_index=ci, content=block["text"], partial=_snapshot(output))
                elif block["type"] == "thinking":
                    output.content[ci] = ThinkingContent(
                        thinking=block["thinking"],
                        thinking_signature=block.get("signature", ""),
                        redacted=block.get("redacted", False),
                    )
                    yield ThinkingEnd(content_index=ci, content=block["thinking"], partial=_snapshot(output))
                elif block["type"] == "toolCall":
                    tc = ToolCall(id=block["id"], name=block["name"], arguments=block["arguments"])
                    output.content[ci] = tc
                    yield ToolCallEnd(content_index=ci, tool_call=tc, partial=_snapshot(output))

            elif event_type == "message_delta":
                delta = event_data.get("delta", {})
                sr = delta.get("stop_reason")
                if sr:
                    stop_reason, err_msg = _map_stop_reason(sr)
                    output.stop_reason = stop_reason
                    if err_msg:
                        output.error_message = err_msg
                usage_raw = event_data.get("usage", {})
                if usage_raw:
                    if usage_raw.get("input_tokens") is not None:
                        output.usage.input = usage_raw["input_tokens"]
                    if usage_raw.get("output_tokens") is not None:
                        output.usage.output = usage_raw["output_tokens"]
                    output.usage.total_tokens = (
                        output.usage.input + output.usage.output
                        + output.usage.cache_read + output.usage.cache_write
                    )
                    _update_cost(output, model)

            elif event_type == "message_stop":
                saw_message_stop = True

        if saw_message_start and not saw_message_stop:
            raise RuntimeError("Anthropic stream ended before message_stop")

        if output.stop_reason == "error":
            raise RuntimeError(output.error_message or "Unknown error")

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
    """Build the Anthropic /v1/messages JSON body."""
    messages = _convert_messages(ctx)
    body: dict[str, Any] = {
        "model": model.id,
        "messages": messages,
        "max_tokens": opts.max_tokens or model.max_tokens,
        "stream": True,
    }

    # System prompt
    if ctx.system_prompt:
        body["system"] = ctx.system_prompt

    # Temperature (incompatible with thinking on some models)
    if opts.temperature is not None and compat.get("supports_temperature", True):
        body["temperature"] = opts.temperature

    # Tools
    if ctx.tools:
        body["tools"] = _convert_tools(ctx.tools, compat)

    # Metadata
    if opts.metadata:
        user_id = opts.metadata.get("user_id")
        if isinstance(user_id, str):
            body["metadata"] = {"user_id": user_id}

    return body


# ═══════════════════════════════════════════════════════════════════
# Simple stream
# ═══════════════════════════════════════════════════════════════════

async def stream_simple(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AsyncGenerator[AssistantMessageEvent, None]:
    """Simplified stream entry point."""
    async for event in stream(model, context, options):
        yield event
