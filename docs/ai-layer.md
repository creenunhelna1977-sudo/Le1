# rho AI 层 — 架构与设计

## 概述

AI 层为 LLM 提供商提供**统一、类型化、流式优先**的接口。每个提供商 — OpenAI、Anthropic、DeepSeek — 都通过同一个抽象层通信，底层处理各异 API 形状、认证流程和传输协议。

## 设计原则

1. **不用提供商 SDK。** 原始 HTTP（`httpx`）给予对 headers、auth、retry、streaming 的完全控制。SDK 隐藏太多且变更频繁。

2. **流式优先。** 每次调用都是流。`complete()` 只是将流收集为单个结果。这符合 LLM 实际工作方式 — token 逐个到达。

3. **类型化事件。** 流产出 discriminated union 事件（`TextDelta`、`ThinkingDelta`、`ToolCallEnd`、`Done`、`Error`），而非原始字符串。消费者用 `match/case` 做模式匹配，获取类型化负载。

4. **Python 习惯。** async generator 替代 push-based 事件发射器。Pydantic discriminated union 替代 TypeBox。`match/case` 做事件分发。

5. **Compat 而非 abstraction。** 提供商不各自拥有 API 模块 — 它们共享 API 实现（如 DeepSeek 用 `openai_completions`）。差异由 **compat 系统**处理：API 模块读取行为标志字典来调整输出。

## 参考：PI 架构

本设计参考了 `reference_pi/pi`（Earendil Works 的 TypeScript agent harness）。关键适配：

| PI (TypeScript) | rho (Python) | 原因 |
|---|---|---|
| `EventStream<T,R>` push 类 | `AsyncGenerator[Event, None]` | Python 有原生 async generator |
| TypeBox schemas | Pydantic v2 models | Python 生态标准 |
| OpenAI/Anthropic SDKs | `httpx` 原始 HTTP | 完全控制，不受 SDK 变更影响 |
| 懒加载 (`lazyApi()`) | 直接 import | Python 包不需要 |
| `Provider<TApi>` 泛型 | `Provider` + 运行时 dispatch | Python 泛型在此规模增加复杂度无收益 |

## 目录结构

```
rho/ai/
├── types.py          # 全部 pydantic 模型 — 单一数据源
├── auth.py           # Auth 类型 + ApiKeyAuth + env var 解析
├── http_client.py    # httpx 封装 + SSE 流式
├── provider.py       # Provider 类 + create_provider() 工厂
├── models.py         # Models 注册中心 — 主入口
├── api/
│   ├── openai_completions.py    # /v1/chat/completions → 事件
│   └── anthropic_messages.py    # /v1/messages → 事件
└── providers/
    ├── openai.py     # GPT-4o, GPT-4o-mini, o3-mini, o4-mini
    └── deepseek.py   # DeepSeek-V3.1, DeepSeek-R1
```

## 数据流

```
用户代码
  │
  ▼
Models.stream(model, context, options)
  │
  ├─► resolve_auth(model, options)
  │     ├─► 按 model.provider 查找 provider
  │     ├─► provider.auth.api_key.resolve(env vars, stored cred)
  │     └─► 合并 headers: auth → model → options
  │
  └─► provider.stream(model, context, resolved_options)
        │
        └─► api_module.stream(model, context, options)
              │
              ├─► detect_compat(model) — 基于 URL/provider 的默认值
              ├─► resolve_compat(model) — 与 model.compat 覆盖合并
              ├─► convert_messages(ctx) → 提供商特定格式
              ├─► build_request_body() → JSON body
              ├─► HttpClient.stream_sse() → 原始 SSE chunks
              │     └─► httpx.AsyncClient.stream(method, url, json, headers)
              │           └─► response.aiter_lines() → SSE 解析
              │
              └─► yield AssistantMessageEvent*
                    ├─► StartEvent
                    ├─► TextStart / TextDelta / TextEnd
                    ├─► ThinkingStart / ThinkingDelta / ThinkingEnd
                    ├─► ToolCallStart / ToolCallDelta / ToolCallEnd
                    └─► DoneEvent | ErrorEvent
```

## 类型层级

### 消息（按 `role` 区分）

```
Message = UserMessage | AssistantMessage | ToolResultMessage
```

- `UserMessage.role = "user"` — content 为 `str` 或 `list[TextContent | ImageContent]`
- `AssistantMessage.role = "assistant"` — content 为 `list[TextContent | ThinkingContent | ImageContent | ToolCall]`
- `ToolResultMessage.role = "toolResult"` — 携带 `tool_call_id`、`tool_name`、结果内容

### 内容块（按 `type` 区分）

```
ContentBlock = TextContent | ThinkingContent | ImageContent | ToolCall
```

- `TextContent.type = "text"` — 纯文本，可选 `text_signature`
- `ThinkingContent.type = "thinking"` — 推理内容，可选 `thinking_signature`、`redacted`
- `ImageContent.type = "image"` — base64 data + mime_type
- `ToolCall.type = "toolCall"` — 函数调用，含 id、name、arguments

### 流事件（按 `type` 区分）

```
AssistantMessageEvent =
    StartEvent           — 流开始，携带部分 AssistantMessage
  | TextStart            — 新文本块在 content_index 开始
  | TextDelta            — 增量文本
  | TextEnd              — 文本块完成，含完整内容
  | ThinkingStart        — 新推理块开始
  | ThinkingDelta        — 增量推理
  | ThinkingEnd          — 推理块完成
  | ToolCallStart        — 新工具调用块开始
  | ToolCallDelta        — 增量 JSON 参数
  | ToolCallEnd          — 工具调用完成，含解析后的参数
  | DoneEvent            — 流成功（reason: stop | length | toolUse）
  | ErrorEvent           — 流失败（reason: error | aborted）
```

## Compat 系统

Compat 系统使得一个 API 实现（`openai_completions.py`）可以服务多个提供商（OpenAI、DeepSeek、together、z.ai 等）。

### 工作原理

1. **`_detect_compat(model)`** — 从 `model.provider` 和 `model.base_url` 自动检测 compat 标志。例如 `base_url` 含 `"deepseek.com"`，则设置 `thinking_format: "deepseek"`、`max_tokens_field: "max_tokens"`。

2. **`_resolve_compat(model)`** — 将自动检测的默认值与显式的 `model.compat` 覆盖合并。模型级覆盖始终优先。

3. **API 模块在关键决策点读取 compat 标志**：
   - `max_tokens_field` — 使用哪个 JSON 字段名
   - `thinking_format` — 如何编码推理参数
   - `requires_reasoning_content_on_assistant_messages` — DeepSeek 特有
   - `supports_developer_role` — 用 `developer` 还是 `system` 角色
   - `requires_tool_result_name` — 部分提供商工具结果需要 `name` 字段
   - `requires_assistant_after_tool_result` — 部分需要填充消息

### 新增 OpenAI 兼容提供商

如果新提供商说 OpenAI completions 协议但有特殊需求：

1. 在 `model.compat` 中设置正确的标志 — 如果 quirk 已被覆盖则无需改代码
2. 如果新 quirk 需要新标志：在 `_detect_compat()` 中添加，在 `_build_request_body()` 或 `_convert_messages()` 中添加标志键读取

## Auth 设计

### 当前（v1）：仅 ApiKeyAuth

```
ProviderAuth
  └─► api_key: ApiKeyAuth
        ├─► name: "OpenAI API key"
        ├─► env_vars: ["OPENAI_API_KEY"]
        └─► resolve(ctx, credential) → AuthResult | None
              ├─► credential.key  （存储的凭证，最高优先级）
              └─► ctx.env(var)    （环境变量，回退）
```

### 未来：OAuth

```
ProviderAuth
  ├─► api_key: ApiKeyAuth | None
  └─► oauth: OAuthAuth | None   ← 预留槽位
        ├─► login(interaction) → OAuthCredential
        ├─► refresh(credential) → OAuthCredential
        └─► to_auth(credential) → ModelAuth
```

添加 OAuth 时，`auth.py` 将拆分为 `auth/` 目录：
```
auth/
├── __init__.py
├── types.py         # Credential、AuthResult 等
├── context.py       # AuthContext
├── credential_store.py
├── helpers.py       # env_api_key_auth, lazy_oauth
├── resolve.py       # resolve_provider_auth
└── oauth/
    ├── anthropic.py
    └── ...
```

## 流式协议

### 为什么用 async generator 而不是 EventStream

PI 使用 `EventStream<T,R>` — push-based 异步可迭代类，含 `push()`、`end()`、`[Symbol.asyncIterator]()`。它存在是因为 TypeScript 没有原生 async generator。

Python 有 `async def ... yield`。一个 async generator：

```python
async def stream(model, ctx, opts) -> AsyncGenerator[Event, None]:
    yield StartEvent(partial=output)
    # ... 解析 SSE ...
    yield TextDelta(delta="hello", ...)
    yield DoneEvent(reason="stop", message=output)
```

比 push 类严格更简单：
- 无需维护类（EventStream 约 90 行）
- 无需队列管理（push/pull 同步）
- 错误自然传播（raise，不 push ErrorEvent）
- 更易测试（直接迭代收集）
- 支持 `async for` / `anext()` / async list comprehension

### 错误处理

API 实现捕获所有异常并产出 `ErrorEvent`：

```python
try:
    yield DoneEvent(reason=output.stop_reason, message=output)
except Exception as exc:
    output.stop_reason = "error"
    output.error_message = str(normalize_provider_error(exc))
    yield ErrorEvent(reason="error", error=output)
```

消费者可内联处理错误：
```python
async for event in models.stream(model, ctx):
    case ErrorEvent(error=msg):
        print(f"错误: {msg.error_message}")
        break
    case DoneEvent(message=msg):
        return msg
```

## 测试策略

### 单元测试（mock HTTP）

每个 API 模块用 `unittest.mock.patch` 替换 `HttpClient.stream_sse`，注入精确的 SSE chunk 序列并验证事件输出：

```python
with patch.object(HttpClient, "stream_sse", mock_stream_sse):
    events = [e async for e in stream(model, ctx, opts)]
    assert events[-1].type == "done"
```

### 集成测试（真实 API）

用真实凭证运行冒烟测试，验证完整管道：auth 解析 → HTTP → SSE 解析 → 事件分发。

### 类型往返测试

验证 `model_dump()` → `model_validate()` 对所有类型（尤其是 discriminated union）保持数据不丢失。

## 决策记录

| 日期 | 决策 | 原因 |
|------|------|------|
| 2026-07-17 | Pydantic v2 而非 dataclass | Discriminated union、验证、序列化 |
| 2026-07-17 | Async generator 而非 EventStream | Python 原生，更简单，无需队列管理 |
| 2026-07-17 | httpx 而非提供商 SDK | 完全 HTTP 控制，单一依赖 |
| 2026-07-17 | auth.py 单文件（v1） | 仅一种 auth 方法；OAuth 到来时拆分 |
| 2026-07-17 | providers/ 目录（非单文件） | 对标 PI，可扩展至 10+ 提供商 |
| 2026-07-17 | Compat dict 而非类型化 compat 模型 | 一个 API 实现服务多个提供商；dict 灵活 |
| 2026-07-17 | v1 不做懒加载 | Python import 系统无 TS bundler 的收益 |
