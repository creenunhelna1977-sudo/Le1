# 添加新提供商

如何为 rho AI 层添加新的 LLM 提供商。

## 快速判断

1. 提供商说已有的 API 协议？→ **复用 API 模块，只加 provider 文件**
2. 提供商说新协议？→ **先实现新的 API 模块**
3. 需要新的 compat 标志？→ **添加到 compat 检测 + API 模块**

## 情况一：OpenAI 兼容的提供商（最常见）

大多数提供商（DeepSeek、together、groq、cerebras、xAI、openrouter 等）使用 OpenAI `/v1/chat/completions` 协议。步骤：

### 1. 创建 `rho/ai/providers/<名称>.py`

```python
"""<提供商名称> provider。"""

from rho.ai.auth import env_api_key_auth, ProviderAuth
from rho.ai.provider import create_provider
from rho.ai.types import Model, ModelCostRates
from rho.ai.api.openai_completions import stream, stream_simple

# ── 模型 ──────────────────────────────────────────────

MY_MODEL = Model(
    id="my-model-id",
    name="我的模型显示名称",
    api="openai-completions",
    provider="my-provider",          # 与下方 provider id 匹配
    base_url="https://api.example.com/v1",
    reasoning=True,                   # 支持 thinking/reasoning？
    input=["text"],                   # ["text"] 或 ["text", "image"]
    cost=ModelCostRates(
        input=0.50, output=2.00,      # $/百万 token
        cache_read=0.10, cache_write=0.50,
    ),
    context_window=128_000,
    max_tokens=8_192,
    # Compat：仅在自动检测不准确时设置
    compat={
        "thinking_format": "deepseek",  # 如果是非标准格式
        "max_tokens_field": "max_tokens",
    },
)

MODELS = [MY_MODEL]

# ── Provider ──────────────────────────────────────────

def my_provider():
    return create_provider(
        id="my-provider",
        name="我的提供商",
        base_url="https://api.example.com/v1",
        auth=ProviderAuth(
            api_key=env_api_key_auth(
                "我的提供商 API key",
                ["MY_PROVIDER_API_KEY"],
            ),
        ),
        models=MODELS,
        api=stream,
        stream_simple=stream_simple,
    )
```

### 2. 注册

```python
from rho.ai import Models
from rho.ai.providers.my_provider import my_provider

models = Models()
models.register(my_provider())
```

### 3. 自动检测不够用？

`_detect_compat()` 的自动检测检查 `model.provider` 和 `model.base_url`。如果你的提供商需要的标志未被检测到：

- **在 `model.compat` 中设置**（快速修复，按模型生效）
- **在 `_detect_compat()` 中增加检测**（正确修复，该提供商所有模型受益）

增加检测的例子：
```python
# 在 openai_completions.py → _detect_compat()
is_my_provider = provider == "my-provider" or "api.example.com" in url

return {
    ...
    "max_tokens_field": "max_tokens" if is_my_provider else "max_completion_tokens",
    ...
}
```

## 情况二：新的 API 协议

如果提供商使用完全不同的协议（如 Google Gemini、Mistral）：

### 1. 创建 `rho/ai/api/<协议>.py`

模块必须导出：

```python
async def stream(
    model: Model,
    context: Context,
    options: StreamOptions | None = None,
) -> AsyncGenerator[AssistantMessageEvent, None]:
    """从此 API 流式获取响应。"""
    # 1. 从 context 构建请求体
    # 2. 通过 HttpClient 发出 HTTP 请求
    # 3. 将响应解析为事件
    # 4. yield 事件
    ...

async def stream_simple(
    model: Model,
    context: Context,
    options: StreamOptions | None = None,
) -> AsyncGenerator[AssistantMessageEvent, None]:
    """简化流（推理级别 facade）。"""
    async for event in stream(model, context, options):
        yield event
```

关键规则：
- 事件必须按序产出：`start` → 内容事件 → `done`/`error`
- 内容块必须用从 0 开始的递增 `content_index`
- 每个事件的 `partial` 必须反映 `AssistantMessage.content` 的当前状态
- Usage 必须累加 — 每个 chunk 可能增加 token
- 错误必须捕获并产出 `ErrorEvent`，不能 raise

### 2. 兼容性

如果协议有提供商特定变体，加入 compat dict（类似 `OpenAICompletionsCompat` 或 `AnthropicMessagesCompat`）。遵循同样的自动检测 + 模型覆盖模式。

### 3. 创建 provider 文件（见情况一）

## Auth 模式

### 标准 env var API key

```python
auth=ProviderAuth(
    api_key=env_api_key_auth("显示名称", ["ENV_VAR_NAME"]),
)
```

`env_api_key_auth` 按顺序尝试：
1. 存储的凭证（未来：credential store）
2. 列表中的每个 env var，返回第一个设置了的

### 多个 env var（如 OAuth token 优先，然后 API key）

```python
auth=ProviderAuth(
    api_key=env_api_key_auth("Anthropic API key", [
        "ANTHROPIC_OAUTH_TOKEN",   # 优先尝试
        "ANTHROPIC_API_KEY",       # 回退
    ]),
)
```

### Keyless / 环境认证提供商

对于不使用 API key 的提供商（如本地 Ollama）：

```python
class KeylessAuth(ApiKeyAuth):
    def __init__(self):
        super().__init__(name="无", env_vars=[])

    async def resolve(self, ctx, credential=None):
        return AuthResult(auth=ModelAuth(), source="keyless")

auth=ProviderAuth(api_key=KeylessAuth())
```

## 模型定义

### 必填字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | `str` | API 模型标识符（发给提供商） |
| `api` | `ApiLit \| str` | 使用哪个 API 协议：`"openai-completions"` 或 `"anthropic-messages"` |
| `provider` | `str` | 必须匹配 provider 的 `id` |
| `base_url` | `str` | 提供商的 API 基础 URL |

### 推荐字段

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `name` | `str` | `""` | 人类可读的显示名称 |
| `reasoning` | `bool` | `False` | 模型是否支持 thinking/reasoning |
| `input` | `list[str]` | `["text"]` | 支持的输入模态 |
| `cost` | `ModelCostRates` | 全零 | $/百万 token 定价 |
| `context_window` | `int` | `128000` | 最大上下文窗口（token） |
| `max_tokens` | `int` | `4096` | 最大输出 token |
| `compat` | `dict` | `{}` | API 特定的 compat 覆盖 |
| `thinking_level_map` | `dict` | `{}` | rho thinking 级别到提供商特定值的映射 |

## 测试你的提供商

### 单元测试（mock HTTP）

```python
async def test_my_provider():
    from rho.ai.api.openai_completions import stream
    from rho.ai.http_client import HttpClient
    from unittest.mock import patch

    model = Model(id="my-model", api="openai-completions",
                  provider="my-provider", base_url="https://api.example.com/v1")
    ctx = Context(messages=[UserMessage(content="你好")])

    chunks = [
        {"choices": [{"delta": {"content": "你好"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}],
         "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}},
    ]

    async def mock_sse(self, url, json_data, headers, method="POST"):
        for chunk in chunks:
            yield chunk

    with patch.object(HttpClient, "stream_sse", mock_sse):
        events = [e async for e in stream(model, ctx,
                  StreamOptions(api_key="test"))]
        assert events[-1].type == "done"
```

### 集成测试（真实 API）

```bash
export MY_PROVIDER_API_KEY="sk-..."
PYTHONPATH=. uv run python -c "
import asyncio
from rho.ai import Models
from rho.ai.providers.my_provider import my_provider
from rho.ai.types import Context, UserMessage, TextDelta, DoneEvent

async def main():
    models = Models()
    models.register(my_provider())
    model = models.get_model('my-provider', 'my-model-id')
    ctx = Context(messages=[UserMessage(content='你好')])
    async for e in models.stream(model, ctx):
        if isinstance(e, TextDelta): print(e.delta, end='')
        elif isinstance(e, DoneEvent): print(f'\n完成: {e.message.usage.total_tokens} tokens')

asyncio.run(main())
"
```
