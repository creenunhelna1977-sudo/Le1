# rho Agent 层 — 设计文档

## 概述

Agent 层做的事很简单：**发消息给 LLM，如果有工具调用就执行，把结果发回去，重复直到模型说停。**

```
用户: "北京天气？"
  → LLM: toolUse get_weather(city="北京")
    → Agent 执行 get_weather → "北京: 晴 22°C"
  → LLM: "北京今天晴天，22度。"  (stop)
→ 结束
```

---

## Python 原生能力

Agent loop 在 Python 里就是两层循环嵌套的 async generator。不需要 EventStream、不需要 AbortController、不需要 PendingMessageQueue。

| 需求 | Python 方案 | 来源 |
|------|-----------|------|
| 流式产生事件 | `async generator` yield | 语言原生 |
| 等待+取消 | `asyncio.Task`（调用方管理） | 标准库 |
| 消息队列 | `collections.deque` | 标准库 |
| 事件分发 | `match/case` | 语言原生 |
| 类型安全 | Pydantic discriminated union | 第三方 |
| 测试 mock | `unittest.mock` patch async generator | 标准库 |

---

## 目录

```
rho/agent/
├── __init__.py
├── types.py       # AgentEvent, AgentTool, AgentToolResult
├── agent.py       # Agent 类 + run() async generator
└── test_agent.py  # 测试
```

v1 三个文件。loop 逻辑在 `agent.py` 里，不需要单独拆 `loop.py` — 总共不会超过 150 行。

---

## 类型

### AgentEvent

```python
AgentEvent = (
    AgentStart
    | TurnStart
    | MessageStart | MessageUpdate | MessageEnd
    | ToolExecutionStart | ToolExecutionEnd
    | TurnEnd
    | AgentEnd
)
```

9 种事件，按 `type` 字段区分。对标 PI 的 AgentEvent 但去掉了 `tool_execution_update`（v1 工具不流式推送进度）。

### AgentTool

```python
@dataclass
class AgentTool:
    name: str
    description: str
    parameters: dict[str, Any]           # JSON Schema
    execute: Callable[..., Awaitable[AgentToolResult]]
    label: str = ""

@dataclass  
class AgentToolResult:
    content: list[TextContent | ImageContent]
    is_error: bool = False
```

对标 AI 层的 `Tool` 但多了 `execute` 函数。`parameters` 直接复用 AI 层的 JSON Schema dict。

---

## 核心设计

### Agent.run() — async generator

```python
class Agent:
    def __init__(self, models, model, tools=(), system_prompt=""):
        self.models = models
        self.model = model
        self.tools = {t.name: t for t in tools}
        self.system_prompt = system_prompt
        self.messages: list[Message] = []

    async def run(self, prompt: str | Message) -> AsyncGenerator[AgentEvent, None]:
        """驱动 agent loop，yield 事件。"""
```

`run()` **就是**循环。它不是启动一个后台任务然后发事件 — 它就是 async generator，调用方 `async for` 直接消费：

```python
agent = Agent(models, model, tools)
async for event in agent.run("北京天气怎么样？"):
    match event:
        case TextDelta():     print(event.delta, end="")
        case ToolExecutionStart(): print(f"执行 {event.tool_name}...")
        case AgentEnd():      print("完成")
```

### 循环逻辑

```python
async def run(self, prompt):
    self.messages.append(_to_user_message(prompt))
    yield AgentStart()

    while True:
        yield TurnStart()

        ctx = Context(
            system_prompt=self.system_prompt,
            messages=list(self.messages),
            tools=[_to_ai_tool(t) for t in self.tools.values()],
        )

        partial = None
        async for ai_event in self.models.stream(self.model, ctx):
            match ai_event:
                case StartEvent(partial=p):
                    partial = p
                    yield MessageStart(message=p)

                case TextDelta() | ThinkingDelta() | ToolCallDelta():
                    partial = ai_event.partial
                    yield MessageUpdate(message=partial, ai_event=ai_event)

                case DoneEvent(message=msg):
                    self.messages.append(msg)
                    yield MessageEnd(message=msg)

                    if msg.stop_reason != "toolUse":
                        yield TurnEnd(message=msg, tool_results=[])
                        yield AgentEnd()
                        return

                    # 执行工具
                    tool_calls = [b for b in msg.content if b.type == "toolCall"]
                    tool_results = await self._execute_tools(tool_calls)
                    yield TurnEnd(message=msg, tool_results=tool_results)
                    break  # 跳出 ai_event 循环，回到 while True 继续下一轮

                case ErrorEvent(error=msg):
                    self.messages.append(msg)
                    yield MessageEnd(message=msg)
                    yield AgentEnd()
                    return
```

### 工具执行（v1: sequential）

```python
async def _execute_tools(self, tool_calls):
    results = []
    for tc in tool_calls:
        tool = self.tools.get(tc.name)
        yield ToolExecutionStart(tool_call_id=tc.id, tool_name=tc.name, args=tc.arguments)

        if tool is None:
            result = AgentToolResult(
                content=[TextContent(text=f"Tool '{tc.name}' not found")],
                is_error=True,
            )
        else:
            try:
                result = await tool.execute(**tc.arguments)
            except Exception as e:
                result = AgentToolResult(content=[TextContent(text=str(e))], is_error=True)

        yield ToolExecutionEnd(tool_call_id=tc.id, tool_name=tc.name,
                               result=result, is_error=result.is_error)

        tr = ToolResultMessage(tool_call_id=tc.id, tool_name=tc.name,
                               content=result.content, is_error=result.is_error)
        self.messages.append(tr)
        yield MessageStart(message=tr)
        yield MessageEnd(message=tr)
        results.append(tr)
    return results
```

### 取消

Agent 不需要内置 cancel 机制。调用方想取消？对 async generator 发 cancel：

```python
task = asyncio.create_task(consume_events(agent.run(prompt)))
# ... 用户按了 Ctrl+C ...
task.cancel()
```

`models.stream()` 内部的 httpx 请求会响应 cancel 抛 `CancelledError`，generator 自然停止。

---

## v1 边界

**做：**

- Agent 类 + `run()` async generator
- AgentEvent 9 种事件类型
- AgentTool + sequential 执行
- 基础错误处理（工具未找到、执行异常）
- 单元测试 + 集成测试

**不做：**

- 并行工具执行 — v1 sequential，后续加 `asyncio.gather`
- steering / followUp — v1 无中断注入需求
- beforeToolCall / afterToolCall hooks — v1 工具直接执行
- subscribe() 观察者 — `async for` 就是消费方式
- asyncio.Task 生命周期管理 — 调用方的责任
- AgentHarness / Session / Compaction — 后续层

---

## 与 AI 层的关系

```
Agent.run()                          AI 层
  │                                    │
  ├─ 构建 Context ──────────────────→ models.stream(model, ctx)
  │                                    │
  ├─ async for ai_event ←─────────── AsyncGenerator[AssistantMessageEvent]
  │   ├─ TextDelta → 转发给调用方       │
  │   └─ DoneEvent → 检查 stopReason   │
  │                                    │
  ├─ toolUse → 执行 AgentTool.execute()
  │   └─ 构建 ToolResultMessage → 加入 self.messages
  │   └─ 回到 while True ──────────→ 新一轮 models.stream()
  │
  └─ stop → AgentEnd
```

Agent 是 AI 层的消费者，驱动 loop。AI 层不知道 Agent 存在。

---

## 测试

```python
# 单元测试：用 async generator mock AI 层
async def test_basic():
    async def mock_stream(model, ctx, opts=None):
        yield StartEvent(partial=AssistantMessage(...))
        yield TextDelta(delta="你好", content_index=0, partial=...)
        yield DoneEvent(reason="stop", message=AssistantMessage(
            content=[TextContent(text="你好")], stop_reason="stop"))

    agent = Agent(models=MockModels(stream=mock_stream), model=...)
    events = [e async for e in agent.run("hi")]
    assert events[-1].type == "agent_end"

# 集成测试：真实 DeepSeek + 真实工具
async def test_with_tool():
    async def get_weather(city: str):
        return AgentToolResult(content=[TextContent(text=f"{city}: 晴")])

    agent = Agent(models=Models([deepseek_provider()]),
                  model=deepseek_chat,
                  tools=[AgentTool(name="get_weather", execute=get_weather, ...)])
    async for event in agent.run("北京天气？"):
        ...
```

---

## 拓展方向

### 短期

**并行工具执行** — 从 `for` 循环改为 `asyncio.gather`：
```python
results = await asyncio.gather(*[
    _execute_one(tc, tools) for tc in tool_calls
])
```

**hooks** — `before_tool_call(tc) -> bool` 和 `after_tool_call(tc, result) -> AgentToolResult`，在 execute 前后插入

### 中期

**AgentHarness** — 组合 Agent + Session + Compaction + Skills，对标 PI harness 层

### 长期

**多 agent** — 一个 harness 管理多个 agent 实例协作
