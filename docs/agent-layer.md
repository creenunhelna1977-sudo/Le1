# rho Agent Layer 设计计划

> 状态：设计阶段，尚未进入实现。
>
> 参考范围：`reference_pi/pi/packages/agent/src/types.ts`、`agent-loop.ts`、`agent.ts` 及对应核心测试。
>
> 目标：理解 PI 的约束和行为，但为 Python 重新设计，不复制 TypeScript 实现。

## 1. 概述

Agent 层只负责一个闭环：

1. 把用户消息和历史记录交给 AI 层。
2. 转发模型流事件。
3. 收到工具调用后执行工具。
4. 把工具结果加入上下文，再次调用模型。
5. 模型停止、出错或调用方取消时结束。

```text
UserMessage
    |
    v
Models.stream_simple()
    |
    +-- text/thinking deltas --> AgentEvent --> caller
    |
    +-- AssistantMessage(stop) ----------------------> end
    |
    +-- AssistantMessage(toolUse)
            |
            v
       execute tools sequentially
            |
            v
       ToolResultMessage[]
            |
            +-------------------------> next model turn
```

Agent 层只能依赖 `rho.ai`，AI 层不能反向依赖 Agent 层。Session、Compaction、Skills、TUI 都位于 Agent 之上，不进入本模块。

## 2. PI 调研结论

### 2.1 PI 为什么拆成两层

PI 有两个核心层次：

- `agent-loop.ts`：低层循环、事件 sink、工具执行、steering/follow-up、hooks。
- `agent.ts`：状态机、订阅器、消息队列、AbortController、`waitForIdle()`、`continue()`。

这种拆分主要服务于 PI 的运行方式：

- TypeScript 没有原生 async generator result 协议，PI 使用 push-based `EventStream`。
- `Agent.prompt()` 在后台生产事件，多个 listener 通过 `subscribe()` 观察。
- 交互式 UI 需要运行中 steering、完成后 follow-up 和显式 abort。
- `AgentMessage` 支持应用自定义消息，调用模型前必须转换。
- Harness 需要 session、compaction、资源刷新和请求 hooks。

rho v1 没有这些上层需求。直接照搬会得到两个几乎重复的 loop API、一套后台任务生命周期和多个暂时没有调用方的 hook。

### 2.2 Python 对应方案

| PI 原语 | rho 方案 | 决策原因 |
|---|---|---|
| `EventStream<AgentEvent, Result>` | `AsyncGenerator[AgentEvent, None]` | Python 原生支持流、背压和取消 |
| `AgentEventSink` | `yield` | 消费者处理完当前事件后，生产者才继续 |
| `AbortController` | `asyncio.Task.cancel()` | Python 原生取消语义 |
| `PendingMessageQueue` | v1 不需要 | 没有 steering/follow-up |
| `AgentMessage` declaration merging | 直接复用 `Message` | v1 没有自定义消息类型 |
| `streamFn` 注入 | 直接传 `Models` | 测试 patch `Models.stream_simple` |
| TypeBox tool validation | v1 使用 Python 调用签名作为最小边界 | 当前没有统一 JSON Schema validator |
| getter/setter 防数组引用泄漏 | 属性返回 `list()` 拷贝 | Python 明确、简单 |

## 3. v1 设计目标

v1 必须完成：

- 文本和 thinking 流事件转发。
- 多次 `run()` 的对话历史累积。
- 单个或多个工具调用的顺序执行。
- 工具结果加入上下文并驱动下一轮模型调用。
- 未知工具、调用参数错误、工具异常的错误结果。
- `stop_reason="length"` 时禁止执行可能被截断的工具参数。
- AI 层 `ErrorEvent` 和调用前异常的完整生命周期。
- 同一个 Agent 不允许两个 `run()` 并发修改 transcript。
- 调用方通过 task cancellation 取消执行。

v1 不追求成为完整 orchestration framework。

## 4. 目录计划

```text
rho/agent/
├── __init__.py       # 公开导出
├── types.py          # AgentTool、AgentToolResult、AgentEvent
└── agent.py          # Agent 状态和 loop

tests/
└── test_agent_unit.py
```

暂不创建 `agent_loop.py`。v1 只有一个公开运行入口，拆出第二个 loop 模块不会形成可复用边界。只有 `agent.py` 接近 500 行，或工具执行发展出并行、hooks、progress 后，才拆 `tool_execution.py`。

## 5. 公开 API 计划

### 5.1 AgentToolResult

```python
class AgentToolResult(BaseModel):
    content: list[TextContent | ImageContent] = Field(default_factory=list)
```

工具成功时返回该类型。工具失败时应抛异常，由 Agent 转成 `is_error=True` 的 `ToolResultMessage`，避免工具作者同时维护“异常”和“错误结果”两种协议。

v1 不加入 `details`、`terminate`、`added_tool_names`。AI 层的 `ToolResultMessage.added_tool_names` 保持默认空列表。

### 5.2 AgentTool

```python
@dataclass(frozen=True, slots=True)
class AgentTool:
    name: str
    description: str
    parameters: dict[str, Any]
    execute: Callable[..., Awaitable[AgentToolResult]]
```

执行方式：

```python
result = await tool.execute(**tool_call.arguments)
```

这样工具定义符合 Python 常见写法：

```python
async def get_weather(city: str) -> AgentToolResult:
    return AgentToolResult(content=[TextContent(text=f"{city}: sunny")])
```

`parameters` 是发给模型的 JSON Schema。v1 不声称它已经完成本地 JSON Schema 校验：

- 缺少参数或出现未知参数时，Python 调用绑定产生 `TypeError`，Agent 转成工具错误。
- 参数类型注解不会自动做运行时校验。
- 第一个需要 enum、嵌套对象或数值边界强校验的工具出现时，再引入统一 validator 或可选 Pydantic args model。

### 5.3 Agent

```python
class Agent:
    def __init__(
        self,
        models: Models,
        model: Model,
        *,
        system_prompt: str = "",
        tools: Sequence[AgentTool] = (),
        messages: Sequence[Message] = (),
        options: SimpleStreamOptions | None = None,
    ) -> None: ...

    @property
    def messages(self) -> list[Message]: ...

    @property
    def is_running(self) -> bool: ...

    def reset(self, messages: Sequence[Message] = ()) -> None: ...

    async def run(
        self,
        prompt: str | UserMessage,
    ) -> AsyncGenerator[AgentEvent, None]: ...
```

设计约束：

- `models` 是真实 `Models` 实例，不增加 `StreamFn`/`AgentBackend` 间接层。
- 每轮调用 `models.stream_simple()`，让 Agent 的 reasoning 配置沿用 AI 层统一入口。
- 构造参数中的 `tools`、`messages` 立即做顶层拷贝。
- `options` 使用 `model_copy(deep=True)` 保存，调用方后续修改原对象不影响 Agent。
- 工具名必须非空且唯一，重复名称在构造时抛 `ValueError`，不能静默覆盖。
- `messages` 属性返回列表拷贝；消息对象本身不做深拷贝。
- `reset()` 在运行中调用应抛 `RuntimeError`。
- 同一个实例并发调用 `run()` 应抛 `RuntimeError`，防止 transcript 交错写入。
- `run()` 的 async generator 被消费时才真正开始执行。

## 6. AgentEvent 协议

事件使用 Pydantic discriminated union，与 AI 层一致：

```python
AgentEvent = Annotated[
    AgentStart
    | AgentEnd
    | TurnStart
    | TurnEnd
    | MessageStart
    | MessageUpdate
    | MessageEnd
    | ToolExecutionStart
    | ToolExecutionEnd,
    Field(discriminator="type"),
]
```

### 6.1 事件字段

| 事件 | 主要字段 | 语义 |
|---|---|---|
| `agent_start` | 无 | 一次 `run()` 开始 |
| `agent_end` | `messages` | 正常或错误结束；携带本次结束时 transcript 快照 |
| `turn_start` | `turn` | 一次模型请求开始，turn 从 1 递增 |
| `turn_end` | `message`, `tool_results` | assistant 响应和本轮工具处理完成 |
| `message_start` | `message` | user、assistant 或 toolResult 消息开始 |
| `message_update` | `message`, `ai_event` | assistant 流式更新 |
| `message_end` | `message` | 消息最终提交到 transcript |
| `tool_execution_start` | id、name、args | 工具开始或准备产生错误结果 |
| `tool_execution_end` | id、name、result、`is_error` | 工具执行完成 |

v1 不做 `tool_execution_update`。工具没有 progress callback，就不应暴露一个永远不会发生的事件。

### 6.2 基本文本事件顺序

```text
agent_start
turn_start(turn=1)
message_start(user)
message_end(user)
message_start(assistant partial)
message_update(assistant partial, TextDelta/ThinkingDelta/...)
message_end(assistant final)
turn_end(assistant final, [])
agent_end(transcript)
```

### 6.3 工具事件顺序

```text
agent_start
turn_start(turn=1)
message_start/end(user)
message_start/update/end(assistant with tool calls)
tool_execution_start(call-1)
tool_execution_end(call-1)
message_start/end(toolResult call-1)
tool_execution_start(call-2)
tool_execution_end(call-2)
message_start/end(toolResult call-2)
turn_end(assistant, [result-1, result-2])
turn_start(turn=2)
message_start/update/end(assistant final)
turn_end(assistant final, [])
agent_end(transcript)
```

事件中的消息使用创建事件时的 Pydantic snapshot，不能让历史事件引用后续被修改的 partial message。

## 7. 核心 loop 计划

伪代码只描述控制流，不是最终实现：

```python
async def run(self, prompt):
    self._enter_run()
    try:
        user_message = _normalize_prompt(prompt)
        self._messages.append(user_message)

        yield AgentStart()
        turn = 1
        yield TurnStart(turn=turn)
        yield MessageStart(message=user_message)
        yield MessageEnd(message=user_message)

        while True:
            context = Context(
                system_prompt=self.system_prompt,
                messages=list(self._messages),
                tools=[tool.to_ai_tool() for tool in self._tools.values()],
            )

            assistant = None
            assistant_started = False
            try:
                async for ai_event in self.models.stream_simple(
                    self.model,
                    context,
                    self.options,
                ):
                    match ai_event:
                        case StartEvent(partial=partial):
                            assistant_started = True
                            yield MessageStart(message=partial)
                        case TextStart() | TextDelta() | TextEnd() \
                           | ThinkingStart() | ThinkingDelta() | ThinkingEnd() \
                           | ToolCallStart() | ToolCallDelta() | ToolCallEnd():
                            yield MessageUpdate(
                                message=ai_event.partial,
                                ai_event=ai_event,
                            )
                        case DoneEvent(message=message):
                            assistant = message
                        case ErrorEvent(error=message):
                            assistant = message
            except Exception as exc:
                assistant = _error_assistant_message(exc)

            if assistant is None:
                assistant = _error_assistant_message(
                    RuntimeError("AI stream ended without a terminal event")
                )

            if not assistant_started:
                yield MessageStart(message=assistant)

            self._messages.append(assistant)
            yield MessageEnd(message=assistant)

            if assistant.stop_reason in {"error", "aborted"}:
                yield TurnEnd(message=assistant, tool_results=[])
                yield AgentEnd(messages=self.messages)
                return

            tool_calls = _tool_calls(assistant)
            if not tool_calls:
                yield TurnEnd(message=assistant, tool_results=[])
                yield AgentEnd(messages=self.messages)
                return

            tool_results = []
            for tool_call in tool_calls:
                yield ToolExecutionStart(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    args=tool_call.arguments,
                )

                if assistant.stop_reason == "length":
                    result = _truncated_tool_result(tool_call)
                    is_error = True
                else:
                    result, is_error = await self._execute_tool(tool_call)
                yield ToolExecutionEnd(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    result=result,
                    is_error=is_error,
                )

                tool_result = ToolResultMessage(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    content=result.content,
                    is_error=is_error,
                )
                tool_results.append(tool_result)
                self._messages.append(tool_result)
                yield MessageStart(message=tool_result)
                yield MessageEnd(message=tool_result)

            yield TurnEnd(message=assistant, tool_results=tool_results)
            turn += 1
            yield TurnStart(turn=turn)
    finally:
        self._leave_run()
```

最终实现仍使用 `match/case` 分发 AI 事件，但会避免为了伪代码整洁而增加只调用一次的 helper。

## 8. 工具执行语义

### 8.1 顺序执行

v1 固定按 assistant content 中的出现顺序执行：

- 事件顺序稳定。
- transcript 顺序稳定。
- 不需要管理多个 task 的完成顺序与取消。
- 对文件修改、shell 等有副作用工具更安全。

并行不是简单替换为 `asyncio.gather()`。一旦支持，必须同时定义事件完成顺序、结果持久化顺序、某个工具失败后的取消策略，因此留到出现真实性能需求时设计。

### 8.2 length 截断保护

如果 assistant 同时满足：

- `stop_reason == "length"`
- content 中存在 `ToolCall`

则所有 tool call 都不得执行。流式 JSON salvage 可能把截断参数修复成“语法合法但语义不完整”的对象。

每个调用仍产生完整生命周期：

```text
tool_execution_start
tool_execution_end(is_error=True)
message_start(toolResult error)
message_end(toolResult error)
```

错误结果加入上下文，让模型下一轮重新发出完整调用。

### 8.3 失败分类

| 情况 | Agent 行为 |
|---|---|
| 工具不存在 | error ToolResult，继续下一轮 |
| Python 参数绑定失败 | error ToolResult，继续下一轮 |
| `execute()` 抛 `Exception` | error ToolResult，继续下一轮 |
| `execute()` 被 task cancel | 不捕获 `CancelledError`，取消整个 run |
| 工具返回错误类型 | 作为工具执行异常处理 |
| 模型 `ErrorEvent` | 写入 assistant error message，结束 run |
| `Models.stream_simple()` 调用前抛错 | 构造 assistant error message，保持完整结束事件 |

工具异常文本会发送给模型。v1 不加入异常类型、traceback 或 provider details，避免把内部信息直接放进 prompt。

## 9. 状态和取消

### 9.1 Transcript 提交规则

- UserMessage 在模型请求前提交。
- AssistantMessage 只在 `DoneEvent` 或 `ErrorEvent` 后提交。
- Partial assistant 只存在于事件，不提前写入 transcript。
- ToolResultMessage 在对应工具结束后立即提交。

这样 AI 请求失败时 transcript 仍然有用户消息和一个明确的 assistant error message，不会留下半个普通 assistant 响应。

### 9.2 task cancellation

调用方负责 task：

```python
async def consume(agent: Agent):
    async for event in agent.run("do work"):
        ...

task = asyncio.create_task(consume(agent))
task.cancel()
```

v1 规则：

- 不提供 `Agent.cancel()`，也不模拟 AbortController。
- 不捕获并吞掉 `asyncio.CancelledError`。
- `finally` 必须恢复 `is_running=False`。
- 取消是异常退出，不保证再产出 `agent_end`。
- 如果取消发生在工具调用完成前，当前 transcript 可能不适合直接续跑；调用方可 `reset()`。等 session/continue 出现时，再设计 aborted message 的持久化语义。

### 9.3 并发保护

虽然 async generator 本身是顺序执行，两个 task 仍可能同时消费两个 `run()`。`is_running` guard 是 v1 必需能力，不属于 PI 的 `waitForIdle()` 复制品。

## 10. PI 功能边界表

PI 当前三个核心测试文件大约覆盖：

- `agent-loop.test.ts`：20 个 loop 场景。
- `agent.test.ts`：19 个状态、订阅、队列和生命周期场景。
- `e2e.test.ts`：10 个 faux provider 集成场景。

下表中的场景数会有交叉，目的是表示 PI 对该能力投入的测试重量，不用于求和。

| PI 功能 | PI 关联场景 | 本质需求 | rho v1 | 原因或触发条件 |
|---|---:|---|---|---|
| 基本文本和事件生命周期 | 4+ | UI 能观察模型响应 | 做 | Agent 的核心职责 |
| 多次 prompt 保留上下文 | 1 | 多轮对话 | 做 | Agent 持有 transcript 的直接价值 |
| thinking 事件和内容保留 | 1 | reasoning 模型支持 | 做 | AI 层已有类型 |
| 工具调用闭环 | 2+ | toolUse 后执行并继续 | 做 | Agent 的核心职责 |
| length 截断保护 | 1 | 禁止执行不完整参数 | 做 | 数据安全要求 |
| 未知工具和执行异常 | 实现分支 + e2e | 错误反馈给模型 | 做 | 工具系统基本边界 |
| 多工具顺序执行 | 多个 | 确定副作用和事件顺序 | 做 | v1 固定 sequential |
| 自定义 `AgentMessage` | 2 | UI 消息与 LLM 消息共存 | 不做 | 出现第一种非 `Message` transcript 项时加 |
| `convertToLlm` | 2 | 自定义消息转模型消息 | 不做 | 与自定义 AgentMessage 同时引入 |
| `transformContext` | 1 | 裁剪或注入上下文 | 不做 | compaction 或 RAG 进入 Agent 上层时加 |
| JSON Schema 参数校验 | 2 | 工具执行前严格验证 | 不做 | 有复杂 schema 工具时选定统一 validator |
| `prepareArguments` | 1 | 兼容模型输出变体 | 不做 | 真实 provider 持续输出错误形状时加 |
| `beforeToolCall` | 1+ | 权限、确认、参数修改 | 不做 | 出现需要用户确认或策略拦截的工具时加 |
| `afterToolCall` | 2 | 审计或修改结果 | 不做 | 出现跨工具统一后处理时加 |
| 并行工具执行 | 4 | 降低多个慢工具的总耗时 | 不做 | 同轮有两个独立且耗时明显的只读工具时加 |
| 工具进度事件 | 2 | 长工具向 UI 报进度 | 不做 | 首个超过数秒且有可报告阶段的工具出现时加 |
| terminate hint | 3 | 工具要求跳过后续模型调用 | 不做 | 有终止类工具时加 |
| `shouldStopAfterTurn` | 1 | turn 边界优雅停止 | 不做 | compaction 或预算策略需要时加 |
| `prepareNextTurn` | 2 | turn 间更换模型或上下文 | 不做 | orchestration 层出现时加 |
| steering queue | 3+ | 运行中注入用户消息 | 不做 | 有实时交互 UI 时加，使用 `deque` |
| follow-up queue | 3+ | 结束后继续排队任务 | 不做 | 有自动编排或交互队列时加 |
| `continue()` | 7+ | 从已有 user/toolResult 尾部恢复 | 不做 | session 恢复或显式 retry UX 出现时加 |
| `subscribe()` | 3 | 多观察者消费事件 | 不做 | 日志、UI、持久化必须独立订阅时加 |
| `waitForIdle()` | 2 | 等待后台生产和 listener barrier | 不做 | rho v1 没有后台生产者 |
| 显式 `abort()` | 3 | UI 主动取消后台任务 | 不做 | 调用方直接持有 asyncio Task |
| sessionId 转发 | 1 | provider cache affinity | 已由 options 支持 | 通过 `SimpleStreamOptions`，不加 Agent 专属字段 |
| Harness/Session/Compaction/Skills | 独立测试目录 | 完整应用运行时 | 不做 | Agent loop 稳定后作为上层模块单独设计 |

## 11. 测试计划

### 11.1 类型测试

- `AgentEvent` 每个分支能由 discriminator 往返解析。
- `AgentToolResult` 拒绝非法 content。
- Agent 构造时复制 tools/messages 顶层容器。
- Agent 拒绝空工具名和重复工具名。
- `messages` getter 不泄漏内部列表引用。

### 11.2 loop 单元测试

用真实 `Models` 对象或 autospec mock，patch `stream_simple` 为 async generator：

1. string prompt 的完整文本事件顺序。
2. `UserMessage` prompt 不被二次包装。
3. Text、Thinking、ToolCall AI 事件映射为 `MessageUpdate`。
4. 多次 `run()` 累积历史，并把完整历史传给下一次 Context。
5. 单工具成功，第二轮收到 ToolResultMessage。
6. 多工具严格顺序执行和持久化。
7. 未知工具生成 error result。
8. 参数绑定 `TypeError` 生成 error result。
9. 工具异常生成 error result。
10. length + tool call 时工具零执行，并驱动模型重试。
11. length + 无 tool call 时正常结束。
12. AI `ErrorEvent` 写入 transcript 并结束。
13. auth/provider 等调用前异常也产生完整结束事件。
14. AI stream 没有 start 或 terminal event 时仍生成完整错误生命周期。
15. 并发 `run()` 被拒绝。
16. cancellation 后 `is_running` 恢复。
17. `reset()` 清空或替换 transcript，运行中拒绝 reset。
18. system prompt、tools、SimpleStreamOptions 正确传入 AI 层。

### 11.3 集成测试

集成测试默认无凭据即 skip，不允许 hardcoded key：

- DeepSeek 基本文本响应。
- DeepSeek 单工具调用到最终文本的完整闭环。

不把真实 API 的具体措辞作为断言，只检查事件顺序、非空内容、tool call/result 对应和最终 stop reason。

## 12. 实施阶段

### 阶段 0：边界确认

- 讨论并确认本文档。
- 特别确认 Tool execute 的 `**arguments` API 和 v1 不做严格 schema validation。
- 确认取消时不保证 `agent_end`，不持久化 aborted partial。

### 阶段 1：类型

- 创建 `rho/agent/types.py`。
- 使用 Pydantic discriminated union 定义事件。
- 使用 dataclass 定义包含 callable 的 `AgentTool`。
- 完成类型往返和容器拷贝测试。

### 阶段 2：核心 loop

- 创建 `rho/agent/agent.py`。
- 实现 prompt 规范化、AI 事件映射、transcript 提交和并发 guard。
- 先通过无工具、错误和多轮上下文测试。

### 阶段 3：工具

- 加入顺序工具执行。
- 完成未知工具、参数错误、执行异常和 length 截断保护。
- 验证每个 ToolResultMessage 的 call id/name 与 assistant ToolCall 对应。

### 阶段 4：集成和导出

- 创建 `rho/agent/__init__.py`。
- 完成 DeepSeek 可选集成测试。
- 更新示例和模块 docstring。
- 单独 commit Agent v1，不混入 Harness 设计。

## 13. v1 验收标准

- `Agent.run()` 是唯一运行入口，返回 async generator。
- 不存在后台 task、EventStream、subscribe 或额外 low-level loop。
- 每次正常/错误结束都有确定的 lifecycle event 顺序。
- 工具执行和 ToolResultMessage 顺序可预测。
- length 截断工具绝不执行。
- AI 错误和工具错误不会破坏 transcript 结构。
- 并发调用不能交错修改消息历史。
- cancellation 不被吞掉，运行状态一定清理。
- 单元测试不访问网络，集成测试无 key 时 skip。
- `rho.ai` 不导入任何 `rho.agent` 模块。

## 14. 后续扩展信号

| 扩展 | 触发信号 | 预计设计位置 |
|---|---|---|
| 严格工具参数验证 | 复杂 schema 导致错误执行 | `AgentTool` args model 或统一 validator |
| 并行工具 | 两个以上独立慢工具成为常态 | `tool_execution.py` + `TaskGroup` |
| 工具进度 | shell/download 等长任务需要 UI 反馈 | execute progress callback + 新事件 |
| 权限 hooks | 文件写入、shell 等需要确认 | Agent 上层 policy，不默认塞进 loop |
| steering/follow-up | 实时交互 UI 出现 | `deque` + 明确 drain point |
| continue/retry | session 恢复或失败重试出现 | transcript 尾部验证 + continuation API |
| max turns/budget | 出现真实无限工具循环或费用控制需求 | turn policy，而不是硬编码分支 |
| custom messages | transcript 需要 UI-only 消息 | AgentMessage + conversion boundary |
| compaction | context window 成为实际限制 | 独立 compaction 层，通过 context transform 接入 |
| session | 需要恢复、分支和持久化 | AgentHarness/Session 上层模块 |

## 15. 当前明确不解决的问题

- 不支持多 Agent 协作。
- 不支持 session persistence。
- 不支持自动 compaction。
- 不支持动态工具注册。
- 不支持运行中切换模型。
- 不支持 provider proxy stream 注入。
- 不支持工具 progress、hooks 或并行执行。
- 不保证取消后的 transcript 可直接 continuation。

这些不是“之后顺手补齐”的清单。只有上表对应触发信号出现时，才重新设计边界。
