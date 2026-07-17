# CLAUDE.md — rho 项目开发约定

## 核心原则

**参考 PI 的设计理念，但不复制 PI 的实现。** PI 是 TypeScript 项目，很多设计是为了弥补 TS 缺少的原语（没有 async generator → EventStream 类；没有 discriminated union → TypeBox；bundler 需要 tree-shaking → lazyApi）。Python 有更优秀的原生能力，rho 要做出对 Python 架构更优化的设计。

### 设计决策优先级

1. **Python 习惯优先** — async generator、context manager、match/case、collections.deque 是 Python 的一等公民，用它们
2. **简洁优先** — 少一层抽象比完整一层好。v1 不做未来可能需要的东西
3. **PI 做参考，不做对标** — 理解 PI 为什么那样做（约束是什么），然后问：Python 有更好的方式吗？

### 具体例子

| PI 做法 | PI 为什么这样做 | rho 做法 | rho 为什么更好 |
|---------|----------------|---------|---------------|
| `EventStream<T,R>` push 类 | TS 无原生 async generator | `AsyncGenerator[Event]` | Python 原生，语法更简洁 |
| `streamFn` 注入 Agent | 测试 seam（替换 faux provider） | 直接传 `Models` 实例 | 少一层间接，mock 用 `unittest.mock.patch` |
| `PendingMessageQueue` 类 | TS 无标准队列 | `collections.deque` | 标准库即用 |
| `AbortController` | Web API | `asyncio.Task.cancel()` | Python 原生取消 |
| getter/setter 防引用泄露 | TS 无 `list()` 拷贝习惯 | 显式 `list()` 拷贝 | Python 习惯，更清晰 |
| `AgentMessage ≠ Message` | 支持自定义消息类型 | v1 直接用 `Message` | 无需求不抽象 |
| TypeBox schemas | 运行时验证 JSON Schema | Pydantic v2 | Python 生态标准 |

### 不做的事

- **不为"完整性"加功能** — PI 有 steering/followUp 是因为它的 agent 需要被外部中断。rho v1 没有这个需求，不加
- **不提前抽象** — AgentHarness、Session、Compaction 是 PI 的重要部分，但在 agent 没跑起来之前不设计。先让 agent loop 工作，再谈封装
- **不强行对标 PI 的目录结构** — PI 拆了 agent-loop.ts / agent.ts / proxy.ts，因为模块之间有 bundler 依赖。Python 不需要这样拆

### 编码风格

- Pydantic v2 discriminated unions: `Annotated[Union[...], Field(discriminator="type")]`
- Async generator 返回类型: `AsyncGenerator[Event, None]`
- `match/case` 做事件分发，不用 if-elif 链
- 文件按职责分，单文件不超过 500 行为佳
- 中文 docstring 用于架构级说明，英文 docstring 用于 API 级
