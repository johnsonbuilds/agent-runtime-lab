# Agent Practice

一个面向学习者的 Agent 原理实践仓库。

这个项目不追求成为 OpenClaw、Hermes Agent 或代码智能体那样的大型应用，而是把大型 Agent 系统中的关键技术拆成短小、可运行、可调试的实验。每个技术点都尽量只回答一个问题：它为什么需要存在，它如何工作，以及它出错时会发生什么。

项目也记录作者自己的学习过程，欢迎准备从事 AI Agent 开发的朋友一起实践、提问和改进。

## 当前内容

### Agent Loop

当前示例展示一个最小的工具调用闭环：

1. 用户发送问题。
2. LLM 决定是否调用工具。
3. Agent 执行工具并把结果追加回消息历史。
4. LLM 根据工具结果生成最终回答。
5. 通过最大迭代次数避免模型陷入无限工具调用。

相关目录：

- `agent_practice/agent/`：Agent loop，只依赖模型和工具接口，不包含具体实现。
- `agent_practice/tools/`：工具注册、schema 和内置工具示例。
- `agent_practice/providers/`：模型供应商适配，目前包含 OpenAI 兼容接口。
- `run_agent.py`：组装模型、工具和 loop，并提供命令行入口。

## 快速开始

需要 Python 3.12+ 和 [uv](https://docs.astral.sh/uv/)。

```bash
uv sync
```

复制环境变量模板并填写自己的配置：

```bash
cp .env.example .env
```

然后运行：

```bash
uv run python run_agent.py "北京今天天气怎么样？"
```

## 环境变量

项目使用 OpenAI 兼容的 Chat Completions 接口：

```dotenv
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://api.example.com/v1
OPENAI_MODEL=your-model
```

支持 OpenAI 以及提供兼容接口的模型服务商。真实密钥只放在本地 `.env`，不要提交到 Git。

## 学习路线

后续会逐步加入独立实验：

- 消息历史与上下文管理
- Tool calling 与参数校验
- 异常处理、重试和超时
- 流式输出
- 多 Agent 协作
- 记忆与检索增强
- 任务规划与状态机
- 可观测性、评估和调试
- MCP 等外部工具协议

每个实验都会保持较小的代码规模，并配套可运行示例和测试。

## 模块化约定

第一阶段只保留三个模块目录，先按职责分组，不追求过早拆分：

1. `agent` 解决“如何循环调用模型和工具”。
2. `tools` 解决“有哪些工具以及如何注册和执行”。
3. `providers` 解决“如何连接不同模型服务商”。

后续只有当一个主题出现独立的实现和示例时，才新增目录，例如
`memory/`、`planning/`、`observability/`。具体组合统一放在 `run_agent.py` 或新的示例入口中，核心模块不直接写死业务选择。

## 参与贡献

欢迎提交 Issue、补充实验、改进测试或分享学习过程。新增内容建议保持单一主题、可独立运行，并说明关键设计取舍。

## License

MIT
