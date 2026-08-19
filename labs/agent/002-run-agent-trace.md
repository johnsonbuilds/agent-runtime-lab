# Lab 002: Run Agent Trace

## Purpose

This lab demonstrates a real OpenAI-compatible LLM run with JSONL tracing.
The trace records the lifecycle of the agent, LLM requests, and tool calls:

- `agent.start` / `agent.end`
- `llm.start` / `llm.error` / `llm.end`
- `tool.start` / `tool.error` / `tool.end`

Each span end event includes its duration and status.

## Configuration

Requires Python 3.12+, `uv`, and an OpenAI-compatible API key. Configure the
provider in `.env`:

```dotenv
LLM_API_KEY=your-api-key
MODEL_ID=gpt-4o-mini
# LLM_BASE_URL=https://your-compatible-provider.example/v1
```

Install dependencies if needed:

```bash
uv sync
```

## Plain LLM Run

Run one prompt and write events to a JSONL file:

```bash
uv run agent-practice \
  --trace runs/manual-plain.jsonl \
  "用一句话介绍什么是 Python。"
```

The default registry is passed to the model on every request. Therefore, even
a simple prompt may cause the model to call `run_command` if it decides that a
tool is useful. `llm.start.data.tool_count` shows how many tools were offered;
the actual tool calls appear in `llm.end.data.tools` and as `tool.*` events.

## Tool Calling Run

Explicitly ask the model to call the built-in shell tool:

```bash
uv run agent-practice \
  --trace runs/manual-tool.jsonl \
  "必须先调用 run_command 执行 pwd，再根据工具结果回答。"
```

A successful run normally contains this event order:

```text
agent.start
llm.start
llm.end
tool.start
tool.end
llm.start
llm.end
agent.end
```

## Interactive Run

Omit the prompt to keep a conversation open. All turns append to the same
trace file and share the same conversation state:

```bash
uv run agent-practice --trace runs/interactive.jsonl
```

Type prompts at the `>` prompt and enter `exit` or `quit` to stop.

## Inspect Events

The output file is JSONL, with one event per line:

```bash
nl -ba runs/manual-tool.jsonl
```

Useful fields include:

- `event_type`: lifecycle event name
- `iteration`: agent loop iteration
- `data.duration_ms`: operation duration
- `data.status`: `success` or `error` on span end events
- `data.tool`: tool name when a tool span is present
- `data.error`: error text on error events

The trace intentionally does not include full prompts, model text, tool
arguments, or tool results.
