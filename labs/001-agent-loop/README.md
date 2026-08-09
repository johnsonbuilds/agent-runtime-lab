# Lab 001: Agent Loop

## Purpose

This lab demonstrates the basic agent execution loop implemented in `agent_runtime/agent/loop.py`.

It shows:

- LLM interaction
- Tool calling flow
- Tool result observation
- Iteration control


## Quick Start

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync

```

Copy the environment variable template and fill in your configuration:

```bash
cp .env.example .env

```

Then run:

```bash
uv run python run_agent.py "How's the weather in Beijing today?"

```