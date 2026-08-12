# Agent Runtime Lab

> A modular AI Agent runtime implementation built from scratch for understanding, experimenting, and extending modern AI Agent architectures.

Agent Runtime Lab is an open-source project that explores the core components behind AI Agent systems.

Instead of treating agents as black-box applications, this project focuses on understanding and implementing the underlying runtime mechanisms:

- Agent Loop
- Tool Calling
- Context Management
- Memory
- Planning
- Execution Runtime
- Reliability Engineering
- Observability


## Why This Project?

Modern AI Agents are moving from short-lived interactions toward long-running autonomous workflows.

The main challenge is no longer only model capability, but also:

- How agents manage state
- How agents execute actions reliably
- How agents recover from failures
- How agents observe and update their environment
- How developers build trustworthy agent systems


This project aims to build a modular Agent Runtime from first principles, where each component can be understood, tested, and extended independently.


## Project Structure


### docs

Contains design documents, technical concepts, architecture explanations, and interview notes.

It explains the "why" behind the Agent Runtime design, including core concepts, implementation decisions, and engineering trade-offs.

### src

Contains the core modular implementation of the Agent Runtime.

Each component is designed as an independent and reusable module, such as agent loop, tool execution, memory management, and reliability mechanisms.

### labs

Contains runnable experiments and quick-start guides for validating modules implemented in `src`.

Each lab provides a simple way to execute and explore a specific capability of the Agent Runtime.

### tests

Contains unit tests and integration tests to verify the correctness and reliability of the runtime components.

agent-runtime-lab/

├── README.md

├── docs/

│   ├── concepts/

│   │   ├── agent-loop.md
│   │   ├── message-lifecycle.md
│   │   ├── tool-calling.md
│   │   └── observation.md
│   │
│   ├── reliability/

│   │   ├── retry.md
│   │   ├── timeout.md
│   │   ├── validation.md
│   │   └── tracing.md
│   │
│   └── interview/

│       ├── agent-loop-qna.md
│       └── system-design.md


├── labs/

│   ├── 001-minimal-agent-loop

│   ├── 002-tool-calling

│   ├── 003-memory

│   ├── 004-planning

│   └── 005-reliable-agent


├── src/

│   └── agent_runtime/

│       ├── agent

│       ├── providers

│       └── tools


└── tests/

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
uv run python run_agent.py "List the files in the current directory"

```

If another project's virtual environment is active, use `uv run --active` or
activate this project's `.venv` first. The warning about `VIRTUAL_ENV` is from
uv and is separate from the agent runtime.

To test the shell tool directly without an LLM:

```bash
PYTHONPATH=src uv run python -c \
  'from agent_runtime.tools import run_command; print(run_command("printf hello"))'
```

Try a non-zero exit code, a working directory, and a timeout:

```bash
PYTHONPATH=src uv run python -c \
  'from agent_runtime.tools import run_command; print(run_command("pwd", cwd="/tmp"))'
PYTHONPATH=src uv run python -c \
  'from agent_runtime.tools import run_command; print(run_command("exit 7"))'
PYTHONPATH=src uv run python -c \
  'from agent_runtime.tools import run_command; print(run_command("sleep 2", timeout=0.1))'
```

## RunTrace

The agent loop emits a provider-independent event stream for each run. Keep it
in memory for tests, or append it as JSONL:

```python
from agent_runtime.agent import RunTrace, run_turn

trace = RunTrace(run_id="run-001", output_path="runs/run-001.jsonl")
answer = run_turn("What's the weather in Singapore?", llm, tools, trace=trace)
```

The event model contains lifecycle, LLM, and tool events: `agent.start`,
`agent.end`, `agent.error`, `llm.request`, `llm.response`, `tool.request`,
`tool.response`, and `tool.error`. Each JSONL record includes
`run_id`, `event_id`, `event_type`, `timestamp`, `iteration`, and `data`.
The default payload is intentionally compact: it records counts, names, IDs,
durations, statuses, and errors, but not full prompts, model text, arguments, or
tool results.

## Environment Variables

The project uses an OpenAI-compatible Chat Completions API:

```dotenv
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://api.example.com/v1
OPENAI_MODEL=your-model

```

Supports OpenAI as well as model providers offering compatible interfaces. Keep your actual API keys strictly in your local `.env` file and do not commit them to Git.

## Learning Roadmap

Independent experiments will be added progressively:

* Message history and context management
* Tool calling and parameter validation
* Error handling, retries, and timeouts
* Streaming output
* Multi-agent collaboration
* Memory and retrieval augmentation
* Task planning and state machines
* Observability, evaluation, and debugging
* External tool protocols such as MCP

Each experiment will maintain a minimal codebase, complete with runnable examples and tests.

## Project Philosophy

This project follows a simple principle:

> Understand agents by rebuilding the fundamental components from scratch.

By decomposing complex agent systems into independent modules, developers can:

- Learn how modern agents work internally
- Experiment with different architectures
- Reuse components like building blocks
- Build customized agent systems


## Who Is This For?

- Engineers learning AI Agent architecture
- Developers building custom agent systems
- Researchers exploring agent runtime design
- Anyone interested in reliable autonomous AI systems

## License

MIT
