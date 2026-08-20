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

### harnesses

Contains declarative agent configuration files (YAML) used to define and run
agent setups without changing code. Each harness describes the prompt, enabled
tools, control limits (e.g. `max_iterations`), memory strategy, recovery
behavior, and verification settings. Harnesses can extend others via `parent`,
making it easy to iterate on variants (e.g. `baseline-v0`, `meta-v3`) for
experiments and evaluations.

### evaluation

Contains benchmark task definitions and the records produced by running the
agent harness against them. The `terminal_bench` subdirectory holds the task
set (e.g. `terminal-bench-2`), while `records` stores timestamped JSON result
files for each evaluation run, useful for comparing agent configurations over
time.

### tests

Contains unit tests and integration tests to verify the correctness and reliability of the runtime components.


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

Harbor is a project dependency (the `eval` group). Run it through `uv run` so it
shares this project's `.venv` with `agent_runtime` and its dependencies:

```bash
uv run harbor run \
  -d terminal-bench/terminal-bench-2 \
  --agent agent_runtime.integrations.harbor:HarborAgent \
  -l 1 \
  --debug
```

`uv run` puts the project's `.venv` on `sys.path` (where `agent_runtime` is
installed editable), so no `PYTHONPATH` prefix is needed.



## RunTrace

The agent loop emits a provider-independent event stream for each run. Keep it
in memory for tests, or append it as JSONL:

```python
from agent_runtime.agent import RunTrace, run_turn

trace = RunTrace(run_id="run-001", output_path="runs/run-001.jsonl")
answer = run_turn("What's the weather in Singapore?", llm, tools, trace=trace)
```

The event model contains lifecycle, LLM, and tool events: `agent.start`,
`agent.end`, `agent.error`, `llm.start`, `llm.error`, `llm.end`, `tool.start`,
`tool.error`, and `tool.end`. Each JSONL record includes
`run_id`, `event_id`, `event_type`, `timestamp`, `iteration`, and `data`.
The default payload is intentionally compact: it records counts, names, IDs,
durations, statuses, and errors, but not full prompts, model text, arguments, or
tool results.

Use `trace.span(name, ...)` to record an operation with explicit start, error,
and end events. The end and error events include `duration_ms`; result metadata
can be added through the yielded metadata dictionary.

## Environment Variables

The project uses an OpenAI-compatible Chat Completions API:

```dotenv
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://api.example.com/v1
MODEL_ID=your-model

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
