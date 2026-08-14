# Lab 003: Run a Terminal-Bench Task

## Purpose

This lab runs the existing `agent-runtime-lab` Agent Loop inside a Harbor
Terminal-Bench trial.

The execution path is:

```text
Harbor BaseAgent
    -> HarborAgent.run()
    -> agent_runtime.run_turn()
    -> OpenAICompatibleLLM
    -> ToolRegistry
    -> HarborShellExecutor
    -> Harbor BaseEnvironment
```

The Agent Loop is not duplicated in the Harbor adapter. Only the execution
environment is changed from local subprocesses to the Harbor sandbox.

## Prerequisites

Requires Python 3.12+, `uv`, Docker or another Harbor environment provider, and
an installed Harbor CLI:

```bash
harbor --help
```

Configure the OpenAI-compatible provider in the project root `.env`:

```dotenv
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://your-compatible-provider.example/v1
OPENAI_MODEL=your-model
```

The Harbor adapter loads `.env` before creating the default LLM. Do not commit
real API keys.

## Run One Trial

The project uses a `src/` layout. When calling the global `harbor` command,
make that directory importable:

```bash
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" harbor run \
  -d terminal-bench/terminal-bench-2 \
  --agent agent_runtime.integrations.harbor:HarborAgent \
  -l 1 \
  --debug
```

`-l 1` means one trial, not one Agent Loop iteration. The runtime may perform
multiple LLM and tool iterations for a single task.

## Monitor A Trial

Harbor creates a job directory similar to:

```text
jobs/2026-08-14__18-22-33/
```

The useful per-trial files are:

```text
jobs/<job>/<trial>/agent/agent-runtime.jsonl
jobs/<job>/<trial>/trial.log
jobs/<job>/<trial>/result.json
```

The runtime trace contains events such as:

```text
agent.start
llm.start
llm.end
tool.start
tool.end
agent.end
```

Inspect the trace while or after a trial:

```bash
nl -ba jobs/<job>/<trial>/agent/agent-runtime.jsonl
```

Useful fields include:

- `event_type`: lifecycle event name
- `iteration`: Agent Loop iteration
- `data.duration_ms`: operation duration
- `data.status`: `success` or `error`
- `data.tool`: tool name
- `data.error`: error information

The adapter also writes runtime information into Harbor's `AgentContext`:

```json
{
  "agent_runtime": {
    "status": "completed",
    "event_count": 12,
    "trace_path": ".../agent-runtime.jsonl"
  }
}
```

## Understand Trial Status

These states are different:

- `completed`: the Agent Loop returned an answer
- `passed`: the Terminal-Bench verifier accepted the task
- `failed`: the agent or verifier reported a failure
- `cancelled`: the trial was stopped before completion

Successful `llm.end` and `tool.end` events do not mean that the benchmark task
has passed. The task must finish and then pass its verifier.

If the terminal shows `canceling trial`, wait for Harbor cleanup to finish
before inspecting the final `result.json`. A manually interrupted trial is
normally recorded as `cancelled`, and it will not have a verifier result.

## Troubleshooting

### `No module named agent_runtime`

Run Harbor with the `PYTHONPATH` prefix shown above, or install the project into
the same environment that provides the Harbor command:

```bash
uv pip install -e .
```

### Missing OpenAI credentials

Check that `.env` exists in the project root and contains a real
`OPENAI_API_KEY`. Alternatively export the variables before running Harbor:

```bash
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://your-compatible-provider.example/v1"
export OPENAI_MODEL="your-model"
```

### The trial appears to run for a long time

A Terminal-Bench task can require many LLM calls and shell commands. Check the
trace before cancelling it. If the latest event is `llm.start`, the runtime is
waiting for the model response. If the latest event is `tool.start`, inspect the
command and the Harbor trial log.

### No verifier result

The verifier result is only available after the agent phase finishes and Harbor
runs the verifier. A manually cancelled or still-running trial will not have a
final verifier result.

## Minimal Smoke Test

Before running a full benchmark task, verify that the adapter and runtime can
be imported:

```bash
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" \
  python -c 'from agent_runtime.integrations.harbor import HarborAgent; print(HarborAgent.name())'
```

This command requires the Harbor Python package to be installed in the current
Python environment. It does not start a trial or call the LLM.
