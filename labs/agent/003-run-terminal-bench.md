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

Requires Python 3.12+, `uv`, Docker or another Harbor environment provider.
Harbor is installed as a project dependency (the `eval` group in
`pyproject.toml`), so it runs inside this project's `.venv`:

```bash
uv run harbor --help
```

Configure the OpenAI-compatible provider in the project root `.env`:

```dotenv
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://your-compatible-provider.example/v1
MODEL_ID=your-model
```

The Harbor adapter loads `.env` before creating the default LLM. Do not commit
real API keys.

## Run One Trial

Run Harbor through `uv run` so it uses this project's `.venv`, where
`agent_runtime` and its dependencies are already installed:

```bash
uv run harbor run \
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

Run Harbor through `uv run` from the project root, so it uses the project's
`.venv` where `agent_runtime` is installed. Avoid the globally installed
`harbor` command (for example one from `uv tool install`): it lives in a
separate environment without `agent_runtime` or its dependencies.

### Missing OpenAI credentials

Check that `.env` exists in the project root and contains a real
`LLM_API_KEY`. Alternatively export the variables before running Harbor:

```bash
export LLM_API_KEY="..."
export LLM_BASE_URL="https://your-compatible-provider.example/v1"
export MODEL_ID="your-model"
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
uv run python -c 'from agent_runtime.integrations.harbor import HarborAgent; print(HarborAgent.name())'
```

This command does not start a trial or call the LLM.
