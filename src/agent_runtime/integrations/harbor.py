"""Harbor BaseAgent adapter for agent-runtime-lab."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from agent_runtime.agent import run_turn
from agent_runtime.execution.harbor import HarborShellExecutor
from agent_runtime.providers import OpenAICompatibleLLM
from agent_runtime.tools import create_default_registry
from agent_runtime.trace import RunEvent, RunTrace


class HarborAgent(BaseAgent):
    """Thin Harbor adapter that delegates execution to the existing runtime."""

    SUPPORTS_WINDOWS = False

    def __init__(self, *args: Any, llm: Any | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.llm = llm if llm is not None else OpenAICompatibleLLM(model=self.model_name)

    @staticmethod
    def name() -> str:
        return "agent-runtime-lab"

    def version(self) -> str:
        return "0.1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        """The runtime needs no environment setup before the first turn."""

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        trace_path = self.logs_dir / "agent-runtime.jsonl"
        runtime_metadata = self._runtime_metadata(context, instruction, trace_path)

        def sync_context() -> None:
            metadata = dict(context.metadata or {})
            metadata["agent_runtime"] = dict(runtime_metadata)
            context.metadata = metadata

        def record_event(event: RunEvent) -> None:
            runtime_metadata["event_count"] = len(trace.events)
            runtime_metadata["last_event"] = event.to_dict()
            sync_context()

        trace = RunTrace(output_path=trace_path, sink=record_event)
        tools = create_default_registry(HarborShellExecutor(environment))
        runtime_metadata["status"] = "running"
        sync_context()

        try:
            answer = await run_turn(instruction, self.llm, tools, trace=trace)
            agent_error = next(
                (event for event in reversed(trace.events)
                 if event.event_type == "agent.error"),
                None,
            )
            runtime_metadata["status"] = "failed" if agent_error else "completed"
            runtime_metadata["answer"] = answer
            if agent_error:
                runtime_metadata["error"] = agent_error.data.get("error", answer)
            sync_context()
        except BaseException as exc:
            runtime_metadata["status"] = "failed"
            runtime_metadata["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            sync_context()
            raise
        finally:
            await trace.flush()

    @staticmethod
    def _runtime_metadata(
        context: AgentContext,
        instruction: str,
        trace_path: Path,
    ) -> dict[str, Any]:
        metadata = dict(context.metadata or {})
        runtime_metadata = dict(metadata.get("agent_runtime") or {})
        runtime_metadata.update({
            "instruction": instruction,
            "trace_path": str(trace_path),
            "event_count": 0,
        })
        metadata["agent_runtime"] = runtime_metadata
        context.metadata = metadata
        return runtime_metadata


__all__ = ["HarborAgent"]
