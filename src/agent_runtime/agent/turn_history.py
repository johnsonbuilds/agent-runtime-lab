"""The turn-local conversation: full history plus its request view.

``messages`` is the single source of truth shared across turns; ``view``
is the (possibly compacted) projection actually sent to the model and
watched by the budget gate.  The two are appended in lockstep through a
single entry point, so the invariant that used to rest on discipline at
every append site — update one list, never forget the other — is now
enforced by construction.
"""

from __future__ import annotations

from typing import Any

from agent_runtime.agent.memory import apply_memory_strategy


class TurnHistory:
    """Conversation history and its request view, appended in lockstep.

    ``view`` starts as an independent copy of the initialized
    conversation so the two never alias.  ``refresh_view`` may replace
    the view wholesale when a memory strategy produces a compacted copy;
    subsequent appends still land in both.  When the strategy returns
    ``messages`` itself (``full_history``), the view keeps its own
    identity — it must never alias the full list, or mirrored appends
    would double up.
    """

    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self.messages = messages
        self.view: list[dict[str, Any]] = list(messages)

    def append(self, message: dict[str, Any]) -> None:
        """Append one message to the history and the request view."""
        self.messages.append(message)
        self.view.append(message)

    async def refresh_view(self, strategy: str, *, trace: Any = None,
                           iteration: int | None = None) -> None:
        """Rebuild the request view via the given memory strategy."""
        new_view = await apply_memory_strategy(
            self.messages, strategy, compact_messages=self.view,
            trace=trace, iteration=iteration)
        if new_view is not self.messages:
            self.view = new_view
