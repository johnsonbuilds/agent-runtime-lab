"""Small, provider-independent event tracing for agent runs."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Generator
from uuid import uuid4


@dataclass(frozen=True)
class RunEvent:
    """One immutable event emitted while an agent run is executing."""

    run_id: str
    event_id: str
    event_type: str
    timestamp: float
    iteration: int
    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RunTrace:
    """Collect events in memory and optionally append them as JSONL."""

    def __init__(self, run_id: str | None = None,
                 output_path: str | Path | None = None,
                 sink: Callable[[RunEvent], None] | None = None) -> None:
        self.run_id = run_id or f"run-{uuid4().hex}"
        self.events: list[RunEvent] = []
        self._output_path = Path(output_path) if output_path else None
        self._sink = sink

    def emit(self, event_type: str, iteration: int = 0,
             **data: Any) -> RunEvent:
        event = RunEvent(self.run_id, f"evt-{uuid4().hex}", event_type,
                         time.time(), iteration, data)
        self.events.append(event)
        if self._output_path:
            self._output_path.parent.mkdir(parents=True, exist_ok=True)
            with self._output_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event.to_dict(), ensure_ascii=True,
                                        default=str) + "\n")
        if self._sink:
            self._sink(event)
        return event

    @contextmanager
    def span(self, name: str, iteration: int | None = None,
             **meta: Any) -> Generator[dict[str, Any], None, None]:
        """Trace an operation with explicit start, error, and end events."""
        event_iteration = iteration if iteration is not None else 0
        span_meta = dict(meta)
        started = time.monotonic()
        self.emit(f"{name}.start", event_iteration, **span_meta)
        failed = False
        try:
            yield span_meta
        except Exception as exc:
            failed = True
            span_meta["duration_ms"] = round((time.monotonic() - started) * 1000, 3)
            error_meta = dict(span_meta)
            error_meta["error"] = str(exc)
            self.emit(f"{name}.error", event_iteration, **error_meta)
            raise
        finally:
            span_meta["duration_ms"] = round((time.monotonic() - started) * 1000, 3)
            span_meta["status"] = "error" if failed else "success"
            self.emit(f"{name}.end", event_iteration, **span_meta)


__all__ = ["RunEvent", "RunTrace"]
