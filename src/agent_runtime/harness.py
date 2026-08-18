"""Harness genome: the named, versioned configuration that decides how an
agent works, expressed as data instead of code.

The runtime engine (loop, trace, events) stays stable; a harness spec
selects prompt, tools, control flow, memory, recovery and verification
behavior.  Specs are loaded from YAML manifests so harness versions can
be diffed, compared, and later mutated one gene at a time:

    baseline-v0  --(enable verification)-->  baseline-v1

Every run records its harness id and a content hash of the genes, so a
trace can always be traced back to the exact harness that produced it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


class HarnessError(ValueError):
    """Raised when a harness manifest is missing or invalid."""


ITERATION_LIMIT_NOTICE = (
    "Iteration limit reached. Summarize the progress and give the "
    "best possible final answer. Do not call tools."
)


def feed_error_and_continue(exc: Exception, *, tool: str | None = None) -> str:
    """Turn a tool failure into an observation and let the loop continue."""
    return f"Tool error: {exc}"


ToolErrorStrategy = Callable[..., str]

TOOL_ERROR_STRATEGIES: dict[str, ToolErrorStrategy] = {
    "feed_error_and_continue": feed_error_and_continue,
}

MEMORY_STRATEGIES: tuple[str, ...] = ("full_history",)


@dataclass(frozen=True)
class PromptGenome:
    system: str = ""
    iteration_limit_notice: str = ITERATION_LIMIT_NOTICE


@dataclass(frozen=True)
class ToolGenome:
    enabled: tuple[str, ...] = ("run_command",)


@dataclass(frozen=True)
class ControlGenome:
    max_iterations: int = 10


@dataclass(frozen=True)
class MemoryGenome:
    strategy: str = "full_history"


@dataclass(frozen=True)
class RecoveryGenome:
    tool_error: str = "feed_error_and_continue"


@dataclass(frozen=True)
class VerificationGenome:
    enabled: bool = False


def _hash_genes(genes: Mapping[str, Any]) -> str:
    canonical = json.dumps(genes, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class HarnessSpec:
    """One harness version: lineage metadata plus seven genes."""

    id: str = "baseline-v0"
    parent: str | None = None
    mutation: str | None = None
    reason: str | None = None
    prompt: PromptGenome = field(default_factory=PromptGenome)
    tools: ToolGenome = field(default_factory=ToolGenome)
    control: ControlGenome = field(default_factory=ControlGenome)
    memory: MemoryGenome = field(default_factory=MemoryGenome)
    recovery: RecoveryGenome = field(default_factory=RecoveryGenome)
    verification: VerificationGenome = field(default_factory=VerificationGenome)
    skills: tuple[str, ...] = ()
    source: str | None = None
    genes_hash: str = ""

    def __post_init__(self) -> None:
        if not self.genes_hash:
            object.__setattr__(self, "genes_hash", _hash_genes(self.genes_dict()))

    def genes_dict(self) -> dict[str, Any]:
        """The behavior-defining genes, excluding lineage and source."""
        return {
            "prompt": asdict(self.prompt),
            "tools": {"enabled": list(self.tools.enabled)},
            "control": asdict(self.control),
            "memory": asdict(self.memory),
            "recovery": asdict(self.recovery),
            "verification": asdict(self.verification),
            "skills": list(self.skills),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "parent": self.parent,
            "mutation": self.mutation,
            "reason": self.reason,
            **self.genes_dict(),
            "source": self.source,
            "genes_hash": self.genes_hash,
        }

    def tool_error_observation(self, exc: Exception, **context: Any) -> str:
        """Apply the recovery gene to a tool failure."""
        try:
            strategy = TOOL_ERROR_STRATEGIES[self.recovery.tool_error]
        except KeyError as missing:
            raise HarnessError(
                f"unknown tool_error strategy: {missing.args[0]!r}") from missing
        return strategy(exc, **context)


DEFAULT_HARNESS = HarnessSpec()


def default_harness() -> HarnessSpec:
    """The built-in baseline; mirrors harnesses/baseline-v0.yaml."""
    return DEFAULT_HARNESS


def _check_keys(section: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = set(section) - allowed
    if unknown:
        raise HarnessError(f"unknown {where} keys: {sorted(unknown)}")


def _optional_text(data: Mapping[str, Any], key: str, where: str) -> str | None:
    value = data.get(key)
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise HarnessError(f"{where}.{key} must be non-empty text or null")
    return value


def _required_text(section: Mapping[str, Any], key: str, where: str,
                   default: Any = None) -> Any:
    value = section.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise HarnessError(f"{where}.{key} must be non-empty text")
    return value


def _name_list(section: Mapping[str, Any], key: str, where: str) -> tuple[str, ...]:
    value = section.get(key, [])
    if not isinstance(value, list):
        raise HarnessError(f"{where}.{key} must be a list")
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise HarnessError(f"{where}.{key} entries must be non-empty text")
    return tuple(value)


def _prompt_genome(data: Mapping[str, Any]) -> PromptGenome:
    section = data.get("prompt") or {}
    if not isinstance(section, Mapping):
        raise HarnessError("prompt must be a mapping")
    _check_keys(section, {"system", "iteration_limit_notice"}, "prompt")
    system = section.get("system", "")
    if not isinstance(system, str):
        raise HarnessError("prompt.system must be text")
    return PromptGenome(
        system=system,
        iteration_limit_notice=_required_text(
            section, "iteration_limit_notice", "prompt", ITERATION_LIMIT_NOTICE),
    )


def _tools_genome(data: Mapping[str, Any]) -> ToolGenome:
    section = data.get("tools") or {}
    if not isinstance(section, Mapping):
        raise HarnessError("tools must be a mapping")
    _check_keys(section, {"enabled"}, "tools")
    return ToolGenome(enabled=_name_list(section, "enabled", "tools"))


def _control_genome(data: Mapping[str, Any]) -> ControlGenome:
    section = data.get("control") or {}
    if not isinstance(section, Mapping):
        raise HarnessError("control must be a mapping")
    _check_keys(section, {"max_iterations"}, "control")
    value = section.get("max_iterations", 10)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise HarnessError("control.max_iterations must be an integer >= 1")
    return ControlGenome(max_iterations=value)


def _memory_genome(data: Mapping[str, Any]) -> MemoryGenome:
    section = data.get("memory") or {}
    if not isinstance(section, Mapping):
        raise HarnessError("memory must be a mapping")
    _check_keys(section, {"strategy"}, "memory")
    strategy = _required_text(section, "strategy", "memory", "full_history")
    if strategy not in MEMORY_STRATEGIES:
        raise HarnessError(
            f"memory.strategy must be one of {list(MEMORY_STRATEGIES)}, "
            f"got {strategy!r}")
    return MemoryGenome(strategy=strategy)


def _recovery_genome(data: Mapping[str, Any]) -> RecoveryGenome:
    section = data.get("recovery") or {}
    if not isinstance(section, Mapping):
        raise HarnessError("recovery must be a mapping")
    _check_keys(section, {"tool_error"}, "recovery")
    tool_error = _required_text(section, "tool_error", "recovery",
                                "feed_error_and_continue")
    if tool_error not in TOOL_ERROR_STRATEGIES:
        raise HarnessError(
            f"recovery.tool_error must be one of "
            f"{sorted(TOOL_ERROR_STRATEGIES)}, got {tool_error!r}")
    return RecoveryGenome(tool_error=tool_error)


def _verification_genome(data: Mapping[str, Any]) -> VerificationGenome:
    section = data.get("verification") or {}
    if not isinstance(section, Mapping):
        raise HarnessError("verification must be a mapping")
    _check_keys(section, {"enabled"}, "verification")
    enabled = section.get("enabled", False)
    if not isinstance(enabled, bool):
        raise HarnessError("verification.enabled must be a boolean")
    return VerificationGenome(enabled=enabled)


def from_dict(data: Mapping[str, Any]) -> HarnessSpec:
    """Validate manifest data and build a harness spec."""
    if not isinstance(data, Mapping):
        raise HarnessError("harness manifest must be a mapping")
    # "source" and "genes_hash" are outputs of to_dict(); accepted so a
    # spec can round-trip, but ignored and recomputed here.
    _check_keys(data, {"id", "parent", "mutation", "reason", "prompt", "tools",
                       "control", "memory", "recovery", "verification", "skills",
                       "source", "genes_hash"},
                "manifest")
    harness_id = data.get("id", "baseline-v0")
    if not isinstance(harness_id, str) or not harness_id.strip():
        raise HarnessError("manifest.id must be non-empty text")
    return HarnessSpec(
        id=harness_id,
        parent=_optional_text(data, "parent", "manifest"),
        mutation=_optional_text(data, "mutation", "manifest"),
        reason=_optional_text(data, "reason", "manifest"),
        prompt=_prompt_genome(data),
        tools=_tools_genome(data),
        control=_control_genome(data),
        memory=_memory_genome(data),
        recovery=_recovery_genome(data),
        verification=_verification_genome(data),
        skills=_name_list(data, "skills", "manifest"),
    )


def load_harness(path: str | Path) -> HarnessSpec:
    """Load and validate a harness manifest from a YAML file."""
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise HarnessError(f"harness manifest not found: {manifest_path}")
    try:
        data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise HarnessError(f"invalid harness YAML in {manifest_path}: {exc}") from exc
    spec = from_dict(data or {})
    object.__setattr__(spec, "source", str(manifest_path))
    return spec


def resolve_harness(value: str | None) -> HarnessSpec:
    """Resolve a CLI/env harness reference to a spec.

    Accepts a file path or a bare id resolved against ``harnesses/``;
    falls back to the built-in baseline when nothing is given.
    """
    if not value:
        return DEFAULT_HARNESS
    direct = Path(value)
    if direct.is_file():
        return load_harness(direct)
    as_id = Path("harnesses") / f"{value}.yaml"
    if as_id.is_file():
        return load_harness(as_id)
    raise HarnessError(
        f"harness {value!r} not found as a file or as {as_id}")


__all__ = [
    "DEFAULT_HARNESS", "HarnessError", "HarnessSpec", "ITERATION_LIMIT_NOTICE",
    "PromptGenome", "ToolGenome", "ControlGenome", "MemoryGenome",
    "RecoveryGenome", "VerificationGenome", "TOOL_ERROR_STRATEGIES",
    "MEMORY_STRATEGIES", "default_harness", "feed_error_and_continue",
    "from_dict", "load_harness", "resolve_harness",
]
