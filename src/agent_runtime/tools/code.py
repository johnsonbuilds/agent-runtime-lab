"""The execute_code composite tool: write a script, run it, observe once.

This is the "code interpreter" execution pattern: instead of N round
trips (write file, run command, read output), the model ships one
complete script per tool call.  Intermediate data stays in workspace
files or in-memory variables; only what the script prints comes back.

Results are returned raw: truncation and spill-to-``.outputs/`` are the
transcript boundary's job (agent_runtime.agent.observations), so every
tool's oversized output is handled uniformly.
"""

from __future__ import annotations

import os
import re
import shlex
from typing import Any
from agent_runtime.execution.base import ShellExecutor, Workspace
from agent_runtime.execution.local import LocalShellExecutor, LocalWorkspace
from agent_runtime.execution.guard import GuardDecision


def _python_binary() -> str:
    return os.getenv("AGENT_RUNTIME_PYTHON", "python3")


LANGUAGE_SPECS: dict[str, dict[str, str]] = {
    "python": {"extension": "py",
               "interpreter": _python_binary,
               "install_hint": "install python3 (e.g. apt-get install -y python3)"},
    "bash": {"extension": "sh",
             "interpreter": "bash",
             "install_hint": ""},
    "r": {"extension": "R",
          "interpreter": "Rscript",
          "install_hint": "install R (e.g. apt-get install -y r-base-core)"},
    "node": {"extension": "js",
             "interpreter": "node",
             "install_hint": "install Node.js (e.g. apt-get install -y nodejs)"},
}
LANGUAGES = {name: spec["extension"] for name, spec in LANGUAGE_SPECS.items()}
SCRIPTS_DIR = ".scripts"
OUTPUTS_DIR = ".outputs"
DEFAULT_TIMEOUT = 120.0

_SEQUENCE = re.compile(r"^(\d{4})\.")


def _interpreter_for(language: str) -> str:
    interpreter = LANGUAGE_SPECS[language]["interpreter"]
    return interpreter() if callable(interpreter) else interpreter


async def _next_script_name(workspace: Workspace, extension: str) -> str:
    """Continue the .scripts/NNNN.ext sequence across all languages."""
    listing = await workspace.list_dir(SCRIPTS_DIR)
    highest = 0
    for entry in listing.get("entries", []):
        match = _SEQUENCE.match(entry.get("name", ""))
        if match:
            highest = max(highest, int(match.group(1)))
    return f"{SCRIPTS_DIR}/{highest + 1:04d}.{extension}"


def _run_command_for(language: str, script: str) -> str:
    return f"{_interpreter_for(language)} {shlex.quote(script)}"


def _guard_decision(executor: ShellExecutor, script: str,
                    language: str) -> GuardDecision | None:
    """Check the script's own text against the executor's guard.

    Only bash scripts are checked, line by line: the wrapper command
    (``bash .scripts/0001.sh``) is always harmless, so what needs
    vetting is the script content.  Python is not textually guarded —
    it is Turing-complete, so any check on its source is trivially
    bypassed and would also block legitimate file work; the real
    boundary for code is the execution environment (container vs
    local).
    """
    guard = getattr(executor, "guard", None)
    if guard is None or language != "bash":
        return None
    for line in script.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        decision = guard.check(stripped)
        if not decision.allowed:
            return decision
    return None


def _with_interpreter_hint(result: dict[str, Any],
                           language: str) -> dict[str, Any]:
    """Append an install hint when the interpreter was not found.

    Exit code 127 means the shell could not find the interpreter binary
    (e.g. ``Rscript: command not found`` on a machine without R).  Turn
    that opaque failure into an actionable hint so the agent can install
    the runtime via ``run_command`` and retry.
    """
    if result.get("exit_code") != 127:
        return result
    interpreter = _interpreter_for(language)
    hint = LANGUAGE_SPECS[language]["install_hint"]
    message = (f"hint: '{interpreter}' was not found; the script was not run."
               f" Install the {language} runtime first via run_command"
               + (f" ({hint})" if hint else ""))
    stderr = result.get("stderr") or ""
    if message not in stderr:
        result = {**result, "stderr": f"{stderr}\n{message}".strip()}
    return result


async def execute_code(code: str, language: str = "python",
                       path: str | None = None, timeout: float | None = DEFAULT_TIMEOUT,
                       *, workspace: Workspace | None = None,
                       executor: ShellExecutor | None = None) -> dict[str, Any]:
    """Write ``code`` to the workspace, execute it, return one result.

    Composes the Workspace and ShellExecutor protocols: the script is
    saved (default ``.scripts/NNNN.ext``, re-runnable later via
    ``run_command``), executed from the workspace root, and the command
    result is returned with ``script_path`` and ``language`` attached.
    """
    ws = workspace or LocalWorkspace()
    ex = executor or LocalShellExecutor()
    if language not in LANGUAGES:
        raise ValueError(f"unsupported language: {language!r} "
                         f"(choose from {sorted(LANGUAGES)})")

    script = path or await _next_script_name(ws, LANGUAGES[language])
    written = await ws.write_file(script, code)
    if "error" in written:
        return {**written, "script_path": script, "language": language}

    cwd = str(ws.root) if ws.root is not None else None
    command = _run_command_for(language, script)
    decision = _guard_decision(ex, code, language)
    if decision is not None and not decision.allowed:
        return {"stdout": "", "stderr": "", "exit_code": None,
                "duration": 0.0,
                "error": {"type": "CommandBlocked",
                          "message": decision.reason or "command blocked"},
                "script_path": script, "language": language}
    result = await ex.execute(command, cwd=cwd, timeout=timeout)
    result = _with_interpreter_hint(result, language)

    # No truncation or spilling here: oversized output is handled once at
    # the transcript boundary (agent_runtime.agent.observations), which
    # spills every tool's bulky results to .outputs/ uniformly.  Returning
    # the raw result keeps tool handlers policy-free.
    return {**result, "script_path": script, "language": language}


__all__ = [
    "DEFAULT_TIMEOUT", "LANGUAGES", "LANGUAGE_SPECS",
    "OUTPUTS_DIR", "SCRIPTS_DIR", "execute_code",
]
