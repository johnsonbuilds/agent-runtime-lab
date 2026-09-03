"""Pre-write syntax gate: reject edited content that no longer parses.

Best-effort by file extension; extensions without a checker pass (fail-open).
The gate exists so that lenient matching (the edit-match ladder) or careless
replacement text can never silently land syntactically-broken content — the
failure must surface at edit time, not at test time.
"""

from __future__ import annotations

import ast
import json
import subprocess

import yaml


def syntax_error(path: str, content: str) -> str | None:
    """Return a human-readable reason if ``content`` fails the gate for ``path``.

    ``None`` means the content passes (or the file type has no checker).
    """
    try:
        if path.endswith(".py"):
            ast.parse(content)
        elif path.endswith(".json"):
            json.loads(content)
        elif path.endswith((".yaml", ".yml")):
            yaml.safe_load(content)
        elif path.endswith((".sh", ".bash")):
            proc = subprocess.run(["bash", "-n"], input=content, text=True,
                                  capture_output=True, timeout=5)
            if proc.returncode != 0:
                detail = (proc.stderr or "").strip().splitlines()
                return detail[-1] if detail else "bash -n rejected the script"
        return None
    except SyntaxError as exc:  # TabError is a SyntaxError subclass
        return f"{type(exc).__name__}: {exc.msg} (line {exc.lineno})"
    except Exception as exc:  # malformed JSON/YAML, bash timeout, ...
        return f"{type(exc).__name__}: {str(exc)[:80]}"


__all__ = ["syntax_error"]
