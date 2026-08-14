"""Execution environment abstractions."""

from .base import ShellExecutor
from .local import LocalShellExecutor

__all__ = ["ShellExecutor", "LocalShellExecutor"]
