"""Shared subprocess primitive for coding-agent editors.

An :class:`~beagle.agents.core.base.Editor` drives a headless coding CLI (cursor-agent,
claude, monet.js …) as a subprocess. The lifecycle is identical across backends — spawn,
feed the prompt on **stdin** (not argv, to dodge the ~128 KB argv cap), enforce a timeout by
killing the whole **process group** (so tool-spawned children die too), and report a plain
outcome. Only the argv construction + stream parsing differ, and those live in each editor's
``edit()``. This module owns the lifecycle so every editor gets the same resilient behavior.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class CliOutcome:
    """The result of one CLI invocation — lifecycle only, no backend parsing."""

    exit_code: int              # 0 ok · 124 timeout · 127 binary-not-found · else child exit code
    stdout: str = ""            # full captured stdout (+stderr merged) — the CLI's stream
    duration_ms: int = 0
    error: str | None = None    # a short message on any non-zero exit; None on success
    log_path: Path | None = None


def _kill_group(proc: subprocess.Popen[Any]) -> None:
    """SIGKILL the child's whole process group (kills descendants that inherit stdout)."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            proc.kill()
        except ProcessLookupError:
            pass


def run_cli(
    argv: list[str],
    *,
    prompt: str,
    cwd: str | Path,
    timeout_s: float,
    log_path: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> CliOutcome:
    """Run ``argv`` with ``prompt`` on stdin in ``cwd``; return a :class:`CliOutcome`.

    Never raises for the ordinary failure modes — a missing binary → exit 127, a timeout →
    exit 124 (after SIGKILLing the group), a spawn error → exit 1 — so the caller branches on
    ``exit_code``/``error`` instead of catching. NUL bytes in the prompt (fatal to fork/exec
    if they ever reach argv) are stripped defensively.
    """
    t0 = time.monotonic()
    prompt = (prompt or "").replace("\x00", "")

    def _elapsed() -> int:
        return int((time.monotonic() - t0) * 1000)

    try:
        proc = subprocess.Popen(
            [str(a) for a in argv], cwd=str(cwd),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, start_new_session=True, env=env,
        )
    except FileNotFoundError:
        return CliOutcome(exit_code=127, duration_ms=_elapsed(), error=f"binary not found: {argv[0]}")
    except OSError as e:  # noqa: BLE001 — surface any spawn failure as an outcome, don't raise
        return CliOutcome(exit_code=1, duration_ms=_elapsed(), error=f"spawn failed: {e}")

    try:
        out, _ = proc.communicate(input=prompt, timeout=timeout_s)
        exit_code = proc.returncode
        error = None if exit_code == 0 else f"exit {exit_code}"
    except subprocess.TimeoutExpired:
        _kill_group(proc)
        out, _ = proc.communicate()          # drain whatever was buffered before the kill
        exit_code, error = 124, f"timeout after {timeout_s}s"

    out = out or ""
    path: Path | None = None
    if log_path is not None:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(out)
    return CliOutcome(exit_code=exit_code, stdout=out, duration_ms=_elapsed(), error=error, log_path=path)


__all__ = ["CliOutcome", "run_cli"]
