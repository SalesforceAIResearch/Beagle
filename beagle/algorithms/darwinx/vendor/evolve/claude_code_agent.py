"""Claude Code as a proposer backend (meta-agent) — a drop-in alternative to
``cursor_agent`` / ``monet_code_agent``.

Runs the **Claude Code** CLI headless (``claude -p … --output-format json``)
against the per-node eval worktree and returns a
:class:`cursor_agent.CursorResult`, so it can replace ``cursor_agent.run``.

Activation: env ``META_AGENT=claude_code`` (default ``cursor`` → this module is
inert). Model defaults to ``claude-opus-4-8``; override via ``claude_code.model``
in the campaign YAML. Reasoning effort (optional) maps to ``--effort``
(``low|medium|high|xhigh|max``).

Gateway path: this box has **no direct egress** to the Salesforce LLM Gateway
Express, and Claude Code speaks only the Anthropic ``/v1/messages`` wire format.
Reaching the gateway therefore requires the **cc_setup** translator+tunnel
(Anthropic→OpenAI) with ``ANTHROPIC_BASE_URL`` pointed at it. ``run`` preflights
that translator and fails fast with actionable guidance if it is not up — it
never silently falls back to a different endpoint.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

from .cursor_agent import CursorResult, DEFAULT_TIMEOUT_S

DEFAULT_CLAUDE_MODEL = "claude-opus-4-8"
_EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
_PROXY_HELP = (
    "On your laptop run `cc_setup/run_claude_code_proxy.sh --remote <node> "
    "--api-key <key>` and keep it running, then export "
    "ANTHROPIC_BASE_URL=http://127.0.0.1:<remote-port> for the campaign."
)


def meta_agent_is_claude_code() -> bool:
    return os.environ.get("META_AGENT", "cursor").strip().lower() in ("claude", "claude_code")


def _claude_bin() -> str | None:
    found = shutil.which("claude")
    if found:
        return found
    fallback = os.path.expanduser("~/.local/bin/claude")
    return fallback if os.path.exists(fallback) else None


def _base_url() -> str:
    """The cc_setup translator URL Claude Code should hit (root, no path)."""
    return (
        os.environ.get("CLAUDE_META_BASE_URL")
        or os.environ.get("ANTHROPIC_BASE_URL")
        or ""
    ).strip().rstrip("/")


def _preflight_proxy(timeout_s: float = 4.0) -> str | None:
    """Return an actionable error string if the cc_setup translator is unreachable, else None.

    The translator serves ``/health`` (see cc_setup/run_claude_code_proxy.sh). A
    missing ``ANTHROPIC_BASE_URL`` or an unreachable translator both yield a
    message that tells the operator exactly how to bring it up.
    """
    base = _base_url()
    if not base:
        return (
            "Claude Code proposer: ANTHROPIC_BASE_URL is not set — no path to the "
            "Salesforce LLM gateway. " + _PROXY_HELP
        )
    try:
        with urllib.request.urlopen(f"{base}/health", timeout=timeout_s) as resp:
            if 200 <= int(getattr(resp, "status", 200) or 200) < 500:
                return None
    except Exception as exc:  # noqa: BLE001
        return (
            f"Claude Code proposer: cc_setup translator at {base} is unreachable "
            f"({type(exc).__name__}: {exc}). " + _PROXY_HELP
        )
    return None


def _child_env() -> dict[str, str]:
    """Env for the spawned ``claude`` so it runs as a fresh top-level session.

    Scrub the parent's Claude Code harness/session vars (``CLAUDECODE``,
    ``CLAUDE_CODE_*``, ``CLAUDE_EFFORT``): inherited, they make the child think
    it is a resumed/child session and ``CLAUDE_EFFORT`` would override our explicit
    ``--effort``.

    Config dir + proxy creds are taken purely from the environment — ``run`` loads
    the project-root ``.env`` first (see :func:`run`), so an operator selects them
    per-box without touching shared code: set ``CLAUDE_CONFIG_DIR`` in ``.env`` to
    isolate the proposer's Claude Code from the interactive ``~/.claude``
    (subscription); leave it unset and Claude uses the default ``~/.claude``.
    ``ANTHROPIC_BASE_URL`` / ``ANTHROPIC_AUTH_TOKEN`` (the cc_setup proxy) likewise
    come from ``.env`` or the shell and are inherited verbatim — no hardcoded path
    or forced override here (that would bake one box's setup into team infra).

    NOTE: a settings.json ``env`` block OVERRIDES process env in Claude Code, and an
    empty ``ANTHROPIC_AUTH_TOKEN`` there nulls auth → 401. Pointing ``CLAUDE_CONFIG_DIR``
    at a proxy-configured dir (whose settings.json carries the proxy env) avoids that;
    the ``ANTHROPIC_API_KEY`` placeholder below only covers a fully-unset token.
    """
    env = dict(os.environ)
    for key in list(env):
        if key == "CLAUDECODE" or key.startswith("CLAUDE_CODE") or key == "CLAUDE_EFFORT":
            env.pop(key, None)
    env.setdefault("ANTHROPIC_API_KEY", env.get("ANTHROPIC_AUTH_TOKEN", "cc-setup-proxy"))
    return env


def _extract_result_text(stdout: str) -> tuple[str, str | None]:
    """Parse ``claude -p --output-format json`` output.

    Schema: a single object ``{"type":"result","subtype":"success",
    "is_error":false,"result":"…"}``. We try the whole stdout first (the format
    emits one object), then fall back to a reverse per-line scan for robustness.
    """
    blob = (stdout or "").strip()
    candidates: list[object] = []
    try:
        candidates.append(json.loads(blob))
    except Exception:  # noqa: BLE001
        for line in reversed(blob.splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    candidates.append(json.loads(line))
                except Exception:  # noqa: BLE001
                    continue
    for obj in candidates:
        if isinstance(obj, dict) and obj.get("type") == "result":
            text = str(obj.get("result") or obj.get("text") or "")
            err = (
                "claude reported is_error"
                if obj.get("is_error") or obj.get("subtype") == "error"
                else None
            )
            return text, err
    return "", None


def run(
    prompt: str,
    *,
    workspace: Path,
    log_path: Path,
    model: str = DEFAULT_CLAUDE_MODEL,
    plan_mode: bool = False,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    reasoning_effort: str | None = None,
    extra_args: list[str] | None = None,
    **_ignored,
) -> CursorResult:
    """Invoke Claude Code headless on the box; return a ``CursorResult`` (drop-in).

    ``plan_mode`` uses Claude Code's native read-only ``--permission-mode plan``
    (analog of cursor's ``--mode plan``); non-plan stages run with
    ``--dangerously-skip-permissions`` so edits/commands proceed unattended in the
    isolated eval worktree.
    """
    # Pick up CLAUDE_CONFIG_DIR / ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN (and any
    # other knobs) from the project-root .env so direct/smoke calls behave like the
    # supervisor (which already loads it). Idempotent; host env wins over .env.
    from . import dotenv_loader
    dotenv_loader.ensure_loaded()
    claude = _claude_bin()
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if claude is None:
        return CursorResult(
            text="", exit_code=127, raw_log_path=log_path,
            error="Claude Code not found (no `claude` on PATH or ~/.local/bin/claude)",
        )
    pre = _preflight_proxy()
    if pre is not None:
        return CursorResult(text="", exit_code=127, raw_log_path=log_path, error=pre)

    workspace = Path(workspace)
    safe_prompt = (prompt or "").replace("\x00", "")  # NUL crashes exec (same guard as cursor_agent)

    cmd = [claude, "-p", safe_prompt, "--output-format", "json",
           "--model", model or DEFAULT_CLAUDE_MODEL]
    if plan_mode:
        cmd += ["--permission-mode", "plan"]
    else:
        cmd += ["--dangerously-skip-permissions"]
    eff = (reasoning_effort or "").strip().lower()
    if eff and eff != "none":
        if eff not in _EFFORT_LEVELS:
            return CursorResult(
                text="", exit_code=2, raw_log_path=log_path,
                error=f"claude_code.reasoning_effort {eff!r} invalid; choose {'|'.join(_EFFORT_LEVELS)}",
            )
        cmd += ["--effort", eff]
    if extra_args:
        cmd += list(extra_args)

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, cwd=str(workspace), env=_child_env(),
            capture_output=True, text=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return CursorResult(
            text="", exit_code=124, raw_log_path=log_path,
            duration_ms=int((time.time() - t0) * 1000),
            error=f"Claude Code proposer timed out after {timeout_s}s",
        )
    except Exception as exc:  # noqa: BLE001
        return CursorResult(text="", exit_code=1, raw_log_path=log_path,
                            error=f"failed to spawn Claude Code: {exc}")
    dur = int((time.time() - t0) * 1000)
    try:
        with open(log_path, "w") as lf:
            lf.write(proc.stdout or "")
            lf.write("\n---STDERR---\n")
            lf.write(proc.stderr or "")
    except Exception:  # noqa: BLE001
        pass

    text, err = _extract_result_text(proc.stdout or "")
    if proc.returncode != 0 and not text:
        err = err or f"Claude Code exited {proc.returncode}: {(proc.stderr or '')[:200]}"
    return CursorResult(text=text, exit_code=proc.returncode, error=err,
                        raw_log_path=log_path, duration_ms=dur)


__all__ = ["run", "meta_agent_is_claude_code", "DEFAULT_CLAUDE_MODEL"]
