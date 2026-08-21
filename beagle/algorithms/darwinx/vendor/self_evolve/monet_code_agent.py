"""Monet code as a proposer backend (meta-agent) — a drop-in alternative to ``cursor_agent``.

Runs the **monet code** CLI headless ON THE LAUNCH BOX (node via nvm + a pre-built monet
clone) against the per-node worktree, parses monet's ``--output-format json`` result,
and returns a :class:`cursor_agent.CursorResult` so it can replace ``cursor_agent.run``.

Activation: env ``META_AGENT=monet_code`` (``monet`` accepted as an alias; default
``cursor`` → this module is inert). Selection + the ``model`` / ``reasoning_effort``
passed in are resolved upstream by ``PipelineConfig`` / ``meta_agent`` — see
``meta_agent.py``. Reasoning effort (optional) maps to monet's ``--effort``
(``none|low|medium|high|max``).

Prereq (one-time): an isolated, built monet clone whose ``bin/monet.js`` is a STABLE
proposer — built from ``monet_code/`` at ``$MONET_META_BIN_DIR`` (default
``~/projects/monet_meta_build``). Verified box-smoke output schema:
``{"type":"result","text":"…"}``. Gateway access is native via ``--provider
llm-gateway-express-local-proxy`` (the same reverse-tunnel the eval already uses),
so — unlike Claude Code — no Anthropic→OpenAI translator is required.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import time
from pathlib import Path

from .cursor_agent import CursorResult, DEFAULT_MODEL, DEFAULT_TIMEOUT_S

_DEFAULT_BIN_DIR = os.path.expanduser("~/projects/MonetEvolve/monet_meta_build")
_PROVIDER = os.environ.get("MONET_META_PROVIDER", "llm-gateway-express-local-proxy")


def meta_agent_is_monet() -> bool:
    return os.environ.get("META_AGENT", "cursor").strip().lower() in ("monet", "monet_code")


def _node_bin() -> str | None:
    """Full path to an nvm-installed node>=20 (subprocess PATH usually lacks it)."""
    cands = sorted(glob.glob(os.path.expanduser("~/.nvm/versions/node/v*/bin/node")), reverse=True)
    for c in cands:
        # prefer v20+ (monet engines: node>=20)
        try:
            major = int(os.path.basename(os.path.dirname(os.path.dirname(c)))[1:].split(".")[0])
            if major >= 20:
                return c
        except Exception:
            continue
    return cands[0] if cands else None


def _monet_js() -> str:
    d = os.environ.get("MONET_META_BIN_DIR", _DEFAULT_BIN_DIR)
    return os.path.join(d, "bin", "monet.js")


def _extract_result_text(stdout: str) -> tuple[str, str | None]:
    """Parse monet's json/stream-json output for the final result text.
    Schema (box-verified): a line ``{"type":"result","text":"…"}``."""
    text, err = "", None
    for line in reversed((stdout or "").splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if obj.get("type") == "result":
            text = str(obj.get("text") or obj.get("result") or "")
            if obj.get("is_error") or obj.get("subtype") == "error":
                err = "monet reported is_error"
            return text, err
    return text, err


def run(
    prompt: str,
    *,
    workspace: Path,
    log_path: Path,
    model: str = DEFAULT_MODEL,
    plan_mode: bool = False,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    reasoning_effort: str | None = None,
    extra_args: list[str] | None = None,
    max_turns: int | None = None,
    **_ignored,
) -> CursorResult:
    """Invoke monet code headless on the box; return a CursorResult (drop-in)."""
    node = _node_bin()
    monet = _monet_js()
    if node is None or not os.path.exists(monet):
        return CursorResult(
            text="", exit_code=127, raw_log_path=Path(log_path),
            error=f"monet meta-agent not built (node={node}, bin={monet}); run build_monet_box.sh",
        )
    workspace = Path(workspace)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    mt = int(max_turns or os.environ.get("MONET_META_MAX_TURNS", "60") or 60)
    safe_prompt = (prompt or "").replace("\x00", "")  # NUL crashes exec (same guard as cursor_agent)

    cmd = [
        node, monet, "-p", safe_prompt, "--cwd", str(workspace), "--model", model,
        "--provider", _PROVIDER, "--output-format", "json",
        "--max-turns", str(mt), "--strict-max-turns", "--all-permissions",
    ]
    # Reasoning effort -> monet's --effort (none|low|medium|high|max). monet's own
    # default is 'none', so we only pass the flag when an effort is configured.
    eff = (reasoning_effort or "").strip().lower()
    if eff:
        cmd += ["--effort", eff]
    if extra_args:
        cmd += list(extra_args)

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return CursorResult(text="", exit_code=124, raw_log_path=log_path,
                            duration_ms=int((time.time() - t0) * 1000),
                            error=f"monet meta-agent timed out after {timeout_s}s")
    except Exception as e:  # noqa: BLE001
        return CursorResult(text="", exit_code=1, raw_log_path=log_path,
                            error=f"failed to spawn monet meta-agent: {e}")
    dur = int((time.time() - t0) * 1000)
    try:
        with open(log_path, "w") as lf:
            lf.write(proc.stdout or "")
            lf.write("\n---STDERR---\n")
            lf.write(proc.stderr or "")
    except Exception:
        pass

    text, err = _extract_result_text(proc.stdout or "")

    # analyze/review/select stages have no monet read-only mode -> discard any edits
    # they made so only the implement stage produces the candidate diff. The agent
    # edits the `monet_code` submodule worktree (every prompt says edit
    # `{{ wt_dir }}/monet_code/`), and `workspace` is the eval superproject; a
    # superproject `git checkout/clean` does NOT reach into a submodule's working
    # tree, so we must reset the submodule worktree directly. Fall back to
    # `workspace` only if there's no nested `monet_code/` (e.g. a caller that
    # already points at the repo).
    if plan_mode:
        target = workspace / "monet_code" if (workspace / "monet_code").is_dir() else workspace
        for args in (["checkout", "--", "."], ["clean", "-fd"]):
            try:
                proc_c = subprocess.run(
                    ["git", "-C", str(target), *args], capture_output=True, timeout=30,
                )
            except Exception as e:  # noqa: BLE001
                # A read-only stage that can't revert its edits must NOT report
                # success — otherwise the leaked edits ride into the candidate diff.
                err = err or f"monet plan-mode cleanup failed ({' '.join(args)}): {e}"
                continue
            if proc_c.returncode != 0:
                err = err or (
                    f"monet plan-mode cleanup failed ({' '.join(args)}): "
                    f"{(proc_c.stderr or b'').decode(errors='ignore')[:200]}"
                )

    if proc.returncode != 0 and not text:
        err = err or f"monet meta-agent exited {proc.returncode}: {(proc.stderr or '')[:200]}"
    return CursorResult(text=text, exit_code=proc.returncode, error=err,
                        raw_log_path=log_path, duration_ms=dur)


__all__ = ["run", "meta_agent_is_monet"]
