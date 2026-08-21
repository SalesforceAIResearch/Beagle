"""Shared config helpers for self-evolve orchestration."""

from __future__ import annotations

from dataclasses import dataclass
import os
import subprocess
import time
from pathlib import Path

import yaml


def _probe_init_timeout_s() -> int:
    """Wall-clock cap for the cursor-agent startup init probe. The 15s default
    is too tight when the box is busy (multiple campaigns + a confirm eval all
    starting cursor-agent at once → slow init), which spuriously aborts the
    supervisor. Override with ``CURSOR_PROBE_INIT_TIMEOUT_S`` (default 45)."""
    try:
        return max(15, int(os.environ.get("CURSOR_PROBE_INIT_TIMEOUT_S", "45")))
    except (TypeError, ValueError):
        return 45


def _probe_init_attempts() -> int:
    """How many times to retry the (flaky-under-load) cursor-agent init probe
    before declaring the model unusable. Override ``CURSOR_PROBE_INIT_ATTEMPTS``
    (default 3)."""
    try:
        return max(1, int(os.environ.get("CURSOR_PROBE_INIT_ATTEMPTS", "3")))
    except (TypeError, ValueError):
        return 3


# coding-bench layout: self_evolve/run_config.py → parents[1] is the repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
# Self-evolve configs live under configs/self_evolve/ to keep their
# cursor_agent/monet/harbor schema separate from coding-bench's flat
# runner configs (model/agent/benchmark/runtime).
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "self_evolve" / "terminal_bench_2_1.yaml"  # was terminal_bench_2.yaml (renamed to _2_1); stale default broke 52 tests + any default-config code path
REQUIRED_CURSOR_CONTEXT = ""

# Env vars that, if set in the supervisor's shell, would override the model
# self-evolve picks for cursor-agent and cause the same background-task
# leak that motivated the per-call managed HOME shim. The preflight refuses
# to launch when any of them disagree with `cursor_agent.model`.
_CURSOR_MODEL_ENV_VARS = (
    "CURSOR_MODEL",
    "CURSOR_AGENT_MODEL",
    "CURSOR_DEFAULT_MODEL",
)


@dataclass(frozen=True)
class CursorModelSelection:
    configured_model: str
    selected_model: str
    actual_by_mode: dict[str, str]
    attempts: list[str]


def load_cursor_model_from_config(config_path: Path | str = DEFAULT_CONFIG_PATH) -> str:
    """Return the Cursor Agent meta-agent model from the benchmark YAML.

    `monet.model` in the same file controls the benchmarked Monet agent inside
    Harbor. This helper intentionally reads only `cursor_agent.model`.
    """
    config_path = Path(config_path)
    try:
        raw = yaml.safe_load(config_path.read_text()) or {}
    except FileNotFoundError as exc:
        raise ValueError(f"self-evolve config does not exist: {config_path}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{config_path}: expected a YAML mapping at top level")

    cursor_cfg = raw.get("cursor_agent")
    if not isinstance(cursor_cfg, dict):
        raise ValueError(f"{config_path}: cursor_agent.model is required")
    model = cursor_cfg.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"{config_path}: cursor_agent.model must be a non-empty string")
    return model.strip()


# ─── Meta-agent (proposer) backends: monet code / Claude Code ────────────
# These blocks are OPTIONAL (the proposer defaults to cursor agent, env
# META_AGENT). Loaders tolerate a missing file/block and never raise, so
# existing cursor-only configs load unchanged.
DEFAULT_CLAUDE_CODE_MODEL = "claude-opus-4-8"


def _safe_config_mapping(config_path: Path | str) -> dict:
    try:
        raw = yaml.safe_load(Path(config_path).read_text()) or {}
    except (FileNotFoundError, OSError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _meta_block(config_path: Path | str, key: str) -> dict:
    block = _safe_config_mapping(config_path).get(key)
    return block if isinstance(block, dict) else {}


def _str_or_none(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def load_meta_agent_from_config(config_path: Path | str = DEFAULT_CONFIG_PATH) -> str | None:
    """Top-level ``meta_agent:`` selector (cursor|monet_code|claude_code; monet/claude
    aliases). None when absent — the launcher then keeps the env value or the cursor
    default. Env ``META_AGENT`` takes precedence over this (see scripts/self_evolve.py).
    """
    return _str_or_none(_safe_config_mapping(config_path).get("meta_agent"))


def load_claude_code_model(config_path: Path | str = DEFAULT_CONFIG_PATH) -> str:
    """Claude Code proposer model. Optional ``claude_code.model``; default opus-4-8."""
    return _str_or_none(_meta_block(config_path, "claude_code").get("model")) \
        or DEFAULT_CLAUDE_CODE_MODEL


def load_claude_code_effort(config_path: Path | str = DEFAULT_CONFIG_PATH) -> str | None:
    """Claude Code reasoning effort. Optional ``claude_code.reasoning_effort``
    (low|medium|high|xhigh|max); None → omit ``--effort``."""
    return _str_or_none(_meta_block(config_path, "claude_code").get("reasoning_effort"))


def load_monet_code_model(config_path: Path | str = DEFAULT_CONFIG_PATH) -> str | None:
    """Monet code proposer model. Optional ``monet_code.model``; None → falls back
    to ``cursor_agent.model`` (the campaign's single model source of truth)."""
    return _str_or_none(_meta_block(config_path, "monet_code").get("model"))


def load_monet_code_effort(config_path: Path | str = DEFAULT_CONFIG_PATH) -> str | None:
    """Monet code reasoning effort. Optional ``monet_code.reasoning_effort``
    (none|low|medium|high|max); None → omit ``--effort`` (monet defaults to none)."""
    return _str_or_none(_meta_block(config_path, "monet_code").get("reasoning_effort"))


def _available_cursor_models() -> dict[str, str]:
    try:
        proc = subprocess.run(
            ["cursor-agent", "models"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {}
    if proc.returncode != 0:
        return {}
    stdout = getattr(proc, "stdout", "")
    if not isinstance(stdout, str):
        return {}
    return _parse_cursor_models(stdout)


def _parse_cursor_models(output: str) -> dict[str, str]:
    models: dict[str, str] = {}
    for raw in output.splitlines():
        line = raw.strip()
        if not line or line.startswith("Available models") or line.startswith("Tip:"):
            continue
        if " - " not in line:
            continue
        slug, display = line.split(" - ", 1)
        slug = slug.strip()
        display = display.strip()
        if slug and display:
            models[slug] = display
    return models


def validate_cursor_model_runtime(
    model: str,
    *,
    workspace: Path | str = REPO_ROOT,
    required_context: str = REQUIRED_CURSOR_CONTEXT,
) -> dict[str, str]:
    """Fail unless the configured `--model` can initialize.

    The YAML slug is the source of truth — passed verbatim to
    `cursor-agent --model`. We probe it for real instead of trusting
    `cursor-agent models` output: bare family slugs like `gpt-5.5` aren't
    in that listing but ARE valid `--model` arguments (cursor-agent maps
    them to the user's account-level reasoning/context preference).

    A listed match is used only for `required_context` enforcement when
    self-evolve is configured to demand a specific context length.
    """
    model = model.strip()
    advertised = _available_cursor_models()
    listed_display = advertised.get(model) if advertised else None
    if listed_display and required_context and required_context not in listed_display:
        raise ValueError(
            f"cursor_agent.model {model!r} is listed as {listed_display!r}, "
            f"not a {required_context} context model"
        )

    from . import cursor_agent

    workspace = Path(workspace)
    actual_by_mode: dict[str, str] = {}
    _timeout = _probe_init_timeout_s()
    _attempts = _probe_init_attempts()
    for mode_name, plan_mode in (("agent", False), ("plan", True)):
        actual = None
        last_exc: Exception | None = None
        for _try in range(1, _attempts + 1):
            try:
                actual = cursor_agent.probe_init_model(
                    model,
                    workspace=workspace,
                    plan_mode=plan_mode,
                    timeout_s=_timeout,
                )
                break
            except RuntimeError as exc:
                last_exc = exc
                if _try < _attempts:
                    time.sleep(5)
        if actual is None:
            raise ValueError(
                f"cursor_agent.model {model!r} runtime probe failed in "
                f"{mode_name} mode after {_attempts} attempt(s) "
                f"(timeout={_timeout}s each): {last_exc}"
            ) from last_exc
        actual_by_mode[mode_name] = actual
        if required_context and required_context not in actual:
            advertised_part = (
                f" advertised by `cursor-agent models` as {listed_display!r},"
                if listed_display
                else ""
            )
            raise ValueError(
                f"cursor_agent.model {model!r}{advertised_part} but actual "
                f"{mode_name} mode initialized {actual!r}; expected "
                f"{required_context} context"
            )
    return actual_by_mode


def validate_no_monet_env_leak(model: str) -> None:
    """Refuse to launch when CURSOR_*_MODEL env vars override the chosen slug.

    A stray `CURSOR_MODEL` / `CURSOR_AGENT_MODEL` / `CURSOR_DEFAULT_MODEL`
    in the supervisor's shell can override the foreground `--model` flag for
    cursor-agent's background tasks the same way the cli-config can. Fail
    fast with the offending var named so the operator unsets it.
    """
    for var in _CURSOR_MODEL_ENV_VARS:
        value = os.environ.get(var)
        if value is None or not value.strip():
            continue
        if value != model:
            raise ValueError(
                f"environment variable {var}={value!r} would override "
                f"cursor_agent.model={model!r} for cursor-agent. Unset it "
                f"before launching self-evolve, or set it to {model!r}."
            )


def select_cursor_model_runtime(
    model: str,
    *,
    workspace: Path | str = REPO_ROOT,
    required_context: str = REQUIRED_CURSOR_CONTEXT,
) -> CursorModelSelection:
    """Validate and return the configured Cursor model.

    `cursor_agent.model` in the benchmark YAML is authoritative. If that slug
    is missing or fails to probe, fail fast instead of silently substituting a
    different model.
    """
    configured = model.strip()
    validate_no_monet_env_leak(configured)
    actual_by_mode = validate_cursor_model_runtime(
        configured,
        workspace=workspace,
        required_context=required_context,
    )
    return CursorModelSelection(
        configured_model=configured,
        selected_model=configured,
        actual_by_mode=actual_by_mode,
        attempts=[],
    )
