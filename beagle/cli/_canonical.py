"""Canonical ``config.yaml`` loader — the self-contained, role-based config that drives both
``beagle evolve`` and ``beagle evaluate`` (and the ``examples/``).

One shape for both modes: a role block (``evolvee`` / ``evolver`` / ``agent``) carries a nested
``harness: {name, version, source}`` (the harness/adapter type + version + INLINE source), a
``model``, and agent-level knobs (``provider`` / ``forward_env`` / ``effort`` / ``max_turns`` /
``timeout`` / ``extra_args``); ``data`` is a list of ``{benchmark, tasks, …}``. This module translates that shape
into the framework's typed :class:`~beagle.config.BeagleConfig` (evolution) or
:class:`~beagle.config.RunConfig` (evaluation) — pydantic validates every field.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import yaml

import beagle as bgl
from beagle.config import AgentConfig, BeagleConfig, RunConfig

#: **First-level** knobs — the shared agent vocabulary, spelled at the top of EVERY agent's role
#: block (uniform across agents; each adapter maps them to its mechanism).
_GENERIC_KNOBS = ("provider", "effort", "max_turns", "max_tokens", "timeout")

#: Agent-harness-specific knobs that USED to sit flat on the role block. Still accepted for
#: backward-compat, but the canonical home is the agent's ``extra_args: {<agent>_args: …}`` block.
_LEGACY_ADAPTER_KNOBS = ("config_path", "monet_args", "container_path", "install_cmd", "output_dir")


def _fold_extra_args(config: dict, ea: object, agent_name: str) -> None:
    """Fold the agent's own ``extra_args:`` block into its flat ``config``.

    ``extra_args:`` is keyed by ``<agent>_args`` (``monet_args`` / ``mini_swe_args``), so a config
    names which knobs belong to which agent. For the agent being built we take *its* block and fold:
    a **list of raw CLI flags** is stored as ``<agent>_args`` (what e.g. monet reads); a **map** of
    named knobs — or a list of single-key maps — is flattened into ``config`` (e.g. mini-swe's
    ``config_path``). A bare ``extra_args`` *list* is back-compat shorthand for monet's argv."""
    if ea is None:
        return
    if isinstance(ea, list):                                  # legacy shorthand: bare monet argv
        config["monet_args"] = list(ea)
        return
    if not isinstance(ea, dict):
        return
    own = ea.get(f"{agent_name.replace('-', '_')}_args")      # mini-swe → mini_swe_args
    if own is None:
        return
    if isinstance(own, dict):                                 # named knobs → flatten
        config.update({k: v for k, v in own.items() if v is not None})
    elif isinstance(own, list) and all(isinstance(x, dict) for x in own):  # list of {knob: val}
        for d in own:
            config.update({k: v for k, v in d.items() if v is not None})
    elif isinstance(own, list):                               # raw CLI argv
        config[f"{agent_name.replace('-', '_')}_args"] = list(own)


def load(path: str | Path) -> dict:
    """Parse a canonical ``config.yaml`` into a plain dict."""
    return yaml.safe_load(Path(path).read_text()) or {}


def _adapter_for(name: str) -> str:
    """Resolve an agent type (e.g. ``cursor-agent``) to the registered adapter that runs it
    (``cursor``): exact match wins, else the longest registered name that prefixes it."""
    names = set(bgl.agents.available())
    if name in names:
        return name
    prefixes = sorted((n for n in names if name.startswith(n)), key=len, reverse=True)
    return prefixes[0] if prefixes else name   # unknown → let build() raise a clear error


def _harness(role: dict) -> dict:
    """The role's nested ``harness`` block (``{name, version, source}``). Renamed from ``agent:`` —
    a config still using the old key gets a clear migration error, not a cryptic KeyError."""
    h = role.get("harness")
    if h is None and role.get("agent") is not None:
        raise ValueError(
            "the nested `agent:` block was renamed to `harness:` — write "
            "`harness: {name, version, source}` under the role block (evolvee/evolver/agent).")
    return h or {}


def agent_dict(role: dict) -> dict:
    """A role block → a declarative :class:`AgentConfig`-shaped dict.

    Canonical shape: the nested ``harness`` block carries the adapter type/version + inline
    ``source``. **First-level** (top of the role block, uniform across EVERY agent) are ``model`` /
    ``forward_env`` plus the shared vocabulary ``provider`` / ``effort`` / ``max_turns`` /
    ``max_tokens`` / ``timeout``. An agent-harness's *own* args live under ``extra_args:``, keyed by
    ``<agent>_args`` (``monet_args`` / ``mini_swe_args``) — so a config names which knobs belong to
    which agent. All fold into the agent's freeform ``config``; knobs spelled flat at the top level
    still fold (backward-compat)."""
    a = _harness(role)
    src = a.get("source") or {}
    config: dict = {}
    if role.get("forward_env"):
        config["forward_env"] = list(role["forward_env"])
    for k in _GENERIC_KNOBS:                                  # first-level vocabulary, every agent
        if role.get(k) is not None:
            config[k] = role[k]

    _fold_extra_args(config, role.get("extra_args"), a.get("name", ""))

    # Backward-compat: adapter knobs spelled flat at the top level still fold (extra_args wins).
    for k in _LEGACY_ADAPTER_KNOBS:
        if role.get(k) is not None and k not in config:
            config[k] = list(role[k]) if k == "monet_args" else role[k]

    if role.get("prompt_override"):
        # Escape hatch (eval/ablation): {system?, instruction?} replaces the agent's own layer-1/2
        # framing. Best-effort — only config-driven adapters apply it. See notes/task-prompt-injection.md.
        config["prompt_override"] = dict(role["prompt_override"])
    if src.get("token_env"):
        config["token_env"] = src["token_env"]
    d: dict = {"name": _adapter_for(a["name"]), "model": role["model"], "config": config}
    if src.get("repo"):
        d["source"] = {"repo": src["repo"], "ref": src.get("ref")}
    return d


def benchmark_dict(g: dict) -> dict:
    """A ``data`` group → :class:`BenchmarkConfig` fields: name + task_ids + per-benchmark knobs.
    OMIT ``tasks`` → ``task_ids`` stays unset (``None`` = the whole suite); a list restricts+orders."""
    d: dict = {"name": g["benchmark"]}
    if g.get("tasks") is not None:
        d["task_ids"] = list(g["tasks"])
    for k in ("dataset", "split", "num_samples", "exclude_task_ids",
              "namespace", "tag", "registry", "image", "options"):
        if g.get(k) is not None:
            d[k] = g[k]
    return d


def check_versions(raw: dict, roles: tuple[str, ...] = ("evolvee", "evolver")) -> None:
    """Version gate — fail loud (``SystemExit``) if a role pins ``agent.version`` that mismatches
    the agent's INSTALLED version. Source-versioned agents (monet, pinned by ``source.ref``) report
    no installed version and are exempt. Runs before any spend."""
    for role in roles:
        block = raw.get(role) or {}
        hb = _harness(block)
        name = hb.get("name")
        declared = hb.get("version")
        if declared is None:
            continue
        agent = bgl.agents.build(AgentConfig.model_validate(agent_dict(block)))
        installed = agent.installed_version()
        if installed is None:
            print(f"[version-gate] {role} {name}: no installed version to check (exempt) — pinned {declared}")
            continue
        if str(declared) != str(installed):
            raise SystemExit(
                f"[version-gate] {role} agent {name!r}: config pins version {declared!r} but the "
                f"installed version is {installed!r} — refusing to run. Update agent.version (or the "
                f"installed binary) so they match.")
        print(f"[version-gate] {role} {name} version {installed} ✓")


def _now_stamp() -> str:
    """A filesystem-safe ``YYYYmmdd-HHMMSS`` stamp for run-dir uniqueness (wrapped so tests pin it)."""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _run_dir(raw: dict, *, default_timestamp: bool) -> tuple[Path, str]:
    """Resolve the run's ``<dir>/<name>``.

    When timestamping is on (``run.timestamp``, defaulting to ``default_timestamp``), append a
    ``-<YYYYmmdd-HHMMSS>`` stamp so every run lands in a FRESH dir — no "job dir already exists /
    can't resume with a different config" collision while you iterate. Set ``run.timestamp: false``
    to pin a stable, resumable dir (the default for ``evolve``, whose campaign dir must persist)."""
    run = raw.get("run") or {}
    name = run.get("name") or "run"
    if run.get("timestamp", default_timestamp):
        name = f"{name}-{_now_stamp()}"
    return Path(run.get("dir", "./tmp")) / name, name


def build_evolution(raw: dict) -> tuple[BeagleConfig, Path, str]:
    """Canonical config → (:class:`BeagleConfig`, run_dir, run_name) for ``beagle evolve``.
    Injects the launch paths (repo_root/campaign/evolvee_checkout) + the evolvee effort into the
    algorithm hparams (evolution trains on ``data[0]`` — DarwinX is single-benchmark)."""
    run_dir, run_name = _run_dir(raw, default_timestamp=False)  # campaign dir must persist (resume)
    data = raw.get("data")
    if not data:
        raise ValueError("evolution config has no `data` — the benchmark + tasks to evolve on")
    hp = dict((raw.get("algorithm") or {}).get("hparams") or {})
    hp.update(repo_root=str(run_dir), campaign=run_name)
    checkout = _harness(raw.get("evolvee") or {}).get("source", {}).get("dir")
    if checkout:
        hp["evolvee_checkout"] = str(Path(checkout).resolve())     # relative to CWD
    if raw["evolvee"].get("effort"):
        hp["evolvee_effort"] = raw["evolvee"]["effort"]
    cfg = BeagleConfig.from_dict({
        "evolvee": agent_dict(raw["evolvee"]),
        "evolver": agent_dict(raw["evolver"]),
        "benchmark": benchmark_dict(data[0]),
        "runtime": {"kind": (raw.get("run") or {}).get("runtime", "xrlenv-cluster")},
        "parallelism": (raw.get("run") or {}).get("parallelism", 1),
        "algorithm": {"name": (raw.get("algorithm") or {}).get("name", "darwinx"), "hparams": hp},
    })
    return cfg, run_dir, run_name


def build_evaluation(raw: dict) -> tuple[RunConfig, Path]:
    """Canonical config → (:class:`RunConfig`, run_dir) for ``beagle evaluate``. Evaluates the
    ``agent`` block on ``data[0]`` (no evolver/algorithm)."""
    run_dir, _ = _run_dir(raw, default_timestamp=True)  # fresh dir per eval; run.timestamp:false pins
    run = raw.get("run") or {}
    cfg = RunConfig.from_dict({
        "model": raw["agent"]["model"],
        "agent": agent_dict(raw["agent"]),
        "benchmark": benchmark_dict(raw["data"][0]),
        "runtime": {"kind": run.get("runtime", "xrlenv-cluster")},
        "parallelism": run.get("parallelism", 1),
        "parallelism_eval_patches": run.get("parallelism_eval_patches"),   # None → falls back to parallelism
        "retry": run.get("retry", {}),      # {infra, content, timeout_multiplier}; default = no retry
    })
    return cfg, run_dir
