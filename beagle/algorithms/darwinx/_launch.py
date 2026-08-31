"""Launch routing for the vendored DarwinX driver — the part of the drop-in that is beagle's
job ("we do the routing", ``notes/darwinx-dropin-contract.md``). It translates one beagle
evolution run into exactly what the vendored driver expects, doing **no** paid work:

* :func:`prepare_import_path` — make the vendored package and the seam-A ``runner.run`` shim
  importable (so ``import evolve…`` and the eval subprocess resolve to us).
* :func:`emit_campaign_config` — write the driver's campaign config (its own YAML schema) from
  the evolvee / evolver / benchmark. This is seam C; it also stashes the evolver as an
  ``AgentConfig`` block so a worker subprocess can rebuild the Editor with **no env var**.
* :func:`build_pipeline_config` — construct the vendored ``PipelineConfig`` from that config plus
  operator hyper-parameters (drift-proof: only fields the dataclass actually declares are passed).
* :func:`read_best` — read the best evolved node back out of the run's genealogy DB → a
  :class:`~beagle.algorithms.base.Candidate`.

``DarwinX.evolve`` calls these in order, injects the evolver Editor (seam B), then runs the
pipeline. The config keys here (``cursor_agent`` / ``monet`` / ``benchmark`` / ``runtime``) are
the hosted driver's own schema — adapter-context naming, produced verbatim so the vendored
readers parse it.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from beagle.algorithms.base import Candidate, CandidateStatus

if TYPE_CHECKING:
    from beagle.agents.core.base import Agent, AgentSource
    from beagle.config import RunConfig

_PKG = Path(__file__).resolve().parent          # …/beagle/algorithms/darwinx
_VENDOR = _PKG / "vendor"
_SHIMS = _PKG / "_shims"

#: Config key the campaign YAML carries the evolver spec under, so a worker subprocess can
#: reconstruct the Editor via ``meta_agent.set_editor_from_spec`` (seam C, no env var).
EVOLVER_SPEC_KEY = "beagle_evolver"


def prepare_import_path() -> None:
    """Make the vendored driver + the seam-A ``runner.run`` shim resolvable (idempotent).

    Two channels, because the driver shells the eval as a subprocess:

    * ``sys.path`` — so *this* process's ``import evolve…`` and ``runner.run`` resolve;
      ``_shims`` precedes ``vendor`` so our ``runner.run`` shadows any other.
    * ``PYTHONPATH`` — so the vendored eval's ``python -m runner.run`` **subprocess** resolves
      our shim too. A subprocess inherits the environment, not ``sys.path``; ``PYTHONPATH`` is
      Python's standard module-resolution channel (not a run-config knob), and the only way a
      ``-m`` subprocess can find the shim.

    A plain ``python -m runner.run`` outside a launch is unaffected — this runs only here.
    """
    import os

    for p in (str(_VENDOR), str(_SHIMS)):   # insert(0) reverses order → _shims ends up first
        if p not in sys.path:
            sys.path.insert(0, p)
    ordered = [str(_SHIMS), str(_VENDOR)]   # _shims first here too
    have = os.environ.get("PYTHONPATH", "").split(os.pathsep) if os.environ.get("PYTHONPATH") else []
    missing = [p for p in ordered if p not in have]
    if missing:
        os.environ["PYTHONPATH"] = os.pathsep.join(missing + have)


# --- evolvee checkout + worktree paths ---------------------------------------

#: Subdir under ``repo_root`` where the vendored worktree carves per-pipeline checkouts of the
#: evolvee (mirrors ``worktree.py``: ``canonical = repo_root / "monet_code"``). The proposer edits
#: a private ``git worktree`` of *this* clone; the eval itself clones the pushed branch separately.
_CANONICAL_SUBDIR = "monet_code"


def ensure_canonical_clone(repo_root: str | Path, evolvee_checkout: str | Path | None = None) -> Path:
    """Ensure ``<repo_root>/monet_code`` is a git clone (what the driver worktrees per pipeline).

    Already there → return it. Otherwise, if ``evolvee_checkout`` (a local experiment-copy path)
    is given, symlink it in — no new clone, no submodule, no wrapper repo. The existing checkout
    *is* the "standalone canonical clone" the driver wants; only the path/name differs.
    """
    repo_root = Path(repo_root)
    canonical = repo_root / _CANONICAL_SUBDIR
    if (canonical / ".git").exists():
        return canonical
    if not evolvee_checkout:
        raise ValueError(
            f"the vendored driver needs a standalone evolvee git clone at {canonical} (it carves "
            f"a per-pipeline worktree of it). None found — point algorithm.hparams.evolvee_checkout "
            f"at your local experiment copy and it will be linked in."
        )
    src = Path(evolvee_checkout).expanduser().resolve()
    if not (src / ".git").exists():
        raise ValueError(f"evolvee_checkout {src} is not a git clone (no .git)")
    if canonical.is_symlink() or canonical.exists():
        if canonical.is_symlink() and canonical.resolve() == src:
            return canonical
        raise ValueError(f"{canonical} already exists and does not point at {src}; remove it first")
    repo_root.mkdir(parents=True, exist_ok=True)
    canonical.symlink_to(src)
    return canonical


def prepare_worktree_env(repo_root: str | Path, worktree_parent: str | Path) -> None:
    """Pin the vendored driver's paths to our run, via ITS OWN documented overrides
    (``DARWINX_EVAL_REPO_ROOT`` / ``DARWINX_EVAL_WORKTREE_PARENT`` / ``DARWINX_EVAL_RESULTS_ROOT``).

    **Must run before the driver is imported**: the module reads these into import-time globals
    *and* into function default args (``add_eval_worktree(..., repo_root=REPO_ROOT)``;
    ``DEFAULT_RESULTS_ROOT``), so setting them after import is too late. These are the driver's own
    env channel — not beagle-introduced; the operator still only sets ``repo_root`` in config, and
    ``_launch`` translates it here.
    """
    import os

    Path(worktree_parent).mkdir(parents=True, exist_ok=True)
    os.environ["DARWINX_EVAL_REPO_ROOT"] = str(Path(repo_root).resolve())
    os.environ["DARWINX_EVAL_WORKTREE_PARENT"] = str(Path(worktree_parent).resolve())
    # Land the driver's eval SCRATCH (isolated ``_iso/<uuid>`` run dirs + the emitted
    # ``_self_evolve_configs`` handshake) UNDER our run dir, not in the vendored tree (whose
    # ``DEFAULT_RESULTS_ROOT`` would otherwise pollute source). An ``_evals/`` subdir keeps it
    # out of the genealogy DB / worktrees.
    os.environ["DARWINX_EVAL_RESULTS_ROOT"] = str(Path(repo_root).resolve() / "_evals")


def align_git_origin(canonical: str | Path, repo_url: str | None, *, remote: str = "origin") -> tuple | None:
    """Point the evolvee clone's ``origin`` at ``repo_url`` (the manifest repo) so the driver
    **pushes** candidate branches to the same fork the eval **clones** from — making the manifest
    the single source of truth for both. The driver's ``git push origin`` reads the checkout's
    local ``origin``, which can drift from the manifest; this re-aligns it. No-op if already
    aligned, ``repo_url`` is empty, or the dir isn't a git repo. Returns ``(old, new)`` if it
    changed, else ``None``.
    """
    import subprocess

    if not repo_url:
        return None
    canonical = str(canonical)
    try:
        cur = subprocess.run(["git", "-C", canonical, "remote", "get-url", remote],
                             capture_output=True, text=True, check=False).stdout.strip()
    except OSError:
        return None
    if not cur or cur == repo_url:
        return None
    subprocess.run(["git", "-C", canonical, "remote", "set-url", remote, repo_url], check=True)
    return (cur, repo_url)


def prepare_git_auth(token_env: str | None) -> str | None:
    """Wire non-interactive GitHub auth for the driver's **host-side** git — fetching the seed θ
    and pushing candidate branches to the evolvee experiment copy — from the token in
    ``$<token_env>``.

    The driver runs git with a restricted env that forwards ``GIT_CONFIG_*`` (but **not** the token
    var), so inject a per-process credential helper through that channel, with the token inlined
    into the value (a ``$VAR`` reference wouldn't resolve in the git subprocess). Nothing is written
    to any ``.gitconfig`` on disk; the token stays in this process's env only. Also disables git's
    interactive prompt so a missing credential fails loud instead of hanging. No-op without a token.
    Returns the token env name used, or ``None``.
    """
    import os

    os.environ["GIT_TERMINAL_PROMPT"] = "0"   # never hang on a username/password prompt
    if not token_env:
        return None
    token = os.environ.get(token_env)
    if not token:
        return None
    n = int(os.environ.get("GIT_CONFIG_COUNT", "0") or "0")
    helper = '!f() { echo username=x-access-token; echo "password=%s"; }; f' % token
    os.environ[f"GIT_CONFIG_KEY_{n}"] = "credential.https://github.com.helper"
    os.environ[f"GIT_CONFIG_VALUE_{n}"] = helper
    os.environ["GIT_CONFIG_COUNT"] = str(n + 1)
    return token_env


def _looks_like_sha(ref: str) -> bool:
    """A ref that's a bare commit sha (≥7 hex chars) — GitHub won't always serve it to a bare
    ``git fetch``, so we don't hand it to the driver as a *fetch* ref (mirrors ``worktree.py``)."""
    return len(ref) >= 7 and all(c in "0123456789abcdefABCDEF" for c in ref)


def prepare_runtime_env(run_config: RunConfig, evolvee_source: AgentSource) -> dict[str, str]:
    """Translate the run's **runtime / seed / concurrency** choices → the driver's env channel
    (bucket 2 — see ``notes/darwinX-migration/darwinx-env-inventory.md``). Set before the pipeline
    runs; the operator sets ``runtime.kind`` / ``evolvee.source`` / ``parallelism`` in config,
    never these env vars. Config wins over any stale value (config is bucket-2's source of truth).
    Returns the vars set (for logging/tests).

    * ``DARWINX_EVOLVE_EVAL_RUNTIME`` ← ``runtime.kind`` (one driver site defaults it to
      ``xrlenv-cluster``, so a ``local`` run must set it explicitly).
    * ``DARWINX_EVOLVE_ROOT_COMMIT`` ← the evolvee baseline ``ref`` (the root θ the driver seeds
      from; resolved locally from the checkout). A non-sha ref also sets ``…_FETCH_REF`` so a
      remote-only branch is fetchable.
    * ``DARWINX_EVAL_HARBOR_N_CONCURRENT`` ← ``parallelism`` (eval trial concurrency).
    """
    import os

    out: dict[str, str] = {}
    kind = getattr(run_config.runtime, "kind", None)
    if kind:
        out["DARWINX_EVOLVE_EVAL_RUNTIME"] = kind
    ref = getattr(evolvee_source, "ref", None)
    if ref:
        out["DARWINX_EVOLVE_ROOT_COMMIT"] = ref
        if not _looks_like_sha(ref):
            out["DARWINX_EVOLVE_ROOT_FETCH_REF"] = ref   # a branch/tag is fetchable; a bare sha isn't
    par = getattr(run_config, "parallelism", None)
    if par:
        out["DARWINX_EVAL_HARBOR_N_CONCURRENT"] = str(par)
    os.environ.update(out)
    return out


def prepare_task_subset_env(run_config: RunConfig) -> dict[str, str]:
    """Translate the benchmark's **task-subset selection** → the driver's ``DARWINX_EVAL_*_TASKS``
    env (bucket 2). The driver's pool/pipeline read these comma-separated lists to stratify the
    eval, so config is their single source of truth (no floating env):

    * ``DARWINX_EVAL_EXCLUDE_TASKS`` ← ``benchmark.exclude_task_ids`` (denylist)
    * ``DARWINX_EVAL_PRIORITY_TASKS`` ← ``benchmark.options.priority_tasks`` (front-loaded)
    * ``DARWINX_EVAL_VARIANCE_TASKS`` ← ``benchmark.options.variance_tasks`` (avg@k probe set)
    * ``DARWINX_EVAL_FULLSET_TASKS``  ← ``benchmark.options.fullset_tasks`` (the scoring set)

    (The *primary* task list — which tasks to evolve on — is separate: it rides ``subset_tasks``
    on the ``PipelineConfig`` from ``benchmark.task_ids``.) Each value may be a list or a bare
    comma-string. Returns the vars set (for logging/tests)."""
    import os

    def _csv(v: Any) -> str:
        return ",".join(str(x) for x in v) if isinstance(v, (list, tuple)) else str(v)

    b = run_config.benchmark
    out: dict[str, str] = {}
    if b.exclude_task_ids:
        out["DARWINX_EVAL_EXCLUDE_TASKS"] = _csv(b.exclude_task_ids)
    opts = dict(b.options or {})
    for opt_key, env_var in (("priority_tasks", "DARWINX_EVAL_PRIORITY_TASKS"),
                             ("variance_tasks", "DARWINX_EVAL_VARIANCE_TASKS"),
                             ("fullset_tasks", "DARWINX_EVAL_FULLSET_TASKS")):
        if opts.get(opt_key):
            out[env_var] = _csv(opts[opt_key])
    os.environ.update(out)
    return out


# --- campaign config (seam C) ------------------------------------------------

def _model_name(agent: Agent, role: str) -> str:
    model = getattr(agent.spec, "model", None)
    if model is None or not getattr(model, "name", ""):
        raise ValueError(
            f"DarwinX launch: the {role} ({agent.spec.name!r}) has no model — set its `model.name` "
            f"in the evolution config ({role}.model)."
        )
    return model.name


def agentconfig_dict(spec: Any) -> dict[str, Any]:
    """An ``AgentConfig``-shaped dict (``{name, config, model?}``) for an :class:`AgentSpec` —
    what :func:`meta_agent.set_editor_from_spec` consumes to rebuild the Editor per-process."""
    d: dict[str, Any] = {"name": spec.name, "config": dict(getattr(spec, "config", {}) or {})}
    model = getattr(spec, "model", None)
    if model is not None and getattr(model, "name", ""):
        d["model"] = {"name": model.name}
    return d


#: evolvee ``config`` keys that ride the benchmarked-agent block at TOP level (the vendored eval
#: reads them there — see ``codingbench_eval.from_self_evolve_config``). Everything else rides the
#: nested ``config`` passthrough so a non-monet evolvee's adapter knobs reach it generically.
_EVOLVEE_TOP_LEVEL_KEYS = ("max_turns", "timeout", "wire_provider", "auth")


def _evolvee_block(evolvee: Agent) -> dict[str, Any]:
    """The benchmarked-agent block: its beagle registry ``name`` + model plus the
    turn-budget / timeout / gateway-wire knobs the vendored eval reads (``build_codingbench_config``
    picks ``max_turns`` + ``timeout`` off this block → the eval's per-rollout budgets).

    ``name`` is what makes the eval **evolvee-agnostic**: the vendored eval resolves the benchmarked
    agent through beagle's registry (``beagle.agents.build(name)``) rather than assuming monet, so any
    registered agent (monet, mini-swe, …) can be the evolvee — selected by ``evolvee.harness.name``.
    Every OTHER evolvee ``config`` knob the target adapter reads (e.g. mini-swe's
    ``provider`` / ``effort`` / ``config_path`` / ``forward_env``) rides the nested ``config`` so a
    non-monet evolvee is fully configured without a monet-shaped block. The monet path is
    behaviourally unchanged: the block now always carries ``name``, but ``monet`` is the eval's
    default, so its budgets still ride the top-level keys and its install/args stay the adapter's
    own defaults."""
    block: dict[str, Any] = {"name": evolvee.spec.name, "model": _model_name(evolvee, "evolvee")}
    cfg = dict(getattr(evolvee.spec, "config", {}) or {})
    for key in _EVOLVEE_TOP_LEVEL_KEYS:
        if cfg.get(key) is not None:
            block[key] = cfg[key]
    # ``agent_source`` is dropped on purpose: the bridge derives it from the benchmark block's
    # repo/ref so the evolved commit and the code the container clones cannot drift apart. An
    # evolvee that genuinely needs its own source sets it in the campaign yaml, which
    # ``_general_agent_block`` honours via ``setdefault``.
    passthrough = {k: v for k, v in cfg.items()
                   if k not in _EVOLVEE_TOP_LEVEL_KEYS and k != "agent_source"}
    if passthrough:
        block["config"] = passthrough
    return block


def _benchmark_block(run_config: RunConfig, src: AgentSource) -> dict[str, Any]:
    """The benchmark block: benchmark identity from the run config + the evolvee's baseline
    ``repo``/``ref`` (the driver evolves branches off this experiment copy). Task selection is
    NOT here — it rides ``subset_tasks`` on the ``PipelineConfig``."""
    b = run_config.benchmark
    block: dict[str, Any] = {"name": b.name}
    for field, val in (("dataset", b.dataset), ("split", b.split), ("namespace", b.namespace),
                       ("tag", b.tag)):
        if val:
            block[field] = val
    if b.options:
        block["options"] = dict(b.options)
    if src.repo:
        block["repo_url"] = src.repo
    if src.ref:
        block["agent_ref"] = src.ref
    return block


def _runtime_block(run_config: RunConfig) -> dict[str, Any]:
    rc = run_config.runtime
    block: dict[str, Any] = {"kind": rc.kind}
    if rc.grpc_host:
        block["grpc_host"] = rc.grpc_host
    if rc.grpc_port:
        block["grpc_port"] = rc.grpc_port
    return block


def emit_campaign_config(*, evolvee: Agent, evolver: Agent, run_config: RunConfig,
                         dest: str | Path) -> Path:
    """Write the vendored driver's campaign config (seam C) → the path it was written to.

    Maps the evolvee → the benchmarked-agent block, the evolver → ``cursor_agent.model`` (the
    proposer model the driver preflights) **and** the :data:`EVOLVER_SPEC_KEY` block (so a
    worker subprocess rebuilds the Editor), the benchmark/runtime → their blocks.
    """
    import yaml

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    src = evolvee.source()  # resolved baseline θ (repo @ ref)
    doc = {
        "cursor_agent": {"model": _model_name(evolver, "evolver")},
        "monet": _evolvee_block(evolvee),
        "benchmark": _benchmark_block(run_config, src),
        "runtime": _runtime_block(run_config),
        EVOLVER_SPEC_KEY: agentconfig_dict(evolver.spec),
    }
    dest.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return dest


# --- pipeline config ---------------------------------------------------------

#: hparam keys that are launch-infra (consumed here), never forwarded to ``PipelineConfig``.
_RESERVED_HPARAMS = frozenset({"repo_root", "reports_root", "campaign", "workdir", "subset_label"})


def build_pipeline_config(pipeline_config_cls: type, *, campaign: str, reports_root: str | Path,
                          repo_root: str | Path, config_path: str | Path, evolver_model: str,
                          subset_tasks: list[str] | None, hparams: dict[str, Any]) -> Any:
    """Construct the vendored ``PipelineConfig``. Drift-proof: only keys the dataclass declares
    are passed, so an unknown hparam is ignored rather than raising ``TypeError``.

    ``cursor_model`` is passed explicitly to bypass its default-factory (which would read a
    stale default config off disk). ``subset_tasks`` carries the task selection; remaining
    hparams that match real fields become hyper-parameter overrides.
    """
    import dataclasses

    valid = {f.name for f in dataclasses.fields(pipeline_config_cls)}
    kwargs: dict[str, Any] = {
        "campaign": campaign,
        "reports_root": Path(reports_root),
        "repo_root": Path(repo_root),
        "config_path": Path(config_path),
    }
    if evolver_model and "cursor_model" in valid:
        kwargs["cursor_model"] = evolver_model
    if subset_tasks and "subset_tasks" in valid:
        kwargs["subset_tasks"] = list(subset_tasks)
        if "subset_label" in valid:
            kwargs["subset_label"] = hparams.get("subset_label", "subset")
    for key, val in hparams.items():
        if key in valid and key not in _RESERVED_HPARAMS and key not in kwargs:
            kwargs[key] = val
    # The driver's PipelineConfig `parent_strategy` default is a *retired* strategy
    # ('high_score_few_children'); its own registry rejects it. Fall back to the driver's canonical
    # default (DEFAULT_STRATEGY_NAME) unless the operator picked one — the original launcher always
    # passed `--parent-strategy`, so relying on the stale default is a migration gap.
    if "parent_strategy" in valid and "parent_strategy" not in kwargs:
        from evolve import parent_selection
        kwargs["parent_strategy"] = parent_selection.DEFAULT_STRATEGY_NAME
    return pipeline_config_cls(**kwargs)


# --- launch the campaign (the two-phase supervisor sequence) ------------------

def _unscored_root_id(tree: Any, cfg: Any) -> str | None:
    """The campaign's root node id if it exists but has **no baseline score yet**, else ``None``.

    The driver's evolve phase can only pick a *scored* parent, so a root with no search-eval must
    be scored by a bootstrap precursor first. ``None`` means either there's no root yet (no DB) or
    the root is already scored (nothing to do).
    """
    db = tree.db_path_for(cfg.reports_root, cfg.campaign)
    if not Path(db).exists():
        return None
    conn = tree.connect(db)
    try:
        row = conn.execute(
            "SELECT id FROM nodes WHERE campaign = ? AND subset = ? AND parent_id IS NULL "
            "ORDER BY created_at ASC LIMIT 1", (cfg.campaign, cfg.subset_label)).fetchone()
        if not row:
            return None
        root_id = row["id"] if hasattr(row, "keys") else row[0]
        scored = tree.search_eval_by_node(conn, campaign=cfg.campaign, subset=cfg.subset_label)
        return None if root_id in scored else root_id
    finally:
        conn.close()


def run_campaign(pipeline_cls: type, cfg: Any, tree: Any, *, log=print) -> int:
    """Drive the driver's two-phase launch that the (un-vendored) supervisor used to do:

    1. Ensure the root exists and is **scored** — the evolve run can only select a scored parent
       (:func:`parent_selection._eligible_nodes`). A fresh campaign's first run bootstraps the root
       but can't pick it (unscored), so we detect that, then
    2. run a **bootstrap precursor** (``bootstrap_only=True`` + the root as ``parent_id_override``)
       that scores the root's baseline, then
    3. run the **evolve** pipeline against the now-scored root.

    Returns the evolve run's rc. Idempotent across re-runs: a run whose root is already scored
    evolves directly; a campaign with a half-bootstrapped (unscored) root is scored then evolved.
    """
    import dataclasses

    root_id = _unscored_root_id(tree, cfg)
    if root_id is None:
        rc = pipeline_cls(cfg).run()                 # bootstraps the root; evolves if already scored
        root_id = _unscored_root_id(tree, cfg)
        if root_id is None:
            return rc                                # root was already scored → that run evolved
    log(f"[darwinx] scoring baseline root {root_id} (bootstrap precursor), then evolving")
    precursor = dataclasses.replace(cfg, bootstrap_only=True, parent_id_override=root_id)
    pipeline_cls(precursor).run()                    # score the root's baseline
    return pipeline_cls(cfg).run()                   # evolve from the scored root


# --- read the winner back ----------------------------------------------------

def _loads(js: str | None) -> list:
    import json

    try:
        val = json.loads(js or "[]")
        return val if isinstance(val, list) else []
    except (ValueError, TypeError):
        return []


def read_best(tree: Any, pool: Any, cfg: Any, *, evolvee_source: AgentSource) -> Candidate | None:
    """Read the best evolved node from the run's genealogy DB → a :class:`Candidate` whose
    ``source`` is the evolvee repo at the winning branch/commit. ``None`` if nothing scored
    (no DB yet, or an empty campaign)."""
    db = tree.db_path_for(cfg.reports_root, cfg.campaign)
    if not Path(db).exists():
        return None
    conn = tree.connect(db)
    try:
        node = pool.best_node(conn, campaign=cfg.campaign, subset=cfg.subset_label)
    finally:
        conn.close()
    if node is None:
        return None
    ref = node.branch_name or node.commit_sha
    src = replace(evolvee_source, ref=ref) if ref else evolvee_source
    status = CandidateStatus.KEPT if node.status == "completed" else CandidateStatus.NO_CHANGE
    return Candidate(
        id=node.id, source=src, parent_id=node.parent_id, status=status, score=node.score,
        improved_tasks=_loads(node.improved_tasks_json),
        regressed_tasks=_loads(node.regressed_tasks_json),
        metadata={"branch": node.branch_name, "commit": node.commit_sha,
                  "resolved_tasks": _loads(node.resolved_tasks_json)},
    )


__all__ = [
    "prepare_import_path", "ensure_canonical_clone", "prepare_worktree_env", "prepare_runtime_env",
    "prepare_task_subset_env",
    "prepare_git_auth", "align_git_origin", "emit_campaign_config", "build_pipeline_config",
    "run_campaign", "read_best",
    "agentconfig_dict", "EVOLVER_SPEC_KEY",
]
