"""coding-bench eval-seam adapter for self_evolve.

This is the Phase-1 bridge between self_evolve's orchestrator and
coding-bench's runner. It exposes the SAME public surface the pipeline
already consumes from :mod:`self_evolve.eval_runner` — ``run_full`` /
``run_subset`` / ``parse_existing_result_dir`` / ``restrict_to_subset``
/ ``task_base`` / ``task_bases`` / ``EvalResult`` /
``EvalInfrastructureError`` — but it drives coding-bench's OWN runner
(``python -m runner.run``) against the ``terminal-bench-v2`` / ``monet``
/ ``local`` stack instead of shelling out to monet_code_eval's
``scripts/run_harbor.sh``.

────────────────────────────────────────────────────────────────────
WHY THIS EXISTS (the eval seam)
────────────────────────────────────────────────────────────────────
In pristine exp_05, ``eval_runner._run_harbor`` runs::

    bash scripts/run_harbor.sh <config.yaml> --include-task-name T ...

inside a per-pipeline monet_code worktree (``cwd``), snapshots the
worktree's ``jobs/`` dir, and parses the new ``jobs/<ts>/result.json``
(Harbor's ``stats.evals.<key>.{metrics,reward_stats,exception_stats}``)
into an :class:`EvalResult`.

coding-bench has no ``run_harbor.sh``. The equivalent surface is:

    python -m runner.run <config.yaml> --results-root <dir>
        → runner.run.main
        → BENCHMARKS["terminal-bench-v2"].runner_key == "harbor"
        → benchmarks.terminal_bench.runner.harbor_runner
        → (per task) agents.monet.harbor_agent.MonetHarborAgent
        → results/runs/<run_id>/run.json

So this adapter:

  1. Translates self_evolve's call (``config_path``, ``task_names``,
     monet commit) into a coding-bench :class:`runner.config.Config`
     (terminal-bench-v2 + monet + local), with ``benchmark.task_ids``
     set to the subset.
  2. Invokes ``python -m runner.run`` as a subprocess (mirrors the
     run_harbor.sh subprocess model, including optional tee logging).
  3. Snapshots ``<results_root>/runs/`` before/after to find the new
     run dir, then lifts ``run.json``'s ``per_task_results`` into an
     :class:`EvalResult` via :func:`parse_run_json`.

────────────────────────────────────────────────────────────────────
MAPPING TABLE  (self_evolve  →  coding-bench)
────────────────────────────────────────────────────────────────────
  eval_runner.run_full/run_subset      runner.run.main (subprocess)
  --include-task-name T  (Harbor CLI)  benchmark.task_ids=[...]
                                       (loader._iter_task_dirs filter)
  cwd = monet_code worktree            agent.config.agent_source ref =
                                       the monet_code commit under eval
                                       (GitClone — see "STUBBED" below)
  jobs/<ts>/result.json                results/runs/<run_id>/run.json
  stats.evals.<k>.reward_stats["1.0"]  per_task_results[].reward >= 1.0
  EvalResult.score (mean reward)       mean(per_task_results[].reward)
  EvalInfrastructureError              run.json errors[] matched against
                                       INFRASTRUCTURE_FAILURE_PATTERNS
  trial name "<task>__<hash>"          task_id == task dir name (no
                                       hash; task_base is identity)

────────────────────────────────────────────────────────────────────
STUBBED / DEFERRED TO PHASE 2  (the central risk)
────────────────────────────────────────────────────────────────────
* monet-commit injection. exp_05 evaluates *uncommitted/committed local
  worktree state* by installing monet from the worktree ``cwd``.
  :class:`agents.monet.harbor_agent.MonetHarborAgent` REQUIRES
  ``agent_source.type == "git_clone"`` and explicitly REJECTS bind
  mounts — it ``git clone --depth 1 --branch <ref> <repo_url>`` inside
  the container. So "evaluate monet at commit X" must map to a *ref a
  container can clone*. This adapter threads a ``monet_ref`` /
  ``monet_repo_url`` through to ``agent.config.agent_source`` but does
  NOT yet solve "make each candidate worktree commit clonable by the
  container" (a local ``file://`` clone isn't reachable from inside a
  container without extra plumbing). Reconciling self_evolve's worktree
  model with GitClone is the #1 Phase-2 decision.

* model/provider + forward_env translation from the self-evolve YAML's
  ``monet:`` block (auth/model/max_turns) into coding-bench's
  model+agent config is approximate here (see
  :func:`build_codingbench_config`); Phase 2 should pin it against the
  live Express/eng-ai-gateway provider.

* num_samples > 1 (pass@k). coding-bench's TB2 path is one trial per
  task by default, so partial-credit (``partially_solved_tasks``) only
  appears when the dataset/loader actually emits repeats. self_evolve's
  trial-level reward model degrades gracefully to task-level here.
"""

from __future__ import annotations

import ipaddress
import json
import os
import select
import socket
import socketserver
import subprocess
import sys
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

import yaml

# Reuse the pristine EvalResult shape + helpers verbatim so the pipeline
# sees an identical contract regardless of which seam produced the
# result. Only the *production* of the result differs.
from .eval_runner import (
    INFRASTRUCTURE_FAILURE_PATTERNS,
    EvalInfrastructureError,
    EvalResult,
    _classify_task_outcomes,
    restrict_to_subset,
    task_base,
    task_bases,
)

__all__ = [
    "EvalResult",
    "EvalInfrastructureError",
    "CodingBenchEvalConfig",
    "run_full",
    "run_subset",
    "parse_existing_result_dir",
    "parse_run_json",
    "restrict_to_subset",
    "build_codingbench_config",
    "task_base",
    "task_bases",
]


# Repo root = the coding-bench clone (self_evolve/ lives directly under
# it). ``python -m runner.run`` must run from here so ``runner`` /
# ``agents`` / ``benchmarks`` import.
REPO_ROOT = Path(__file__).resolve().parents[1]
# beagle patch: the default lands eval SCRATCH (``_iso/<uuid>/…`` + the emitted
# ``_self_evolve_configs`` handshake) inside the *vendored* tree, polluting source. beagle's
# ``_launch`` sets ``MONET_EVAL_RESULTS_ROOT`` to the run dir so it lands with the run instead
# (read at import — _launch sets it before the driver is imported, per the drop-in contract).
DEFAULT_RESULTS_ROOT = (
    Path(os.environ["MONET_EVAL_RESULTS_ROOT"])
    if os.environ.get("MONET_EVAL_RESULTS_ROOT")
    else REPO_ROOT / "results"
)
# Vendored TB2 task tree (submodule). Matches benchmark.dataset in
# configs/test_monet_cloud_tb2.yaml.
DEFAULT_TBENCH_DATASET = "benchmarks/terminal_bench/vendor"
# The monet fork the container clones. This MUST be a build that carries
# the provider self-evolve drives monet through: the live campaign uses
# ``--provider llm-gateway-express-local-proxy`` (the Express LLM gateway
# behind a loopback reverse tunnel), which only exists in monet_code's
# exp_05 lineage. We point at ``yifan-zhang_sfemu/monet_code`` — an OWNED
# fork of that lineage (forked from ``zeyuan-chen_sfemu/monet_code``), so
# it inherits the same provider build, the ``develop`` tip, and the full
# history (``d5c17da`` etc.) while ALSO being a repo we can push to.
# Owning it matters for candidate-commit injection below: the throwaway
# branch a child's evolved commit is pushed to lands on this SAME repo the
# container clones from, instead of relying on someone else's fork as the
# push target. (The earlier vendored pin ``juntao-tan_sfemu/monet_code@eval``
# predates the Express provider and fails at runtime with ``Unknown
# provider: llm-gateway-express-local-proxy``.) ``develop`` is the
# published tip the self-evolve campaign packs its monet from; candidate
# injection overrides the ref per child (see "STUBBED" note + below).
# Override for YOUR fork via ``SELF_EVOLVE_MONET_REPO_URL`` (e.g. in .env): the
# campaign both clones the agent from AND pushes evolved candidate commits to
# this repo, so it must be a fork you can push to (forked from the same
# Express-provider lineage). Defaults to the owned yifan-zhang fork.
DEFAULT_MONET_REPO_URL = (
    os.environ.get("SELF_EVOLVE_MONET_REPO_URL", "").strip()
    or "https://github.com/yifan-zhang_sfemu/monet_code.git"
)
DEFAULT_MONET_REF = "develop"

# monet install command run inside the trial container (as the agent
# user, which is root on the TB2 task images). TB2 base images are
# minimal (e.g. bare ``ubuntu:24.04``) and frequently ship no Node — or
# a Node too old for monet's deps — so we bootstrap Node 20 before
# ``npm ci``/``npm link``. Lifted verbatim from
# configs/test_monet_cloud_tb2.yaml, the validated cloud-sweep config.
# The node-bootstrap preamble (everything before the final ``cd <path>
# && npm ci``). Kept as a separate constant because the bash body
# contains ``${ID:-}`` etc. that would collide with ``str.format`` — the
# container path is appended by hand in ``build_codingbench_config``.
_MONET_NODE_BOOTSTRAP = """\
need_install=1
if command -v node >/dev/null 2>&1; then
  if node -e 'const [maj,min]=process.versions.node.split(".").map(Number); process.exit((maj>20||(maj===20&&min>=5))?0:1)'; then
    need_install=0
  fi
fi
if [ "$need_install" = "1" ]; then
  . /etc/os-release
  if [ "${ID:-}" = "alpine" ]; then
    apk add --no-cache nodejs-current npm || apk add --no-cache nodejs npm
  else
    if ! command -v curl >/dev/null 2>&1; then
      apt-get update -qq && apt-get install -y --no-install-recommends curl ca-certificates gnupg
    fi
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    apt-get install -y nodejs
  fi
  for bin in node npm npx; do
    if [ -x "/usr/bin/$bin" ] && [ -e "/usr/local/bin/$bin" ]; then
      ln -sf "/usr/bin/$bin" "/usr/local/bin/$bin"
    fi
  done
fi
"""


def _monet_install_cmd(container_path: str) -> str:
    """monet install command run inside the trial container (as root on
    the TB2 task images). Bootstraps Node 20 if absent/too old, then
    ``npm ci``/``npm link`` from the cloned source. Mirrors the validated
    cloud-sweep config (configs/test_monet_cloud_tb2.yaml)."""
    return (
        _MONET_NODE_BOOTSTRAP
        + f"cd {container_path} && npm ci --omit=dev && npm link"
    )


# ──────────────────────────────────────────────────────────────────────
# Runtime selection — local Docker vs the xrlenv cluster
# ──────────────────────────────────────────────────────────────────────
# coding-bench's runner picks a container runtime from ``config.runtime.kind``
# against its ``runner.run.RUNTIMES`` registry (``local`` →
# LocalDockerRuntime, ``xrlenv-cluster`` → XrlenvDockerRuntime). The seam
# defaults to ``local`` (one host, host Docker daemon) and is switchable to
# ``xrlenv-cluster`` (fan trials across cluster nodes) per campaign. The
# cluster path additionally swaps the Harbor environment to
# ``xrlenv_plugins.harbor:XrlenvHarborEnvironmentCluster`` — that wiring
# lives in benchmarks/terminal_bench/runner.py (it branches on the runtime
# instance), so the seam only has to emit the right ``runtime`` block and
# let the runner do the rest. Per-task resource caps and node fan-out are
# the cluster environment plugin's job, not the seam's.
_RUNTIME_ENV = "SELF_EVOLVE_EVAL_RUNTIME"
# Mirrors runner.run.RUNTIMES — keep in sync if a new backend lands there.
_VALID_RUNTIME_KINDS = ("local", "xrlenv-cluster")


def _resolve_runtime_kwargs(runtime_block: Any) -> dict[str, Any]:
    """Resolve the runtime selector into :class:`CodingBenchEvalConfig` kwargs.

    Precedence: ``$SELF_EVOLVE_EVAL_RUNTIME`` env override > the YAML
    ``runtime:`` block > the dataclass default. The xrlenv connection
    overrides (``grpc_host`` / ``grpc_port`` / ``grpc_secure``) are read
    from the YAML block only when present; the consumer token is never
    accepted here (it is a secret, read from ``XRLENV_CONSUMER_TOKEN`` in
    the env). An unknown ``kind`` raises so a typo fails loud at campaign
    start rather than silently running local.
    """
    out: dict[str, Any] = {}
    block = runtime_block if isinstance(runtime_block, dict) else {}

    kind = os.environ.get(_RUNTIME_ENV, "").strip()
    if not kind:
        yaml_kind = block.get("kind")
        kind = yaml_kind.strip() if isinstance(yaml_kind, str) else ""
    if kind:
        if kind not in _VALID_RUNTIME_KINDS:
            raise ValueError(
                f"unknown eval runtime {kind!r}; valid: {list(_VALID_RUNTIME_KINDS)} "
                f"(set via {_RUNTIME_ENV} or the self-evolve YAML runtime.kind)"
            )
        out["runtime_kind"] = kind

    host = block.get("grpc_host")
    if isinstance(host, str) and host.strip():
        out["xrlenv_grpc_host"] = host.strip()
    port = block.get("grpc_port")
    if isinstance(port, int):
        out["xrlenv_grpc_port"] = port
    secure = block.get("grpc_secure")
    if isinstance(secure, bool):
        out["xrlenv_grpc_secure"] = secure
    return out


@dataclass(frozen=True)
class CodingBenchEvalConfig:
    """Static knobs for the coding-bench seam, resolved once per campaign.

    These are the bits self_evolve's per-iteration ``run_full`` /
    ``run_subset`` calls don't carry but coding-bench's runner needs.
    Phase 2 populates this from the self-evolve YAML + campaign context;
    Phase 1 ships defaults that mirror configs/test_monet_cloud_tb2.yaml
    with ``runtime: local``.
    """

    dataset: str = DEFAULT_TBENCH_DATASET
    # Benchmark id passed to the coding-bench runner. Default keeps the
    # terminal-bench-v2 seam; a held-out CROSS-BENCHMARK eval (cross_bench.py)
    # points this + `dataset` at another registered benchmark (e.g.
    # swe-bench-verified) via `dataclasses.replace`.
    benchmark_name: str = "terminal-bench-v2"
    # WAI/browser support (browser-use harness eval, e.g. webarena-infinity).
    # Auto-enabled when benchmark_name == "webarena-infinity" (see
    # from_self_evolve_config); the terminal path is byte-identical when off.
    benchmark_split: "str | None" = None
    benchmark_options: "dict | None" = None
    browser_mode: bool = False
    browser_start_url: str = "http://localhost:9000"
    browser_max_actions: int = 30
    browser_max_images: int = 5
    # Harbor namespace/tag for non-TB2 harbor benchmarks (e.g. TerminalWorld:
    # harbor-terminalworld-verified). Emitted only when set; TB2 path unchanged.
    benchmark_namespace: "str | None" = None
    benchmark_tag: "str | None" = None
    monet_repo_url: str = DEFAULT_MONET_REPO_URL
    monet_ref: str = DEFAULT_MONET_REF
    monet_token_env: str = "GH_TOKEN"
    monet_container_path: str = "/opt/agent"
    # Model the benchmarked monet agent talks to (inside the container).
    # This becomes monet's ``--model`` arg, so it MUST be a bare monet
    # model id (e.g. ``gpt-5.5``) — NOT a LiteLLM-prefixed id like
    # ``openai/gpt-5.5`` (that prefix is a coding-bench/LiteLLM routing
    # hint and is meaningless to monet's own CLI).
    model_name: str = "gpt-5.5"
    # ``model.provider`` is consumed only by coding-bench's *own* LiteLLM
    # calls (``runner.model_call.call_model``). The monet TB2 path never
    # routes through that — monet makes its own API calls inside the
    # container via ``--provider`` (see ``monet_wire_provider``). This
    # value must still be a valid ``runner.config.Provider`` so the
    # generated Config validates; it is otherwise inert here.
    model_provider: str = "eng-ai-gateway"
    # monet CLI provider arg (`monet --provider <...>`). The live
    # self-evolve campaign drives monet through the Express LLM gateway's
    # local loopback proxy; this is the matching monet provider id (see
    # monet_code ``PROVIDER_LLM_GATEWAY_EXPRESS_LOCAL_PROXY``).
    monet_wire_provider: str = "llm-gateway-express-local-proxy"
    # Host env vars carrying the Express gateway bearer key + the local
    # proxy base URL. Both are forwarded into the trial container (see
    # ``build_codingbench_config``); the URL is rewritten to a
    # container-reachable address by the relay in ``_run``.
    express_key_env: str = "LLM_GATEWAY_EXPRESS_API_KEY"
    express_proxy_url_env: str = "LLM_GATEWAY_EXPRESS_LOCAL_PROXY_URL"
    # Host env vars carrying the SFR gateway bearer key + base URL, used
    # when ``monet_wire_provider == "sfr-gateway"`` (the cluster path: SFR
    # is a real network endpoint reachable from remote nodes, so no
    # host-loopback relay is needed). They are renamed into the container
    # as ``OPENAI_GATEWAY_API_KEY`` / ``OPENAI_BASE_URL`` — the names
    # monet's ``--provider sfr-gateway`` reads (see build_codingbench_config).
    sfr_key_env: str = "X_API_KEY"
    sfr_url_env: str = "SFR_GATEWAY_OPENAI_URL"
    max_turns: int = 50
    timeout: int = 1800
    parallelism: int = 1
    # Trials per task in a single runner pass (the TB2 runner DOES honor this:
    # it expands tasks x num_samples into ``<task>__s<idx>`` trials). The seam
    # defaults to 1 (best-of-N is done by re-running, see ``_run``); the
    # variance-aware avg@k path (``run_subset_sampled``) bumps it to k so we get
    # k independent samples per task in one pass for a denoised pass-rate.
    num_samples: int = 1
    # Container runtime for coding-bench's runner. ``local`` spins trial
    # containers on the host Docker daemon (single node, serial-ish);
    # ``xrlenv-cluster`` fans every trial across the xrlenv cluster (see
    # docs/self_evolve/CLUSTER_LAUNCH.md). Selectable per campaign via the
    # self-evolve YAML ``runtime:`` block or the ``SELF_EVOLVE_EVAL_RUNTIME``
    # env override; defaults to ``local`` so existing campaigns are unchanged.
    runtime_kind: str = "local"
    # Optional explicit xrlenv connection overrides (xrlenv-cluster only).
    # When left ``None`` the cluster client falls back to ``xrlenv.from_env()``
    # reading ``XRLENV_GRPC_HOST`` / ``XRLENV_GRPC_PORT`` /
    # ``XRLENV_GRPC_SECURE`` from the runner subprocess env — the recommended
    # path (see the cluster runbook). The consumer TOKEN is DELIBERATELY not a
    # field here: it is a secret and must never be serialised into the on-disk
    # runner config; it is read from ``XRLENV_CONSUMER_TOKEN`` in the env.
    xrlenv_grpc_host: str | None = None
    xrlenv_grpc_port: int | None = None
    xrlenv_grpc_secure: bool | None = None
    results_root: Path = DEFAULT_RESULTS_ROOT

    @classmethod
    def from_self_evolve_config(
        cls,
        config_path: "Path | str | None",
        *,
        results_root: "Path | None" = None,
        parallelism: int | None = None,
    ) -> "CodingBenchEvalConfig":
        """Build a :class:`CodingBenchEvalConfig` from a self-evolve YAML.

        Reads the ``monet:`` block (the benchmarked agent's model + turn
        budget + credential profile) and the optional ``runtime:`` block
        (the container runtime selector). The cursor-agent meta-agent
        model lives under ``cursor_agent:`` and is consumed elsewhere
        (run_config); it is intentionally NOT read here.

        ``monet.model`` becomes the bare monet model id the container's
        ``monet --model`` uses. ``monet.auth`` (or the optional
        ``monet.wire_provider`` override) selects the ``monet --provider``
        id. Everything else falls back to the dataclass defaults (which
        already mirror configs/test_monet_cloud_tb2.yaml + the live
        Express gateway), so a missing/partial block degrades gracefully.

        Runtime selection precedence (highest first):
          1. ``$SELF_EVOLVE_EVAL_RUNTIME`` env var (``local`` |
             ``xrlenv-cluster``) — flip the cluster on without editing the
             campaign YAML.
          2. the YAML ``runtime.kind`` (+ optional ``runtime.grpc_host`` /
             ``grpc_port`` / ``grpc_secure`` overrides).
          3. the dataclass default (``local``).
        The xrlenv consumer token is never read from the YAML — it must be
        supplied via ``XRLENV_CONSUMER_TOKEN`` in the env (see the cluster
        runbook).
        """
        auth_to_wire = {
            "openai": "openai",
            "sfr_gateway": "sfr-gateway",
            "llm_gateway_express_local_proxy": "llm-gateway-express-local-proxy",
        }
        kwargs: dict[str, Any] = {}
        raw: dict[str, Any] = {}
        if config_path is not None:
            try:
                loaded = yaml.safe_load(Path(config_path).read_text()) or {}
                raw = loaded if isinstance(loaded, dict) else {}
            except (OSError, yaml.YAMLError):
                raw = {}
            monet = raw.get("monet")
            if isinstance(monet, dict):
                model = monet.get("model")
                if isinstance(model, str) and model.strip():
                    kwargs["model_name"] = model.strip()
                max_turns = monet.get("max_turns")
                if isinstance(max_turns, int) and max_turns > 0:
                    kwargs["max_turns"] = max_turns
                # beagle: honor a config-driven per-rollout timeout (evolvee.timeout in config.yaml
                # → the campaign monet block), the same env→config shift as max_turns above.
                timeout = monet.get("timeout")
                if isinstance(timeout, int) and timeout > 0:
                    kwargs["timeout"] = timeout
                wire = monet.get("wire_provider")
                if isinstance(wire, str) and wire.strip():
                    kwargs["monet_wire_provider"] = wire.strip()
                else:
                    auth = monet.get("auth")
                    if isinstance(auth, str) and auth.strip() in auth_to_wire:
                        kwargs["monet_wire_provider"] = auth_to_wire[auth.strip()]
            bench = raw.get("benchmark")
            if isinstance(bench, dict):
                bname = bench.get("name")
                if isinstance(bname, str) and bname.strip():
                    kwargs["benchmark_name"] = bname.strip()
                    if bname.strip() == "webarena-infinity":
                        kwargs["browser_mode"] = True
                        kwargs["dataset"] = None   # WAI -> vendored answers unless overridden below
                ds = bench.get("dataset")
                if isinstance(ds, str) and ds.strip():
                    kwargs["dataset"] = ds.strip()
                sp = bench.get("split")
                if isinstance(sp, str) and sp.strip():
                    kwargs["benchmark_split"] = sp.strip()
                opts = bench.get("options")
                if isinstance(opts, dict) and opts:
                    kwargs["benchmark_options"] = dict(opts)
                ns = bench.get("namespace")
                if isinstance(ns, str) and ns.strip():
                    kwargs["benchmark_namespace"] = ns.strip()
                tg = bench.get("tag")
                if isinstance(tg, str) and tg.strip():
                    kwargs["benchmark_tag"] = tg.strip()
                bref = bench.get("agent_ref") or bench.get("monet_ref")
                if isinstance(bref, str) and bref.strip():
                    kwargs["monet_ref"] = bref.strip()
                brepo = bench.get("repo_url") or bench.get("monet_repo_url")
                if isinstance(brepo, str) and brepo.strip():
                    kwargs["monet_repo_url"] = brepo.strip()
        kwargs.update(_resolve_runtime_kwargs(raw.get("runtime")))
        if results_root is not None:
            kwargs["results_root"] = results_root
        if parallelism is not None:
            kwargs["parallelism"] = parallelism
        return cls(**kwargs)


# ──────────────────────────────────────────────────────────────────────
# Config translation
# ──────────────────────────────────────────────────────────────────────
def _agent_block(cb_cfg, forward_env):
    """Agent config for the runner. BROWSER mode (webarena-infinity) emits the
    browser monet_args + npm-ci install; else the terminal monet path (unchanged)."""
    src = {
        "agent_source": {
            "type": "git_clone",
            "repo_url": cb_cfg.monet_repo_url,
            "ref": cb_cfg.monet_ref,
            "token_env": cb_cfg.monet_token_env,
            "container_path": cb_cfg.monet_container_path,
        },
    }
    if cb_cfg.browser_mode:
        src["install_cmd"] = f"cd {cb_cfg.monet_container_path} && npm ci"
        src["monet_args"] = [
            "--provider", cb_cfg.monet_wire_provider,
            "--browser-mode", "execute",
            "--browser-start-url", cb_cfg.browser_start_url,
            "--headless",
            "--max-images", str(cb_cfg.browser_max_images),
            "--browser-max-actions", str(cb_cfg.browser_max_actions),
            "--trajectory-dir", "/work/trajectory",
            "--yolo",
            "--output-format", "stream-json",
        ]
    else:
        src["install_cmd"] = _monet_install_cmd(cb_cfg.monet_container_path)
        src["monet_args"] = [
            "--provider", cb_cfg.monet_wire_provider,
            *(["--effort", os.environ["MONET_EVAL_EFFORT"].strip()]
              if os.environ.get("MONET_EVAL_EFFORT", "").strip() else []),
            "--all-permissions",
            "--no-monet-md",
            "--output-format", "stream-json",
        ]
    src["forward_env"] = forward_env
    src["max_turns"] = cb_cfg.max_turns
    src["timeout"] = cb_cfg.timeout
    return {"name": "monet", "config": src}


def _benchmark_block(cb_cfg, task_ids):
    b = {
        "name": cb_cfg.benchmark_name,
        "split": cb_cfg.benchmark_split,
        "task_ids": task_ids,
        "num_samples": cb_cfg.num_samples,
    }
    # WAI defaults to its vendored answers when dataset is unset; only the
    # terminal path (and explicit WAI overrides) carry a dataset path.
    if cb_cfg.dataset:
        b["dataset"] = cb_cfg.dataset
    if cb_cfg.benchmark_namespace:
        b["namespace"] = cb_cfg.benchmark_namespace
    if cb_cfg.benchmark_tag:
        b["tag"] = cb_cfg.benchmark_tag
    if cb_cfg.benchmark_options:
        b["options"] = dict(cb_cfg.benchmark_options)
    return b


def build_codingbench_config(
    *,
    task_names: list[str],
    cb_cfg: CodingBenchEvalConfig,
) -> dict[str, Any]:
    """Produce a coding-bench ``runner.config.Config``-shaped dict.

    The returned dict validates against :class:`runner.config.Config`
    (``extra="forbid"`` on every sub-model). ``benchmark.task_ids`` is
    the self_evolve subset; an empty list means "no filter → full
    sweep" (encoded as ``None`` because the loader treats ``None`` as
    unfiltered).

    NOTE: the monet ``forward_env`` / install_cmd here are a faithful
    copy of configs/test_monet_cloud_tb2.yaml's local-friendly variant.
    The exact provider/credential wiring for the live Express tunnel is
    a Phase-2 refinement (see module docstring).
    """
    task_ids = list(task_names) if task_names else None

    # Forward the Express gateway credentials into the trial container so
    # monet's ``--provider llm-gateway-express-local-proxy`` resolves:
    #   * LLM_GATEWAY_EXPRESS_API_KEY      → bearer key (monet reads it).
    #   * LLM_GATEWAY_EXPRESS_LOCAL_PROXY_URL → base URL monet POSTs to.
    # The URL value handed to the container is the container-reachable
    # one the relay in ``_run`` writes back into the runner env (a
    # loopback ``127.0.0.1`` value would be the container's own loopback,
    # not the host's, so the relay rewrites it to the docker bridge
    # gateway). ``forward_env`` maps container-env-name → host-env-name.
    #
    # Provider-dependent credential wiring:
    #   * sfr-gateway (the cluster path): monet's ``--provider sfr-gateway``
    #     reads ``OPENAI_BASE_URL`` + ``OPENAI_GATEWAY_API_KEY`` inside the
    #     container. The SFR gateway is a real network endpoint reachable
    #     from remote cluster nodes, so no host-loopback relay is involved;
    #     we just rename the host's SFR creds into those container names.
    #   * llm-gateway-express-local-proxy (the local path): forward the
    #     Express key + (relayed) proxy URL passthrough, as before.
    if cb_cfg.monet_wire_provider == "sfr-gateway":
        forward_env = [
            {"OPENAI_GATEWAY_API_KEY": cb_cfg.sfr_key_env},
            {"OPENAI_BASE_URL": cb_cfg.sfr_url_env},
        ]
    else:
        forward_env = [
            {cb_cfg.express_key_env: cb_cfg.express_key_env},
            {"LLM_GATEWAY_EXPRESS_API_KEY_LIST": "LLM_GATEWAY_EXPRESS_API_KEY_LIST"},
            {cb_cfg.express_proxy_url_env: cb_cfg.express_proxy_url_env},
        ]

    return {
        "model": {
            "name": cb_cfg.model_name,
            "provider": cb_cfg.model_provider,
            "api_base": None,
            # Empty params: monet owns its own sampling defaults. We
            # deliberately don't pin ``temperature`` (gpt-5.5 is a
            # reasoning model that 400s on any non-default temperature,
            # and monet has no ``--temperature`` flag anyway) or
            # ``max_tokens`` (monet picks a model-appropriate cap).
            "params": {},
        },
        "agent": _agent_block(cb_cfg, forward_env),
        "benchmark": _benchmark_block(cb_cfg, task_ids),
        "parallelism": cb_cfg.parallelism,
        "runtime": _build_runtime_block(cb_cfg),
    }


# Env vars xrlenv.from_env() reads for cluster (connect) mode. The seam
# requires at least the host + token before it will run a cluster eval, so a
# misconfigured campaign fails fast instead of silently degrading to local.
_XRLENV_HOST_ENV = "XRLENV_GRPC_HOST"
_XRLENV_TOKEN_ENV = "XRLENV_CONSUMER_TOKEN"


def _require_cluster_creds(cb_cfg: CodingBenchEvalConfig) -> None:
    """Guard: ``runtime: xrlenv-cluster`` must have real cluster creds.

    ``xrlenv.from_env()`` falls back to LocalDocker mode when no gRPC host
    is set — so without this guard a cluster campaign with missing creds
    would run on the local daemon (slow, single node) and look like it
    "worked". We require the gRPC host (from the config's
    ``runtime.grpc_host`` or ``$XRLENV_GRPC_HOST``) AND the consumer token
    (``$XRLENV_CONSUMER_TOKEN`` only — never the config) and otherwise raise
    a clear, actionable error. Local mode is unaffected.
    """
    if cb_cfg.runtime_kind != "xrlenv-cluster":
        return
    host = cb_cfg.xrlenv_grpc_host or os.environ.get(_XRLENV_HOST_ENV, "").strip()
    token = os.environ.get(_XRLENV_TOKEN_ENV, "").strip()
    missing: list[str] = []
    if not host:
        missing.append(f"a gRPC host (runtime.grpc_host or ${_XRLENV_HOST_ENV})")
    if not token:
        missing.append(f"${_XRLENV_TOKEN_ENV}")
    if missing:
        raise RuntimeError(
            "runtime=xrlenv-cluster selected but the cluster is not "
            f"configured: missing {', '.join(missing)}. xrlenv would "
            "silently fall back to local Docker, so this run is refused. "
            "Start the control-plane/tunnel with scripts/xrlenv_up.sh and "
            "export XRLENV_GRPC_HOST / XRLENV_GRPC_PORT / "
            "XRLENV_CONSUMER_TOKEN (see docs/self_evolve/CLUSTER_LAUNCH.md). "
            "To run on a single node instead, use runtime=local."
        )


def _build_runtime_block(cb_cfg: CodingBenchEvalConfig) -> dict[str, Any]:
    """Render the coding-bench ``runtime:`` sub-config from the seam config.

    ``local`` emits the bare ``{kind: local}``. ``xrlenv-cluster`` emits
    ``{kind: xrlenv-cluster}`` plus only the explicitly-set connection
    overrides — leaving the rest unset so the runner's
    ``XrlenvClusterRuntimeConfig`` lets ``xrlenv.from_env()`` read
    ``XRLENV_GRPC_HOST`` / ``XRLENV_GRPC_PORT`` / ``XRLENV_GRPC_SECURE``
    (and the secret ``XRLENV_CONSUMER_TOKEN``) from the runner subprocess
    env. The token is never written into this on-disk config.
    """
    if cb_cfg.runtime_kind == "local":
        return {"kind": "local"}
    if cb_cfg.runtime_kind == "xrlenv-cluster":
        block: dict[str, Any] = {"kind": "xrlenv-cluster"}
        if cb_cfg.xrlenv_grpc_host is not None:
            block["grpc_host"] = cb_cfg.xrlenv_grpc_host
        if cb_cfg.xrlenv_grpc_port is not None:
            block["grpc_port"] = cb_cfg.xrlenv_grpc_port
        if cb_cfg.xrlenv_grpc_secure is not None:
            block["grpc_secure"] = cb_cfg.xrlenv_grpc_secure
        return block
    raise ValueError(
        f"unknown runtime_kind {cb_cfg.runtime_kind!r}; valid: "
        f"{list(_VALID_RUNTIME_KINDS)}"
    )


def write_config(config: dict[str, Any], dest: Path) -> Path:
    """Serialise a config dict to YAML at ``dest`` and return it."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return dest


# ──────────────────────────────────────────────────────────────────────
# Public API — mirrors eval_runner
# ──────────────────────────────────────────────────────────────────────
def run_full(
    *,
    config_path: Path | None = None,
    cwd: Path | None = None,
    subset: str,
    task_names: list[str],
    job_name: str | None = None,
    extra_args: list[str] | None = None,
    extra_env: dict[str, str] | None = None,
    tee_log_path: Path | None = None,
    cb_cfg: CodingBenchEvalConfig | None = None,
) -> EvalResult:
    """coding-bench analogue of :func:`eval_runner.run_full`.

    ``config_path`` / ``cwd`` are accepted for signature compatibility
    with the pristine seam but are advisory here: coding-bench drives
    everything off the generated config + REPO_ROOT, not a worktree
    ``cwd``. ``task_names == []`` means a full sweep.

    When ``cb_cfg`` is omitted, the static knobs (monet model/provider/
    turn budget) are derived from the self-evolve ``config_path`` YAML via
    :meth:`CodingBenchEvalConfig.from_self_evolve_config`, so the
    pipeline's per-iteration calls thread the campaign's configured model
    through unchanged.
    """
    return _run(
        subset=subset,
        task_names=task_names,
        job_name=job_name,
        cwd=cwd,
        extra_args=extra_args,
        extra_env=extra_env,
        tee_log_path=tee_log_path,
        cb_cfg=cb_cfg or CodingBenchEvalConfig.from_self_evolve_config(config_path),
    )


def run_subset(
    *,
    config_path: Path | None = None,
    cwd: Path | None = None,
    task_names: list[str],
    job_name: str | None = None,
    extra_env: dict[str, str] | None = None,
    tee_log_path: Path | None = None,
    cb_cfg: CodingBenchEvalConfig | None = None,
) -> EvalResult:
    """coding-bench analogue of :func:`eval_runner.run_subset` (mini-eval).

    As with :func:`run_full`, an omitted ``cb_cfg`` is derived from the
    self-evolve ``config_path`` YAML.
    """
    if not task_names:
        raise ValueError("run_subset requires a non-empty task_names list")
    return _run(
        subset=f"adhoc:{len(task_names)}tasks",
        task_names=task_names,
        job_name=job_name,
        cwd=cwd,
        extra_env=extra_env,
        tee_log_path=tee_log_path,
        cb_cfg=cb_cfg or CodingBenchEvalConfig.from_self_evolve_config(config_path),
        # Mini-evals stay single-attempt (matches the old harness, which only
        # applied multi-attempt to baselines/final evals). Best-of-N is for the
        # reported full-set score, not the inner-loop gate.
        allow_multi_attempt=False,
    )


def _per_task_rows_with_trial_fallback(run_dir: Path) -> list[dict[str, Any]]:
    """Return the run's ``per_task_results`` rows, reconstructing them from the
    per-trial ``result.json`` files when ``run.json`` is missing or unreadable.

    Harbor occasionally fails to assemble the top-level ``run.json`` (a transient
    cluster/gateway hiccup) even though every per-trial ``raw/trials/*/result.json``
    was written. Without this fallback a missing ``run.json`` makes the caller
    (equivalence probe / mini-eval) raise, which the equivalence gate catches as
    "probe execution failed" and then **fails open** — silently letting a
    regressing candidate through. Reconstructing from the surviving trial files
    keeps the preservation check honest. The synthesized rows expose the same
    keys (``task_id``/``reward``/``error``/``resolved``) the aggregation loop and
    :func:`_row_is_agent_timeout` rely on.
    """
    import re
    run_dir = Path(run_dir)
    try:
        raw = json.loads((run_dir / "run.json").read_text())
        rows = raw.get("per_task_results")
        if rows:
            return rows
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        pass
    rows: list[dict[str, Any]] = []
    for p in sorted(run_dir.glob("raw/trials/*/result.json")):
        try:
            d = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        task_name = str(d.get("task_name") or "")
        if task_name:
            base = task_name.split("/")[-1]
        else:
            trial_name = str(d.get("trial_name") or p.parent.name)
            base = re.sub(r"__s\d+$", "", re.sub(r"^[0-9a-fA-F]+__", "", trial_name))
        reward = ((d.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        meta = (d.get("agent_result") or {}).get("metadata") or {}
        err = d.get("exception_info") or meta.get("error") or meta.get("stream_error")
        rows.append({
            "task_id": base,
            "reward": reward,
            "error": (str(err) if err else None),
            "resolved": (float(reward) >= 1.0) if reward is not None else False,
        })
    return rows


def _rows_to_samples(rows, *, absorb_timeouts, absorb_transient):
    """Pure: per-trial rows -> {base_task: [0/1,...]}. ``absorb_timeouts`` MUST be
    False for any SCORED eval (budget-timeout = real fail per protocol); only the
    mini-eval search screen may absorb. (legitimacy audit 2026-07-02)"""
    import re
    samples: dict[str, list[float]] = {}
    for row in rows:
        tid = str(row.get("task_id"))
        base = re.sub(r"__s\d+$", "", tid)
        reward = row.get("reward")
        if reward is None:
            if row.get("error"):
                continue
            reward = 1.0 if row.get("resolved") else 0.0
        if absorb_transient and _row_is_transient_infra(row):
            continue
        if absorb_timeouts and _row_is_agent_timeout(row):
            continue
        samples.setdefault(base, []).append(1.0 if float(reward) >= 1.0 else 0.0)
    return samples


def _aggregate_task_samples(run_dir: Path, *, absorb_timeouts=None, absorb_transient=None) -> dict[str, list[float]]:
    """Group a run's per-trial rewards by base task id (see _rows_to_samples).
    Params default to the env-driven policy; the SCORED full-avg@k path passes
    explicit False so a budget-timeout counts as a real failure."""
    if absorb_timeouts is None:
        absorb_timeouts = _absorb_contention_timeouts()
    if absorb_transient is None:
        absorb_transient = _absorb_transient_infra()
    return _rows_to_samples(
        _per_task_rows_with_trial_fallback(Path(run_dir)),
        absorb_timeouts=absorb_timeouts, absorb_transient=absorb_transient,
    )

@dataclass(frozen=True)
class SampledEvalResult:
    """Result of a denoised avg@k eval over a task subset."""
    rates: dict[str, tuple[float, int]]   # task -> (pass_rate, n_samples)
    samples: dict[str, list[float]]       # task -> per-sample 0/1 outcomes
    job_dir: Path


def run_subset_sampled(
    *,
    config_path: Path | None = None,
    cwd: Path | None = None,
    task_names: list[str],
    k_samples: int,
    job_name: str | None = None,
    extra_env: dict[str, str] | None = None,
    tee_log_path: Path | None = None,
    cb_cfg: CodingBenchEvalConfig | None = None,
    absorb_timeouts: bool | None = None,
    absorb_transient: bool | None = None,
) -> SampledEvalResult:
    """Denoised avg@k probe: run each task ``k_samples`` times in one pass and
    return per-task pass-rates + raw samples + the run dir.

    This is the variance-aware primitive the 3-stage cascade uses for the
    mini-eval screen and the equivalence-gate probes, replacing the k=1
    ``run_subset`` whose single-sample verdicts were dominated by noise. Reuses
    all of :func:`_run`'s machinery (candidate-commit injection, Express relay,
    cluster creds) by routing through a ``num_samples=k`` config and a single
    pass (``allow_multi_attempt=False``), then re-aggregating the raw trials.
    """
    if not task_names:
        raise ValueError("run_subset_sampled requires a non-empty task_names list")
    if k_samples < 1:
        raise ValueError(f"k_samples must be >= 1, got {k_samples}")
    base_cfg = cb_cfg or CodingBenchEvalConfig.from_self_evolve_config(config_path)
    res = _run(
        subset=f"avgk:{len(task_names)}tasks@k{k_samples}",
        task_names=task_names,
        job_name=job_name,
        cwd=cwd,
        extra_env=extra_env,
        tee_log_path=tee_log_path,
        cb_cfg=replace(base_cfg, num_samples=k_samples),
        allow_multi_attempt=False,
    )
    samples = _aggregate_task_samples(
        res.job_dir, absorb_timeouts=absorb_timeouts, absorb_transient=absorb_transient,
    )
    rates = {t: (sum(v) / len(v), len(v)) for t, v in samples.items() if v}
    return SampledEvalResult(rates=rates, samples=samples, job_dir=Path(res.job_dir))


def run_full_avg_k(
    *,
    config_path: Path | None = None,
    cwd: Path | None = None,
    task_names: list[str],
    k_samples: int,
    job_name: str | None = None,
    extra_env: dict[str, str] | None = None,
    tee_log_path: Path | None = None,
    cb_cfg: CodingBenchEvalConfig | None = None,
) -> EvalResult:
    """Full-set eval scored as **avg@k** (mean reward over k samples per task),
    the metric the public TB2.1 leaderboard reports — as opposed to best-of-N
    (pass@N), which over-reports. Builds a per-task-mean ``run.json`` shape and
    lifts it through :func:`_build_eval_result` so downstream scoring /
    solved-task classification is identical to the single-pass path.
    """
    sres = run_subset_sampled(
        config_path=config_path, cwd=cwd, task_names=task_names,
        k_samples=k_samples, job_name=job_name, extra_env=extra_env,
        tee_log_path=tee_log_path, cb_cfg=cb_cfg,
        absorb_timeouts=False,  # SCORED eval: budget-timeout = fail
    )
    rows = [
        {"task_id": t, "reward": rate, "resolved": rate >= 1.0}
        for t, (rate, _n) in sres.rates.items()
    ]
    merged_raw = {
        "per_task_results": rows,
        "errors": [],
        "totals": {
            "num_tasks": len(rows),
            "num_tasks_resolved": sum(1 for r in rows if r["reward"] >= 1.0),
            "num_tasks_errored": 0,
        },
        "_avg_at_k": k_samples,
    }
    return _build_eval_result(
        merged_raw, sres.job_dir,
        subset=f"full@avg{k_samples}", task_names=list(task_names),
    )


# ──────────────────────────────────────────────────────────────────────
# Provider unification — Express local-proxy → trial-container reachability
# ──────────────────────────────────────────────────────────────────────
# self-evolve's live campaign drives monet through the Express LLM
# gateway via a loopback reverse tunnel on the host
# (``LLM_GATEWAY_EXPRESS_LOCAL_PROXY_URL=http://127.0.0.1:<port>/``). A
# Harbor trial container can't reach the host's loopback — ``127.0.0.1``
# inside the container is the container itself. monet_code_eval's
# ``scripts/run_harbor.py`` solves this by standing up a tiny TCP relay
# bound to a docker-reachable host address (the docker bridge gateway)
# that forwards to the loopback proxy, then rewriting the proxy URL it
# forwards into containers. We port that bootstrap here verbatim so the
# coding-bench seam reaches the SAME gateway the campaign already uses.


class _ThreadingTcpServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def _is_loopback_host(hostname: str | None) -> bool:
    return hostname in {"127.0.0.1", "localhost", "::1"}


def _docker_bridge_gateway_ip() -> str | None:
    try:
        result = subprocess.run(
            [
                "docker", "network", "inspect", "bridge", "--format",
                "{{range .IPAM.Config}}{{if .Gateway}}{{println .Gateway}}"
                "{{end}}{{end}}",
            ],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=5, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        gateway = line.strip()
        try:
            ip = ipaddress.ip_address(gateway)
        except ValueError:
            continue
        if ip.version == 4:
            return gateway
    return None


def _docker_reachable_host_ip() -> str:
    override = os.environ.get("MONET_EVAL_DOCKER_HOST_PROXY_HOST")
    if override:
        return override
    docker_gateway = _docker_bridge_gateway_ip()
    if docker_gateway:
        return docker_gateway
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def _relay_public_url(source_url: str, *, public_host: str, public_port: int) -> str:
    parsed = urlparse(source_url)
    netloc = f"{public_host}:{public_port}"
    path = parsed.path or "/"
    return urlunparse(("http", netloc, path, "", parsed.query, ""))


def _start_tcp_relay(*, upstream_host: str, upstream_port: int) -> _ThreadingTcpServer:
    class Handler(socketserver.BaseRequestHandler):
        def handle(self) -> None:
            try:
                upstream = socket.create_connection(
                    (upstream_host, upstream_port), timeout=10,
                )
            except OSError:
                return
            with upstream:
                sockets = [self.request, upstream]
                while True:
                    readable, _, _ = select.select(sockets, [], [], 60)
                    if not readable:
                        continue
                    for src in readable:
                        try:
                            data = src.recv(65536)
                        except OSError:
                            return
                        if not data:
                            return
                        dst = upstream if src is self.request else self.request
                        try:
                            dst.sendall(data)
                        except OSError:
                            return

    server = _ThreadingTcpServer(("0.0.0.0", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _prepare_express_proxy_relay(
    env: dict[str, str], *, url_env: str,
) -> _ThreadingTcpServer | None:
    """Make a loopback Express proxy URL reachable from trial containers.

    Reads ``env[url_env]`` (the host-side
    ``LLM_GATEWAY_EXPRESS_LOCAL_PROXY_URL``). If it points at the host's
    loopback, stand up a TCP relay on a docker-reachable host address and
    rewrite ``env[url_env]`` in place to the relay's URL. Returns the
    server so the caller can close it after the run (None when no relay
    was needed, e.g. the URL is already a routable host/IP).
    """
    configured = env.get(url_env)
    if not configured:
        return None
    parsed = urlparse(configured)
    if parsed.scheme != "http" or not _is_loopback_host(parsed.hostname):
        return None
    relay = _start_tcp_relay(
        upstream_host=parsed.hostname or "127.0.0.1",
        upstream_port=parsed.port or 80,
    )
    public_url = _relay_public_url(
        configured,
        public_host=_docker_reachable_host_ip(),
        public_port=relay.server_address[1],
    )
    env[url_env] = public_url
    print(
        "[codingbench_eval] exposing loopback Express local proxy to "
        f"trial containers as {public_url}",
        file=sys.stderr,
    )
    return relay


# ──────────────────────────────────────────────────────────────────────
# Candidate-commit injection — make a child's evolved commit clonable
# ──────────────────────────────────────────────────────────────────────
# THE PHASE-2 #1 BLOCKER FIX.
#
# self_evolve produces a candidate as a *local* commit in a per-pipeline
# ``<eval_dir>/monet_code`` git worktree. ``MonetHarborAgent`` REQUIRES a
# ``git_clone`` source and ``git clone --depth 1 --branch <ref> <url>``s
# monet *inside the trial container* — so a local-only commit is not
# clonable and the seam fell back to ``develop`` (DEFAULT_MONET_REF),
# making every child evaluate the BASELINE → ``no_change`` forever.
#
# Fix (preferred approach from PHASE1_NOTES decision #1.a — "throwaway
# remote branch"): before each eval, push the worktree's HEAD commit to a
# short-lived branch on the configured monet repo, point
# ``agent_source.ref`` at that branch, run the eval, then delete the
# branch. The push target is ``cb_cfg.monet_repo_url`` (=
# ``DEFAULT_MONET_REPO_URL`` — the OWNED ``yifan-zhang_sfemu`` fork), NOT
# the worktree submodule's ``origin`` remote. Deriving it from the cfg
# guarantees the candidate ref lands on the SAME repo the trial container
# clones from, and only depends on push access to a repo we own (so the
# submodule's ``origin`` can stay pointed at whatever it was cloned from
# — it does not need re-pointing). ``git clone --branch`` resolves branch
# names (under ``refs/heads/*``) — NOT bare SHAs or custom
# ``refs/evolve/*`` namespaces — so the throwaway MUST be a real branch.
# Gated + degrades gracefully: any failure (no cwd, no monet_code repo,
# push error) silently falls back to the configured default ref,
# preserving the prior behaviour rather than crashing the campaign.

# Set to "0"/"false"/"no" to disable injection (fall back to the default
# published ref — the pre-fix behaviour).
_CANDIDATE_INJECTION_ENV = "SELF_EVOLVE_CANDIDATE_COMMIT_INJECTION"
# Throwaway branch namespace on the monet origin. Cleaned up after each
# eval; the prefix lets a stray-branch sweep find leftovers.
_CANDIDATE_BRANCH_PREFIX = "evolve-eval"


@dataclass(frozen=True)
class _CandidateRef:
    """A pushed throwaway branch the trial container can clone."""

    branch: str
    repo_url: str
    sha: str
    monet_dir: Path


def _candidate_injection_enabled() -> bool:
    val = os.environ.get(_CANDIDATE_INJECTION_ENV, "1").strip().lower()
    return val not in {"0", "false", "no", "off"}


def _git(args: list[str], *, cwd: Path, timeout: int = 300) -> subprocess.CompletedProcess:
    """Run git inheriting the full env (so the supervisor's
    ``GIT_CONFIG_*`` gh-credential-helper wiring authenticates pushes)."""
    return subprocess.run(
        ["git", *args], cwd=str(cwd), text=True,
        capture_output=True, timeout=timeout, check=False,
    )


def _sanitize_branch_token(token: str) -> str:
    safe = "".join(c if (c.isalnum() or c in "-_.") else "-" for c in token)
    return safe.strip("-.") or "eval"


def _publish_candidate_commit(
    *, monet_dir: Path, job_name: str | None, repo_url: str,
) -> _CandidateRef | None:
    """Push the worktree's HEAD to a throwaway branch on ``repo_url``.

    ``repo_url`` is the configured monet repo (``cb_cfg.monet_repo_url`` =
    the owned fork), pushed to BY URL rather than via the submodule's
    ``origin`` remote name — so the candidate ref lands on the same repo
    the trial container clones from and the submodule remote is irrelevant.

    Returns the :class:`_CandidateRef` to clone, or ``None`` if anything
    prevents publishing (caller then falls back to the default ref).
    """
    head = _git(["rev-parse", "HEAD"], cwd=monet_dir, timeout=30)
    if head.returncode != 0:
        return None
    sha = (head.stdout or "").strip()
    if not sha:
        return None
    if not repo_url:
        return None
    token = _sanitize_branch_token(job_name or "eval")
    branch = f"{_CANDIDATE_BRANCH_PREFIX}/{token}-{sha[:12]}-{os.getpid()}"
    # --force so a re-run reusing the same branch name (same job_name +
    # sha + pid) overwrites cleanly rather than failing non-fast-forward.
    # Push by URL (not the ``origin`` remote) so the target is the owned
    # fork regardless of what the worktree's submodule remote points at.
    push = _git(
        ["push", "--force", repo_url, f"{sha}:refs/heads/{branch}"],
        cwd=monet_dir, timeout=300,
    )
    if push.returncode != 0:
        print(
            "[codingbench_eval] candidate-commit injection: could not push "
            f"{sha[:12]} to {repo_url} ({(push.stderr or '').strip()[:200]}); "
            "falling back to default monet ref",
            file=sys.stderr,
        )
        return None
    print(
        f"[codingbench_eval] candidate-commit injection: evaluating {sha[:12]} "
        f"via throwaway branch {branch} on {repo_url}",
        file=sys.stderr,
    )
    return _CandidateRef(branch=branch, repo_url=repo_url, sha=sha, monet_dir=monet_dir)


def _delete_candidate_ref(ref: _CandidateRef) -> None:
    """Best-effort delete of the throwaway branch after the eval."""
    try:
        r = _git(
            ["push", ref.repo_url, "--delete", ref.branch],
            cwd=ref.monet_dir, timeout=120,
        )
        if r.returncode != 0:
            print(
                "[codingbench_eval] candidate-commit injection: cleanup of "
                f"{ref.branch} failed ({(r.stderr or '').strip()[:200]})",
                file=sys.stderr,
            )
    except (OSError, subprocess.SubprocessError) as e:
        print(
            f"[codingbench_eval] candidate-commit injection: cleanup of "
            f"{ref.branch} raised {e!r}",
            file=sys.stderr,
        )


def _maybe_inject_candidate_commit(
    *, cwd: Path | None, job_name: str | None, cb_cfg: CodingBenchEvalConfig,
) -> tuple[CodingBenchEvalConfig, _CandidateRef | None]:
    """If a monet worktree lives under ``cwd``, publish its HEAD commit as
    a clonable throwaway branch and rewrite ``cb_cfg`` to point at it.

    Returns ``(cb_cfg, candidate_ref)``. ``candidate_ref`` is ``None`` when
    injection is disabled/unavailable, in which case ``cb_cfg`` is returned
    unchanged (default-ref fallback).
    """
    if not _candidate_injection_enabled() or cwd is None:
        return cb_cfg, None
    monet_dir = Path(cwd) / "monet_code"
    # ``.git`` is a *file* in a linked worktree and a dir in a plain clone;
    # ``exists()`` covers both.
    if not (monet_dir / ".git").exists():
        return cb_cfg, None
    ref = _publish_candidate_commit(
        monet_dir=monet_dir, job_name=job_name, repo_url=cb_cfg.monet_repo_url,
    )
    if ref is None:
        # Candidate worktree EXISTS but HEAD could not be published; silently
        # scoring DEFAULT_MONET_REF would misreport the baseline as this candidate.
        # Fail loud. (legitimacy audit 2026-07-02)
        raise RuntimeError(
            "candidate-commit injection failed: could not publish HEAD of "
            f"{monet_dir} (repo_url={cb_cfg.monet_repo_url!r}). Refusing to "
            f"silently evaluate the '{DEFAULT_MONET_REF}' baseline as the candidate."
        )
    return replace(cb_cfg, monet_ref=ref.branch, monet_repo_url=ref.repo_url), ref


# ──────────────────────────────────────────────────────────────────────
# Best-of-N (multi-attempt) scoring
# ──────────────────────────────────────────────────────────────────────
# The colleague's 0.775 reference was measured best-of-2 (Harbor's
# ``final_eval_n_attempts: 2``): each task gets up to 2 attempts and counts
# RESOLVED if ANY attempt passes. coding-bench's TB2 loader yields exactly
# one trial per task (``num_samples`` is not honored by the TB2 path, and
# duplicating a task within one run collides on the per-run container
# namespace), so we implement best-of-N at the seam by running the eval in
# up to N passes and merging per task by MAX reward (resolved-if-any).
#
# Optimisation: pass ``p>=2`` re-runs ONLY the tasks not yet resolved. For
# the resolved/MAX-reward outcome this is provably identical to running every
# task N times — a task already at reward 1.0 cannot improve, and a task at
# reward r<1.0 is exactly the one a fresh attempt might lift. This makes
# best-of-2 cost ~1.4x a single pass instead of 2x.
#
# Attempts are read from ``SELF_EVOLVE_EVAL_ATTEMPTS`` (campaign-wide) and
# from the pipeline's ``--n-attempts N`` final-eval flag (whichever is
# larger). run_full (bootstrap baseline + final evals) honors it; run_subset
# (mini-evals) always uses 1 attempt, matching the old harness.
_EVAL_ATTEMPTS_ENV = "SELF_EVOLVE_EVAL_ATTEMPTS"


def _eval_attempts(env: dict[str, str], extra_args: list[str] | None) -> int:
    n = 1
    raw = (env.get(_EVAL_ATTEMPTS_ENV) or "").strip()
    if raw.isdigit():
        n = max(n, int(raw))
    args = list(extra_args or [])
    for i, a in enumerate(args):
        val: str | None = None
        if a in ("--n-attempts", "--n_attempts") and i + 1 < len(args):
            val = args[i + 1]
        elif a.split("=", 1)[0] in ("--n-attempts", "--n_attempts") and "=" in a:
            val = a.split("=", 1)[1]
        if val is not None:
            try:
                n = max(n, int(val))
            except ValueError:
                pass
    return max(1, n)


def _row_effective_reward(row: dict[str, Any]) -> float:
    r = row.get("reward")
    if r is None:
        return 1.0 if row.get("resolved") else 0.0
    try:
        return float(r)
    except (TypeError, ValueError):
        return 0.0


def _row_is_resolved(row: dict[str, Any] | None) -> bool:
    """A task counts resolved when an attempt scored >=1.0 with no error."""
    if row is None:
        return False
    return _row_effective_reward(row) >= 1.0 and not row.get("error")


def _merge_best_row(prev: dict[str, Any] | None, new: dict[str, Any]) -> dict[str, Any]:
    """Keep the better of two attempts for the same task (best-of-N).

    A clean resolved row always wins; otherwise the higher effective reward
    wins; a no-error row breaks ties over an errored one.
    """
    if prev is None:
        return new
    if _row_is_resolved(new) and not _row_is_resolved(prev):
        return new
    if _row_is_resolved(prev) and not _row_is_resolved(new):
        return prev
    pr, nr = _row_effective_reward(prev), _row_effective_reward(new)
    if nr > pr:
        return new
    if nr == pr and prev.get("error") and not new.get("error"):
        return new
    return prev


def _grade_swebench_and_rewrite(run_dir, cb_cfg) -> None:
    """SWE-bench Verified is 2-phase: ``runner.run`` only writes ``predictions.json``
    and its inline ``resolved`` is trajectory-completion, NOT test-pass. Run the
    official grader and rewrite each per-task ``reward``/``resolved`` in run.json with
    the REAL graded verdict, so the self_evolve loop optimizes test-pass. Scoped to
    ``swe-bench-verified`` — a byte-for-byte no-op for every other benchmark.
    (Assumes pass@1: one patch per instance in predictions.json.)"""
    import re as _re
    if (getattr(cb_cfg, "benchmark_name", "") or "").strip() != "swe-bench-verified":
        return
    run_dir = Path(run_dir)
    preds = run_dir / "predictions.json"
    if not preds.exists():
        return
    from benchmarks.swe_bench_verified import grading as _swev_grading
    runtime_kind = (os.environ.get("SELF_EVOLVE_EVAL_RUNTIME") or "xrlenv-cluster").strip()
    try:
        nw = int(os.environ.get("MONET_EVAL_HARBOR_N_CONCURRENT", "6"))
    except ValueError:
        nw = 6
    overall = _swev_grading.grade(
        predictions_path=preds, run_dir=run_dir,
        runtime_kind=runtime_kind, num_workers=nw,
        dataset=getattr(cb_cfg, "dataset", None),
    )
    resolved = json.loads((Path(overall).parent / "eval_results.json").read_text())
    rj = run_dir / "run.json"
    try:
        d = json.loads(rj.read_text())
    except (OSError, json.JSONDecodeError):
        return
    for row in d.get("per_task_results") or []:
        tid = _re.sub(r"__s\d+$", "", str(row.get("task_id") or ""))
        if tid in resolved:
            r = 1.0 if resolved[tid] else 0.0
            row["reward"] = r
            row["resolved"] = bool(resolved[tid])
            vr = row.get("verifier_result") or {}
            rw = vr.get("rewards") or {}
            rw["reward"] = r
            vr["rewards"] = rw
            row["verifier_result"] = vr
    rj.write_text(json.dumps(d))


def _run(
    *,
    subset: str,
    task_names: list[str],
    job_name: str | None,
    cwd: Path | None = None,
    extra_args: list[str] | None = None,
    extra_env: dict[str, str] | None,
    tee_log_path: Path | None,
    cb_cfg: CodingBenchEvalConfig,
    allow_multi_attempt: bool = True,
) -> EvalResult:
    results_root = Path(cb_cfg.results_root)
    # Per-call results-root isolation. FIX for the chronic ~27% "mini-eval
    # failed: [Errno 2] ... run.json" race: multiple workers run eval subprocesses
    # CONCURRENTLY into a shared results_root/runs, so the before/after snapshot
    # diff could select ANOTHER worker's half-written run dir and crash on
    # json.loads(run_dir/"run.json"). A private runs dir per call fixes it.
    import uuid as _uuid_iso
    results_root = results_root / "_iso" / _uuid_iso.uuid4().hex
    runs_dir = results_root / "runs"

    # Fail loud if cluster mode is selected without reachable cluster creds.
    # xrlenv.from_env() SILENTLY degrades to local Docker when no gRPC host
    # is configured — for a self-evolve campaign that means "I asked for the
    # fast cluster but quietly got the slow single-node path". Refuse that.
    _require_cluster_creds(cb_cfg)

    # Make the child's evolved commit (the worktree HEAD under ``cwd``)
    # clonable by the trial container; falls back to the default ref when
    # unavailable. ``candidate`` is cleaned up in the ``finally`` below.
    cb_cfg, candidate = _maybe_inject_candidate_commit(
        cwd=cwd, job_name=job_name, cb_cfg=cb_cfg,
    )

    # Honor an eval-concurrency override (the pipeline's bootstrap lift +
    # any campaign-wide ``MONET_EVAL_HARBOR_N_CONCURRENT`` setting). Read
    # from the merged env (os.environ + extra_env) so an explicit per-call
    # extra_env wins. Default (env unset) preserves parallelism 1.
    merged_env = {**os.environ, **(extra_env or {})}
    parallelism_override = _eval_parallelism_override(merged_env)
    if parallelism_override is not None and parallelism_override != cb_cfg.parallelism:
        print(
            f"[codingbench_eval] eval concurrency set to {parallelism_override} "
            f"(from {_EVAL_PARALLELISM_ENV})",
            file=sys.stderr,
        )
        cb_cfg = replace(cb_cfg, parallelism=parallelism_override)

    cfg_dir = results_root / "_self_evolve_configs"

    env = {**os.environ, **(extra_env or {})}
    sanitized_extra = _sanitize_runner_extra_args(extra_args)
    attempts = _eval_attempts(merged_env, extra_args) if allow_multi_attempt else 1
    if attempts > 1:
        print(
            f"[codingbench_eval] best-of-{attempts} scoring: a task counts "
            "resolved if ANY attempt passes; pass 2+ re-runs only unresolved tasks",
            file=sys.stderr,
        )

    # Provider unification: relay the host-loopback Express proxy onto a
    # docker-reachable address and rewrite the URL the runner forwards into
    # trial containers. Started once, reused across all best-of-N passes.
    relay = _prepare_express_proxy_relay(
        env, url_env=cb_cfg.express_proxy_url_env,
    )

    # Per-task best row across attempts (best-of-N), in first-seen order.
    merged_rows: dict[str, dict[str, Any]] = {}
    task_order: list[str] = []
    # Top-level (harness-crash) error entries per task, latest pass wins.
    pending_errors: dict[str, dict[str, Any]] = {}
    first_run_dir: Path | None = None
    try:
        pass_task_names = list(task_names)  # [] == full sweep on pass 1
        for attempt in range(1, attempts + 1):
            config = build_codingbench_config(
                task_names=pass_task_names, cb_cfg=cb_cfg,
            )
            cfg_name = (
                f"{job_name or subset.replace(':', '_')}"
                f"{'' if attempt == 1 else f'_attempt{attempt}'}.yaml"
            )
            cfg_path = write_config(config, cfg_dir / cfg_name)
            before = _snapshot_runs(runs_dir)
            cmd = [
                sys.executable, "-m", "runner.run", str(cfg_path),
                "--results-root", str(results_root),
            ]
            if job_name:
                cmd += ["--campaign-id", job_name]
            cmd += sanitized_extra
            rc = _invoke_runner(cmd, env=env, tee_log_path=tee_log_path)
            after = _snapshot_runs(runs_dir)
            new = sorted(after - before)
            if not new:
                raise RuntimeError(
                    f"runner.run exited rc={rc} but no new run dir appeared "
                    f"under {runs_dir}; the run likely crashed before writing "
                    "run.json"
                )
            run_dir = runs_dir / new[-1]
            if first_run_dir is None:
                first_run_dir = run_dir
            _grade_swebench_and_rewrite(run_dir, cb_cfg)
            raw_p = json.loads((run_dir / "run.json").read_text())
            for row in raw_p.get("per_task_results") or []:
                tid = str(row.get("task_id"))
                if tid not in merged_rows:
                    task_order.append(tid)
                merged_rows[tid] = _merge_best_row(merged_rows.get(tid), row)
                # A produced row supersedes a prior harness-crash error.
                pending_errors.pop(tid, None)
            for e in raw_p.get("errors") or []:
                tid = str(e.get("task_id") or "")
                if tid and tid not in merged_rows:
                    pending_errors[tid] = e
                elif not tid:
                    # Global/harness crash with no task_id — keep for the
                    # whole-job abort check.
                    pending_errors.setdefault("", e)

            if attempt >= attempts:
                break
            # Re-run only the tasks not yet resolved (best-of-N optimisation).
            retry = [t for t in task_order if not _row_is_resolved(merged_rows.get(t))]
            # Tasks that only ever crashed at the harness level (no row yet)
            # are also worth one more attempt.
            retry += [t for t in pending_errors if t and t not in merged_rows]
            retry = list(dict.fromkeys(retry))
            if not retry:
                break
            print(
                f"[codingbench_eval] best-of-{attempts}: attempt {attempt} left "
                f"{len(retry)} task(s) unresolved; re-running them in attempt "
                f"{attempt + 1}",
                file=sys.stderr,
            )
            pass_task_names = retry
    finally:
        if relay is not None:
            relay.server_close()
        if candidate is not None:
            _delete_candidate_ref(candidate)

    if first_run_dir is None:
        raise RuntimeError("best-of-N eval produced no run dir")

    if attempts == 1:
        # Preserve the single-pass path exactly (read the on-disk run.json).
        return parse_run_json(first_run_dir, subset=subset, task_names=task_names)

    # Synthesise a merged run.json (per-task best across attempts) and score
    # it through the same path as a single run.
    n_resolved = sum(1 for t in task_order if _row_is_resolved(merged_rows[t]))
    n_errored = sum(1 for t in task_order if merged_rows[t].get("error"))
    merged_raw: dict[str, Any] = {
        "per_task_results": [merged_rows[t] for t in task_order],
        "errors": [pending_errors[t] for t in pending_errors],
        "totals": {
            "num_tasks": len(task_order),
            "num_tasks_resolved": n_resolved,
            "num_tasks_errored": n_errored,
        },
        "_best_of": attempts,
    }
    return _build_eval_result(
        merged_raw, first_run_dir, subset=subset, task_names=task_names,
    )


def _snapshot_runs(runs_dir: Path) -> set[str]:
    if not runs_dir.is_dir():
        return set()
    return {p.name for p in runs_dir.iterdir() if p.is_dir()}


def _invoke_runner(
    cmd: list[str],
    *,
    env: dict[str, str],
    tee_log_path: Path | None,
) -> int:
    """Run ``python -m runner.run`` from REPO_ROOT, optionally teeing.

    Mirrors :func:`eval_runner._run_harbor_command`'s tee behaviour but
    without the post-result grace/kill (coding-bench's runner exits
    cleanly after writing run.json — there's no lingering Harbor
    viewer process to reap).
    """
    if tee_log_path is None:
        return subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=False).returncode

    tee_log_path.parent.mkdir(parents=True, exist_ok=True)
    with tee_log_path.open("ab") as log_f:
        proc = subprocess.Popen(
            cmd, cwd=str(REPO_ROOT), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0,
        )
        assert proc.stdout is not None
        for chunk in iter(lambda: proc.stdout.read(4096), b""):
            log_f.write(chunk)
            log_f.flush()
            buf = getattr(sys.stdout, "buffer", None)
            if buf is not None:
                buf.write(chunk)
                buf.flush()
        return proc.wait()


def parse_existing_result_dir(run_dir: Path) -> EvalResult:
    """Parse a coding-bench run dir's ``run.json`` without running anything.

    coding-bench analogue of :func:`eval_runner.parse_existing_result_dir`
    (which parsed a Harbor ``jobs/<ts>/result.json``).
    """
    return parse_run_json(Path(run_dir).resolve(), subset="full", task_names=[])


# ──────────────────────────────────────────────────────────────────────
# Per-task infrastructure-failure tolerance (full-set robustness fix)
# ──────────────────────────────────────────────────────────────────────
# On the full TB2 set, a handful of tasks (e.g. ``mteb-retrieve``) error as
# Harbor "infrastructure failures" because they need network the trial
# containers don't have. The pristine behaviour raised
# :class:`EvalInfrastructureError` on the FIRST such error, aborting the
# entire baseline/final eval — which made any full-set campaign impossible
# to complete and confounded an A/B (one arm aborting, the other not).
#
# With this flag ON (the default), a PER-TASK infra failure is tolerated:
# the offending task is recorded as unrunnable and EXCLUDED from the score's
# denominator (it counts neither for nor against the arm), and the eval
# CONTINUES. We still raise on a WHOLE-JOB infra failure — a global error
# with no ``task_id`` (the harness itself crashed), or the degenerate case
# where every produced task row is infra-broken so there is nothing
# scoreable left. ``_run`` already raises when no ``run.json`` appears at
# all, which is the other whole-job-failure mode.
#
# Set ``SELF_EVOLVE_TOLERATE_TASK_INFRA_FAILURE=0`` to restore the strict
# "abort on any infra failure" behaviour.
_TOLERATE_TASK_INFRA_ENV = "SELF_EVOLVE_TOLERATE_TASK_INFRA_FAILURE"


def _tolerate_task_infra_failures() -> bool:
    val = os.environ.get(_TOLERATE_TASK_INFRA_ENV, "1").strip().lower()
    return val not in {"0", "false", "no", "off"}


# ── Transient gateway/tunnel error absorption ─────────────────────────────
# Unlike the Docker-network INFRASTRUCTURE_FAILURE_PATTERNS (which surface in
# run.json ``errors[]``), transient connectivity failures show up as a normal
# trial row with ``reward=0`` and an ``error`` string like
# ``"monet exited rc=1: Error: fetch failed"`` — i.e. the agent-under-test
# could not reach the LLM gateway (e.g. the laptop Express tunnel blipped).
# These are NOT genuine task failures and must NOT be scored as reward-0, or a
# 2-minute tunnel drop silently depresses the whole eval (observed: 79/139
# trials scored 0 during a Cursor-update reconnect). We detect them by error
# substring and EXCLUDE such trials from the score denominator (a sibling
# attempt / sample that ran clean still counts; if *every* trial is transient-
# infra we abort as a whole-job failure). Toggle off with
# ``ATELIER_ABSORB_TRANSIENT_INFRA=0``.
_TRANSIENT_INFRA_SUBSTRINGS = (
    "fetch failed",
    "econnrefused", "econnreset", "etimedout", "eai_again",
    "socket hang up",
    "connection refused", "connection reset", "connection timed out",
    "network is unreachable", "could not connect", "failed to fetch",
    "502 bad gateway", "503 service unavailable", "504 gateway timeout",
    # bare "429"/"rate limit"/"too many requests" REMOVED (legitimacy audit
    # 2026-07-02): agent-controllable free-text (gaming vector) + coincidental
    # matches. Genuine gateway rate-limits surface as a real 0 / external re-run.
)


def _absorb_transient_infra() -> bool:
    val = os.environ.get("ATELIER_ABSORB_TRANSIENT_INFRA", "1").strip().lower()
    return val not in {"0", "false", "no", "off"}


# ── Agent-timeout (contention) absorption ─────────────────────────────────
# An ``AgentTimeoutError`` (the agent ran past the task's declared budget)
# surfaces as a normal trial row with ``reward=0`` and an ``error`` like
# ``"AgentTimeoutError: Agent execution timed out after 900.0 seconds"``. Some
# of these are GENUINE (a hard/slow task), but under a CONTENDED shared cluster
# the same task runs slower and times out spuriously — and a single timed-out
# sample on a tiny mini-eval set poisons the gradient (counts as a hard 0).
# When ``MONET_EVAL_ABSORB_TIMEOUTS=1`` we drop timeout trials from the score
# denominator (a clean sibling sample still counts; if EVERY trial timed out we
# abort as a whole-job infra failure rather than report a meaningless 0.0). This
# is DEFAULT OFF so the trustworthy ground-truth eval (run clean/serial on an
# isolated cluster group) keeps the specialist-comparable as-measured semantics
# (genuine timeouts = 0); turn it ON only for contention-prone campaign
# mini-evals where the timeout is far more likely a scheduling artifact.
_AGENT_TIMEOUT_SUBSTRINGS = (
    "agenttimeouterror",
    "agent execution timed out",
    "timed out after",
)


def _absorb_contention_timeouts() -> bool:
    val = os.environ.get("MONET_EVAL_ABSORB_TIMEOUTS", "0").strip().lower()
    return val in {"1", "true", "yes", "on"}


def _row_is_agent_timeout(row: dict[str, Any]) -> bool:
    """True iff this trial row is an agent-timeout failure (reward<1 with a
    'timed out' error string) rather than a graded wrong-answer."""
    err = row.get("error")
    if not err:
        return False
    try:
        r = float(row.get("reward") if row.get("reward") is not None
                  else (1.0 if row.get("resolved") else 0.0))
    except (TypeError, ValueError):
        r = 0.0
    if r >= 1.0:
        return False  # passed despite a noted slowness — keep it
    blob = str(err).lower()
    return any(s in blob for s in _AGENT_TIMEOUT_SUBSTRINGS)


def _row_is_transient_infra(row: dict[str, Any]) -> bool:
    """True iff this trial row is a transient gateway/tunnel failure (reward<1
    with a connectivity error string) rather than a genuine task failure."""
    err = row.get("error")
    if not err:
        return False
    try:
        r = float(row.get("reward") if row.get("reward") is not None
                  else (1.0 if row.get("resolved") else 0.0))
    except (TypeError, ValueError):
        r = 0.0
    if r >= 1.0:
        return False  # it passed despite a noted error — keep it
    blob = str(err).lower()
    return any(s in blob for s in _TRANSIENT_INFRA_SUBSTRINGS)


def _classify_infra_errors(raw: dict[str, Any]) -> tuple[set[str], bool]:
    """Split ``run.json`` ``errors[]`` into per-task vs whole-job infra failures.

    Returns ``(per_task_infra_ids, has_global_infra)``:
      * ``per_task_infra_ids`` — task ids whose error matched an
        :data:`INFRASTRUCTURE_FAILURE_PATTERNS` substring and that carry a
        ``task_id`` (a single unrunnable task).
      * ``has_global_infra`` — True if an infra-matching error has NO
        ``task_id`` (the harness crashed globally → whole-job failure).
    """
    errors = raw.get("errors") or []
    per_task_ids: set[str] = set()
    has_global = False
    for e in errors:
        blob = f"{e.get('message', '')}\n{e.get('traceback', '')}".lower()
        if not any(p in blob for p in INFRASTRUCTURE_FAILURE_PATTERNS):
            continue
        tid = str(e.get("task_id") or "").strip()
        if tid:
            per_task_ids.add(tid)
        else:
            has_global = True
    return per_task_ids, has_global


# ──────────────────────────────────────────────────────────────────────
# Eval concurrency override
# ──────────────────────────────────────────────────────────────────────
# The pipeline lifts ``harbor.n_concurrent_bootstrap`` for the one-time
# baseline by exporting ``MONET_EVAL_HARBOR_N_CONCURRENT`` into the eval
# subprocess env. The pristine seam ignored that env (defaulting to
# parallelism 1, i.e. one container at a time), which made a full-set eval
# of ~90 tasks take many hours. Honor the same env here so the bootstrap
# lift works AND a campaign can set a sensible global eval concurrency.
_EVAL_PARALLELISM_ENV = "MONET_EVAL_HARBOR_N_CONCURRENT"


def _eval_parallelism_override(env: dict[str, str]) -> int | None:
    raw = (env.get(_EVAL_PARALLELISM_ENV) or "").strip()
    if not raw:
        return None
    try:
        n = int(raw)
    except ValueError:
        return None
    return n if n > 0 else None


# ``runner.run`` CLI flags that exist in the Harbor seam (which forwards to
# ``run_harbor.sh``) but are NOT understood by coding-bench's ``runner.run``
# argparse. The pipeline injects ``--n-attempts <N>`` for full-set final
# evals (``self_evolve.pipeline._final_eval_extra_args``); passing it through
# verbatim makes ``python -m runner.run`` exit non-zero with "unrecognized
# arguments", which the seam surfaces as "no new run dir" → every full-set
# final eval crashes. coding-bench's TB2 path is one trial per task
# (``benchmark.num_samples`` owns pass@k, set to 1 here), and parse_run_json
# scores at task level — so the Harbor attempt-count flag has no coding-bench
# analogue and is stripped. Stripping it also keeps children's final evals
# consistent with the one-trial-per-task bootstrap baseline they're measured
# against. ``--n-attempts`` takes a value, so drop the following token too.
_UNSUPPORTED_RUNNER_VALUE_FLAGS = ("--n-attempts", "--n_attempts")


def _sanitize_runner_extra_args(extra_args: list[str] | None) -> list[str]:
    """Drop Harbor-only flags ``runner.run`` would reject (e.g. ``--n-attempts``)."""
    args = list(extra_args or [])
    cleaned: list[str] = []
    dropped: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        # Handle both "--flag value" and "--flag=value" spellings.
        flag = a.split("=", 1)[0]
        if flag in _UNSUPPORTED_RUNNER_VALUE_FLAGS:
            if "=" in a:
                dropped.append(a)
                i += 1
            else:
                dropped.append(" ".join(args[i:i + 2]))
                i += 2
            continue
        cleaned.append(a)
        i += 1
    if dropped:
        print(
            "[codingbench_eval] dropping Harbor-only eval flag(s) not understood "
            f"by runner.run: {dropped} (the attempt count is honored separately "
            "via best-of-N passes — see _eval_attempts)",
            file=sys.stderr,
        )
    return cleaned


def parse_run_json(
    run_dir: Path,
    *,
    subset: str,
    task_names: list[str],
) -> EvalResult:
    """Lift coding-bench's ``run.json`` into self_evolve's :class:`EvalResult`.

    run.json shape (see runner.run._assemble_row)::

        {
          "totals": {"num_tasks": N, "num_tasks_resolved": R,
                     "num_tasks_errored": E, ...},
          "per_task_results": [
            {"task_id": "...", "resolved": bool, "reward": float|None,
             "error": str|None, ...}, ...
          ],
          "errors": [{"task_id": "...", "kind": "...", "message": "..."}],
          ...
        }

    Each per-task row becomes one entry in ``rewards_per_task`` keyed by
    ``task_id`` (TB2 task ids carry no ``__<hash>`` suffix, so they're
    already task-base names). ``score`` is the mean reward across rows;
    a row's reward falls back to ``1.0 if resolved else 0.0`` when the
    verifier didn't emit a numeric reward.

    Per-task infrastructure failures (a task that can't run for infra
    reasons, e.g. missing network) are tolerated by default: the task is
    dropped from the score's denominator and the eval continues. See
    :func:`_tolerate_task_infra_failures`. Whole-job infra failures still
    raise :class:`EvalInfrastructureError`.
    """
    run_dir = Path(run_dir)
    run_json_path = run_dir / "run.json"
    if not run_json_path.is_file():
        raise FileNotFoundError(f"{run_dir} has no run.json")
    raw = json.loads(run_json_path.read_text())
    return _build_eval_result(raw, run_dir, subset=subset, task_names=task_names)


def _build_eval_result(
    raw: dict[str, Any],
    run_dir: Path,
    *,
    subset: str,
    task_names: list[str],
) -> EvalResult:
    """Lift a (possibly best-of-N merged) ``run.json`` dict into an EvalResult.

    Split out of :func:`parse_run_json` so the best-of-N path in :func:`_run`
    can synthesise a merged ``raw`` (per-task best across attempts) and reuse
    the exact same scoring + infra-tolerance logic.
    """
    run_dir = Path(run_dir)
    rows = raw.get("per_task_results") or []
    totals = raw.get("totals") or {}

    # ── Per-task infra-failure tolerance ──────────────────────────────
    # Decide BEFORE building per-task rewards whether to abort (whole-job
    # failure or strict mode) or to drop the unrunnable tasks and continue.
    infra_ids, infra_global = _classify_infra_errors(raw)
    excluded_infra_tasks: list[str] = []
    if infra_ids or infra_global:
        tolerate = _tolerate_task_infra_failures()
        run_task_ids = {str(r.get("task_id")) for r in rows}
        scoreable = run_task_ids - infra_ids
        # Abort (raise) when: strict mode, OR a global/harness crash, OR
        # there is nothing scoreable left (every produced row is infra-broken,
        # or no row was produced at all alongside infra errors).
        if (not tolerate) or infra_global or not scoreable:
            failures = _infrastructure_failures_from_raw(raw, run_dir)
            raise EvalInfrastructureError(
                _format_infrastructure_failure_message(run_dir, failures),
                job_dir=run_dir,
                failures=failures,
            )
        # Tolerate: drop the unrunnable task rows from the denominator. Record
        # every infra-classified task id (whether or not it managed to emit a
        # row) so the result is auditable.
        excluded_infra_tasks = sorted(infra_ids)
        rows = [r for r in rows if str(r.get("task_id")) not in infra_ids]
        print(
            "[codingbench_eval] per-task infra-failure tolerance: excluding "
            f"{sorted(infra_ids)} from the score denominator and continuing "
            f"({len(rows)} scoreable task rows remain in {run_dir.name})",
            file=sys.stderr,
        )

    # Drop transient gateway/tunnel-failure trials (reward<1 + connectivity
    # error) from the denominator so a tunnel blip can't masquerade as task
    # failures. A clean sibling sample/attempt of the same task still scores.
    excluded_transient: list[str] = []
    if _absorb_transient_infra():
        kept_rows = []
        for r in rows:
            if _row_is_transient_infra(r):
                excluded_transient.append(str(r.get("task_id")))
            else:
                kept_rows.append(r)
        if excluded_transient and not kept_rows:
            # Every trial was a transient-infra failure → whole-job tunnel
            # outage. Abort rather than report a meaningless 0.0.
            failures = _infrastructure_failures_from_raw(raw, run_dir)
            raise EvalInfrastructureError(
                f"all {len(excluded_transient)} trials failed with transient "
                f"gateway/tunnel errors (e.g. 'fetch failed') in {run_dir.name} "
                f"— treating as whole-job infra failure (tunnel down)",
                job_dir=run_dir,
                failures=failures,
            )
        if excluded_transient:
            print(
                "[codingbench_eval] transient-infra absorption: excluding "
                f"{len(excluded_transient)} tunnel/gateway-errored trial(s) "
                f"from the score denominator in {run_dir.name} "
                f"(e.g. {excluded_transient[:3]})",
                file=sys.stderr,
            )
        rows = kept_rows

    # Drop contention-suspect agent-timeout trials (reward<1 + 'timed out'
    # error) when MONET_EVAL_ABSORB_TIMEOUTS=1 — same rationale as the transient
    # absorption above, but for cluster-contention timeouts. Default off.
    excluded_timeout: list[str] = []
    if _absorb_contention_timeouts():
        kept_rows = []
        for r in rows:
            if _row_is_agent_timeout(r):
                excluded_timeout.append(str(r.get("task_id")))
            else:
                kept_rows.append(r)
        if excluded_timeout and not kept_rows:
            failures = _infrastructure_failures_from_raw(raw, run_dir)
            raise EvalInfrastructureError(
                f"all {len(excluded_timeout)} trials failed with agent timeouts "
                f"in {run_dir.name} — treating as whole-job infra failure "
                "(cluster likely starved); not reporting 0.0",
                job_dir=run_dir,
                failures=failures,
            )
        if excluded_timeout:
            print(
                "[codingbench_eval] contention-timeout absorption: excluding "
                f"{len(excluded_timeout)} agent-timeout trial(s) from the score "
                f"denominator in {run_dir.name} (e.g. {excluded_timeout[:3]})",
                file=sys.stderr,
            )
        rows = kept_rows

    per_task: dict[str, float] = {}
    trial_order: list[str] = []
    passing: list[str] = []
    failing: list[str] = []
    for row in rows:
        tid = str(row.get("task_id"))
        reward = row.get("reward")
        if reward is None:
            reward = 1.0 if row.get("resolved") else 0.0
        reward = float(reward)
        if tid not in per_task:
            trial_order.append(tid)
        per_task[tid] = reward
        if reward >= 1.0 and not row.get("error"):
            passing.append(tid)
        else:
            failing.append(tid)

    if per_task:
        score = sum(per_task.values()) / len(per_task)
    else:
        # No per-task rows — fall back to totals if present.
        n = int(totals.get("num_tasks") or 0)
        score = (int(totals.get("num_tasks_resolved") or 0) / n) if n else 0.0

    solved, unsolved, partial, outcomes, task_rewards = _classify_task_outcomes(
        per_task, trial_order,
    )

    # Stash the tolerated/excluded infra tasks so reports + reruns can audit
    # which tasks were treated as unrunnable on this eval (non-destructive —
    # ``raw`` is the in-memory parse, not the on-disk run.json).
    if excluded_infra_tasks:
        raw = {**raw, "_excluded_infra_tasks": excluded_infra_tasks}

    result = EvalResult(
        job_dir=run_dir,
        config_path=run_dir / "run.json",
        subset=subset,
        task_names=list(task_names),
        n_trials=len(rows) or int(totals.get("num_tasks") or 0),
        n_errors=int(totals.get("num_tasks_errored") or 0),
        score=score,
        passing_tasks=passing,
        failing_tasks=failing,
        rewards_per_task=per_task,
        solved_tasks=solved,
        unsolved_tasks=unsolved,
        partially_solved_tasks=partial,
        task_outcomes=outcomes,
        task_rewards=task_rewards,
        raw=raw,
    )

    # Any infra failure that survives to here is, by construction, a
    # tolerated per-task one (the abort path above already raised on
    # strict-mode / whole-job failures). Return the continued-eval result.
    return result


def _infrastructure_failures_from_raw(
    raw: dict[str, Any], job_dir: Path,
) -> list[dict[str, str]]:
    """Match ``run.json`` ``errors[]`` against the infra-failure substrings.

    Shared by :func:`infrastructure_failures` (result-based) and the
    abort path in :func:`parse_run_json` (raw-based, before a result
    exists). Each returned dict is one infra-classified error.
    """
    errors = raw.get("errors") or []
    failures: list[dict[str, str]] = []
    for e in errors:
        blob = f"{e.get('message', '')}\n{e.get('traceback', '')}".lower()
        matched = next(
            (p for p in INFRASTRUCTURE_FAILURE_PATTERNS if p in blob),
            None,
        )
        if matched is None:
            continue
        failures.append({
            "trial": str(e.get("task_id", "")),
            "exception": str(e.get("kind", "")),
            "pattern": matched,
            "trial_log": str(job_dir / "raw" / f"{e.get('task_id', '')}.log"),
        })
    return failures


def infrastructure_failures(result: EvalResult) -> list[dict[str, str]]:
    """coding-bench analogue of :func:`eval_runner.infrastructure_failures`.

    Harbor surfaced infra crashes as ``exception_stats`` + per-trial
    ``trial.log``; coding-bench surfaces them as top-level ``run.json``
    ``errors[]`` entries (the harbor_runner's ``except BaseException``
    path stamps ``kind``/``message``/``traceback``). We match the
    message/traceback against the same infra-failure substrings the
    pristine seam used so the pipeline's revert-on-infra logic keeps
    working unchanged.
    """
    return _infrastructure_failures_from_raw(result.raw, result.job_dir)


def _format_infrastructure_failure_message(
    run_dir: Path, failures: list[dict[str, str]],
) -> str:
    examples = ", ".join(f["trial"] for f in failures[:5])
    more = "" if len(failures) <= 5 else f", ... ({len(failures)} total)"
    return (
        f"coding-bench eval infrastructure failure in {run_dir}: "
        f"{examples}{more}. The score is invalid and must not be used."
    )
