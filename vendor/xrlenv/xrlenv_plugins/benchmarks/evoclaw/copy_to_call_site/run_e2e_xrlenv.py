#!/usr/bin/env python3
"""Faithful EvoClaw run entry on the xrlenv cluster.

Installs the in-process ``docker`` interceptor, registers the ``oracle`` agent,
relaxes ``run_e2e``'s ``--agent`` choices, then hands off to EvoClaw's
**own** ``harness.e2e.run_e2e.main()`` unchanged. EvoClaw's orchestrator / DAG /
evaluator run exactly as upstream; only the ``docker`` calls are rerouted.

Run from the EvoClaw checkout (so ``harness.*`` resolves) with the cluster + data
vars in the checkout's ``.env`` / ``.env_private`` (auto-loaded; shell wins).
``--workspace-root`` is **required** (never derived — it's where EvoClaw reads the
repo data AND writes trials). ``--srs-root`` (from ``--workspace-root``) and
``--image`` (from ``--repo-name``) are filled in when omitted::

    # .env: XRLENV_GRPC_HOST/PORT/CONSUMER_TOKEN, EVOCLAW_DATA_ROOT
    .venv/bin/python xrlenv_onboard/run_e2e_xrlenv.py \
        --agent oracle --model none --milestones 1 --force \
        --repo-name navidrome_navidrome_v0.57.0_v0.58.0 \
        --workspace-root "$EVOCLAW_DATA_ROOT/navidrome_navidrome_v0.57.0_v0.58.0"

See README.md for full setup. All other args are forwarded to ``run_e2e``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
# The EvoClaw checkout root (parent of xrlenv_onboard/) — same robust derivation
# scripts/run_all.py uses (Path(__file__).resolve().parent.parent). Per-user
# workspaces land under here by default; the golden cache does so only as a last
# resort, when no shared /fsx cache root is present (see
# _default_golden_cache_root). Both paths are overridable via CLI flags.
_PROJECT_ROOT = _HERE.parent


def _default_workspace_root_base() -> str:
    """Default per-user workspace base: ``<project-root>/results``.
    Overridden by ``--workspace-root-base``."""
    return str(_PROJECT_ROOT / "results")


# Shared golden-tar cache — migrated off the per-user ``<checkout>/golden_cache``
# so every EvoClaw user shares one cache AND the xrlenv sysbox real-bind
# allowlist (policy.allowed_host_paths) can point at a stable SHARED path rather
# than a personal home dir (which others can't read). The oracle mounts the
# per-task golden tars (a subdir under this root) read-only into the container.
#
# There is no default shared path — the shared filesystem is laid out differently
# per cluster — so set ``EVOCLAW_GOLDEN_CACHE_ROOT`` (below) to your shared cache
# root. With it unset the cache falls back to ``<project-root>/golden_cache``.
# Operators wanting auto-probed cluster-specific candidates can populate this
# tuple (probed in order; the first whose PARENT dir exists wins).
_SHARED_GOLDEN_CACHE_ROOTS: tuple[Path, ...] = ()
_GOLDEN_CACHE_ROOT_ENV = "EVOCLAW_GOLDEN_CACHE_ROOT"


def _default_golden_cache_root() -> Path:
    """Default golden-tar cache root, resolved in this order:

    1. ``$EVOCLAW_GOLDEN_CACHE_ROOT`` — an explicit answer always wins.
    2. the first shared root in ``_SHARED_GOLDEN_CACHE_ROOTS`` whose PARENT
       directory exists on this box. The parent, not the root itself: the cache
       directory is created on first use, so probing it would never match on a
       cluster that has not run EvoClaw yet.
    3. ``<project-root>/golden_cache`` — a private per-checkout cache, for a box
       with no shared mount at all (a laptop, or a new cluster).

    Whatever this resolves to must ALSO be allowlisted in the cluster's
    ``policy.allowed_host_paths`` (``slurm_scripts/clusters.yaml``) or the
    oracle's read-only bind of the golden dir is denied at sandbox create.
    Overridden per-run by ``--golden-cache-root``.
    """
    if override := os.environ.get(_GOLDEN_CACHE_ROOT_ENV):
        return Path(override).expanduser()
    for root in _SHARED_GOLDEN_CACHE_ROOTS:
        if root.parent.is_dir():
            return root
    return _PROJECT_ROOT / "golden_cache"


def _ensure_imports() -> None:
    """Make ``xrlenv``, the sibling shim, and EvoClaw's ``harness`` importable."""
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))  # docker_shim / oracle siblings
    try:
        import xrlenv  # noqa: F401
    except ImportError:
        repo = os.environ.get("XRLENV_REPO", "")
        if repo and os.path.isdir(repo):
            sys.path.insert(0, repo)
    # EvoClaw's harness is importable because we run from its checkout (cwd) or
    # EVOCLAW_SOURCE_ROOT points at it.
    src = os.environ.get("EVOCLAW_SOURCE_ROOT")
    if src and src not in sys.path:
        sys.path.insert(0, src)


_EXTRA_AGENTS = ("oracle",)


def _relax_agent_choices() -> None:
    """Let ``run_e2e``'s argparse accept our extra agents (no upstream edit)."""
    orig = argparse.ArgumentParser.add_argument

    def patched(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        choices = kwargs.get("choices")
        if choices and "claude-code" in choices:
            kwargs["choices"] = [*choices, *(a for a in _EXTRA_AGENTS if a not in choices)]
        return orig(self, *args, **kwargs)

    argparse.ArgumentParser.add_argument = patched  # type: ignore[method-assign]


# EvoClaw reads these from --workspace-root (metadata + DAG + milestone list).
_WORKSPACE_REQUIRED = ("metadata.json", "dependencies.csv", "milestones.csv")


def _validate_workspace_root(ws: Path) -> None:
    missing = [f for f in _WORKSPACE_REQUIRED if not (ws / f).exists()]
    if missing:
        raise SystemExit(
            f"--workspace-root {ws} is missing EvoClaw repo data: {missing}\n"
            "EvoClaw reads the repo data from --workspace-root (metadata.json, "
            "dependencies.csv, milestones.csv, srs/, test_results/) — it is not an "
            "arbitrary scratch dir. Use the repo data dir, or pass --workspace-root-base "
            "(default <project-root>/results) so the wrapper builds a symlinked "
            "per-user workspace (see below)."
        )


def _resolve_workspace_root() -> None:
    """Set ``--workspace-root`` without copying or writing into the shared dataset.

    Priority:
      1. explicit ``--workspace-root`` (validated to hold the repo data); else
      2. ``--workspace-root-base`` (your writable dir; default
         ``<project-root>/results``, or ``$EVOCLAW_WORKSPACE_ROOT``) — build/refresh
         ``<base>/<repo>`` of **symlinks** into the shared ``EVOCLAW_DATA_ROOT``
         (no copy; ``e2e_trial/`` writes there; dataset untouched), and inject it;
         else
      3. fail loud, explaining both.
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--workspace-root", default=None)
    p.add_argument("--repo-name", default=None)
    known, _ = p.parse_known_args(sys.argv[1:])

    if known.workspace_root:
        _validate_workspace_root(Path(known.workspace_root).expanduser())
        return

    base = _CFG.workspace_root_base if _CFG else _default_workspace_root_base()
    data_root = os.environ.get("EVOCLAW_DATA_ROOT")
    if base and data_root and known.repo_name:
        import workspace

        ws = workspace.link_workspace(Path(data_root), Path(base), known.repo_name)
        print(f"[xrlenv] per-user workspace (symlinks into EVOCLAW_DATA_ROOT, no copy): {ws}")
        sys.argv.extend(["--workspace-root", str(ws)])
        return

    raise SystemExit(
        "No --workspace-root. Pick one (neither copies the dataset nor writes into it):\n"
        "  (a) --workspace-root-base <dir> (default <project-root>/results, or\n"
        "      $EVOCLAW_WORKSPACE_ROOT) — the wrapper builds <base>/<repo> by symlinking\n"
        "      the shared EVOCLAW_DATA_ROOT; trials write there, the dataset is never\n"
        "      touched. (needs EVOCLAW_DATA_ROOT + --repo-name); or\n"
        "  (b) pass --workspace-root <repo data dir> explicitly."
    )


def _derive_default_args(argv, base_image_ref):  # type: ignore[no-untyped-def]
    """Extra argv filling `--srs-root` (from the *explicit* `--workspace-root`)
    and `--image` (from `--repo-name`) when omitted. `--workspace-root` is never
    derived — see :func:`_require_workspace_root`.

    ``base_image_ref(repo_name) -> str | None`` builds the pullable base ref.
    """
    p = argparse.ArgumentParser(add_help=False)
    for a in ("--repo-name", "--workspace-root", "--srs-root", "--image"):
        p.add_argument(a, default=None)
    known, _ = p.parse_known_args(argv)
    extra: list[str] = []
    if not known.srs_root and known.workspace_root:
        extra += ["--srs-root", str(Path(known.workspace_root) / "srs")]
    if not known.image and known.repo_name:
        ref = base_image_ref(known.repo_name)
        if ref:
            extra += ["--image", ref]
    return extra


def _inject_remove_container() -> None:
    """EvoClaw keeps the long-lived agent container running unless
    ``--remove-container`` is passed (``run_e2e.py`` cleanup: default keeps it).
    On the cluster that orphaned container is later reaped by the watchdog
    instead of released gracefully. Inject the flag so EvoClaw tears the agent
    container down at trial end. Opt out (to inspect it) with
    ``--keep-container``."""
    if "--remove-container" in sys.argv:
        return
    if _CFG and _CFG.keep_container:
        return
    sys.argv.append("--remove-container")
    print("[xrlenv] injecting --remove-container so EvoClaw removes the agent "
          "container at trial end (--keep-container to keep it)")


def _configure_testbed_copy() -> None:
    """Make EvoClaw's whole-``/testbed`` copy an EXPLICIT, off-by-default choice.

    EvoClaw's ``cleanup()`` runs ``docker cp {container}:/testbed .`` — it
    exfiltrates the ENTIRE repo (source + build output + node_modules, hundreds
    of MB for these repos) out to the host purely as a debug artifact. Grading
    does NOT need it. On the cluster that copy goes container→node→control
    plane→client, and xrlenv now caps how much a single ``get_archive`` may
    relay through the control plane (``XRLENV_MAX_GET_ARCHIVE_RELAY_BYTES``,
    default 128 MiB) — a whole-``/testbed`` copy over that cap is REFUSED at the
    transport. That refusal fails only that one copy (EvoClaw's cleanup catches
    it and warns); the eval/grading is unaffected.

    So we DISABLE the copy by default (inject ``--skip-testbed-copy``) and make
    enabling it explicit via the ``--copy-testbed`` flag. When it's on we warn
    loudly that a large testbed may not fully transfer under the relay cap."""
    if not (_CFG and _CFG.copy_testbed):
        if "--skip-testbed-copy" not in sys.argv:
            sys.argv.append("--skip-testbed-copy")
        print("[xrlenv] /testbed debug-copy DISABLED (default). "
              "Pass --copy-testbed to export it.")
        return
    print(
        "[xrlenv] WARNING: /testbed debug-copy is ON. EvoClaw will try to copy "
        "the ENTIRE /testbed out of each eval container. xrlenv caps a single "
        "get_archive relayed through the control plane "
        "(XRLENV_MAX_GET_ARCHIVE_RELAY_BYTES, default 128 MiB); a /testbed above "
        "that cap is REFUSED at the transport, so you may NOT get the full "
        "testbed. That single copy fails cleanly — the eval/grading is "
        "unaffected. For full large-artifact capture, use xrlenv's artifact-"
        "export primitive rather than the control-plane copy path.",
        flush=True,
    )


def _inject_derived_args() -> None:
    def base_ref(repo_name: str) -> str | None:
        try:
            import image_resolution
            import oracle

            # The AGENT BASE image uses a DIFFERENT tag than the milestone/eval
            # images. Upstream's quarantine (auto-on per repo) forces the eval
            # offline (GOPROXY=off, pip offline), so the base MUST be the
            # offline-dependency-closure image hyd2apse/<short>:base-offline-<v>
            # (deps baked: /wheelhouse + go/cargo/npm closure), while the
            # per-milestone eval images stay <mid>-<v>. Hence the base tag DEFAULTS
            # to `offline-<image_tag>`. --base-image-tag overrides (pass plain
            # 'v0.9' to use the legacy non-offline base, e.g. for --unprotected).
            mtag = _CFG.image_tag if _CFG else image_resolution._DEFAULT_TAG
            base_tag = (_CFG.base_image_tag if (_CFG and _CFG.base_image_tag)
                        else f"offline-{mtag}")
            registry = _CFG.image_registry if _CFG else ""
            return image_resolution.dockerhub_ref(
                f"{repo_name}/base", oracle._repo_map(),
                image_tag=base_tag, image_registry=registry,
            )
        except Exception:
            return None

    extra = _derive_default_args(sys.argv[1:], base_ref)
    if extra:
        print(f"[xrlenv] derived args (srs-root from --workspace-root, image from --repo-name): {extra}")
        sys.argv.extend(extra)


def _selected_milestones(ws: Path, dag: Path, spec: str | None) -> list[str]:
    from harness.e2e.milestone_selection import (  # type: ignore[import-not-found]
        load_graph,
        read_base_ids,
        select_prefix,
        topological_order,
    )

    mcsv = ws / "milestones.csv"
    if spec:
        return list(select_prefix(dag, spec, milestones_csv=mcsv))
    # No prefix: extract exactly the workspace's curated selected set (what
    # DAGManager evaluates: <trial_root>/selected_milestone_ids.txt is copied from
    # here), in topological order — NOT the whole DAG. This aligns golden
    # extraction with evaluation and lets a single-id selected file drive a
    # one-milestone run (the per-milestone sweep). Falls back to the full DAG.
    nodes, edges = load_graph(dag, mcsv)
    ordered = list(topological_order(nodes, edges))
    selected = read_base_ids(ws / "selected_milestone_ids.txt")
    return [m for m in ordered if m in selected] if selected else ordered


def _make_yd_fixes_subprocess_safe() -> None:
    """Ensure the eval-level YD patches reach EvoClaw's ``ProcessPoolExecutor`` children.

    EvoClaw evaluates in a ``ProcessPoolExecutor`` (``orchestrator.py``, no explicit
    ``mp_context``). Our fixes are in-process monkey-patches: a **fork** child inherits them,
    but a **spawn**/**forkserver** child boots a fresh interpreter that would not. Two layers,
    belt-and-suspenders:

    1. **Pin fork** as the default start method (fork is available on Linux, our cluster), so
       the executor's ``ProcessPoolExecutor()`` inherits our patches regardless of the
       interpreter default (Python 3.14 flips Linux to forkserver). No-op if already fork.
    2. **Startup re-apply hook** for any child that *is* spawned anyway (e.g. if upstream ever
       passes ``mp_context="spawn"``): export ``EVOCLAW_APPLY_YD_FIXES=1`` and prepend the
       ``_yd_bootstrap`` dir (its ``sitecustomize.py``) + this dir (for ``yd_fixes``) to
       ``PYTHONPATH``. Both are inherited by children; the child's ``site`` init imports the
       hook, which re-runs ``apply_yd_fixes`` (idempotent).
    """
    import multiprocessing as mp

    try:
        if "fork" in mp.get_all_start_methods() and mp.get_start_method(allow_none=True) != "fork":
            mp.set_start_method("fork", force=True)
    except Exception:
        pass  # can't pin fork (e.g. no fork on this platform) -- layer 2 still covers spawn

    os.environ["EVOCLAW_APPLY_YD_FIXES"] = "1"
    boot = str(_HERE / "_yd_bootstrap")   # holds sitecustomize.py
    this = str(_HERE)                     # holds yd_fixes.py
    parts = [boot, this] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    # de-dup while preserving order
    seen: set[str] = set()
    os.environ["PYTHONPATH"] = os.pathsep.join(p for p in parts if not (p in seen or seen.add(p)))


def _prepare_oracle_golden(prefix: str) -> None:
    """For ``--agent oracle``: extract golden END src for the selected milestones
    and expose it via ``EVOCLAW_GOLDEN_DIR`` (mounted into the agent container)."""
    import docker_shim
    import oracle

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--workspace-root", type=Path)
    p.add_argument("--dag-path", type=Path, default=None)
    p.add_argument("--srs-root", type=Path, default=None)
    p.add_argument("--milestones", default=None)
    args, _ = p.parse_known_args()
    if not args.workspace_root:
        raise SystemExit("--workspace-root is required for --agent oracle")
    ws = args.workspace_root
    dag = args.dag_path or (ws / "dependencies.csv")
    mids = _selected_milestones(ws, dag, args.milestones)
    if not mids:
        raise SystemExit("no milestones selected for the oracle")

    # Stable content cache keyed by (repo, data-version tag): a cached <mid>.tar
    # is reused across runs — the milestone image is only acquired on a miss.
    # Root + tag are explicit flags (--golden-cache-root / --image-tag); only the
    # root's DEFAULT consults $EVOCLAW_GOLDEN_CACHE_ROOT, since the shared cache
    # lives at a different path per cluster.
    tag = _CFG.image_tag if _CFG else "v0.9"
    cache_root = Path(_CFG.golden_cache_root).expanduser() if _CFG else _default_golden_cache_root()
    golden_dir = (cache_root / f"{ws.name}__{tag}").resolve()
    refresh = bool(_CFG and _CFG.golden_refresh)
    print(f"[oracle] golden cache: {golden_dir} (milestones: {mids}, refresh={refresh})")
    oracle.extract_selected(
        docker_shim.client(),
        workspace_root=ws,
        milestone_ids=mids,
        golden_dir=golden_dir,
        name_prefix=prefix,
        refresh=refresh,
    )
    # Mount ONLY this task's selected tars, not the whole shared cache: the cache
    # accumulates every milestone's tar across runs, and the shim binds the mounted
    # dir as a single gRPC put — a big repo's full cache (dubbo ~306 MiB) exceeds the
    # 128 MiB message limit and the bind silently fails (oracle sees no golden).
    # A per-task dir of hardlinks (same fs, no copy) keeps the bind to the tars the
    # task actually needs (one, for --parallelization-level milestone).
    mount_dir = (cache_root / ".mount" / f"{ws.name}__{tag}__{os.getpid()}").resolve()
    mount_dir.mkdir(parents=True, exist_ok=True)
    for m in mids:
        src, dst = golden_dir / f"{m}.tar", mount_dir / f"{m}.tar"
        if not src.is_file():
            continue
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        try:
            os.link(src, dst)          # hardlink — instant, no copy
        except OSError:
            import shutil
            shutil.copy2(src, dst)     # cross-device fallback
    os.environ["EVOCLAW_GOLDEN_DIR"] = str(mount_dir)
    global _ORACLE_MOUNT_DIR
    _ORACLE_MOUNT_DIR = mount_dir      # recorded so run() tears it down after the task


def _cleanup_oracle_mount(keep: bool = False) -> None:
    """Remove this task's per-run golden hardlink-mount dir (``.mount/<repo>__<tag>__<pid>``).

    The dir holds hardlinks to the shared golden cache, so removing it frees no real data
    but stops ``.mount/`` from accumulating one dead dir per task. No-op with ``keep=True``
    (``--keep-container``, where the bind source should survive for debugging) or if no mount
    dir was created (non-oracle runs). Never raises."""
    global _ORACLE_MOUNT_DIR
    d, _ORACLE_MOUNT_DIR = _ORACLE_MOUNT_DIR, None
    if keep or d is None:
        return
    try:
        import shutil
        shutil.rmtree(d, ignore_errors=True)
    except Exception:
        pass


# EvoClaw's tag "debounce" (wait for the agent to stop amending) + inter-recovery
# cooldown are meaningless for a deterministic no-LLM agent that tags once.
# (The oracle→watcher race that caused a rare Completed:0 is fixed in the oracle
# agent itself — it lingers after tagging so EvoClaw's 2s tag-watcher reliably
# sees the tag before the process exits; see oracle._ORACLE_SCRIPT. That's
# the right layer: recovery_wait only applies on agent *errors*, not clean exits.)
_FAST_TIMING = {"debounce_seconds": 0, "max_debounce_wait": 0, "recovery_wait_seconds": 0}


def _fast_config_dict(base: dict) -> dict:  # type: ignore[type-arg]
    """Return ``base`` with the retry/timing waits zeroed (pure; for testing)."""
    cfg = dict(base)
    rt = dict(cfg.get("retry_and_timing", {}))
    rt.update(_FAST_TIMING)
    cfg["retry_and_timing"] = rt
    return cfg


def _maybe_fast_config() -> None:
    """For ``--agent oracle`` with no explicit ``--config``: write an
    eval config with debounce + recovery waits zeroed and inject ``--config``.
    Opt out with ``--no-fast-oracle``."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--agent")
    p.add_argument("--config")
    p.add_argument("--workspace-root")
    known, _ = p.parse_known_args(sys.argv[1:])
    if known.agent not in _EXTRA_AGENTS or known.config:
        return
    if _CFG and _CFG.no_fast_oracle:
        return
    import yaml  # EvoClaw dep
    from harness.e2e import run_e2e  # type: ignore[import-not-found]

    # Base on the config EvoClaw would use (repo's workspace one, else harness
    # default), so we only override the timing and keep everything else.
    base_path = None
    if known.workspace_root:
        wc = Path(known.workspace_root) / "e2e_config.yaml"
        base_path = wc if wc.is_file() else None
    if base_path is None:
        base_path = Path(run_e2e.__file__).parent / "e2e_config.yaml"
    base = yaml.safe_load(base_path.read_text()) if base_path.is_file() else {}
    # pid-unique so concurrent sweep runs don't clobber each other's config.
    out = _HERE / "tmp" / f"oracle-fast-e2e-config-{os.getpid()}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(_fast_config_dict(base or {})))
    print(f"[xrlenv] {known.agent}: fast eval config (debounce/recovery waits = 0): {out}")
    sys.argv.extend(["--config", str(out)])


# ── Wrapper CLI flags ─────────────────────────────────────────────────────────
# All onboarding behaviour is EXPLICIT command-line flags — not env vars — so a
# stale value in .env_private can never silently change a run. (Deployment config
# that IS environment — EVOCLAW_DATA_ROOT, the cluster coords — stays in .env.)
_CFG: argparse.Namespace | None = None
# This task's per-run golden hardlink-mount dir (cache_root/.mount/<repo>__<tag>__<pid>),
# recorded by _prepare_oracle_golden so run() can tear it down (see _cleanup_oracle_mount).
_ORACLE_MOUNT_DIR: Path | None = None


def _parse_wrapper_flags() -> argparse.Namespace:
    """Consume the onboarding's own flags off ``sys.argv`` (before ``run_e2e``
    ever parses it) and return them. Run once at the top of ``main()``."""
    # allow_abbrev=False is LOAD-BEARING: without it, argparse prefix-matches
    # run_e2e's ``--workspace-root`` onto our ``--workspace-root-base`` and eats
    # it, so run_e2e never sees --workspace-root and silently rebuilds the FULL
    # repo workspace (every per-milestone task runs the whole repo). Do not remove.
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    p.add_argument("--copy-testbed", action="store_true",
                   help="re-enable EvoClaw's whole-/testbed debug copy at trial end (default: OFF)")
    p.add_argument("--keep-container", action="store_true",
                   help="keep the long-lived agent container after the trial (default: remove it)")
    p.add_argument("--golden-refresh", action="store_true",
                   help="re-extract the golden tar even on a cache hit")
    p.add_argument("--no-fast-oracle", action="store_true",
                   help="keep EvoClaw's real debounce/recovery waits (don't zero them for the oracle)")
    p.add_argument("--apply-yd-fixes", action="store_true",
                   help="apply opt-in local corrections to known UPSTREAM eval-protocol "
                        "bugs (currently: preserve untracked GT test files across the "
                        "evaluator's git clean, e.g. element e662c19/fba5938). Default "
                        "OFF = faithful, leaderboard-comparable. See yd_fixes.py.")
    p.add_argument("--cpu-pinning", action="store_true",
                   help="enable xrlenv cpuset-pinning for THIS worker's containers: each "
                        "gets ceil(cpus) dedicated cores so nproc==cpus (stops go/cargo/"
                        "jest oversubscribing on big nodes). Uses xrlenv's existing "
                        "RuntimeLimits.cpu_pinning via an onboarding runtime-patch — no "
                        "xrlenv-core change. In a batch, don't set this globally — the "
                        "driver routes it per-milestone via --cpu-pinning-milestone. Default OFF.")
    p.add_argument("--cluster-retries", type=int, default=4, metavar="N",
                   help="in-place retries on a transient CP/node blip (default: 4)")
    p.add_argument("--cluster-retry-base-s", type=float, default=3.0, metavar="S",
                   help="retry backoff base seconds — 3,6,12,24 (default: 3)")
    p.add_argument("--mem-per-cpu-gb", type=float, default=2.0, metavar="G",
                   help="memory per CPU for acquired containers, GiB — each container's "
                        "--memory cap = its cpus x this when EvoClaw declared none "
                        "(0=cluster default; default: 2). In a sweep the driver sets this.")
    p.add_argument("--oracle-tag-settle-s", type=int, default=12, metavar="S",
                   help="oracle lingers this long after tagging so EvoClaw's ~2s tag-watcher catches it (default: 12)")
    p.add_argument("--workspace-root-base", default=_default_workspace_root_base(), metavar="DIR",
                   help="per-user writable base for symlinked workspaces "
                        "(default: $EVOCLAW_WORKSPACE_ROOT if set, else <project-root>/results)")
    p.add_argument("--golden-cache-root", default=str(_default_golden_cache_root()), metavar="DIR",
                   help="content cache root for the extracted golden tars "
                        "(default: $EVOCLAW_GOLDEN_CACHE_ROOT if set, else the "
                        "first shared /fsx cache root present on this box, else "
                        "<project-root>/golden_cache; here: "
                        f"{_default_golden_cache_root()}). Must be covered by the "
                        "cluster's policy.allowed_host_paths")
    # Image config: tag + registry are flags (not env). The one image knob kept as
    # env is EVOCLAW_GOZERO_BASE_IMAGE (required, no default; read
    # inside image_resolution). image_resolution owns the tag default; imported
    # here (its dir is on sys.path by the time main() calls this).
    import image_resolution
    p.add_argument("--image-tag", default=image_resolution._DEFAULT_TAG, metavar="TAG",
                   help="tag EvoClaw's milestone/golden images are published under; "
                        "keys the golden cache too "
                        f"(default: {image_resolution._DEFAULT_TAG})")
    p.add_argument("--base-image-tag", default=None, metavar="TAG",
                   help="tag for the AGENT BASE image ONLY (default: "
                        "'offline-<image_tag>', i.e. upstream's offline-closure base "
                        "hyd2apse/<short>:base-offline-<v> that the auto-on quarantine "
                        "requires). Milestone/eval images stay at --image-tag. Pass "
                        "plain '<v>' (e.g. 'v0.9') to use the legacy non-offline base.")
    p.add_argument("--image-registry", default="", metavar="HOST",
                   help="explicit Docker Hub mirror-host prefix for milestone refs "
                        "(default: empty → docker.io routes via the node mirror)")
    # Fleet reservation (opt-in). This task schedules a FLEET of containers (one
    # long-lived agent + one or more heavier eval containers); declaring the
    # peak footprint lets xrlenv reserve it on one node so the evals aren't
    # starved by greedy admission. Both are REQUIRED TOGETHER when fleet
    # reservation is used — no default: the operator states the footprint
    # deliberately (per-milestone task ~= agent + 1 eval; per-repo task ~= agent
    # + 4 evals, EvoClaw's fixed ThreadPoolExecutor). Neither set = fleet off
    # (legacy per-container admission). Exactly one set = fail loud (below).
    p.add_argument("--fleet-footprint-cpu", type=float, default=None, metavar="C",
                   help="fleet reservation: the task's PEAK cpu footprint (whole cores). "
                        "REQUIRED with --fleet-footprint-mem-gb; no default")
    p.add_argument("--fleet-footprint-mem-gb", type=float, default=None, metavar="G",
                   help="fleet reservation: the task's PEAK memory footprint (GiB). "
                        "REQUIRED with --fleet-footprint-cpu; no default")
    known, rest = p.parse_known_args(sys.argv[1:])
    # Strip our flags so run_e2e's own argparse never sees them.
    sys.argv[:] = [sys.argv[0], *rest]
    return known


def _resolve_fleet_footprint(
    fleet_cpu: float | None,
    fleet_mem_gb: float | None,
    fleet_id: str,
) -> tuple[str | None, float | None, int | None]:
    """Turn the two footprint flags into fleet-install args.

    Returns ``(fleet_id, cpu_request, mem_request_bytes)`` — all three set when
    fleet reservation is ON, all ``None`` when OFF. Contract (mirrors xrlenv's
    both-or-neither label rule + the operator's "nothing slips through"):

    - both flags set   -> fleet ON with this task's ``fleet_id``.
    - neither set       -> fleet OFF (legacy per-container admission).
    - exactly one set   -> ``SystemExit`` (fail loud, no silent partial).
    - non-positive value -> ``SystemExit``.
    """
    if (fleet_cpu is None) != (fleet_mem_gb is None):
        raise SystemExit(
            "run_e2e_xrlenv: --fleet-footprint-cpu and --fleet-footprint-mem-gb "
            "must be given TOGETHER (a fleet declares both cpu and memory) or "
            f"neither (fleet off). Got cpu={fleet_cpu!r}, mem_gb={fleet_mem_gb!r}.",
        )
    if fleet_cpu is None:
        return None, None, None
    if fleet_cpu <= 0 or (fleet_mem_gb is not None and fleet_mem_gb <= 0):
        raise SystemExit(
            "run_e2e_xrlenv: fleet footprint must be positive; got "
            f"cpu={fleet_cpu}, mem_gb={fleet_mem_gb}.",
        )
    assert fleet_mem_gb is not None  # both-or-neither guaranteed above
    return fleet_id, fleet_cpu, int(fleet_mem_gb * (1024 ** 3))


def main() -> int:
    # `import xrlenv` auto-loads the nearest `.env` at package import — but ONLY
    # `.env` (EvoClaw's committed placeholder), never `.env_private`. If that runs
    # first it seeds os.environ with the placeholder and our loader (shell-wins)
    # would keep it — no `unset` can fix that, since the re-load happens every
    # import. Our loader reads BOTH files with correct precedence, so make it
    # authoritative: disable xrlenv's autoload and run ours FIRST, before any
    # `import xrlenv`, so the *true* shell is the precedence baseline.
    os.environ.setdefault("XRLENV_DOTENV", "off")
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))
    import env_loader

    env_loader.load_project_dotenv()  # .env + .env_private, shell-wins, warns on shadow

    global _CFG
    _CFG = _parse_wrapper_flags()  # explicit flags off sys.argv, before run_e2e sees them

    _ensure_imports()

    # Resolve --workspace-root (explicit, or symlinked per-user from
    # --workspace-root-base) before any cluster work — needs the .env vars above.
    _resolve_workspace_root()

    import docker_shim  # sibling
    import oracle
    from harness.e2e.agents import get_agent_framework  # type: ignore[import-not-found]

    # Importing oracle runs its @register_framework decorator. Apply the
    # tag-settle flag to the module, reference the class (so the import isn't
    # dead), and fail loud if registration didn't take.
    oracle._TAG_SETTLE_S = _CFG.oracle_tag_settle_s
    _ = oracle.OracleFramework
    for _name in _EXTRA_AGENTS:
        get_agent_framework(_name)  # raises ValueError if not registered

    prefix = os.environ.get("EVOCLAW_RUN_PREFIX", f"xrl-{os.getpid()}-")
    labels = {"benchmark": "evoclaw"}
    group = os.environ.get("XRLENV_GROUP_ID")
    if group:
        labels["xrlenv.group_id"] = group

    # Fleet reservation (opt-in). The fleet_id is this process's own identity
    # (the sweep runs one process per task — agent + its evals share it), so all
    # of a task's containers land in one reservation.
    fleet_id, fleet_cpu_request, fleet_mem_request_bytes = _resolve_fleet_footprint(
        _CFG.fleet_footprint_cpu, _CFG.fleet_footprint_mem_gb, prefix,
    )
    if fleet_id is not None:
        print(
            f"[fleet] reservation ON: fleet_id={fleet_id} footprint "
            f"cpu={fleet_cpu_request} mem={_CFG.fleet_footprint_mem_gb:.1f}GiB",
            file=sys.stderr,
        )

    docker_shim.install(
        name_prefix=prefix,
        labels=labels,
        cluster_retries=_CFG.cluster_retries,
        cluster_retry_base_s=_CFG.cluster_retry_base_s,
        container_mem_per_cpu_gb=_CFG.mem_per_cpu_gb,
        fleet_id=fleet_id,
        fleet_cpu_request=fleet_cpu_request,
        fleet_mem_request_bytes=fleet_mem_request_bytes,
    )
    _relax_agent_choices()

    # Point EvoClaw's milestone-image resolution at pullable Docker Hub refs so
    # the cluster mirror can pull on acquire (DESIGN.md §5.2). Base image is the
    # operator's --image (pass a pullable hyd2apse/<short>:base-<v>).
    import image_resolution  # sibling

    image_resolution.install(
        image_tag=_CFG.image_tag,
        image_registry=_CFG.image_registry,
    )

    # Fill --srs-root (from the explicit --workspace-root) + --image (from --repo-name).
    _inject_derived_args()

    # No-LLM agents: skip EvoClaw's 120s tag-debounce + 60s recovery cooldowns.
    _maybe_fast_config()

    # Ask EvoClaw to remove the long-lived agent container at trial end (else the
    # cluster watchdog reaps the orphan ~15 min later).
    _inject_remove_container()

    # /testbed debug-copy is OFF by default (it's a whole-repo exfil the grader
    # doesn't need, and it can exceed the control-plane get_archive relay cap).
    _configure_testbed_copy()

    if "--agent" in sys.argv and "oracle" in sys.argv:
        _prepare_oracle_golden(prefix)

    # Opt-in local corrections to upstream eval-protocol bugs (default OFF). Applied
    # as runtime monkey-patches on the harness, so the vendored files stay pristine.
    if _CFG.apply_yd_fixes:
        import yd_fixes  # sibling
        # Make the eval-level patches survive EvoClaw's ProcessPoolExecutor regardless of
        # the multiprocessing start method (fork inherits them; spawn/forkserver need the
        # startup hook) -- see _make_yd_fixes_subprocess_safe.
        _make_yd_fixes_subprocess_safe()
        yd_fixes.apply_yd_fixes()
    elif not os.environ.get("_XRLENV_YD_WARNED"):
        # Loud warning that known upstream eval-protocol bugs are NOT corrected. Skipped
        # when a parent sweep (run_all_xrlenv) already printed it once at the top.
        import yd_fixes  # sibling
        yd_fixes.warn_yd_fixes_off()

    # Opt-in cpuset-pinning (default OFF): runtime-patch the xrlenv compat layer's
    # runtime-limits assembler so acquires carry cpu_pinning=True — no xrlenv-core edit.
    if _CFG.cpu_pinning:
        import cpu_pinning  # sibling
        cpu_pinning.apply_cpu_pinning()

    from harness.e2e import run_e2e

    try:
        return int(run_e2e.main() or 0)
    finally:
        # Safety net: force-remove any container EvoClaw left registered (e.g. on
        # a crash/interrupt before its own teardown), then restore subprocess.
        docker_shim.cleanup_containers()
        docker_shim.uninstall()
        # Remove this task's per-run golden hardlink-mount dir so .mount/ doesn't
        # accumulate one dir per task forever (the tars are hardlinks -> the shared
        # golden_cache is untouched). Skipped with --keep-container (kept for debug).
        _cleanup_oracle_mount(keep=bool(getattr(_CFG, "keep_container", False)))


if __name__ == "__main__":
    sys.exit(main())
