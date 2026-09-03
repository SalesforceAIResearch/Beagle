#!/usr/bin/env python3
"""Build the shared harbor task cache for the seta-env benchmark.

Self-contained, ``--stage``-driven pipeline — the same shape as terminalworld /
terminal_bench_2_1's ``build_cache.py`` — that clones ``camel-ai/seta-env`` and
lands each ``Harbor-Dataset/<id>/`` under a dedicated ``seta-env/`` shard of the
unified harbor cache (``<cache>/seta-env/<id>/``), so seta tasks are namespaced
away from terminal-bench-2's in the same cache.

One command — ``--stage all`` (the default) — yields a correct cache: it clones
the catalog, then applies three kinds of cache-level repair (no separate stage for
any — they are just part of building a correct cache):

- **Migration-repair overlays** (``patches/<id>/`` full-file copies). The
  ``camel-ai/seta-env`` Harbor-Dataset conversion dropped runtime-critical config
  the original ``Dataset/<id>/`` still has; each overlay restores the smallest
  faithful piece (e.g. 309's solve.sh guards on the pre-Harbor ``/oracle`` run
  path, but harbor runs the oracle from ``/solution``). See ``patches/README.md``.
- **Base-image restore** (``BASE_IMAGE_FIX_TASKS``). The migration swapped these
  tasks' original ``FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:<date>`` base
  (which bakes python3/curl/uv/wget/tmux) for bare ``ubuntu:24.04``, so their
  identical solve.sh fails ``command not found``. ``--stage all`` rewrites the FROM
  back; ``build_plan_gen`` then builds these ``type: local`` from the restored
  cache Dockerfile (a git build would use upstream's broken FROM). See STATUS.md §1.3.
- **DinD sysbox markers** — ``[environment.env] XRLENV_CONTAINER_RUNTIME`` (+
  ``XRLENV_INNER_DOCKERD`` / ``XRLENV_INSTALL_DOCKERD`` companions) for tasks whose
  oracle needs a docker daemon / NET_ADMIN / SYS_ADMIN / systemd that plain
  ``runc`` can't host (``SYSBOX_TASKS``). Task-level *routing*, not content.
- **Verifier-as-root markers** — ``[verifier] user = "root"`` for custom-user tasks
  (``VERIFIER_ROOT_TASKS``). Harbor 0.20 runs the verifier as the image USER; a task
  that sets a non-root USER then runs its (root-assuming: ``apt``/``su -l``) verifier
  as non-root and scores 0. Restores terminal-bench's root-verifier contract; the
  agent still runs as the task user. task.toml edit → no image rebuild.

``all`` is idempotent (re-running re-applies the overlays + base-restore + markers).
Exclusions stay in ``black_list.txt`` (build-unbuildable + runtime-excluded),
handled at build-plan + sweep time. ``--stage sysbox`` (re)writes only the markers.

    # normal path — clone + overlays + markers, idempotent, safe to re-run:
    python xrlenv_plugins/benchmarks/seta/build_cache.py --stage all \\
        --dest "$XRLENV_BENCHMARK_CACHE"

``--dest`` defaults to ``$XRLENV_BENCHMARK_CACHE`` (else ``~/.cache/harbor/tasks``).
``--repo`` / ``--ref`` override the upstream git URL / ref. Point every xrlenv
consumer's ``XRLENV_BENCHMARK_CACHE`` at the same root — seta and terminal-bench-2
coexist in it under separate shards.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

# Shard subdir name == image namespace, matching build_plan_gen.py / the sweep.
SHARD = "seta-env"
DEFAULT_REPO = "https://github.com/camel-ai/seta-env.git"
DEFAULT_REF = "main"

# Curated per-task overlays repairing Harbor-migration damage (see patches/README.md).
# Each ``patches/<task_id>/<rel>`` file is copied over the populated cache task,
# so the fix survives re-populate. Applied by ``--stage all``.
PATCHES_DIR = Path(__file__).resolve().parent / "patches"

# Base-image restore (Harbor-migration damage). The conversion swapped every task's
# original ``FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:<date>`` (the fixed
# T-Bench base that bakes python3 / curl / uv / wget / tmux / … ) for bare
# ``ubuntu:24.04``, so a task whose IDENTICAL solve.sh assumes one of those tools
# fails ``command not found``. Restoring the original base is the faithful, uniform
# fix; ``build_cache.py --stage all`` rewrites the FROM in these tasks' cache
# Dockerfiles and ``build_plan_gen`` then builds them ``type: local`` from the
# restored cache. See STATUS.md / README §1.3.
TBENCH_BASE = "ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624"
_BARE_BASE = "ubuntu:24.04"

# Tasks a base-restore fixes on its OWN — their solve.sh just needs a common tool the
# t-bench base baked. Each VALIDATED green after the rebuild (oracle reward 1.0).
# Grown one proven task at a time. NOT here (blacklisted): tasks whose solve INSTALLS
# or BREAKS the tool itself, so a base restore alone doesn't fix them —
#   * 197 java-PPA, 409/962 py3.8-3.9, 414 adb, 535 custom-gm, 308 gcc/timeout
#     (solve installs the tool; install fails for another reason);
#   * 15/304/729/1092 (curl in the base, but the solve's apt/dpkg/permission edits
#     break the verifier's curl->uv bootstrap); 172 (solve pins an unavailable
#     `wget=<version>`). These 5 were rebuilt on the t-bench base but STILL fail —
#     re-blacklisted after the 2026-08-04 validation sweep (10/15 green).
BASE_IMAGE_FIX_TASKS = frozenset({
    "240", "367", "390", "617", "906", "953",   # python3
    "60", "827",                                  # curl (verifier uv bootstrap)
    "1203",                                       # uv (via curl)
    "723",                                        # tmux
})

# Verifier-as-root marker (Harbor-migration behavior, not a base-image issue). A seta
# task whose Dockerfile sets a non-root ``USER`` (``[metadata] sets_custom_user = true``)
# gets BOTH its agent AND its verifier run as that user by harbor 0.20 — because
# ``[verifier] user`` is unset (None → the image USER; harbor single_step.py runs the
# verifier as ``task.config.verifier.user``). But terminal-bench's verifier contract is
# ROOT: the stock ``tests/test.sh`` does ``apt-get install curl`` (→ the uv bootstrap)
# and the tests do ``su -l <user>`` — both need root. As the non-root image USER the
# bootstrap dies (``curl: command not found``) and ``su`` fails (``Authentication
# failure``) → reward 0. The faithful CACHE-LEVEL fix is ``[verifier] user = "root"`` in
# task.toml — surgical (only the verifier phase; the agent/solve still runs as the task
# user) and NO image rebuild (task.toml is read at trial time). Verified end-to-end: with
# the verifier as root, 15's oracle scores reward 1.0.
#
# Scoped to the FAILING custom-user tasks only (the 9 passing custom-user tasks run tests
# that are fine as the task user — forcing root risks a regression, so leave them). NOT
# here: 407 (host-sysctl) / 999 (build-unbuildable) also set a custom user but fail for a
# DIFFERENT primary reason. Grown one oracle-validated task at a time.
VERIFIER_ROOT_TASKS = frozenset({"15", "304", "729", "1092"})


def _harbor_cache_root(dest: str | None) -> Path:
    """The cache ROOT: ``dest``, else ``$XRLENV_BENCHMARK_CACHE``, else fail loud.

    ``benchmark_cache_root`` is the single implementation — it rejects the retired
    XRLENV_HARBOR_CACHE var / xrlenv_harbor_cache path (renamed 2026-07-31) and raises
    when nothing is set. This used to fall back to a home-directory cache instead, which
    answers an operator error with a plausible-but-wrong directory. Lazy import to match
    the plugin style (plugin -> xrlenv is allowed).
    """
    from xrlenv_plugins.benchmarks._benchmark_cache import benchmark_cache_root

    return Path(benchmark_cache_root(dest)).expanduser()


def _shard_dir(dest: str | None) -> Path:
    return _harbor_cache_root(dest) / SHARD


def _count_tasks(shard_dir: Path) -> int:
    return sum(1 for _ in shard_dir.glob("*/task.toml"))


def is_populated(shard_dir: Path) -> bool:
    """True if the shard already holds at least one task — the "cache present"
    signal that makes ``populate`` a no-op on re-run."""
    return shard_dir.is_dir() and _count_tasks(shard_dir) > 0


def _tasks_to_move(harbor_dataset: Path, shard_dir: Path) -> list[Path]:
    """The ``Harbor-Dataset/<id>/`` dirs to move into the shard: those that carry
    a ``task.toml`` and are not already present (per-task idempotency). Pure — no
    I/O beyond the directory scan — so the selection is unit-testable."""
    return [
        d for d in sorted(harbor_dataset.iterdir())
        if d.is_dir()
        and (d / "task.toml").is_file()
        and not (shard_dir / d.name).exists()
    ]


def populate(
    dest: str | None = None, *, repo: str = DEFAULT_REPO, ref: str = DEFAULT_REF,
) -> int:
    """Clone the seta-env catalog into the shard (idempotent). Returns the number
    of task dirs newly materialized. Raises ``SystemExit`` on a clone failure or
    an unexpected upstream layout."""
    shard = _shard_dir(dest)
    if is_populated(shard):
        print(
            f">> {shard} already populated ({_count_tasks(shard)} tasks) — "
            f"skipping clone.",
            file=sys.stderr,
        )
        return 0

    shard.mkdir(parents=True, exist_ok=True)
    # Stage the clone next to the cache (same filesystem) so moving task dirs into
    # the shard is a rename, not a cross-fs copy. Cleaned up in `finally`.
    staging = shard.parent.parent / f".seta-staging.{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)
    print(f">> cloning {repo} ({ref}) -> staging", file=sys.stderr)
    try:
        try:
            subprocess.run(
                ["git", "clone", "--branch", ref, "--depth", "1", repo, str(staging)],
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise SystemExit(
                f"git clone failed (exit {exc.returncode}): {repo} @ {ref}",
            ) from exc

        harbor_dataset = staging / "Harbor-Dataset"
        if not any(harbor_dataset.glob("*/task.toml")):
            raise SystemExit(
                f"clone has no Harbor-Dataset/*/task.toml under {harbor_dataset} "
                f"— upstream layout changed?",
            )

        moved = 0
        for task_dir in _tasks_to_move(harbor_dataset, shard):
            shutil.move(str(task_dir), str(shard / task_dir.name))
            moved += 1
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    print(
        f">> merged {moved} seta task(s) into {shard} ",
        file=sys.stderr,
    )
    return moved


# ── sysbox DinD routing markers (task-level routing, not content) ──────────────
# A seta task marked here gets ``[environment.env] XRLENV_CONTAINER_RUNTIME`` (+
# companions) written into its task.toml, so the cluster plug-in routes it to a
# sysbox-runc node and brings up an unprivileged nested dockerd. Same knobs
# terminalworld's build_cache writes; see xrlenv_plugins/harbor/environment.py
# (_start_inner_dockerd) for how the plug-in consumes them. Grown one PROVEN task
# at a time — a marked task hard-fails on a cluster with no sysbox node.


@dataclass(frozen=True)
class SysboxTask:
    """One seta task routed to the sysbox-runc (DinD) pool.

    ``inner_dockerd`` brings up a nested dockerd after acquire (for a task whose
    oracle talks to ``/var/run/docker.sock``). ``install_dockerd`` apt-installs
    the docker engine first — for a CLI-only image (``docker-ce-cli`` with no
    daemon) that assumed a host socket. The remaining fields mirror
    terminalworld's ``SysboxTask`` so the shared marker writer applies the same
    ``[environment.env]`` knobs; seta doesn't use them yet but keeping the shape
    identical means a future seta DinD task needs no new plumbing."""

    task: str
    runtime: str
    inner_dockerd: bool
    reason: str
    install_dockerd: bool = False
    dockerd_legacy_store: bool = False
    systemd_init: bool = False
    agent_user: str | None = None
    verifier_user: str | None = None


# The seta sysbox set — each validated end-to-end on the dev sysbox cluster
# (oracle reward 1.0). Grown one proven task at a time, mirroring terminalworld's
# discipline. These tasks' oracles need a privilege plain runc can't grant, that
# sysbox provides unprivileged (no privileged host, no xrlenv_plugins/harbor change):
# a nested docker daemon (DinD), NET_ADMIN (iptables / ip netns), SYS_ADMIN (mount),
# or a systemd PID 1. Every id below was run under its marker and scored reward 1.0.
SYSBOX_TASKS: tuple[SysboxTask, ...] = (
    # ── DinD: image needs a live docker daemon ────────────────────────────────
    SysboxTask(
        task="8",
        runtime="sysbox-runc",
        inner_dockerd=True,
        reason=(
            "oracle test runs `docker ps` as the developer user; image ships the "
            "full docker-ce daemon. Under runc there is no daemon (Cannot connect "
            "to the Docker daemon); sysbox + nested dockerd → reward 1.0."
        ),
    ),
    SysboxTask(
        task="1004",
        runtime="sysbox-runc",
        inner_dockerd=True,
        install_dockerd=True,
        reason=(
            "multi-network DinD; image ships docker-ce-CLI only (no daemon), so "
            "install the engine before nesting. sysbox + install + nested dockerd "
            "→ reward 1.0."
        ),
    ),
    # ── iptables (unprivileged NET_ADMIN under sysbox) ────────────────────────
    SysboxTask(task="1117", runtime="sysbox-runc", inner_dockerd=False,
               reason="iptables firewall rules; sysbox unprivileged NET_ADMIN → reward 1.0."),
    SysboxTask(task="1347", runtime="sysbox-runc", inner_dockerd=False,
               reason="iptables rules; sysbox unprivileged NET_ADMIN → reward 1.0."),
    # ── mount / mkfs (unprivileged SYS_ADMIN under sysbox) ────────────────────
    SysboxTask(task="311", runtime="sysbox-runc", inner_dockerd=False,
               reason="mount inside the container; sysbox unprivileged SYS_ADMIN → reward 1.0."),
    SysboxTask(task="119", runtime="sysbox-runc", inner_dockerd=False,
               reason="mount/filesystem ops; sysbox unprivileged SYS_ADMIN → reward 1.0."),
    SysboxTask(task="1225", runtime="sysbox-runc", inner_dockerd=False,
               reason="mount/filesystem ops; sysbox unprivileged SYS_ADMIN → reward 1.0."),
    SysboxTask(task="830", runtime="sysbox-runc", inner_dockerd=False,
               reason="mount/filesystem ops; sysbox unprivileged SYS_ADMIN → reward 1.0."),
    # ── ip netns / veth (unprivileged NET_ADMIN under sysbox) ─────────────────
    SysboxTask(task="1059", runtime="sysbox-runc", inner_dockerd=False,
               reason="ip netns/link (multi-interface); sysbox unprivileged NET_ADMIN → reward 1.0."),
    SysboxTask(task="484", runtime="sysbox-runc", inner_dockerd=False,
               reason="ip netns/veth; sysbox unprivileged NET_ADMIN → reward 1.0."),
    # ── systemd services (unprivileged PID-1 init under sysbox) ───────────────
    SysboxTask(task="345", runtime="sysbox-runc", inner_dockerd=False, systemd_init=True,
               reason="`systemctl start …`; sysbox unprivileged systemd PID 1 → reward 1.0."),
    # ── cap_add recovered from the Dataset/ compose (migration dropped it; sysbox
    #    grants the caps unprivileged, so no privileged host / no plugin change) ──
    SysboxTask(task="846", runtime="sysbox-runc", inner_dockerd=False,
               reason="Dataset/846 compose dropped `cap_add: [NET_RAW, NET_ADMIN]` "
               "(command stays sleep infinity); sysbox grants them → reward 1.0."),
)


def _table_lookup(parsed: dict[str, object], table: str) -> dict[str, object]:
    """Walk a dotted TOML table path (e.g. ``environment.env``) in a parsed doc,
    returning the sub-table dict (or ``{}`` if absent)."""
    node: object = parsed
    for part in table.split("."):
        node = node.get(part, {}) if isinstance(node, dict) else {}
    return node if isinstance(node, dict) else {}


def _ensure_table_key(text: str, table: str, key: str, value: str) -> tuple[str, bool]:
    """Surgically ensure ``[table]`` carries ``key = "value"`` in a task.toml.

    If ``key`` is already present in ``[table]`` with that value, returns
    ``(text, False)``. Otherwise inserts ``key = "value"`` under the existing
    ``[table]`` header, or appends a fresh ``[table]`` block at EOF. Only ADDS a
    missing key — never rewrites an existing one. Pure function (unit-testable)."""
    if str(_table_lookup(tomllib.loads(text), table).get(key)) == value:
        return text, False
    entry = f'{key} = "{value}"\n'
    header = re.compile(r"^\[" + re.escape(table) + r"\]\s*$")
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if header.match(line):
            lines.insert(i + 1, entry)
            return "".join(lines), True
    tail = "".join(lines)
    if tail and not tail.endswith("\n"):
        tail += "\n"
    return tail + f"\n[{table}]\n" + entry, True


def apply_sysbox_marker(text: str, spec: SysboxTask) -> tuple[str, str]:
    """Apply a task's sysbox routing markers to a task.toml given as ``text``.

    Ensures ``[environment.env]`` carries ``XRLENV_CONTAINER_RUNTIME`` (+ the
    ``XRLENV_INNER_DOCKERD`` / ``XRLENV_INSTALL_DOCKERD`` /
    ``XRLENV_DOCKERD_LEGACY_STORE`` / ``XRLENV_SYSTEMD_INIT`` companions per the
    spec) and, when set, ``[agent] user`` / ``[verifier] user``. Returns
    ``(new_text, status)`` where ``status`` is ``"patched"`` or ``"already"``.
    Pure function — unit-testable in isolation."""
    pairs: list[tuple[str, str, str]] = [
        ("environment.env", "XRLENV_CONTAINER_RUNTIME", spec.runtime),
    ]
    if spec.inner_dockerd:
        pairs.append(("environment.env", "XRLENV_INNER_DOCKERD", "1"))
    if spec.install_dockerd:
        pairs.append(("environment.env", "XRLENV_INSTALL_DOCKERD", "1"))
    if spec.dockerd_legacy_store:
        pairs.append(("environment.env", "XRLENV_DOCKERD_LEGACY_STORE", "1"))
    if spec.systemd_init:
        pairs.append(("environment.env", "XRLENV_SYSTEMD_INIT", "1"))
    if spec.agent_user:
        pairs.append(("agent", "user", spec.agent_user))
    if spec.verifier_user:
        pairs.append(("verifier", "user", spec.verifier_user))

    changed = False
    for table, key, value in pairs:
        text, c = _ensure_table_key(text, table, key, value)
        changed = changed or c
    return text, ("patched" if changed else "already")


def apply_all_sysbox_markers(
    shard_dir: Path, only: list[str] | None,
) -> list[tuple[str, str]]:
    """Apply the ``SYSBOX_TASKS`` markers (optionally filtered to ``only``) to
    their task.toml. Returns ``[(task, status), …]``. Verifies each result parses
    and carries the runtime marker (fail-loud post-condition)."""
    results: list[tuple[str, str]] = []
    for spec in SYSBOX_TASKS:
        if only is not None and spec.task not in only:
            continue
        toml_path = shard_dir / spec.task / "task.toml"
        if not toml_path.is_file():
            raise SystemExit(
                f"sysbox-marker target missing: {toml_path} — is the cache "
                f"populated? Run `--stage populate` first.",
            )
        new_text, status = apply_sysbox_marker(toml_path.read_text(), spec)
        if status == "patched":
            toml_path.write_text(new_text)
        env = tomllib.loads(toml_path.read_text()).get("environment", {}).get("env", {})
        assert env.get("XRLENV_CONTAINER_RUNTIME") == spec.runtime, (
            f"post-condition failed for {spec.task}: marker not applied"
        )
        results.append((spec.task, status))
    if only:
        unknown = set(only) - {s.task for s in SYSBOX_TASKS}
        if unknown:
            raise SystemExit(
                f"--tasks names not in SYSBOX_TASKS: {sorted(unknown)}. Add a "
                f"SysboxTask row for it first.",
            )
    return results


# ── verifier-as-root markers ([verifier] user = "root" for custom-user tasks) ──


def apply_verifier_root_marker(text: str) -> tuple[str, str]:
    """Ensure a task.toml's ``[verifier]`` table carries ``user = "root"``. Pure;
    returns ``(new_text, "patched"|"already")``. If ``[verifier] user`` is already
    present (any value) it is left untouched (a task that deliberately pins a
    different verifier user is respected) — this also avoids inserting a duplicate
    key, since ``_ensure_table_key`` keys on value-match, not presence."""
    if _table_lookup(tomllib.loads(text), "verifier").get("user") is not None:
        return text, "already"
    new_text, changed = _ensure_table_key(text, "verifier", "user", "root")
    return new_text, ("patched" if changed else "already")


def apply_all_verifier_root_markers(
    shard_dir: Path, only: list[str] | None,
) -> list[tuple[str, str]]:
    """Write ``[verifier] user = "root"`` into the ``VERIFIER_ROOT_TASKS`` task.toml
    (optionally filtered to ``only``). Returns ``[(task, status), …]``, with a
    fail-loud post-condition that the marker parses back as root."""
    results: list[tuple[str, str]] = []
    for task in sorted(VERIFIER_ROOT_TASKS, key=int):
        if only is not None and task not in only:
            continue
        toml_path = shard_dir / task / "task.toml"
        if not toml_path.is_file():
            raise SystemExit(
                f"verifier-root target missing: {toml_path} — is the cache "
                f"populated? Run `--stage populate` first.",
            )
        new_text, status = apply_verifier_root_marker(toml_path.read_text())
        if status == "patched":
            toml_path.write_text(new_text)
        got = str(tomllib.loads(toml_path.read_text()).get("verifier", {}).get("user"))
        assert got == "root", (
            f"post-condition failed for {task}: [verifier] user not root ({got!r})"
        )
        results.append((task, status))
    if only:
        unknown = set(only) - VERIFIER_ROOT_TASKS
        if unknown:
            raise SystemExit(
                f"--tasks names not in VERIFIER_ROOT_TASKS: {sorted(unknown)}.",
            )
    return results


# ── Stage: patch (curated full-file overlays; repairs migration damage) ───────


def _apply_patch(task_dir: Path, patch_dir: Path) -> list[str]:
    """Overlay every file under ``patch_dir`` onto ``task_dir`` (full-file
    replacement, preserving relative layout + exec bits). A patch file with no
    counterpart is still copied (a patch may add a missing file). Returns the
    relative paths overridden. Mirrors terminalworld's ``_apply_patch``."""
    overridden: list[str] = []
    for f in sorted(patch_dir.rglob("*")):
        if not f.is_file() or f.name == "README.md":
            continue
        rel = f.relative_to(patch_dir)
        target = task_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, target)
        overridden.append(rel.as_posix())
    return overridden


def apply_all_patches(shard_dir: Path) -> int:
    """Apply every ``patches/<task_id>/`` overlay to its present task in the
    shard. Idempotent; a task absent from the shard is skipped with a note.
    Returns the number of tasks patched."""
    if not PATCHES_DIR.is_dir():
        return 0
    patched = 0
    for patch_dir in sorted(PATCHES_DIR.iterdir()):
        if not patch_dir.is_dir():
            continue  # skip README.md etc.
        tid = patch_dir.name
        dest = shard_dir / tid
        if not (dest / "task.toml").is_file():
            print(
                f"   [patch] {tid}: SKIP — not present in shard (populate first)",
                file=sys.stderr,
            )
            continue
        files = _apply_patch(dest, patch_dir)
        if files:
            patched += 1
            print(f"   [patch] {tid}: overrode {files}", file=sys.stderr)
    return patched


def restore_base_images(shard_dir: Path) -> int:
    """Rewrite the ``FROM`` line of each ``BASE_IMAGE_FIX_TASKS`` cache Dockerfile
    from bare ``ubuntu:24.04`` back to the original t-bench base (:data:`TBENCH_BASE`),
    which bakes the common tools the identical solve.sh assumes. Targeted +
    idempotent (a Dockerfile already on the t-bench base is left alone). A task
    absent from the shard is skipped. Returns the number of Dockerfiles rewritten.
    ``build_plan_gen`` builds these tasks ``type: local`` from the rewritten cache."""
    n = 0
    pat = re.compile(rf"^(FROM\s+){re.escape(_BARE_BASE)}(\s|$)", re.MULTILINE)
    for tid in sorted(BASE_IMAGE_FIX_TASKS):
        df = shard_dir / tid / "environment" / "Dockerfile"
        if not df.is_file():
            print(f"   [base] {tid}: SKIP — not present in shard", file=sys.stderr)
            continue
        text = df.read_text()
        if TBENCH_BASE in text:
            continue  # already restored
        new = pat.sub(rf"\g<1>{TBENCH_BASE}\g<2>", text)
        if new != text:
            df.write_text(new)
            n += 1
            print(f"   [base] {tid}: FROM {_BARE_BASE} -> t-bench base", file=sys.stderr)
        else:
            print(f"   [base] {tid}: WARN — no `FROM {_BARE_BASE}` line to rewrite",
                  file=sys.stderr)
    return n


# Dropped-command restore (Harbor-migration damage). The conversion dropped the
# compose ``command:`` from some single-service tasks, so harbor boots the bare image
# and the task's premise (a running service / spawned workers / seeded state) is
# absent → the oracle fails. We restore it by baking a boot wrapper as the task
# Dockerfile's ENTRYPOINT: ``( <command> ) & exec "$@"`` — the recovered command runs
# in the background to set the task up, then the wrapper execs harbor's CMD (which the
# raw-container path sets to ``sleep infinity``) as PID 1 to keep the container alive.
# The image ENTRYPOINT is preserved because the raw-container acquire sets docker CMD,
# not entrypoint; a baked ``CMD`` is dead (harbor overrides it). ``build_plan_gen``
# builds these ``type: local`` from the patched cache.
#
# Commands are recovered VERBATIM from the upstream ``Dataset/<id>/docker-compose.yaml``
# (READ FROM GIT — the repo's Dataset/ working tree may be a partial checkout; use
# ``git show HEAD:Dataset/<id>/docker-compose.yaml``). The ``sh -c "<payload>"`` wrapper
# is unwrapped to ``<payload>`` (our wrapper is already a shell); exec-form arrays are
# joined (e.g. ``[start.sh, sleep, infinity]`` → ``start.sh sleep infinity``).
#
# SCOPE — a task is included ONLY if it BOTH (a) failed the oracle AND (b) has no
# surviving Harbor Dockerfile ENTRYPOINT (a baked CMD doesn't count — harbor overrides
# it). Many tasks carry a dropped command yet PASS (their solve.sh or a build-time RUN
# does the setup); those are left untouched. 227 is oracle-proven; the rest are the
# same structural class and are gated per-task by the operator's oracle rebuild (still
# -failing ones get re-blacklisted with the real reason). Tasks that ALSO need a
# capability (cap_add/privileged) are NOT here — they need a sysbox marker too and are
# tracked separately (deferred). See STATUS.md "dropped-command".
# Validated 8/11 green under the oracle (2026-08-04). The 3 that still failed the oracle
# after the ENTRYPOINT restore need MORE than the dropped command (re-blacklisted with the
# real reason, black_list.txt): 775 (the on-disk config-drift the premise needs is not set
# up — the oracle reload correctly yields the unchanged config), 1246 (oracle timing race:
# stopped-worker log-freshness vs a 10 s test threshold), 1309 (dropped-command + SYSTEMD —
# grafana via `service grafana-server start`; needs a sysbox systemd marker too).
DROPPED_COMMAND_TASKS: dict[str, str] = {
    "26": "/usr/local/bin/start_rsync.sh && sleep infinity",
    "227": "/server/start_server.sh && sleep infinity",
    "412": "/start-services.sh",
    "475": "myserver infinity & sleep infinity",
    "669": "/usr/sbin/sshd && sleep infinity",
    "946": "/app/startup.sh",
    "1287": "/usr/bin/supervisord -n -c /etc/supervisor/conf.d/supervisord.conf",
    "1349": "/opt/start_services.sh sleep infinity",
}
_BOOT_SCRIPT = ".xrlenv-boot.sh"


def restore_dropped_commands(shard_dir: Path) -> int:
    """For ``DROPPED_COMMAND_TASKS``, bake a boot wrapper (``( <cmd> ) & exec "$@"``)
    as the task Dockerfile's ENTRYPOINT so the dropped compose command runs at boot.
    Writes the wrapper into the cache build context and appends COPY/ENTRYPOINT
    (idempotent — keyed on the wrapper already being referenced in the Dockerfile).
    Returns the number of Dockerfiles newly patched. ``build_plan_gen`` builds these
    ``type: local`` so the ENTRYPOINT lands in the rebuilt image."""
    n = 0
    for tid, cmd in sorted(DROPPED_COMMAND_TASKS.items()):
        env = shard_dir / tid / "environment"
        df = env / "Dockerfile"
        if not df.is_file():
            print(f"   [cmd] {tid}: SKIP — not present in shard", file=sys.stderr)
            continue
        (env / _BOOT_SCRIPT).write_text(
            "#!/bin/sh\n"
            "# xrlenv: restore the compose `command:` the Harbor migration dropped —\n"
            "# run it in the background to set the task up, then exec harbor's CMD\n"
            "# (sleep infinity on the raw-container path) as PID 1 to stay alive.\n"
            f"( {cmd} ) &\n"
            'exec "$@"\n',
        )
        (env / _BOOT_SCRIPT).chmod(0o755)
        text = df.read_text()
        if _BOOT_SCRIPT not in text:
            if not text.endswith("\n"):
                text += "\n"
            text += (
                "\n# xrlenv: restore the dropped compose command at boot\n"
                f"COPY {_BOOT_SCRIPT} /{_BOOT_SCRIPT}\n"
                f"RUN chmod +x /{_BOOT_SCRIPT}\n"
                f'ENTRYPOINT ["/{_BOOT_SCRIPT}"]\n'
            )
            df.write_text(text)
            n += 1
            print(f"   [cmd] {tid}: ENTRYPOINT restores `{cmd}`", file=sys.stderr)
    return n


def _report_sysbox(shard_dir: Path, only: list[str] | None) -> None:
    """Apply the sysbox markers and print a per-task report. Shared by the
    ``sysbox`` stage and the sysbox step of ``all``."""
    results = apply_all_sysbox_markers(shard_dir, only)
    print("\nsysbox markers:", file=sys.stderr)
    for task, status in results:
        mark = "+ marked" if status == "patched" else "= already marked"
        print(f"  {mark:18s} seta-env/{task}  -> sysbox-runc", file=sys.stderr)
    print(
        "\nThese tasks route to a node advertising sysbox-runc. Running them on a "
        "cluster with no sysbox node fails loud (expected). For a runc-only cache, "
        "use `--stage populate`, not `all`.",
        file=sys.stderr,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="build_cache",
        description=(
            "Materialize the seta-env task catalog into a shared harbor cache "
            "(a git clone into the seta-env/ shard). `all` (the default) also "
            "applies the curated migration-repair overlays (patches/) and the DinD "
            "sysbox routing markers — so one command yields a correct cache."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--stage", choices=("all", "populate", "sysbox"), default="all",
        help="all (default): populate (if missing) + apply patches/ overlays + "
        "write the sysbox markers, in order — the normal path. populate: clone the "
        "catalog only (needs network). sysbox: (re)write the SYSBOX_TASKS markers "
        "only. The migration-repair overlays are always applied by `all` (no "
        "separate stage). `all` is idempotent — safe to re-run.",
    )
    p.add_argument(
        "--dest", default=os.environ.get("XRLENV_BENCHMARK_CACHE"),
        help="Shared harbor cache ROOT (the shard lands under <dest>/seta-env/). "
        "Defaults to $XRLENV_BENCHMARK_CACHE, else ~/.cache/harbor/tasks.",
    )
    p.add_argument(
        "--tasks", default=None,
        help="For --stage sysbox: comma-separated subset of SYSBOX_TASKS to mark. "
        "Default: every task in SYSBOX_TASKS.",
    )
    p.add_argument(
        "--repo", default=DEFAULT_REPO,
        help=f"Upstream git URL (default {DEFAULT_REPO}).",
    )
    p.add_argument(
        "--ref", default=DEFAULT_REF,
        help=f"Upstream git ref — branch / tag / sha (default {DEFAULT_REF}).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    shard = _shard_dir(args.dest)
    only = (
        [t.strip() for t in args.tasks.split(",") if t.strip()]
        if args.tasks else None
    )

    # sysbox-only edits an EXISTING cache — fail loud if unpopulated.
    if args.stage == "sysbox":
        if not is_populated(shard):
            raise SystemExit(
                f"--stage sysbox needs a populated cache at {shard}. Run "
                f"`--stage populate` first.",
            )
        _report_sysbox(shard, only)
        return 0

    # all / populate: clone, then (for `all`) apply the curated migration-repair
    # overlays + the sysbox markers. There is no standalone `patch` stage — the
    # overlays are just part of building a correct cache (`all`), not a knob an
    # operator has to remember. `all` is idempotent, so re-running it on a
    # populated cache simply re-applies the overlays + markers.
    moved = populate(args.dest, repo=args.repo, ref=args.ref)
    if args.stage == "all":
        n = apply_all_patches(shard)
        print(f">> applied {n} migration-repair overlay(s)", file=sys.stderr)
        b = restore_base_images(shard)
        print(f">> restored the t-bench base image on {b} task(s)", file=sys.stderr)
        c = restore_dropped_commands(shard)
        print(f">> restored the dropped compose command on {c} task(s)", file=sys.stderr)
        vr = apply_all_verifier_root_markers(shard, only)
        vr_patched = sum(1 for _, s in vr if s == "patched")
        print(
            f">> set [verifier] user=root on {vr_patched}/{len(vr)} custom-user task(s)",
            file=sys.stderr,
        )
        _report_sysbox(shard, only)
    print(
        f"\n{_count_tasks(shard)} task(s) in {shard}"
        + (f" ({moved} newly cloned)" if moved else "")
        + f".\nPoint consumers at:  export XRLENV_BENCHMARK_CACHE={shard.parent}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
