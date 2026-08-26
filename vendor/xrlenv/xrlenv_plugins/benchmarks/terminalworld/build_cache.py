#!/usr/bin/env python3
"""Build a shared harbor task cache for the TerminalWorld ``verified`` split.

Self-contained pipeline, mirroring ``terminal_bench_2_1/build_cache.py`` in
shape (``--stage`` driven, idempotent), but with a TerminalWorld-specific
populate step because the dataset lives on the HuggingFace Hub, not a local
harbor export.

Stages
------
    populate  Pull the ``verified`` task list from ``EuniAI/TerminalWorld`` on
              the Hub, download only those ``artifacts/<id>.tar.gz``, extract
              each into ``<dest>/terminalworld-verified/<id>/``, and normalize
              its ``task.toml`` (drop the harbor-deprecated ``memory`` /
              ``storage`` string keys when the canonical ``_mb`` integers are
              also present — 97 of the 200 verified tasks won't load without
              this). Needs network + ``datasets`` / ``huggingface_hub``.
    patch     Overlay the curated ``patches/<task_id>/`` full-file fixes onto
              the extracted tasks (a partial reference ``solve.sh``, a missing
              verifier user, …). No network. See ``patches/README.md``.
    sysbox    Surgically mark the curated ``SYSBOX_TASKS`` for the sysbox pool by
              inserting ``[environment.env] XRLENV_CONTAINER_RUNTIME`` (+ the
              companion ``XRLENV_INNER_DOCKERD`` / ``XRLENV_INSTALL_DOCKERD`` /
              ``XRLENV_DOCKERD_LEGACY_STORE`` / ``XRLENV_SYSTEMD_INIT`` markers)
              into their ``task.toml``. Case-by-case: only the specific tasks in
              ``SYSBOX_TASKS`` get a marker (``--tasks`` narrows further). A
              marked task hard-fails on a cluster with no sysbox node
              (``BackendCapabilityMissing``), so a **runc-only** cache should use
              ``--stage patch``, not ``all``.
    all       The full one-command setup: populate (only if missing), then patch,
              then sysbox. Runs the three in the order that matters (sysbox last,
              so a task.toml overlay from ``patch`` can't clobber a marker). The
              default; use it to prep a cache for a full run on a sysbox-capable
              cluster.

The shard name (``terminalworld-verified``) doubles as the image namespace: a
task lands at ``<dest>/terminalworld-verified/<id>/`` and its image at
``<registry>/terminalworld-verified/<id>:main`` (built by the sibling
``xrlenv_plugins/benchmarks/<name>/`` build scripts). Keeping the two identical is what
lets the oracle sweep resolve ``dataset`` and image ref from the one shard name.

How xrlenv consumers pick it up
-------------------------------
xrlenv's harbor onboarding resolves a task by local *path* (``TaskConfig(path=…)``
→ harbor ``LocalTaskId``), searching under ``$XRLENV_BENCHMARK_CACHE``. A local task
is used as-is — harbor never re-downloads or overwrites it — so the normalized
``task.toml`` + patched ``solve.sh`` this script writes are authoritative. Point
every consumer's ``XRLENV_BENCHMARK_CACHE`` at the shared root this script writes.

Env overrides (populate):
    TW_REPO_ID           HF dataset id        (default EuniAI/TerminalWorld)
    TW_CONFIG            dataset config name  (default verified)
    TW_SPLIT             split name           (default test)
    TW_SHARD             cache shard == image namespace (default terminalworld-verified)
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tarfile
import tomllib
from dataclasses import dataclass
from pathlib import Path

# ── Dataset identity ──────────────────────────────────────────────────────────
REPO_ID = os.environ.get("TW_REPO_ID", "EuniAI/TerminalWorld")
CONFIG = os.environ.get("TW_CONFIG", "verified")
SPLIT = os.environ.get("TW_SPLIT", "test")
# Shard subdir name == image namespace. The name every consumer's shard-scan
# sees and the sibling benchmark build scripts push under.
SHARD = os.environ.get("TW_SHARD", "terminalworld-verified")

# Curated per-task fixes live beside this script. Each ``patches/<task_id>/<rel>``
# file is overlaid (full-file replacement) onto the extracted task dir AFTER
# extraction+normalization, so the fixes survive re-populate. See patches/README.md.
PATCHES_DIR = Path(__file__).resolve().parent / "patches"


# ── Stage 1: populate (HF download + extract + task.toml normalize) ───────────


def _safe_extractall(tf: tarfile.TarFile, dest: Path) -> None:
    """Extract with the ``data`` filter where available (Python >=3.12); fall
    back to a plain extract on older interpreters."""
    try:
        tf.extractall(dest, filter="data")  # type: ignore[call-arg]
    except TypeError:
        tf.extractall(dest)  # older interpreter without the data filter


def normalize_task_toml_text(text: str) -> tuple[str, bool]:
    """Strip harbor-deprecated ``memory`` / ``storage`` string keys from a
    task.toml's ``[environment]`` section when the canonical ``memory_mb`` /
    ``storage_mb`` are also present. Returns ``(new_text, changed)``.

    TerminalWorld's generation pipeline emits BOTH the deprecated string field
    and the canonical ``_mb`` integer, and ~half the verified set sets them
    inconsistently (e.g. ``memory = "2G"`` alongside ``memory_mb = 4096``).
    harbor's ``EnvironmentConfig`` rejects that conflict outright, so those
    tasks won't load at all. ``memory``/``storage`` are deprecated in harbor (it
    migrates them INTO ``memory_mb``/``storage_mb`` when the canonical field is
    absent), so the least-lossy fix is to drop the deprecated duplicate and keep
    harbor's canonical field. Edits are surgical (line-level) to preserve the
    rest of the file, including the harbor-canary comment.

    Pure function (no I/O) so the logic is unit-testable in isolation.
    """
    env = tomllib.loads(text).get("environment", {})
    strip = {
        legacy
        for legacy, canonical in (("memory", "memory_mb"), ("storage", "storage_mb"))
        if legacy in env and canonical in env
    }
    if not strip:
        return text, False

    out: list[str] = []
    section: str | None = None
    for line in text.splitlines(keepends=True):
        header = re.match(r"\s*\[([^\]]+)\]", line)
        if header:
            section = header.group(1).strip()
        if section == "environment":
            key = re.match(r"\s*([A-Za-z_]+)\s*=", line)
            if key and key.group(1) in strip:
                continue  # drop the deprecated duplicate
        out.append(line)
    return "".join(out), True


def _normalize_task_toml(path: Path) -> bool:
    new_text, changed = normalize_task_toml_text(path.read_text(encoding="utf-8"))
    if changed:
        path.write_text(new_text, encoding="utf-8")
    return changed


def _locate_task_dir(extracted: Path, task_id: str) -> Path:
    """Find the dir holding ``task.toml`` inside the extracted tree. The tarball
    nests everything under ``<task_id>/``, but accept a flat layout too."""
    if (extracted / task_id / "task.toml").is_file():
        return extracted / task_id
    if (extracted / "task.toml").is_file():
        return extracted
    matches = sorted(extracted.rglob("task.toml"))
    if len(matches) != 1:
        raise SystemExit(
            f"ERROR: {task_id}: expected exactly one task.toml in the artifact, "
            f"found {len(matches)}",
        )
    return matches[0].parent


def populate(shard_dir: Path) -> tuple[int, int]:
    """Download the ``verified`` split into ``shard_dir`` (idempotent). Returns
    ``(moved, normalized)``. Skips tasks already present, so it's safe to re-run
    and coexists with the tb2 / seta / turing shards in the same cache."""
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    print(f">> Loading {REPO_ID} [{CONFIG}/{SPLIT}] task list", file=sys.stderr)
    ds = load_dataset(REPO_ID, CONFIG, split=SPLIT)
    rows = [(r["task_id"], r["artifact_path"]) for r in ds]

    shard_dir.mkdir(parents=True, exist_ok=True)
    todo = [
        (tid, ap) for tid, ap in rows
        if not (shard_dir / tid / "task.toml").is_file()
    ]
    print(
        f">> {len(rows)} verified task(s); {len(rows) - len(todo)} already "
        f"present, {len(todo)} to fetch into {shard_dir}",
        file=sys.stderr,
    )

    # Stage extraction on the same filesystem as the cache so the final move into
    # the shard is a fast rename, not a cross-fs copy. Cleaned up on exit.
    staging = shard_dir.parent.parent / f".terminalworld-staging.{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)

    moved = 0
    normalized = 0
    try:
        for i, (tid, ap) in enumerate(todo, 1):
            tar_path = hf_hub_download(REPO_ID, ap, repo_type="dataset")
            work = staging / tid
            shutil.rmtree(work, ignore_errors=True)
            work.mkdir(parents=True)
            with tarfile.open(tar_path) as tf:
                _safe_extractall(tf, work)
            src = _locate_task_dir(work, tid)
            dest = shard_dir / tid
            shutil.rmtree(dest, ignore_errors=True)
            shutil.move(str(src), str(dest))
            moved += 1
            if _normalize_task_toml(dest / "task.toml"):
                normalized += 1
            if i % 25 == 0 or i == len(todo):
                print(f"   [{i}/{len(todo)}] {tid}", file=sys.stderr)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return moved, normalized


# ── Stage 2: patch (curated full-file overlays) ───────────────────────────────


def _apply_patch(task_dir: Path, patch_dir: Path) -> list[str]:
    """Overlay every file under ``patch_dir`` onto ``task_dir`` (full-file
    replacement), preserving relative layout and exec bits. Returns the relative
    paths overridden, for logging. A patch file with no counterpart in the task
    is still copied (lets a patch add a missing file)."""
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
    shard. Idempotent; a task absent from the shard is skipped with a SKIP note.
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


# ── task.toml cpuset-pinning markers (correctness fix, safe on any cluster) ───
# Tasks whose solve.sh sizes build parallelism to ``nproc`` (``make -j$(nproc)``).
# harbor applies only a CFS quota + memory cap and never sets cpuset, so ``nproc``
# inside the container reports the *host* core count (e.g. 192 on the dev node),
# not the task's declared ``cpus``. That fans the build out to ~host-count jobs →
# OOM and/or a race on generated headers (tw_179356: ``No rule to make target
# build/build.h`` when version.o compiles before genbuild.sh runs). The harbor
# plugin reads ``[environment.env] XRLENV_CPU_PINNING`` and sizes the affinity
# mask to the declared ``cpus`` so ``nproc`` == the task budget. Same mechanism
# as terminal_bench_2_1's ENV_PATCHES. Applied in the ``patch`` stage.
CPU_PINNING_TASKS: tuple[str, ...] = (
    "tw_179356",  # Denarius Qt build: make -j$(nproc) races on build/build.h at host nproc
    "tw_245032",  # openal-soft + fmt-11.2 build: make -j$(nproc) at host nproc (192) inside an
                  # 8GB/4cpu container OOM-kills cc1plus (heavy templates); pin → nproc=4 fits
                  # (6→11/12 tests). NOT yet green: the verifier hardcodes libopenal.so.1.25.1
                  # but the baked image is 1.25.2 (no tag/network to pin 1.25.1) — needs an
                  # image rebuild at 1.25.1. Pin kept so it passes once the image is aligned.
    # Also pin the two contention-flaky tasks so they get their DECLARED cores
    # dedicated (no CFS-quota sharing with concurrent trials on the host):
    "tw_528959",  # EXCLUDED 2026-07-08 (STATUS.md Failed): CPython-from-source (make -j2,
                  # cpus=2) busts its own 2700s ceiling EVEN uncontended — unreasonable task
                  # config. Marker left inert (run_full_sweep.sh EXCLUDE drops it); only bites
                  # if you deliberately re-run it with a bumped --timeout-multiplier.
    "tw_234227",  # gdb/ptrace backtrace (cpus=1): needs stable timing to be deterministic
    "tw_650591",  # sysbox DinD (ip netns / runc run / mounts): timing-sensitive verifier
                  # flaked to reward 0 under conc-32 CPU contention (2026-07-08 full sweep,
                  # the lone straggler at 186/187 — no exception, oracle ran, verifier
                  # scored 0). Pin → dedicated cores → stable DinD timing → reliable pass.
)


def apply_cpu_pinning(shard_dir: Path) -> list[tuple[str, str]]:
    """Insert ``[environment.env] XRLENV_CPU_PINNING = "1"`` into each
    ``CPU_PINNING_TASKS`` task.toml (surgical, idempotent). Returns
    ``[(task, status), …]``."""
    results: list[tuple[str, str]] = []
    for tid in CPU_PINNING_TASKS:
        toml_path = shard_dir / tid / "task.toml"
        if not toml_path.is_file():
            raise SystemExit(
                f"cpu-pinning target missing: {toml_path} — populate first.",
            )
        new_text, changed = _ensure_table_key(
            toml_path.read_text(), "environment.env", "XRLENV_CPU_PINNING", "1",
        )
        if changed:
            toml_path.write_text(new_text)
        results.append((tid, "patched" if changed else "already"))
    return results


# ── Stage 2b: drop a redundant compose ``privileged: true`` ───────────────────

# Multi-service tasks whose compose declares a blanket ``privileged: true`` on
# top of the ``cap_add: [NET_ADMIN, NET_RAW]`` they actually enumerate. These are
# iptables/routing stacks — configuring rules in a service's own network
# namespace needs only those two Level-1 caps, which the control-plane
# ``KwargsPolicy`` already permits by default. The extra ``privileged`` is
# over-broad (all caps + /dev + relaxed seccomp/AppArmor + cgroup rw) and forces
# either a cluster-wide ``allow_privileged: true`` opt-in or a sysbox build-out.
# Dropping the redundant flag lets these run under plain runc with just the
# permitted caps — no ``allow_privileged``, no sysbox. If a task turns out to
# genuinely need more than the caps, drop it from this list (or remove the whole
# transform) and re-populate to restore the authored compose.
COMPOSE_DROP_PRIVILEGED: tuple[str, ...] = (
    "tw_304270",  # st2/lb iptables peers on 172.16.70.0/24
    "tw_304271",  # same shape on 10.71.238.0/24
    "tw_305044",  # stapp iptables peers on 192.168.20.0/24
)

# A block-style ``privileged: <truthy>`` line (the only form the corpus uses),
# with an optional trailing comment. We remove the whole line.
_PRIVILEGED_TRUE_RE = re.compile(
    r"^[ \t]*privileged:[ \t]*(true|yes|on|1)[ \t]*(#.*)?$", re.IGNORECASE,
)


def _task_compose_path(task_dir: Path) -> Path | None:
    for name in ("docker-compose.yaml", "docker-compose.yml"):
        p = task_dir / "environment" / name
        if p.is_file():
            return p
    return None


def drop_compose_privileged(shard_dir: Path) -> list[tuple[str, str]]:
    """Strip the redundant service-level ``privileged: true`` from each
    ``COMPOSE_DROP_PRIVILEGED`` task's compose (keeping its ``cap_add``). Surgical
    (line-level — the rest of the file is byte-preserved), idempotent, and
    validated: after the strip the doc must still parse and no service may retain a
    truthy ``privileged`` (else we leave the file untouched and fail loud rather
    than write a half-fixed compose). Returns ``[(task, status), …]``."""
    import yaml

    results: list[tuple[str, str]] = []
    for tid in COMPOSE_DROP_PRIVILEGED:
        compose_path = _task_compose_path(shard_dir / tid)
        if compose_path is None:
            raise SystemExit(
                f"compose-privilege target missing: {shard_dir / tid}/environment/"
                f"docker-compose.yaml — populate first.",
            )
        original = compose_path.read_text()
        new_text = "".join(
            ln for ln in original.splitlines(keepends=True)
            if not _PRIVILEGED_TRUE_RE.match(ln.rstrip("\n"))
        )
        # Always validate the RESULT carries no truthy ``privileged`` — this also
        # catches forms the line-strip can't reach (e.g. flow-style
        # ``main: {privileged: true}``): fail loud rather than silently report
        # "already" and leave a service the CP vet will then reject.
        doc = yaml.safe_load(new_text) or {}
        services = doc.get("services") if isinstance(doc, dict) else None
        stragglers = [
            name for name, svc in (services or {}).items()
            if isinstance(svc, dict) and svc.get("privileged")
        ]
        if stragglers:
            raise SystemExit(
                f"{tid}: privileged remains on service(s) {stragglers} after the "
                f"line-strip (unhandled compose form) — refusing to leave a "
                f"half-fixed compose.",
            )
        changed = new_text != original
        if changed:
            compose_path.write_text(new_text)
        results.append((tid, "patched" if changed else "already"))
    return results


# ── Stage 3: sysbox markers (opt-in, case-by-case task.toml env inserts) ──────


@dataclass(frozen=True)
class SysboxTask:
    """A task routed to the sysbox pool by a surgical ``[environment.env]``
    marker in its ``task.toml``. ``inner_dockerd`` additionally opts the task
    into the harbor cluster plugin's dockerd bring-up (``exec dockerd`` + wait
    for the socket) — for DinD tasks whose image ships a docker daemon but
    starts nothing (a real TerminalWorld VM boots it via systemd; harbor's
    DooD host-socket mount was the shortcut xrlenv correctly refuses).

    ``agent_user`` / ``verifier_user`` (optional) set ``[agent] user`` /
    ``[verifier] user`` — harbor honors these (``trial.py`` runs the agent/
    verifier as that user). Needed by system-admin tasks whose image bakes a
    non-root ``USER`` but whose reference solve.sh does root-only ops
    (``ip netns``, ``runc``, mounts). Faithful — the reference solution
    genuinely needs root, exactly like the ``[verifier] user = "root"`` fix for
    an image that drops privileges before the tests. Kept in the opt-in sysbox
    path (not ``--stage patch``) because the root need is entailed by routing
    these tasks to the nested-runtime pool."""

    task: str
    runtime: str
    inner_dockerd: bool
    reason: str
    agent_user: str | None = None
    verifier_user: str | None = None
    # Bring the nested dockerd up with the legacy image store (schema2 pushes,
    # not OCI) — for a task whose docker tooling only understands schema2
    # manifests (e.g. tw_709166's dockdiver). Faithful to the older-docker env
    # the task targets. Sets [environment.env] XRLENV_DOCKERD_LEGACY_STORE.
    dockerd_legacy_store: bool = False
    # Install the docker engine before starting it — for a CLI-only DooD image
    # (docker-ce-cli, no daemon; it relied on the host socket). Sets
    # [environment.env] XRLENV_INSTALL_DOCKERD.
    install_dockerd: bool = False
    # Boot the container with systemd as PID 1 (acquire command=[/sbin/init])
    # for a task that needs a real init (`systemctl start …`, running services).
    # Sysbox provides unprivileged systemd PID 1. Sets [environment.env]
    # XRLENV_SYSTEMD_INIT.
    systemd_init: bool = False


# The sysbox recovery set, grown one proven task at a time (see
# tmp/sysbox-terminalworld-recovery-plan.md). Start with the decisive probe and
# the other DinD tasks whose image already ships a docker daemon (docker.io) —
# the clean nested-dockerd case, no in-container engine install needed. CLI-only
# DinD images (docker-ce-cli) and systemd/netns tasks are deferred until the
# daemon-shipping set is proven green.
SYSBOX_TASKS: tuple[SysboxTask, ...] = (
    SysboxTask(
        task="tw_245733",
        runtime="sysbox-runc",
        inner_dockerd=True,
        reason=(
            "decisive probe: solve.sh is `docker pull ubuntu:latest`; image "
            "ships docker.io (daemon present); single-service; non-privileged."
        ),
    ),
    SysboxTask(
        task="tw_247958",
        runtime="sysbox-runc",
        inner_dockerd=True,
        reason="DinD daemon-shipper (docker.io), single-service, non-privileged.",
    ),
    SysboxTask(
        task="tw_709166",
        runtime="sysbox-runc",
        inner_dockerd=True,
        # dockdiver only understands docker schema2 manifests; a modern nested
        # dockerd's containerd store pushes OCI → dockdiver's dump 404s. Legacy
        # store restores schema2. (The exec-based sysbox upload path is what lets
        # the legacy/overlay2 store work — it sidesteps the put_archive resolv.conf
        # 500 that overlay2 otherwise triggers.)
        dockerd_legacy_store=True,
        reason=(
            "DinD daemon-shipper (docker.io); dockdiver builds + dumps a local "
            "registry image. compose privileged:true is dropped — sysbox grants "
            "the caps unprivileged. Needs a solve.sh completion patch (shipped "
            "oracle writes an empty result.txt) + legacy image store (dockdiver "
            "is schema2-only)."
        ),
    ),
    SysboxTask(
        task="tw_650591",
        runtime="sysbox-runc",
        inner_dockerd=True,
        # Image bakes `USER user`, but the solve does root-only ops (ip netns,
        # runc run, mounts) → run the oracle + verifier as root. Faithful: it's
        # a system-admin task that genuinely needs root.
        agent_user="root",
        verifier_user="root",
        reason=(
            "DinD + unprivileged netns: docker export busybox, runc run, and "
            "ip netns/link/bridge/veth — all under sysbox without host privilege "
            "(compose privileged:true dropped). Image is USER user; needs a root "
            "oracle for the netns/runc ops. The hardest daemon-shipper."
        ),
    ),
    SysboxTask(
        task="tw_313581",
        runtime="sysbox-runc",
        inner_dockerd=False,
        # NOT systemd_init: CentOS 7's systemd v219 can't bring up D-Bus under
        # sysbox ("Operation not permitted"), so systemctl is non-functional for
        # these tasks. Instead the solve.sh patch starts dbus + firewalld in the
        # foreground (the way the image's own Dockerfile documents), which needs
        # sysbox's unprivileged NET_ADMIN for firewalld to manage nftables.
        reason=(
            "firewalld: shipped solve wrote the zone XML but never started the "
            "daemon (test_firewalld_process_running fails). solve.sh patch starts "
            "dbus + firewalld manually under sysbox NET_ADMIN, then adds the "
            "permanent 3306/tcp rule."
        ),
    ),
    # Investigation batch (2026-07-07): high-probability sysbox candidates from
    # the STATUS.md "need investigation" set — DinD (docker-API failures) and
    # single-service netns/iptables (NET_ADMIN failures, like the firewalld win).
    SysboxTask(
        task="tw_586787", runtime="sysbox-runc", inner_dockerd=True,
        reason="DinD (docker.io) + ip netns + runc; nested dockerd under sysbox.",
    ),
    SysboxTask(
        task="tw_435744", runtime="sysbox-runc", inner_dockerd=True, install_dockerd=True,
        reason="CLI-only DinD: docker registry + build + oras; install engine, then nest.",
    ),
    SysboxTask(
        task="tw_526185", runtime="sysbox-runc", inner_dockerd=True, install_dockerd=True,
        reason="CLI-only DinD: docker exec trino/redis; install engine, then nest.",
    ),
    SysboxTask(
        task="tw_222108", runtime="sysbox-runc", inner_dockerd=False,
        reason="netns/veth (ip link/netns) — sysbox unprivileged NET_ADMIN.",
    ),
    SysboxTask(
        task="tw_583114", runtime="sysbox-runc", inner_dockerd=False,
        reason="runc + netns — sysbox unprivileged runc/netns.",
    ),
    SysboxTask(
        task="tw_305688", runtime="sysbox-runc", inner_dockerd=False,
        reason="iptables (nftables) — sysbox unprivileged NET_ADMIN.",
    ),
    SysboxTask(
        task="tw_333762", runtime="sysbox-runc", inner_dockerd=False,
        reason="iptables (solve is systemd-aware, falls back to ln -sf) — sysbox NET_ADMIN.",
    ),
    # NOTE: tw_18948 / tw_7829 (lldb/gdb debugger) are NOT sysbox-marked — their
    # `'A' packet error 8` / no-crash failures were a seccomp/personality issue
    # (debuggers disable ASLR via personality()), fixed runtime-independently in the
    # solve overlays (`disable-aslr false` / `disable-randomization off`), not sysbox.
    SysboxTask(
        task="tw_118507", runtime="sysbox-runc", inner_dockerd=False,
        reason="singularity SIF build from docker://ubuntu needs userns (rootless "
        "build); truncates at 'Exploding layer' under plain runc.",
    ),
    # CLI-only DinD group: images ship docker-ce-cli ONLY (no daemon — DooD by
    # design, relying on the host socket). install_dockerd apt-installs docker-ce
    # from the image's configured repo, then nests it like the daemon-shippers.
    *(
        SysboxTask(
            task=t,
            runtime="sysbox-runc",
            inner_dockerd=True,
            install_dockerd=True,
            reason="CLI-only DinD (docker-ce-cli, no daemon) — install docker-ce, then nest.",
        )
        for t in (
            "tw_27806", "tw_333322", "tw_16553", "tw_347571", "tw_11696",
            "tw_582345", "tw_105786", "tw_268653", "tw_420790", "tw_27037",
        )
    ),
)


def _table_lookup(parsed: dict, table: str) -> dict:
    """Walk a dotted TOML table path (e.g. ``environment.env``) in a parsed
    doc, returning the sub-table dict (or ``{}`` if absent)."""
    node: object = parsed
    for part in table.split("."):
        node = node.get(part, {}) if isinstance(node, dict) else {}
    return node if isinstance(node, dict) else {}


def _ensure_table_key(text: str, table: str, key: str, value: str) -> tuple[str, bool]:
    """Surgically ensure ``[table]`` carries ``key = "value"`` in a task.toml.

    If ``key`` is already present in ``[table]`` with that value, returns
    ``(text, False)``. Otherwise inserts ``key = "value"`` right under the
    existing ``[table]`` header, or appends a fresh ``[table]`` block at EOF (a
    valid standalone TOML table). Only ADDS a missing key — never rewrites an
    existing one (we control the marker values). Returns ``(new_text, changed)``.
    Pure function so the logic is unit-testable in isolation.
    """
    if str(_table_lookup(tomllib.loads(text), table).get(key)) == value:
        return text, False
    entry = f'{key} = "{value}"\n'
    header = re.compile(r"^\[" + re.escape(table) + r"\]\s*$")
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if header.match(line):
            lines.insert(i + 1, entry)  # first entry inside the table
            return "".join(lines), True
    tail = "".join(lines)
    if tail and not tail.endswith("\n"):
        tail += "\n"
    return tail + f"\n[{table}]\n" + entry, True


def apply_sysbox_marker(text: str, spec: SysboxTask) -> tuple[str, str]:
    """Apply a task's sysbox routing markers (+ optional run-as-user) to a
    task.toml given as ``text``.

    Ensures ``[environment.env]`` carries ``XRLENV_CONTAINER_RUNTIME`` (+
    ``XRLENV_INNER_DOCKERD`` when ``spec.inner_dockerd``), and — when set —
    ``[agent] user`` / ``[verifier] user``. Each key is inserted surgically
    (under its table header, or a fresh table appended at EOF), composing with
    the curated ``patches/`` overlays without clobbering them. Returns
    ``(new_text, status)`` where ``status`` is ``"patched"`` (something changed)
    or ``"already"`` (all present). Pure function — unit-testable in isolation.
    """
    pairs: list[tuple[str, str, str]] = [
        ("environment.env", "XRLENV_CONTAINER_RUNTIME", spec.runtime),
    ]
    if spec.inner_dockerd:
        pairs.append(("environment.env", "XRLENV_INNER_DOCKERD", "1"))
    if spec.dockerd_legacy_store:
        pairs.append(("environment.env", "XRLENV_DOCKERD_LEGACY_STORE", "1"))
    if spec.install_dockerd:
        pairs.append(("environment.env", "XRLENV_INSTALL_DOCKERD", "1"))
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
    their task.toml. Returns ``[(task, status), …]``. Verifies each result
    parses and carries the markers (fail-loud post-condition)."""
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


# ── Helpers ───────────────────────────────────────────────────────────────────


def _count_tasks(shard_dir: Path) -> int:
    return sum(1 for _ in shard_dir.glob("*/task.toml"))


def is_populated(shard_dir: Path) -> bool:
    return shard_dir.is_dir() and _count_tasks(shard_dir) > 0


# ── CLI ───────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="build_cache",
        description=(
            "Materialize the TerminalWorld verified split into a shared harbor "
            "cache. Stages (each idempotent): populate -> patch -> sysbox; "
            "`all` (default) runs all three in that order."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--stage",
        choices=("all", "populate", "patch", "sysbox"),
        default="all",
        help="all (default): the full setup — populate (if missing) + patch + "
        "sysbox, in order. populate: download+normalize only (needs network). "
        "patch: curated solve.sh/task.toml overlays only (safe on any cluster). "
        "sysbox: the SYSBOX_TASKS routing markers only. For a RUNC-ONLY cache "
        "(no sysbox node), use `patch`, not `all`.",
    )
    p.add_argument(
        "--dest",
        default=os.environ.get("XRLENV_BENCHMARK_CACHE"),
        help="Shared harbor cache ROOT (the shard lands under "
        f"<dest>/{SHARD}/). Defaults to $XRLENV_BENCHMARK_CACHE. Point every "
        "xrlenv consumer's XRLENV_BENCHMARK_CACHE at this path.",
    )
    p.add_argument(
        "--tasks",
        default=None,
        help="For --stage sysbox: comma-separated subset of SYSBOX_TASKS to "
        "mark. Default: every task in SYSBOX_TASKS.",
    )
    return p


def _resolve_shard(dest: str | None) -> Path:
    # Fail loud BEFORE resolving: a caller still pointing at the retired
    # XRLENV_HARBOR_CACHE var/path would materialize the shard under the wrong
    # (stale/absent) cache. Guard first, then keep the SHARD-append + "no
    # destination" error behavior unchanged.
    from xrlenv_plugins.benchmarks._benchmark_cache import guard_legacy_cache_env
    guard_legacy_cache_env(dest)
    if not dest:
        raise SystemExit(
            "error: no destination — pass --dest or set XRLENV_BENCHMARK_CACHE.",
        )
    return Path(dest).expanduser() / SHARD


def _report_sysbox(shard_dir: Path, only: list[str] | None) -> None:
    """Apply the sysbox markers and print a per-task report. Shared by the
    ``sysbox`` stage and the sysbox step of ``all``."""
    results = apply_all_sysbox_markers(shard_dir, only)
    print("\nsysbox markers:", file=sys.stderr)
    for task, status in results:
        mark = "+ marked" if status == "patched" else "= already marked"
        print(f"  {mark:18s} {task}  -> sysbox-runc", file=sys.stderr)
    print(
        "\nThese tasks route to a node advertising sysbox-runc. Running them on "
        "a cluster with no sysbox node fails loud (BackendCapabilityMissing) — "
        "expected. For a runc-only cache, re-populate/patch without --stage all.",
        file=sys.stderr,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    shard_dir = _resolve_shard(args.dest)
    shard_dir.parent.mkdir(parents=True, exist_ok=True)

    only = (
        [t.strip() for t in args.tasks.split(",") if t.strip()]
        if args.tasks else None
    )

    if args.stage == "sysbox":
        if not is_populated(shard_dir):
            raise SystemExit(
                f"cannot mark: {shard_dir} is not populated. Run "
                f"`--stage populate` (or `--stage all`) first.",
            )
        _report_sysbox(shard_dir, only)
        return 0

    moved = normalized = 0
    if args.stage in ("all", "populate"):
        if args.stage == "populate" or not is_populated(shard_dir):
            moved, normalized = populate(shard_dir)
        else:
            print(
                f">> {shard_dir} already populated "
                f"({_count_tasks(shard_dir)} tasks) — skipping download.",
                file=sys.stderr,
            )
    if args.stage == "populate":
        print(
            f"\npopulated {moved} task(s) ({normalized} task.toml normalized) "
            f"-> {shard_dir}",
            file=sys.stderr,
        )
        return 0

    # patch (also reached by --stage all)
    if not is_populated(shard_dir):
        raise SystemExit(
            f"cannot patch: {shard_dir} is not populated. Run "
            f"`--stage populate` first (or `--stage all`).",
        )
    patched = apply_all_patches(shard_dir)
    pinned = apply_cpu_pinning(shard_dir)
    n_pinned = sum(1 for _, s in pinned if s == "patched")
    dropped = drop_compose_privileged(shard_dir)
    n_dropped = sum(1 for _, s in dropped if s == "patched")
    total = _count_tasks(shard_dir)
    print(
        f"\nOK: {total} task(s) in {shard_dir}"
        + (f" ({moved} onboarded, {normalized} normalized)" if moved else "")
        + f"; applied {patched} curated patch(es)"
        + (f", {n_pinned} cpu-pinning marker(s)" if n_pinned else "")
        + (f", {n_dropped} compose privilege-drop(s)" if n_dropped else "")
        + ".",
        file=sys.stderr,
    )

    # sysbox — the final step of `all` (after patch, so a task.toml overlay
    # can't clobber a marker). `--stage patch` stops here (runc-only cache).
    if args.stage == "all":
        _report_sysbox(shard_dir, only=None)

    print(
        f"Point consumers at:  export XRLENV_BENCHMARK_CACHE={shard_dir.parent}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
