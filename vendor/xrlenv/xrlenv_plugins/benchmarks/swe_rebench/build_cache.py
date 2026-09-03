#!/usr/bin/env python3
"""Build a shared harbor task cache for the SWE-rebench leaderboard benchmark.

Self-contained pipeline, mirroring ``frontier_swe/build_cache.py`` in shape
(``--stage`` driven, idempotent), retargeted at **SWE-rebench** — a
harbor-format task corpus published as a **Harbor Hub package dataset**
(``swe-rebench/swe-rebench-leaderboard``, 860 curated Python SWE tasks from
Nebius AI R&D). Each task unpacks to ``task.toml`` + ``instruction.md`` +
``environment/Dockerfile`` + ``tests/{test.sh,parser.py,config.json}`` +
``solution/solve.sh`` — the exact harbor filesystem contract
``TaskConfig(path=…)`` consumes. All 860 ship a reference solution, so the
whole corpus is oracle-gateable.

Stages
------
    populate  Download the dataset from the Harbor Hub into
              ``<dest>/swe-rebench/<task_id>/`` and normalize each
              ``task.toml``. Needs network (anonymous — the dataset is
              public); no git clone, no HF token.
    repin     Write each task's authoritative prebuilt image into its
              ``task.toml`` as ``[environment] docker_image``. See
              "Why repin" below.
    patch     Apply the curated content fixes, of two kinds: full-file
              ``patches/<id>/`` overlays (starts EMPTY), and programmatic
              ``task.toml`` edits — resource routing (``XRLENV_CPU_PINNING``
              markers + fair memory overrides, :data:`CPU_PINNING_TASKS` /
              :data:`MEMORY_OVERRIDES`) and hermeticity env
              (:data:`HERMETICITY_ENV`). Same two-kind split
              terminal_bench_2_1 uses (its ``PATCHES`` + ``ENV_PATCHES``).
    all       populate (only if missing) -> repin -> patch. The default.

Why repin
---------
SWE-rebench tasks carry **no** ``[environment] docker_image``. Upstream ships a
prebuilt image per task on Docker Hub (``swerebench/sweb.eval.x86_64.<slug>``)
and expresses it as the ``FROM`` of a three-line ``environment/Dockerfile``
that adds ``ENV _JAVA_OPTIONS=""``, an ``uv`` install, and ``mkdir -p /logs``.
The xrlenv harbor cluster environment resolves an image ref at acquire; it does
not build on acquire. So ``--stage repin`` writes the authoritative upstream ref
into ``task.toml``, turning the whole corpus into a pull-on-demand
``type: registry`` plan with **nothing to build**.

The ref is read from the task's own ``tests/config.json`` (upstream's declared
``docker_image``) and **cross-checked against the Dockerfile's ``FROM``** — a
mismatch fails loud rather than silently pinning the wrong image.

What repinning drops, and why it is safe
----------------------------------------
Two of the Dockerfile's three lines are provably inert under harbor 0.20:

* ``RUN mkdir -p /logs`` — harbor creates the log tree itself. Its
  ``empty_dirs`` (``harbor/environments/base.py``) runs
  ``mkdir -p /logs/verifier && chmod 777`` in-container before the verifier
  phase, and every one of the 860 ``tests/test.sh`` opens with its own
  ``mkdir -p /logs/verifier``.
* ``ENV _JAVA_OPTIONS=""`` — a guard against a JVM echoing
  ``Picked up _JAVA_OPTIONS`` into parsed test output. No task's ``test.sh``
  or ``solve.sh`` references it, and no base image sets it.
* the ``uv`` install — **the base images already ship uv.** 17 tasks' verifiers
  run ``uv run pytest`` / ``uv pip install``; all 17 were run on-cluster against
  the plain upstream image with nothing built (2026-09-01) and 16 solved, the
  17th failing on an unrelated upstream packaging defect. No trial produced
  ``uv: command not found``, and passing trials show uv working from
  ``/root/.cache/uv/``. See STATUS.md.


Env overrides (populate):
    SWE_REBENCH_DATASET       Harbor Hub package (default
                              swe-rebench/swe-rebench-leaderboard)
    SWE_REBENCH_DATASET_REF   package ref/version (default latest)
    SWE_REBENCH_SHARD         cache shard name (default swe-rebench)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tomllib
from pathlib import Path

# ── Dataset identity ──────────────────────────────────────────────────────────
# The Harbor Hub package dataset. Resolved anonymously through harbor's own
# registry client (``PackageDatasetClient``) — it is public, so no HARBOR_API_KEY
# / login is needed. Overridable so a fork or a pinned snapshot can be swapped in
# without a code edit.
DATASET = os.environ.get("SWE_REBENCH_DATASET", "swe-rebench/swe-rebench-leaderboard")
DATASET_REF = os.environ.get("SWE_REBENCH_DATASET_REF", "latest")
# Shard subdir name == the namespace every consumer enumerates (one name, double
# duty — same convention as the frontier-swe / deep-swe shards).
SHARD = os.environ.get("SWE_REBENCH_SHARD", "swe-rebench")

# Curated per-task fixes live beside this script. Each ``patches/<id>/<rel>`` file
# is overlaid (full-file replacement) onto the populated task dir AFTER
# populate+repin, so the fixes survive a re-populate. Starts empty.
PATCHES_DIR = Path(__file__).resolve().parent / "patches"

# A task dir is valid iff it carries a task.toml (the harbor-format anchor).
_TASK_ANCHOR = "task.toml"
# Every SWE-rebench task ships a reference solution, so all are oracle-gateable.
_SOLVE_ANCHOR = os.path.join("solution", "solve.sh")

# Provenance of the last populate: the resolved dataset content hash + task count.
# A dotfile, so shard discovery (which scans for directories) never sees it.
_PROVENANCE = ".dataset-version.json"

# ── Resource routing (written by --stage patch) ──────────────────────────────────────
# harbor applies a CFS cpu quota + a hard memory cap but NEVER a cpuset, so on a
# big host ``nproc`` inside a ``cpus = 1`` container reports the HOST's core
# count (192 on this fleet), not the task's budget. Any pool sized from
# ``os.cpu_count()`` — joblib/loky, pytest-xdist ``-n auto``, dask/ray,
# OpenMP/BLAS threads — then fans out ~192 ways inside an 8 GB cap and is
# SIGKILL'd. The harbor plug-in documents the surgical remedy: a per-task
# ``[environment.env] XRLENV_CPU_PINNING = "1"`` marker that sizes the affinity
# mask to ``ceil(cpus)`` so ``nproc`` equals the task budget again, while the
# quota and memory cap still apply.
#
# Every id below was MEASURED: it failed the 2026-09-01 sweep with a
# ``test.sh: line N: <pid> Killed`` SIGKILL, reproduced at concurrency 4 (so it
# is not a contention artefact), and flipped to reward 1 when re-run with
# cpu pinning. Grow this set only the same way. See STATUS.md.
CPU_PINNING_TASKS: frozenset[str] = frozenset({
    "ImperialCollegeLondon__virtual_ecosystem-1232",
    "SciTools__iris-6754",
    "calliope-project__calliope-854",
    "copier-org__copier-2646",
    "joshuadavidthomas__django-bird-239",
    "modelcontextprotocol__python-sdk-1864",
    "networkx__networkx-8369",
    "owkin__PyDESeq2-356",
    "pybamm-team__PyBaMM-4871",
    "sktime__skpro-574",
    "sktime__sktime-8723",
    "sktime__sktime-8921",
    "sktime__sktime-8937",
    "vyperlang__vyper-4462",
    "vyperlang__vyper-4677",
    "vyperlang__vyper-4801",
})

# Tasks that still exceed their memory cap AFTER pinning, with the value that
# was measured to make the oracle pass.
#
# FAIRNESS RULE (enforced by :func:`_assert_memory_override_is_fair`, not just
# documented): a memory override is permitted ONLY where upstream declared no
# memory for the task. SWE-rebench states resource intent when it has one —
# 10 of the 860 carry an explicit ``harbor_cpus``/``harbor_memory`` in
# ``tests/config.json`` and their task.toml faithfully reflects it (2 cpu /
# 16G). For the other 850 both fields are ``null`` and the ``1 cpu / 8G`` in
# task.toml is purely the dataset converter's blanket default. Raising a
# converter default is not competing on a different resource envelope than the
# benchmark intended; overriding an upstream-declared value WOULD be, so the
# guard refuses it.
MEMORY_OVERRIDES: dict[str, str] = {
    "ImperialCollegeLondon__virtual_ecosystem-1232": "16G",
    "calliope-project__calliope-854": "16G",
    "pybamm-team__PyBaMM-4871": "16G",
    # Still SIGKILL'd at 16G with pinning; measured to pass at 32G.
    "owkin__PyDESeq2-356": "32G",
}

# The marker the harbor plug-in reads (xrlenv_plugins/harbor/environment.py).
_CPU_PINNING_MARKER = "XRLENV_CPU_PINNING"

# ── Hermeticity routing (also written by --stage patch) ──────────────────────
# Distinct from the resource routing above: these do not change the task's
# compute envelope, they stop the verifier from reaching the network mid-grade.
#
# ``CQCL__guppylang-1259``: its test.sh runs ``uv run pytest``, and ``uv run``
# re-resolves the workspace before every invocation. PEP-517 *build* requirements
# are not covered by the lockfile, so the resolve pulls whatever hatchling is
# current on PyPI. Since 2026-08 that is 1.32.0, which rejects the task's
# ``readme = "../README.md"`` ("Readme path must be within the project
# directory") — the package never builds and every F2P plus 33 P2P report
# NOT_FOUND. The task was authored 2025-09 against a hatchling that accepted it.
#
# ``UV_NO_SYNC=1`` tells ``uv run`` to use the environment the image already
# ships instead of re-resolving. Measured: reward 1, resolved True, F2P PASSED,
# 33/33 P2P, and zero build/download lines in the verifier log — so this fixes
# the grade AND removes a live PyPI dependency from the verify phase.
HERMETICITY_ENV: dict[str, dict[str, str]] = {
    "CQCL__guppylang-1259": {"UV_NO_SYNC": "1"},
}


# ── task.toml normalization (pure) ────────────────────────────────────────────


def normalize_task_toml_text(text: str) -> tuple[str, bool]:
    """Strip the harbor-deprecated ``memory`` / ``storage`` string keys from a
    task.toml's ``[environment]`` (and ``[verifier.environment]``) section when
    the canonical ``memory_mb`` / ``storage_mb`` are also present. Returns
    ``(new_text, changed)``.

    harbor's ``EnvironmentConfig`` carries BOTH the deprecated string field and
    the canonical ``_mb`` integer; if a task sets them inconsistently the model
    rejects the conflict and the task won't load. The least-lossy fix is to drop
    the deprecated duplicate and keep the canonical ``_mb`` field. Surgical
    (line-level) so the rest of the file is byte-preserved. Pure (no I/O) so the
    logic is unit-testable in isolation.

    A **no-op for today's SWE-rebench corpus** — it sets only the deprecated
    string form (``memory = '8G'``), which harbor coerces cleanly to
    ``memory_mb = 8192``. Kept defensively so an upstream schema change that
    starts emitting both forms doesn't silently break the load.
    """
    parsed = tomllib.loads(text)

    def _conflicts(section: dict[str, object]) -> set[str]:
        return {
            legacy
            for legacy, canonical in (
                ("memory", "memory_mb"),
                ("storage", "storage_mb"),
            )
            if legacy in section and canonical in section
        }

    env = parsed.get("environment", {})
    ver_env = (parsed.get("verifier", {}) or {}).get("environment", {})
    strip_env = _conflicts(env if isinstance(env, dict) else {})
    strip_ver = _conflicts(ver_env if isinstance(ver_env, dict) else {})
    if not strip_env and not strip_ver:
        return text, False

    out: list[str] = []
    section: str | None = None
    for line in text.splitlines(keepends=True):
        header = re.match(r"\s*\[([^\]]+)\]", line)
        if header:
            section = header.group(1).strip()
        strip = (
            strip_env if section == "environment"
            else strip_ver if section == "verifier.environment"
            else set()
        )
        if strip:
            key = re.match(r"\s*([A-Za-z_]+)\s*=", line)
            if key and key.group(1) in strip:
                continue  # drop the deprecated duplicate
        out.append(line)
    return "".join(out), True


def set_environment_docker_image(text: str, image_ref: str) -> tuple[str, bool]:
    """Set ``[environment] docker_image = "<image_ref>"`` in a task.toml.

    Returns ``(new_text, changed)``; ``changed`` is False when the key is
    already present with exactly this value (so ``--stage repin`` is
    idempotent).

    Pure (no I/O) — the unit tests drive this directly.
    """
    return _set_environment_key(text, "docker_image", f'"{image_ref}"', image_ref)


def set_environment_memory(text: str, memory: str) -> tuple[str, bool]:
    """Set ``[environment] memory = '<memory>'`` (e.g. ``'16G'``).

    Keeps upstream's single-quoted string form rather than introducing the
    canonical ``memory_mb``, so the edit is a one-line, in-place value change
    that ``normalize_task_toml_text`` has nothing to reconcile.
    """
    return _set_environment_key(text, "memory", f"'{memory}'", memory)


def _set_environment_key(
    text: str, key: str, literal: str, parsed_value: object,
) -> tuple[str, bool]:
    """Set ``[environment] <key> = <literal>`` in a task.toml.

    ``parsed_value`` is what ``tomllib`` should report once written — used only
    for the idempotency short-circuit. Rewrites an existing key in place, else
    appends it to the end of the existing ``[environment]`` section, else
    appends a fresh ``[environment]`` section. Line-level, so everything else in
    the file is byte-preserved.
    """
    parsed = tomllib.loads(text)
    env = parsed.get("environment")
    if isinstance(env, dict) and env.get(key) == parsed_value:
        return text, False

    line = f"{key} = {literal}\n"
    lines = text.splitlines(keepends=True)

    # Locate the [environment] section: its header index and the index one past
    # its last line (the next section header, or EOF).
    start = end = None
    for i, raw in enumerate(lines):
        header = re.match(r"\s*\[([^\]]+)\]", raw)
        if header is None:
            continue
        name = header.group(1).strip()
        if name == "environment":
            start = i
        elif start is not None and end is None:
            # Any later header ends the section — including the [environment.env]
            # SUB-table, since a bare key placed after it would belong to the
            # sub-table, not to [environment].
            end = i
    if start is None:
        # No [environment] section at all — append one.
        prefix = "" if not text or text.endswith("\n") else "\n"
        return text + f"{prefix}\n[environment]\n{line}", True
    if end is None:
        end = len(lines)

    for i in range(start + 1, end):
        if re.match(rf"\s*{re.escape(key)}\s*=", lines[i]):
            lines[i] = line
            return "".join(lines), True

    # Insert after the section's last non-blank line so trailing blank lines
    # (and the following section header) stay where they are.
    insert_at = end
    while insert_at > start + 1 and not lines[insert_at - 1].strip():
        insert_at -= 1
    lines.insert(insert_at, line)
    return "".join(lines), True


def _dockerfile_base_ref(text: str) -> str | None:
    """The image ref of the Dockerfile's (last) ``FROM``, or None."""
    matches = re.findall(r"^\s*FROM\s+(\S+)", text, re.MULTILINE)
    return matches[-1] if matches else None


def upstream_image_ref(task_dir: Path) -> str:
    """The authoritative prebuilt image ref for a task.

    Read from the task's own ``tests/config.json`` (upstream's declared
    ``docker_image``) and cross-checked against ``environment/Dockerfile``'s
    ``FROM``. Fails loud on a missing field or a mismatch between the two — a
    silently-wrong pin would run the oracle against the wrong repo snapshot.
    """
    cfg_path = task_dir / "tests" / "config.json"
    if not cfg_path.is_file():
        raise SystemExit(
            f"ERROR: {task_dir.name}: no {cfg_path} — SWE-rebench tasks are "
            f"expected to ship tests/config.json (re-run --stage populate).",
        )
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise SystemExit(
            f"ERROR: {task_dir.name}: unreadable config.json: {exc}",
        ) from exc
    ref = cfg.get("docker_image") or cfg.get("image_name")
    if not ref:
        raise SystemExit(
            f"ERROR: {task_dir.name}: tests/config.json declares no "
            f"'docker_image' — cannot pin an image for this task.",
        )

    dockerfile = task_dir / "environment" / "Dockerfile"
    if dockerfile.is_file():
        base = _dockerfile_base_ref(dockerfile.read_text(encoding="utf-8"))
        if base is not None and base != ref:
            raise SystemExit(
                f"ERROR: {task_dir.name}: image ref disagreement — "
                f"tests/config.json says {ref!r} but environment/Dockerfile "
                f"FROM says {base!r}. Refusing to pin an ambiguous image.",
            )
    return str(ref)


# ── Stage 1: populate ─────────────────────────────────────────────────────────


def populate_hub(shard_dir: Path) -> tuple[int, int]:
    """Download every task of the Harbor Hub package dataset into the shard.

    Uses harbor's own ``PackageDatasetClient`` (dataset -> task list) plus
    ``TaskClient`` (per-task archive download + extract) in **export** mode, so
    tasks land at ``<shard_dir>/<task_id>/`` exactly as
    ``harbor download <dataset> --export`` would place them. Idempotent:
    ``overwrite=False`` makes an already-present task a no-op.

    Returns ``(downloaded, normalized)``.
    """
    import asyncio

    from harbor.registry.client.package import (  # type: ignore[import-untyped]
        PackageDatasetClient,
    )
    from harbor.tasks.client import TaskClient  # type: ignore[import-untyped]

    async def _run() -> tuple[int, int]:
        print(f">> resolving Harbor Hub dataset {DATASET}@{DATASET_REF}", file=sys.stderr)
        client = PackageDatasetClient()
        metadata = await client.get_dataset_metadata(f"{DATASET}@{DATASET_REF}")
        task_ids = list(metadata.task_ids)
        if not task_ids:
            raise SystemExit(
                f"ERROR: {DATASET}@{DATASET_REF} resolved to 0 tasks — refusing to "
                f"produce an empty shard. Check the dataset name / ref.",
            )
        print(
            f">> {len(task_ids)} task(s) at {metadata.version}; downloading into "
            f"{shard_dir}",
            file=sys.stderr,
        )
        shard_dir.mkdir(parents=True, exist_ok=True)

        done = 0

        def _on_complete(_task_id: object, _result: object) -> None:
            nonlocal done
            done += 1
            if done % 50 == 0 or done == len(task_ids):
                print(f"   [{done}/{len(task_ids)}]", file=sys.stderr)

        result = await TaskClient().download_tasks(
            task_ids=task_ids,
            overwrite=False,
            output_dir=shard_dir,
            export=True,
            on_task_download_complete=_on_complete,
        )
        downloaded = sum(1 for r in result.results if not r.cached)

        # Fail loud on a half-populated shard rather than letting a later stage
        # define a silently-smaller green set.
        landed = _count_tasks(shard_dir)
        if landed < len(task_ids):
            raise SystemExit(
                f"ERROR: expected {len(task_ids)} task(s) with {_TASK_ANCHOR} under "
                f"{shard_dir}, found {landed} — the populate is incomplete. "
                f"Re-run --stage populate.",
            )

        normalized = 0
        for toml_path in sorted(shard_dir.glob(f"*/{_TASK_ANCHOR}")):
            if _normalize_task_toml(toml_path):
                normalized += 1

        (shard_dir / _PROVENANCE).write_text(
            json.dumps(
                {
                    "dataset": DATASET,
                    "ref": DATASET_REF,
                    "resolved_version": str(metadata.version),
                    "task_count": len(task_ids),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return downloaded, normalized

    return asyncio.run(_run())


def _normalize_task_toml(path: Path) -> bool:
    new_text, changed = normalize_task_toml_text(path.read_text(encoding="utf-8"))
    if changed:
        path.write_text(new_text, encoding="utf-8")
    return changed


# ── Stage 2: repin (write the authoritative prebuilt image ref) ───────────────


def repin_all(shard_dir: Path) -> int:
    """Write ``[environment] docker_image`` into every populated task.

    Idempotent: a task already pinned to the same ref is left byte-identical.
    Returns the number of tasks whose task.toml changed.
    """
    pinned = 0
    for toml_path in sorted(shard_dir.glob(f"*/{_TASK_ANCHOR}")):
        task_dir = toml_path.parent
        new_text, changed = set_environment_docker_image(
            toml_path.read_text(encoding="utf-8"), upstream_image_ref(task_dir),
        )
        if changed:
            toml_path.write_text(new_text, encoding="utf-8")
            pinned += 1
    return pinned


# ── patch, part 2: resource routing (cpu-pinning + fair memory) ──────────


def set_environment_env_marker(text: str, key: str, value: str) -> tuple[str, bool]:
    """Set ``[environment.env] <key> = "<value>"`` in a task.toml.

    ``[environment.env]`` is a sub-table of ``[environment]``; a bare key may
    not follow it inside the parent, so the sub-table is appended at EOF when
    absent (always valid TOML) and edited in place when present. Idempotent.
    """
    parsed = tomllib.loads(text)
    env = (parsed.get("environment") or {}).get("env") or {}
    if env.get(key) == value:
        return text, False

    line = f'{key} = "{value}"\n'
    lines = text.splitlines(keepends=True)
    start = end = None
    for i, raw in enumerate(lines):
        header = re.match(r"\s*\[([^\]]+)\]", raw)
        if header is None:
            continue
        if header.group(1).strip() == "environment.env":
            start = i
        elif start is not None and end is None:
            end = i
    if start is None:
        prefix = "" if not text or text.endswith("\n") else "\n"
        return text + f"{prefix}\n[environment.env]\n{line}", True
    if end is None:
        end = len(lines)
    for i in range(start + 1, end):
        if re.match(rf"\s*{re.escape(key)}\s*=", lines[i]):
            lines[i] = line
            return "".join(lines), True
    insert_at = end
    while insert_at > start + 1 and not lines[insert_at - 1].strip():
        insert_at -= 1
    lines.insert(insert_at, line)
    return "".join(lines), True


def _assert_memory_override_is_fair(task_dir: Path) -> None:
    """Refuse to override memory for a task upstream sized itself.

    The whole justification for :data:`MEMORY_OVERRIDES` is that the value being
    raised is a converter default, not an upstream decision. If upstream DID
    declare ``harbor_memory`` for this task, raising it would run the oracle in
    a bigger envelope than the benchmark intends — so fail loud instead.
    """
    cfg_path = task_dir / "tests" / "config.json"
    if not cfg_path.is_file():
        raise SystemExit(
            f"ERROR: {task_dir.name}: no tests/config.json — cannot verify that a "
            f"memory override is fair. Refusing.",
        )
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    declared = cfg.get("harbor_memory")
    if declared:
        raise SystemExit(
            f"ERROR: {task_dir.name}: upstream DECLARES harbor_memory={declared!r}; "
            f"overriding it would give the oracle a different resource envelope "
            f"than the benchmark intends. Remove it from MEMORY_OVERRIDES "
            f"(exclude the task instead if it cannot pass as declared).",
        )


def apply_resource_routing(shard_dir: Path) -> tuple[int, int, list[str]]:
    """Write the cpu-pinning markers + fair memory overrides.

    Returns ``(pinned, memory_bumped, missing)``. Idempotent. ``missing`` names
    marked tasks absent from the shard — a corpus-drift signal for the caller.
    """
    pinned = bumped = 0
    missing: list[str] = []
    for task_id in sorted(CPU_PINNING_TASKS | MEMORY_OVERRIDES.keys()):
        task_dir = shard_dir / task_id
        toml_path = task_dir / _TASK_ANCHOR
        if not toml_path.is_file():
            missing.append(task_id)
            continue
        text = toml_path.read_text(encoding="utf-8")
        changed = False
        if task_id in CPU_PINNING_TASKS:
            text, did = set_environment_env_marker(text, _CPU_PINNING_MARKER, "1")
            changed |= did
            if did:
                pinned += 1
        if task_id in MEMORY_OVERRIDES:
            _assert_memory_override_is_fair(task_dir)
            text, did = set_environment_memory(text, MEMORY_OVERRIDES[task_id])
            changed |= did
            if did:
                bumped += 1
        if changed:
            toml_path.write_text(text, encoding="utf-8")
    return pinned, bumped, missing


def apply_hermeticity_env(shard_dir: Path) -> tuple[int, list[str]]:
    """Write the :data:`HERMETICITY_ENV` markers. Returns ``(written, missing)``.

    Kept separate from :func:`apply_resource_routing` because the fairness
    question is different: these vars change how the verifier resolves its own
    dependencies, never how much CPU or memory it gets, so no envelope guard
    applies. Idempotent.
    """
    written = 0
    missing: list[str] = []
    for task_id, env in sorted(HERMETICITY_ENV.items()):
        toml_path = shard_dir / task_id / _TASK_ANCHOR
        if not toml_path.is_file():
            missing.append(task_id)
            continue
        text = toml_path.read_text(encoding="utf-8")
        changed = False
        for key, value in sorted(env.items()):
            text, did = set_environment_env_marker(text, key, value)
            changed |= did
        if changed:
            toml_path.write_text(text, encoding="utf-8")
            written += 1
    return written, missing


# ── Stage 3: patch (curated full-file overlays) ───────────────────────────────


def _apply_patch(task_dir: Path, patch_dir: Path) -> list[str]:
    """Overlay every file under ``patch_dir`` onto ``task_dir`` (full-file
    replacement), preserving relative layout + exec bits. Returns the relative
    paths overridden. A patch file with no counterpart is still copied (lets a
    patch ADD a missing file)."""
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
    """Apply every ``patches/<id>/`` overlay to its present task in the shard.
    Idempotent; a task absent from the shard is skipped with a SKIP note. Returns
    the number of tasks patched (0 while ``patches/`` is empty)."""
    if not PATCHES_DIR.is_dir():
        return 0
    patched = 0
    for patch_dir in sorted(PATCHES_DIR.iterdir()):
        if not patch_dir.is_dir():
            continue  # skip README.md etc.
        tid = patch_dir.name
        dest = shard_dir / tid
        if not (dest / _TASK_ANCHOR).is_file():
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


# ── Helpers ───────────────────────────────────────────────────────────────────


def _count_tasks(shard_dir: Path) -> int:
    return sum(1 for _ in shard_dir.glob(f"*/{_TASK_ANCHOR}"))


def _count_gateable(shard_dir: Path) -> int:
    return sum(1 for _ in shard_dir.glob(f"*/{_SOLVE_ANCHOR}"))


def is_populated(shard_dir: Path) -> bool:
    return shard_dir.is_dir() and _count_tasks(shard_dir) > 0


# ── CLI ───────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="build_cache",
        description=(
            "Materialize the SWE-rebench task corpus into a shared harbor cache. "
            "Stages (each idempotent): populate -> repin -> patch; `all` "
            "(default) runs all three."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--stage",
        choices=("all", "populate", "repin", "patch"),
        default="all",
        help="all (default): populate (if missing) + repin + patch. populate: "
        "download+normalize only (needs network). repin: write each task's "
        "authoritative docker_image. patch: curated content fixes — patches/ "
        "overlays plus the task.toml resource routing (cpu-pinning markers + "
        "fair memory overrides).",
    )
    p.add_argument(
        "--dest",
        default=os.environ.get("XRLENV_BENCHMARK_CACHE"),
        help="Shared cache ROOT (the shard lands under "
        f"<dest>/{SHARD}/). Defaults to $XRLENV_BENCHMARK_CACHE. Point every xrlenv "
        "consumer's XRLENV_BENCHMARK_CACHE at this path.",
    )
    return p


def _resolve_shard(dest: str | None) -> Path:
    # Hard-reject the retired cache env var/path first (renamed 2026-07-31:
    # XRLENV_HARBOR_CACHE -> XRLENV_BENCHMARK_CACHE, xrlenv_harbor_cache ->
    # xrlenv_benchmark_cache). Reusing the old var/path reads stale/absent data and
    # yields unreliable results, so fail loud before resolving a shard from it. Lazy
    # import to match the plugin style (plugin -> xrlenv core).
    from xrlenv_plugins.benchmarks._benchmark_cache import guard_legacy_cache_env

    guard_legacy_cache_env(dest)
    if not dest:
        raise SystemExit(
            "error: no destination — pass --dest or set XRLENV_BENCHMARK_CACHE.",
        )
    return Path(dest).expanduser() / SHARD


def _require_populated(shard_dir: Path, stage: str) -> None:
    if not is_populated(shard_dir):
        raise SystemExit(
            f"cannot {stage}: {shard_dir} is not populated. Run "
            f"`--stage populate` first (or `--stage all`).",
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    shard_dir = _resolve_shard(args.dest)
    shard_dir.parent.mkdir(parents=True, exist_ok=True)

    downloaded = normalized = 0
    if args.stage in ("all", "populate"):
        if args.stage == "populate" or not is_populated(shard_dir):
            downloaded, normalized = populate_hub(shard_dir)
        else:
            print(
                f">> {shard_dir} already populated "
                f"({_count_tasks(shard_dir)} tasks) — skipping download.",
                file=sys.stderr,
            )
    if args.stage == "populate":
        print(
            f"\npopulated {downloaded} task(s) ({normalized} task.toml normalized) "
            f"-> {shard_dir}",
            file=sys.stderr,
        )
        return 0

    pinned = 0
    if args.stage in ("all", "repin"):
        _require_populated(shard_dir, "repin")
        pinned = repin_all(shard_dir)
        if args.stage == "repin":
            print(
                f"\nOK: pinned docker_image on {pinned} task(s) in {shard_dir}",
                file=sys.stderr,
            )
            return 0

    # patch (also reached by --stage all) — curated content fixes, of two kinds:
    # file overlays from patches/, and the programmatic task.toml resource
    # routing. Same split terminal_bench_2_1 uses (PATCHES + ENV_PATCHES).
    _require_populated(shard_dir, "patch")
    patched = apply_all_patches(shard_dir)
    pinned_n, bumped_n, missing = apply_resource_routing(shard_dir)
    hermetic_n, missing_env = apply_hermeticity_env(shard_dir)
    missing += missing_env
    if missing:
        raise SystemExit(
            f"ERROR: {len(missing)} routed task(s) absent from the shard: "
            f"{missing}. The corpus drifted — re-check CPU_PINNING_TASKS / "
            f"MEMORY_OVERRIDES / HERMETICITY_ENV against STATUS.md.",
        )
    total = _count_tasks(shard_dir)
    gateable = _count_gateable(shard_dir)
    print(
        f"\nOK: {total} task(s) in {shard_dir} "
        f"({gateable} oracle-gateable — ship solution/solve.sh)"
        + (f"; {downloaded} downloaded, {normalized} normalized" if downloaded else "")
        + f"; pinned docker_image on {pinned}"
        + f"; applied {patched} curated overlay(s), cpu-pinning marker on "
        + f"{pinned_n}, memory override on {bumped_n}, hermeticity env on "
        + f"{hermetic_n}.",
        file=sys.stderr,
    )
    print(
        f"Point consumers at:  export XRLENV_BENCHMARK_CACHE={shard_dir.parent}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
