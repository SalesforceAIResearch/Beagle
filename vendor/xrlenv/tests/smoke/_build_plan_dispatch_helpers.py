"""Shared helpers for the build-plan dispatch smokes (P1.7.C.2).

Two execution modes:

- **local** — apply the plan against an in-process ``LocalRuntime``
  + the host's Docker daemon. Asserts via ``docker image inspect``
  + the in-memory ``BuildOutcome``.
- **remote** — POST the plan to a running control plane via the
  admin API (``/api/build/apply`` + ``/api/build/plans/<plan_id>``).
  Asserts via the JSON response. Cannot inspect per-node image
  sizes today (the cluster doesn't expose those over the admin
  API yet); calibration tests are local-only.

Both modes return the same dict shape so test bodies can stay
mode-agnostic.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from xrlenv.control.build_plan import BuildPlan, compute_plan_id
from xrlenv.errors import ManifestInvalid

# ──────────────────────────────────────────────────────────────────────────────
# Docker daemon probes
# ──────────────────────────────────────────────────────────────────────────────


def docker_available() -> bool:
    """``docker info`` succeeds. Skip the whole module on False."""
    if shutil.which("docker") is None:
        return False
    try:
        r = subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return r.returncode == 0


def image_present_locally(ref: str) -> bool:
    """``docker images -q <ref>`` returns a non-empty digest."""
    try:
        r = subprocess.run(
            ["docker", "images", "-q", ref],
            check=False, capture_output=True, text=True, timeout=10,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return r.returncode == 0 and bool(r.stdout.strip())


def local_image_uncompressed_bytes(ref: str) -> int | None:
    """``docker image inspect <ref> -f '{{.Size}}'`` — uncompressed
    on-disk size. Returns ``None`` when the image is absent or the
    inspect call fails."""
    try:
        r = subprocess.run(
            ["docker", "image", "inspect", ref, "-f", "{{.Size}}"],
            check=False, capture_output=True, text=True, timeout=10,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if r.returncode != 0:
        return None
    try:
        return int(r.stdout.strip())
    except ValueError:
        return None


def local_image_created_at(ref: str) -> str | None:
    """``docker image inspect <ref> -f '{{.Created}}'`` — RFC3339
    creation timestamp. Used to detect "no new pull happened" on a
    no-op re-apply. Returns ``None`` if absent."""
    try:
        r = subprocess.run(
            ["docker", "image", "inspect", ref, "-f", "{{.Created}}"],
            check=False, capture_output=True, text=True, timeout=10,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if r.returncode != 0:
        return None
    return r.stdout.strip() or None


# ──────────────────────────────────────────────────────────────────────────────
# Harbor cache walker for fresh-task selection
# ──────────────────────────────────────────────────────────────────────────────


def _harbor_cache_root() -> Path:
    explicit = os.environ.get("XRLENV_BENCHMARK_CACHE")
    if explicit:
        return Path(explicit).expanduser()
    return Path("~/.cache/harbor/tasks").expanduser()


def discover_tb2_tasks() -> list[str]:
    """Every tb2 task id present in the harbor cache."""
    from xrlenv_plugins.images_build.terminal_bench_2.build_plan_gen import (
        _discover_all_tasks,
    )
    return _discover_all_tasks()


def pick_fresh_tb2_tasks(
    n: int, *, exclude: Iterable[str], require_uncached: bool = False,
    namespace: str = "alexgshaw", tag: str = "20251031",
) -> list[str]:
    """Pick ``n`` tb2 task ids that aren't in ``exclude``.

    With ``require_uncached=True`` (used in the fresh-8 test), also
    filter out tasks whose ``namespace/<task>:tag`` image is already
    pulled locally — guaranteeing the next apply does real
    registry work. Skips with a clear message if fewer than ``n``
    candidates remain.
    """
    excl = set(exclude)
    candidates = [t for t in discover_tb2_tasks() if t not in excl]
    if require_uncached:
        candidates = [
            t for t in candidates
            if not image_present_locally(f"{namespace}/{t}:{tag}")
        ]
    if len(candidates) < n:
        pytest.skip(
            f"only {len(candidates)} tb2 task(s) available "
            f"(excluding {len(excl)}, require_uncached={require_uncached}); "
            f"need {n}. Populate the harbor cache or relax the filter.",
        )
    return candidates[:n]


# ──────────────────────────────────────────────────────────────────────────────
# Calibration table — registry-probe hint vs measured uncompressed
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class CalibrationRow:
    image_ref: str
    hint_bytes: int
    actual_bytes: int | None
    """Uncompressed on-disk size, or None if image not present."""

    @property
    def ratio(self) -> float | None:
        if self.actual_bytes is None or self.hint_bytes <= 0:
            return None
        return self.actual_bytes / self.hint_bytes


def collect_calibration_rows(plan: BuildPlan) -> list[CalibrationRow]:
    """One row per entry, with the actual on-disk size pulled from
    ``docker image inspect``. Always called after a successful apply
    in local mode."""
    rows: list[CalibrationRow] = []
    for e in plan.entries:
        rows.append(CalibrationRow(
            image_ref=e.image_ref,
            hint_bytes=e.placement.size_hint_bytes,
            actual_bytes=local_image_uncompressed_bytes(e.image_ref),
        ))
    return rows


def format_calibration_table(rows: list[CalibrationRow]) -> str:
    """Pretty-printed table; written to stdout + the calibrated
    artifact next to the canonical YAML."""
    lines = [
        f"{'image_ref':<55}{'hint MB':>10}{'actual MB':>12}{'ratio':>8}",
        "-" * 85,
    ]
    for r in rows:
        hint_mb = r.hint_bytes / 1024**2
        if r.actual_bytes is None:
            actual_str = "(absent)"
            ratio_str = "  -"
        else:
            actual_str = f"{r.actual_bytes / 1024**2:.1f}"
            ratio_str = f"{r.ratio:.2f}x" if r.ratio is not None else "  -"
        lines.append(
            f"{r.image_ref[:55]:<55}{hint_mb:>10.1f}{actual_str:>12}{ratio_str:>8}",
        )
    return "\n".join(lines)


def write_calibrated_plan(
    plan: BuildPlan, rows: list[CalibrationRow], output_path: Path,
) -> None:
    """Write a side-artifact YAML with ``cluster-reported`` sizes for
    every entry the cluster materialized at least once.

    Doesn't touch the canonical ``build_plan.yaml`` — preserves it
    as the registry-probe artifact while giving operators a calibrated
    one to promote later via the (still-deferred) ``xrlenv build
    calibrate`` flow.
    """
    import yaml

    actual_by_ref = {r.image_ref: r.actual_bytes for r in rows}
    raw = plan.model_dump(mode="json", exclude_none=True)
    for e in raw.get("entries", []):
        actual = actual_by_ref.get(e["image_ref"])
        if actual is None:
            continue
        e["placement"]["size_hint_bytes"] = int(actual)
        e["placement"]["size_hint_source"] = "cluster-reported"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(raw, sort_keys=False))


# ──────────────────────────────────────────────────────────────────────────────
# Local-mode apply (build_local_runtime + coordinator.apply)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class ApplyResult:
    """Mode-agnostic apply outcome for assertions in test bodies."""
    plan_id: str
    status: str
    successes: int
    failures: int
    error_summary: list[str]
    raw: dict[str, Any]


async def _local_apply_async(
    plan: BuildPlan, *,
    state_db: Path, runs_root: Path,
    dry_run: bool, force: bool, eager: bool,
) -> ApplyResult:
    from xrlenv.control.runtime import build_local_runtime

    runs_root.mkdir(parents=True, exist_ok=True)
    runtime = build_local_runtime(
        runs_root=runs_root, state_db_path=state_db,
        skip_stale_node_sweep=True,
    )
    await runtime.start()
    try:
        outcome = await runtime.build_coordinator.apply(
            plan, dry_run=dry_run, force=force, eager=eager,
            applied_by="smoke-local",
        )
        return ApplyResult(
            plan_id=outcome.plan_id,
            status=outcome.status,
            successes=outcome.successes,
            failures=outcome.failures,
            error_summary=list(outcome.error_summary),
            raw=outcome.model_dump(mode="json"),
        )
    finally:
        await runtime.shutdown()


def apply_plan_local(
    plan: BuildPlan, *,
    state_db: Path, runs_root: Path,
    dry_run: bool = False, force: bool = False, eager: bool = True,
) -> ApplyResult:
    return asyncio.run(_local_apply_async(
        plan, state_db=state_db, runs_root=runs_root,
        dry_run=dry_run, force=force, eager=eager,
    ))


# ──────────────────────────────────────────────────────────────────────────────
# Remote-mode apply via the admin HTTP API
# ──────────────────────────────────────────────────────────────────────────────


def _remote_config() -> tuple[str, int, str | None] | None:
    """Resolve the admin endpoint for remote-mode smokes.

    Reuses the existing xrlenv env-var convention so an operator who
    already has ``XRLENV_GRPC_HOST`` exported (SDK convention) gets
    remote mode for free. ``XRLENV_ADMIN_HOST`` and
    ``XRLENV_ADMIN_PORT`` are overrides for the rare case where the
    admin HTTP endpoint lives on a different host or port from the
    gRPC endpoint (defaults: same host, port 8080 for admin vs
    50051 for gRPC).

    Resolution order:

    - host: ``XRLENV_ADMIN_HOST`` -> ``XRLENV_GRPC_HOST`` -> None.
    - port: ``XRLENV_ADMIN_PORT`` -> 8080.
    - token: ``XRLENV_OPERATOR_TOKEN`` -> ``~/.xrlenv/secrets/operator.token``.
    """
    host = os.environ.get("XRLENV_ADMIN_HOST") or os.environ.get("XRLENV_GRPC_HOST")
    if not host:
        return None
    port = int(os.environ.get("XRLENV_ADMIN_PORT", "8080"))
    token = os.environ.get("XRLENV_OPERATOR_TOKEN")
    if not token:
        token_path = Path.home() / ".xrlenv" / "secrets" / "operator.token"
        if token_path.is_file():
            try:
                token = token_path.read_text().strip() or None
            except OSError:
                token = None
    return host, port, token


def apply_plan_remote(
    plan: BuildPlan, *,
    dry_run: bool = False, force: bool = False, eager: bool = True,
    poll_interval_s: float = 3.0, poll_timeout_s: float = 1800.0,
) -> ApplyResult:
    """POST ``/api/build/apply``, then poll ``/api/build/plans/<id>``
    until terminal. Mirrors the ``--connect-host`` CLI path but
    returns a structured result instead of writing to stdout."""
    import httpx

    cfg = _remote_config()
    if cfg is None:
        pytest.skip(
            "remote smoke skipped: set $XRLENV_GRPC_HOST (or "
            "$XRLENV_ADMIN_HOST as an override) and $XRLENV_OPERATOR_TOKEN "
            "to a running ``xrlenv up`` admin endpoint.",
        )
    host, port, token = cfg
    base = f"http://{host}:{port}"
    headers: dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = {
        "plan": plan.model_dump(mode="json", exclude_none=True),
        "dry_run": dry_run, "force": force, "eager": eager,
        "applied_by": "smoke-remote",
    }
    plan_id = compute_plan_id(plan)
    with httpx.Client(base_url=base, headers=headers, timeout=15.0) as client:
        r = client.post("/api/build/apply", json=body)
    if r.status_code in (401, 403):
        pytest.fail(f"remote admin auth failed ({r.status_code}): "
                    "set $XRLENV_OPERATOR_TOKEN to an operator-role token")
    if r.status_code == 503:
        pytest.fail(
            "remote admin 503: build coordinator not wired on that "
            "control plane (run xrlenv up with the latest binary)",
        )
    if r.status_code == 400:
        # Plan validation / dispatch-time rejection. Mirror the local
        # path, which raises ManifestInvalid synchronously, so the
        # seta-env rejection smoke can match the same exception type.
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        raise ManifestInvalid(str(detail))
    if r.status_code not in (200, 202):
        pytest.fail(f"remote /api/build/apply returned {r.status_code}: {r.text}")
    payload = r.json()
    plan_id = str(payload.get("plan_id") or plan_id)

    if dry_run or payload.get("status") == "no_op_already_completed":
        return ApplyResult(
            plan_id=plan_id,
            status=str(payload.get("status") or "dry_run"),
            successes=0, failures=0, error_summary=[],
            raw=payload,
        )

    deadline = time.monotonic() + poll_timeout_s
    # Separate, shorter deadline for "the plan record is at least
    # persisted." If the coordinator's _apply_per_image_ref raises
    # before record_build_plan (e.g. plan_placements raising
    # InsufficientCapacity when zero nodes are connected), the
    # background asyncio task swallows the error, the plan is never
    # persisted, and /api/build/plans/<plan_id> returns 404 forever.
    # Without this guard, the poll loop would hang for the full
    # poll_timeout_s (default 30 min). 30s is plenty for the
    # coordinator to call record_build_plan on a healthy cluster.
    persistence_deadline = time.monotonic() + 30.0
    poll_url = f"/api/build/plans/{plan_id}"
    while True:
        if time.monotonic() > deadline:
            pytest.fail(
                f"remote plan {plan_id} did not reach terminal status "
                f"within {poll_timeout_s:.0f}s",
            )
        time.sleep(poll_interval_s)
        with httpx.Client(
            base_url=base, headers=headers, timeout=15.0,
        ) as client:
            pr = client.get(poll_url)
        if pr.status_code == 404:
            if time.monotonic() > persistence_deadline:
                pytest.fail(
                    f"remote plan {plan_id} never appeared in state.db "
                    "(404 from /api/build/plans/<plan_id> for >30s after "
                    "POST). The coordinator's apply likely raised before "
                    "record_build_plan — common causes:\n"
                    "  - zero nodes connected (cluster just started, "
                    "or all nodes lost). Check `xrlenv nodes`.\n"
                    "  - InsufficientCapacity from the FFD bin-packer "
                    "(no node has room for at least one entry).\n"
                    "Look for `build coordinator apply raised` in the "
                    "admin server logs for the underlying exception.",
                )
            continue
        if pr.status_code != 200:
            continue
        snap = pr.json()
        if snap.get("status") in ("in_flight",):
            continue
        per_status = snap.get("per_status") or {}
        successes = int(per_status.get("done", 0))
        failures = int(per_status.get("failed", 0))
        error_summary = [
            f"{a['node_id']}/{a['image_ref']}: {a.get('error') or 'unknown'}"
            for a in (snap.get("assignments") or [])
            if a.get("status") == "failed"
        ]
        return ApplyResult(
            plan_id=plan_id,
            status=str(snap.get("status")),
            successes=successes,
            failures=failures,
            error_summary=error_summary[:20],
            raw=snap,
        )


def apply_plan(
    plan: BuildPlan, *, mode: str,
    state_db: Path, runs_root: Path,
    dry_run: bool = False, force: bool = False, eager: bool = True,
) -> ApplyResult:
    if mode == "local":
        return apply_plan_local(
            plan, state_db=state_db, runs_root=runs_root,
            dry_run=dry_run, force=force, eager=eager,
        )
    if mode == "remote":
        return apply_plan_remote(
            plan, dry_run=dry_run, force=force, eager=eager,
        )
    raise ValueError(f"unknown mode {mode!r}; expected 'local' or 'remote'")


# ──────────────────────────────────────────────────────────────────────────────
# pytest fixtures
# ──────────────────────────────────────────────────────────────────────────────


def runtime_modes() -> list[str]:
    """Modes the suite runs against. Always includes ``local``;
    includes ``remote`` only when an admin endpoint is configured
    via ``$XRLENV_ADMIN_HOST`` or the standard ``$XRLENV_GRPC_HOST``
    (otherwise the remote tests would dominate skip noise)."""
    modes = ["local"]
    if os.environ.get("XRLENV_ADMIN_HOST") or os.environ.get("XRLENV_GRPC_HOST"):
        modes.append("remote")
    return modes


def smoke_artifact_dir(label: str) -> Path:
    """Per-run output directory under ``<repo>/tmp/`` per the
    project convention."""
    repo_root = Path(__file__).resolve().parents[2]
    ts = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    out = repo_root / "tmp" / f"smoke-build-plan-{label}-{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_summary(out_dir: Path, name: str, payload: dict[str, Any]) -> None:
    (out_dir / name).write_text(json.dumps(payload, indent=2, default=str))
