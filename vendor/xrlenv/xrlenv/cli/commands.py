"""Operator CLI command implementations (Slice 5b, spec 09).

Phase-0 operator workflows are read-mostly: the consumer drives via
``Client``; operators want to inspect what's happening. These commands
all open the on-disk state directly:

- :class:`SqliteStateStore` opened against ``~/.xrlenv/state.db`` (WAL
  mode is safe for concurrent readers — the running control plane
  keeps the same file open and writes through it).
- :class:`PlatformJsonlSink` walking ``~/.xrlenv/runs/<date>/<id>/``.

This means every command works without an admin RPC: zero new wire
format, zero auth surface, no risk of an operator's ``xrlenv events``
loop holding open a gRPC stream that pins control-plane resources.
The trade-off is that read commands only work on the same host as
the control plane — phase-0 is single-host so this is the right
trade. A remote-CLI admin RPC lands in a later slice if multi-host
operations become a need.

Mutating commands (``drain``, ``reload``) deliberately do not ship
in this slice; they need an admin RPC to talk to the running control
plane.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import json
import logging
import os
import re
import subprocess
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import IO, Any, TextIO

import yaml

from xrlenv import paths
from xrlenv.backends.base import ImageRecord, SandboxBackend
from xrlenv.control.state import (
    EventRecord,
    RolloutRecord,
    SandboxRecord,
    SqliteStateStore,
)
from xrlenv.control.trajectory_sink import PlatformJsonlSink
from xrlenv.node.image_pins import DEFAULT_PIN_FILE, load_image_pins

# All three derive from ``$XRLENV_HOME`` (default ``~/.xrlenv``); see
# :mod:`xrlenv.paths`. Evaluated at import — after ``xrlenv/__init__.py`` has
# auto-loaded ``.env`` into ``os.environ`` — so a checkout's ``.env`` can
# relocate the whole tree for a side-by-side dev cluster.
DEFAULT_XRLENV_HOME = paths.xrlenv_home()
DEFAULT_STATE_DB = paths.state_db_path()
DEFAULT_RUNS_ROOT = paths.runs_root()
DEFAULT_NODES_YAML = Path.cwd() / "nodes.yaml"


# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────


def _open_state(state_db: Path, *, read_only: bool = True) -> SqliteStateStore:
    """Open the state store — READ-ONLY BY DEFAULT.

    A read-only open never runs ``PRAGMA journal_mode`` (which, env-unset,
    defaults to WAL and would flip a control plane's ``TRUNCATE`` DB back to WAL,
    recreating the ``-shm`` mmap SIGBUS exposure on a network filesystem), nor
    creates ``-wal``/``-shm`` sidecars, nor stamps metadata. The default is
    ``read_only=True`` on purpose: the safe path is the default, so a *new*
    pure-read CLI command can't silently inherit the mutating open. The handful
    of commands that actually WRITE state (``fairshare set``, ``db prune``,
    ``build cancel``) must pass ``read_only=False`` explicitly.
    """
    if not state_db.exists():
        raise FileNotFoundError(
            f"state.db not found at {state_db}. Pass --state-db <path> if the "
            "control plane uses a non-default location, or start the control "
            "plane first."
        )
    return SqliteStateStore(state_db, read_only=read_only)


def _open_sink(runs_root: Path) -> PlatformJsonlSink:
    return PlatformJsonlSink(runs_root)


def _print_table(rows: Sequence[Sequence[str]], *, out: TextIO) -> None:
    """Render rows as fixed-width columns. Empty input prints nothing."""
    if not rows:
        return
    widths = [
        max(len(str(row[i])) for row in rows)
        for i in range(len(rows[0]))
    ]
    for row in rows:
        out.write(
            "  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))
        )
        out.write("\n")


def _emit_json(payload: Any, *, out: TextIO) -> None:
    out.write(json.dumps(payload, indent=2, default=str))
    out.write("\n")


_DURATION_RE = re.compile(r"^(\d+)([smhd])$")


def parse_duration(spec: str) -> float:
    """Parse ``5m`` / ``2h`` / ``30s`` / ``1d`` into seconds.

    Public so the CLI dispatcher can call it on argparse types.
    """
    m = _DURATION_RE.fullmatch(spec.strip())
    if not m:
        raise ValueError(
            f"invalid duration {spec!r}; expected like 30s, 5m, 2h, 1d"
        )
    value = int(m.group(1))
    unit = m.group(2)
    return value * {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]


# ──────────────────────────────────────────────────────────────────────────────
# nodes
# ──────────────────────────────────────────────────────────────────────────────


def _load_nodes_yaml(path: Path) -> list[dict[str, Any]]:
    """Read the operator's nodes.yaml roster. Returns ``[]`` if missing."""
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        return []
    nodes = raw.get("nodes") or []
    if not isinstance(nodes, list):
        return []
    return [n for n in nodes if isinstance(n, dict)]


def cmd_nodes(
    *,
    state_db: Path = DEFAULT_STATE_DB,
    nodes_yaml: Path = DEFAULT_NODES_YAML,
    output_format: str = "text",
    out: TextIO,
) -> int:
    """List nodes from the live registry mirror, the roster, and the
    active-sandbox counts.

    The union of three sources surfaces every relevant node:

    - ``state.list_nodes()`` — :class:`NodeRegistry`'s persistent shadow,
      so a freshly-attached but idle node shows up immediately (no need
      to wait for the first rollout). Status column tracks
      ``connected`` vs ``lost`` (heartbeat watchdog tripped).
    - ``nodes.yaml`` — the operator-managed roster. Nodes named here
      that haven't connected yet still appear so the operator can see
      what they're expecting.
    - ``state.list_sandboxes()`` — covers the pre-Slice-4 case where a
      node had sandboxes recorded against it before the registry table
      existed (or for an in-process runtime that doesn't write to the
      registry table).
    """
    rostered = _load_nodes_yaml(nodes_yaml)
    rostered_ids: set[str] = {
        str(n["id"]) for n in rostered if isinstance(n.get("id"), str)
    }

    sandboxes_by_node: dict[str, int] = {}
    nodes_by_id: dict[str, Any] = {}
    # READ-ONLY: `nodes` is a pure query — never flip the live DB's journal mode
    # or write to it (see _open_state). Safe to run against a control plane's
    # TRUNCATE-on-Lustre store without re-arming the WAL -shm SIGBUS.
    state = _open_state(state_db, read_only=True)
    try:
        for sb in state.list_sandboxes():
            sandboxes_by_node[sb.node_id] = sandboxes_by_node.get(sb.node_id, 0) + 1
        for n in state.list_nodes():
            nodes_by_id[n.node_id] = n
    finally:
        state.close()

    seen_ids: set[str] = set(sandboxes_by_node) | rostered_ids | set(nodes_by_id)
    now = time.time()
    rows: list[dict[str, Any]] = []
    for nid in sorted(seen_ids):
        roster_entry = next(
            (n for n in rostered if n.get("id") == nid), None
        )
        # Accept both the spec-09 example key (``expected_address``) and the
        # typed-loader key (``address`` per xrlenv/control/nodes_yaml.py).
        addr: Any = None
        if roster_entry is not None:
            addr = (
                roster_entry.get("expected_address")
                or roster_entry.get("address")
            )
        live = nodes_by_id.get(nid)
        if live is not None:
            status = live.status
            last_seen_age = max(0.0, now - live.last_seen_at)
        else:
            status = "absent"
            last_seen_age = None
        rows.append(
            {
                "id": nid,
                "status": status,
                "last_seen_age_s": last_seen_age,
                "rostered": roster_entry is not None,
                "cloud": (roster_entry or {}).get("cloud"),
                "expected_address": addr,
                "active_sandboxes": sandboxes_by_node.get(nid, 0),
                # P6 step-2c (observability) — advertised CPU-isolation
                # capability + last-known pinnable-CPU counts. Absent nodes /
                # pre-P6 rows read false / 0 / 0.
                "isolation_capable": bool(getattr(live, "isolation_capable", False)),
                "pinned_cpus_free": int(getattr(live, "pinned_cpus_free", 0)),
                "pinned_cpus_total": int(getattr(live, "pinned_cpus_total", 0)),
            }
        )

    if output_format == "json":
        _emit_json(rows, out=out)
    else:
        table = [[
            "NODE", "STATUS", "LAST_SEEN", "ROSTERED", "CLOUD",
            "EXPECTED ADDR", "ACTIVE_SBX", "CPU_ISOLATION",
        ]]
        for r in rows:
            age = r["last_seen_age_s"]
            last_seen_label = (
                "-" if age is None else
                ("now" if age < 1 else f"{age:.0f}s ago" if age < 60 else f"{age/60:.0f}m ago")
            )
            # "yes 6/8" (capable, 6 of 8 pinnable CPUs free) / "no" / "no 0/0".
            iso = "yes" if r["isolation_capable"] else "no"
            if r["pinned_cpus_total"]:
                iso += f" {r['pinned_cpus_free']}/{r['pinned_cpus_total']}"
            table.append(
                [
                    str(r["id"]),
                    str(r["status"]),
                    last_seen_label,
                    "yes" if r["rostered"] else "no",
                    str(r["cloud"] or "-"),
                    str(r["expected_address"] or "-"),
                    str(r["active_sandboxes"]),
                    iso,
                ]
            )
        _print_table(table, out=out)
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# rollouts
# ──────────────────────────────────────────────────────────────────────────────


def _filter_rollouts(
    rollouts: Iterable[RolloutRecord],
    *,
    status: str | None,
    template: str | None,
    since_s: float | None,
) -> list[RolloutRecord]:
    out: list[RolloutRecord] = []
    cutoff_ts = (time.time() - since_s) if since_s is not None else None
    for r in rollouts:
        if status and r.status.value != status:
            continue
        if template and r.template != template:
            continue
        if cutoff_ts is not None and r.created_at < cutoff_ts:
            continue
        out.append(r)
    return out


def cmd_rollouts(
    *,
    state_db: Path = DEFAULT_STATE_DB,
    status: str | None = None,
    template: str | None = None,
    since: str | None = None,
    output_format: str = "text",
    out: TextIO,
) -> int:
    """List rollouts with optional --status / --template / --since filters."""
    since_s = parse_duration(since) if since else None
    state = _open_state(state_db)
    try:
        records = _filter_rollouts(
            state.list_rollouts(),
            status=status,
            template=template,
            since_s=since_s,
        )
    finally:
        state.close()
    records.sort(key=lambda r: r.created_at, reverse=True)

    if output_format == "json":
        _emit_json(
            [
                {
                    "rollout_id": r.rollout_id,
                    "template": r.template,
                    "status": r.status.value,
                    "reason": r.reason,
                    "node_id": r.node_id,
                    "task_key": r.task_key,
                    "group_id": r.group_id,
                    "created_at": r.created_at,
                    "step_count": len(r.steps),
                    "final_reward": r.final_reward,
                }
                for r in records
            ],
            out=out,
        )
    else:
        table = [["ROLLOUT_ID", "STATUS", "TEMPLATE", "NODE", "STEPS", "REWARD", "AGE_S"]]
        now = time.time()
        for r in records:
            table.append(
                [
                    r.rollout_id[:16],
                    r.status.value,
                    r.template,
                    r.node_id or "-",
                    str(len(r.steps)),
                    f"{r.final_reward:.3f}",
                    f"{now - r.created_at:.0f}",
                ]
            )
        _print_table(table, out=out)
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# replay
# ──────────────────────────────────────────────────────────────────────────────


def cmd_replay(
    rollout_id: str,
    *,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    output_format: str = "text",
    out: TextIO,
) -> int:
    """Print a sealed trajectory's summary + steps."""
    sink = _open_sink(runs_root)
    try:
        traj = sink.read(rollout_id)
    except FileNotFoundError as exc:
        out.write(f"error: {exc}\n")
        return 1

    if output_format == "json":
        _emit_json(traj.model_dump(), out=out)
    else:
        out.write(f"rollout_id    : {traj.rollout_id}\n")
        out.write(f"template      : {traj.template}\n")
        out.write(f"status        : {traj.status.value}\n")
        if traj.reason:
            out.write(f"reason        : {traj.reason}\n")
        out.write(f"final_reward  : {traj.final_reward}\n")
        out.write(f"steps         : {len(traj.steps)}\n")
        if traj.metadata:
            out.write("metadata      :\n")
            out.write(json.dumps(traj.metadata, indent=2, default=str))
            out.write("\n")
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# events
# ──────────────────────────────────────────────────────────────────────────────


def _filter_events(
    events: Iterable[EventRecord],
    *,
    rollout_id: str | None,
    since_s: float | None,
) -> list[EventRecord]:
    out: list[EventRecord] = []
    cutoff_ts = (time.time() - since_s) if since_s is not None else None
    for e in events:
        if rollout_id and e.rollout_id != rollout_id:
            continue
        if cutoff_ts is not None and e.ts < cutoff_ts:
            continue
        out.append(e)
    return out


def cmd_events(
    *,
    state_db: Path = DEFAULT_STATE_DB,
    since: str | None = None,
    rollout_id: str | None = None,
    output_format: str = "text",
    out: TextIO,
) -> int:
    """Stream the events log with optional --since DURATION + --rollout ID."""
    since_s = parse_duration(since) if since else None
    state = _open_state(state_db)
    try:
        events = list(state.events_since(0))
    finally:
        state.close()
    events = _filter_events(events, rollout_id=rollout_id, since_s=since_s)

    if output_format == "json":
        _emit_json(
            [
                {
                    "seq": e.seq, "ts": e.ts, "rollout_id": e.rollout_id,
                    "sandbox_id": e.sandbox_id, "kind": e.kind,
                    "payload": e.payload,
                }
                for e in events
            ],
            out=out,
        )
    else:
        table = [["SEQ", "TS", "KIND", "ROLLOUT", "SANDBOX", "PAYLOAD"]]
        for e in events:
            table.append(
                [
                    str(e.seq),
                    f"{e.ts:.3f}",
                    e.kind,
                    (e.rollout_id or "-")[:16],
                    (e.sandbox_id or "-")[:16],
                    json.dumps(e.payload, default=str)[:80],
                ]
            )
        _print_table(table, out=out)
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# audit (spec 19) — separate table, separate retention, separate command.
# ``xrlenv events`` shows rollout-lifecycle events; auth events
# (``auth.token_used`` / ``auth.denied``) live in the ``audit`` table per
# spec 20's retention matrix and are surfaced here. Operators verifying
# node-connect after a fresh deploy use this to confirm the bidi stream
# attached even when no rollouts have run yet.
# ──────────────────────────────────────────────────────────────────────────────


def cmd_audit(
    *,
    state_db: Path = DEFAULT_STATE_DB,
    since: str | None = None,
    kind: str | None = None,
    role: str | None = None,
    output_format: str = "text",
    out: TextIO,
) -> int:
    """Stream the spec-19 audit log with optional --since / --kind / --role filters."""
    since_s = parse_duration(since) if since else None
    cutoff_ts = (time.time() - since_s) if since_s is not None else None

    state = _open_state(state_db)
    try:
        rows = list(state.audit_since(0))
    finally:
        state.close()
    filtered = []
    for r in rows:
        if cutoff_ts is not None and r.ts < cutoff_ts:
            continue
        if kind and r.kind != kind:
            continue
        if role and r.role != role:
            continue
        filtered.append(r)

    if output_format == "json":
        _emit_json(
            [
                {
                    "seq": r.seq, "ts": r.ts, "kind": r.kind,
                    "role": r.role, "result": r.result,
                    "method": r.method, "source": r.source,
                    "digest_hint": r.digest_hint,
                    "payload": r.payload,
                }
                for r in filtered
            ],
            out=out,
        )
    else:
        table = [["SEQ", "TS", "KIND", "ROLE", "RESULT", "METHOD", "SOURCE"]]
        for r in filtered:
            table.append(
                [
                    str(r.seq),
                    f"{r.ts:.3f}",
                    r.kind,
                    r.role or "-",
                    r.result,
                    (r.method or "-")[:36],
                    (r.source or "-")[:24],
                ]
            )
        _print_table(table, out=out)
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# tail / attach (file-following commands)
# ──────────────────────────────────────────────────────────────────────────────


def _resolve_run_dir(runs_root: Path, rollout_id: str) -> Path:
    """Walk ``runs_root/<date>/`` to find the rollout's directory."""
    if not runs_root.exists():
        raise FileNotFoundError(f"runs root {runs_root} does not exist")
    for date_dir in sorted(runs_root.iterdir()):
        candidate = date_dir / rollout_id
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"no run dir for rollout {rollout_id} under {runs_root}"
    )


def _follow_file(
    path: Path,
    *,
    out: TextIO,
    stop_after_s: float | None,
    poll_interval_s: float = 0.5,
    max_lines: int | None = None,
) -> int:
    """Print existing contents then tail-follow.

    ``stop_after_s`` short-circuits the loop after that many wall-clock
    seconds (used in tests so the CLI doesn't block forever).
    ``max_lines`` caps the total number of lines emitted.
    """
    started = time.monotonic()
    emitted = 0
    fp: IO[str] | None = None
    try:
        if path.exists():
            fp = path.open("r", encoding="utf-8")
            for line in fp:
                out.write(line)
                emitted += 1
                if max_lines is not None and emitted >= max_lines:
                    return 0
        while True:
            if stop_after_s is not None and time.monotonic() - started >= stop_after_s:
                return 0
            if fp is None and path.exists():
                fp = path.open("r", encoding="utf-8")
            if fp is not None:
                line = fp.readline()
                if line:
                    out.write(line)
                    out.flush()
                    emitted += 1
                    if max_lines is not None and emitted >= max_lines:
                        return 0
                    continue
            time.sleep(poll_interval_s)
    finally:
        if fp is not None:
            fp.close()


def cmd_tail(
    rollout_id: str,
    *,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    stop_after_s: float | None = None,
    out: TextIO,
) -> int:
    """Follow trajectory.jsonl for an in-flight rollout."""
    try:
        run_dir = _resolve_run_dir(runs_root, rollout_id)
    except FileNotFoundError as exc:
        out.write(f"error: {exc}\n")
        return 1
    return _follow_file(run_dir / "trajectory.jsonl", out=out, stop_after_s=stop_after_s)


def cmd_attach(
    rollout_id: str,
    *,
    state_db: Path = DEFAULT_STATE_DB,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    stop_after_s: float | None = None,
    out: TextIO,
) -> int:
    """Print the rollout's snapshot + recent events, then tail coordinator.log.

    Phase-0 ``attach`` is read-only inspection (no interactive shell). It's
    the operator's "what's happening with this rollout right now" tool —
    snapshot + the per-rollout event log streamed live.
    """
    state = _open_state(state_db)
    try:
        try:
            record = state.get_rollout(rollout_id)
        except KeyError as exc:
            out.write(f"error: {exc}\n")
            return 1

        sandbox_record: SandboxRecord | None = None
        if record.sandbox_id is not None:
            try:
                sandbox_record = state.get_sandbox(record.sandbox_id)
            except KeyError:
                sandbox_record = None
        events = [
            e for e in state.events_since(0) if e.rollout_id == rollout_id
        ]
    finally:
        state.close()

    out.write(f"=== rollout {record.rollout_id} ===\n")
    out.write(f"template      : {record.template}\n")
    out.write(f"status        : {record.status.value}\n")
    if record.reason:
        out.write(f"reason        : {record.reason}\n")
    out.write(f"node_id       : {record.node_id or '-'}\n")
    out.write(f"sandbox_id    : {record.sandbox_id or '-'}\n")
    if sandbox_record is not None:
        out.write(
            f"sandbox.backend: {sandbox_record.backend} "
            f"(ref={sandbox_record.backend_ref})\n"
        )
    out.write(f"steps         : {len(record.steps)}\n")
    out.write(f"final_reward  : {record.final_reward}\n")
    out.write(f"task_key      : {record.task_key or '-'}\n")
    out.write(f"group_id      : {record.group_id or '-'}\n")
    out.write("\n")

    out.write("--- recent events ---\n")
    for e in events[-10:]:
        out.write(
            f"  seq={e.seq} ts={e.ts:.3f} kind={e.kind} "
            f"payload={json.dumps(e.payload, default=str)[:80]}\n"
        )
    if not events:
        out.write("  (no events recorded)\n")
    out.write("\n")

    try:
        run_dir = _resolve_run_dir(runs_root, rollout_id)
    except FileNotFoundError:
        out.write("(no run dir on disk; nothing to follow)\n")
        return 0
    coordinator_log = run_dir / "coordinator.log"
    out.write(f"--- following {coordinator_log} ---\n")
    return _follow_file(coordinator_log, out=out, stop_after_s=stop_after_s)


# ──────────────────────────────────────────────────────────────────────────────
# images / warmup (spec 15)
# ──────────────────────────────────────────────────────────────────────────────


def _build_local_docker_backend() -> SandboxBackend:
    """Construct a phase-0 single-host DockerBackend for the CLI to query.

    The image-management methods on ``SandboxBackend`` only need the
    Docker client; we don't go through ``build_local_runtime`` because
    the operator might run ``xrlenv images`` against a control plane
    that is itself running in another process — in phase-0 single-host
    deployments both share ``$DOCKER_HOST`` so a fresh DockerBackend
    sees the same image cache the running NodeAgent does.
    """
    import xrlenv
    from xrlenv.backends.docker import (
        DockerBackend,
        DockerBackendConfig,
        _default_stub_transport,
    )

    xrlenv_pkg_path = Path(xrlenv.__file__).resolve().parent
    return DockerBackend(
        DockerBackendConfig(
            runs_root=DEFAULT_RUNS_ROOT,
            xrlenv_pkg_path=xrlenv_pkg_path,
            stub_transport=_default_stub_transport(),
        ),
    )


def _format_bytes(n: int) -> str:
    if n >= 1024**3:
        return f"{n / 1024**3:.1f}G"
    if n >= 1024**2:
        return f"{n / 1024**2:.1f}M"
    if n >= 1024:
        return f"{n / 1024:.1f}K"
    return f"{n}B"


def _images_summary(images: list[ImageRecord], pin_set: set[str]) -> dict[str, Any]:
    """Aggregate per-image records into the spec-15-shaped tier summary."""
    total_size = sum(img.size_bytes for img in images)
    pinned_count = sum(1 for img in images if img.name in pin_set)
    return {
        "total_count": len(images),
        "total_size_bytes": total_size,
        "pinned_count": pinned_count,
    }


def cmd_images(
    *,
    pin_file: Path | None = None,
    output_format: str = "text",
    out: TextIO,
    backend: SandboxBackend | None = None,
) -> int:
    """List images cached on this host with their pin/size state.

    Phase-0 surface (spec 15 §"Operator surface"): single-host listing.
    Multi-node remote view lands when the control plane fans cache-status
    reports up the heartbeat (phase 1).
    """
    pin_set = load_image_pins(pin_file or DEFAULT_PIN_FILE)
    drv = backend or _build_local_docker_backend()
    images = asyncio.run(drv.list_images())
    images.sort(key=lambda i: (-i.size_bytes, i.name))
    free = asyncio.run(drv.free_disk_bytes())
    summary = _images_summary(images, pin_set)

    if output_format == "json":
        _emit_json(
            {
                "summary": {
                    **summary,
                    "free_disk_bytes": free,
                    "pin_count": len(pin_set),
                },
                "images": [
                    {
                        "name": img.name,
                        "size_bytes": img.size_bytes,
                        "pinned": img.name in pin_set,
                        "digest": img.digest,
                    }
                    for img in images
                ],
            },
            out=out,
        )
        return 0

    out.write(
        f"summary: {summary['total_count']} images, "
        f"total={_format_bytes(summary['total_size_bytes'])}, "
        f"pinned={summary['pinned_count']}/{len(pin_set)}, "
        f"free_disk={_format_bytes(free)}\n"
    )
    table = [["IMAGE", "SIZE", "PINNED"]]
    for img in images:
        table.append(
            [img.name, _format_bytes(img.size_bytes), "yes" if img.name in pin_set else "no"]
        )
    _print_table(table, out=out)
    return 0


def cmd_warmup(
    images: Sequence[str],
    *,
    deadline_s: float = 600.0,
    out: TextIO,
    backend: SandboxBackend | None = None,
) -> int:
    """Pre-pull a list of images so they're warm before the next rollout.

    Phase-0 surface: explicit operator-driven warming via ``docker pull``
    against the local daemon. Phase-1 ``client.warmup(...)`` SDK + control-
    plane fan-out + per-instance cache-key derivation lands later.

    Concurrency is bounded by the backend's pull semaphore (the
    ImageCacheManager owns that ceiling); this CLI just dispatches one
    pull per image and reports per-image success/failure.
    """
    if not images:
        out.write("error: at least one image is required\n")
        return 2

    drv = backend or _build_local_docker_backend()

    async def _pull_all() -> list[tuple[str, str | None]]:
        results: list[tuple[str, str | None]] = []
        for image in images:
            t0 = time.monotonic()
            try:
                await drv.pull_image(image, timeout_s=deadline_s)
                elapsed = time.monotonic() - t0
                out.write(f"pulled {image} ({elapsed:.1f}s)\n")
                results.append((image, None))
            except Exception as exc:
                out.write(f"failed  {image}: {exc}\n")
                results.append((image, str(exc)))
        return results

    results = asyncio.run(_pull_all())
    failures = [r for r in results if r[1] is not None]
    return 0 if not failures else 1


# ──────────────────────────────────────────────────────────────────────────────
# stub-runtime layer (D12 stage 1 helper — three-stage build pipeline)
# ──────────────────────────────────────────────────────────────────────────────


def stub_runtime_dockerfile_path() -> Path:
    """Path to the canonical stub-runtime Dockerfile snippet.

    The snippet is shipped as ``xrlenv/sandbox_stub/Dockerfile.stub-runtime``
    so version pins for pydantic / aiohttp / pyyaml live in one place
    and propagate to every benchmark plug-in's next build automatically.
    """
    import xrlenv

    return (
        Path(xrlenv.__file__).resolve().parent
        / "sandbox_stub" / "Dockerfile.stub-runtime"
    )


def _inspect_image_user(image: str) -> str:
    """Return the upstream image's ``Config.User`` (or ``"root"``
    when empty / inspect fails).

    Used by ``cmd_stub_runtime_layer`` to thread the upstream USER
    through to the stub-runtime layer Dockerfile so installing as
    root for apt+pip doesn't permanently flip the runtime user
    (audit M1, 2026-04-29). Empty Config.User means "Docker's
    default" which is root, so we surface it as root explicitly.
    """
    try:
        result = subprocess.run(
            ["docker", "inspect", "--format={{.Config.User}}", image],
            capture_output=True, text=True, check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "root"
    user = result.stdout.strip()
    return user if user else "root"


def cmd_stub_runtime_layer(
    *,
    base: str,
    out_tag: str,
    runner: Callable[[Sequence[str]], int] | None = None,
    upstream_user_resolver: Callable[[str], str] | None = None,
    out: TextIO,
) -> int:
    """``xrlenv stub-runtime layer --base <tag> --out <tag>``.

    Build the canonical stub-runtime layer on top of ``base`` and tag
    the result as ``out_tag``. Plug-ins' ``build-task-images.sh``
    scripts call this so the platform's stub Python deps live in
    one pinned place — see ``xrlenv/sandbox_stub/Dockerfile.stub-runtime``.

    Three-stage benchmark image pipeline:

    - **Stage 1** (per plug-in) — upstream task env from
      ``<task>/environment/Dockerfile``, tagged
      ``<bench>-base/<task>:<ver>``.
    - **Stage 2** (this command, platform-owned) — stub-runtime
      layer (apt-install python+pip if missing, then pip install
      pydantic+aiohttp+pyyaml). Preserves the upstream image's
      ``Config.User`` so installing as root for apt+pip doesn't
      permanently flip the runtime user.
    - **Stage 3** (per plug-in, optional) — adapter-runtime deps if
      the EnvAdapter imports its own libraries (e.g. swebench's
      harness package).

    ``runner`` and ``upstream_user_resolver`` are seams for tests;
    callers pass fakes to record the docker invocation / inject a
    canned upstream USER without actually shelling out.
    """
    snippet = stub_runtime_dockerfile_path()
    if not snippet.is_file():
        out.write(
            f"error: stub-runtime Dockerfile missing at {snippet}; "
            "is xrlenv installed correctly?\n"
        )
        return 2

    resolver = upstream_user_resolver or _inspect_image_user
    upstream_user = resolver(base)

    cmd = [
        "docker", "build",
        "--build-arg", f"BASE_IMAGE={base}",
        "--build-arg", f"UPSTREAM_USER={upstream_user}",
        "--file", str(snippet),
        "--tag", out_tag,
        str(snippet.parent),
    ]
    out.write(f"$ {' '.join(cmd)}\n")
    if runner is not None:
        return runner(cmd)
    return subprocess.run(cmd, check=False).returncode


# ──────────────────────────────────────────────────────────────────────────────
# tokens issue (spec 19, Slice 8)
# ──────────────────────────────────────────────────────────────────────────────


def cmd_tokens_issue(
    role: str,
    *,
    owner: str | None = None,
    display_name: str | None = None,
    secrets_root: Path | None = None,
    out: TextIO,
) -> int:
    """Issue a fresh bearer token for ``role`` (node / consumer / operator).

    Without ``--owner``: writes mode-0600 ``<secrets_root>/<role>.token`` and
    prints the token once (the legacy single shared role-token,
    ``owner_id="default"``). Spec 19 §"Token lifecycle": *the CLI prints the
    new token once and refuses to print it again* — operators paste it into
    the systemd EnvironmentFile (node) or `XRLENV_{CONSUMER,OPERATOR}_TOKEN`
    env var.

    With ``--owner <id>``: mints a **per-user** token for that tenant
    (multi-user, ``notes/multi-user-fairshare-plan.md``). The bearer's
    SHA-256 + owner_id are appended to ``users.json``; the plaintext is
    printed once and never stored. Many per-user tokens of the same role
    coexist, so a running control plane hot-reloads the new user without a
    restart. ``role=node`` is rejected — nodes are infra, not tenants.
    """
    from xrlenv.control.security import (
        DEFAULT_SECRETS_ROOT,
        ROLE_DEFAULT_SCOPE,
        generate_token,
        token_digest_hint,
        write_secret_file,
        write_user_record,
    )

    if role not in ROLE_DEFAULT_SCOPE:
        out.write(
            f"error: unknown role {role!r}; expected one of {sorted(ROLE_DEFAULT_SCOPE)}\n"
        )
        return 2
    target_root = secrets_root or DEFAULT_SECRETS_ROOT

    if owner is not None:
        if not owner.strip():
            out.write("error: --owner must be a non-empty tenant id\n")
            return 2
        if role == "node":
            out.write(
                "error: --owner is not valid with role=node — nodes are "
                "infrastructure, not tenants. Mint a node token without "
                "--owner.\n"
            )
            return 2
        users_path = target_root / "users.json"
        token = generate_token(role)
        token_id = write_user_record(
            users_path, token=token, role=role,  # type: ignore[arg-type]
            owner_id=owner.strip(), display_name=display_name,
        )
        out.write(
            f"issued {role} token for owner={owner.strip()}"
            + (f" ({display_name})" if display_name else "")
            + f" (token_id={token_id})\n"
            f"  recorded at: {users_path} (hashed; plaintext not stored)\n"
            f"  scope:       {ROLE_DEFAULT_SCOPE[role]}\n"
            f"  raw token:   {token}\n"
            "\n"
            f"{_token_distribution_hint(role)}"
            "Copy the token now — it will not be shown again.\n"
        )
        return 0

    target = target_root / f"{role}.token"
    if target.exists():
        out.write(
            f"error: {target} already exists; rotate via "
            "`xrlenv tokens rotate` or remove the file first\n"
        )
        return 1
    token = generate_token(role)
    write_secret_file(target, token)
    out.write(
        f"issued {role} token (digest_hint={token_digest_hint(token)})\n"
        f"  stored at: {target} (mode 0600)\n"
        f"  scope:     {ROLE_DEFAULT_SCOPE[role]}\n"
        f"  raw token: {token}\n"
        "\n"
        f"{_token_distribution_hint(role)}"
        "Copy the token now — it will not be shown again.\n"
    )
    if role == "consumer":
        out.write(_shared_consumer_note())
    return 0


def _shared_consumer_note() -> str:
    """Warn-only nudge printed after a SHARED (no-``--owner``) consumer token
    is minted by ``tokens issue`` / ``tokens rotate``.

    A shared consumer token is owner_id="default". It authenticates rollouts
    fine, but on the admin panel it is *not* full admin — it grants a read-only
    view scoped to owner_id="default" jobs, and owner-crossing pages
    (fair-share) + write actions still require an operator token. Surfacing the
    reasons at mint time heads off the common "I issued a consumer token but
    can't see fair-share / admin" confusion. Shared between the two commands so
    the wording stays in lock-step.
    """
    return (
        "\n"
        'NOTE: this is a SHARED consumer token (owner_id="default"). It is\n'
        "      not an admin credential — on the admin panel it is read-only\n"
        '      and scoped to owner_id="default" jobs; the fair-share page and\n'
        "      write actions need an operator token "
        "(`xrlenv tokens issue operator`).\n"
        "      For a multi-tenant setup, mint a per-user token instead so\n"
        "      each tenant's rollouts are stamped + isolated by owner:\n"
        "          xrlenv tokens issue consumer --owner <id>\n"
    )


def _token_distribution_hint(role: str) -> str:
    """Per-role hint about where each freshly issued token belongs.

    Keeps ``tokens issue`` / ``rotate`` output operator-actionable
    instead of generic. Empty string for unrecognized roles so the CLI
    doesn't spam an awkward line if a new role appears before this
    helper is updated.
    """
    if role == "node":
        return (
            "Paste it into the node's systemd unit at "
            "[Service] Environment=XRLENV_NODE_TOKEN=<paste>.\n"
        )
    if role == "consumer":
        return (
            "Set XRLENV_CONSUMER_TOKEN in the user's .env / shell, or pass "
            "to Client.grpc(token=...). The same token also opens the admin "
            "panel read-only (scoped to their jobs): browse to "
            "http://<control-plane>:8080/ and paste this token on the sign-in "
            "page (use `log out` in the nav to switch tokens).\n"
        )
    if role == "viewer":
        return (
            "Share with teammates who need read-only admin panel access. "
            "Browser flow: navigate to http://<control-plane>:8080/ and paste "
            "this token on the sign-in page.\n"
        )
    if role == "operator":
        return (
            "Use for admin-write workflows: `xrlenv build apply "
            "--operator-token <token>` (CLI) or paste this token on the admin "
            "panel's sign-in page (browser); `log out` to switch tokens.\n"
        )
    return ""


def _parse_grace(spec: str) -> float:
    """Like :func:`parse_duration` but also accepts a bare integer (seconds).

    ``"0"`` → 0 (immediate cutover, the default). ``"3600"`` → 3600s.
    ``"24h"`` → 86400s. Negative values are rejected.
    """
    stripped = spec.strip()
    if not stripped:
        raise ValueError("empty --grace value")
    if stripped.lstrip("-").isdigit():
        as_int = int(stripped)
        if as_int < 0:
            raise ValueError(f"--grace must be non-negative; got {spec!r}")
        return float(as_int)
    return parse_duration(stripped)


def cmd_tokens_rotate(
    role: str,
    *,
    grace: str = "0",
    secrets_root: Path | None = None,
    out: TextIO,
) -> int:
    """``xrlenv tokens rotate <role> [--grace 24h]`` — replace ``role``'s
    bearer token. Immediate cutover by default; ``--grace`` keeps the
    prior token valid for an overlap window (deployment-rollover only).

    On-disk effect:

    - ``<secrets_root>/<role>.token`` is rewritten with a fresh secret.
    - With ``--grace > 0``, ``<role>.token.previous.json`` is created
      with the prior token + an ISO-8601 ``grace_until`` timestamp; the
      control plane's :class:`TokenStore` accepts that token until its
      grace expires.

    The new token is printed once. Operators paste it into the
    consumer's env / systemd EnvironmentFile, exactly like ``tokens
    issue``.
    """
    from xrlenv.control.security import (
        DEFAULT_SECRETS_ROOT,
        ROLE_DEFAULT_SCOPE,
        generate_token,
        token_digest_hint,
        token_full_id,
        write_grace_record,
        write_secret_file,
    )

    if role not in ROLE_DEFAULT_SCOPE:
        out.write(
            f"error: unknown role {role!r}; expected one of {sorted(ROLE_DEFAULT_SCOPE)}\n"
        )
        return 2
    try:
        grace_s = _parse_grace(grace)
    except ValueError as exc:
        out.write(f"error: {exc}\n")
        return 2
    target_root = secrets_root or DEFAULT_SECRETS_ROOT
    target = target_root / f"{role}.token"
    if not target.exists():
        out.write(
            f"error: {target} does not exist; nothing to rotate. "
            "Use `xrlenv tokens issue` first.\n"
        )
        return 1
    prior_token = target.read_text(encoding="utf-8").strip()
    if not prior_token:
        out.write(
            f"error: {target} is empty; nothing to rotate. "
            "Use `xrlenv tokens issue` first.\n"
        )
        return 1
    new_token = generate_token(role)
    grace_path = target_root / f"{role}.token.previous.json"
    # Order: write the previous-token sidecar first when grace > 0 so
    # the control plane's reload can't see the new active token without
    # also seeing its grace partner. With grace == 0, drop any stale
    # sidecar instead.
    if grace_s > 0:
        grace_until = _dt.datetime.now(_dt.UTC) + _dt.timedelta(seconds=grace_s)
        write_grace_record(grace_path, prior_token, grace_until)
    elif grace_path.exists():
        grace_path.unlink()
    write_secret_file(target, new_token)
    summary = (
        f"rotated {role} token "
        f"(new digest_hint={token_digest_hint(new_token)}, "
        f"new token_id={token_full_id(new_token)})\n"
        f"  stored at: {target} (mode 0600)\n"
        f"  scope:     {ROLE_DEFAULT_SCOPE[role]}\n"
        f"  raw token: {new_token}\n"
    )
    if grace_s > 0:
        summary += (
            f"  prior token kept valid for: {grace_s:g}s "
            f"(see {grace_path})\n"
        )
    else:
        summary += "  prior token: invalidated immediately\n"
    summary += (
        "\n"
        f"{_token_distribution_hint(role)}"
        "Copy the token now — it will not be shown again.\n"
    )
    # Rotate only ever rewrites the shared ``<role>.token`` (no per-owner
    # path), so a rotated consumer token is always owner_id="default" — same
    # nudge as ``tokens issue`` so neither path silently mints a shared token.
    if role == "consumer":
        summary += _shared_consumer_note()
    out.write(summary)
    return 0


def cmd_tokens_revoke(
    token_id_or_prefix: str,
    *,
    secrets_root: Path | None = None,
    out: TextIO,
) -> int:
    """``xrlenv tokens revoke <token-id>`` — mark a token revoked so a
    running control plane refuses it on the next ``maybe_reload``.

    Matches by exact 12-char ``token_id`` or by any ≥6-char unique
    prefix. An ambiguous prefix returns code 2; an unknown id returns
    code 1. Idempotent — re-revoking an already-revoked id is a no-op
    success.
    """
    from xrlenv.control.security import (
        DEFAULT_SECRETS_ROOT,
        TokenStore,
        append_revocation,
    )

    target_root = secrets_root or DEFAULT_SECRETS_ROOT
    store = TokenStore.load(secrets_root=target_root)
    try:
        full_id = store.revoke(token_id_or_prefix)
    except ValueError as exc:
        out.write(f"error: {exc}\n")
        return 2
    except LookupError as exc:
        out.write(f"error: {exc}\n")
        return 1
    revoked_path = target_root / "revoked.json"
    append_revocation(revoked_path, full_id)
    out.write(
        f"revoked token_id={full_id} (persisted to {revoked_path})\n"
        "The control plane will reject this token on its next "
        "hot-reload check.\n",
    )
    return 0


def cmd_tokens_list(
    *,
    secrets_root: Path | None = None,
    out: TextIO,
) -> int:
    """``xrlenv tokens list`` — show active token + grace + revocation
    state per role, with token_ids the operator can pass to ``revoke``.

    Never prints the raw token bytes — only digest hints + token_ids.
    """
    from xrlenv.control.security import (
        DEFAULT_SECRETS_ROOT,
        TokenStore,
        token_sha256,
    )

    target_root = secrets_root or DEFAULT_SECRETS_ROOT
    store = TokenStore.load(secrets_root=target_root)
    if store.is_empty:
        out.write(
            f"no tokens loaded from {target_root} or env vars; "
            "use `xrlenv tokens issue <role>` to create one.\n"
        )
        return 0
    out.write(f"tokens loaded from {target_root}:\n")
    grace_map = dict(store._grace_expires)
    for role in store.known_roles:
        active_token = store._by_role[role]
        active_identity = store._by_token.get(active_token)
        if active_identity is None:
            # Step-5 collision reconciliation (TokenStore.load) dropped this
            # role's active token from the legacy map because it has the same
            # value as a per-user token: the per-user identity now wins (see the
            # WARN emitted at load). `_by_role` still points at that token, so
            # surface the shadowing honestly instead of KeyError-ing here.
            shadow = store._by_token_sha.get(token_sha256(active_token))
            if shadow is not None:
                out.write(
                    f"  {role:<9} shadowed token_id={shadow.token_id} "
                    f"digest_hint={shadow.digest_hint} owner={shadow.owner_id} "
                    f"(shared {role} token value collides with a per-user "
                    f"bearer; authenticates as owner={shadow.owner_id})\n",
                )
            else:
                out.write(
                    f"  {role:<9} unresolved active token is not in the "
                    f"identity map (revoked or reconciled away)\n",
                )
            continue
        out.write(
            f"  {role:<9} active   token_id={active_identity.token_id} "
            f"digest_hint={active_identity.digest_hint} owner=default\n",
        )
        for tok, expires_at in grace_map.items():
            identity = store._by_token.get(tok)
            if identity is None or identity.role != role:
                continue
            remaining = max(0.0, expires_at - time.time())
            out.write(
                f"  {role:<9} grace    token_id={identity.token_id} "
                f"digest_hint={identity.digest_hint} "
                f"remaining={remaining:.0f}s\n",
            )
    users = store.users()
    if users:
        out.write("per-user tokens (multi-user):\n")
        for identity in users:
            label = f" ({identity.display_name})" if identity.display_name else ""
            state = (
                "revoked"
                if identity.token_id in store._revoked_token_ids
                else "user"
            )
            out.write(
                f"  {identity.role:<9} {state:<8} token_id={identity.token_id} "
                f"digest_hint={identity.digest_hint} "
                f"owner={identity.owner_id}{label}\n",
            )
    if store._revoked_token_ids:
        out.write("revoked token_ids (verify path refuses these):\n")
        for tid in sorted(store._revoked_token_ids):
            out.write(f"  {tid}\n")
    return 0


def cmd_build_apply(
    *,
    plan_path: Path | None,
    benchmark: str | None,
    smoke: bool,
    instances: str | None,
    all_: bool,
    build_path: str | None,
    replication: int | None,
    reserved_runtime_gb: int,
    buffer_gb: int,
    dry_run: bool,
    force: bool,
    state_db: Path,
    runs_root: Path,
    out: TextIO,
    connect_host: str | None = None,
    connect_port: int = 8080,
    operator_token: str | None = None,
    eager: bool = False,
    fill_missing: bool = False,
    tarball_max_bytes: int | None = None,
    skip_if_present: bool = False,
    concurrency: int | None = None,
) -> int:
    """``xrlenv build apply`` — distribute image builds across the cluster.

    Two input modes:

    1. ``--plan PATH`` — declarative ``build-plan.yaml`` (source of
       truth, persisted in state.db; idempotent re-applies).
    2. Imperative shorthand: ``--benchmark NAME --smoke|--instances|--all``
       — CLI lowers to a transient :class:`BuildPlan` and feeds the same
       coordinator. Useful for one-off operator iteration; ``--dry-run``
       prints the placement so you can save it as a YAML.
    """
    import asyncio

    from xrlenv.api.constants import DEFAULT_BUILD_TARBALL_MAX_BYTES
    from xrlenv.control.build_plan import (
        BenchmarkBuildSpec,
        BenchmarkSelection,
        BuildBudget,
        BuildPlan,
        load_build_plan,
        resolve_tarball_sources,
    )
    from xrlenv.control.image_planner import InsufficientCapacity
    from xrlenv.control.runtime import build_local_runtime
    from xrlenv.errors import ManifestInvalid

    # Validate the input-mode XOR.
    if (plan_path is None) == (benchmark is None):
        out.write(
            "error: pass either --plan PATH or --benchmark NAME (not both, "
            "not neither)\n",
        )
        return 2

    # --fill-missing flag conflicts.
    if fill_missing and force:
        out.write(
            "error: --fill-missing and --force are mutually exclusive. "
            "--force means 'rebuild everything from scratch'; "
            "--fill-missing means 'only build what isn't present'. "
            "Pick one.\n",
        )
        return 2
    if fill_missing and eager:
        out.write(
            "error: --fill-missing and --eager are mutually exclusive. "
            "--eager asserts the full plan fits at apply time; "
            "--fill-missing only places the missing subset, so the "
            "assertion is moot.\n",
        )
        return 2
    if fill_missing and connect_host is None:
        out.write(
            "error: --fill-missing requires --connect-host. The cluster "
            "inventory probe needs a live control plane with connected "
            "node-agents; the local-only LocalRuntime path doesn't have "
            "the inventory_provider wired.\n",
        )
        return 2

    # Lower imperative shorthand to a transient plan when --benchmark is set.
    if plan_path is not None:
        try:
            plan = load_build_plan(plan_path)
            # Sub-slice 1.b: tarball entries' bytes are loaded
            # operator-side here so the plan that reaches the
            # coordinator (local OR cluster admin) is wire-ready.
            # The base_dir resolves relative tarball ``path`` values
            # against the plan YAML's directory, matching how YAML
            # references are typically authored.
            plan = resolve_tarball_sources(
                plan,
                max_bytes=(
                    tarball_max_bytes
                    if tarball_max_bytes is not None
                    else DEFAULT_BUILD_TARBALL_MAX_BYTES
                ),
                base_dir=plan_path.parent,
            )
        except ManifestInvalid as exc:
            out.write(f"error: {exc}\n")
            return 2
    else:
        # benchmark is set; validate the selection trio.
        assert benchmark is not None
        selected = sum((bool(smoke), bool(instances), bool(all_)))
        if selected != 1:
            out.write(
                "error: with --benchmark, pass exactly one of --smoke / "
                "--instances LIST / --all\n",
            )
            return 2
        sel_kwargs: dict[str, Any] = {}
        if smoke:
            sel_kwargs["smoke"] = True
        elif instances:
            sel_kwargs["instances"] = tuple(
                s.strip() for s in instances.split(",") if s.strip()
            )
        else:
            sel_kwargs["all"] = True
        try:
            plan = BuildPlan(
                replication=replication or 1,
                budget=BuildBudget(
                    reserved_runtime_gb=reserved_runtime_gb,
                    buffer_gb=buffer_gb,
                ),
                benchmarks=(BenchmarkBuildSpec(
                    name=benchmark,
                    selection=BenchmarkSelection(**sel_kwargs),
                    build_path=build_path,
                ),),
            )
        except Exception as exc:
            out.write(f"error: invalid plan: {exc}\n")
            return 2

    runs_root.mkdir(parents=True, exist_ok=True)

    # P1.6.f cluster-RPC: when --connect-host is set, dispatch via the
    # admin server's POST /api/build/apply + poll instead of building a
    # LocalRuntime. The active-cluster safety guard below is local-only
    # — the cluster path is the operator's explicit opt-in.
    if connect_host is not None:
        return _build_apply_via_admin(
            host=connect_host, port=connect_port,
            operator_token=operator_token, plan=plan,
            dry_run=dry_run, force=force, eager=eager,
            fill_missing=fill_missing,
            skip_if_present=skip_if_present, out=out,
            concurrency=concurrency,
        )

    # Audit P1.6-H1 guard: refuse to run the local-only path against a
    # state.db that's currently fronting a live control plane. Prior to
    # this guard, the LocalRuntime startup-sweep would mark genuinely-
    # connected nodes as ``lost`` and corrupt the operator's live view
    # of the cluster. Mirror the 30s heartbeat-staleness window the
    # admin's nodes view uses.
    if state_db.exists():
        from xrlenv.control.state import SqliteStateStore as _Store

        _check_state = _Store(state_db)
        try:
            now = time.time()
            live = [
                n for n in _check_state.list_nodes(status="connected")
                if (now - n.last_seen_at) < 30.0
            ]
        finally:
            _check_state.close()
        if live:
            ids = ", ".join(n.node_id for n in live[:5])
            out.write(
                "error: refusing to run local-only ``xrlenv build apply`` "
                "while another control plane appears active.\n"
                f"       {len(live)} node(s) heartbeated within the last 30s: {ids}\n"
                "       Running the local path against this state.db would "
                "mark those connected nodes as ``lost``.\n"
                "       To dispatch to the live cluster instead, pass "
                "``--connect-host <admin-host>`` (see\n"
                "       docs/deployment/build_plans.md). Or stop "
                "``xrlenv up`` first if you really want the local path.\n",
            )
            return 2

    async def _run() -> int:
        runtime = build_local_runtime(
            runs_root=runs_root, state_db_path=state_db,
            skip_stale_node_sweep=True,
        )
        await runtime.start()
        try:
            try:
                outcome = await runtime.build_coordinator.apply(
                    plan, dry_run=dry_run, force=force, eager=eager,
                    applied_by="cli",
                    skip_if_present=skip_if_present,
                    concurrency=concurrency,
                )
            except InsufficientCapacity as exc:
                out.write(f"error: {exc}\n")
                return 1
            except ManifestInvalid as exc:
                out.write(f"error: {exc}\n")
                return 2

            placement = outcome.placement
            if placement is not None:
                out.write(
                    f"plan_id: {outcome.plan_id}  "
                    f"status: {outcome.status}\n",
                )
                out.write(
                    f"placements: {len(placement.assignments)} pre-built; "
                    f"{outcome.deferred} deferred (lazy)\n",
                )
                for node_id, rows in sorted(placement.assignments_by_node.items()):
                    out.write(f"  {node_id}: {len(rows)} image(s)\n")
                    for a in rows[:10]:
                        out.write(f"    - {a.image_ref}\n")
                    if len(rows) > 10:
                        out.write(f"    ... +{len(rows) - 10} more\n")
                if outcome.deferred:
                    out.write(
                        f"  (+{outcome.deferred} deferred — registered as "
                        f"`registered`; lazy-built on first rollout)\n",
                    )
            if outcome.failures:
                out.write(f"failures: {outcome.failures}\n")
                for line in outcome.error_summary:
                    out.write(f"  ! {line}\n")
            if outcome.status in ("completed", "dry_run", "no_op_already_completed"):
                return 0
            if outcome.status == "rejected_in_flight":
                out.write(
                    "another plan is already in_flight — wait for it to finish "
                    "or `xrlenv build cancel`\n",
                )
                return 2
            # partial_failure or anything else: non-zero exit.
            return 1
        finally:
            await runtime.shutdown()

    return asyncio.run(_run())


def _resolve_operator_token(explicit: str | None) -> str | None:
    """Pick the operator token from (in priority order): explicit
    --operator-token CLI flag, ``$XRLENV_OPERATOR_TOKEN`` env var, or
    ``$XRLENV_HOME/secrets/operator.token`` (default
    ``~/.xrlenv/secrets/operator.token``, mode 0600). Returns None
    when no token is configured — the admin server will accept
    anonymous loopback requests in that case."""
    if explicit:
        return explicit
    env_val = os.environ.get("XRLENV_OPERATOR_TOKEN")
    if env_val:
        return env_val
    path = DEFAULT_XRLENV_HOME / "secrets" / "operator.token"
    if path.is_file():
        try:
            return path.read_text().strip() or None
        except OSError:
            return None
    return None


def _build_apply_via_admin(
    *,
    host: str,
    port: int,
    operator_token: str | None,
    plan: Any,
    dry_run: bool,
    force: bool,
    eager: bool,
    out: TextIO,
    fill_missing: bool = False,
    skip_if_present: bool = False,
    concurrency: int | None = None,
) -> int:
    """P1.6.f cluster-RPC — POST the plan to the admin server's
    apply endpoint, poll until terminal, print results.

    The admin endpoint kicks off a background asyncio task and
    returns 202 + ``{plan_id, status: "in_flight"}`` immediately;
    the CLI polls ``GET /api/build/plans/<plan_id>`` every ~3s
    until the plan reaches a terminal status.
    """
    import httpx

    base = f"http://{host}:{port}"
    token = _resolve_operator_token(operator_token)
    headers: dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    body: dict[str, object] = {
        "plan": plan.model_dump(mode="json", exclude_none=True),
        "dry_run": dry_run,
        "force": force,
        "eager": eager,
        "fill_missing": fill_missing,
        "skip_if_present": skip_if_present,
        "applied_by": "operator-cli",
    }
    if concurrency is not None:
        body["concurrency"] = concurrency

    try:
        with httpx.Client(base_url=base, headers=headers, timeout=15.0) as client:
            r = client.post("/api/build/apply", json=body)
    except httpx.HTTPError as exc:
        out.write(
            f"error: cannot reach admin at {base}: {exc}\n"
            "       Confirm xrlenv up is running with "
            "--admin-port "
            f"{port} and reachable.\n",
        )
        return 2

    if r.status_code == 401:
        out.write(
            "error: 401 unauthorized — operator token required.\n"
            "       Pass --operator-token, set $XRLENV_OPERATOR_TOKEN, "
            f"or place the token at {DEFAULT_XRLENV_HOME / 'secrets' / 'operator.token'}.\n",
        )
        return 2
    if r.status_code == 403:
        out.write("error: 403 forbidden — token role is not 'operator'.\n")
        return 2
    if r.status_code == 503:
        out.write(
            "error: 503 — admin server reachable but no build coordinator wired.\n"
            "       This is expected when xrlenv up was started without the "
            "build path; restart with the latest control plane.\n",
        )
        return 2
    if r.status_code == 400:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        out.write(f"error: 400 from admin: {detail}\n")
        return 2
    if r.status_code not in (200, 202):
        out.write(f"error: admin returned {r.status_code}: {r.text}\n")
        return 1

    payload = r.json()
    plan_id = str(payload.get("plan_id") or "")

    if dry_run:
        out.write(
            f"plan_id: {plan_id}  status: {payload.get('status')}\n",
        )
        placement = payload.get("placement") or []
        out.write(f"placements: {len(placement)} image(s)\n")
        by_node: dict[str, list[dict[str, Any]]] = {}
        for a in placement:
            by_node.setdefault(a["node_id"], []).append(a)
        for node_id, rows in sorted(by_node.items()):
            out.write(f"  {node_id}: {len(rows)} image(s)\n")
            for a in rows[:10]:
                out.write(f"    - {a['image_ref']}\n")
            if len(rows) > 10:
                out.write(f"    ... +{len(rows) - 10} more\n")
        return 0

    if payload.get("status") == "no_op_already_completed":
        out.write(
            f"plan_id: {plan_id}  status: no_op_already_completed\n"
            "(this plan was already applied + completed; pass --force "
            "to rebuild)\n",
        )
        return 0

    # in_flight → poll.
    out.write(f"plan_id: {plan_id}\nstatus: in_flight  (polling every 3s)\n")
    poll_url = f"/api/build/plans/{plan_id}"
    # Silence httpx's per-request INFO log (one line per 3s poll
    # floods the operator's terminal with no signal). Restored
    # before returning so other CLI subcommands are unaffected.
    httpx_logger = logging.getLogger("httpx")
    prior_httpx_level = httpx_logger.level
    httpx_logger.setLevel(logging.WARNING)
    last_counts = ""
    poll_started_at = time.monotonic()
    last_progress_emit_at = 0.0
    # Persistence-deadline tolerance: the admin server returns 202
    # the moment it spawns the background apply task, but the
    # ``record_build_plan`` call doesn't fire until the coordinator
    # passes the source-type gate, dispatcher checks, budget snapshot,
    # and FFD placement. On a healthy cluster that's <1 s; under
    # load (slow nodes, large plans, network blips) it can take 5-15 s.
    # Tolerate 404 for up to 30 s after the POST so the CLI doesn't
    # exit on a transient race that the smoke's apply_plan_remote
    # helper already handles.
    persistence_deadline = time.monotonic() + 30.0
    try:
        return _poll_until_terminal(
            base=base, headers=headers, poll_url=poll_url,
            plan_id=plan_id,
            persistence_deadline=persistence_deadline,
            poll_started_at=poll_started_at,
            last_counts=last_counts,
            last_progress_emit_at=last_progress_emit_at,
            out=out,
        )
    finally:
        httpx_logger.setLevel(prior_httpx_level)


def _poll_until_terminal(
    *, base: str, headers: dict[str, str], poll_url: str,
    plan_id: str, persistence_deadline: float,
    poll_started_at: float, last_counts: str,
    last_progress_emit_at: float, out: TextIO,
) -> int:
    import httpx

    while True:
        time.sleep(3.0)
        try:
            with httpx.Client(
                base_url=base, headers=headers, timeout=15.0,
            ) as client:
                r = client.get(poll_url)
        except httpx.HTTPError as exc:
            out.write(f"warning: poll failed: {exc}; retrying...\n")
            continue
        if r.status_code == 404:
            if time.monotonic() > persistence_deadline:
                out.write(
                    f"error: plan {plan_id} never appeared in state.db "
                    "(404 from /api/build/plans/<plan_id> for >30s after "
                    "POST). The coordinator's apply likely raised before "
                    "record_build_plan — common causes:\n"
                    "  - zero nodes connected (cluster just started, "
                    "or all nodes lost). Check `xrlenv nodes`.\n"
                    "  - InsufficientCapacity from the FFD bin-packer "
                    "(no node has room for at least one entry).\n"
                    "Look for `build coordinator apply raised` in the "
                    "admin server logs for the underlying exception.\n",
                )
                return 1
            continue
        if r.status_code != 200:
            out.write(
                f"error: poll returned {r.status_code}: {r.text}\n",
            )
            return 1
        snap = r.json()
        per = snap.get("per_status") or {}
        counts = (
            f"done={per.get('done', 0)} "
            f"failed={per.get('failed', 0)} "
            f"building={per.get('building', 0)} "
            f"pending={per.get('pending', 0)}"
        )
        cancelled = per.get("cancelled", 0)
        if cancelled:
            counts += f" cancelled={cancelled}"
        deferred = per.get("registered", 0)
        if deferred:
            counts += f" deferred={deferred}"
        evicted = per.get("evicted", 0)
        if evicted:
            counts += f" evicted={evicted}"
        # Progress emission policy: print on every count change,
        # plus a heartbeat every 30 s when nothing's moving so the
        # operator sees the CLI is alive (long-running builds can
        # sit at the same `building=N` line for minutes per entry).
        now = time.monotonic()
        elapsed = int(now - poll_started_at)
        if counts != last_counts or (now - last_progress_emit_at) >= 30.0:
            out.write(f"  [{elapsed:4d}s] {counts}\n")
            last_counts = counts
            last_progress_emit_at = now
        if snap["status"] in ("completed", "partial_failure", "cancelled", "superseded"):
            out.write(f"\nstatus: {snap['status']}\n")
            failures = [
                a for a in snap.get("assignments", [])
                if a.get("status") == "failed"
            ]
            if failures:
                out.write(f"failures: {len(failures)}\n")
                for f in failures[:10]:
                    out.write(
                        f"  ! {f['node_id']}/{f['image_ref']}: "
                        f"{f.get('error') or 'unknown'}\n",
                    )
            return 0 if snap["status"] == "completed" else 1


def _resolve_plan_id(
    state: Any, plan_id_or_prefix: str, *, out: TextIO,
) -> Any | None:
    """Return the unique ``BuildPlanRecord`` whose ``plan_id`` either
    equals or starts with ``plan_id_or_prefix``.

    Operators copy short ids out of the admin panel (which renders
    the first 12 chars). Forcing a full SHA-256 paste is hostile;
    the CLI accepts any unique prefix (>=4 chars). Behavior:

    - exact match wins, even when the prefix would also match other
      plan_ids (defensive — pasting a complete id never errors).
    - prefix match: returns the unique match; errors on ambiguity
      and lists candidates so the operator can disambiguate.
    - no match: writes a clear error to ``out``, returns None.
    """
    record = state.get_build_plan(plan_id_or_prefix)
    if record is not None:
        return record
    if len(plan_id_or_prefix) < 4:
        out.write(
            f"error: plan_id prefix {plan_id_or_prefix!r} is too short "
            f"(need at least 4 chars to disambiguate).\n",
        )
        return None
    matches = [
        p for p in state.list_build_plans()
        if p.plan_id.startswith(plan_id_or_prefix)
    ]
    if not matches:
        out.write(f"error: plan_id {plan_id_or_prefix!r} not found\n")
        return None
    if len(matches) > 1:
        out.write(
            f"error: plan_id prefix {plan_id_or_prefix!r} is ambiguous "
            f"({len(matches)} plans match):\n",
        )
        for m in matches[:10]:
            out.write(f"  {m.plan_id}  ({m.name or '(unnamed)'}, {m.status})\n")
        if len(matches) > 10:
            out.write(f"  ... and {len(matches) - 10} more\n")
        out.write("Pass more characters of the plan_id.\n")
        return None
    return matches[0]


def cmd_build_status(
    *,
    plan_id: str | None,
    state_db: Path,
    out: TextIO,
) -> int:
    """``xrlenv build status`` — read the persisted snapshot.

    Without ``--plan``, prints the most recent applied plan + a
    summary of its assignment statuses. With ``--plan PLAN_ID``, prints
    the per-row breakdown for that plan. ``PLAN_ID`` accepts either
    a full SHA-256 plan_id or a unique prefix (>=4 chars) — handy
    for pasting the 12-char short id from the admin /builds panel.

    Drift detection between the snapshot and live node inventory is
    a phase-1.6.d follow-on (it requires walking each node's live
    ``report_images()`` and diffing — for distributed mode that's a
    connect-mode operation).
    """
    if not state_db.exists():
        out.write(f"error: state.db not found at {state_db}\n")
        return 2
    from xrlenv.control.state import SqliteStateStore

    state = SqliteStateStore(state_db, read_only=True)  # read: build-plan status
    try:
        if plan_id is None:
            plans = state.list_build_plans()
            if not plans:
                out.write("no build plans applied yet\n")
                return 0
            plan = plans[0]  # most recent
        else:
            plan_record = _resolve_plan_id(state, plan_id, out=out)
            if plan_record is None:
                return 1
            plan = plan_record

        rows = state.list_assignments(plan.plan_id)
        per_status: dict[str, int] = {}
        for r in rows:
            per_status[r.status] = per_status.get(r.status, 0) + 1

        out.write(
            f"plan_id: {plan.plan_id}\n"
            f"status:  {plan.status}\n"
            f"applied: {plan.applied_at:.0f}  by: {plan.applied_by}\n",
        )
        out.write(f"assignments: {len(rows)} total\n")
        for status in (
            "done", "failed", "cancelled",
            "building", "pending", "registered", "evicted",
        ):
            n = per_status.get(status, 0)
            if n:
                label = {
                    "registered": "registered (deferred — lazy on first rollout)",
                    "evicted": "evicted (cache reclaim — re-build on next ensure_present)",
                    "cancelled": "cancelled (operator-cancelled via xrlenv build cancel)",
                }.get(status, status)
                out.write(f"  {label}: {n}\n")

        # Show first 10 failures with their error string.
        failures = [r for r in rows if r.status == "failed"]
        if failures:
            out.write(f"first {min(10, len(failures))} failure(s):\n")
            for r in failures[:10]:
                out.write(
                    f"  ! {r.node_id}/{r.image_ref}: {r.error or 'unknown'}\n",
                )
        return 0
    finally:
        state.close()


def cmd_build_cancel(
    *,
    plan_id: str,
    state_db: Path,
    out: TextIO,
    connect_host: str | None = None,
    connect_port: int = 8080,
    operator_token: str | None = None,
) -> int:
    """``xrlenv build cancel --plan <id>`` — cancel an in-flight build
    plan. Accepts a full plan_id or a unique prefix (>=4 chars), same
    as ``xrlenv build status``.

    Two modes:

    **Cluster-side cancel** (``--connect-host`` set, the recommended
    path for a live cluster apply): POSTs to the admin server's
    ``/api/build/cancel`` endpoint, which marks the plan ``cancelled``
    AND dispatches a ``CancelBuildImageCommand`` over the spec-21
    stream to each node currently building an assignment. Each node
    interrupts its in-flight ``docker build`` (best-effort: kills the
    running build container labeled with the matching cancel-key,
    cancels the asyncio task) and the BuildImage command_id completes
    with a ``failed: cancelled by operator`` reply. Pending
    assignments transition to ``cancelled``; the admin /builds panel
    converges on the new state.

    **Local-only cancel** (no ``--connect-host``): updates the plan
    record's status to ``cancelled`` in state.db so the admin panel
    and operator polling converge, BUT does not interrupt any
    in-flight ``docker build`` already running on a remote node.
    Useful when the cluster is unreachable, when there are no live
    builds, or when you just want to clear a stuck plan record so
    re-apply (with or without ``--force``) goes through cleanly.

    Use case for cluster-side: operator's CLI disconnected mid-apply
    and the cluster is still grinding through a 30-min build of a
    plan they no longer want.
    """
    if connect_host is not None:
        return _build_cancel_via_admin(
            host=connect_host, port=connect_port,
            operator_token=operator_token,
            plan_id=plan_id, out=out,
        )

    if not state_db.exists():
        out.write(f"error: state.db not found at {state_db}\n")
        return 2
    from xrlenv.control.state import SqliteStateStore

    state = SqliteStateStore(state_db, read_only=False)  # WRITE: cancels a build plan
    try:
        plan = _resolve_plan_id(state, plan_id, out=out)
        if plan is None:
            return 1
        if plan.status in ("completed", "cancelled", "superseded"):
            out.write(
                f"plan {plan.plan_id} is already terminal "
                f"(status={plan.status}); no-op.\n",
            )
            return 0
        prior = plan.status
        state.update_build_plan_status(plan.plan_id, "cancelled")
        out.write(
            f"plan {plan.plan_id} ({plan.name or '(unnamed)'}): "
            f"{prior} → cancelled\n"
            "Note: this is a local-only cancel — in-flight builds on "
            "cluster nodes are NOT interrupted. To stop running builds, "
            "re-run with --connect-host to reach the admin server.\n",
        )
        return 0
    finally:
        state.close()


def _build_cancel_via_admin(
    *,
    host: str,
    port: int,
    operator_token: str | None,
    plan_id: str,
    out: TextIO,
) -> int:
    """Cluster-side cancel via the admin ``/api/build/cancel`` endpoint."""
    import httpx

    base = f"http://{host}:{port}"
    token = _resolve_operator_token(operator_token)
    headers: dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    body = {"plan_id": plan_id}
    try:
        with httpx.Client(base_url=base, headers=headers, timeout=60.0) as client:
            r = client.post("/api/build/cancel", json=body)
    except httpx.HTTPError as exc:
        out.write(
            f"error: cannot reach admin at {base}: {exc}\n"
            "       Confirm xrlenv up is running with --admin-port "
            f"{port} and reachable.\n",
        )
        return 2

    if r.status_code == 401:
        out.write(
            "error: 401 unauthorized — operator token required.\n"
            "       Pass --operator-token, set $XRLENV_OPERATOR_TOKEN, "
            f"or place the token at {DEFAULT_XRLENV_HOME / 'secrets' / 'operator.token'}.\n",
        )
        return 2
    if r.status_code == 403:
        out.write("error: 403 forbidden — token role is not 'operator'.\n")
        return 2
    if r.status_code == 404:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        out.write(f"error: 404 — {detail}\n")
        return 1
    if r.status_code == 409:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        out.write(f"error: 409 — {detail}\n")
        return 1
    if r.status_code == 400:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        out.write(f"error: 400 — {detail}\n")
        return 1
    if r.status_code != 200:
        out.write(f"error: admin returned {r.status_code}: {r.text}\n")
        return 1

    payload = r.json()
    plan_id_full = str(payload.get("plan_id") or plan_id)
    status = str(payload.get("status") or "")
    cancelled_count = int(payload.get("cancelled_count") or 0)
    errors = payload.get("errors") or []
    note = payload.get("note")

    if note:
        out.write(f"plan {plan_id_full}: {note}\n")
        return 0

    out.write(
        f"plan {plan_id_full}: status → {status}; "
        f"{cancelled_count} assignment(s) cancelled\n",
    )
    if errors:
        out.write(f"{len(errors)} per-node error(s):\n")
        for e in errors:
            out.write(
                f"  - node={e.get('node_id')} "
                f"image={e.get('image_ref')}: {e.get('error')}\n",
            )
        # Cluster-side cancel had partial errors but the plan is
        # marked cancelled — still a success at the operator-visible
        # layer (the offending builds will fail on their own / be
        # GCed on node disconnect). Exit 0.
    return 0


def cmd_build_calibrate(
    *,
    plan_path: Path,
    output_path: Path,
    out: TextIO,
    connect_host: str | None,
    connect_port: int = 8080,
    operator_token: str | None = None,
) -> int:
    """``xrlenv build calibrate`` — replace size hints with measured
    values from the cluster.

    Walks the input YAML's entries, asks the admin server for the
    max ``size_bytes`` each ``image_ref`` reports across connected
    nodes, and writes a calibrated YAML where every measured entry
    has its ``placement.size_hint_bytes`` updated to the cluster
    max and ``placement.size_hint_source`` set to ``cluster-reported``.
    Unmeasured entries (no node has materialized that ref yet) keep
    their operator-supplied hints.

    Output is written to a separate file so the operator can diff +
    decide before promoting (typical flow: ``xrlenv build calibrate
    --plan plan.yaml --output plan.calibrated.yaml --connect-host
    127.0.0.1`` → review → ``mv plan.calibrated.yaml plan.yaml ; git
    commit``).

    Why operator-driven (F5 lock): the calibrated sizes feed FFD
    placement, which directly affects which nodes get which images
    on the next apply. Auto-on-apply would silently reshape the
    cluster between unrelated runs; explicit invocation gives the
    operator a chance to review.

    --connect-host is required: there's no useful local-only
    fallback (without a cluster, there's nothing to measure).
    """
    if connect_host is None:
        out.write(
            "error: --connect-host is required for ``xrlenv build "
            "calibrate``. There's no useful local-only fallback "
            "(without a cluster, there's nothing to measure).\n",
        )
        return 2
    if not plan_path.is_file():
        out.write(f"error: plan not found at {plan_path}\n")
        return 2

    import httpx
    import yaml as _yaml

    from xrlenv.control.build_plan import load_build_plan
    from xrlenv.errors import ManifestInvalid

    try:
        plan = load_build_plan(plan_path)
    except ManifestInvalid as exc:
        out.write(f"error: {exc}\n")
        return 2

    base = f"http://{connect_host}:{connect_port}"
    token = _resolve_operator_token(operator_token)
    headers: dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    body = {"plan": plan.model_dump(mode="json", exclude_none=True)}
    try:
        with httpx.Client(base_url=base, headers=headers, timeout=60.0) as client:
            r = client.post("/api/build/calibrate", json=body)
    except httpx.HTTPError as exc:
        out.write(
            f"error: cannot reach admin at {base}: {exc}\n"
            "       Confirm xrlenv up is running with --admin-port "
            f"{connect_port} and reachable.\n",
        )
        return 2

    if r.status_code == 401:
        out.write(
            "error: 401 unauthorized — operator token required.\n"
            "       Pass --operator-token, set $XRLENV_OPERATOR_TOKEN, "
            f"or place the token at {DEFAULT_XRLENV_HOME / 'secrets' / 'operator.token'}.\n",
        )
        return 2
    if r.status_code == 503:
        out.write("error: 503 — admin server not wired for cluster reachability.\n")
        return 2
    if r.status_code != 200:
        out.write(f"error: admin returned {r.status_code}: {r.text}\n")
        return 1

    payload = r.json()
    calibrated: dict[str, int] = dict(payload.get("calibrated") or {})
    unmeasured: list[str] = list(payload.get("unmeasured") or [])
    nodes_queried: int = int(payload.get("nodes_queried") or 0)
    unreachable: list[dict[str, str]] = list(payload.get("nodes_unreachable") or [])

    out.write(
        f"calibrate: {len(calibrated)} measured / "
        f"{len(unmeasured)} unmeasured "
        f"({nodes_queried} node(s) queried)\n",
    )
    if unreachable:
        out.write(f"unreachable nodes ({len(unreachable)}):\n")
        for u in unreachable[:10]:
            out.write(
                f"  - {u.get('node_id')}: {u.get('error')}\n",
            )

    # Walk the YAML in place. We re-parse the YAML structure (rather
    # than re-serializing the pydantic model) so operator comments +
    # field ordering survive the round-trip — operators commonly
    # hand-edit these files.
    raw_yaml = _yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    if not isinstance(raw_yaml, dict):
        out.write(f"error: {plan_path} is malformed YAML\n")
        return 1
    entries = raw_yaml.get("entries")
    if not isinstance(entries, list):
        out.write(
            f"error: {plan_path} has no per-image-ref ``entries:`` list "
            "to calibrate (legacy benchmark-driven plans aren't "
            "supported by calibrate yet).\n",
        )
        return 1

    updated = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        image_ref = entry.get("image_ref")
        if image_ref not in calibrated:
            continue
        placement = entry.setdefault("placement", {})
        if not isinstance(placement, dict):
            continue
        new_size = int(calibrated[image_ref])
        placement["size_hint_bytes"] = new_size
        placement["size_hint_source"] = "cluster-reported"
        updated += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        _yaml.safe_dump(raw_yaml, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    out.write(f"wrote {output_path} ({updated} entries updated)\n")
    if unmeasured:
        out.write(
            f"{len(unmeasured)} entries kept their operator-supplied "
            "sizes (no node has materialized them yet):\n",
        )
        for ref in unmeasured[:10]:
            out.write(f"  - {ref}\n")

    # Plan-id-change warning. Only emit when at least one entry
    # actually changed (size_hint_bytes or size_hint_source flipped);
    # if calibrate is a no-op, the plan_id is identical and there's
    # nothing to flag.
    old_plan_id: str | None = None
    new_plan_id: str | None = None
    if updated > 0:
        try:
            from xrlenv.control.build_plan import (
                compute_plan_id,
                load_build_plan,
            )

            old_plan_id = compute_plan_id(load_build_plan(plan_path))
            new_plan_id = compute_plan_id(load_build_plan(output_path))
        except Exception:
            # Plan-id computation is best-effort; never fail the
            # calibrate over it.
            old_plan_id = new_plan_id = None
        if old_plan_id and new_plan_id and old_plan_id != new_plan_id:
            out.write(
                f"\nnote: the calibrated YAML has a fresh plan_id\n"
                f"  input:  {old_plan_id[:12]}\n"
                f"  output: {new_plan_id[:12]}\n"
                f"plan_id is content-hashed; size-hint changes shift it.\n"
                f"After you promote {output_path.name} into the canonical\n"
                f"YAML, the next ``xrlenv build apply`` creates a NEW\n"
                f"build_plans row (the old plan_id stays around as audit\n"
                f"history). The new dispatch re-runs every entry, even\n"
                f"ones already on the cluster — docker's layer cache\n"
                f"keeps the wall-clock low (~5-10s per already-built\n"
                f"image). See ``docs/technical_details/images/build_plan.md``\n"
                f"§ Important: calibrating changes the plan_id.\n",
            )
    return 0


def cmd_image_evict(
    *,
    image_ref: str,
    force: bool = False,
    out: TextIO,
    connect_host: str | None,
    connect_port: int = 8080,
    operator_token: str | None = None,
) -> int:
    """``xrlenv images evict`` — remove an image from every node's cache.

    Fans an eviction out to every connected node via the admin API so
    the next acquire re-pulls the current registry digest. The escape
    hatch for the mutable-tag staleness problem: after a rebuild +
    re-push under the *same* tag, a node never re-pulls on its own
    (``ensure_present`` short-circuits on local presence), so it keeps
    serving the old bytes until evicted.

    In-use / pinned images are skipped node-side unless ``--force``, so a
    live rollout is never disrupted. Exits non-zero only when a node
    *errored* (failed / unreachable) — an all-absent or all-in-use
    result is a successful, actionable no-op, not a command failure.
    """
    if connect_host is None:
        out.write(
            "error: --connect-host is required for ``xrlenv image "
            "evict`` (it fans out to the cluster via the admin API).\n",
        )
        return 2

    import httpx

    base = f"http://{connect_host}:{connect_port}"
    token = _resolve_operator_token(operator_token)
    headers: dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    body = {"image_ref": image_ref, "force": bool(force)}
    try:
        with httpx.Client(base_url=base, headers=headers, timeout=120.0) as client:
            r = client.post("/api/image/evict", json=body)
    except httpx.HTTPError as exc:
        out.write(
            f"error: cannot reach admin at {base}: {exc}\n"
            "       Confirm xrlenv up is running with --admin-port "
            f"{connect_port} and reachable.\n",
        )
        return 2

    if r.status_code == 401:
        out.write(
            "error: 401 unauthorized — operator token required.\n"
            "       Pass --operator-token, set $XRLENV_OPERATOR_TOKEN, "
            f"or place the token at {DEFAULT_XRLENV_HOME / 'secrets' / 'operator.token'}.\n",
        )
        return 2
    if r.status_code == 503:
        out.write("error: 503 — admin server not wired for cluster reachability.\n")
        return 2
    if r.status_code != 200:
        out.write(f"error: admin returned {r.status_code}: {r.text}\n")
        return 1

    payload = r.json()
    results = list(payload.get("results") or [])
    nodes_queried = int(payload.get("nodes_queried") or 0)
    nodes_evicted = int(payload.get("nodes_evicted") or 0)
    total_reclaimed = int(payload.get("total_reclaimed_bytes") or 0)

    def _gib(n: int) -> str:
        return f"{n / (1024 ** 3):.2f} GiB"

    out.write(
        f"evict {image_ref}{' --force' if force else ''}: "
        f"{nodes_evicted} evicted / {nodes_queried} node(s) queried, "
        f"{_gib(total_reclaimed)} reclaimed\n",
    )
    errors = 0
    for res in results:
        node_id = res.get("node_id")
        status = res.get("status")
        if status == "evicted":
            removed = res.get("removed") or []
            tail = f", {', '.join(removed)}" if removed else ""
            out.write(
                f"  - {node_id}: evicted "
                f"({_gib(int(res.get('reclaimed_bytes') or 0))}{tail})\n",
            )
        elif status == "absent":
            out.write(f"  - {node_id}: absent (nothing to evict)\n")
        elif status == "in_use":
            out.write(
                f"  - {node_id}: in use — skipped "
                "(re-run with --force to evict anyway)\n",
            )
        else:  # failed / unreachable
            errors += 1
            out.write(f"  - {node_id}: {status}: {res.get('error', '')}\n")
    return 1 if errors else 0


# ──────────────────────────────────────────────────────────────────────────────
# fairshare — live multi-user fair-share policy (read + tune)
# ──────────────────────────────────────────────────────────────────────────────


def _print_fairshare(policy: Any, counts: dict[str, int], out: TextIO) -> None:
    """Render a fair-share policy + current per-owner usage."""
    if not policy.enabled:
        out.write(
            "fair-share: DISABLED (no default cap set)\n"
            "  All owners run uncapped. Enable with "
            "`xrlenv fairshare set --default-cap <N>`.\n"
        )
    else:
        out.write(
            f"fair-share: ENABLED  default_cap={policy.capacity_basis}\n"
        )
    active = set(counts) | set(policy.overrides)
    if active:
        out.write("per-owner:\n")
        for owner in sorted(active):
            running = counts.get(owner, 0)
            ov = policy.overrides.get(owner)
            bits = [f"running={running}"]
            if policy.enabled:
                cap = policy.cap_for(owner, active)
                bits.append(
                    "effective_cap=uncapped"
                    if cap is None else f"effective_cap={cap}"
                )
            if ov is not None:
                if ov.hard_cap is not None:
                    bits.append(f"owner_cap={ov.hard_cap}")
                if ov.uncapped:
                    bits.append("UNCAPPED")
                if ov.blocked:
                    bits.append("BLOCKED")
            out.write(f"  {owner:<20} {'  '.join(bits)}\n")
    else:
        out.write("per-owner: (no active owners, no overrides)\n")
    if policy.enabled:
        out.write(
            "  (`--default-cap` applies to owners without an override; "
            "`--owner ... --cap` overrides one owner; `--uncap` bypasses "
            "fair-share for one owner. Real cluster resources are still "
            "enforced by the scheduler.)\n"
        )


def cmd_fairshare_show(
    *,
    state_db: Path = DEFAULT_STATE_DB,
    out: TextIO,
) -> int:
    """``xrlenv fairshare show`` — print the live fair-share policy + usage."""
    state = _open_state(state_db)
    try:
        policy = state.get_fairness_policy()
        counts = state.running_counts_by_owner()
    finally:
        state.close()
    _print_fairshare(policy, counts, out)
    return 0


def cmd_fairshare_set(
    *,
    default_cap: int | None = None,
    disable: bool = False,
    owner: str | None = None,
    cap: int | None = None,
    uncap: bool = False,
    recap: bool = False,
    block: bool = False,
    unblock: bool = False,
    clear_owner: str | None = None,
    state_db: Path = DEFAULT_STATE_DB,
    out: TextIO,
) -> int:
    """``xrlenv fairshare set`` — tune the live fair-share policy.

    Writes to the same ``state.db`` the control plane reads each drain pass,
    so changes take effect within a few seconds **without** a restart and
    **without** killing running jobs — lowering a cap or pausing only stops
    *new* admissions (use ``xrlenv ...`` cancel for a hard reclaim).
    """
    if block and unblock:
        out.write("error: pass at most one of --block / --unblock\n")
        return 2
    if uncap and recap:
        out.write("error: pass at most one of --uncap / --recap\n")
        return 2
    if disable and default_cap is not None:
        out.write("error: pass at most one of --default-cap / --disable\n")
        return 2
    owner_flags = cap is not None or uncap or recap or block or unblock
    if owner is None and owner_flags:
        out.write("error: --owner is required with --cap/--uncap/--recap/--block/--unblock\n")
        return 2
    if cap is not None and (uncap or recap):
        out.write("error: pass at most one of --cap / --uncap / --recap\n")
        return 2
    if uncap and block:
        out.write("error: pass at most one of --uncap / --block\n")
        return 2
    # Range validation (audit M6): reject nonsensical numbers up front. A
    # negative owner cap is the worst — cap_for returns owner caps verbatim and
    # the gate treats even zero running sandboxes as at-cap (0 >= -1), silently
    # creating an always-blocked owner instead of the explicit --block control.
    if default_cap is not None and default_cap < 1:
        out.write("error: --default-cap must be >= 1 (use --disable to turn fairness off)\n")
        return 2
    if cap is not None and cap < 1:
        out.write("error: --cap must be >= 1 (use --block to stop all new admissions)\n")
        return 2
    state = _open_state(state_db, read_only=False)  # WRITE: sets fairness policy
    try:
        current = state.get_fairness_policy()
        new_basis = current.capacity_basis
        new_floor = current.floor
        touched_global = False
        if disable:
            new_basis = None
            touched_global = True
        elif default_cap is not None:
            new_basis = default_cap
            touched_global = True
        if touched_global:
            state.set_fairness_global(capacity_basis=new_basis, floor=new_floor)

        if clear_owner is not None:
            state.clear_fairness_owner(clear_owner)

        if owner is not None:
            ov = current.overrides.get(owner)
            new_weight = ov.weight if ov is not None else 1.0
            new_hard_cap = ov.hard_cap if ov is not None else None
            new_uncapped = ov.uncapped if ov is not None else False
            new_blocked = ov.blocked if ov is not None else False
            if recap:
                new_hard_cap = None
                new_uncapped = False
                new_blocked = False
            elif uncap:
                new_hard_cap = None
                new_uncapped = True
                new_blocked = False
            elif cap is not None:
                new_hard_cap = cap
                new_uncapped = False
            if block:
                new_blocked = True
            elif unblock:
                new_blocked = False
            state.set_fairness_owner(
                owner, weight=new_weight, hard_cap=new_hard_cap,
                uncapped=new_uncapped, blocked=new_blocked,
            )

        policy = state.get_fairness_policy()
        counts = state.running_counts_by_owner()
    finally:
        state.close()
    _print_fairshare(policy, counts, out)
    out.write(
        "\nApplied. The control plane picks this up on its next admission "
        "drain (seconds) — no restart, running jobs untouched.\n"
    )
    return 0


def cmd_db_prune(
    *,
    state_db: Path = DEFAULT_STATE_DB,
    audit_retention_days: int | None = 30,
    events_retention_days: int | None = 14,
    raw_rollout_retention_days: int | None = 14,
    out: TextIO,
) -> int:
    """Hard-delete ``state.db`` rows past their retention window (spec 20 matrix).

    The control plane runs this automatically every 24 h; this command triggers
    it on demand — e.g. to reclaim immediately before ``xrlenv db vacuum``. Safe
    while the control plane runs (deletes are batched; WAL keeps readers
    unblocked), though running it while quiescent avoids contending with the
    live writer.
    """
    state = _open_state(state_db, read_only=False)  # WRITE: deletes expired rows
    try:
        counts = state.prune_expired(
            now=time.time(),
            audit_retention_days=audit_retention_days,
            events_retention_days=events_retention_days,
            raw_rollout_retention_days=raw_rollout_retention_days,
        )
    finally:
        state.close()
    total = sum(counts.values())
    out.write(
        f"pruned {total} expired row(s): audit={counts['audit']} "
        f"events={counts['events']} raw_rollouts={counts['raw_rollouts']}\n"
    )
    if total:
        out.write(
            "note: DELETE frees pages for reuse but does not shrink the file; "
            "run `xrlenv db vacuum` (control plane stopped) to reclaim disk.\n"
        )
    return 0


def cmd_db_vacuum(*, state_db: Path = DEFAULT_STATE_DB, out: TextIO) -> int:
    """``VACUUM`` ``state.db`` to return freed pages to the filesystem.

    Run with the control plane STOPPED — ``VACUUM`` needs exclusive access and
    fails with "database is locked" if ``xrlenv up`` holds the database.
    Checkpoints the WAL first, then rewrites the file.
    """
    import sqlite3

    if not state_db.exists():
        raise FileNotFoundError(f"state.db not found at {state_db}")
    before = state_db.stat().st_size
    conn = sqlite3.connect(str(state_db))
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.execute("VACUUM")
        conn.commit()
    except sqlite3.OperationalError as exc:
        out.write(
            f"VACUUM failed: {exc}. Is the control plane running? Stop "
            "`xrlenv up` first — VACUUM needs exclusive access to the database.\n"
        )
        return 1
    finally:
        conn.close()
    after = state_db.stat().st_size
    out.write(
        f"VACUUM complete: {before / 1e6:.1f} MB -> {after / 1e6:.1f} MB "
        f"(reclaimed {(before - after) / 1e6:.1f} MB)\n"
    )
    return 0


__all__ = [
    "DEFAULT_NODES_YAML",
    "DEFAULT_RUNS_ROOT",
    "DEFAULT_STATE_DB",
    "cmd_attach",
    "cmd_audit",
    "cmd_build_apply",
    "cmd_build_calibrate",
    "cmd_build_cancel",
    "cmd_build_status",
    "cmd_db_prune",
    "cmd_db_vacuum",
    "cmd_events",
    "cmd_fairshare_set",
    "cmd_fairshare_show",
    "cmd_images",
    "cmd_nodes",
    "cmd_replay",
    "cmd_rollouts",
    "cmd_stub_runtime_layer",
    "cmd_tail",
    "cmd_tokens_issue",
    "cmd_tokens_list",
    "cmd_tokens_revoke",
    "cmd_tokens_rotate",
    "cmd_warmup",
    "parse_duration",
    "stub_runtime_dockerfile_path",
]
