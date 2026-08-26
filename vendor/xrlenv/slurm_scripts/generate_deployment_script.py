#!/usr/bin/env python3
"""Generate and synchronize per-cluster Slurm scripts from clusters.yaml.

``clusters.yaml`` is the single source of truth for cluster topology; the
templates under ``templates/`` are the single source of script *prose*. Every
committed ``generated/deploy_<name>.sh`` /
``generated/<name>_xrlenv_{node,control}.sh`` is a pure function of those two
inputs, so adding a cluster is one YAML block and adding a worker is one YAML
line — no new files, no edits to this script. The outputs live under
``slurm_scripts/generated/`` (see ``GENERATED_DIRNAME``) so the source files
stay unmixed with the artifacts they produce.

Two rendering paths, deliberately kept separate:

* ``rebuild_all`` renders a whole script from ``templates/``. This is how new
  clusters are scaffolded and how ``--check`` proves the committed scripts still
  match template + config (the golden gate).
* ``render_all`` patches only the topology-bearing lines of scripts that already
  exist, preserving any local edit. This is the incremental path used when a
  cluster's hosts move.
"""

from __future__ import annotations

import argparse
import difflib
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "clusters.yaml"
# Generated artifacts — deploy_<n>.sh, <n>_xrlenv_{node,control}.sh, and the
# runtime-written <n>_hyperpod_nodes.yaml roster — live in this subdirectory of
# slurm_scripts/, NOT flat alongside the source. Quarantining every generated
# file under one directory keeps `ls slurm_scripts/` showing only inputs
# (clusters.yaml + templates/ + this script + the hand-maintained *.sh helpers).
# Always spelled ``SCRIPT_DIR / GENERATED_DIRNAME`` at CALL time rather than
# frozen into a Path constant, so the unit tests that monkeypatch SCRIPT_DIR
# still redirect the output.
GENERATED_DIRNAME = "generated"
# Legacy hint only: the authoritative cluster list is the set of top-level keys
# in clusters.yaml (minus ``defaults``). Adding a cluster must not require an
# edit here — that was the whole point of the template refactor.
CLUSTERS = ("dev", "prod", "cn")
# Top-level key holding values inherited by every cluster that does not override
# them. Not a cluster; skipped by cluster discovery.
DEFAULTS_KEY = "defaults"
REQUIRED_FIELDS = {
    "control_plane",
    "registry",
    "workers",
    "sysbox_pool",
    "cpu_isolation_pool",
}
# Optional per-cluster fields. Every one has a derived default (see
# ``_parse_cluster``) so a new cluster only spells out what it does differently.
# They are OPTIONAL rather than required so that a minimal cluster block stays
# minimal — and so the schema stayed backward-compatible when it grew.
OPTIONAL_FIELDS = {
    "checkout",
    "local_disk_root",
    "node_env",
    "state_db",
    "control_log",
    "tunnel",
    "deploy_registry",
    "partition",
    "account",
    "grpc_port",
    "sysbox_max_concurrent",
    "allowed_host_paths",
}
# Mount point of the box's DEDICATED local data volume, or None when the cluster
# has none. One fact, two consumers, which is why it is a single knob:
#   * Docker's data-root on workers — relocated there by ``bootstrap-aws.sh
#     --hyperpod`` so image layers do not fill the small root disk. With no such
#     volume the flag must be omitted entirely: set_docker_data_root.sh refuses a
#     target that is not a real block device, and refuses one that resolves to
#     the root device, so there is nothing valid to point it at.
#   * ``state_db`` on the control plane — must be local (a WAL on Lustre faults
#     with SIGBUS), so it defaults under this root when there is one.
DEFAULT_LOCAL_DISK_ROOT = "/opt/sagemaker"
# Where state.db goes when the cluster has NO dedicated volume: still local disk
# (that is the hard requirement), just the root filesystem.
NO_VOLUME_STATE_DIR = "/var/lib"
ALLOWED_FIELDS = REQUIRED_FIELDS | OPTIONAL_FIELDS
TUNNEL_REQUIRED = {"admin", "metrics"}
# Per-node env stamped into /etc/xrlenv/node.env at bootstrap. This exists
# because ``sudo`` in the node script passes an explicit ALLOWLIST — anything not
# named on that command line is stripped, so setting a knob in ``.env`` alone
# never reaches the bootstrap. Rendering the assignments into the sudo line is
# the only path that works.
#
# The knobs are read by ``xrlenv/cli/bootstrap.py`` (written into node.env) and
# then by ``xrlenv/node/cli.py`` (into ImageCacheConfig etc.). Names are
# restricted to XRLENV_* so this cannot be used to inject arbitrary environment
# into a root sudo invocation, and values must be shell-safe for the same reason.
NODE_ENV_NAME_RE = re.compile(r"^XRLENV_[A-Z0-9_]+$")
NODE_ENV_VALUE_RE = re.compile(r"^[A-Za-z0-9_.:/=+-]*$")
DEFAULT_PARTITION = "ml.m7i.48xlarge"
DEFAULT_ACCOUNT = "<your-slurm-account>"
DEFAULT_GRPC_PORT = 50051
DEFAULT_SYSBOX_MAX_CONCURRENT = 4
# Empty by default, deliberately. allowed_host_paths is a SECURITY allowlist
# (policy.allowed_host_paths): it authorizes real read-only host mounts
# into sandboxes. It is also cluster-specific — the shared filesystem is laid out
# differently per cluster, so a path that is valid on one is meaningless on
# another. Inheriting a mount permission a cluster never asked for is the wrong
# default in both directions, so each cluster declares its own.
DEFAULT_ALLOWED_HOST_PATHS: tuple[str, ...] = ()
DEFAULT_TUNNEL_ADMIN = 8080
DEFAULT_TUNNEL_METRICS = 9090
# template basename -> committed filename pattern (``{n}`` = cluster name)
TEMPLATES = {
    "deploy.sh.j2": "deploy_{n}.sh",
    "xrlenv_node.sh.j2": "{n}_xrlenv_node.sh",
    "xrlenv_control.sh.j2": "{n}_xrlenv_control.sh",
}
# The node/control scripts are committed WITHOUT a trailing newline; the deploy
# script WITH one. Templates are authored to match byte-for-byte, so Jinja must
# not add or strip one (keep_trailing_newline=True).

REGISTRY_REQUIRED = {"mirror_host", "private_host"}
REGISTRY_OPTIONAL = {"mirror_port", "private_port", "scratch_host", "scratch_port"}
REGISTRY_ALLOWED = REGISTRY_REQUIRED | REGISTRY_OPTIONAL
DEFAULT_MIRROR_PORT = 5010
DEFAULT_PRIVATE_PORT = 5011
DEFAULT_SCRATCH_PORT = 5012
HOST_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ConfigError(ValueError):
    """Raised when cluster topology cannot safely generate shell scripts."""


@dataclass(frozen=True)
class RegistryTopology:
    """Per-cluster image-registry endpoints. All three hosts default to the same
    shared box today to save resources, but the schema lets a cluster run its
    own registries — on their own ports. Ports are optional in clusters.yaml and
    default to the project conventions (:5010 mirror, :5011 private, :5012
    scratch). ``scratch_host`` is optional and defaults to ``private_host`` — the
    scratch (build-on-demand) registry is a third registry that usually shares
    the private/mirror box."""

    mirror_host: str
    private_host: str
    scratch_host: str
    mirror_port: int = DEFAULT_MIRROR_PORT
    private_port: int = DEFAULT_PRIVATE_PORT
    scratch_port: int = DEFAULT_SCRATCH_PORT


@dataclass(frozen=True)
class Cluster:
    """One cluster's full deployment identity.

    The first six fields are the topology and keep their historical positional
    order. Everything after them is a per-cluster *runtime* resource that must
    be disjoint between clusters (state DB, control log, login-node tunnel
    ports) or a Slurm/knob default; each carries a derived default so a minimal
    cluster block stays minimal.
    """

    name: str
    control_plane: str
    registry: RegistryTopology
    workers: tuple[str, ...]
    sysbox_pool: tuple[str, ...]
    cpu_isolation_pool: tuple[str, ...]
    checkout: str = ""
    # "" means "not spelled out, use the default"; None means "this cluster has
    # NO dedicated local volume" — a meaningful value, not a missing one.
    local_disk_root: str | None = DEFAULT_LOCAL_DISK_ROOT
    state_db: str = ""
    control_log: str = ""
    tunnel_admin: int = DEFAULT_TUNNEL_ADMIN
    tunnel_metrics: int = DEFAULT_TUNNEL_METRICS
    deploy_registry: bool = True
    partition: str = DEFAULT_PARTITION
    account: str = DEFAULT_ACCOUNT
    grpc_port: int = DEFAULT_GRPC_PORT
    sysbox_max_concurrent: int = DEFAULT_SYSBOX_MAX_CONCURRENT
    allowed_host_paths: tuple[str, ...] = DEFAULT_ALLOWED_HOST_PATHS
    # Ordered so rendering is deterministic (a dict would still render stably in
    # 3.7+, but a tuple makes the frozen dataclass hashable and the order explicit).
    node_env: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        # Derived defaults live here (not in _parse_cluster) so a Cluster built
        # directly — as the unit tests do — is still fully renderable.
        if not self.checkout:
            object.__setattr__(self, "checkout", str(SCRIPT_DIR.parent))
        if not self.state_db:
            base = (
                self.local_disk_root
                if self.local_disk_root
                else NO_VOLUME_STATE_DIR
            )
            object.__setattr__(
                self, "state_db", f"{base}/xrlenv-{self.name}/state.db"
            )
        if not self.control_log:
            # LOCAL disk, co-located with state.db — NOT Lustre. The CP writes
            # --log-file with a synchronous handler on the event-loop thread, so
            # a Lustre stall on it freezes the control plane and false-marks the
            # fleet lost (2026-08-21). Mirrors the state_db "must be local"
            # requirement above.
            base = (
                self.local_disk_root
                if self.local_disk_root
                else NO_VOLUME_STATE_DIR
            )
            object.__setattr__(
                self,
                "control_log",
                f"{base}/xrlenv-{self.name}/logs/xrlenv-up-control.log",
            )


def _host_list(value: Any, *, cluster: str, field: str, allow_empty: bool) -> tuple[str, ...]:
    if not isinstance(value, list) or (not allow_empty and not value):
        suffix = " (may be empty)" if allow_empty else ""
        raise ConfigError(f"{cluster}.{field} must be a non-empty YAML list{suffix}")
    if not all(isinstance(host, str) and HOST_RE.fullmatch(host) for host in value):
        raise ConfigError(
            f"{cluster}.{field} entries must be hostnames containing only letters, "
            "digits, dots, underscores, or hyphens"
        )
    hosts = tuple(value)
    if len(hosts) != len(set(hosts)):
        raise ConfigError(f"{cluster}.{field} contains duplicate hosts")
    return hosts


def _parse_port(value: Any, *, cluster: str, key: str) -> int:
    # ``bool`` is an ``int`` subclass — reject it so ``mirror_port: true`` is an
    # error, not port 1.
    if isinstance(value, bool) or not isinstance(value, int) or not (1 <= value <= 65535):
        raise ConfigError(
            f"{cluster}.registry.{key} must be an integer port in 1..65535"
        )
    return int(value)


def _parse_registry(value: Any, *, cluster: str) -> RegistryTopology:
    if not isinstance(value, dict):
        raise ConfigError(f"{cluster}.registry must be a YAML mapping")
    fields = set(value)
    missing = sorted(REGISTRY_REQUIRED - fields)
    extra = sorted(fields - REGISTRY_ALLOWED)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if extra:
            details.append(f"unknown: {', '.join(extra)}")
        raise ConfigError(f"{cluster}.registry has invalid fields ({'; '.join(details)})")
    for key in ("mirror_host", "private_host"):
        host = value[key]
        if not isinstance(host, str) or not HOST_RE.fullmatch(host):
            raise ConfigError(f"{cluster}.registry.{key} must be a valid hostname")
    # scratch_host is optional and defaults to the private host — the scratch
    # (build-on-demand) registry is a third registry that normally shares the
    # private/mirror box, so a cluster running one registry pair gets a scratch
    # endpoint on the same host for free.
    scratch_host = value.get("scratch_host", value["private_host"])
    if not isinstance(scratch_host, str) or not HOST_RE.fullmatch(scratch_host):
        raise ConfigError(f"{cluster}.registry.scratch_host must be a valid hostname")
    return RegistryTopology(
        mirror_host=value["mirror_host"],
        private_host=value["private_host"],
        scratch_host=scratch_host,
        mirror_port=_parse_port(
            value.get("mirror_port", DEFAULT_MIRROR_PORT),
            cluster=cluster, key="mirror_port",
        ),
        private_port=_parse_port(
            value.get("private_port", DEFAULT_PRIVATE_PORT),
            cluster=cluster, key="private_port",
        ),
        scratch_port=_parse_port(
            value.get("scratch_port", DEFAULT_SCRATCH_PORT),
            cluster=cluster, key="scratch_port",
        ),
    )


def _parse_tunnel(value: Any, *, cluster: str) -> tuple[int, int]:
    """Login-node port forwards. These are the one resource every cluster shares
    (the login node), so they MUST differ between clusters — cross-checked in
    ``_check_disjoint``."""
    if not isinstance(value, dict):
        raise ConfigError(f"{cluster}.tunnel must be a YAML mapping")
    fields = set(value)
    missing = sorted(TUNNEL_REQUIRED - fields)
    extra = sorted(fields - TUNNEL_REQUIRED)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if extra:
            details.append(f"unknown: {', '.join(extra)}")
        raise ConfigError(f"{cluster}.tunnel has invalid fields ({'; '.join(details)})")
    return (
        _parse_port(value["admin"], cluster=cluster, key="tunnel.admin"),
        _parse_port(value["metrics"], cluster=cluster, key="tunnel.metrics"),
    )


def _parse_node_env(value: Any, *, cluster: str) -> tuple[tuple[str, str], ...]:
    """Per-node ``XRLENV_*`` env, rendered into the node script's sudo line.

    Names are constrained to ``XRLENV_*`` and values to a shell-safe charset:
    these are interpolated into a ``sudo`` command that runs the bootstrap as
    root, so an unconstrained name or a value containing quotes/spaces/``$``
    would be a command-injection vector, not merely a syntax error.
    """
    if not isinstance(value, dict):
        raise ConfigError(f"{cluster}.node_env must be a YAML mapping")
    out: list[tuple[str, str]] = []
    for key in sorted(value):
        if not isinstance(key, str) or not NODE_ENV_NAME_RE.fullmatch(key):
            raise ConfigError(
                f"{cluster}.node_env key {key!r} must match XRLENV_[A-Z0-9_]+ "
                "(only xrlenv's own knobs may be stamped into node.env)"
            )
        raw = value[key]
        if isinstance(raw, bool) or raw is None:
            raise ConfigError(
                f"{cluster}.node_env[{key}] must be a string or number, not {raw!r}"
            )
        text = str(raw)
        if not NODE_ENV_VALUE_RE.fullmatch(text):
            raise ConfigError(
                f"{cluster}.node_env[{key}]={text!r} contains characters that are "
                "unsafe to interpolate into the bootstrap's sudo command line"
            )
        out.append((key, text))
    return tuple(out)


def _parse_cluster(
    name: str, value: Any, *, defaults: dict[str, Any] | None = None
) -> Cluster:
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a YAML mapping")
    # ``defaults:`` supplies inherited values; an explicit per-cluster key always
    # wins. Merged before validation so an unknown key in defaults is caught too.
    if defaults:
        value = {**defaults, **value}
    fields = set(value)
    if not (fields >= REQUIRED_FIELDS and fields <= ALLOWED_FIELDS):
        missing = sorted(REQUIRED_FIELDS - fields)
        extra = sorted(fields - ALLOWED_FIELDS)
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if extra:
            details.append(f"unknown: {', '.join(extra)}")
        raise ConfigError(f"{name} has invalid fields ({'; '.join(details)})")

    control_plane = value["control_plane"]
    if not isinstance(control_plane, str) or not HOST_RE.fullmatch(control_plane):
        raise ConfigError(f"{name}.control_plane must be a valid hostname")

    registry = _parse_registry(value["registry"], cluster=name)

    workers = _host_list(value["workers"], cluster=name, field="workers", allow_empty=False)
    sysbox_pool = _host_list(
        value["sysbox_pool"], cluster=name, field="sysbox_pool", allow_empty=True
    )
    cpu_isolation_pool = _host_list(
        value["cpu_isolation_pool"],
        cluster=name,
        field="cpu_isolation_pool",
        allow_empty=True,
    )

    worker_set = set(workers)
    for field, pool in (
        ("sysbox_pool", sysbox_pool),
        ("cpu_isolation_pool", cpu_isolation_pool),
    ):
        outside = sorted(set(pool) - worker_set)
        if outside:
            raise ConfigError(f"{name}.{field} contains non-workers: {', '.join(outside)}")
    overlap = sorted(set(sysbox_pool) & set(cpu_isolation_pool))
    if overlap:
        raise ConfigError(
            f"{name}.sysbox_pool and {name}.cpu_isolation_pool overlap: "
            f"{', '.join(overlap)}"
        )
    if control_plane in worker_set:
        raise ConfigError(f"{name}.control_plane must not also be a worker")

    tunnel_admin, tunnel_metrics = (
        _parse_tunnel(value["tunnel"], cluster=name)
        if "tunnel" in value
        else (DEFAULT_TUNNEL_ADMIN, DEFAULT_TUNNEL_METRICS)
    )
    deploy_registry = value.get("deploy_registry", True)
    if not isinstance(deploy_registry, bool):
        raise ConfigError(f"{name}.deploy_registry must be true or false")
    host_paths = value.get("allowed_host_paths", list(DEFAULT_ALLOWED_HOST_PATHS))
    if not isinstance(host_paths, list) or not all(
        isinstance(p, str) and p and " " not in p for p in host_paths
    ):
        raise ConfigError(
            f"{name}.allowed_host_paths must be a list of space-free paths"
        )
    for key in ("checkout", "state_db", "control_log", "partition", "account"):
        if key in value and (not isinstance(value[key], str) or not value[key]):
            raise ConfigError(f"{name}.{key} must be a non-empty string")
    # local_disk_root is tri-state: absent -> the conventional /opt/sagemaker;
    # an explicit path -> that mount; explicit null -> the cluster HAS no
    # dedicated volume. ``null`` must stay distinguishable from ``absent``, so it
    # is read with a sentinel rather than ``.get(key)``.
    _MISSING = object()
    local_disk_root: Any = value.get("local_disk_root", _MISSING)
    if local_disk_root is _MISSING:
        local_disk_root = DEFAULT_LOCAL_DISK_ROOT
    elif local_disk_root is not None and (
        not isinstance(local_disk_root, str)
        or not local_disk_root.startswith("/")
    ):
        raise ConfigError(
            f"{name}.local_disk_root must be an absolute path, or null when the "
            "cluster's boxes have no dedicated local data volume"
        )
    max_conc = value.get("sysbox_max_concurrent", DEFAULT_SYSBOX_MAX_CONCURRENT)
    if isinstance(max_conc, bool) or not isinstance(max_conc, int) or max_conc < 0:
        raise ConfigError(f"{name}.sysbox_max_concurrent must be a non-negative int")

    return Cluster(
        name,
        control_plane,
        registry,
        workers,
        sysbox_pool,
        cpu_isolation_pool,
        checkout=value.get("checkout", ""),
        local_disk_root=local_disk_root,
        state_db=value.get("state_db", ""),
        control_log=value.get("control_log", ""),
        tunnel_admin=tunnel_admin,
        tunnel_metrics=tunnel_metrics,
        deploy_registry=deploy_registry,
        partition=value.get("partition", DEFAULT_PARTITION),
        account=value.get("account", DEFAULT_ACCOUNT),
        grpc_port=_parse_port(
            value.get("grpc_port", DEFAULT_GRPC_PORT), cluster=name, key="grpc_port"
        ),
        sysbox_max_concurrent=max_conc,
        allowed_host_paths=tuple(host_paths),
        node_env=_parse_node_env(value["node_env"], cluster=name)
        if "node_env" in value
        else (),
    )


def _load_raw(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"configuration file does not exist: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("configuration root must be a YAML mapping")
    return raw


# Fields naming durable on-disk state. A cluster that omits one still works —
# the derivation fills it in — but it would silently RELOCATE that state if the
# derivation ever changed, so clusters.yaml declares them explicitly and this
# warns when one does not. Advisory, never fatal.
_EXPLICIT_BY_CONVENTION = ("state_db", "control_log", "checkout")


def warn_implicit_state_paths(path: Path, selected: tuple[str, ...]) -> list[str]:
    """Names of (cluster, field) pairs relying on a derived value."""
    raw = _load_raw(path)
    defaults = raw.get(DEFAULTS_KEY) or {}
    return [
        f"{name}.{f}"
        for name in selected
        for f in _EXPLICIT_BY_CONVENTION
        if f not in (raw.get(name) or {}) and f not in defaults
    ]


def cluster_names(path: Path) -> tuple[str, ...]:
    """Every cluster declared in clusters.yaml, in file order.

    This — not a constant in this file — is the authoritative cluster list, so
    declaring a new cluster requires no code change.
    """
    return tuple(k for k in _load_raw(path) if k != DEFAULTS_KEY)


def load_config(path: Path, selected: tuple[str, ...]) -> tuple[Cluster, ...]:
    raw = _load_raw(path)
    defaults = raw.get(DEFAULTS_KEY) or {}
    if not isinstance(defaults, dict):
        raise ConfigError(f"{DEFAULTS_KEY} must be a YAML mapping")
    if unknown := sorted(set(defaults) - ALLOWED_FIELDS):
        raise ConfigError(f"{DEFAULTS_KEY} has unknown fields: {', '.join(unknown)}")

    missing = [name for name in selected if name not in raw]
    if missing:
        raise ConfigError(f"configuration is missing clusters: {', '.join(missing)}")
    return tuple(
        _parse_cluster(name, raw[name], defaults=defaults) for name in selected
    )


def _replace_once(text: str, pattern: str, replacement: str, *, path: Path) -> str:
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise ConfigError(f"{path}: expected exactly one match for {pattern!r}, found {count}")
    return updated


def render_deploy(text: str, cluster: Cluster, path: Path) -> str:
    text = _replace_once(
        text,
        r'^CP_NODE="[^"]*"(.*)$',
        rf'CP_NODE="{cluster.control_plane}"\1',
        path=path,
    )
    text = _replace_once(
        text,
        r"^SYSBOX_POOL=\([^)]*\)$",
        f"SYSBOX_POOL=({' '.join(cluster.sysbox_pool)})",
        path=path,
    )
    return _replace_once(
        text,
        r"^CPU_ISOLATION_POOL=\([^)]*\)$",
        f"CPU_ISOLATION_POOL=({' '.join(cluster.cpu_isolation_pool)})",
        path=path,
    )


def render_node(text: str, cluster: Cluster, path: Path) -> str:
    text = _replace_once(
        text,
        r"^(#SBATCH --nodes=)\S+(.*)$",
        rf"\g<1>{len(cluster.workers)}\2",
        path=path,
    )
    # Only the SBATCH allocation lines are patchable here. The runtime topology
    # (control plane + registry hosts/ports) is no longer baked into the node
    # script as ``:=`` defaults — it is required from .env (require_env in the
    # template), synced from clusters.yaml via --env-cluster — so there is no
    # ``XRLENV_GRPC_HOST:=`` line for --patch-only to rewrite.
    return _replace_once(
        text,
        r"^#SBATCH --nodelist=.*$",
        f"#SBATCH --nodelist={','.join(cluster.workers)}",
        path=path,
    )


def render_control(text: str, cluster: Cluster, path: Path) -> str:
    text = _replace_once(
        text,
        r"^#SBATCH --nodelist=.*$",
        f"#SBATCH --nodelist={cluster.control_plane}",
        path=path,
    )
    text = _replace_once(
        text,
        r"^(\s*--grpc-host )\S+(\s+--grpc-port .*)$",
        rf"\g<1>{cluster.control_plane}\2",
        path=path,
    )
    return _replace_once(
        text,
        r"^(# scancel .* && ssh .*)\s+\S+$",
        rf"\g<1> {cluster.control_plane}",
        path=path,
    )


# Topology keys the generator keeps in sync in a checkout's ``.env`` so the
# CONSUMER path (``xrlenv.from_env`` / the benchmark oracle sweeps, which read
# ``.env`` to dial the control plane + registry) tracks clusters.yaml. These are
# the lines that change when a cluster's hosts/ports move; every secret line
# (``*_TOKEN``, ``DOCKERHUB_*``, ``*_HTTP_SECRET``) and ``XRLENV_GRPC_PORT`` are
# left untouched. clusters.yaml is the single source of truth; ``.env`` just
# carries the derived values alongside the hand-maintained secrets.
_ENV_TOPOLOGY_KEYS = (
    "XRLENV_GRPC_HOST",
    "XRLENV_MIRROR_REGISTRY_HOST",
    "XRLENV_MIRROR_REGISTRY_PORT",
    "XRLENV_PRIVATE_REGISTRY_HOST",
    "XRLENV_PRIVATE_REGISTRY_PORT",
    "XRLENV_SCRATCH_REGISTRY_HOST",
    "XRLENV_SCRATCH_REGISTRY_PORT",
)


def _render_env_var(text: str, key: str, value: str, *, path: Path) -> str:
    """Replace the ``KEY=...`` line in ``.env`` text, or append it if absent.
    Every other line (secrets, comments, untouched keys) is preserved
    byte-for-byte."""
    # ``value`` is a hostname (HOST_RE) or a port (digits) — no backslashes, so a
    # plain string repl carries no ``\g``/backref-interpretation risk.
    updated, count = re.subn(
        rf"^{re.escape(key)}=.*$",
        f"{key}={value}",
        text,
        flags=re.MULTILINE,
    )
    if count > 1:
        raise ConfigError(f"{path}: {key} appears {count} times; expected at most one")
    if count == 0:
        sep = "" if text == "" or text.endswith("\n") else "\n"
        return f"{text}{sep}{key}={value}\n"
    return updated


def _env_topology_values(cluster: Cluster) -> dict[str, str]:
    """The topology ``key -> value`` mapping clusters.yaml expects in a
    checkout's ``.env``. These are infrastructure addresses (hosts/ports), NOT
    secrets — unlike the ``*_TOKEN`` / ``DOCKERHUB_*`` lines render_env never
    touches — so they are safe to print in a drift report."""
    return {
        "XRLENV_GRPC_HOST": cluster.control_plane,
        "XRLENV_MIRROR_REGISTRY_HOST": cluster.registry.mirror_host,
        "XRLENV_MIRROR_REGISTRY_PORT": str(cluster.registry.mirror_port),
        "XRLENV_PRIVATE_REGISTRY_HOST": cluster.registry.private_host,
        "XRLENV_PRIVATE_REGISTRY_PORT": str(cluster.registry.private_port),
        "XRLENV_SCRATCH_REGISTRY_HOST": cluster.registry.scratch_host,
        "XRLENV_SCRATCH_REGISTRY_PORT": str(cluster.registry.scratch_port),
    }


def render_env(text: str, cluster: Cluster, path: Path) -> str:
    values = _env_topology_values(cluster)
    for key in _ENV_TOPOLOGY_KEYS:
        text = _render_env_var(text, key, values[key], path=path)
    return text


def env_topology_drift(
    text: str, cluster: Cluster,
) -> list[tuple[str, str | None, str]]:
    """Per topology key whose ``.env`` value differs from what clusters.yaml
    would render, return ``(key, current_value_or_None, wanted_value)``. Empty
    when in sync. Only the topology keys are compared — so a caller can NAME the
    exact drift (which key is missing / stale, and to what) without a raw diff
    that could echo an adjacent secret line. render_env only ever rewrites these
    keys, so this is the complete .env topology drift."""
    values = _env_topology_values(cluster)
    drift: list[tuple[str, str | None, str]] = []
    for key in _ENV_TOPOLOGY_KEYS:
        wanted = values[key]
        match = re.search(rf"^{re.escape(key)}=(.*)$", text, flags=re.MULTILINE)
        current = match.group(1) if match is not None else None
        if current != wanted:
            drift.append((key, current, wanted))
    return drift


def render_all(
    clusters: tuple[Cluster, ...],
    *,
    env_cluster: str | None = None,
    env_path: Path | None = None,
) -> dict[Path, str]:
    rendered: dict[Path, str] = {}
    generated = SCRIPT_DIR / GENERATED_DIRNAME
    for cluster in clusters:
        transformations = {
            generated / f"deploy_{cluster.name}.sh": render_deploy,
            generated / f"{cluster.name}_xrlenv_node.sh": render_node,
            generated / f"{cluster.name}_xrlenv_control.sh": render_control,
        }
        for path, transform in transformations.items():
            try:
                current = path.read_text(encoding="utf-8")
            except FileNotFoundError as exc:
                raise ConfigError(f"deployment script does not exist: {path}") from exc
            rendered[path] = transform(current, cluster, path)
    if env_cluster is not None:
        if env_path is None:
            raise ConfigError("env_path is required when env_cluster is set")
        target = next((c for c in clusters if c.name == env_cluster), None)
        if target is None:
            raise ConfigError(
                f"--env-cluster {env_cluster!r} is not among the selected clusters"
            )
        try:
            current = env_path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise ConfigError(f"env file does not exist: {env_path}") from exc
        rendered[env_path] = render_env(current, target, env_path)
    return rendered


def _template_env() -> Environment:
    """Jinja configured for shell templates.

    Delimiters are moved off ``{{``/``{%`` because the scripts contain Go
    template syntax in docker format strings (``{{.ID}} {{.Names}}``,
    ``{{.CgroupDriver}}``) that Jinja would otherwise try to evaluate.
    ``keep_trailing_newline`` preserves each template's exact final byte, which
    is how the node/control scripts stay newline-free at EOF like the committed
    originals.
    """
    return Environment(
        loader=FileSystemLoader(SCRIPT_DIR / "templates"),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
        variable_start_string="@{", variable_end_string="}@",
        block_start_string="@%", block_end_string="%@",
        comment_start_string="@#", comment_end_string="#@",
    )


def _template_context(cluster: Cluster) -> dict[str, Any]:
    n = cluster.name
    state_db_decl = f'STATE_DB="{cluster.state_db}"'
    # The committed deploy scripts align STATE_DB's trailing comment with
    # CP_NODE's, whose padding is a literal 23 spaces after the quoted host.
    # Reproducing that keeps regeneration a no-op on the existing files.
    comment_col = len(f'CP_NODE="{cluster.control_plane}"') + 23
    return {
        "name": n,
        "checkout": cluster.checkout,
        "control_plane": cluster.control_plane,
        "workers": list(cluster.workers),
        "nodelist": ",".join(cluster.workers),
        "node_count": len(cluster.workers),
        "sysbox_pool": list(cluster.sysbox_pool),
        "cpu_isolation_pool": list(cluster.cpu_isolation_pool),
        "registry_mirror_host": cluster.registry.mirror_host,
        "registry_private_host": cluster.registry.private_host,
        "registry_scratch_host": cluster.registry.scratch_host,
        "registry_mirror_port": cluster.registry.mirror_port,
        "registry_private_port": cluster.registry.private_port,
        "registry_scratch_port": cluster.registry.scratch_port,
        "deploy_registry": cluster.deploy_registry,
        "state_db": cluster.state_db,
        "state_db_pad": " " * max(1, comment_col - len(state_db_decl)),
        # A cluster with no dedicated volume must NOT pass --hyperpod: the
        # relocation helper refuses a non-block-device target and refuses one on
        # the root device, so the bootstrap aborts and no agent is ever installed
        # (cn, 2026-08-08 — all four nodes exited 1 before installing anything).
        # Rendered inline into the bootstrap call, so a cluster without a volume
        # simply omits the flag — no conditional block, hence no stray blank line
        # in the generated script.
        "hyperpod_flag": "--hyperpod " if cluster.local_disk_root else "",
        # Extra assignments spliced into the bootstrap's sudo allowlist. Each line
        # carries its own continuation backslash and trailing newline, so an EMPTY
        # node_env renders to the empty string and the surrounding script is
        # byte-identical to a cluster that declares none.
        "node_env_lines": "".join(
            f'        {k}="{v}" \\\n' for k, v in cluster.node_env
        ),
        "control_log": cluster.control_log,
        "tunnel_admin": cluster.tunnel_admin,
        "tunnel_metrics": cluster.tunnel_metrics,
        "partition": cluster.partition,
        "account": cluster.account,
        "grpc_port": cluster.grpc_port,
        "sysbox_max_concurrent": cluster.sysbox_max_concurrent,
        "allowed_host_paths": list(cluster.allowed_host_paths),
        "node_job": f"{n}-xrlenv-nodes",
        "control_job": f"{n}-xrlenv-control",
        # Repo-root-relative paths to the sibling generated scripts. They live
        # under slurm_scripts/generated/ (see GENERATED_DIRNAME), so every
        # cross-reference — sbatch targets in the deploy script, the roster path
        # in the control script, the scancel/ssh hints — carries that segment.
        "node_script": f"slurm_scripts/{GENERATED_DIRNAME}/{n}_xrlenv_node.sh",
        "control_script": f"slurm_scripts/{GENERATED_DIRNAME}/{n}_xrlenv_control.sh",
        "deploy_script": f"slurm_scripts/{GENERATED_DIRNAME}/deploy_{n}.sh",
        "node_script_name": f"{n}_xrlenv_node.sh",
        "control_script_name": f"{n}_xrlenv_control.sh",
        "deploy_script_name": f"deploy_{n}.sh",
        "node_slurm_script": f"./slurm_scripts/{GENERATED_DIRNAME}/{n}_xrlenv_node.sh",
        "nodes_yaml": f"./slurm_scripts/{GENERATED_DIRNAME}/{n}_hyperpod_nodes.yaml",
    }


def rebuild_all(clusters: tuple[Cluster, ...]) -> dict[Path, str]:
    """Render every script for every cluster from ``templates/``.

    Unlike ``render_all`` this does not read the existing files, so it also
    scaffolds a brand-new cluster — and it is what ``--check`` compares against
    to prove the committed scripts are still exactly template + config.
    """
    env = _template_env()
    rendered: dict[Path, str] = {}
    generated = SCRIPT_DIR / GENERATED_DIRNAME
    for cluster in clusters:
        context = _template_context(cluster)
        for template_name, target in TEMPLATES.items():
            rendered[generated / target.format(n=cluster.name)] = env.get_template(
                template_name
            ).render(**context)
    return rendered


def _check_disjoint(clusters: tuple[Cluster, ...]) -> None:
    """Reject clusters that would collide at runtime.

    Only meaningful when several clusters are loaded together, which is why the
    full-config path (no ``--cluster``) is the one that gates a deploy. Each
    resource below is shared infrastructure: the login node hosts every tunnel,
    and two control planes on one box would share a state DB and log file.
    """
    for label, key in (
        ("Slurm job prefix", lambda c: c.name),
        ("state_db", lambda c: c.state_db),
        ("control_log", lambda c: c.control_log),
        ("tunnel.admin port", lambda c: c.tunnel_admin),
        ("tunnel.metrics port", lambda c: c.tunnel_metrics),
    ):
        seen: dict[Any, str] = {}
        for cluster in clusters:
            value = key(cluster)
            if value in seen:
                raise ConfigError(
                    f"{label} {value!r} is used by both {seen[value]} and "
                    f"{cluster.name}; every cluster needs its own"
                )
            seen[value] = cluster.name


def _validate_slurm_nodes(clusters: tuple[Cluster, ...]) -> list[str]:
    """Cross-check every declared host against the live Slurm node list.

    Catches the failure this whole schema exists to prevent: a hostname left
    behind after a cluster is reprovisioned into a new subnet. Slurm node names
    are IP-derived, so every name changes on reprovision and a stale one only
    surfaces as ``sbatch: Invalid node name specified`` mid-deploy.
    """
    # ``-N`` is load-bearing: without it sinfo emits a COMPRESSED hostlist
    # (``ip-10-1-98-[43,127,133]``) that no exact-match lookup can resolve, which
    # makes every host look absent. ``-N`` prints one row per node per partition.
    try:
        proc = subprocess.run(
            ["sinfo", "-N", "-h", "-o", "%N"], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ConfigError(f"could not run sinfo to validate node names: {exc}") from exc
    if proc.returncode != 0:
        raise ConfigError(f"sinfo failed: {proc.stderr.strip() or proc.returncode}")
    known = {line.strip() for line in proc.stdout.splitlines() if line.strip()}
    problems = []
    for cluster in clusters:
        for label, hosts in (
            ("control_plane", (cluster.control_plane,)),
            ("workers", cluster.workers),
        ):
            for host in hosts:
                if host not in known:
                    problems.append(f"{cluster.name}.{label}: {host} is not a Slurm node")
    return problems


def show_diff(path: Path, current: str, rendered: str) -> None:
    relative = path.relative_to(SCRIPT_DIR.parent)
    sys.stdout.writelines(
        difflib.unified_diff(
            current.splitlines(keepends=True),
            rendered.splitlines(keepends=True),
            fromfile=str(relative),
            tofile=str(relative),
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update dev/prod/cn Slurm deployment scripts from clusters.yaml."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"cluster YAML path (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--cluster",
        action="append",
        dest="clusters",
        help="cluster to update; repeat for multiple clusters (default: every "
        "cluster declared in clusters.yaml)",
    )
    parser.add_argument(
        "--patch-only",
        action="store_true",
        help="update ONLY the topology lines (control plane, nodelist, pools) of "
        "existing scripts instead of re-rendering them from templates/. Preserves "
        "local edits, but silently ignores every other clusters.yaml field — use "
        "only to touch up a live cluster's hosts without a full regeneration",
    )
    parser.add_argument(
        "--validate-slurm",
        action="store_true",
        help="additionally check every control_plane/worker against `sinfo`, so a "
        "hostname left over from a previous cluster fails here instead of "
        "mid-deploy with 'Invalid node name specified'",
    )
    parser.add_argument(
        "--env-cluster",
        help="also sync this cluster's topology (control-plane + registry hosts) "
        "into the local checkout's .env, which the consumer path (xrlenv.from_env "
        "/ benchmark oracle sweeps) reads to dial the control plane",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="show stale generated topology and exit nonzero without writing",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    # The cluster list comes from clusters.yaml, not a constant here, so adding a
    # cluster never requires editing this file.
    try:
        declared = cluster_names(config_path)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if unknown := [c for c in (args.clusters or ()) if c not in declared]:
        print(
            f"error: unknown cluster(s): {', '.join(unknown)} "
            f"(clusters.yaml declares: {', '.join(declared)})",
            file=sys.stderr,
        )
        return 2
    selected = tuple(args.clusters or declared)
    env_cluster = args.env_cluster
    if env_cluster is not None and env_cluster not in selected:
        print(
            f"error: --env-cluster {env_cluster} is not in the selected clusters "
            f"({', '.join(selected)})",
            file=sys.stderr,
        )
        return 2
    # ``.env`` lives at the checkout root (one level up from slurm_scripts). The
    # generator only ever touches the LOCAL checkout's .env — prod's .env is in a
    # separate checkout, so run the generator there (with --env-cluster prod) to
    # sync it.
    env_path = SCRIPT_DIR.parent / ".env" if env_cluster is not None else None
    try:
        clusters = load_config(config_path, selected)
        _check_disjoint(load_config(config_path, declared))
        if implicit := warn_implicit_state_paths(config_path, selected):
            print(
                f"warning: relying on a derived value for {', '.join(implicit)} — "
                "declare it in clusters.yaml so durable state cannot move if the "
                "derivation changes",
                file=sys.stderr,
            )
        if args.validate_slurm and (problems := _validate_slurm_nodes(clusters)):
            for problem in problems:
                print(f"error: {problem}", file=sys.stderr)
            return 2
        # Rendering whole scripts from templates/ is the DEFAULT, and --check
        # always uses it: these files are generated artifacts, so the only correct
        # content is exactly what templates/ + clusters.yaml produce. That also
        # makes --check the GOLDEN GATE — a hand-edit to a generated script, or a
        # drifted template, fails CI rather than a deploy.
        #
        # --patch-only is the narrow escape hatch. It rewrites just the topology
        # lines, so it CANNOT see fields the patch patterns don't cover
        # (allowed_host_paths, state_db, tunnel ports, checkout, …) — which is
        # exactly why it is not the default: editing one of those in clusters.yaml
        # and running the patch path would silently do nothing.
        if args.patch_only and not args.check:
            rendered = render_all(clusters, env_cluster=env_cluster, env_path=env_path)
        else:
            rendered = rebuild_all(clusters)
            if env_cluster is not None:
                assert env_path is not None
                target = next(c for c in clusters if c.name == env_cluster)
                rendered[env_path] = render_env(
                    env_path.read_text(encoding="utf-8"), target, env_path
                )
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    changed = []
    for path, output in rendered.items():
        current = path.read_text(encoding="utf-8") if path.exists() else None
        if current is None:
            changed.append(path)
            if args.check:
                rel = path.relative_to(SCRIPT_DIR.parent)
                print(f"{rel}: does not exist (run without --check to create)",
                      file=sys.stderr)
            continue
        if current != output:
            changed.append(path)
            if args.check:
                if env_path is not None and path == env_path:
                    # NAME the drifted topology keys (hosts/ports — safe) rather
                    # than a raw diff, which could echo an adjacent secret line.
                    # These keys are render_env's whole footprint, so this is the
                    # complete .env drift and tells the operator exactly what to
                    # fix + to what.
                    rel = path.relative_to(SCRIPT_DIR.parent)
                    assert env_cluster is not None
                    target = next(c for c in clusters if c.name == env_cluster)
                    drift = env_topology_drift(current, target)
                    print(
                        f"{rel}: {len(drift)} topology key(s) out of sync with "
                        f"clusters.yaml (cluster {env_cluster}) — "
                        f"`--env-cluster {env_cluster}` will fix:",
                        file=sys.stderr,
                    )
                    for key, cur, want in drift:
                        if cur is None:
                            print(f"    missing: {key}  (clusters.yaml: {want})",
                                  file=sys.stderr)
                        else:
                            print(f"    stale:   {key}  (.env: {cur!r} -> "
                                  f"clusters.yaml: {want!r})", file=sys.stderr)
                else:
                    show_diff(path, current, output)

    if args.check:
        if changed:
            print(f"{len(changed)} deployment script(s) are out of date", file=sys.stderr)
            return 1
        print("deployment scripts are up to date")
        return 0

    for path in changed:
        existed = path.exists()
        # Generated scripts live under slurm_scripts/generated/; create it if a
        # fresh checkout or a new cluster lands here before the directory exists.
        # (.env, the one non-generated output, sits at the repo root, which does.)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered[path], encoding="utf-8")
        # The deploy orchestrator is run directly (`bash deploy_<n>.sh`, but also
        # `./deploy_<n>.sh`) while the node/control scripts are only ever handed
        # to sbatch. Mark a freshly scaffolded deploy script executable so a new
        # cluster needs no manual chmod; never touch an existing file's mode.
        if not existed and path.name.startswith("deploy_"):
            path.chmod(path.stat().st_mode | 0o111)
        print(f"{'updated' if existed else 'created'} {path.relative_to(SCRIPT_DIR.parent)}")
    if not changed:
        print("deployment scripts are already up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
