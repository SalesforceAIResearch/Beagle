#!/usr/bin/env python3
"""Synchronize Slurm scripts with clusters.yaml topology."""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "clusters.yaml"
CLUSTERS = ("example",)
REQUIRED_FIELDS = {
    "control_plane",
    "registry",
    "workers",
    "sysbox_pool",
    "cpu_isolation_pool",
}
REGISTRY_REQUIRED = {"mirror_host", "private_host"}
REGISTRY_OPTIONAL = {"mirror_port", "private_port"}
REGISTRY_ALLOWED = REGISTRY_REQUIRED | REGISTRY_OPTIONAL
DEFAULT_MIRROR_PORT = 5010
DEFAULT_PRIVATE_PORT = 5011
HOST_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ConfigError(ValueError):
    """Raised when cluster topology cannot safely generate shell scripts."""


@dataclass(frozen=True)
class RegistryTopology:
    """Per-cluster image-registry endpoints. Both hosts default to the same
    shared box today to save resources, but the schema lets a cluster run its
    own registry — on its own ports. Ports are optional in clusters.yaml and
    default to the project conventions (:5010 mirror, :5011 private)."""

    mirror_host: str
    private_host: str
    mirror_port: int = DEFAULT_MIRROR_PORT
    private_port: int = DEFAULT_PRIVATE_PORT


@dataclass(frozen=True)
class Cluster:
    name: str
    control_plane: str
    registry: RegistryTopology
    workers: tuple[str, ...]
    sysbox_pool: tuple[str, ...]
    cpu_isolation_pool: tuple[str, ...]


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
    return RegistryTopology(
        mirror_host=value["mirror_host"],
        private_host=value["private_host"],
        mirror_port=_parse_port(
            value.get("mirror_port", DEFAULT_MIRROR_PORT),
            cluster=cluster, key="mirror_port",
        ),
        private_port=_parse_port(
            value.get("private_port", DEFAULT_PRIVATE_PORT),
            cluster=cluster, key="private_port",
        ),
    )


def _parse_cluster(name: str, value: Any) -> Cluster:
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a YAML mapping")
    fields = set(value)
    if fields != REQUIRED_FIELDS:
        missing = sorted(REQUIRED_FIELDS - fields)
        extra = sorted(fields - REQUIRED_FIELDS)
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

    return Cluster(
        name, control_plane, registry, workers, sysbox_pool, cpu_isolation_pool
    )


def load_config(path: Path, selected: tuple[str, ...]) -> tuple[Cluster, ...]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"configuration file does not exist: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("configuration root must be a YAML mapping")

    missing = [name for name in selected if name not in raw]
    if missing:
        raise ConfigError(f"configuration is missing clusters: {', '.join(missing)}")
    return tuple(_parse_cluster(name, raw[name]) for name in selected)


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
    text = _replace_once(
        text,
        r"^#SBATCH --nodelist=.*$",
        f"#SBATCH --nodelist={','.join(cluster.workers)}",
        path=path,
    )
    return _replace_once(
        text,
        r'(: "\$\{XRLENV_GRPC_HOST:=)[^}]*(\}")',
        rf"\g<1>{cluster.control_plane}\2",
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


def render_env(text: str, cluster: Cluster, path: Path) -> str:
    values = {
        "XRLENV_GRPC_HOST": cluster.control_plane,
        "XRLENV_MIRROR_REGISTRY_HOST": cluster.registry.mirror_host,
        "XRLENV_MIRROR_REGISTRY_PORT": str(cluster.registry.mirror_port),
        "XRLENV_PRIVATE_REGISTRY_HOST": cluster.registry.private_host,
        "XRLENV_PRIVATE_REGISTRY_PORT": str(cluster.registry.private_port),
    }
    for key in _ENV_TOPOLOGY_KEYS:
        text = _render_env_var(text, key, values[key], path=path)
    return text


def render_all(
    clusters: tuple[Cluster, ...],
    *,
    env_cluster: str | None = None,
    env_path: Path | None = None,
) -> dict[Path, str]:
    rendered: dict[Path, str] = {}
    for cluster in clusters:
        transformations = {
            SCRIPT_DIR / f"deploy_{cluster.name}.sh": render_deploy,
            SCRIPT_DIR / f"{cluster.name}_xrlenv_node.sh": render_node,
            SCRIPT_DIR / f"{cluster.name}_xrlenv_control.sh": render_control,
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
        description="Update Slurm deployment scripts from clusters.yaml."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"cluster YAML path (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--cluster",
        choices=CLUSTERS,
        action="append",
        dest="clusters",
        help="cluster to update; repeat for multiple clusters (default: all)",
    )
    parser.add_argument(
        "--env-cluster",
        choices=CLUSTERS,
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
    selected = tuple(args.clusters or CLUSTERS)
    env_cluster = args.env_cluster
    if env_cluster is not None and env_cluster not in selected:
        print(
            f"error: --env-cluster {env_cluster} is not in the selected clusters "
            f"({', '.join(selected)})",
            file=sys.stderr,
        )
        return 2
    # ``.env`` lives at the checkout root (one level up from slurm_scripts). The
    # generator only ever touches the LOCAL checkout's .env — another cluster's .env is in a
    # separate checkout, so run the generator there (with --env-cluster example) to
    # sync it.
    env_path = SCRIPT_DIR.parent / ".env" if env_cluster is not None else None
    try:
        clusters = load_config(args.config.resolve(), selected)
        rendered = render_all(clusters, env_cluster=env_cluster, env_path=env_path)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    changed = []
    for path, output in rendered.items():
        current = path.read_text(encoding="utf-8")
        if current != output:
            changed.append(path)
            if args.check:
                if env_path is not None and path == env_path:
                    # Never print .env contents — it carries secrets. Report drift only.
                    rel = path.relative_to(SCRIPT_DIR.parent)
                    print(
                        f"{rel}: topology keys out of date (diff suppressed — "
                        "file contains secrets)",
                        file=sys.stderr,
                    )
                else:
                    show_diff(path, current, output)

    if args.check:
        if changed:
            print(f"{len(changed)} deployment script(s) are out of date", file=sys.stderr)
            return 1
        print("deployment scripts are up to date")
        return 0

    for path in changed:
        path.write_text(rendered[path], encoding="utf-8")
        print(f"updated {path.relative_to(SCRIPT_DIR.parent)}")
    if not changed:
        print("deployment scripts are already up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
