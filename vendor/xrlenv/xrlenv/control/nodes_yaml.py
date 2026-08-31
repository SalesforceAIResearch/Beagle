"""Loader for the operator-managed ``nodes.yaml`` inventory (spec 09).

The control plane uses this to know which nodes it *expects* to see —
distinct from the runtime ``NodeRegistry`` which tracks which nodes have
actually connected. Slice 3.5 ships the loader + schema; the auth-token
path lands when shared bearer tokens (spec 19 phase-0) are wired into the
gRPC server in Slice 4.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from xrlenv.control.kwargs_policy import DEFAULT_POLICY, KwargsPolicy
from xrlenv.errors import ManifestInvalid

LOGGER = logging.getLogger(__name__)


class NodeEntry(BaseModel):
    """One row in ``nodes.yaml``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    address: str | None = None
    cloud: str | None = None
    instance_type: str | None = None
    backends: tuple[str, ...] = ()
    auth_token_env: str | None = None
    """Name of the env var the control plane reads to obtain this node's
    shared bearer token. Slice 4 wires this into auth checks; Slice 3.5
    just records it.
    """
    sysbox: bool = False
    """§5.3 / sysbox pool — operator opt-in marking this node as a member of
    the dedicated Sysbox node pool. Purely declarative inventory: it drives
    (a) the deploy tooling, which runs ``xrlenv_plugins/sysbox/
    install_sysbox_node.sh`` on pool nodes so their docker advertises the
    ``sysbox-runc`` runtime, and (b) operator/CLI validation (e.g. warn if
    ``allowed_runtimes`` includes ``sysbox-runc`` but no node is marked). The
    control plane's placement filter routes ``container_runtime='sysbox-runc'``
    acquires to whatever nodes actually advertise the runtime at connect
    (NodeHello) — this flag is the operator's intent, the advertisement is the
    ground truth. Sysbox is a container-escape surface, so keep the pool to
    dev / single-tenant nodes (see xrlenv_plugins/sysbox/README.md)."""
    max_concurrent_by_runtime: dict[str, int] = Field(default_factory=dict)
    """Per-node cap on concurrently *running* containers of a given OCI runtime
    (design-per-node-runtime-concurrency-cap.md). e.g.
    ``{sysbox-runc: 8}`` bounds concurrent sysbox containers on this node so a
    create/exec storm can't overwhelm sysbox-fs into a FUSE wedge — the scheduler
    counts running + in-flight sysbox sessions and holds overflow in the
    admission queue rather than placing more. Empty (the default) ⇒ unlimited ⇒
    current behavior for every runtime. Distinct from the node-side
    ``sysbox_create_concurrency`` (which serializes the create *burst* but does
    not bound steady-state concurrency). Keys are runtime names as advertised at
    NodeHello (``runc``, ``sysbox-runc``); an entry for a runtime the node
    doesn't run is simply never hit."""


class NodesInventory(BaseModel):
    """Validated form of ``nodes.yaml``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: int = 1
    nodes: tuple[NodeEntry, ...] = Field(default_factory=tuple)
    policy: KwargsPolicy = Field(default_factory=lambda: DEFAULT_POLICY)
    """Cluster-wide docker-kwarg policy (issue #6). Defaults work for
    swebench / terminal-bench / coding-bench / SCUBA-style KVM
    benchmarks; operators only need to touch this section when they
    want to restrict (``denied_caps``) or opt in to a Level-2 risk
    (``allow_host_network``, ``allow_privileged``,
    ``allowed_host_paths``). See ``xrlenv/control/kwargs_policy.py``
    for the four-tier model.
    """

    def by_id(self) -> dict[str, NodeEntry]:
        return {n.id: n for n in self.nodes}

    def sysbox_pool(self) -> tuple[NodeEntry, ...]:
        """The nodes marked ``sysbox: true`` — the dedicated Sysbox pool the
        deploy tooling installs the patched runtime on (see
        ``xrlenv_plugins/sysbox/``)."""
        return tuple(n for n in self.nodes if n.sysbox)


def load_nodes_yaml(path: Path) -> NodesInventory:
    """Parse + validate ``nodes.yaml`` at ``path``.

    Raises :class:`ManifestInvalid` when the schema is wrong; returns an
    empty inventory when the file is absent so the in-process / single-node
    deployments don't need to ship a placeholder file.
    """
    if not path.exists():
        LOGGER.info("nodes.yaml not present at %s; assuming single-node deployment", path)
        return NodesInventory()
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ManifestInvalid(f"{path}: top-level must be a mapping")
    try:
        return NodesInventory.model_validate(raw)
    except Exception as exc:
        raise ManifestInvalid(f"{path}: {exc}") from exc


def _coerce_entry(raw: Any) -> NodeEntry:
    """Convenience: pydantic handles dict→NodeEntry; this keeps a hand-typed
    helper around in case Slice 4's CLI surface needs to add per-row defaults
    that depend on environment-variable lookups.
    """
    return NodeEntry.model_validate(raw)


__all__ = ["NodeEntry", "NodesInventory", "load_nodes_yaml"]
