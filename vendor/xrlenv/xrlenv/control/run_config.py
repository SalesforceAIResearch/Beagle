"""Run-config: per-experiment policy supplied by the user (client side).

Manifests carry the **immutable benchmark contract** — name, adapter
binding, instances / assets, reward command, and the static fields the
plug-in author owns. Per-experiment **policy** — deadlines, idle TTL,
init_params, mounts, resource budgets for non-Pattern-A templates,
backend choice (which sandbox runtime), network policy — lives here,
in a YAML file the user points at via ``Client(run_config=...)``.

Layering at rollout time:

1. **Plug-in contract** (``manifest.yaml``) — immutable.
2. **Per-instance data** (resolver / asset block) — supplied by the
   plug-in's resolver for Pattern A; manifest for Pattern B / Simple.
3. **User policy** (this module, ``Client(run_config=...)``) —
   per-trainer-process; the same control plane can serve multiple
   trainers with different run-configs simultaneously.
4. **Per-rollout SDK kwargs** (``client.rollout(deadline=..., ...)``)
   — overrides 3 at the call site.

The control plane itself does **not** load a run-config; binding
policy at the cluster level would make ``xrlenv up`` single-tenant.
The run-config is purely a client-side convenience for hoisting the
"don't restate this on every rollout call" values into one file.

Schema (YAML):

.. code-block:: yaml

    version: 1
    manifests:
      terminal-bench-2:
        deadlines:
          hard_s: 1800
          step_timeout_s: 90
        idle_ttl_s: 600
        init_params:
          verbose: true
      hello-shell:
        deadlines:
          hard_s: 60

Each top-level key under ``manifests:`` is the manifest's ``name:``
field — i.e. the string the consumer passes as ``client.rollout(template=...)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from xrlenv.backends.base import NetworkPolicy
from xrlenv.errors import ManifestInvalid


class DeadlinesPolicy(BaseModel):
    """Per-template deadline knobs. Every field is optional; omitted
    fields fall through to platform defaults at rollout time."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    hard_s: float | None = None
    step_timeout_s: float | None = None
    setup_timeout_s: float | None = None
    teardown_timeout_s: float | None = None
    init_timeout_s: float | None = None


class TemplatePolicy(BaseModel):
    """The user's policy for one manifest."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    deadlines: DeadlinesPolicy | None = None
    idle_ttl_s: float | None = None
    init_params: dict[str, Any] = Field(default_factory=dict)
    """Static config dict merged into ``adapter.setup(init_params)``.
    The plug-in author defines what keys the adapter accepts; the user
    supplies values here. Per-rollout ``init={...}`` kwargs override
    these on the call site."""
    backend: str | None = None
    """Which sandbox runtime to use for this template — ``docker``
    (phase 0) or ``cubesandbox`` (phase 1, KVM-only). The plug-in
    author doesn't set this because different operators run different
    hardware; the user picks what works on their cluster."""
    network: NetworkPolicy | None = None
    """Network policy for sandboxes of this template — one of
    ``none``, ``open``, ``egress-allowlist`` (phase 1). Typed as
    :data:`xrlenv.backends.base.NetworkPolicy` so pydantic rejects
    typos at load time (the Docker backend silently treats unknown
    values as bridge networking, so unvalidated strings would be a
    "fail-open" footgun for hermetic workloads). Pattern A's resolver
    overrides this per-task in Slice 9b; for Pattern B / Simple
    templates this is the operating value."""


class RunConfig(BaseModel):
    """Top-level run-config: schema version + per-manifest policy map."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: int = 1
    manifests: dict[str, TemplatePolicy] = Field(default_factory=dict)

    def policy_for(self, manifest_name: str) -> TemplatePolicy | None:
        """Return the policy for ``manifest_name`` if the run-config
        covers it, otherwise ``None``. Callers fall back to platform
        defaults (or hard-error, depending on context) when this
        returns ``None``.
        """
        return self.manifests.get(manifest_name)


def load_run_config(path: str | Path) -> RunConfig:
    """Load + validate a run-config YAML file.

    Raises :class:`ManifestInvalid` for shape errors so the operator
    error path matches the manifest loader's.
    """
    p = Path(path)
    if not p.is_file():
        raise ManifestInvalid(f"run-config file not found: {p}")
    raw = yaml.safe_load(p.read_text()) or {}
    if not isinstance(raw, dict):
        raise ManifestInvalid(
            f"{p}: top-level run-config must be a mapping; got {type(raw).__name__}"
        )
    try:
        return RunConfig.model_validate(raw)
    except Exception as exc:
        raise ManifestInvalid(f"{p}: invalid run-config: {exc}") from exc


__all__ = [
    "DeadlinesPolicy",
    "RunConfig",
    "TemplatePolicy",
    "load_run_config",
]
