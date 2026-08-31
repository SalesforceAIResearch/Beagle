"""Platform-default constants for runtime fields the manifest no longer carries.

The manifest is the **immutable benchmark contract** — identity,
adapter binding, instances/assets, reward command. It carries no
operational policy. Per-rollout policy (deadlines, idle TTL,
init_params, mounts, backend, network) lives in the user's
run-config and is plumbed through the rollout request.

When the run-config doesn't supply a value for a field that *must*
have one to drive a rollout (backend, network), the runtime falls
back to the constants in this module — never to a manifest field.
The fallback exists so a quick smoke run with a bare-minimum
run-config (or no run-config at all) still works; production
deployments should declare these explicitly in their run-config.
"""

from __future__ import annotations

from xrlenv.backends.base import NetworkPolicy

DEFAULT_BACKEND: str = "docker"
"""The only backend that ships in phase 0. Phase 1 adds
``cubesandbox`` (KVM-only); operators on KVM-capable hardware can
flip to it via the run-config's ``backend:`` field."""

DEFAULT_NETWORK: NetworkPolicy = "open"
"""The most permissive policy — sandboxes can reach the public
internet. Tighten via the run-config's ``network: none`` for
hermetic tasks. Pattern A's resolver overrides this per-task in
Slice 9b."""


__all__ = ["DEFAULT_BACKEND", "DEFAULT_NETWORK"]
