"""Pattern A — instance resolver (spec 06 §"Importing existing benchmark task suites").

Used by case-1 RL training plug-ins where the benchmark ships one
Docker image *per task instance*. Hard-coding hundreds of templates
is the wrong shape: instead, one *outer* template declares an
:class:`InstanceResolver` that maps each task's ``instance_id`` to
its image, resources, mounts, and adapter init overrides.

The resolver is loaded lazily by import name so the platform doesn't
take a hard dep on any benchmark plug-in: the catalog can register a
manifest that *references* a resolver class
(e.g. ``xrlenv_plugins.benchmarks.<your_plugin>.adapter.YourInstanceResolver``)
without that plug-in's upstream library being installed. The actual
import happens at rollout-start time, when the operator has installed
the benchmark plug-in.

```{note}
Pattern A applies to **case-1** RL training plug-ins under the
slim pivot. Case-2/3 evaluation harnesses (SWE-bench, harbor) do
their own per-task resolution inside the harness — see
``xrlenv_plugins/harbor/`` for the harbor-shape adapter pattern,
``xrlenv/compat/docker_client.py`` for the docker-py drop-in.
```

Phase-0 surface:

- :class:`InstanceResolverDecl` — the manifest's serialized handle to
  a resolver (module + class + index_path).
- :class:`InstanceResolver` Protocol — what implementers must provide.
- :class:`ResolvedInstance` — what ``resolve(instance_id)`` returns;
  the coordinator overlays its fields onto the outer template.
- :func:`load_resolver` — import a resolver by spec.
- :func:`apply_to_manifest` — produce the per-rollout effective
  manifest by overlaying a ``ResolvedInstance``'s fields.

Phase-1+ extensions (not in this slice): cache_key derivation for
warmup Layer 1 (spec 15 §"Layer 1"), bulk enumeration via
``enumerate_instances`` for ``xrlenv analyze`` (spec 16).
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from xrlenv.backends.base import MountSpec, NetworkPolicy, ResourceSpec

if TYPE_CHECKING:
    from xrlenv.control.template_catalog import TemplateManifest

LOGGER = logging.getLogger(__name__)


class InstanceResolverDecl(BaseModel):
    """Manifest serialization of an :class:`InstanceResolver` reference.

    YAML form (per spec 06 Pattern A example)::

        instances:
          module: xrlenv_plugins.benchmarks.<your_plugin>.adapter
          class: YourInstanceResolver
          index_path: ./tasks/

    ``index_path`` is forwarded to the resolver's constructor verbatim
    (paths interpreted relative to the manifest dir during YAML load).
    Resolver-specific config goes in ``options`` so the schema doesn't
    need a new field per benchmark.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    module: str
    class_name: str = Field(..., alias="class")
    index_path: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class VerifierUpload(BaseModel):
    """One verifier-asset injection (D12 stage 1).

    The resolver computes a tarball host-side at resolve time; the
    coordinator's :func:`xrlenv.control.reward.compute_in_sandbox_final_reward`
    ships the bytes to the node and extracts them into ``target_dir``
    immediately before running the manifest's ``reward.cmd``. The
    grader files therefore do not exist in the sandbox during the
    agent's ``step()`` loop — closes audit H1's timing-isolation
    half.

    ``tarball`` is gzipped tar (``mode="w:gz"`` from :mod:`tarfile`);
    Docker's ``put_archive`` auto-detects gzip. Keep payloads small
    (<1 MB typical for terminal-bench tasks; the bidi message size
    cap is ~16 MB).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    target_dir: str
    tarball: bytes


#: A1 / D20 (P1.2) — re-exported here so :class:`ResolvedInstance`
#: can declare it without importing from ``template_catalog`` (which
#: imports back from this module — circular). Both definitions
#: must stay in lockstep; the canonical contract documentation
#: lives in :data:`xrlenv.control.template_catalog.ImagePinMode`.
ImagePinMode = Literal[
    "registry_digest", "per_node_local", "shared_storage", "scratch_build",
]


class ResolvedInstance(BaseModel):
    """One resolver hit. Fields override the outer manifest's defaults
    when applied via :func:`apply_to_manifest`.

    All fields except ``instance_id`` are optional — a resolver that
    only varies the image leaves resources and mounts untouched.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    instance_id: str
    image: str | None = None
    resources: ResourceSpec | None = None
    mounts: tuple[MountSpec, ...] | None = None
    init_params: dict[str, Any] = Field(default_factory=dict)
    """Extra ``env_adapter.init_params`` keys merged into the outer
    manifest's defaults; the resolver wins on key collisions."""
    network: NetworkPolicy | None = None
    """Per-task network policy (Pattern A). When the resolver supplies
    a value here it wins over the user's run-config / per-rollout
    network kwarg — the benchmark author knows whether a particular
    task needs hermetic isolation (e.g. the prompt explicitly forbids
    web access) or open egress (e.g. a task that fetches a package).
    Leave ``None`` to let the request-level value (run-config /
    per-rollout kwarg / ``DEFAULT_NETWORK``) drive the policy."""
    verifier_uploads: tuple[VerifierUpload, ...] = ()
    """Per-task verifier-asset injections (D12 stage 1). Each entry is
    a (target_dir, tarball) pair the coordinator extracts into the
    sandbox at reward time, *not* at image-build time. Empty for
    benchmarks that don't need timing-isolated grader assets."""
    image_pin_mode: ImagePinMode | None = None
    """A1 / D20 (P1.2) — per-instance override of the outer manifest's
    image_pin_mode. ``None`` defers to the manifest-level value
    (typical case). Set this when a single benchmark mixes
    distribution strategies — e.g., most tasks are
    ``per_node_local`` builds but one task uses a registry-pulled
    image. The applied manifest carries the resolved value so the
    catalog's overlay-validation path sees the correct mode."""


class InstanceResolver(Protocol):
    """Maps an ``instance_id`` to a :class:`ResolvedInstance`.

    Implementations live next to the EnvAdapter that consumes them
    (e.g. ``xrlenv_plugins.benchmarks.<your_plugin>.adapter.YourInstanceResolver``).
    The constructor receives the :class:`InstanceResolverDecl` so it
    can read ``index_path`` and any ``options``.
    """

    def __init__(self, decl: InstanceResolverDecl) -> None: ...

    def resolve(self, instance_id: str) -> ResolvedInstance: ...

    def enumerate_instances(self) -> list[str]:
        """Optional: return every known instance id. Phase-0 callers may
        ignore this; ``xrlenv analyze`` (spec 16) consumes it. Default
        implementations return an empty list."""
        return []


# ──────────────────────────────────────────────────────────────────────────────
# Loader
# ──────────────────────────────────────────────────────────────────────────────


def load_resolver(decl: InstanceResolverDecl) -> InstanceResolver:
    """Import + construct the resolver class named in ``decl``.

    Imports happen lazily so a manifest that references e.g.
    ``xrlenv_plugins.benchmarks.<your_plugin>.adapter.YourInstanceResolver``
    can be loaded at register time even when the benchmark plug-in
    isn't fully wired; the import failure surfaces only at the first
    rollout that needs it.
    """
    try:
        module = importlib.import_module(decl.module)
    except ImportError as exc:
        raise InstanceResolverImportError(
            f"could not import resolver module {decl.module!r}: {exc}"
        ) from exc
    try:
        cls = getattr(module, decl.class_name)
    except AttributeError as exc:
        raise InstanceResolverImportError(
            f"module {decl.module!r} has no attribute {decl.class_name!r}"
        ) from exc
    return cls(decl)  # type: ignore[no-any-return]


class InstanceResolverImportError(RuntimeError):
    """Raised when :func:`load_resolver` cannot find the resolver class.

    Distinguished from a generic ImportError so the coordinator can
    map it onto a clean ``RolloutFailed("resolver_unavailable")``
    rather than crashing the start_rollout call with a generic
    traceback.
    """


# ──────────────────────────────────────────────────────────────────────────────
# Manifest overlay
# ──────────────────────────────────────────────────────────────────────────────


def apply_to_manifest(
    manifest: TemplateManifest,
    resolved: ResolvedInstance,
) -> TemplateManifest:
    """Return a per-rollout effective manifest with resolver fields
    overlaid on the outer manifest.

    Spec 06 §"Pattern A" semantics:

    - ``image`` from the resolver replaces the manifest's image.
    - ``resources`` from the resolver wholly replaces the manifest's
      resources (no per-field merge — benchmarks that vary cpu also
      tend to vary mem + disk together).
    - ``mounts`` extend the manifest's mounts (resolver mounts append
      after manifest defaults, so per-instance bind-mounts win on
      overlapping sandbox_path lookups).
    - ``init_params`` merge: resolver wins on key collisions.

    The manifest's original ``digest`` is left untouched — it identifies
    the *outer* manifest, not the per-instance derivative. Per-instance
    image identity rides on ``image`` itself (which the catalog has
    already pinned to a digest at register-time, spec 19).
    """
    updates: dict[str, Any] = {}
    if resolved.image is not None:
        updates["image"] = resolved.image
    if resolved.resources is not None:
        updates["resources"] = resolved.resources
    if resolved.mounts is not None:
        merged_mounts = (*(manifest.resources.mounts or ()), *resolved.mounts)
        # ``resources`` may already be in updates (resolver replaced it);
        # apply mount extension on top of whatever resource value wins.
        base_resources = updates.get("resources") or manifest.resources
        updates["resources"] = base_resources.model_copy(
            update={"mounts": merged_mounts},
        )
    if resolved.init_params:
        existing = dict(manifest.env_adapter.init_params or {})
        existing.update(resolved.init_params)
        updates["env_adapter"] = manifest.env_adapter.model_copy(
            update={"init_params": existing},
        )
    if resolved.image_pin_mode is not None:
        # A1 / D20 (P1.2) — resolver may override the manifest-level
        # image_pin_mode for this specific instance (e.g., one task's
        # image is registry-pulled while the rest are per-node-built).
        # The applied manifest carries the resolved value so the
        # catalog's overlay-validation path (validate_overlay) and the
        # coordinator's audit-event payload see the correct mode.
        updates["image_pin_mode"] = resolved.image_pin_mode
    if not updates:
        return manifest
    return manifest.model_copy(update=updates)


def index_path_relative_to(decl: InstanceResolverDecl, base: Path) -> Path | None:
    """Resolve ``decl.index_path`` against ``base`` (the manifest's dir).

    Helper for resolver implementations that need to read a vendored
    instances directory shipped alongside the manifest.
    """
    if decl.index_path is None:
        return None
    p = Path(decl.index_path)
    return p if p.is_absolute() else (base / p).resolve()


__all__ = [
    "InstanceResolver",
    "InstanceResolverDecl",
    "InstanceResolverImportError",
    "ResolvedInstance",
    "VerifierUpload",
    "apply_to_manifest",
    "index_path_relative_to",
    "load_resolver",
]
