"""TemplateCatalog (spec 03 / spec 06).

Loads ``manifest.yaml`` files from disk, validates them against the
manifest schema (immutable benchmark contract: identity, adapter,
instances/assets, reward command), computes a stable content digest
(spec 00 invariant 4), and exposes a typed :class:`TemplateManifest`
for the coordinator and scheduler to consume.

Per-experiment policy (deadlines, idle TTL, init_params) lives in the
user's run-config, not the manifest — see
:mod:`xrlenv.control.run_config` for the loader and
``Client(run_config=...)`` for the client-side binding.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from xrlenv.backends.base import MountSpec, ResourceSpec, TemplateRef
from xrlenv.control.assets import AssetSpec
from xrlenv.control.image_builder import ImageBuilderDecl
from xrlenv.control.instance_resolver import InstanceResolverDecl
from xrlenv.envs.base import REWARD_MODES
from xrlenv.errors import ManifestInvalid

#: A1 / D20 (P1.2) — how the platform identifies the image's bytes.
#:
#: - ``"registry_digest"``: image is registry-published; the digest is
#:   the same on every node; the catalog pins centrally at register
#:   time via the wired ``digest_resolver``. Default for backward
#:   compat with phase-0 manifests; the right choice when every node
#:   pulls from a shared registry.
#: - ``"per_node_local"``: image is built or shipped to each node
#:   individually; each host has its own bytes (and its own per-host
#:   ``Id``); the control plane has **no** authoritative view of the
#:   bytes. The catalog skips central digest pinning entirely. Use
#:   this for ``build-task-images.sh``-style workflows where the
#:   operator runs the build on each VM.
#: - ``"shared_storage"``: image lives on read-only shared storage
#:   (NFS/S3-backed mount) every node maps; pinned by the storage
#:   layer's content hash, not via a registry. Reserved for phase-2
#:   deployments — accepted by the schema today but treated like
#:   ``per_node_local`` (no central pinning) in P1.2.
ImagePinMode = Literal["registry_digest", "per_node_local", "shared_storage"]

LOGGER = logging.getLogger(__name__)

RewardOnError = Literal["fail_rollout", "zero_reward", "partial"]
RewardOutputFormat = Literal[
    "exit_code",     # reward = 1.0 if exit_code == 0 else 0.0 (terminal-bench-2 style)
    "stdout_float",  # reward = float(last non-empty line of stdout) — generic graders
    "json_stdout",   # reward = json.loads(stdout)[score_key] — wrapped graders
    "json_file",     # reward = json.loads(read(output_path))[score_key] — OSWorld / SWE-bench style
]
RewardAggregator = Literal[
    "mean",          # sum(scores) / len(scores) — equal-weight average
    "sum",           # sum(scores) — additive
    "weighted_sum",  # sum(score * weight) — default; honors per-grader weights
    "max",           # max(scores) — best-of
    "min",           # min(scores) — worst-of (e.g. safety floors)
    "first",         # scores[0] — treats grader[0] as canonical; rest informational
]


class GraderSpec(BaseModel):
    """One in-sandbox grader (spec 02 RewardContract, multi-grader extension).

    Each grader runs as ``node.run_in_sandbox(cmd, timeout_s=...)`` after the
    rollout reaches ``done``. Per-grader scores are stored in
    ``trajectory.metadata.rewards = {name: float}`` so the consumer / admin can
    inspect each sub-score; the aggregate goes to ``trajectory.final_reward``
    via the contract's ``aggregator``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    cmd: tuple[str, ...]
    output_format: RewardOutputFormat = "stdout_float"
    output_path: str | None = None
    score_key: str = "score"
    weight: float = 1.0
    timeout_s: float | None = None
    """Per-grader timeout; falls through to ``RewardContract.timeout_s`` when
    unset so simple manifests don't have to repeat the value per grader."""

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        if not v:
            raise ValueError("grader name must be non-empty")
        return v

    @field_validator("cmd")
    @classmethod
    def _validate_cmd(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if not v:
            raise ValueError("grader cmd must be non-empty")
        return v

    @model_validator(mode="after")
    def _validate_output_path(self) -> GraderSpec:
        if self.output_format == "json_file" and not self.output_path:
            raise ValueError(
                "grader.output_path is required when output_format=json_file"
            )
        return self


class RewardContract(BaseModel):
    """Per-template reward configuration (spec 02 RewardContract).

    Two forms for ``in_sandbox_final``, both shipped in Slice 4.5:

    - **Single-grader (convenience)**: set ``cmd`` + ``output_format`` etc.
      directly on the contract. Used for one-grader templates (terminal-bench
      ``test.sh``, simple shell scripts).
    - **Multi-grader (explicit)**: leave ``cmd`` empty and set ``graders`` to
      a list of :class:`GraderSpec`. Per-grader scores are kept; the
      ``aggregator`` (default ``weighted_sum``) computes the final reward.
      Used when a benchmark has independent sub-rewards (SWE-bench
      correctness vs regression safety; OSWorld task-completion vs
      trajectory quality; safety floors).

    The ``output_format`` knob (single-grader form) and per-:class:`GraderSpec`
    knob (multi-grader form) lets templates wire today's benchmark
    conventions without wrapper scripts:

    - terminal-bench-2 (`test.sh` exits 0/1) → ``output_format: exit_code``
    - OSWorld evaluators (write a JSON result file) → ``json_file``
    - SWE-bench harness (structured pytest report) → ``json_file`` or ``json_stdout``
    - bespoke shell graders that print one number → ``stdout_float`` (default)
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: str
    timeout_s: float = 60.0
    on_error: RewardOnError = "fail_rollout"

    # Single-grader convenience form.
    cmd: tuple[str, ...] = ()
    output_format: RewardOutputFormat = "stdout_float"
    output_path: str | None = None
    score_key: str = "score"

    # Multi-grader explicit form.
    graders: tuple[GraderSpec, ...] = ()
    aggregator: RewardAggregator = "weighted_sum"

    @field_validator("mode")
    @classmethod
    def _validate_mode(cls, v: str) -> str:
        if v not in REWARD_MODES:
            raise ValueError(
                f"reward.mode must be one of {sorted(REWARD_MODES)}; got {v!r}"
            )
        return v

    @model_validator(mode="after")
    def _validate_grader_shape(self) -> RewardContract:
        # Mutually exclusive: cmd vs graders.
        if self.cmd and self.graders:
            raise ValueError(
                "reward: 'cmd' and 'graders' are mutually exclusive — use one form"
            )
        # Single-grader json_file requires output_path.
        if self.output_format == "json_file" and not self.output_path and not self.graders:
            raise ValueError(
                "reward.output_path is required when output_format=json_file"
            )
        # Grader names must be unique.
        if self.graders:
            seen: set[str] = set()
            for g in self.graders:
                if g.name in seen:
                    raise ValueError(
                        f"reward.graders: duplicate grader name {g.name!r}"
                    )
                seen.add(g.name)
        # in_sandbox_final must have at least one grader.
        if self.mode == "in_sandbox_final" and not self.cmd and not self.graders:
            raise ValueError(
                "reward.mode=in_sandbox_final requires either 'cmd' (single-grader "
                "form) or 'graders' (multi-grader form)"
            )
        return self

    def effective_graders(self) -> tuple[GraderSpec, ...]:
        """Normalize single-grader convenience form into a one-element tuple
        of :class:`GraderSpec`. Multi-grader form returns ``self.graders``
        unchanged. Coordinator code only ever iterates this tuple — never
        branches on which form the manifest used.
        """
        if self.graders:
            return self.graders
        if not self.cmd:
            return ()
        return (
            GraderSpec(
                name="default",
                cmd=self.cmd,
                output_format=self.output_format,
                output_path=self.output_path,
                score_key=self.score_key,
                weight=1.0,
                timeout_s=None,
            ),
        )


class EnvAdapterDecl(BaseModel):
    """Manifest's ``env_adapter:`` block (spec 14)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    module: str
    class_name: str
    init_params: dict[str, Any] = Field(default_factory=dict)


class TemplateManifest(BaseModel):
    """Validated template manifest (spec 06 / spec 00 invariant 4).

    The manifest carries the **immutable benchmark contract** — name,
    version, adapter binding, instance resolver / asset list, reward
    contract. Per-experiment **policy** (deadlines, idle TTL, mounts,
    resource budgets, ``backend``, ``network``) is **not** part of
    the manifest; it lives in the user's run-config (see
    :class:`xrlenv.control.run_config.RunConfig`) and is merged at
    rollout time. ``backend`` and ``network`` were dropped from the
    schema in commit ``b5d602b`` — the loader rejects them with a
    pointer to the run-config.

    Pattern A plug-ins source per-task image / resources from the
    resolver; Pattern B and Simple templates carry ``image`` here
    because that's part of the benchmark identity. The runtime falls
    back to platform defaults from
    :mod:`xrlenv.control.defaults` for missing policy at rollout
    time.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    version: str
    digest: str
    image: str | None = None
    """Outer image. Required for Pattern B / Simple templates; may be
    omitted for Pattern A (the resolver supplies the per-task image)."""
    resources: ResourceSpec
    """Resource budget envelope. For Pattern A this is a placeholder —
    the resolver returns per-task resources that override this at
    rollout time; the manifest-level value is only used by the
    capacity estimator's per-template ceiling. Pattern B / Simple
    templates take resources from the run-config layer (filled into
    the manifest at load time)."""
    env_adapter: EnvAdapterDecl
    reward: RewardContract
    init_cmd: tuple[str, ...] = ()
    instances: InstanceResolverDecl | None = None
    """Spec 06 §"Pattern A": when set, the coordinator calls
    :func:`xrlenv.control.instance_resolver.load_resolver` to map each
    rollout's ``init.instance_id`` to a per-instance image / resources /
    mounts / init_params overlay. Outer template stays one manifest;
    benchmark suites with hundreds of per-task images don't need
    hundreds of templates."""
    image_builder: ImageBuilderDecl | None = None
    """P1.6 — control-plane-driven image builds. When set, the build
    coordinator can dispatch ``BuildImagesCommand`` to node-agents
    that load this builder via
    :func:`xrlenv.control.image_builder.load_image_builder` and run
    it in-process. Optional: plug-ins that don't ship a builder fall
    back to operator-side ``build-task-images.sh`` workflows."""
    assets: tuple[AssetSpec, ...] = ()
    """Spec 06 §"Pattern B": large external blobs (qcow2 disks, model
    checkpoints, dataset shards) tracked alongside images in the
    spec-15 cache. Each entry carries an integrity digest + size; the
    cache fetches via the registered :class:`AssetFetcher` for the
    URI scheme."""
    hard_s_default: float = 600.0
    ttl_default_s: float = 3600.0
    """Spec-09 GC layer-1 safety net (default 1 h).

    The DeadlineWatcher always arms the per-rollout hard-deadline timer at
    rollout start; if the consumer didn't pass an explicit
    :class:`Deadline.hard_s`, the coordinator falls back to
    ``min(hard_s_default, ttl_default_s)``. These platform-default
    values are used when the run-config doesn't supply explicit
    deadlines for the template; the run-config + per-rollout SDK
    kwargs override them when present.
    """
    init_timeout_s: float = 120.0
    setup_timeout_s: float = 60.0
    step_timeout_s: float = 30.0
    teardown_timeout_s: float = 30.0
    image_pin_mode: ImagePinMode = "registry_digest"
    """A1 / D20 (P1.2) — declares how this template's image bytes are
    identified across the cluster. ``registry_digest`` is the default
    (catalog pins centrally at register time); ``per_node_local`` skips
    central pinning for benchmarks built per-host (terminal-bench-2
    today); ``shared_storage`` is reserved for phase-2 NFS/S3 mounts.
    See :data:`ImagePinMode` for the full contract."""
    raw: dict[str, Any] = Field(default_factory=dict)

    def template_ref(self) -> TemplateRef:
        if self.image is None:
            raise ValueError(
                f"template {self.name!r}: image is None — Pattern A "
                "templates must have their image resolved per-task "
                "before template_ref() is called"
            )
        return TemplateRef(name=self.name, image=self.image, digest=self.digest)


# ──────────────────────────────────────────────────────────────────────────────
# Catalog
# ──────────────────────────────────────────────────────────────────────────────


ImageDigestResolver = Callable[[str], str | None]
"""``image_ref -> "sha256:abcd..."`` (or None if the resolver can't pin).

Spec 19 §"Image and asset supply chain": at register time the catalog
rewrites mutable tags into ``image@sha256:...`` so spec-00 invariant 4
(template manifests immutable for the duration of a training run) holds.
The runtime hands the catalog a Docker-backed resolver; tests hand a
canned mapping or ``None``.
"""

AuditCallback = Callable[[str, dict[str, Any]], None]
"""Spec 19 §"Audit logging" hook: ``(kind, payload) -> None``."""


# Spec 19 §"Bind mounts and shared caches": these prefixes are *always*
# refused, even when the operator overrides the allowlist. Templates that
# want a writable / readable host path under any of these are mis-built.
_DENIED_MOUNT_PREFIXES: tuple[str, ...] = (
    "/etc",
    "/root",
    "/home",
    "/proc",
    "/sys",
    "/var/run/docker.sock",
)
# /dev itself is denied except for the controlled set the runtime injects
# (per spec 19 §"Sandbox runtime hardening").
_ALLOWED_DEV_PATHS: frozenset[str] = frozenset(
    {"/dev/null", "/dev/zero", "/dev/random", "/dev/urandom", "/dev/tty"}
)


class MountDenied(ManifestInvalid):
    """Raised when a template's mount touches a spec-19 denied prefix."""


def _check_mount_allowlist(host_path: str) -> str | None:
    """Return None if the path is allowed; reason string otherwise.

    Public so tests + admin tooling can ask the same question without
    having to construct a TemplateManifest.
    """
    p = host_path.rstrip("/") or "/"
    for denied in _DENIED_MOUNT_PREFIXES:
        if p == denied or p.startswith(denied + "/"):
            return f"mount path {host_path!r} matches denied prefix {denied!r}"
    if (p == "/dev" or p.startswith("/dev/")) and p not in _ALLOWED_DEV_PATHS:
        return (
            f"mount path {host_path!r} is under /dev but not one of the "
            f"controlled set ({sorted(_ALLOWED_DEV_PATHS)})"
        )
    return None


def _is_pinned_by_digest(image_ref: str | None) -> bool:
    if image_ref is None:
        return False
    return "@sha256:" in image_ref


class TemplateCatalog:
    """In-memory catalog of registered templates.

    Templates can be registered from disk (``register_dir``) or programmatically
    (``register``). The latter is convenient for tests; the former is the
    canonical path operators use via ``xrlenv template register`` (Slice 3).

    Slice 8 (spec 19) added two register-time guards:

    - **Mount allowlist** — :py:func:`_check_mount_allowlist` rejects any
      ``ResourceSpec.mounts`` entry pointing at the denied prefixes
      (``/etc`` etc.). Audit event ``mount.denied`` fires before the
      :class:`MountDenied` raise.
    - **Image digest pinning** — when ``digest_resolver`` is wired, the
      catalog asks it for ``sha256:...`` and rewrites ``image:tag`` into
      ``image@sha256:...``; tag-only refs registered without a resolver
      log a warning + audit event ``template.image_unpinned`` so the
      operator can fix them in the next iteration.
    """

    def __init__(
        self,
        *,
        digest_resolver: ImageDigestResolver | None = None,
        audit_callback: AuditCallback | None = None,
    ) -> None:
        self._templates: dict[str, TemplateManifest] = {}
        self._digest_resolver = digest_resolver
        self._audit = audit_callback

    def register(self, manifest: TemplateManifest) -> TemplateManifest:
        """Validate + (optionally) pin + store. Returns the registered
        manifest after digest rewriting (so callers see the pinned form).
        """
        self._validate_mounts(manifest)
        manifest = self._maybe_pin_image(manifest)
        self._templates[manifest.name] = manifest
        if self._audit is not None:
            self._audit(
                "template.registered",
                {
                    "name": manifest.name,
                    "version": manifest.version,
                    "digest": manifest.digest,
                    "image": manifest.image,
                    "pinned_by_digest": _is_pinned_by_digest(manifest.image),
                },
            )
        return manifest

    def register_dir(self, root: Path) -> list[TemplateManifest]:
        """Walk ``root`` and register every ``manifest.yaml`` found."""
        registered: list[TemplateManifest] = []
        for yaml_path in sorted(root.rglob("manifest.yaml")):
            manifest = load_manifest(yaml_path)
            registered.append(self.register(manifest))
        return registered

    def register_paths(self, paths: Iterable[Path]) -> list[TemplateManifest]:
        """Register an explicit list of ``manifest.yaml`` files.

        Used by the plug-in discovery path
        (:func:`xrlenv.control.template_discovery.find_plugin_manifest_files`)
        to skip the rglob walk: each plug-in lives at a known
        canonical depth (``xrlenv_plugins/<cat>/<name>/manifest.yaml``)
        so the discovery returns the paths directly.
        """
        registered: list[TemplateManifest] = []
        for yaml_path in paths:
            manifest = load_manifest(yaml_path)
            registered.append(self.register(manifest))
        return registered

    def get(self, name: str) -> TemplateManifest:
        if name not in self._templates:
            raise KeyError(f"unknown template {name!r}")
        return self._templates[name]

    def validate_overlay(self, manifest: TemplateManifest) -> TemplateManifest:
        """Run register-time security checks on a post-resolver overlay.

        Pattern A manifests omit ``image`` and let the resolver provide
        per-task image / resources / mounts at rollout time. The
        register-time validators in :py:meth:`register` (mount allowlist
        + image digest pinning) skip Pattern-A manifests because the
        fields they check aren't populated yet. This method runs the
        same validators on a fully-overlaid manifest — call it from the
        coordinator after :func:`xrlenv.control.instance_resolver.apply_to_manifest`
        but before placement / sandbox creation.

        Returns the (possibly digest-rewritten) manifest. Raises
        :class:`MountDenied` when a resolver-supplied mount touches a
        spec-19 denied prefix; logs a warning + audit event when an
        unpinned image makes it past the resolver.
        """
        self._validate_mounts(manifest)
        return self._maybe_pin_image(manifest)

    def list(self) -> list[TemplateManifest]:
        return list(self._templates.values())

    # ── Slice 8 (spec 19) helpers ────────────────────────────────────────────

    def _validate_mounts(self, manifest: TemplateManifest) -> None:
        for mount in manifest.resources.mounts or ():
            reason = _check_mount_allowlist(mount.host_path)
            if reason is None:
                continue
            if self._audit is not None:
                self._audit(
                    "mount.denied",
                    {
                        "template": manifest.name,
                        "host_path": mount.host_path,
                        "sandbox_path": mount.sandbox_path,
                        "reason": reason,
                    },
                )
            raise MountDenied(
                f"template {manifest.name!r}: {reason}"
            )

    def _maybe_pin_image(self, manifest: TemplateManifest) -> TemplateManifest:
        """Rewrite ``image:tag`` → ``image@sha256:...`` when possible.

        Returns the (possibly mutated) manifest. Mutates via
        :py:meth:`pydantic.BaseModel.model_copy` because TemplateManifest
        is frozen.
        """
        if manifest.image is None:
            # Pattern A — resolver supplies the per-task image; nothing
            # for the catalog to pin at registration time.
            return manifest
        if manifest.image_pin_mode != "registry_digest":
            # A1 / D20 (P1.2) — manifests declared as ``per_node_local``
            # or ``shared_storage`` are NOT centrally pinned; the bytes
            # are identified by the destination node (per-node-local) or
            # the shared-storage layer's content hash. Skip the
            # digest-resolver path entirely so we don't accidentally pin
            # to a control-plane-local digest that no other node has
            # (the buildx local-only ``RepoDigests`` trap that the
            # 5a38e78 symptom-fix patched). The audit event still fires
            # so spec-19 can account for the registration with an
            # accurate ``digest_source`` field.
            if self._audit is not None:
                self._audit(
                    "template.image_unpinned",
                    {
                        "template": manifest.name,
                        "image": manifest.image,
                        "reason": "image_pin_mode",
                        "image_pin_mode": manifest.image_pin_mode,
                        "digest_source": (
                            "per_node" if manifest.image_pin_mode == "per_node_local"
                            else "shared_storage"
                        ),
                    },
                )
            return manifest
        if _is_pinned_by_digest(manifest.image):
            return manifest
        if self._digest_resolver is None:
            LOGGER.warning(
                "template %r: image %r is not pinned by digest and no "
                "digest_resolver is wired; the manifest is registered as-is "
                "(spec 19 §\"Image and asset supply chain\" — fix the "
                "manifest or wire a Docker-backed resolver in production)",
                manifest.name, manifest.image,
            )
            if self._audit is not None:
                self._audit(
                    "template.image_unpinned",
                    {"template": manifest.name, "image": manifest.image},
                )
            return manifest
        try:
            digest = self._digest_resolver(manifest.image)
        except Exception:
            LOGGER.exception(
                "template %r: digest resolver raised on image %r; "
                "registering unpinned",
                manifest.name, manifest.image,
            )
            if self._audit is not None:
                self._audit(
                    "template.image_unpinned",
                    {"template": manifest.name, "image": manifest.image,
                     "reason": "resolver_error"},
                )
            return manifest
        if digest is None:
            # Intentional: resolver checked and found no registry-
            # resolvable digest (the image is locally-built per node
            # — see ``DockerBackend.resolve_image_digest`` and the
            # ``buildx local-only RepoDigests`` discussion under D20).
            # This is the right outcome for per-node-built images;
            # the audit event still fires so spec-19 can account for
            # the unpinned registration, but log at DEBUG because the
            # operator has no action to take. (D20 will replace this
            # with an explicit ``image_pin_mode=per_node_local`` flag
            # so the resolver doesn't even get called for those.)
            LOGGER.debug(
                "template %r: digest resolver returned None for image %r; "
                "registering unpinned (no registry-resolvable digest — "
                "expected for locally-built per-node images)",
                manifest.name, manifest.image,
            )
            if self._audit is not None:
                self._audit(
                    "template.image_unpinned",
                    {"template": manifest.name, "image": manifest.image,
                     "reason": "resolver_returned_none"},
                )
            return manifest
        if not digest.startswith("sha256:"):
            # The resolver returned something that isn't None and isn't
            # a sha256 digest — that's a resolver-implementation bug,
            # not an unpinnable image. Loud warning to surface it.
            LOGGER.warning(
                "template %r: digest resolver returned %r; registering "
                "unpinned (resolver should return 'sha256:<hex>' or None)",
                manifest.name, digest,
            )
            if self._audit is not None:
                self._audit(
                    "template.image_unpinned",
                    {"template": manifest.name, "image": manifest.image,
                     "reason": "resolver_returned_malformed_digest"},
                )
            return manifest
        # Strip the tag (everything after the last ``:`` that isn't part of
        # a port number). Pinned form: ``<repo>@sha256:...``.
        repo = manifest.image.split("@", 1)[0].rsplit(":", 1)[0]
        pinned = f"{repo}@{digest}"
        return manifest.model_copy(update={"image": pinned})


# ──────────────────────────────────────────────────────────────────────────────
# Loader / validator
# ──────────────────────────────────────────────────────────────────────────────


def load_manifest(path: Path) -> TemplateManifest:
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ManifestInvalid(f"{path}: top-level must be a mapping")
    try:
        return _manifest_from_raw(raw, manifest_dir=path.parent.resolve())
    except ManifestInvalid:
        raise
    except Exception as exc:
        raise ManifestInvalid(f"{path}: {exc}") from exc


def _manifest_from_raw(
    raw: dict[str, Any],
    *,
    manifest_dir: Path | None = None,
) -> TemplateManifest:
    name = _require_str(raw, "name")
    # The manifest is the immutable benchmark contract. ``backend`` and
    # ``network`` are user-side policy — different operators run
    # different hardware (KVM-or-not), and per-task tasks may legitimately
    # want different network policy — so they live in the run-config, not
    # here. Reject them with a precise pointer instead of silently honouring
    # a transitional fallback the runtime no longer reads.
    for forbidden, where in (
        ("backend", "manifests.<name>.backend in the run-config"),
        ("network", "manifests.<name>.network in the run-config (or per-task "
                    "via the resolver for Pattern A)"),
    ):
        if forbidden in raw:
            raise ManifestInvalid(
                f"template {name!r}: {forbidden!r} is not a manifest field "
                f"(it's user-side policy). Move it to {where}."
            )

    version = str(raw.get("version") or "0.1")

    has_instances = bool(raw.get("instances"))
    image: str | None
    if "image" in raw and raw["image"] is not None:
        image = _require_str(raw, "image")
    elif has_instances:
        # Pattern A: per-task image from the resolver.
        image = None
    else:
        raise ManifestInvalid(
            f"template {name!r}: 'image' is required for templates without an "
            "'instances:' block (Pattern A plug-ins source images per-task "
            "via the resolver instead)"
        )

    # Per-experiment policy (resources / deadlines) is no longer part
    # of the manifest contract. If absent we fall back to conservative
    # platform defaults so existing capacity / scheduler code paths keep
    # working; the run-config layer overrides at rollout time.
    resources_block = raw.get("resources") or {}
    resources = _build_resources(resources_block)

    env_adapter = _build_adapter(raw.get("env_adapter") or {})
    reward = _build_reward(raw.get("reward") or {"mode": "env_step"})

    init_block = raw.get("init") or {}
    init_cmd = tuple(init_block.get("cmd") or ())

    deadlines = raw.get("deadlines") or {}
    hard_s = float(deadlines.get("hard_s_default") or 600.0)
    ttl_default = float(deadlines.get("ttl_default_s") or 3600.0)

    instances_block = raw.get("instances")
    instances_decl: InstanceResolverDecl | None = None
    if instances_block:
        if not isinstance(instances_block, dict):
            raise ManifestInvalid(
                "manifest field 'instances' must be a mapping with "
                "'module' + 'class' (+ optional index_path / options)"
            )
        try:
            instances_decl = InstanceResolverDecl.model_validate(instances_block)
        except Exception as exc:
            raise ManifestInvalid(
                f"manifest field 'instances' is malformed: {exc}"
            ) from exc
        # Audit M3: anchor a relative ``index_path`` to the manifest's
        # directory at load time so the resolver always sees an
        # absolute path. Without this, ``./tasks/`` resolves against
        # the operator's CWD when ``xrlenv up`` runs — which makes
        # behaviour depend on where the daemon was launched and turns
        # the manifest-local path into a trap.
        if (
            instances_decl.index_path
            and manifest_dir is not None
            and not Path(instances_decl.index_path).is_absolute()
        ):
            absolute = (manifest_dir / instances_decl.index_path).resolve()
            instances_decl = instances_decl.model_copy(
                update={"index_path": str(absolute)},
            )

    image_builder_block = raw.get("image_builder")
    image_builder_decl: ImageBuilderDecl | None = None
    if image_builder_block:
        if not isinstance(image_builder_block, dict):
            raise ManifestInvalid(
                "manifest field 'image_builder' must be a mapping with "
                "'module' + 'class'"
            )
        try:
            image_builder_decl = ImageBuilderDecl.model_validate(image_builder_block)
        except Exception as exc:
            raise ManifestInvalid(
                f"manifest field 'image_builder' is malformed: {exc}"
            ) from exc

    assets_block = raw.get("assets") or ()
    if assets_block and not isinstance(assets_block, list):
        raise ManifestInvalid(
            "manifest field 'assets' must be a list of asset declarations"
        )
    asset_specs: list[AssetSpec] = []
    for entry in assets_block:
        if not isinstance(entry, dict):
            raise ManifestInvalid(
                f"manifest 'assets' entry must be a mapping; got {entry!r}"
            )
        try:
            asset_specs.append(AssetSpec.model_validate(entry))
        except Exception as exc:
            raise ManifestInvalid(
                f"manifest 'assets' entry is malformed: {exc}"
            ) from exc

    digest = _compute_digest(raw)

    image_pin_mode_raw = raw.get("image_pin_mode") or "registry_digest"
    if image_pin_mode_raw not in (
        "registry_digest", "per_node_local", "shared_storage",
    ):
        raise ManifestInvalid(
            f"manifest field 'image_pin_mode' must be one of "
            f"'registry_digest', 'per_node_local', 'shared_storage'; "
            f"got {image_pin_mode_raw!r}"
        )
    image_pin_mode = cast(ImagePinMode, image_pin_mode_raw)

    return TemplateManifest(
        name=name,
        version=version,
        digest=digest,
        image=image,
        resources=resources,
        env_adapter=env_adapter,
        reward=reward,
        init_cmd=init_cmd,
        instances=instances_decl,
        image_builder=image_builder_decl,
        assets=tuple(asset_specs),
        hard_s_default=hard_s,
        ttl_default_s=ttl_default,
        init_timeout_s=float(deadlines.get("init_timeout_s") or 120.0),
        setup_timeout_s=float(deadlines.get("setup_timeout_s") or 60.0),
        step_timeout_s=float(deadlines.get("step_timeout_s") or 30.0),
        teardown_timeout_s=float(deadlines.get("teardown_timeout_s") or 30.0),
        image_pin_mode=image_pin_mode,
        raw=raw,
    )


def _require_str(raw: dict[str, Any], key: str) -> str:
    val = raw.get(key)
    if not isinstance(val, str) or not val:
        raise ManifestInvalid(f"manifest field {key!r} must be a non-empty string")
    return val


def _build_resources(raw: dict[str, Any]) -> ResourceSpec:
    cpu_request = float(raw.get("cpu_request") or 0.25)
    cpu_limit = float(raw.get("cpu_limit") or max(cpu_request, 1.0))
    mem_request = _parse_bytes(raw.get("mem_request") or "256Mi")
    mem_limit = _parse_bytes(raw.get("mem_limit") or "1Gi")
    disk_request = _parse_bytes(raw.get("disk_request") or "256Mi")
    gpu_required = bool(raw.get("gpu_required") or False)
    mounts = tuple(_build_mount(m) for m in (raw.get("mounts") or ()))
    return ResourceSpec(
        cpu_request=cpu_request,
        cpu_limit=cpu_limit,
        mem_request_bytes=mem_request,
        mem_limit_bytes=mem_limit,
        disk_request_bytes=disk_request,
        gpu_required=gpu_required,
        mounts=mounts,
    )


def _build_mount(raw: dict[str, Any]) -> MountSpec:
    return MountSpec(
        host_path=_require_str(raw, "host_path"),
        sandbox_path=_require_str(raw, "sandbox_path"),
        readonly=bool(raw.get("readonly", True)),
    )


def _build_adapter(raw: dict[str, Any]) -> EnvAdapterDecl:
    module = _require_str(raw, "module")
    class_name = _require_str(raw, "class")
    init_params = raw.get("init_params") or {}
    if not isinstance(init_params, dict):
        raise ManifestInvalid("env_adapter.init_params must be a mapping")
    return EnvAdapterDecl(module=module, class_name=class_name, init_params=init_params)


def _build_reward(raw: dict[str, Any]) -> RewardContract:
    mode = str(raw.get("mode") or "env_step")
    if mode not in REWARD_MODES:
        raise ManifestInvalid(
            f"reward.mode must be one of {sorted(REWARD_MODES)}; got {mode!r}"
        )
    graders_raw = raw.get("graders") or ()
    graders = tuple(_build_grader(g) for g in graders_raw)
    return RewardContract(
        mode=mode,
        cmd=tuple(raw.get("cmd") or ()),
        timeout_s=float(raw.get("timeout_s") or 60.0),
        on_error=cast(RewardOnError, str(raw.get("on_error") or "fail_rollout")),
        output_format=cast(
            RewardOutputFormat, str(raw.get("output_format") or "stdout_float")
        ),
        output_path=raw.get("output_path"),
        score_key=str(raw.get("score_key") or "score"),
        graders=graders,
        aggregator=cast(
            RewardAggregator, str(raw.get("aggregator") or "weighted_sum")
        ),
    )


def _build_grader(raw: dict[str, Any]) -> GraderSpec:
    return GraderSpec(
        name=_require_str(raw, "name"),
        cmd=tuple(raw.get("cmd") or ()),
        output_format=cast(
            RewardOutputFormat, str(raw.get("output_format") or "stdout_float")
        ),
        output_path=raw.get("output_path"),
        score_key=str(raw.get("score_key") or "score"),
        weight=float(raw.get("weight") or 1.0),
        timeout_s=(
            float(raw["timeout_s"]) if raw.get("timeout_s") is not None else None
        ),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


_BYTE_PATTERN = re.compile(r"^\s*([0-9.]+)\s*([KMGT]i?B?|B)?\s*$", re.IGNORECASE)
_SI_FACTORS: dict[str, int] = {
    "B": 1,
    "KB": 1000, "MB": 1000**2, "GB": 1000**3, "TB": 1000**4,
    "KIB": 1024, "MIB": 1024**2, "GIB": 1024**3, "TIB": 1024**4,
    "KI": 1024, "MI": 1024**2, "GI": 1024**3, "TI": 1024**4,
    "K": 1000, "M": 1000**2, "G": 1000**3, "T": 1000**4,
}


def _parse_bytes(value: Any) -> int:
    """Parse spec-01 byte strings (``"8GB"`` SI, ``"4Gi"`` binary).

    Integers pass through unchanged; strings are matched against the regex
    and multiplied by the appropriate factor.
    """
    if isinstance(value, int):
        return value
    if not isinstance(value, str):
        raise ManifestInvalid(f"byte value must be int or str; got {type(value).__name__}")
    match = _BYTE_PATTERN.match(value)
    if not match:
        raise ManifestInvalid(f"could not parse byte value {value!r}")
    qty_s, suffix = match.groups()
    suffix = (suffix or "B").upper()
    factor = _SI_FACTORS.get(suffix)
    if factor is None:
        raise ManifestInvalid(f"unknown byte suffix {suffix!r} in {value!r}")
    return int(float(qty_s) * factor)


def _compute_digest(raw: dict[str, Any]) -> str:
    """Stable sha256 of the canonicalised manifest body (spec 00 invariant 4)."""
    body = json.dumps(raw, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(body.encode("utf-8")).hexdigest()


__all__ = [
    "EnvAdapterDecl",
    "GraderSpec",
    "RewardAggregator",
    "RewardContract",
    "RewardOnError",
    "RewardOutputFormat",
    "TemplateCatalog",
    "TemplateManifest",
    "load_manifest",
]


def parse_bytes(value: Any) -> int:
    """Public re-export of the byte parser for test convenience."""
    return _parse_bytes(value)


def collect_manifests(root: Path) -> Iterable[TemplateManifest]:
    """Generator form of :py:meth:`TemplateCatalog.register_dir`."""
    for yaml_path in sorted(root.rglob("manifest.yaml")):
        yield load_manifest(yaml_path)
