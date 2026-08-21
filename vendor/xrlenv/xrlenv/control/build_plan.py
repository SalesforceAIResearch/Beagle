"""``build-plan.yaml`` schema — two shapes coexisting.

The original P1.6 shape was **benchmark-driven**: top-level
``benchmarks: list[BenchmarkBuildSpec]`` where each entry names a
manifest (``swebench-verified``, ``terminal-bench-2``) and a
selection (smoke / instances / all). Dispatch ran via the
benchmark's registered ``BenchmarkImageBuilder``. P1.7.D deleted
the in-tree builders, so the benchmark-driven path is dormant
(but kept for any external plug-in that still registers a
builder).

The P1.7.C.2 shape is **per-image-ref**: top-level
``entries: list[BuildEntry]`` where each entry describes one
image to materialize, with a ``context_source`` discriminated
union (``registry`` / ``git`` / ``tarball``). This shape is
benchmark-agnostic — the schema, RPC, and node-side handler
don't know about harbor / tb2 / swebench. Adapters and
per-benchmark generators emit YAML in this shape; the cluster
materializes whatever the YAML says.

Both shapes share top-level ``budget`` + ``replication`` knobs
and route through the same ``image_planner.plan_opportunistic_placements``
bin-packer.

Per-image-ref schema example::

    version: 1
    replication: 1
    budget:
      reserved_runtime_gb: 30
      buffer_gb: 10
    entries:
      - image_ref: alexgshaw/fix-git:20251031
        context_source: { type: registry }
        placement:
          preferred_home_count: 1
          size_hint_bytes: 1500000000
          size_hint_source: registry-probe

      - image_ref: hb__seta-task-0
        context_source:
          type: git
          repo: https://github.com/camel-ai/seta-env
          ref: main
          subdir: Harbor-Dataset/0/environment
          dockerfile: Dockerfile
        placement:
          preferred_home_count: 1
          size_hint_bytes: 4500000000
          size_hint_source: heuristic
        pinned: false                         # never evict if true
        priority: 0                            # higher = build first when budget tight

Benchmark-driven schema example (legacy, still supported)::

    version: 1
    replication: 1
    budget: { reserved_runtime_gb: 30, buffer_gb: 10 }
    benchmarks:
      - name: swebench-verified
        selection: { smoke: true }

Validator rejects mixing the two shapes — ``entries`` and
``benchmarks`` are mutually exclusive.

Decisions locked (2026-05-08 design pass):

- ``size_hint_source`` distinguishes ``registry-probe`` (accurate),
  ``cluster-reported`` (post-build authoritative), and ``heuristic``
  (estimated; bin-packer adds safety margin).
- ``plan_id`` content-addresses the canonicalised plan; re-applying
  the same YAML is a no-op.
- ``pinned: true`` flows through to ``ImageCacheManager.pin`` after
  successful build; pinned images skip eviction.
- Plan-apply does a pin-budget sanity check upfront — refuses to
  dispatch if total pinned bytes per node exceeds the per-node
  image-cache budget.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from xrlenv.errors import ManifestInvalid


class BuildBudget(BaseModel):
    """Per-node disk budget knobs.

    The bin-packer subtracts ``reserved_runtime_gb`` (running
    containers + scratch) and ``buffer_gb`` (margin before LRU
    evict kicks in) from each node's free disk, then places images
    against what remains. ``cap_per_node_gb`` overrides the budget
    upward when the operator wants to cap usage on, e.g., a small
    test VM.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    reserved_runtime_gb: int = 30
    buffer_gb: int = 10
    cap_per_node_gb: int | None = None


class BenchmarkSelection(BaseModel):
    """One-of-three selection: smoke, explicit instances, or all.

    Exactly one field must be set (or implicitly truthy). The schema
    rejects multi-select to avoid ambiguous selection semantics.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    smoke: bool = False
    instances: tuple[str, ...] = ()
    all: bool = False

    @model_validator(mode="after")
    def _exactly_one(self) -> BenchmarkSelection:
        present = sum(
            1 for v in (self.smoke, bool(self.instances), self.all) if v
        )
        if present != 1:
            raise ValueError(
                "selection must set exactly one of "
                "'smoke: true' / 'instances: [...]' / 'all: true'",
            )
        return self

    def to_kwargs(self) -> dict[str, Any]:
        """Lower to the dict shape ``BenchmarkImageBuilder.enumerate_image_refs``
        expects."""
        if self.smoke:
            return {"smoke": True}
        if self.instances:
            return {"instances": list(self.instances)}
        return {"all": True}


class BenchmarkBuildSpec(BaseModel):
    """One benchmark's slot in the cluster-wide plan.

    ``replication`` is optional per-benchmark override; absence means
    "use the plan's top-level default." Per-image override is reserved
    for a phase-2 follow-on.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    """Manifest name (e.g. ``swebench-verified``). Must match a
    template registered in the catalog with an ``image_builder``
    block; the coordinator rejects plans referencing unknown or
    builder-less benchmarks before any dispatch."""

    selection: BenchmarkSelection
    """Which images to build for this benchmark."""

    build_path: str | None = None
    """Plug-in-specific build mode (e.g. ``pull-and-retag`` or
    ``build-locally`` for swebench-verified). Forwarded as a kwarg
    to the builder's ``build(...)`` call. ``None`` lets the builder
    pick its default."""

    replication: int | None = None
    """Per-benchmark replication override. ``None`` defers to the
    plan-level default. Must be >=1 when set."""

    @model_validator(mode="after")
    def _validate(self) -> BenchmarkBuildSpec:
        if self.replication is not None and self.replication < 1:
            raise ValueError("replication must be >= 1")
        if not self.name:
            raise ValueError("benchmark.name must be non-empty")
        return self


SizeHintSource = Literal["registry-probe", "cluster-reported", "heuristic"]
"""Provenance of an entry's ``size_hint_bytes``.

- ``registry-probe`` — sum of unique layer sizes from the registry
  manifest API at generation time. Accurate for ``type: registry``
  entries.
- ``cluster-reported`` — actual size after the cluster built or
  pulled the image (from ``ReportImagesCommand``). Authoritative.
- ``heuristic`` — estimated (e.g. base-image size x 1.3 or generic
  default). Bin-packer adds safety margin; ``xrlenv build calibrate``
  promotes to ``cluster-reported`` after first successful build.
"""


class RegistrySource(BaseModel):
    """Pull from a Docker registry. No build required."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["registry"] = "registry"


class GitSource(BaseModel):
    """Build from a Dockerfile in a git repo.

    Cluster nodes clone the repo (cached at
    ``~/.xrlenv/build-context-cache/<repo-hash>/<ref>/``) and run
    ``docker build -f <dockerfile> <subdir>``. Re-builds against the
    same ref are cache-hits.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["git"] = "git"
    repo: str
    """Git URL — ``https://...`` or ``git@...``. SSH URLs require
    operator-side credentials on each data-plane node."""

    ref: str = "main"
    """Branch, tag, or commit sha. Cluster fetches this exact ref."""

    subdir: str = "."
    """Path within the repo that's the docker build context.
    Defaults to repo root."""

    dockerfile: str = "Dockerfile"
    """Dockerfile name within ``subdir``. Defaults to ``Dockerfile``."""


class TarballSource(BaseModel):
    """Build from a docker context tarball shipped to the node.

    The operator's CLI loads the tarball at apply time and ships
    bytes via ``BuildImageCommand``. The CLI enforces a size cap
    operator-side (default 100 MB, tunable via
    ``xrlenv build apply --build-tarball-max-bytes``) so oversized
    payloads reject before any wire traffic — operators iterate
    locally rather than failing mid-cluster on a too-big context.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["tarball"] = "tarball"
    path: str
    """Operator-local path to a ``.tar`` / ``.tar.gz`` build context.
    The CLI loads + ships; the path itself is never sent to the
    cluster, only the bytes."""

    dockerfile: str = "Dockerfile"
    """Dockerfile name relative to the tarball root."""

    content_b64: str | None = None
    """Apply-time-only payload: base64-encoded bytes of the tarball
    at ``path``. Populated by :func:`resolve_tarball_sources` (called
    by the CLI immediately after YAML load) and consumed by the
    wire transport. **Never set in the YAML.** Excluded from
    :func:`compute_plan_id` so the plan_id reflects operator intent
    (the YAML), not which bytes happen to be on disk at apply time
    — re-running ``xrlenv build apply`` with an unchanged Dockerfile
    is a no-op even though the bytes are re-loaded."""


class LocalSource(BaseModel):
    """Build from a Dockerfile in a directory that **already exists on the
    build host** — no clone, no extract, no byte-shipping. The build tool
    ``docker build``s ``path`` in place.

    This is the least-lossy, least-stateful source: git and tarball both exist
    only to *materialize* a directory before ``docker build``; when the build
    context is already a directory on the host, using it directly skips that
    copy entirely. seta-env clones its repo into the shared FSx
    ``build-context-cache`` first; a harbor task cache is *already* a directory
    tree on shared FSx, so ``local`` builds it where it sits.

    The trade-off is portability: a local path only resolves on hosts where that
    exact directory is mounted. That is correct ONLY on a cluster-shared
    filesystem — every build node sees the same path — so ``shared_fs`` is
    REQUIRED. It names the shared-fs topology (e.g. ``hyperpod``) that
    guarantees the path on every build node, serving both as a machine-readable
    assertion and as operator-visible documentation of *why* a bare local path
    is safe here. The Slurm build fan-out
    (``slurm_scripts/build_and_push_images.sh``) already relies on exactly this
    property ("any nodes with docker + the shared FSx home").

    **Build-host-only.** This source is consumed by
    ``scripts/build_and_push_images.py`` (run directly or Slurm-sharded). The
    cluster ``xrlenv build apply`` path — which ships sources to nodes that may
    not share the path — rejects ``local`` entries; build them on a shared-fs
    build host and apply a registry-source plan instead.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["local"] = "local"
    path: str
    """Absolute path to a directory on the build host that IS the docker build
    context (the Dockerfile and everything it ``COPY``s live here). Must resolve
    identically on every build node — hence ``shared_fs``."""

    dockerfile: str = "Dockerfile"
    """Dockerfile name within ``path``. Defaults to ``Dockerfile``."""

    shared_fs: str
    """REQUIRED. Names the cluster-shared filesystem / topology that guarantees
    ``path`` is mounted identically on every build node (e.g. ``hyperpod``).
    Non-empty by validation — you cannot declare a local source without stating
    the shared-fs assumption that makes a bare local path safe."""

    @model_validator(mode="after")
    def _validate(self) -> LocalSource:
        if not self.path:
            raise ValueError("local source path must be non-empty")
        if not self.shared_fs:
            raise ValueError(
                "local source requires shared_fs naming the cluster-shared "
                "filesystem that guarantees `path` is mounted on every build "
                "node (e.g. shared_fs: hyperpod). A local path is not portable; "
                "this field is the explicit assertion that it resolves "
                "cluster-wide.",
            )
        return self


# Discriminated union — pydantic picks the right model from the ``type`` tag.
ContextSource = RegistrySource | GitSource | TarballSource | LocalSource


class EntryPlacement(BaseModel):
    """Placement hints for one image entry.

    ``preferred_home_count`` is a soft preference — the bin-packer
    tries to land copies on N nodes. Under disk pressure copies can
    evict; build-on-acquire backfills if the image is needed again.
    Pair with ``pinned: true`` for hard persistence.

    ``size_hint_bytes`` feeds FFD bin-packing. Source-typed via
    ``size_hint_source``; bin-packer adds safety margin to
    ``heuristic``-tagged entries.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    preferred_home_count: int = 1
    size_hint_bytes: int
    size_hint_source: SizeHintSource = "heuristic"

    @model_validator(mode="after")
    def _validate(self) -> EntryPlacement:
        if self.preferred_home_count < 1:
            raise ValueError("preferred_home_count must be >= 1")
        if self.size_hint_bytes < 0:
            raise ValueError("size_hint_bytes must be >= 0")
        return self


class BuildEntry(BaseModel):
    """One image's slot in a per-image-ref build plan.

    Schema-agnostic about the benchmark — the entry says "build /
    pull this image_ref via this context_source." Adapters and
    per-benchmark generators emit YAML in this shape; the cluster
    has no benchmark-specific dispatch.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    image_ref: str
    """Final image tag the cluster will materialize. Matches the
    string consumers will pass to ``acquire_container(image=...)``."""

    context_source: ContextSource = Field(discriminator="type")
    placement: EntryPlacement

    pinned: bool = False
    """When True, the resulting image is pinned in
    ``ImageCacheManager`` and never evicts until the operator
    unpins or the image is explicitly removed."""

    priority: int = 0
    """Higher = built first when the plan exceeds cluster budget.
    Ties broken by entry order. Default 0."""

    labels: dict[str, str] = Field(default_factory=dict)
    """Extra Docker labels applied to the built image. The cluster
    automatically adds ``xrlenv.image.rebuild-cost`` based on
    ``context_source.type`` (registry → cheap, git → expensive,
    tarball → medium); operator labels merge on top, but the
    rebuild-cost label is reserved and won't be overridden."""

    @model_validator(mode="after")
    def _validate(self) -> BuildEntry:
        if not self.image_ref:
            raise ValueError("image_ref must be non-empty")
        return self


class BuildPlan(BaseModel):
    """Cluster-wide build plan.

    Two coexisting shapes. Set EXACTLY ONE of:

    - ``benchmarks`` (legacy P1.6) — list of ``BenchmarkBuildSpec``.
      Routes through registered ``BenchmarkImageBuilder``s.
    - ``entries`` (P1.7.C.2) — list of ``BuildEntry``, per-image-ref,
      benchmark-agnostic. Routes through generic ``BuildImageCommand``.

    ``replication`` is the default applied to legacy ``benchmarks``
    entries; per-image-ref ``entries`` use ``EntryPlacement.preferred_home_count``
    instead. ``budget`` carves disk off each node before placement
    in both shapes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: Literal[1] = 1
    name: str | None = None
    """Operator-friendly label (e.g. ``terminal-bench-2-phase-0``).
    Shown in the admin ``/builds`` panel so plan rows associate
    with their YAML files at a glance instead of by ``plan_id``.
    Excluded from ``compute_plan_id`` so renaming a plan doesn't
    change its content-hash and trigger a fresh dispatch.
    Optional; missing names render as ``(unnamed)`` in the panel."""
    replication: int = 1
    budget: BuildBudget = Field(default_factory=BuildBudget)
    benchmarks: tuple[BenchmarkBuildSpec, ...] = ()
    entries: tuple[BuildEntry, ...] = ()

    @model_validator(mode="after")
    def _validate(self) -> BuildPlan:
        if self.replication < 1:
            raise ValueError("plan.replication must be >= 1")
        has_bench = bool(self.benchmarks)
        has_entries = bool(self.entries)
        if has_bench and has_entries:
            raise ValueError(
                "plan.benchmarks and plan.entries are mutually exclusive — "
                "use one shape or the other, not both",
            )
        if not has_bench and not has_entries:
            raise ValueError(
                "plan must contain either 'benchmarks' (legacy) or "
                "'entries' (per-image-ref)",
            )
        if has_bench:
            seen_b: set[str] = set()
            for b in self.benchmarks:
                if b.name in seen_b:
                    raise ValueError(
                        f"benchmark {b.name!r} listed twice; the planner "
                        "expects one entry per benchmark — combine selections "
                        "in a single block",
                    )
                seen_b.add(b.name)
        if has_entries:
            seen_e: set[str] = set()
            for e in self.entries:
                if e.image_ref in seen_e:
                    raise ValueError(
                        f"image_ref {e.image_ref!r} listed twice; merge into a "
                        "single entry (use placement.preferred_home_count for "
                        "replication)",
                    )
                seen_e.add(e.image_ref)
        return self

    def replication_for(self, benchmark: str) -> int:
        """Legacy lookup; returns plan-default for unknown benchmarks
        when the plan is in per-image-ref mode."""
        for b in self.benchmarks:
            if b.name == benchmark:
                return b.replication or self.replication
        if self.entries:
            return self.replication
        raise KeyError(benchmark)

    def is_per_image_ref(self) -> bool:
        return bool(self.entries)


def load_build_plan(path: Path) -> BuildPlan:
    """Read + validate a ``build-plan.yaml`` from disk.

    Raises :class:`ManifestInvalid` with a useful message on schema
    failures — same error class as ``load_manifest`` for consistency
    with the catalog's existing error surface.
    """
    if not path.is_file():
        raise ManifestInvalid(f"build plan not found at {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ManifestInvalid(
            f"build-plan.yaml at {path} must be a mapping; got {type(raw).__name__}",
        )
    try:
        return BuildPlan.model_validate(raw)
    except Exception as exc:
        raise ManifestInvalid(f"build-plan.yaml at {path} is malformed: {exc}") from exc


def resolve_tarball_sources(
    plan: BuildPlan, *, max_bytes: int,
    base_dir: Path | None = None,
) -> BuildPlan:
    """Load operator-side tarball bytes into every ``TarballSource``
    entry's ``content_b64`` field.

    Called by the CLI immediately after YAML parse and before the
    plan reaches the coordinator (local or cluster). After this
    function returns, the plan is "wire-ready": every TarballSource
    carries its bytes inline and no subsequent layer needs to touch
    operator disk.

    Behavior:

    - For each entry whose ``context_source`` is a ``TarballSource``:
      reads the file at ``path`` (resolved relative to ``base_dir``
      if given, else cwd / absolute), validates ``len(bytes) <=
      max_bytes``, base64-encodes, returns a fresh ``BuildPlan``
      with that entry updated.
    - Idempotent: entries whose ``content_b64`` is already set get
      passed through (re-applies don't re-load).
    - Raises ``ManifestInvalid`` on any failure (file missing,
      oversized, unreadable) with a clear message naming the
      offending image_ref + path.

    The plan_id is unchanged by this function — see
    :func:`compute_plan_id` for the canonicalisation that strips
    ``content_b64`` before hashing.
    """
    import base64

    new_entries = []
    any_changed = False
    for e in plan.entries:
        if not isinstance(e.context_source, TarballSource):
            new_entries.append(e)
            continue
        src = e.context_source
        if src.content_b64 is not None:
            new_entries.append(e)
            continue
        path = Path(src.path)
        if not path.is_absolute() and base_dir is not None:
            path = base_dir / path
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise ManifestInvalid(
                f"build plan rejected: tarball entry {e.image_ref!r} "
                f"could not load {src.path!r}: {exc}",
            ) from exc
        if len(data) > max_bytes:
            raise ManifestInvalid(
                f"build plan rejected: tarball entry {e.image_ref!r} "
                f"is {len(data) / 1024**2:.1f} MB at {src.path!r}, "
                f"over the {max_bytes / 1024**2:.0f} MB cap. Either "
                f"trim the build context (.dockerignore is your "
                f"friend) or raise the cap with ``xrlenv build "
                f"apply --build-tarball-max-bytes <bytes>``.",
            )
        encoded = base64.b64encode(data).decode("ascii")
        new_src = src.model_copy(update={"content_b64": encoded})
        new_entries.append(e.model_copy(update={"context_source": new_src}))
        any_changed = True
    if not any_changed:
        return plan
    return plan.model_copy(update={"entries": new_entries})


def compute_plan_id(plan: BuildPlan) -> str:
    """Stable ``sha256`` over the canonicalised plan.

    Re-applying the *same* YAML produces the same id — the build
    coordinator treats this as the idempotency key (notes/
    phase-1-to-do.md, idempotency layer 1).

    Canonicalisation: sort keys, drop ``None`` values, drop the
    ``name`` field (operator-only metadata; renaming a plan must
    not change its content-hash or every rename would trigger a
    fresh dispatch), drop ``content_b64`` from every tarball entry
    (apply-time runtime payload — see :class:`TarballSource`), and
    force-fold the JSON encoding so whitespace / quoting variations
    don't perturb the hash.
    """
    raw = plan.model_dump(mode="json", exclude_none=True)
    raw.pop("name", None)
    # Strip apply-time tarball bytes — same plan_id whether or not
    # the CLI has resolved them yet. ``entries`` is a list of dicts
    # at this point; each may carry ``context_source.content_b64``.
    for entry in raw.get("entries", []):
        ctx = entry.get("context_source")
        if isinstance(ctx, dict):
            ctx.pop("content_b64", None)
    canon = json.dumps(raw, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


__all__ = [
    "BenchmarkBuildSpec",
    "BenchmarkSelection",
    "BuildBudget",
    "BuildEntry",
    "BuildPlan",
    "ContextSource",
    "EntryPlacement",
    "GitSource",
    "LocalSource",
    "RegistrySource",
    "SizeHintSource",
    "TarballSource",
    "compute_plan_id",
    "load_build_plan",
    "resolve_tarball_sources",
]
