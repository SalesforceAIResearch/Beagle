"""Dispatch primitive — coordinator's view of "execute this build on
that node" (P1.6.b).

Two implementations land across the slice progression:

- :class:`InProcessNodeBuilder` (this file, P1.6.b) — calls the
  plug-in's :class:`BenchmarkImageBuilder` directly in-process. Used
  by :class:`xrlenv.control.runtime.LocalRuntime`. Useful for laptop
  workflows + the bash shim (P1.6.d).
- A gRPC-backed implementation lands in P1.6.c, riding on the
  spec-21 :class:`BuildImagesCommand` proto. Same Protocol surface so
  the build coordinator doesn't care which one runs.

The Protocol is intentionally async-iterator shaped so the gRPC
variant can stream per-image completion events as they happen rather
than buffering the whole batch.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from xrlenv.control.image_builder import (
    BuildResult,
    ImageBuilderImportError,
    load_image_builder,
)
from xrlenv.control.image_planner import PlanAssignment

LOGGER = logging.getLogger(__name__)


class BuildJob(BaseModel):
    """One node's slice of a plan — what the dispatch sends."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str
    assignments: tuple[PlanAssignment, ...]
    builder_per_benchmark: dict[str, BuilderRef] = Field(default_factory=dict)
    """Map benchmark name → :class:`BuilderRef` so the executor knows
    which Python class to instantiate per-image. The coordinator fills
    this from the catalog at dispatch time so every node sees the same
    pinned builder regardless of when it joins the cluster."""
    build_kwargs_per_benchmark: dict[str, dict[str, str]] = Field(
        default_factory=dict,
    )
    """Per-benchmark kwargs forwarded to ``BenchmarkImageBuilder.build``
    (e.g. ``{"build_path": "build-locally"}`` for swebench-verified)."""
    force: bool = False
    lazy_registrations: tuple[PlanAssignment, ...] = ()
    """Audit P1.6.g-H1 fix (2026-05-05): refs the operator wants the
    node to know about — register the (image_ref → BuilderRef + kwargs)
    mapping so a later ``ensure_present`` can dispatch lazily — but
    NOT build synchronously here. Used for opportunistic-mode
    deferred rows (status=registered): they land on this node's
    preferred-home, the lazy hook fires when the first rollout
    needs them, and ``backend.pull_image`` is bypassed for refs the
    benchmark builder is the only producer for."""


class BuilderRef(BaseModel):
    """Module + class — same shape as :class:`ImageBuilderDecl` but
    decoupled so the dispatch protocol doesn't import the manifest types."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    module: str
    class_name: str


class NodeBuilder(Protocol):
    """Coordinator's dispatch surface.

    One concrete impl per runtime (LocalRuntime → in-process,
    DistributedRuntime → gRPC). The coordinator yields ``BuildResult``
    rows as they arrive so the state store can mark each assignment
    ``done`` / ``failed`` incrementally.
    """

    def execute(
        self, job: BuildJob,
    ) -> AsyncIterator[BuildResult]:
        """Run ``job`` and stream per-image results."""


# ──────────────────────────────────────────────────────────────────────────────
# In-process implementation (LocalRuntime path)
# ──────────────────────────────────────────────────────────────────────────────


class InProcessNodeBuilder:
    """Loads each benchmark's builder lazily + dispatches in-process.

    Caches loaded builders per ``(module, class)`` pair so a 50-image
    plan reuses the same ``Tb2ImageBuilder`` instance for all 50 builds.

    P1.6.g — when constructed with a ``node_agent`` reference, registers
    each assignment's ``(image_ref → BuilderRef)`` mapping into the
    agent's lazy-builder dict before dispatch. The agent's image
    cache uses that mapping later when ``ensure_present`` fires for a
    ref that wasn't (or was, but got evicted) materialized at apply
    time.
    """

    def __init__(self, node_agent: Any | None = None) -> None:
        self._cache: dict[tuple[str, str], object] = {}
        self._node_agent = node_agent

    async def execute(
        self, job: BuildJob,
    ) -> AsyncIterator[BuildResult]:
        # Register lazy-builder mappings up front (P1.6.g — H3 lazy
        # lifecycle). Even if the synchronous build dispatch below
        # fails or is skipped, the agent retains the mapping so a
        # later ``ensure_present`` call can re-trigger the build.
        # Audit P1.6.g-H1 fix (2026-05-05): include lazy_registrations
        # too — those are deferred rows that didn't fit the budget
        # at apply time but need their builder mapping registered so
        # the lazy hook can produce them on first rollout.
        if self._node_agent is not None and (
            job.assignments or job.lazy_registrations
        ):
            mapping: dict[str, tuple[BuilderRef, dict[str, str]]] = {}
            for assignment in (*job.assignments, *job.lazy_registrations):
                ref = job.builder_per_benchmark.get(assignment.benchmark)
                if ref is None:
                    continue
                kwargs = dict(job.build_kwargs_per_benchmark.get(
                    assignment.benchmark, {},
                ))
                mapping[assignment.image_ref] = (ref, kwargs)
            if mapping:
                self._node_agent.register_lazy_image_builders(mapping)

        for assignment in job.assignments:
            ref = job.builder_per_benchmark.get(assignment.benchmark)
            if ref is None:
                yield BuildResult(
                    image_ref=assignment.image_ref, status="failed",
                    error=(
                        f"no image_builder registered for benchmark "
                        f"{assignment.benchmark!r}"
                    ),
                )
                continue
            kwargs = dict(job.build_kwargs_per_benchmark.get(
                assignment.benchmark, {},
            ))
            try:
                builder = self._get_builder(ref)
            except ImageBuilderImportError as exc:
                yield BuildResult(
                    image_ref=assignment.image_ref, status="failed",
                    error=f"builder load failed: {exc}",
                )
                continue
            try:
                result = await builder.build(  # type: ignore[attr-defined]
                    image_ref=assignment.image_ref,
                    kwargs=kwargs, force=job.force,
                )
            except Exception as exc:
                LOGGER.exception(
                    "InProcessNodeBuilder.build raised on %s",
                    assignment.image_ref,
                )
                yield BuildResult(
                    image_ref=assignment.image_ref, status="failed",
                    error=f"builder.build raised: {exc}",
                )
                continue
            yield result

    def _get_builder(self, ref: BuilderRef) -> object:
        key = (ref.module, ref.class_name)
        if key not in self._cache:
            from xrlenv.control.image_builder import ImageBuilderDecl

            decl = ImageBuilderDecl.model_validate({
                "module": ref.module, "class": ref.class_name,
            })
            self._cache[key] = load_image_builder(decl)
        return self._cache[key]


# ──────────────────────────────────────────────────────────────────────────────
# gRPC-backed implementation (DistributedRuntime path; P1.6.c)
# ──────────────────────────────────────────────────────────────────────────────


class GrpcNodeBuilder:
    """Dispatch a node's build job via the spec-21 ``BuildImagesCommand``.

    Wraps a node-lookup callable (the same shape :class:`NodeRegistry`
    exposes for ``cfg.node_lookup``) and yields the per-image
    :class:`BuildResult` rows the node's reply carries. Phase-A is
    batched — the entire job lands in one reply, then this iterator
    fans them out so the build-coordinator can update one assignment
    row at a time.
    """

    def __init__(
        self,
        *,
        node_lookup: Callable[[str], Any | None],
    ) -> None:
        self._node_lookup = node_lookup

    async def execute(
        self, job: BuildJob,
    ) -> AsyncIterator[BuildResult]:
        transport = self._node_lookup(job.node_id)
        if transport is None:
            for a in job.assignments:
                yield BuildResult(
                    image_ref=a.image_ref, status="failed",
                    error=f"node {job.node_id!r} has no live transport",
                )
            return
        try:
            results = await transport.build_images(
                assignments=list(job.assignments),
                builder_per_benchmark=dict(job.builder_per_benchmark),
                kwargs_per_benchmark=dict(job.build_kwargs_per_benchmark),
                force=bool(job.force),
                lazy_registrations=list(job.lazy_registrations),
            )
        except Exception as exc:
            LOGGER.exception(
                "GrpcNodeBuilder.build_images failed for %s",
                job.node_id,
            )
            for a in job.assignments:
                yield BuildResult(
                    image_ref=a.image_ref, status="failed",
                    error=f"build_images RPC failed: {exc}",
                )
            return
        for r in results:
            yield r


__all__ = [
    "BuildJob",
    "BuilderRef",
    "GrpcNodeBuilder",
    "InProcessNodeBuilder",
    "NodeBuilder",
]
