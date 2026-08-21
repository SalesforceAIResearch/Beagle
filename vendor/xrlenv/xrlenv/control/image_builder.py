"""Per-benchmark image builder (P1.6.a — control-plane-driven image builds).

The phase-1 ``xrlenv build apply`` flow drives image builds from the
control plane. Each benchmark plug-in ships a builder next to its
adapter / instance resolver — same plug-in mechanism, same manifest
shape — and the node-agent dispatches builds in-process when the
control plane assigns it work.

Phase-1.6 surface:

- :class:`ImageBuilderDecl` — the manifest's serialized handle to a
  builder (module + class).
- :class:`BenchmarkImageBuilder` Protocol — what plug-in implementers
  must provide.
- :class:`BuildResult` — what each ``build`` call returns; consumed
  by the build coordinator to update :class:`build_plan_assignments`.
- :func:`load_image_builder` — import a builder by spec.

Why this lives next to :mod:`xrlenv.control.instance_resolver` rather
than under ``xrlenv/node/``: the builder declaration is part of the
benchmark contract carried in the manifest (spec 06), and the catalog
validates the field at register time. The actual ``build()`` calls
happen on the node-agent; the node imports this module to load and
dispatch the builder. Single source of truth.

What this module deliberately does NOT do:

- It doesn't own the bin-packing or the dispatch lifecycle — those
  live in :mod:`xrlenv.control.image_planner` and
  :mod:`xrlenv.control.build_coordinator` (P1.6.b).
- It doesn't ship per-benchmark logic — that's the plug-in's job
  (mechanism not policy). The two in-tree builders for terminal-bench-2
  and swebench-verified live alongside their adapters under
  ``xrlenv_plugins/benchmarks/<name>/image_builder.py``.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, ClassVar, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

LOGGER = logging.getLogger(__name__)


class ImageBuilderDecl(BaseModel):
    """Manifest serialization of a :class:`BenchmarkImageBuilder` reference.

    YAML form::

        image_builder:
          module: xrlenv_plugins.benchmarks.<your_plugin>.image_builder
          class: YourImageBuilder

    No ``init_params`` field by design — builders are stateless from
    the manifest's point of view; per-call configuration arrives via
    the ``kwargs`` argument to :py:meth:`BenchmarkImageBuilder.build`.
    Builders that need plugin-static configuration can read it from
    the manifest's other fields after construction.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    module: str
    class_name: str = Field(..., alias="class")


class BuildResult(BaseModel):
    """One ``build()`` outcome.

    Consumed by the P1.6.b build coordinator to update the per-row
    ``build_plan_assignments`` status. ``status="done"`` means the
    image is locally tagged and label-conformant; ``status="failed"``
    means the build raised and ``error`` carries the message.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    image_ref: str
    status: Literal["done", "failed"]
    bytes_pulled: int = 0
    """Approximate bytes the build pulled from network (Dockerhub +
    upstream registries). Best-effort; 0 when unknown. Used only for
    operator-visible totals on the admin /builds page."""
    duration_s: float = 0.0
    """Wall-clock seconds the build took. 0.0 when unknown."""
    error: str | None = None
    """Failure message when ``status="failed"``. None on success."""


class BenchmarkImageBuilder(Protocol):
    """One per benchmark plug-in. Plug-ins implement; node-agent dispatches.

    Implementations live next to their case-1 plug-in's adapter /
    resolver (e.g.
    ``xrlenv_plugins.benchmarks.<your_plugin>.image_builder.YourImageBuilder``).
    The constructor receives the :class:`ImageBuilderDecl`; per-build
    configuration arrives via :py:meth:`build`'s ``kwargs``.

    Implementations should be **idempotent**: calling ``build`` twice
    for the same ``image_ref`` with ``force=False`` should be a fast
    no-op when the local Docker daemon already has the tag.
    """

    IMAGE_SIZE_HINT_BYTES: ClassVar[int]
    """Static upper-bound estimate of one final-tag image's on-disk size,
    used by the P1.6.b bin-packer at plan time before any image is
    actually built. Conservative — overshoot is OK; undershoot causes
    the planner to overcommit a node's disk budget. Reality is measured
    after the build and the snapshot updated; if reality consistently
    exceeds the hint by >20%, log a warning so the plug-in author
    bumps it."""

    def __init__(self, decl: ImageBuilderDecl) -> None: ...

    def enumerate_image_refs(self, *, selection: dict[str, Any]) -> list[str]:
        """Resolve a plan's ``selection`` block to the concrete image
        refs the build will produce.

        ``selection`` is the manifest-style sub-block from build-plan.yaml,
        carrying one of ``{"smoke": True}``, ``{"instances": [...]}``,
        ``{"all": True}``, or any plug-in-specific shape. Returns the
        final-tag image refs (e.g. ``["terminal-bench-2/fix-git:0.1",
        ...]``) — what the planner places and what the node ultimately
        builds. The list is deduplicated and stable-ordered.
        """

    async def build(
        self,
        *,
        image_ref: str,
        kwargs: dict[str, Any],
        force: bool,
    ) -> BuildResult:
        """Build one image referenced by ``image_ref``.

        ``kwargs`` carries plug-in-specific knobs (e.g.
        ``{"build_path": "build-locally"}`` for swebench-verified's
        offline mode). ``force=True`` rebuilds even when the local tag
        exists — the platform still skips when it's safe; ``force``
        is the operator's escape hatch for upstream-Dockerfile or
        stub-runtime-deps changes.
        """


# ──────────────────────────────────────────────────────────────────────────────
# Loader
# ──────────────────────────────────────────────────────────────────────────────


class ImageBuilderImportError(RuntimeError):
    """Raised when :func:`load_image_builder` cannot find the builder.

    Distinguished from generic :class:`ImportError` so the build
    coordinator can map it onto a clean per-assignment ``failed``
    row with a meaningful error message rather than crashing the
    apply call.
    """


def load_image_builder(decl: ImageBuilderDecl) -> BenchmarkImageBuilder:
    """Import + construct the builder class named in ``decl``.

    Imports happen lazily so a manifest that references a builder
    module can register at catalog time even when the benchmark's
    plug-in package isn't fully wired yet — the import failure
    surfaces only at the first ``build apply`` that targets this
    benchmark.
    """
    try:
        module = importlib.import_module(decl.module)
    except ImportError as exc:
        raise ImageBuilderImportError(
            f"could not import builder module {decl.module!r}: {exc}",
        ) from exc
    try:
        cls = getattr(module, decl.class_name)
    except AttributeError as exc:
        raise ImageBuilderImportError(
            f"module {decl.module!r} has no attribute {decl.class_name!r}",
        ) from exc
    return cls(decl)  # type: ignore[no-any-return]


__all__ = [
    "BenchmarkImageBuilder",
    "BuildResult",
    "ImageBuilderDecl",
    "ImageBuilderImportError",
    "load_image_builder",
]
