"""Control-plane scratch build-on-demand preparation.

Turns a ``scratch_build`` template (one carrying an ``image_build:`` block)
into the two things the node needs to materialize it:

1. the **content-addressed scratch ref** — ``<scratch-host>/scratch/<input_digest>``,
   stable across the fleet and across runs (built once, drift-free); and
2. the **build source** shipped to the node — a :class:`GitSource` (repo
   clone) or a :class:`TarballSource` (the local ``context:`` dir tarred +
   base64-encoded inline).

The node's :meth:`GitSourceBuilder.build_and_push` then builds this source and
pushes it to the scratch registry embedded in the ref.

See ``notes/scratch-registry-build-on-demand.md``.
"""

from __future__ import annotations

import base64
import io
import os
import tarfile
from pathlib import Path

from xrlenv.control.build_plan import GitSource, TarballSource
from xrlenv.errors import XRLEnvError
from xrlenv.image_build import (
    ImageBuildSpec,
    compute_context_digest_for_dir,
    compute_input_digest,
    git_context_digest,
    scratch_ref,
)


def resolve_scratch_image(
    spec: ImageBuildSpec,
    *,
    scratch_host: str,
) -> tuple[str, GitSource | TarballSource]:
    """Compute the ``(scratch_ref, build_source)`` for an ``image_build`` spec.

    ``scratch_host`` is ``host:port`` of the scratch registry (from
    ``XRLENV_SCRATCH_REGISTRY_HOST``/``_PORT``). Raises :class:`XRLEnvError`
    when ``scratch_host`` is empty (build-on-demand is unconfigured) or the
    local ``context:`` directory is missing.
    """
    if not scratch_host:
        raise XRLEnvError(
            "scratch build-on-demand is not configured: set "
            "XRLENV_SCRATCH_REGISTRY_HOST (and _PORT) so image_build templates "
            "have a registry to build into. See "
            "notes/scratch-registry-build-on-demand.md.",
        )
    if spec.git is not None:
        context_digest = git_context_digest(spec.git)
        source: GitSource | TarballSource = GitSource(
            repo=spec.git.repo,
            ref=spec.git.ref,
            subdir=spec.git.subdir,
            dockerfile=spec.git.dockerfile,
        )
    else:
        assert spec.context is not None  # ImageBuildSpec enforces context xor git
        context_dir = Path(spec.context)
        if not context_dir.is_dir():
            raise XRLEnvError(
                f"image_build context {spec.context!r} is not a directory; "
                "the build context must exist on the control plane so it can "
                "be shipped to the build node.",
            )
        context_digest = compute_context_digest_for_dir(context_dir)
        source = _tarball_source_from_dir(context_dir, dockerfile=spec.dockerfile)
    input_digest = compute_input_digest(spec, context_digest=context_digest)
    return scratch_ref(scratch_host, input_digest), source


def durable_ref_for(spec: ImageBuildSpec, scratch_ref: str) -> str | None:
    """The durable destination ref for a scratch image, or ``None`` when the
    spec has no ``durable_to``.

    When ``durable_to`` already carries a tag (``:`` after the last ``/``) it
    is used verbatim; otherwise the content-addressed ``input_digest`` (taken
    from ``scratch_ref``) is appended as the tag, so the durable copy is itself
    content-addressed and stable.
    """
    if not spec.durable_to:
        return None
    dst = spec.durable_to
    tail = dst.rsplit("/", 1)[-1]
    if ":" in tail:
        return dst
    input_digest = scratch_ref.rsplit("/", 1)[-1]
    return f"{dst}:{input_digest}"


def _tarball_source_from_dir(
    context_dir: Path, *, dockerfile: str,
) -> TarballSource:
    """Tar ``context_dir``'s contents at the tar root (gzip) and inline them
    base64 into a :class:`TarballSource` — the node untars and builds against
    the root, finding ``dockerfile`` there."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        # arcname="." places the directory *contents* at the tar root, so the
        # Dockerfile lands at ``<extract>/dockerfile`` where the node looks.
        tf.add(str(context_dir), arcname=".", recursive=True)
    content_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return TarballSource(
        path=str(context_dir), dockerfile=dockerfile, content_b64=content_b64,
    )


def scratch_registry_host_from_env() -> str | None:
    """The scratch registry ``host:port`` from the environment, or ``None``
    when unset (build-on-demand disabled). ``XRLENV_SCRATCH_REGISTRY_HOST`` is
    the host; ``XRLENV_SCRATCH_REGISTRY_PORT`` defaults to ``5012``."""
    host = os.environ.get("XRLENV_SCRATCH_REGISTRY_HOST")
    if not host:
        return None
    port = os.environ.get("XRLENV_SCRATCH_REGISTRY_PORT", "5012")
    return f"{host}:{port}"


__all__ = [
    "durable_ref_for",
    "resolve_scratch_image",
    "scratch_registry_host_from_env",
]
