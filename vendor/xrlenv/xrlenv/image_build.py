"""``image_build`` — bring-your-own-Dockerfile, build-on-demand.

A template (spec 06's ``image_build:`` block) can ship a **Dockerfile +
build context** instead of a prebuilt image ref. The platform builds it
**once for the fleet, on demand**, into a quota-bounded scratch registry
and distributes it by pull. This module is the shared foundation for that
path:

- :class:`ImageBuildSpec` — the declarative build spec (local ``context:``
  dir *or* a ``git:`` context), parsed from the manifest / resolver return.
- :func:`compute_input_digest` — the **content-addressed tag** over the
  build inputs. Same Dockerfile + context + build-args ⇒ same digest ⇒
  built exactly once, drift-free across nodes.
- :func:`scratch_ref` — the ``<scratch-host>/scratch/<input_digest>`` ref.

It lives at the package top level (like :mod:`xrlenv.image_refs`) so both
the control plane (register-time content-addressing) and the node
(build-time) import it without a control⇄node dependency.

Full design + open sub-decisions: ``notes/scratch-registry-build-on-demand.md``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: Registry namespace under which content-addressed build-on-demand images
#: are pushed (``<scratch-host>/scratch/<input_digest>``).
SCRATCH_NAMESPACE = "scratch"


class GitContext(BaseModel):
    """A build context hosted in a git repo (the ``git:`` alternative to a
    local ``context:`` dir).

    The node clones ``repo`` at ``ref`` and runs
    ``docker build -f <dockerfile> <subdir>``. The tuple
    ``(repo, ref, subdir, dockerfile)`` fully pins the context, so it is
    what the content-addressed digest hashes (see
    :func:`compute_input_digest`) — no repo clone is needed just to
    compute the tag.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    repo: str
    """Git URL — ``https://...`` or ``git@...``."""

    ref: str = "main"
    """Branch, tag, or (preferably) an immutable commit sha. A moving ref
    like ``main`` means the content-addressed tag does *not* change when
    the branch advances — pin a sha for reproducibility."""

    subdir: str = "."
    """Path within the repo that is the docker build context."""

    dockerfile: str = "Dockerfile"
    """Dockerfile name within ``subdir``."""

    @model_validator(mode="after")
    def _validate(self) -> GitContext:
        if not self.repo:
            raise ValueError("image_build.git.repo must be non-empty")
        return self


class ImageBuildSpec(BaseModel):
    """A template's on-demand Dockerfile build declaration.

    Exactly one of ``context`` (a local directory holding a Dockerfile +
    build context) or ``git`` (a repo-hosted context) must be set.
    ``context`` is the primary "bring your own Dockerfile" path; ``git`` is
    the alternative for contexts that already live in a repo.

    Setting an ``image_build`` block implies the scratch build-on-demand
    path (``image_pin_mode="scratch_build"``); the user does not hand-edit
    the pin mode.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    context: str | None = None
    """Local directory holding a Dockerfile + build context. The platform
    tars it, content-addresses it, ships the bytes to the build node, and
    builds. Mutually exclusive with ``git``."""

    git: GitContext | None = None
    """Repo-hosted build context. Mutually exclusive with ``context``."""

    dockerfile: str = "Dockerfile"
    """Dockerfile name within ``context``. Ignored when ``git`` is set
    (``git.dockerfile`` applies there)."""

    build_args: dict[str, str] = Field(default_factory=dict)
    """``--build-arg`` values, folded into the content-addressed tag so a
    changed build-arg forces a distinct image."""

    durable_to: str | None = None
    """Optional user-owned registry endpoint (``host:port/repo``) the fleet
    can reach over the LAN. When set, the built image is copied there
    (digest-preserved) and survives scratch GC. When omitted, the image
    lives only in the scratch registry and may be GC'd at any time."""

    tag: str | None = None
    """Optional human-readable tag override. When ``None`` (the default and
    recommended), the scratch ref is content-addressed via
    :func:`compute_input_digest` — that is what guarantees build-once +
    no drift. An explicit tag is honored but bypasses input dedup."""

    @model_validator(mode="after")
    def _validate(self) -> ImageBuildSpec:
        if (self.context is None) == (self.git is None):
            raise ValueError(
                "image_build requires exactly one of 'context' (a local "
                "Dockerfile dir) or 'git' (a repo-hosted context)",
            )
        if self.context is not None and not self.context:
            raise ValueError("image_build.context must be a non-empty path")
        return self

    def effective_dockerfile(self) -> str:
        """The Dockerfile name that actually applies, whichever source is set."""
        return self.git.dockerfile if self.git is not None else self.dockerfile


def _sha256_hexdigest_of_json(payload: object) -> str:
    """Stable sha256 over a JSON-canonicalised payload (sorted keys, tight
    separators) so equal inputs always hash identically."""
    canon = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def compute_context_digest_for_dir(path: str | Path) -> str:
    """Deterministic sha256 over a build-context directory's file tree.

    Walks ``path`` in sorted order and hashes, per entry, its POSIX
    relative path, an executable-bit marker, and either the file's content
    sha256 (regular files) or the symlink target (symlinks). The result is
    independent of walk order and of absolute location, so two byte-identical
    context trees hash identically regardless of where they live.

    Conservative by design: it hashes **every regular file and symlink**,
    ignoring ``.dockerignore``. That can only ever *over*-rebuild (two
    contexts that differ solely in ignored files get distinct digests); it
    never wrongly dedupes two genuinely different contexts. Honoring
    ``.dockerignore`` is a later optimization.

    Symlinks that point at *directories* are not traversed (``os.walk`` with
    ``followlinks=False``) and so do not contribute to the digest. This stays
    on the conservative side — it can only over-rebuild, never falsely dedupe
    a file-content change — but two trees differing solely by a
    symlink-to-directory hash identically.

    Raises :class:`FileNotFoundError` / :class:`NotADirectoryError` if
    ``path`` is not an existing directory.
    """
    root = Path(path)
    if not root.is_dir():
        raise NotADirectoryError(f"image_build context is not a directory: {root}")

    entries: list[tuple[str, str, str]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            abspath = Path(dirpath) / name
            rel = abspath.relative_to(root).as_posix()
            if abspath.is_symlink():
                kind = "L"
                digest = hashlib.sha256(
                    os.readlink(abspath).encode("utf-8"),
                ).hexdigest()
            else:
                kind = "F"
                digest = _sha256_of_file(abspath)
            exec_bit = "x" if os.access(abspath, os.X_OK) else "-"
            entries.append((rel, f"{kind}{exec_bit}", digest))

    entries.sort()
    return _sha256_hexdigest_of_json(entries)


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_context_digest(git: GitContext) -> str:
    """Content digest of a git-hosted context — hashes the pinning tuple
    ``(repo, ref, subdir, dockerfile)`` without cloning anything.

    Because it hashes ``ref`` verbatim, a moving branch produces a stable
    digest across branch advances; pin a commit sha for true reproducibility.
    """
    return _sha256_hexdigest_of_json(
        {
            "repo": git.repo,
            "ref": git.ref,
            "subdir": git.subdir,
            "dockerfile": git.dockerfile,
        },
    )


def compute_input_digest(
    spec: ImageBuildSpec,
    *,
    context_digest: str,
    base_image_digest: str = "",
) -> str:
    """The content-addressed build-input digest — the scratch tag body.

    Hashes everything that determines the built bytes:

    - ``base_image_digest`` — the resolved ``FROM`` image ``@sha256`` when
      known (build-time, strongest). Empty string when unresolved
      (register-time, content still pinned to the Dockerfile's ``FROM``
      *tag* via ``context_digest``); passing it later yields a distinct,
      stronger-pinned digest.
    - ``context_digest`` — :func:`compute_context_digest_for_dir` for a
      local ``context:``, or :func:`git_context_digest` for a ``git:``
      source.
    - the effective Dockerfile name — so selecting ``Dockerfile.gpu`` vs
      ``Dockerfile`` within the *same* context tree produces a distinct
      image.
    - ``build_args`` — a changed build-arg forces a distinct image.

    Same inputs ⇒ same digest ⇒ built exactly once for the fleet.
    """
    payload = {
        "v": 1,
        "base_image_digest": base_image_digest,
        "context_digest": context_digest,
        "dockerfile": spec.effective_dockerfile(),
        "build_args": sorted(spec.build_args.items()),
    }
    return _sha256_hexdigest_of_json(payload)


def scratch_ref(
    scratch_host: str,
    input_digest: str,
    *,
    namespace: str = SCRATCH_NAMESPACE,
) -> str:
    """The content-addressed scratch registry ref for a build-input digest.

    ``scratch_host`` is ``host:port`` (e.g. ``cp-box:5012``). The returned
    ref is ``<host:port>/<namespace>/<input_digest>`` — a plain tag; the
    image's own manifest ``@sha256`` is resolved after the first build and
    pinned separately for the run.
    """
    if not scratch_host:
        raise ValueError("scratch_ref requires a non-empty scratch_host")
    if not input_digest:
        raise ValueError("scratch_ref requires a non-empty input_digest")
    if not namespace:
        raise ValueError("scratch_ref requires a non-empty namespace")
    return f"{scratch_host}/{namespace}/{input_digest}"


__all__ = [
    "SCRATCH_NAMESPACE",
    "GitContext",
    "ImageBuildSpec",
    "compute_context_digest_for_dir",
    "compute_input_digest",
    "git_context_digest",
    "scratch_ref",
]
