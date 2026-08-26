"""Scratch-registry garbage-collection decision logic.

The scratch registry (:5012) holds build-on-demand images and must stay
bounded — otherwise it reintroduces the unbounded-growth problem it exists to
avoid. This module is the **pure decision core**: given the current scratch
repos (digest + size + last-used) and the set of digests active runs still
reference, decide which to reclaim by **TTL** (age out) and **per-namespace
quota** (oldest-first until under cap).

The one hard rule (spec 00 invariant 4, ``notes/scratch-registry-build-on-
demand.md``): **never reclaim a digest an active run references.** Operators
also set the TTL comfortably beyond the longest run, so a mid-run reclaim is
impossible in practice; the exemption set is the belt-and-suspenders backstop.

The registry-facing plumbing (list repos, read sizes/mtimes, delete manifests,
run ``registry garbage-collect``) lives in ``deploy/registry/scratch_registry_gc.py``;
this module is import-only + fully unit-tested so the eviction policy is
verifiable without a live registry.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from xrlenv.image_build import SCRATCH_NAMESPACE
from xrlenv.image_refs import repo_path


@dataclass(frozen=True)
class ScratchImage:
    """One scratch-registry image manifest, as seen by the GC."""

    repo: str
    """Repository name, e.g. ``scratch/<input_digest>``."""
    digest: str
    """Manifest digest, ``sha256:...`` — the identity an active run pins."""
    size_bytes: int
    """Unique on-disk footprint attributable to this image."""
    last_used_at: float
    """Epoch seconds of the most recent push/pull (or manifest mtime when the
    registry doesn't expose pull time)."""


def select_reclaim_targets(
    images: list[ScratchImage],
    *,
    now: float,
    ttl_seconds: float | None,
    quota_bytes: int | None,
    exempt_digests: frozenset[str],
) -> list[ScratchImage]:
    """Return the images to reclaim, honoring the active-run exemption.

    An image is **exempt** when its manifest ``digest`` OR its ``repo`` is in
    ``exempt_digests`` — so the control plane can exempt by the content-
    addressed repo (``scratch/<input_digest>``) it knows without resolving the
    post-build manifest digest. See :func:`active_scratch_repos`.

    Two passes, both skipping any exempt image:

    1. **TTL** — reclaim every non-exempt image older than ``ttl_seconds``.
    2. **Quota** — if the *total* footprint (including exempt images, which
       occupy space but can't be evicted) still exceeds ``quota_bytes``,
       reclaim the oldest non-exempt images not already picked until the total
       is under quota.

    ``ttl_seconds`` / ``quota_bytes`` of ``None`` disable that pass. Passing
    both ``None`` reclaims nothing. Deterministic: ties broken by
    ``(last_used_at, repo, digest)``. If exempt images alone exceed the quota,
    the quota can't be met — that's correct (active runs are never evicted);
    the caller logs the shortfall.
    """
    def _is_exempt(img: ScratchImage) -> bool:
        return img.digest in exempt_digests or img.repo in exempt_digests

    reclaimed: dict[str, ScratchImage] = {}

    # Pass 1 — TTL.
    if ttl_seconds is not None:
        for img in images:
            if _is_exempt(img):
                continue
            if now - img.last_used_at > ttl_seconds:
                reclaimed[img.digest] = img

    # Pass 2 — quota. Total counts every still-resident image (exempt +
    # not-yet-reclaimed); we can only evict non-exempt, not-yet-reclaimed ones.
    if quota_bytes is not None:
        resident = [i for i in images if i.digest not in reclaimed]
        total = sum(i.size_bytes for i in resident)
        if total > quota_bytes:
            evictable = sorted(
                (i for i in resident if not _is_exempt(i)),
                key=lambda i: (i.last_used_at, i.repo, i.digest),
            )
            for img in evictable:
                if total <= quota_bytes:
                    break
                reclaimed[img.digest] = img
                total -= img.size_bytes

    # Stable, oldest-first ordering for the caller's delete loop.
    return sorted(
        reclaimed.values(),
        key=lambda i: (i.last_used_at, i.repo, i.digest),
    )


def active_scratch_repos(
    image_status_pairs: Iterable[tuple[str | None, str]],
) -> set[str]:
    """The scratch repo names (``scratch/<input_digest>``) referenced by
    non-terminal sandboxes — the GC's active-run exemption set.

    Input is ``(image, status)`` pairs from the sandbox table; a ``destroyed``
    status or a non-scratch image is skipped. The result is intended to be
    served by the control plane and consumed via the GC's ``--exempt-url``,
    matched against :attr:`ScratchImage.repo` by
    :func:`select_reclaim_targets`.
    """
    repos: set[str] = set()
    prefix = f"{SCRATCH_NAMESPACE}/"
    for image, status in image_status_pairs:
        if not image or status == "destroyed":
            continue
        repo = repo_path(image)
        if repo.startswith(prefix):
            repos.add(repo)
    return repos


__all__ = ["ScratchImage", "active_scratch_repos", "select_reclaim_targets"]
