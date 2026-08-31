"""Docker image-reference helpers shared across planes.

A single home for the registry-host normalization rule so the control
plane (``xrlenv build calibrate`` matching, ``xrlenv images evict``
fan-out) and the data plane (node-side cache eviction matching) agree
byte-for-byte on "do these two refs name the same image". Duplicating
the rule is exactly the kind of drift CLAUDE.md warns about — keep it
here.
"""

from __future__ import annotations


def registry_agnostic_ref(ref: str) -> str:
    """Return ``ref`` with any leading registry host stripped.

    Build plans and consumer configs store registry-*agnostic* image
    refs (e.g. ``xrlenv-webarena-infinity/substrate:1ca77813``) — the
    registry is supplied separately at push time so one plan/config
    works across registries. But a node that *pulled* that image from a
    private registry reports + holds it under its registry-qualified
    Docker tag (e.g.
    ``<registry-host>:5011/xrlenv-webarena-infinity/substrate:1ca77813``).
    Cross-referencing the two (calibrate, evict) requires normalizing
    both sides to the same registry-agnostic form first.

    Docker's own rule: the substring before the first ``/`` is a
    registry host **only** when it contains a ``.`` or ``:`` or equals
    ``localhost`` — otherwise the whole path is a Docker-Hub-relative
    repository with no registry to strip (``library/python:3.12`` keeps
    its ``library`` component). A ``host:port`` colon lives in that
    first segment, never after the first ``/``, so this never trims a
    repository path or a ``:tag`` / ``@sha256`` suffix.
    """
    head, slash, rest = ref.partition("/")
    if slash and ("." in head or ":" in head or head == "localhost"):
        return rest
    return ref


def same_image(a: str, b: str) -> bool:
    """True iff ``a`` and ``b`` name the same image modulo registry host.

    Used by node-side eviction to match an operator-supplied ref
    (possibly bare, possibly registry-qualified) against the tags a node
    actually holds. A digest ref (``repo@sha256:...``) matches only its
    exact registry-agnostic form — we never equate a tag with a digest.
    """
    return registry_agnostic_ref(a) == registry_agnostic_ref(b)


def repo_path(ref: str) -> str:
    """The bare **repository path** of ``ref`` — registry host, ``:tag``, and
    ``@sha256:...`` digest all stripped.

    ``<registry-host>:5011/ns/img:main`` and ``ns/img@sha256:abc`` both collapse
    to ``ns/img``. This is the deliberately-looser sibling of
    :func:`registry_agnostic_ref` (which keeps the tag/digest): it exists for
    **calibrate**, where a plan lists an image by its ``:tag`` but the node
    pulled it **by digest** (the control plane digest-pins ``:tag`` →
    ``@sha256:...`` for invariant 4), so the daemon holds it untagged / under an
    ``@sha256`` ref and a tag-preserving match misses it. Matching by repo path
    reunites the two. NOT for eviction — there, equating a tag with a digest
    could remove the wrong image, so eviction stays strict (:func:`same_image`).
    """
    bare = registry_agnostic_ref(ref)          # drop the registry host
    bare = bare.split("@", 1)[0]               # drop a @sha256:... digest
    slash = bare.rfind("/")
    colon = bare.rfind(":")
    if colon > slash:                          # a ':' AFTER the last '/' is a tag
        bare = bare[:colon]
    return bare


def manifest_digest(ref: str | None) -> str | None:
    """The bare ``sha256:...`` manifest digest of a digest ref, else ``None``.

    ``public.ecr.aws/d3j8x8q7/swe-bench-202605@sha256:abc`` → ``sha256:abc``;
    a tag ref or bare repo → ``None``. The digest is content-addressed and
    globally unique, so it is the robust key for cross-referencing a plan ref's
    **resolved** digest (``RegistryDigestResolver.resolve`` returns
    ``host/repo@sha256:…``) against the ``RepoDigests`` a node reports for an
    image it pulled — independent of registry-host or repo spelling on either
    side. Used by ``xrlenv build calibrate``'s digest-match fallback to
    attribute a digest-pinned (held-untagged) image to its plan entry when many
    tags share one repository and the tag/repo-path matchers can't.
    """
    if not ref:
        return None
    _, sep, dig = ref.partition("@")
    if sep and dig.startswith("sha256:"):
        return dig
    return None


def has_explicit_tag(ref: str) -> bool:
    """True iff ``ref`` carries an explicit ``:tag`` (not a digest, not
    untagged).

    Gates calibrate's repo-path fallback. A node that pulled a plan image **by
    digest** (the control plane digest-pins ``:tag`` → ``@sha256:...`` for
    invariant 4) holds it *untagged* / under an ``@sha256`` ref — no explicit
    tag — so :func:`repo_path` can safely reunite it with the plan's tagged ref.
    But a node that holds a genuinely **different tag** of the same repo (e.g. a
    stale ``ns/img:20251031`` left over from a prior build, while the plan pins
    ``ns/img:20260403``) *does* carry an explicit tag — and it is NOT the plan's
    image. Crediting it would attribute the wrong (often 0-byte, fully-shared)
    size to the plan ref. So the fallback must fire only when this returns
    ``False``: an explicit-but-different tag is a real mismatch, not a digest
    pull.

    ``ns/img@sha256:abc`` → ``False`` (digest, no tag). ``ns/img`` → ``False``
    (untagged). ``ns/img:v1`` → ``True``. Registry host + digest are ignored.
    """
    bare = registry_agnostic_ref(ref)          # drop the registry host
    bare = bare.split("@", 1)[0]               # a digest is not a tag
    slash = bare.rfind("/")
    colon = bare.rfind(":")
    return colon > slash                       # a ':' AFTER the last '/' is a tag
