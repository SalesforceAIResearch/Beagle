"""Point EvoClaw's image resolution at pullable Docker Hub refs (DESIGN.md §5.2).

EvoClaw names milestone images after a local *retag* that ``scripts/pull_images.sh``
produces: ``<repo_full>/<milestone>:<v>`` — a registry-less, multi-level name that
exists in no registry, so the cluster can't pull it on acquire. The original
publishable image is ``hyd2apse/<short>:<milestone>-<v>`` on Docker Hub, which the
cluster's pull-through **mirror** serves transparently (nodes' ``registry-mirrors``
route ``docker.io`` refs).

This module monkeypatches ``harness.e2e.image_version.resolve_image`` (and the copy
bound into ``harness.e2e.evaluator``) to return the Docker Hub ref instead of the
local retag. The ``short ↔ full`` repo map is **parsed from EvoClaw's own
``pull_images.sh``** (its source of truth) rather than hardcoded — the short names
are not derivable from the full names (e.g. ``apache_dubbo_…`` → ``dubbo``).

The **agent base image** needs no override: the operator passes a pullable
``--image hyd2apse/<short>:base-<v>`` directly.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable
from pathlib import Path

LOGGER = logging.getLogger("xrlenv.evoclaw.image_resolution")

_DEFAULT_TAG = "v0.9"
_DEFAULT_ORG = "hyd2apse"

# Corrected go-zero **base** image. Upstream's published
# ``hyd2apse/go-zero:base-v0.9`` ships WITHOUT ``.git`` (go-zero's
# ``.dockerignore`` excludes ``.git`` and — unlike the milestone Dockerfiles,
# which ``COPY .git`` back — the base build didn't), so no agent can git-tag a
# submission and the run stalls. The corrected base image is the ONE image knob
# kept as an environment variable (``EVOCLAW_GOZERO_BASE_IMAGE``), mirroring
# EvoClaw's own image config. It is REQUIRED (no default): build the gitfixed base
# from ``go-zero-gitfix.Dockerfile``, push it to your private registry, and point
# the env var at it (see the evoclaw README).


def _gozero_base_ref() -> str | None:
    """Corrected go-zero **base** image ref, or None to fall through to the normal
    milestone-style ref. Read from env ``EVOCLAW_GOZERO_BASE_IMAGE`` — REQUIRED,
    no default: point it at the gitfixed go-zero base image in your private
    registry, or set it EMPTY to disable the redirect and use the upstream
    ``base-v0.9`` directly (e.g. once upstream republishes it with ``.git``).
    """
    if "EVOCLAW_GOZERO_BASE_IMAGE" not in os.environ:
        raise SystemExit(
            "ERROR: EVOCLAW_GOZERO_BASE_IMAGE is not set. Set it (in .env) to the "
            "gitfixed go-zero base image ref in your private registry, e.g. "
            "<registry-host>:5011/go-zero:base-v0.9-gitfix (build it with "
            "xrlenv_plugins/benchmarks/evoclaw/go-zero-gitfix.Dockerfile), or set "
            "it empty to use the upstream base. See "
            "xrlenv_plugins/benchmarks/evoclaw/README.md.",
        )
    ref = os.environ["EVOCLAW_GOZERO_BASE_IMAGE"].strip()
    return ref or None


def load_repo_short_map(evoclaw_root: Path) -> dict[str, str]:
    """Parse ``REPO_FULL[short]="full"`` pairs → ``{full.lower(): short}``."""
    script = evoclaw_root / "scripts" / "pull_images.sh"
    if not script.is_file():
        raise FileNotFoundError(
            f"cannot find EvoClaw's pull_images.sh at {script} to read the "
            "repo short↔full map; set EVOCLAW_SOURCE_ROOT or pass evoclaw_root",
        )
    text = script.read_text(encoding="utf-8")
    pairs = re.findall(r'REPO_FULL\[([^\]]+)\]="([^"]+)"', text)
    return {full.lower(): short for short, full in pairs}


def dockerhub_ref(
    image_base: str,
    full_to_short: dict[str, str],
    *,
    image_tag: str = _DEFAULT_TAG,
    image_registry: str = "",
) -> str | None:
    """``<repo_full>/<milestone>`` → ``[<reg>]<org>/<short>:<milestone>-<tag>``.

    Returns None if the repo isn't a known EvoClaw repo (caller falls back to the
    original resolver). ``image_tag`` / ``image_registry`` are wrapper flags
    (``--image-tag`` / ``--image-registry``), threaded in so they never hide in the
    environment. ``image_registry`` optionally prefixes an explicit mirror host
    (default empty → docker.io routes through the node mirror). The go-zero base
    redirect reads env ``EVOCLAW_GOZERO_BASE_IMAGE`` (see :func:`_gozero_base_ref`).
    """
    parts = image_base.split("/")
    repo_full, milestone = parts[0], parts[-1]
    short = full_to_short.get(repo_full.lower())
    if short is None:
        return None
    # Per-repo BASE-image override for repos whose upstream base image is broken.
    # go-zero's published base ships without .git (see _gozero_base_ref); only its
    # base ref is redirected — its milestone images are fine.
    if short == "go-zero" and milestone == "base":
        fixed = _gozero_base_ref()
        if fixed:
            return fixed
    prefix = image_registry
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return f"{prefix}{_DEFAULT_ORG}/{short}:{milestone}-{image_tag}"


def install(
    evoclaw_root: Path | None = None,
    *,
    image_tag: str = _DEFAULT_TAG,
    image_registry: str = "",
) -> Callable[[str], str]:
    """Monkeypatch ``resolve_image`` to emit pullable Docker Hub milestone refs.

    ``image_tag`` / ``image_registry`` come from the wrapper flags. Returns the
    installed override (for tests). Unknown repos fall back to the original
    ``resolve_image`` so non-EvoClaw image bases are untouched.
    """
    import harness.e2e.evaluator as evaluator  # type: ignore[import-not-found]
    import harness.e2e.image_version as image_version  # type: ignore[import-not-found]

    if evoclaw_root is None:
        # harness/ lives at the EvoClaw repo root → root = parents[2] of image_version.
        f = image_version.__file__
        if not f:
            raise RuntimeError("cannot locate harness.e2e.image_version on disk")
        evoclaw_root = Path(f).resolve().parents[2]
    full_to_short = load_repo_short_map(evoclaw_root)
    original = image_version.resolve_image

    def override(image_base: str) -> str:
        ref = dockerhub_ref(
            image_base, full_to_short,
            image_tag=image_tag, image_registry=image_registry,
        )
        if ref is None:
            return original(image_base)
        LOGGER.info("resolve_image(%s) -> %s", image_base, ref)
        return ref

    image_version.resolve_image = override
    evaluator.resolve_image = override  # evaluator did `from ... import resolve_image`
    return override
