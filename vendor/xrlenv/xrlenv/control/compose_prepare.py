"""CP-side preparation of a compose document before it goes on the wire (§2.3).

The coordinator, after minting ``rollout_id`` + deriving ``project_name`` and
resolving the main image tag→digest, rewrites the (already-vetted) compose
document to:

1. **Stamp the reserved ``xrlenv.*`` labels** into every service — label
   ownership lives in **core**, not the plugin (the plugin never sees the
   CP-minted ``rollout_id``), mirroring how the single-container path has the node
   merge ``xrlenv.rollout_id`` / ``session_kind``. The ``session_kind`` is split by
   role: ``raw`` on the ``main`` service (a CP session — must stay visible +
   matched in the raw-GC node-truth diff), ``compose`` on the sidecars (not CP
   sessions — ``raw`` would make them node-only orphans the existing sweep
   force-destroys). See ``notes/multi-service-compose-step3-plan.md`` §2.3 / R9.

2. **Pin the main image** to the digest-resolved ref, consistently with
   ``manifest.image`` + the ensure-present ``images`` list, so the node never
   ``ensure_present``s a tag while ``docker compose`` runs a digest.

Pure ``dict`` → ``dict`` (the coordinator parses / dumps the YAML around it), so
it is fully unit-testable without a node.
"""
from __future__ import annotations

import copy
import ipaddress
from dataclasses import dataclass
from typing import Any

__all__ = [
    "LABEL_COMPOSE_PROJECT",
    "LABEL_ROLLOUT_ID",
    "LABEL_SESSION_KIND",
    "SESSION_KIND_COMPOSE",
    "SESSION_KIND_MAIN",
    "PreparedCompose",
    "pin_images",
    "prepare_compose",
    "stamp_and_pin",
    "subnet_claims",
    "subnets_overlap",
]

LABEL_ROLLOUT_ID = "xrlenv.rollout_id"
LABEL_COMPOSE_PROJECT = "xrlenv.compose_project"
LABEL_SESSION_KIND = "xrlenv.session_kind"

# ``main`` is a CP session → ``raw`` (visible + matched in the node-truth diff).
# Sidecars are not CP sessions → ``compose`` (invisible to the raw-filtered sweep,
# so the existing raw-GC never force-destroys them). §2.3 / R9.
SESSION_KIND_MAIN = "raw"
SESSION_KIND_COMPOSE = "compose"


def _normalize_labels(labels: Any) -> dict[str, str]:
    """Compose ``labels`` may be a map (``{k: v}``) or a list (``["k=v"]``);
    normalize to a ``{str: str}`` map so we can merge the reserved keys."""
    if isinstance(labels, dict):
        return {str(k): str(v) for k, v in labels.items()}
    if isinstance(labels, list):
        out: dict[str, str] = {}
        for item in labels:
            s = str(item)
            key, sep, value = s.partition("=")
            out[key] = value if sep else ""
        return out
    return {}


def stamp_and_pin(
    compose: dict[str, Any],
    *,
    rollout_id: str,
    project_name: str,
    main_service: str = "main",
    main_image_ref: str | None = None,
) -> dict[str, Any]:
    """Return a deep copy of ``compose`` with the reserved ``xrlenv.*`` labels
    stamped into every service (``session_kind=raw`` on ``main_service``,
    ``=compose`` on the rest) and the ``main_service`` image pinned to
    ``main_image_ref`` when given.

    The reserved keys are set **last** so a task can't override them (mirrors the
    single-container "operators should not override those keys" contract). Services
    with no ``main_service`` present are left unstamped-for-main only — every
    service still gets ``rollout_id`` + ``compose_project`` + a ``session_kind``.
    """
    out = copy.deepcopy(compose)
    services = out.get("services")
    if not isinstance(services, dict):
        return out
    for name, svc in services.items():
        if not isinstance(svc, dict):
            continue
        labels = _normalize_labels(svc.get("labels"))
        labels[LABEL_ROLLOUT_ID] = rollout_id
        labels[LABEL_COMPOSE_PROJECT] = project_name
        labels[LABEL_SESSION_KIND] = (
            SESSION_KIND_MAIN if name == main_service else SESSION_KIND_COMPOSE
        )
        svc["labels"] = labels
        if name == main_service and main_image_ref:
            svc["image"] = main_image_ref
            # a build: stanza (if any survived) would shadow the pinned image and
            # re-introduce a tag; the node never has a build context anyway.
            svc.pop("build", None)
            svc.pop("pull_policy", None)
    return out


def subnet_claims(compose: dict[str, Any]) -> tuple[str, ...]:
    """Every pinned subnet CIDR the compose declares under
    ``networks.*.ipam.config[].subnet`` (§4.2 / 3b).

    A static-IP task pins a subnet its solve.sh hard-codes; two projects claiming
    overlapping subnets can't co-locate (docker refuses overlapping networks on one
    daemon). The coordinator feeds these to ``place(exclude_node_ids=…)`` so such
    projects don't land on the same node. Empty for service-DNS-only tasks (docker
    auto-assigns per project → unbounded concurrency). Core-owned: the CP has the
    vetted document, so it derives the claims itself rather than depending on the
    plugin."""
    claims: list[str] = []
    networks = compose.get("networks")
    if not isinstance(networks, dict):
        return ()
    for net in networks.values():
        if not isinstance(net, dict):
            continue
        ipam = net.get("ipam")
        if not isinstance(ipam, dict):
            continue
        for entry in ipam.get("config") or []:
            if isinstance(entry, dict) and entry.get("subnet"):
                claims.append(str(entry["subnet"]))
    return tuple(claims)


def subnets_overlap(a: str, b: str) -> bool:
    """True if the two CIDRs overlap (share any address). Unparseable input falls
    back to a conservative string-equality match — never a false "disjoint"."""
    try:
        net_a = ipaddress.ip_network(a, strict=False)
        net_b = ipaddress.ip_network(b, strict=False)
    except ValueError:
        return a == b
    return net_a.overlaps(net_b)


def _main_image(compose: dict[str, Any], main_service: str) -> str | None:
    services = compose.get("services")
    if not isinstance(services, dict):
        return None
    svc = services.get(main_service)
    if isinstance(svc, dict) and svc.get("image"):
        return str(svc["image"])
    return None


def pin_images(
    images: list[str],
    *,
    original_main_ref: str | None,
    resolved_main_ref: str | None,
) -> list[str]:
    """Return ``images`` with ``original_main_ref`` replaced by
    ``resolved_main_ref`` (the digest-pinned main image), ensuring the resolved
    ref is present exactly once. No-op when no resolution happened."""
    out = list(images or [])
    if not resolved_main_ref:
        return out
    out = [resolved_main_ref if ref == original_main_ref else ref for ref in out]
    if resolved_main_ref not in out:
        out.append(resolved_main_ref)
    return out


@dataclass(frozen=True)
class PreparedCompose:
    """The fully-prepared compose stack the coordinator sends to the node:
    ``compose`` (label-stamped + main-image-pinned) and ``images`` (the
    ensure-present list, with the main image pinned to the **same** resolved ref).
    Keeping both in one object is what guarantees the node never
    ``ensure_present``s a tag while ``docker compose`` runs the resolved digest."""

    compose: dict[str, Any]
    images: list[str]


def prepare_compose(
    compose: dict[str, Any],
    images: list[str],
    *,
    rollout_id: str,
    project_name: str,
    main_service: str = "main",
    resolved_main_ref: str | None = None,
) -> PreparedCompose:
    """The single CP-side compose-prep entry point: stamp reserved labels, pin the
    main image in **both** the compose document and the ``images`` ensure-present
    list to ``resolved_main_ref`` (§2.3, threaded consistently).

    ``resolved_main_ref`` is the coordinator's tag→digest resolution of the main
    image; ``None`` leaves images as tags (labels are still stamped)."""
    original_main_ref = _main_image(compose, main_service)
    prepared = stamp_and_pin(
        compose,
        rollout_id=rollout_id,
        project_name=project_name,
        main_service=main_service,
        main_image_ref=resolved_main_ref,
    )
    pinned_images = pin_images(
        images,
        original_main_ref=original_main_ref,
        resolved_main_ref=resolved_main_ref,
    )
    return PreparedCompose(compose=prepared, images=pinned_images)
