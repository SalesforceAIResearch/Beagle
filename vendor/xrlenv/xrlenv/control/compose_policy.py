"""CP-side security vetting for a multi-service compose project (§3.5).

``docker compose up`` runs a task's compose document on a node's daemon, executing
its per-service ``privileged`` / ``cap_add`` / ``devices`` / host bind-mounts /
``network_mode`` directly. The single-container acquire path is gated by the
control plane's :class:`~xrlenv.control.kwargs_policy.KwargsPolicy` (the node does
**not** independently enforce ``allow_privileged`` — it applies what the CP
approved), so the compose gate lives CP-side too: the coordinator vets the compose
against the same policy **before** issuing the node command, and the node only
ever runs an already-vetted document.

This module is a thin adapter — it maps each compose service to the existing
:func:`~xrlenv.control.kwargs_policy.validate_kwargs` call the single-acquire path
uses, so there are **zero** new policy semantics: ``allow_privileged``,
``denied_caps``, ``allowed_devices``, ``allowed_host_paths``, ``allow_host_network``
and the always-fatal Level-3 escapes all behave identically to a plain
``containers.run``. Rejections across every service collect into one
:class:`~xrlenv.control.kwargs_policy.KwargsPolicyViolation` (fail-loud,
non-retryable) so an operator sees every problem at once.

Reject, don't strip: consistent with ``KwargsPolicy`` everywhere else, a
policy-violating compose fails the acquire loudly rather than being silently
rewritten (which would break the task confusingly downstream). The corpus's 7
multi-service tasks pass under the operator's existing ``allow_privileged`` opt-in
and default cap allowlist, and none mount a host path.
"""
from __future__ import annotations

from typing import Any

from xrlenv.control.kwargs_policy import (
    DEFAULT_POLICY,
    KwargsPolicy,
    KwargsPolicyViolation,
    KwargsRejection,
    validate_kwargs,
)

__all__ = ["vet_compose_project"]


def _services(compose: dict[str, Any]) -> dict[str, Any]:
    services = compose.get("services")
    return services if isinstance(services, dict) else {}


def _as_list(value: Any) -> list[Any]:
    """Compose fields like ``cap_add`` / ``devices`` are lists; be tolerant of a
    scalar (a malformed doc) by wrapping it, and of ``None`` by dropping it."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _host_binds(service: dict[str, Any]) -> list[str]:
    """Extract host bind sources from a service's ``volumes:`` as
    ``"<host>:<container>"`` specs for ``validate_kwargs(binds=…)``.

    Named volumes and anonymous/tmpfs mounts are **not** host binds and are
    skipped — only a bind whose source is a host path (absolute, ``.``/``~``
    relative, or an explicit ``type: bind``) is subject to ``allowed_host_paths``.
    Both compose forms are handled: the short ``"src:dst[:mode]"`` string and the
    long ``{type: bind, source: /host, target: /c}`` mapping.
    """
    binds: list[str] = []
    for entry in _as_list(service.get("volumes")):
        if isinstance(entry, str):
            src = entry.split(":", 1)[0]
            if src.startswith(("/", ".", "~")):
                binds.append(entry)
        elif isinstance(entry, dict):
            # Long form. A host bind has type: bind (default when a source path
            # is given); named volumes (type: volume) and tmpfs are not host binds.
            vtype = entry.get("type")
            source = entry.get("source")
            if source is None:
                continue
            src_str = str(source)
            is_bind = vtype == "bind" or (
                vtype is None and src_str.startswith(("/", ".", "~"))
            )
            if is_bind:
                target = entry.get("target", "")
                binds.append(f"{src_str}:{target}")
    return binds


def _network_mode(service: dict[str, Any]) -> str | None:
    """A service's ``network_mode``, if it sets one. ``network_mode: service:*``
    (intra-project) and ``none`` are safe and validate as clean; ``host`` and
    ``container:*`` are gated/blocked by ``validate_kwargs`` exactly as on the
    single-acquire path."""
    nm = service.get("network_mode")
    return str(nm) if nm else None


def _service_rejections(
    name: str, service: Any, policy: KwargsPolicy,
) -> list[KwargsRejection]:
    """Vet one service, returning its rejections tagged with the service name so a
    multi-service violation names the offending service. ``service`` is ``Any``
    because a parsed compose can carry a non-dict (``null``) service value."""
    if not isinstance(service, dict):
        return []
    raw = validate_kwargs(
        devices=_as_list(service.get("devices")) or None,
        cap_add=_as_list(service.get("cap_add")) or None,
        privileged=bool(service.get("privileged", False)),
        network_mode=_network_mode(service),
        pid_mode=service.get("pid"),
        ipc_mode=service.get("ipc"),
        cgroup_parent=service.get("cgroup_parent"),
        cpuset_cpus=service.get("cpuset"),
        binds=_host_binds(service) or None,
        userns_mode=service.get("userns_mode"),
        platform=service.get("platform"),
        runtime=service.get("runtime"),
        policy=policy,
    )
    # Tag each rejection with the service so ``KwargsPolicyViolation`` points at it.
    return [
        r.model_copy(update={"kwarg": f"services.{name}.{r.kwarg}"})
        for r in raw
    ]


def vet_compose_project(
    compose: dict[str, Any],
    *,
    policy: KwargsPolicy = DEFAULT_POLICY,
) -> None:
    """Vet every service in ``compose`` against ``policy``; raise on any violation.

    Raises :class:`~xrlenv.control.kwargs_policy.KwargsPolicyViolation` carrying
    the full cross-service rejection list (never just the first) so the operator
    can fix everything in one pass. A clean compose returns ``None``.

    The compose is the **rewritten, image-ref** document the plugin sends (§4.1)
    — build contexts are already gone — so only runtime container fields are
    vetted. Docker networks the task declares are project-scoped and not a policy
    surface; ``network_mode`` (which *joins* a namespace) is.
    """
    rejections: list[KwargsRejection] = []
    for name, service in _services(compose).items():
        rejections.extend(_service_rejections(name, service, policy))
    if rejections:
        raise KwargsPolicyViolation(rejections)
