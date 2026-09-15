"""CP-side security vetting for a multi-service compose project (§3.5).

``docker compose up`` runs a task's compose document on a node's daemon, executing
its per-service ``privileged`` / ``cap_add`` / ``devices`` / host bind-mounts /
``network_mode`` directly. The single-container acquire path is gated by the
control plane's :class:`~xrlenv.control.kwargs_policy.KwargsPolicy` (the node does
**not** independently enforce ``allow_privileged`` — it applies what the CP
approved), so the compose gate lives CP-side too: the coordinator vets the compose
against the same policy **before** issuing the node command, and the node only
ever runs an already-vetted document.

This module maps each compose service to the existing
:func:`~xrlenv.control.kwargs_policy.validate_kwargs` call the single-acquire path
uses: ``allow_privileged``,
``denied_caps``, ``allowed_devices``, ``allowed_host_paths``, ``allow_host_network``
and the always-fatal Level-3 escapes all behave identically to a plain
``containers.run``. It also rejects storage declarations that bypass project
scoping: global/external volumes, custom backing stores, and mounts inherited
from external containers. Rejections across every service collect into one
:class:`~xrlenv.control.kwargs_policy.KwargsPolicyViolation` (fail-loud,
non-retryable) so an operator sees every problem at once.

Reject, don't strip: consistent with ``KwargsPolicy`` everywhere else, a
policy-violating compose fails the acquire loudly rather than being silently
rewritten (which would break the task confusingly downstream). Project-local
volumes preserve their native mount paths and sharing between services.
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
    skipped here (named volume definitions are vetted separately). Only a bind
    whose source is a host path (absolute, ``.``/``~``
    relative, or an explicit ``type: bind``) is subject to ``allowed_host_paths``.
    Both compose forms are handled: the short ``"src:dst[:mode]"`` string and the
    long ``{type: bind, source: /host, target: /c}`` mapping.
    """
    binds: list[str] = []
    for entry in _as_list(service.get("volumes")):
        if isinstance(entry, str):
            if ":" not in entry:
                continue  # Target only: Docker allocates a private anonymous volume.
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


def _storage_rejections(compose: dict[str, Any]) -> list[KwargsRejection]:
    """Keep named volumes project-local without rewriting benchmark storage.

    A read-only external mount is also a cross-project communication channel:
    another rollout can write to it. Privileged/host-path opt-ins do not grant
    permission to attach arbitrary external volumes or containers.
    """
    rejected: list[KwargsRejection] = []

    def reject(path: str, reason: str, hint: str) -> None:
        rejected.append(KwargsRejection(kwarg=path, level=3, reason=reason, hint=hint))

    if compose.get("include"):
        reject("include", "Included documents are not resolved before policy vetting.",
               "Submit a self-contained Compose document with all resources resolved.")

    volumes = compose.get("volumes")
    if isinstance(volumes, dict):
        for name, definition in volumes.items():
            if not isinstance(definition, dict):
                continue
            path = f"volumes.{name}"
            if definition.get("name"):
                reject(f"{path}.name", "An explicit volume name bypasses Compose project scoping.",
                       "Remove name so Compose allocates a project-local volume.")
            if definition.get("external"):
                reject(f"{path}.external", "External volumes can share state across projects.",
                       "Remove external and initialize a project-local volume instead.")
            if definition.get("driver") not in (None, "", "local"):
                reject(f"{path}.driver", "Custom volume drivers can share an external backing store.",
                       "Use the local driver without driver_opts.")
            if definition.get("driver_opts"):
                reject(f"{path}.driver_opts", "Volume driver options can attach host or remote storage.",
                       "Use a project-local volume without driver_opts.")

    services = _services(compose)
    for name, service in services.items():
        if not isinstance(service, dict):
            continue
        if service.get("extends"):
            reject(f"services.{name}.extends", "Inherited services are not resolved before policy vetting.",
                   "Submit a self-contained Compose document with inheritance resolved.")
        for source in _as_list(service.get("volumes_from")):
            parts = source.split(":") if isinstance(source, str) else []
            if (not parts or parts[0] not in services or parts[0] == "container"
                    or len(parts) > 2 or (len(parts) == 2 and parts[1] not in ("ro", "rw"))):
                reject(f"services.{name}.volumes_from",
                       "Volumes may only be inherited from a declared service in this project.",
                       "Reference a local service, optionally followed by :ro or :rw.")
        for index, mount in enumerate(_as_list(service.get("volumes"))):
            # Vet the same source Docker will use. Node-side interpolation could
            # otherwise turn a seemingly named mount into an unvetted host bind.
            source = None
            mount_type = None
            if isinstance(mount, str):
                # A target-only expression could expand to source:target too.
                source = mount.split(":", 1)[0]
            elif isinstance(mount, dict):
                source = mount.get("source")
                mount_type = mount.get("type")
            if any(isinstance(value, str) and "$" in value for value in (source, mount_type)):
                reject(f"services.{name}.volumes[{index}]",
                       "Mount sources and types must be resolved before policy vetting.",
                       "Submit a literal mount source and type; node-side interpolation is not allowed.")
    return rejected


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
    — build contexts are already gone. Storage must remain scoped to the
    project; deferred includes/inheritance must be resolved before submission.
    Network topology is outside this storage check; ``network_mode`` is still
    vetted through the existing kwarg policy.
    """
    rejections = _storage_rejections(compose)
    for name, service in _services(compose).items():
        rejections.extend(_service_rejections(name, service, policy))
    if rejections:
        raise KwargsPolicyViolation(rejections)
