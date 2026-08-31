"""Generic harbor multi-service compose helpers — pure (``yaml`` + stdlib only).

Shared by two consumers so the ``build:`` → ``image:`` mapping and the
``<task_id>-<service>`` naming live in **one** place (no build-vs-run drift):

- ``xrlenv_plugins.benchmarks.<benchmark>.build_plan_gen`` — enumerates the
  services that ship a ``build:`` context and names the image each becomes, so
  the build+push step produces exactly the refs the eval will resolve.
- ``xrlenv_plugins.harbor.environment.XrlenvHarborEnvironmentCluster`` — at
  ``start()``, rewrites the task's compose so every ``build:`` service points at
  its pushed image ref (no build context ever ships to the node), injects
  per-sidecar cgroup caps, and reads the stack footprint + pinned subnet claims
  the scheduler needs.

This module imports **no** ``xrlenv`` or ``harbor`` symbols on purpose: the
build generator is deliberately lightweight (see its module docstring) and must
import these helpers without pulling the harbor runtime in. ``harbor``'s package
``__init__`` is lazy (PEP 562) precisely so ``import
xrlenv_plugins.harbor.compose`` stays harbor-free.

Terminology: a task's compose is *multi-service* when it declares more than one
service. Harbor execs the agent into the service named ``main`` (its base
build-compose defines it); every other service is a *sidecar*. A ``build:``
service whose context resolves to the environment dir (``.``) is the task's own
image (the canonical ``<task_id>`` ref already built today); a ``build:`` service
with a *sub-directory* context (e.g. ``./solr-node``) is a distinct image that
needs its own build entry.
"""
from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import PurePosixPath
from typing import Any

__all__ = [
    "DEFAULT_SIDECAR_CPU",
    "DEFAULT_SIDECAR_MEM_MB",
    "MAIN_CONTEXT",
    "MAIN_KEEPALIVE",
    "MAIN_SERVICE",
    "assemble_project",
    "build_context",
    "build_dockerfile",
    "default_image_refs",
    "ensure_main_service",
    "image_refs",
    "is_canonical_main_build",
    "is_multi_service",
    "is_safe_relative_context",
    "iter_build_services",
    "load_compose",
    "local_tag_service_names",
    "registry_namespace_and_tag",
    "rewrite_to_image_refs",
    "service_map",
    "sidecar_footprint",
    "subdir_build_services",
    "subnet_claims",
]

MAIN_SERVICE = "main"
MAIN_CONTEXT = "."

# Harbor's base compose keeps ``main`` alive with this exact command
# (``docker-compose-build.yaml`` / ``-prebuilt.yaml``) so the agent can ``exec``
# into it; the cluster reproduces it as the fill-missing-only default.
MAIN_KEEPALIVE = ["sh", "-c", "sleep infinity"]

# Q2 default per-sidecar reservation/cap when a service declares no
# ``deploy.resources``. Sidecars in the corpus are light (a DB, a mock server);
# these are safe, slightly-generous defaults.
DEFAULT_SIDECAR_CPU = 1.0
DEFAULT_SIDECAR_MEM_MB = 1024


def load_compose(text: str) -> dict[str, Any]:
    """Parse compose YAML text into a dict. Empty / null docs yield ``{}``."""
    import yaml

    doc = yaml.safe_load(text)
    return doc if isinstance(doc, dict) else {}


def service_map(compose: dict[str, Any]) -> dict[str, Any]:
    """The ``services:`` mapping (``{}`` when absent)."""
    services = compose.get("services")
    return services if isinstance(services, dict) else {}


def is_multi_service(compose: dict[str, Any]) -> bool:
    """True when the compose declares more than one service — the predicate that
    routes a task onto the compose-project path instead of the single acquire."""
    return len(service_map(compose)) > 1


def _normalize_context(ctx: str) -> str:
    """Normalize a build ``context`` to a POSIX-relative string; ``"."`` and
    ``"./"`` collapse to ``"."`` and ``"./solr-node"`` to ``"solr-node"``."""
    normalized = str(PurePosixPath(ctx.strip()))
    return "." if normalized in (".", "") else normalized


def build_context(service: dict[str, Any]) -> str | None:
    """The normalized build context of a service, or ``None`` if it has no
    ``build:`` stanza (an ``image:``-only service — e.g. ``postgres:14``).

    Handles both compose forms: ``build: ./dir`` (string) and
    ``build: {context: ./dir, dockerfile: Dockerfile}`` (mapping, default
    context ``.``)."""
    build = service.get("build")
    if build is None:
        return None
    if isinstance(build, str):
        return _normalize_context(build)
    if isinstance(build, dict):
        return _normalize_context(str(build.get("context", ".")))
    return None


# A ``build:`` service at the ROOT context (``.``) but with a *custom* ``dockerfile:``
# (e.g. chess-mate's ``Dockerfile.game``) is a DISTINCT image, not the task's canonical
# ``<id>`` image. ``None`` and (normalized) ``Dockerfile`` are the default.
_DEFAULT_DOCKERFILES = frozenset({None, "Dockerfile"})


def build_dockerfile(service: dict[str, Any]) -> str | None:
    """The ``dockerfile:`` a ``build:`` mapping names, POSIX-normalized (so
    ``./Dockerfile`` → ``Dockerfile``, not mistaken for a custom sidecar dockerfile),
    or ``None`` for the default. A string ``build:`` / ``image:``-only service → ``None``."""
    build = service.get("build")
    if isinstance(build, dict) and build.get("dockerfile") is not None:
        return str(PurePosixPath(str(build["dockerfile"])))
    return None


def is_canonical_main_build(service: dict[str, Any]) -> bool:
    """True iff this ``build:`` service produces the task's *canonical main* image: the
    ROOT context (``.``) built with the DEFAULT Dockerfile. A root context with a custom
    ``dockerfile:`` (e.g. ``Dockerfile.game``) is a DISTINCT sidecar image and maps to
    its own ``<task_id>-<service>`` ref instead."""
    return (
        build_context(service) == MAIN_CONTEXT
        and build_dockerfile(service) in _DEFAULT_DOCKERFILES
    )


def iter_build_services(compose: dict[str, Any]) -> dict[str, str]:
    """``{service_name: normalized_context}`` for every service with a ``build:``
    stanza. ``image:``-only services are omitted (they resolve to a pull)."""
    out: dict[str, str] = {}
    for name, svc in service_map(compose).items():
        if not isinstance(svc, dict):
            continue
        ctx = build_context(svc)
        if ctx is not None:
            out[name] = ctx
    return out


def subdir_build_services(compose: dict[str, Any]) -> dict[str, str]:
    """``{service: normalized_context}`` for every ``build:`` service that needs a
    **distinct** image beyond the task's own ``<id>`` image — a sub-directory context
    **or** a root context with a *custom* ``dockerfile:`` (e.g. chess-mate's ``game``
    built from ``Dockerfile.game``). Empty for the common case where the only ``build:``
    service is the canonical main."""
    out: dict[str, str] = {}
    for n, svc in service_map(compose).items():
        if not isinstance(svc, dict):
            continue
        ctx = build_context(svc)
        if ctx is not None and not is_canonical_main_build(svc):
            out[n] = ctx
    return out


def is_safe_relative_context(context: str) -> bool:
    """True if a (normalized) build context stays **within** the task
    environment dir — relative, not absolute, and no ``..`` escape.

    Benchmark composes are semi-untrusted authored content, so before a
    consumer joins a context onto the task's ``environment/`` dir it should
    reject one that would escape it (``build: /etc`` or ``build: ../../x``) —
    otherwise a ``type: local`` build context could point outside the task, or a
    future runtime path could stage host files. The invariant lives here so both
    the build-plan generator (which enforces it) and any later consumer share one
    definition. ``"."`` (the environment dir itself) is safe."""
    normalized = _normalize_context(context)
    path = PurePosixPath(normalized)
    return not path.is_absolute() and ".." not in path.parts


def default_image_refs(
    task_id: str,
    compose: dict[str, Any],
    *,
    namespace: str,
    tag: str = "main",
    main_ref: str | None = None,
) -> dict[str, str]:
    """The canonical ``{service: image_ref}`` for every ``build:`` service.

    The single source of truth for the ``<task_id>-<service>`` naming, shared by
    the build-plan generator (what to push) and the runtime rewrite (what to
    reference):

    - context ``.`` **with the default Dockerfile** → the task's own image, ``main_ref``
      if given else ``<namespace>/<task_id>:<tag>`` (matches today's single-service ref);
    - sub-dir context **or a custom ``dockerfile:``** (e.g. ``Dockerfile.game``) →
      ``<namespace>/<task_id>-<service>:<tag>``.

    ``image:``-only services are absent from the result — the rewrite leaves
    their public ``image:`` untouched.
    """
    safe_tag = tag.replace("/", "-").replace(":", "-")
    canonical_main = main_ref or f"{namespace}/{task_id}:{safe_tag}"
    refs: dict[str, str] = {}
    for name, svc in service_map(compose).items():
        if not isinstance(svc, dict) or build_context(svc) is None:
            continue
        if is_canonical_main_build(svc):
            refs[name] = canonical_main
        else:
            refs[name] = f"{namespace}/{task_id}-{name}:{safe_tag}"
    return refs


def registry_namespace_and_tag(main_ref: str | None) -> tuple[str | None, str]:
    """Split a **private-registry** main-image ref into ``(namespace, tag)`` for deriving
    the sub-dir sidecar refs — e.g. ``<registry-host>:5011/lhtb/chess-mate:main`` →
    ``("<registry-host>:5011/lhtb", "main")``.

    This lets a **repinned** main image supply the sidecar namespace, so a multi-service
    compose task's sidecars resolve from the already-resolved main ref without requiring
    a separate namespace source (the ``xrlenv_image_template`` kwarg, if present, drives
    main-image resolution but sub-dir sidecars always derive their namespace from the
    already-repinned main ref). Returns
    ``(None, "main")`` for a docker.io-relative ref (first path segment carries no ``.`` /
    ``:`` / ``localhost`` → not a private registry, so there is no derivable namespace)."""
    if not main_ref:
        return None, "main"
    head = main_ref.split("/", 1)[0]
    if "." not in head and ":" not in head and head != "localhost":
        return None, "main"  # docker.io-relative — no private-registry namespace to split
    last_slash = main_ref.rfind("/")
    last_colon = main_ref.rfind(":")
    if last_colon > last_slash:            # a ':' AFTER the last '/' is the tag
        body, tag = main_ref[:last_colon], main_ref[last_colon + 1:]
    else:
        body, tag = main_ref, "main"
    namespace = body.rsplit("/", 1)[0] if "/" in body else None
    return (namespace or None), (tag or "main")


def _inject_missing_caps(
    service: dict[str, Any], *, cpu: float, mem_mb: int,
) -> None:
    """Inject a default cap for **each** cpu/mem dimension the author left
    unspecified, in place. Symmetric with :func:`sidecar_footprint`, which
    reserves a default for each missing dimension independently: an all-or-nothing
    "any limit → inject neither" rule would reserve (say) default memory in the
    footprint while leaving memory uncapped in the compose, so a sidecar declaring
    only ``cpus`` could still exceed its reserved memory and OOM the node. Reads
    the declared value from either the service-level (``cpus`` / ``mem_limit``) or
    ``deploy.resources.limits`` form; injects the service-level form for the
    absent dimension only — never overriding an author's explicit sizing."""
    declared_cpu, declared_mem = _service_limit_cpu_mem(service)
    if declared_cpu is None:
        service["cpus"] = float(cpu)
    if declared_mem is None:
        service["mem_limit"] = f"{int(mem_mb)}m"


def rewrite_to_image_refs(
    compose: dict[str, Any],
    ref_for_service: Callable[[str], str | None] | dict[str, str],
    *,
    main_service: str = MAIN_SERVICE,
    sidecar_cpu: float = DEFAULT_SIDECAR_CPU,
    sidecar_mem_mb: int = DEFAULT_SIDECAR_MEM_MB,
) -> dict[str, Any]:
    """Return a deep copy of ``compose`` with every ``build:`` service replaced
    by an ``image:`` ref (and ``pull_policy: build`` dropped), plus a per-sidecar
    cgroup cap injected where the author declared none.

    ``ref_for_service`` maps a service name to its image ref — pass the dict from
    :func:`default_image_refs` or a callable. Every service with a ``build:``
    stanza **must** resolve to a non-``None`` ref — a missing dict key *or* a
    callable returning ``None`` raises ``KeyError`` (never a silent
    ``image: None``); a service *without* ``build:`` is rewritten only when its
    name resolves to a non-``None`` ref. This lets a
    caller also repoint services that reference a **local build tag** via
    ``image:`` + ``pull_policy: never`` (e.g. tw_299387's
    ``image: terminalworld-env-299387``) at the pushed registry ref — those don't
    ship a ``build:`` but still must not be pulled by their local name on a node.

    Any rewritten service also has ``pull_policy`` dropped (``build`` is invalid
    once ``build:`` is gone; ``never`` would block pulling the registry ref).

    The cap injection makes the scheduler's footprint reservation *enforced*: an
    uncapped sidecar otherwise can't be held to its reserved share and could OOM
    the node. ``main`` is never capped here — harbor's own resources override
    sizes it.
    """
    is_dict = isinstance(ref_for_service, dict)
    mapping = ref_for_service if is_dict else {}

    def resolve(name: str, *, required: bool) -> str | None:
        if is_dict:
            if name in mapping:
                return mapping[name]
            ref = None
        else:
            ref = ref_for_service(name)  # type: ignore[operator]
        if ref is None and required:
            # A build service MUST resolve to a ref in either mapping form —
            # dropping ``build:`` and writing ``image: None`` would produce a
            # silently-broken compose. Fail loud symmetrically for dict + callable.
            raise KeyError(
                f"no image ref supplied for build service {name!r}",
            )
        return ref

    out = copy.deepcopy(compose)
    for name, svc in service_map(out).items():
        if not isinstance(svc, dict):
            continue
        has_build = svc.get("build") is not None
        ref = resolve(name, required=has_build)
        if has_build or ref is not None:
            svc.pop("build", None)
            svc.pop("pull_policy", None)
            svc["image"] = ref
        if name != main_service:
            _inject_missing_caps(svc, cpu=sidecar_cpu, mem_mb=sidecar_mem_mb)
    return out


def _service_limit_cpu_mem(service: dict[str, Any]) -> tuple[float | None, int | None]:
    """Read a service's declared cpu (cores) + mem (MiB) from ``deploy.resources.
    limits`` or service-level ``cpus`` / ``mem_limit``. ``None`` where absent."""
    cpu: float | None = None
    mem_mb: int | None = None
    deploy = service.get("deploy")
    if isinstance(deploy, dict):
        limits = (deploy.get("resources") or {}).get("limits") or {}
        if limits.get("cpus") is not None:
            cpu = float(limits["cpus"])
        if limits.get("memory") is not None:
            mem_mb = _mem_to_mb(str(limits["memory"]))
    if cpu is None and service.get("cpus") is not None:
        cpu = float(service["cpus"])
    if mem_mb is None and service.get("mem_limit") is not None:
        mem_mb = _mem_to_mb(str(service["mem_limit"]))
    return cpu, mem_mb


def _mem_to_mb(value: str) -> int:
    """Parse a compose memory string (``"512m"`` / ``"2g"`` / ``"1073741824"``)
    to MiB. Bare integers are treated as bytes (compose's rule)."""
    v = value.strip().lower()
    mult = 1
    if v.endswith("b"):
        v = v[:-1]
    if v and v[-1] in "kmg":
        mult = {"k": 1024, "m": 1024**2, "g": 1024**3}[v[-1]]
        v = v[:-1]
    try:
        raw_bytes = float(v) * mult
    except ValueError:
        return 0
    return int(raw_bytes // (1024 * 1024))


def sidecar_footprint(
    compose: dict[str, Any],
    *,
    main_service: str = MAIN_SERVICE,
    default_cpu: float = DEFAULT_SIDECAR_CPU,
    default_mem_mb: int = DEFAULT_SIDECAR_MEM_MB,
) -> tuple[float, int]:
    """Aggregate ``(cpu_cores, mem_mb)`` the **sidecars** contribute to the stack
    footprint (Q2): each service other than ``main`` counts its declared
    ``deploy.resources`` / ``cpus`` / ``mem_limit`` where present, else the flat
    default. The caller adds ``main``'s own declared size to get the whole-stack
    footprint the scheduler reserves via ``place(reserve=...)``."""
    total_cpu = 0.0
    total_mem = 0
    for name, svc in service_map(compose).items():
        if name == main_service or not isinstance(svc, dict):
            continue
        cpu, mem_mb = _service_limit_cpu_mem(svc)
        total_cpu += cpu if cpu is not None else default_cpu
        total_mem += mem_mb if mem_mb is not None else default_mem_mb
    return total_cpu, total_mem


def subnet_claims(compose: dict[str, Any]) -> list[str]:
    """Every pinned subnet CIDR declared under ``networks.*.ipam.config[].subnet``.

    These are what the scheduler treats as node-exclusive (§4.2): a task that
    pins ``172.16.70.0/24`` and hard-codes static IPs the solve.sh reaches can't
    co-locate with another project claiming an overlapping subnet on the same
    node. Empty for service-DNS-only tasks (docker auto-assigns per project →
    unbounded concurrency)."""
    claims: list[str] = []
    networks = compose.get("networks")
    if not isinstance(networks, dict):
        return claims
    for net in networks.values():
        if not isinstance(net, dict):
            continue
        ipam = net.get("ipam")
        if not isinstance(ipam, dict):
            continue
        for entry in ipam.get("config") or []:
            if isinstance(entry, dict) and entry.get("subnet"):
                claims.append(str(entry["subnet"]))
    return claims


# ──────────────────────────────────────────────────────────────────────────────
# Runtime assembly (step 4b) — the effective, image-ref-only compose the cluster
# ships, plus its ensure-present image list. Pure (yaml/stdlib only), so the
# plugin can build the whole document consumer-side without a local docker.
# ──────────────────────────────────────────────────────────────────────────────


def ensure_main_service(
    compose: dict[str, Any],
    *,
    main_ref: str,
    keepalive: list[str] | None = None,
) -> dict[str, Any]:
    """Return a deep copy of ``compose`` with a runnable ``main`` service
    guaranteed, filling **only** the fields the task left absent — never
    overwriting an explicit one.

    This reproduces harbor's base+task compose layering: the base
    (``docker-compose-build.yaml``) contributes ``main`` (image + ``command:
    sleep infinity``) and the task compose layers on top, so an explicit
    ``main.command`` / ``main.image`` in the task wins ("later ``-f`` wins").
    Fill-missing-only semantics mirror that:

    - **no ``main`` service** → inject ``{image: main_ref, command: keepalive}``;
    - ``main`` present with **neither ``image`` nor ``build``** → set ``image`` =
      ``main_ref`` (a bare ``main`` declaring neither; a ``main`` that builds from
      ``.`` or carries a local tag is repointed by the rewrite instead, so its
      ``image`` is left for that stage);
    - ``main`` present with **no ``command``** → set ``command`` = ``keepalive`` so
      it stays alive for ``exec``; **an explicit ``main.command`` is left
      untouched** (byte-for-byte faithful to the task);
    - every other explicit ``main`` field (``environment``, ``working_dir``,
      ``depends_on``, …) is preserved verbatim.

    ``keepalive`` defaults to :data:`MAIN_KEEPALIVE` (harbor's exact base command).
    """
    kb = list(keepalive) if keepalive is not None else list(MAIN_KEEPALIVE)
    out = copy.deepcopy(compose)
    services = out.get("services")
    if not isinstance(services, dict):
        services = {}
        out["services"] = services
    main = services.get(MAIN_SERVICE)
    if not isinstance(main, dict):
        services[MAIN_SERVICE] = {"image": main_ref, "command": kb}
        return out
    if "image" not in main and "build" not in main:
        main["image"] = main_ref
    if "command" not in main:
        main["command"] = kb
    return out


def local_tag_service_names(compose: dict[str, Any]) -> list[str]:
    """Service names that reference a **local build tag** — an ``image:`` with
    ``pull_policy: never`` and **no** ``build:`` stanza.

    Such an image was built locally by harbor (never pushed under that name), so it
    won't exist on a node and must be repointed to a pushed ref. In the
    TerminalWorld corpus the only locally-built image is the task's *own* image
    (tw_299387's ``main`` + ``fake-token`` both use ``image:
    terminalworld-env-299387`` + ``pull_policy: never``), so a caller maps these to
    the canonical ``main`` ref. A public ``image:`` (``postgres:14``, no
    ``pull_policy: never``) is left alone — it resolves to a registry pull.
    Returns names in declaration order (``main`` included if it qualifies)."""
    out: list[str] = []
    for name, svc in service_map(compose).items():
        if not isinstance(svc, dict) or svc.get("build") is not None:
            continue
        pull_policy = str(svc.get("pull_policy", "")).strip().lower()
        if pull_policy == "never" and svc.get("image") is not None:
            out.append(name)
    return out


def image_refs(compose: dict[str, Any]) -> list[str]:
    """Sorted, de-duplicated ``image:`` refs across all services — the
    ensure-present list for a rewritten (image-ref-only) compose (``main`` +
    sidecar refs + public sidecars). A service with no ``image`` is skipped (should
    not occur once :func:`assemble_project` has run)."""
    refs = {
        str(svc["image"])
        for svc in service_map(compose).values()
        if isinstance(svc, dict) and svc.get("image")
    }
    return sorted(refs)


def assemble_project(
    compose: dict[str, Any],
    *,
    task_id: str,
    main_ref: str,
    namespace: str | None = None,
    tag: str = "main",
    main_command: list[str] | None = None,
    sidecar_cpu: float = DEFAULT_SIDECAR_CPU,
    sidecar_mem_mb: int = DEFAULT_SIDECAR_MEM_MB,
) -> tuple[dict[str, Any], list[str]]:
    """Assemble the effective, image-ref-only compose the cluster ships + its
    ensure-present image list. Returns ``(rewritten_doc, images)``. Pure.

    Ties the pieces together:

    1. :func:`ensure_main_service` (fill-missing-only ``main``);
    2. build the ``{service: ref}`` map — :func:`default_image_refs` for every
       ``build:`` service (context ``.`` → ``main_ref``; sub-dir →
       ``<namespace>/<task_id>-<service>:<tag>``, the ref ``build_plan_gen``
       pushed) + local-tag repoints (:func:`local_tag_service_names` → ``main_ref``)
       + ``main`` → ``main_ref``;
    3. :func:`rewrite_to_image_refs` (build→image, ``pull_policy`` dropped,
       per-sidecar cgroup caps injected);
    4. :func:`image_refs` collects the ensure-present list.

    ``namespace`` / ``tag`` name the **sub-dir** build services only. A task **with**
    sub-dir build services requires a non-``None`` ``namespace`` (the caller derives
    it from the repinned main ref via :func:`registry_namespace_and_tag`); passing
    ``None`` there raises ``ValueError`` rather than emitting a ref that won't match
    what was pushed. A task with **no** sub-dir builds needs no ``namespace`` — every
    ``build:``/local-tag service resolves to ``main_ref``.
    """
    with_main = ensure_main_service(compose, main_ref=main_ref, keepalive=main_command)
    subdir = subdir_build_services(with_main)
    if subdir and namespace is None:
        raise ValueError(
            f"task {task_id!r} has sub-dir build service(s) {sorted(subdir)} that "
            f"each need a pushed <namespace>/<task_id>-<service>:<tag> ref, but no "
            f"image namespace was resolved — the repinned main image ref must be a "
            f"private-registry ref with a '{{task_id}}' sub-path so the runtime ref "
            f"matches what build_plan_gen pushed.",
        )
    refs = default_image_refs(
        task_id, with_main, namespace=namespace or "", tag=tag, main_ref=main_ref,
    )
    for name in local_tag_service_names(with_main):
        refs.setdefault(name, main_ref)
    refs.setdefault(MAIN_SERVICE, main_ref)
    rewritten = rewrite_to_image_refs(
        with_main, refs, main_service=MAIN_SERVICE,
        sidecar_cpu=sidecar_cpu, sidecar_mem_mb=sidecar_mem_mb,
    )
    return rewritten, image_refs(rewritten)
