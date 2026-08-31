"""FastAPI admin server (spec 13).

The admin page ownership contract lives in :mod:`xrlenv.admin`; keep the
route handlers here aligned with that matrix. Pages are rendered with
Jinja2 and auto-refreshed via a ``<meta http-equiv="refresh">`` tag
until the planned HTMX/SSE upgrade.

The server takes no live coordinator handles — it opens
:class:`SqliteStateStore` against the on-disk ``state.db`` and walks the
``PlatformJsonlSink`` runs root. WAL mode is concurrent-reader-safe so
the running control plane keeps writing through the same file. This
keeps the admin panel's failure modes orthogonal to the control plane:
a slow render, a 500, or a long-poll never blocks the consumer-facing
path.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import os
import statistics
import threading
import time
import warnings
from collections import Counter
from collections.abc import Callable, Iterable
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import quote

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict, Field, SkipValidation

from xrlenv._version import __version__
from xrlenv.control.state import (
    BuildAssignmentStatus,
    RawRolloutStatus,
    RolloutRecord,
    SqliteStateStore,
)
from xrlenv.control.trajectory_cache import TrajectoryCache, TrajectoryCacheConfig
from xrlenv.image_refs import (
    has_explicit_tag,
    manifest_digest,
    registry_agnostic_ref,
    repo_path,
)
from xrlenv.node.hw_probe import HardwareInfo
from xrlenv.node.trajectory_reader import JsonlTrajectoryReader
from xrlenv.types import RolloutStatus, Trajectory

LOGGER = logging.getLogger(__name__)

_ADMIN_ROOT = Path(__file__).resolve().parent
_TEMPLATE_DIR = _ADMIN_ROOT / "templates"
_STATIC_DIR = _ADMIN_ROOT / "static"

# Cookie session for the browser auth path (B7.4, 2026-06-07). The browser
# signs in via ``POST /login`` (token → HttpOnly cookie) instead of HTTP basic
# auth, so the operator can log out and switch to a different consumer / viewer
# / operator token — impossible under browser-cached basic auth. See
# ``_require_role`` and the ``/login`` + ``/logout`` routes.
_SESSION_COOKIE = "xrlenv_admin_session"
# The token is re-verified against the TokenStore on every request, so a
# revocation still takes effect immediately regardless of cookie lifetime; the
# max-age just spares the operator a daily re-login.
_SESSION_MAX_AGE_S = 7 * 24 * 3600
# Roles that may browse (GET) the panel. Write (POST) routes stay operator-only.
_ADMIN_READ_ROLES = {"consumer", "viewer", "operator"}

# Pages whose GET is operator-only (not just write routes). ``/users`` exposes
# every tenant's activity, so a per-user ``consumer`` / ``viewer`` token must
# not read it — it's gated like a write route even though it's a GET.
_OPERATOR_ONLY_GET_PREFIXES = ("/users",)


def _is_operator_only_path(path: str) -> bool:
    """Whether ``path``'s GET requires the operator role (not just read)."""
    return any(
        path == prefix or path.startswith(prefix + "/")
        for prefix in _OPERATOR_ONLY_GET_PREFIXES
    )
RolloutPageSize = Literal[32, 64, 128, 256]
_ROLLOUT_PAGE_SIZE_OPTIONS: tuple[RolloutPageSize, ...] = (32, 64, 128, 256)
_DEFAULT_ROLLOUT_PAGE_SIZE: RolloutPageSize = 32
_MAX_ROLLOUT_PAGE_SIZE = max(_ROLLOUT_PAGE_SIZE_OPTIONS)
ImagePageSize = Literal[25, 50, 100, 200]
_IMAGE_PAGE_SIZE_OPTIONS: tuple[ImagePageSize, ...] = (25, 50, 100, 200)
_DEFAULT_IMAGE_PAGE_SIZE: ImagePageSize = 50
_MAX_IMAGE_PAGE_SIZE = max(_IMAGE_PAGE_SIZE_OPTIONS)
ImageIncludeFilter = Literal["default", "intermediates", "foreign", "all"]
_IMAGE_INCLUDE_OPTIONS: tuple[tuple[ImageIncludeFilter, str], ...] = (
    ("default", "xrlenv images only"),
    ("intermediates", "+ intermediates"),
    ("foreign", "+ foreign images"),
    ("all", "all images"),
)
_IMAGE_FREE_DISK_CRITICAL_BYTES = 10 * 1024**3
_IMAGE_FREE_DISK_WARN_BYTES = 50 * 1024**3


# ──────────────────────────────────────────────────────────────────────────────
# Lazy-import helpers — xrlenv.cli.commands transitively pulls
# xrlenv.control.distributed_runtime, which in turn imports this module.
# Lazy-importing breaks the cycle while still letting us reuse the proven
# CLI helpers verbatim.
# ──────────────────────────────────────────────────────────────────────────────


def _load_nodes_yaml_lazy(path: Path) -> list[dict[str, Any]]:
    from xrlenv.cli.commands import _load_nodes_yaml

    return _load_nodes_yaml(path)


def parse_duration_lazy(spec: str) -> float:
    from xrlenv.cli.commands import parse_duration

    return parse_duration(spec)


class AdminBindError(RuntimeError):
    """Raised when the configured bind violates the spec-19 admin guard."""


class AdminServerConfig(BaseModel):
    """Runtime config for :class:`AdminServer`."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    state_db: Path
    runs_root: Path
    nodes_yaml: Path | None = None
    host: str = "127.0.0.1"
    port: int = 8080
    refresh_interval_s: int = 0
    """Default auto-reload cadence (seconds) for the rendered list/overview
    pages, via meta-refresh. ``0`` (the default) means **off** — pages do not
    auto-refresh until the operator picks an interval from the per-page
    auto-refresh selector (or passes ``?refresh=<seconds>``). Set a positive
    value to make every page auto-refresh by default."""
    rollout_page_size: RolloutPageSize = Field(
        default=_DEFAULT_ROLLOUT_PAGE_SIZE,
        ge=1,
        le=_MAX_ROLLOUT_PAGE_SIZE,
    )
    """Default number of rollout rows shown on each admin list page."""
    allow_public: bool = False
    """When False, binding to a non-loopback address raises
    :class:`AdminBindError`. Spec 19 phase-0 guard."""
    trajectory_cache_config: TrajectoryCacheConfig | None = None
    """Optional override for the on-disk trajectory cache (LRU + TTL).
    When ``None`` the spec-17 defaults apply: 5 GB budget, 1 h TTL."""
    node_lookup: Any = None
    """Optional ``Callable[[str], NodeTransport | None]``. When provided,
    cache misses dispatch a ``FetchTrajectoryCommand`` to the rollout's
    owning node via the bidi gRPC stream (true distributed multi-host).
    When ``None``, the cache falls back to reading the local
    ``runs_root`` — correct for phase-0 single-host."""
    build_coordinator: Any = None
    """Optional :class:`xrlenv.control.build_coordinator.BuildCoordinator`.
    When wired, the ``POST /api/build/apply`` + ``GET /api/build/plans/<id>``
    endpoints route operator build-apply requests through this
    coordinator (which in DistributedRuntime fan-out via
    :class:`GrpcNodeBuilder`). When ``None``, those endpoints return
    503 — the admin server is reachable but no cluster-wide build
    dispatch is available."""
    token_store: Any = None
    """Optional :class:`xrlenv.control.security.TokenStore`. When wired,
    the build-apply API endpoints require an operator-role bearer
    token; missing/invalid token → 401. When ``None`` the endpoints
    fall back to localhost-only access (loopback bind already gates
    public exposure via :class:`AdminBindError`)."""

    # ── Cluster-info banner on the overview page (cosmetic) ──────────────────
    # All optional; each is shown only when set. Populated at the ``xrlenv up``
    # wiring point (see ``build_distributed_runtime``) — the admin server can't
    # discover them on its own (the gRPC bind + the registry env vars live
    # outside ``AdminServerConfig``).
    control_plane_endpoint: str | None = None
    """The ``host:port`` node agents dial for the control-plane gRPC stream
    (the resolved advertise host, not a ``0.0.0.0`` wildcard). Shown on the
    overview so an operator can copy it into a node's bootstrap."""
    registry_mirror: str | None = None
    """The pull-through registry mirror configured for the cluster
    (``XRLENV_REGISTRY_MIRROR``), or ``None`` when nodes pull direct."""
    private_registry: str | None = None
    """The private / custom build registry images are pushed to + pulled from
    (``XRLENV_PRIVATE_REGISTRY``), or ``None`` when unset."""


@dataclass(frozen=True)
class RolloutPage:
    records: list[RolloutRecord]
    page: int
    page_size: int
    has_next: bool


def _resolve_rollout_page_size(
    requested: int | None, default: int,
) -> int:
    page_size = requested if requested is not None else default
    if page_size not in _ROLLOUT_PAGE_SIZE_OPTIONS:
        raise HTTPException(
            status_code=422,
            detail=(
                "page_size must be one of "
                f"{', '.join(str(v) for v in _ROLLOUT_PAGE_SIZE_OPTIONS)}"
            ),
        )
    return page_size


def _resolve_image_page_size(requested: int | None) -> int:
    page_size = requested if requested is not None else _DEFAULT_IMAGE_PAGE_SIZE
    if page_size not in _IMAGE_PAGE_SIZE_OPTIONS:
        raise HTTPException(
            status_code=422,
            detail=(
                "page_size must be one of "
                f"{', '.join(str(v) for v in _IMAGE_PAGE_SIZE_OPTIONS)}"
            ),
        )
    return page_size


def _resolve_image_include(
    include: str | None,
    *,
    show_intermediate: int,
    show_external: int,
) -> tuple[ImageIncludeFilter, bool, bool]:
    """Resolve the operator's ``?include=`` choice into the
    three-tuple the snapshot builder consumes:
    ``(include_filter_name, hide_intermediate, xrlenv_only)``.

    First-time-operator UX (2026-05-11): the no-flag default is
    ``"all"``. Operators bootstrap a cluster, pull a bunch of
    benchmark images via ``docker pull`` or the source-build flow,
    open the panel, and expect to see them. The previous default
    (``"default"`` = "xrlenv-only-no-intermediates-no-foreign")
    hid every image not registered in the template catalog —
    common scenario, the operator sees "Showing 0 of N" and thinks
    the panel is broken. Showing everything by default and letting
    them filter down is the right shape for first-visit.

    Explicit ``include=default`` still works the same (xrlenv-only),
    so any operator who bookmarks a filtered URL keeps it.
    """
    if include in {None, ""}:
        show_i = bool(show_intermediate)
        show_e = bool(show_external)
        if show_i and show_e:
            return "all", False, False
        if show_i:
            return "intermediates", False, True
        if show_e:
            return "foreign", True, False
        # No filter set at all → show everything. Operators filter
        # down via the include dropdown when they want a slice.
        return "all", False, False

    if include == "default":
        return "default", True, True
    if include == "intermediates":
        return "intermediates", False, True
    if include == "foreign":
        return "foreign", True, False
    if include == "all":
        return "all", False, False
    raise HTTPException(
        status_code=422,
        detail=(
            "include must be one of "
            f"{', '.join(value for value, _label in _IMAGE_INCLUDE_OPTIONS)}"
        ),
    )


def _admin_auth_challenge_headers() -> dict[str, str]:
    """``WWW-Authenticate`` for programmatic 401s. We advertise ``Bearer``
    (the CLI transport), deliberately **not** ``Basic``: a ``Basic`` challenge
    makes the browser pop its native credential dialog, and the browser then
    caches those creds and replays them on every request with no
    app-controllable logout — the exact reason an operator couldn't switch
    tokens. The browser signs in through the cookie-session ``/login`` page
    instead (B7.4)."""
    return {"WWW-Authenticate": 'Bearer realm="xrlenv-admin"'}


def _safe_next(raw: str | None) -> str:
    """Sanitize a ``?next=`` post-login redirect target to a same-site path.

    Open-redirect guard: only a path beginning with a single ``/`` is honored.
    A protocol-relative ``//evil.example`` (which a browser resolves to another
    host) or any absolute URL falls back to ``/``.
    """
    if not raw or not raw.startswith("/") or raw.startswith("//"):
        return "/"
    return raw


def _wants_html(request: Request) -> bool:
    """True when the caller looks like a browser navigation (GET that accepts
    HTML), so an auth failure should redirect to ``/login`` rather than return
    a JSON 401. A CLI / ``curl`` caller (``Accept: */*``) gets the JSON 401."""
    if request.method != "GET":
        return False
    return "text/html" in request.headers.get("accept", "")


def _sibling_loopback(host: str) -> str | None:
    """Return the sibling-family loopback for a loopback ``host``,
    or ``None`` for non-loopback / wildcard binds.

    The "sibling" pairing:

    - ``127.0.0.1``  / ``localhost`` / ``""`` → ``"::1"``
    - ``::1``                                  → ``"127.0.0.1"``
    - anything else                            → ``None``

    Used by :meth:`AdminServer.start` to bring up a second uvicorn
    listener so the admin panel is reachable from clients that
    resolved ``localhost`` to the other family (most commonly: VS
    Code Remote SSH's port-forwarding, which dials whichever family
    the remote OS prefers — ::1 first on Linux). Without the
    sibling bind, operators using VS Code's port-forward see a
    silent ``ERR_EMPTY_RESPONSE`` even with xrlenv up running.

    Wildcard binds (``0.0.0.0``, ``::``) return ``None``: those
    already cover both families on Linux, and the bind guard refuses
    them outright without auth tokens anyway.
    """
    if not host:
        return "::1"
    if host.lower() == "localhost":
        return "::1"
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return None
    if not ip.is_loopback:
        return None
    if ip.version == 4:
        return "::1"
    return "127.0.0.1"


def _format_host_for_url(host: str) -> str:
    """``::1`` needs to be wrapped in brackets when emitted inside a
    URL (``http://[::1]:8080``); IPv4 stays as-is."""
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return host
    if ip.version == 6:
        return f"[{host}]"
    return host


def _is_loopback(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host in {"localhost", ""}


def _enforce_bind_guard(cfg: AdminServerConfig) -> None:
    """Spec 19 §"Admin-bind guard": loopback binds always allowed. A
    public bind is allowed only when (a) the operator explicitly opted
    into ``--admin-allow-public`` AND (b) a :class:`TokenStore` is wired
    with at least one shared ``viewer`` / ``operator`` role-token
    registered. The second condition rejects unauthenticated *and*
    management-less public binds: a per-user ``consumer`` token now
    grants read-only, owner-scoped access, but the guard still wants an
    explicit ``viewer``/``operator`` present before public exposure so
    the panel has a management-capable identity (and isn't exposed with
    only an empty store). The operator normally issues an operator token
    anyway, so this passes in the common case; the guard just refuses the
    empty / no-admin-token misconfiguration up front instead of a
    mysteriously locked-out browser.
    """
    if _is_loopback(cfg.host):
        return
    if not cfg.allow_public:
        raise AdminBindError(
            f"refusing to bind admin server on public address {cfg.host!r}: "
            "pass --admin-allow-public to override "
            "(unauthenticated public binds remain refused regardless)."
        )
    store = cfg.token_store
    if store is None or getattr(store, "is_empty", True):
        raise AdminBindError(
            f"refusing public admin bind on {cfg.host!r} with no auth "
            "configured. Issue at least one viewer or operator token "
            "(`xrlenv tokens issue viewer`, `xrlenv tokens issue operator`) "
            "and wire the TokenStore before exposing the admin server "
            "outside loopback. The SSH-tunnel workaround "
            "(`ssh -L 8080:localhost:8080 user@control-plane`) is still a "
            "safe alternative when you don't want to manage tokens."
        )
    admin_roles = {"viewer", "operator"}
    known_roles = set(getattr(store, "known_roles", ()))
    if not (admin_roles & known_roles):
        raise AdminBindError(
            f"refusing public admin bind on {cfg.host!r}: the wired "
            "TokenStore holds no admin-capable tokens "
            f"(found roles={sorted(known_roles)!r}; admin routes require "
            "viewer or operator). Issue one via "
            "`xrlenv tokens issue viewer` or "
            "`xrlenv tokens issue operator` before exposing the panel."
        )


_REFRESH_OPTIONS_S = (0, 5, 10, 30, 60)
# Cluster-health page: node signals, long-running/queued sessions, failure rate.
_HEALTH_CHECK_COUNT = 3
# Coarse display thresholds for highlighting an unhealthy node row — NOT
# the Stage-3 admission controller, just a 5-second-glance signal.
_HEALTH_HEARTBEAT_STALE_S = 30.0
_HEALTH_CREATE_P95_HIGH_MS = 30_000.0
# Age past which a session is surfaced in the long-running/queued triage table.
# Deliberately coarse (2x the 1h default hard deadline): this is an age
# heuristic for the operator to eyeball, NOT a failure signal — long-horizon
# rollouts and persistent substrate containers legitimately exceed it, so it
# does not flip the health banner.
_HEALTH_LONG_RUNNING_AGE_S = 2 * 3600.0


def _refresh_context(
    request: Request,
    cfg: AdminServerConfig,
    *,
    default_s: int | None = None,
) -> dict[str, Any]:
    """Per-page auto-refresh controls for list/overview pages.

    The operator can override the configured default with ``?refresh=``.
    ``off`` / ``0`` disables the meta-refresh tag for the current URL.
    Rollout-detail intentionally does not use this helper; artifact
    inspection pages stay no-refresh to preserve scroll/search state.
    """
    effective_default_s = cfg.refresh_interval_s if default_s is None else default_s
    refresh_s = _resolve_refresh_s(request, effective_default_s)
    options = list(_REFRESH_OPTIONS_S)
    for seconds in (cfg.refresh_interval_s, effective_default_s):
        if seconds not in options and seconds > 0:
            options.append(seconds)
    options.sort()
    return {
        "refresh_s": refresh_s,
        "refresh_default_s": effective_default_s,
        "refresh_configurable": True,
        "refresh_options_s": options,
    }


def _resolve_refresh_s(request: Request, default_s: int) -> int:
    raw = request.query_params.get("refresh")
    if raw is None:
        return default_s
    if raw.lower() == "off":
        return 0
    try:
        value = int(raw)
    except ValueError:
        return default_s
    if value < 0:
        return default_s
    return value


# Registry-host normalization for calibrate (and evict) ref matching now
# lives in :mod:`xrlenv.image_refs` so the node-side eviction matcher uses
# the identical rule. Re-exported under the original private name so the
# in-module call sites + tests keep importing it from here.
_registry_agnostic_ref = registry_agnostic_ref
_repo_path = repo_path
_has_explicit_tag = has_explicit_tag
_manifest_digest = manifest_digest


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────────────────────


def build_admin_app(cfg: AdminServerConfig) -> FastAPI:
    """Construct the FastAPI app the admin server runs.

    The factory is split out from :class:`AdminServer` so tests can
    drive it through ``fastapi.testclient.TestClient`` without standing
    up uvicorn.
    """
    if not _TEMPLATE_DIR.exists():
        raise RuntimeError(f"admin templates dir missing: {_TEMPLATE_DIR}")

    templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))
    started_at = time.time()
    cache = TrajectoryCache(cfg.trajectory_cache_config)
    local_reader = JsonlTrajectoryReader(cfg.runs_root)
    # Stale-while-revalidate cache for the cluster image-rows fan-out so the
    # /images pages stay responsive while a heavy build makes each node's
    # report_images slow. One instance per app; request-triggered.
    image_snapshot = _ImageRowsSnapshot()

    async def _fetch_trajectory(rollout_id: str) -> Trajectory:
        """Cache fetch_fn with two-source resolution.

        Phase-0 / phase-1 architecture: the platform-jsonl sink runs
        on the control plane and writes ``<runs_root>/<date>/<rid>/
        trajectory.jsonl`` regardless of which node ran the sandbox.
        So in *every* topology the admin runs against, the body
        should normally be on the control plane's local disk.

        Resolution order:

        1. **Local-disk read first.** Cheap (one ``open``); always
           the right answer when the sink wrote here. Validates that
           the user's empirical observation ("all trajectories live
           under ``~/.xrlenv/runs/``") matches the admin's read path.
        2. **Bidi fetch fallback** when local read raises
           ``FileNotFoundError`` AND ``cfg.node_lookup`` is wired
           AND ``node_id`` is known. Reserved for future multi-host
           topologies where the body legitimately lives node-side
           (spec-18 sandbox sessions, node-side sinks). On bidi
           errors (node disconnected, gRPC failure), we re-raise the
           original ``FileNotFoundError`` rather than the bidi
           exception so the caller's ``except FileNotFoundError``
           branch handles it as "missing" — same shape it would have
           seen with no fallback wired.

        Pre-fix the order was reversed: bidi-first, then fallback to
        local *only when ``node_lookup`` was unwired*. That meant a
        transient bidi error in distributed mode threw away a
        perfectly good local file, surfacing as "No trajectory body
        on disk" on a page where ``meta.json`` + ``coordinator.log``
        + verifier files were all visible. Reported by the user
        2026-05-01.
        """
        try:
            local: Trajectory = await asyncio.to_thread(
                local_reader.read_range, rollout_id,
            )
            return local
        except FileNotFoundError as local_exc:
            # File isn't in the control plane's runs_root. Try the
            # bidi path before giving up — covers multi-host setups
            # where the sink is node-side.
            node_id: str | None = None
            if cfg.state_db.exists():
                store = SqliteStateStore(cfg.state_db, read_only=True)
                try:
                    try:
                        record = store.get_rollout(rollout_id)
                        node_id = record.node_id
                    except KeyError:
                        pass
                finally:
                    store.close()
            if cfg.node_lookup is not None and node_id is not None:
                transport = cfg.node_lookup(node_id)
                if transport is not None:
                    try:
                        fetched: Trajectory = await transport.fetch_trajectory(
                            rollout_id,
                        )
                        return fetched
                    except Exception as bidi_exc:
                        # Re-raise the local FileNotFoundError so the
                        # caller's missing-trajectory branch fires
                        # (instead of the generic Exception path that
                        # logs "trajectory fetch failed"). The bidi
                        # error becomes the FileNotFoundError's
                        # ``__cause__`` for log-trail visibility.
                        raise local_exc from bidi_exc
            raise

    app = FastAPI(
        title="XRLEnv admin",
        description="Phase-0 read-only cluster dashboard (spec 13).",
        version=__version__,
    )
    if _STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    # ── Auth middleware (B7.3, refined 2026-05-11) ───────────────────────────
    #
    # **Loopback binds bypass auth entirely.** When the admin server is
    # bound to a loopback address (127.0.0.1 / localhost / ::1 — the
    # default), the protection boundary is the operator's SSH tunnel,
    # not HTTP basic auth. Adding basic auth on top of a loopback bind
    # gives zero security uplift (anyone who reached the loopback
    # already passed SSH) and inflicts a real UX cost (Chrome /
    # Safari auto-upgrade ``http://localhost`` to https, which our
    # HTTP-only server can't speak — the failure surfaces as a
    # mysterious "Internal Server Error" in the browser with nothing
    # in the server log).
    #
    # On a **public bind** (``--admin-allow-public``), auth engages:
    #
    #   - ``GET`` routes accept ``consumer`` / ``viewer`` / ``operator``
    #     (read-only browsing — nodes, builds, rollouts, images,
    #     ``GET /api/build/plans/<id>``). A per-user ``consumer`` token
    #     gets a read-only view scoped to its own ``owner_id``.
    #   - Every other method (currently only ``POST``) needs
    #     ``operator`` (writes — apply / cancel / calibrate, plus
    #     future destructive admin actions).
    #   - ``/healthz`` / ``/static/*`` / ``/login`` / ``/logout`` stay
    #     open: load balancers and the sign-in page need them without
    #     credentials.
    #
    # **Two transports, by caller.** Programmatic callers (the CLI) send
    # ``Authorization: Bearer <token>``. Browsers sign in through the
    # cookie-session ``/login`` page (B7.4) — NOT HTTP basic auth, which
    # the browser caches and replays with no app-controllable logout (so
    # an operator who signed in as one consumer could never switch to
    # another). When an unauthenticated *browser* GET hits a gated route
    # we 303-redirect to ``/login?next=…`` instead of returning a bare
    # 401, so the operator lands on the sign-in form rather than a JSON
    # blob or the browser's native basic-auth popup.
    #
    # The bind guard (``_enforce_bind_guard``) ensures a public bind
    # is only allowed when an admin-capable token is wired, so
    # "public + auth off" is rejected at startup, not at request
    # time — public binds can't accidentally serve unauthenticated.

    bind_is_loopback = _is_loopback(cfg.host)

    @app.middleware("http")
    async def _admin_auth_mw(
        request: Request,
        call_next: Callable[[Request], Any],
    ) -> Any:
        if bind_is_loopback:
            # Loopback: SSH-tunnel is the protection boundary; auth
            # is redundant and harmful (HTTPS-upgrade gotcha).
            return await call_next(request)
        path = request.url.path
        if (
            path in ("/healthz", "/login", "/logout")
            or path.startswith("/static")
        ):
            return await call_next(request)
        try:
            if request.method == "GET" and not _is_operator_only_path(path):
                _require_read_role(request)
            else:
                _require_operator(request)
        except HTTPException as exc:
            if exc.status_code == 401 and _wants_html(request):
                # Browser navigation with no/expired session → send the
                # operator to the sign-in form (preserving where they were
                # headed), not a JSON 401.
                nxt = request.url.path
                if request.url.query:
                    nxt = f"{nxt}?{request.url.query}"
                return RedirectResponse(
                    url=f"/login?next={quote(nxt, safe='')}",
                    status_code=303,
                )
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail},
                headers=exc.headers or None,
            )
        return await call_next(request)

    # ── /healthz ─────────────────────────────────────────────────────────────

    @app.get("/healthz", response_class=PlainTextResponse)
    async def healthz() -> str:
        return "ok"

    # ── /login + /logout (cookie session, B7.4) ─────────────────────────────
    #
    # The browser auth path is a logout-able cookie session, not HTTP basic
    # auth. ``POST /login`` verifies a pasted token against the TokenStore and
    # drops it into an HttpOnly cookie; ``_require_role`` re-verifies that
    # cookie on every later request (so a revoked token stops working at once).
    # ``POST /logout`` clears the cookie — the whole point of the feature: the
    # operator can sign out and switch to a different consumer / viewer /
    # operator token, which browser-cached basic auth made impossible. Both
    # routes are exempt from the auth middleware (you must reach the sign-in
    # form while signed out) and short-circuit on loopback (no auth there).

    def _verify_cookie(request: Request) -> Any:
        store = cfg.token_store
        if store is None:
            return None
        store.maybe_reload()
        if store.is_empty:
            return None
        cookie_token = request.cookies.get(_SESSION_COOKIE)
        if not cookie_token:
            return None
        return store.verify(cookie_token)

    @app.get("/login", response_class=HTMLResponse)
    async def login_form(request: Request, next: str = "/") -> Any:
        target = _safe_next(next)
        if bind_is_loopback:
            # Loopback bypasses auth entirely — there is nothing to sign into.
            return RedirectResponse(target, status_code=303)
        if _verify_cookie(request) is not None:
            return RedirectResponse(target, status_code=303)
        resp = templates.TemplateResponse(
            request, "login.html", {"next": target, "error": None},
        )
        # Clear any stale / revoked cookie so the form starts from a clean slate.
        if request.cookies.get(_SESSION_COOKIE):
            resp.delete_cookie(_SESSION_COOKIE, path="/")
        return resp

    @app.post("/login")
    async def login_submit(request: Request) -> Any:
        # Parse the urlencoded form body by hand so we don't pull in
        # python-multipart (only a transitive dep) just for two fields.
        from urllib.parse import parse_qs

        raw = (await request.body()).decode("utf-8", "replace")
        form = parse_qs(raw, keep_blank_values=True)
        token = (form.get("token", [""])[0]).strip()
        target = _safe_next(form.get("next", ["/"])[0])
        store = cfg.token_store
        if store is not None:
            store.maybe_reload()
        identity = (
            store.verify(token)
            if (store is not None and not store.is_empty and token)
            else None
        )
        if identity is None or identity.role not in _ADMIN_READ_ROLES:
            return templates.TemplateResponse(
                request,
                "login.html",
                {
                    "next": target,
                    "error": (
                        "That token isn't valid for the admin panel. Paste a "
                        "consumer, viewer, or operator token."
                    ),
                },
                status_code=401,
            )
        resp = RedirectResponse(target, status_code=303)
        resp.set_cookie(
            _SESSION_COOKIE,
            token,
            max_age=_SESSION_MAX_AGE_S,
            httponly=True,
            samesite="lax",
            path="/",
        )
        return resp

    @app.post("/logout")
    async def logout() -> Any:
        resp = RedirectResponse("/login", status_code=303)
        resp.delete_cookie(_SESSION_COOKIE, path="/")
        return resp

    # ── / (overview) ─────────────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def overview(request: Request) -> HTMLResponse:
        snapshot = await _gather_overview(cfg, started_at)
        return templates.TemplateResponse(
            request, "overview.html",
            {
                **snapshot,
                "active_page": "overview",
                **_refresh_context(request, cfg),
            },
        )

    # ── /nodes ───────────────────────────────────────────────────────────────

    @app.get("/nodes", response_class=HTMLResponse)
    async def nodes(request: Request) -> HTMLResponse:
        rows = await _gather_nodes(cfg)
        dist = await asyncio.to_thread(_node_distribution_blocking, cfg, rows)
        return templates.TemplateResponse(
            request, "nodes.html",
            {
                "rows": rows,
                "dist": dist,
                "active_page": "nodes",
                **_refresh_context(request, cfg),
            },
        )

    # ── /users (operator-only) ───────────────────────────────────────────────
    #
    # Per-tenant raw-rollout scoreboard: total / released / failed /
    # cancelled / reaped + active + success%, grouped by ``owner_id``.
    # Operator-gated in the auth middleware (see ``_is_operator_only_path``)
    # because it exposes every tenant's activity, which a per-user
    # ``consumer`` token must not see.

    @app.get("/users", response_class=HTMLResponse)
    async def users(request: Request) -> HTMLResponse:
        data = await asyncio.to_thread(_users_blocking, cfg)
        return templates.TemplateResponse(
            request, "users.html",
            {
                "rows": data["rows"],
                "totals": data["totals"],
                # Thread the cumulative-tally provenance into the template so
                # the H4 banner can render the real inception timestamp and the
                # raw-rollout retention window (span_start..span_end) instead of
                # the "since lifetime tracking was enabled" fallback text.
                "span_start": data["span_start"],
                "span_end": data["span_end"],
                "inception": data["inception"],
                "active_page": "users",
                **_refresh_context(request, cfg),
            },
        )

    # ── /rollouts ────────────────────────────────────────────────────────────
    #
    # P1.7.B.3 (UX split): /rollouts is a dropdown nav; default
    # landing is /rollouts/raw (case-2/3 evaluation harnesses,
    # the platform's primary audience under the slim pivot).
    # /rollouts/template is the case-1 trainer-driven view, kept
    # for trainer integrations that still depend on it.

    @app.get("/rollouts", include_in_schema=False)
    async def rollouts_root(request: Request) -> RedirectResponse:
        # Bare /rollouts → default landing (case-2/3 raw rollouts).
        # Preserve query string so URLs like
        # ``/rollouts?raw_status=failed`` redirect cleanly to
        # ``/rollouts/raw?raw_status=failed`` rather than dropping
        # the filter on the floor.
        target = "/rollouts/raw"
        if request.url.query:
            target = f"{target}?{request.url.query}"
        return RedirectResponse(target, status_code=302)

    @app.get("/rollouts/raw", response_class=HTMLResponse)
    async def rollouts_raw(
        request: Request,
        raw_status: str | None = Query(
            None,
            description=(
                "Filter by status "
                "(acquiring | running | released | cancelled | failed | reaped)"
            ),
        ),
        status: str | None = Query(
            None,
            description=(
                "Alias for raw_status — accepted so legacy / "
                "externally-bookmarked URLs that use ?status=… still "
                "filter correctly. raw_status wins if both are set."
            ),
        ),
        since: str | None = Query(
            None,
            description="Created within DURATION (e.g. 5m / 2h / 1d)",
        ),
        task_key: str | None = Query(
            None,
            description=(
                "Filter by scheduler anti-affinity key (e.g. an "
                "instance_id). Sourced from the acquire_container "
                "kwarg or the xrlenv.task_key docker label."
            ),
        ),
        group_id: str | None = Query(
            None,
            description=(
                "Filter by operator-supplied grouping key (e.g. a "
                "harness run_id). Sourced from the xrlenv.group_id "
                "docker label."
            ),
        ),
        page: int = Query(1, ge=1, description="One-indexed page"),
        page_size: int | None = Query(
            None, ge=1, le=_MAX_ROLLOUT_PAGE_SIZE,
            description="Rows per page",
        ),
    ) -> HTMLResponse:
        effective_page_size = _resolve_rollout_page_size(
            page_size, cfg.rollout_page_size,
        )
        # Accept ``?status=`` as an alias for the canonical
        # ``?raw_status=``: FastAPI silently drops unknown query
        # params, so without this fall-through callers using the
        # short name (including the overview card link in older
        # builds) would land on an unfiltered listing — surfacing
        # as e.g. 256 "running" rows when the DB has 38.
        if raw_status is None:
            raw_status = status
        since_after = _parse_since(since)  # None on bad input
        # Empty-string query params arrive as ``""`` (default-when-
        # submitted-with-blank-input from the filter form); coerce to
        # None so the SQL clauses don't filter on the empty string.
        task_key = task_key or None
        group_id = group_id or None
        offset = (page - 1) * effective_page_size
        # Fetch one extra row to detect "has next page" without
        # a separate COUNT query.
        records, total = await asyncio.to_thread(
            _list_raw_rollouts_paginated_blocking,
            cfg, raw_status, since_after, task_key, group_id,
            effective_page_size + 1, offset, _caller_owner_id(request),
        )
        has_next = len(records) > effective_page_size
        records = records[:effective_page_size]
        now = time.time()
        refresh_ctx = _refresh_context(request, cfg)
        refresh_ctx["refresh_url"] = str(
            request.url.include_query_params(
                page=page, page_size=effective_page_size,
            ),
        )
        return templates.TemplateResponse(
            request, "rollouts_raw.html",
            {
                "records": records,
                "raw_status": raw_status,
                "raw_statuses": list(_RAW_ROLLOUT_STATUSES),
                "since": since,
                "task_key": task_key,
                "group_id": group_id,
                "page": page,
                "page_size": effective_page_size,
                "page_size_options": _ROLLOUT_PAGE_SIZE_OPTIONS,
                "has_prev": page > 1,
                "has_next": has_next,
                "prev_page_url": str(
                    request.url.include_query_params(
                        page=max(1, page - 1),
                        page_size=effective_page_size,
                    ),
                ),
                "next_page_url": str(
                    request.url.include_query_params(
                        page=page + 1,
                        page_size=effective_page_size,
                    ),
                ),
                "total": total,
                "now": now,
                "active_page": "rollouts_raw",
                **refresh_ctx,
            },
        )

    @app.get("/rollouts/template", response_class=HTMLResponse)
    async def rollouts_template(
        request: Request,
        status: str | None = Query(None, description="Filter by status enum"),
        template: str | None = Query(None, description="Filter by template name"),
        since: str | None = Query(
            None,
            description="Created within DURATION (e.g. 5m)",
        ),
        page: int = Query(1, ge=1, description="One-indexed rollout page"),
        page_size: int | None = Query(
            None, ge=1, le=_MAX_ROLLOUT_PAGE_SIZE,
            description="Rows per page",
        ),
    ) -> HTMLResponse:
        effective_page_size = _resolve_rollout_page_size(
            page_size, cfg.rollout_page_size,
        )
        rollout_page = await _gather_rollouts(
            cfg,
            status=status,
            template=template,
            since=since,
            page=page,
            page_size=effective_page_size,
            owner_id=_caller_owner_id(request),
        )
        records = rollout_page.records[:rollout_page.page_size]
        has_next = rollout_page.has_next or len(rollout_page.records) > rollout_page.page_size
        now = time.time()
        durations = await asyncio.to_thread(
            _rollout_durations_blocking, cfg, records, now,
        )
        refresh_ctx = _refresh_context(request, cfg)
        refresh_ctx["refresh_url"] = str(
            request.url.include_query_params(
                page=rollout_page.page,
                page_size=rollout_page.page_size,
            )
        )
        return templates.TemplateResponse(
            request, "rollouts.html",
            {
                "records": records,
                "durations": durations,
                "status": status, "template": template, "since": since,
                "page": rollout_page.page,
                "page_size": rollout_page.page_size,
                "page_size_options": _ROLLOUT_PAGE_SIZE_OPTIONS,
                "has_prev": rollout_page.page > 1,
                "has_next": has_next,
                "prev_page_url": str(
                    request.url.include_query_params(
                        page=max(1, rollout_page.page - 1),
                        page_size=rollout_page.page_size,
                    )
                ),
                "next_page_url": str(
                    request.url.include_query_params(
                        page=rollout_page.page + 1,
                        page_size=rollout_page.page_size,
                    )
                ),
                "statuses": [s.value for s in RolloutStatus],
                "now": now,
                "active_page": "rollouts_template",
                **refresh_ctx,
            },
        )

    # ── /raw-rollouts/{id} ─────────────────────────────────────────────────

    @app.get("/raw-rollouts/{rollout_id}", response_class=HTMLResponse)
    async def raw_rollout_detail(
        request: Request, rollout_id: str,
    ) -> HTMLResponse:
        record = await asyncio.to_thread(
            _get_raw_rollout_blocking, cfg, rollout_id,
        )
        if record is None or _owner_forbidden(
            _caller_owner_id(request), getattr(record, "owner_id", None),
        ):
            # A scoped caller gets the same 404 as a truly-missing id, so the
            # panel never reveals that another tenant's rollout exists.
            raise HTTPException(
                status_code=404,
                detail=f"raw rollout {rollout_id} not found",
            )
        # Best-effort artifact-path resolution against the
        # control-plane host's filesystem. The cluster never
        # uploads artifacts; if the consumer ran on a different
        # machine, the path is just a string here.
        artifact_status, artifact_listing = (
            _resolve_artifact_path(record.artifact_path)
            if record.artifact_path else ("missing", [])
        )
        return templates.TemplateResponse(
            request, "raw_rollout_detail.html",
            {
                "record": record,
                "created_at_iso": _iso(record.created_at),
                "finished_at_iso": (
                    _iso(record.finished_at)
                    if record.finished_at else None
                ),
                "artifact_status": artifact_status,
                "artifact_listing": artifact_listing,
                "active_page": "rollouts_raw",
                **_refresh_context(request, cfg),
            },
        )

    # ── /raw-rollouts/{id}/artifact/{file_path} (P1.7.B.3) ─────────────────

    @app.get("/raw-rollouts/{rollout_id}/artifact/{file_path:path}")
    async def raw_rollout_artifact(
        request: Request, rollout_id: str, file_path: str,
    ) -> Any:
        """Stream a single artifact file from the consumer-recorded
        ``record.artifact_path`` directory.

        Path-traversal guard: the resolved file MUST be a child of
        the resolved artifact_path; symlinks pointing outside are
        rejected. ``os.access(R_OK)`` first so unreadable files
        surface as 403 rather than 500.

        Text files (UTF-8 / ASCII / common log encodings) are served
        as ``text/plain`` so the browser renders inline. Binary or
        oversize files are served as ``application/octet-stream``
        with a download disposition. Max file size capped at
        :data:`_ARTIFACT_PREVIEW_MAX_BYTES` to avoid blowing up the
        browser on a multi-GB log; oversized text files get
        truncated with a banner.
        """
        record = await asyncio.to_thread(
            _get_raw_rollout_blocking, cfg, rollout_id,
        )
        if record is None or _owner_forbidden(
            _caller_owner_id(request), getattr(record, "owner_id", None),
        ) or not record.artifact_path:
            # Owner-gate the artifact bytes themselves — without this a scoped
            # caller blocked from the detail page could still read another
            # tenant's files by guessing the id + path.
            raise HTTPException(
                status_code=404,
                detail=f"raw rollout {rollout_id} has no artifact_path",
            )
        resolved = _resolve_artifact_file(record.artifact_path, file_path)
        if resolved is None:
            raise HTTPException(
                status_code=404,
                detail=f"artifact file {file_path!r} not found",
            )
        kind, body, size = await asyncio.to_thread(
            _read_artifact_file, resolved,
        )
        if kind == "denied":
            raise HTTPException(
                status_code=403,
                detail=f"artifact file {file_path!r} is not readable",
            )
        if kind == "text":
            return PlainTextResponse(body)
        # Binary or too-large-to-inline: stream with content-disposition.
        return FileResponse(
            str(resolved),
            media_type="application/octet-stream",
            filename=resolved.name,
            headers={
                "X-Artifact-Size-Bytes": str(size),
            },
        )

    # ── /sandboxes ───────────────────────────────────────────────────────────

    @app.get("/sandboxes", response_class=HTMLResponse)
    async def sandboxes(
        request: Request,
        node: str | None = Query(None, description="Filter by node id"),
        template: str | None = Query(None, description="Filter by template name"),
    ) -> HTMLResponse:
        rows = await _gather_sandboxes(
            cfg, node=node, template=template,
            owner_id=_caller_owner_id(request),
        )
        return templates.TemplateResponse(
            request, "sandboxes.html",
            {
                "rows": rows, "node": node, "template": template,
                "now": time.time(),
                "active_page": "sandboxes",
                **_refresh_context(request, cfg),
            },
        )

    # ── /capacity ────────────────────────────────────────────────────────────

    @app.get("/capacity", response_class=HTMLResponse)
    async def capacity(request: Request) -> HTMLResponse:
        snapshot = await _gather_capacity(cfg)
        return templates.TemplateResponse(
            request, "capacity.html",
            {
                **snapshot,
                "active_page": "capacity",
                **_refresh_context(request, cfg),
            },
        )

    @app.get("/fairshare", response_class=HTMLResponse)
    async def fairshare(request: Request) -> HTMLResponse:
        # Admin-only: the page lists every tenant's usage, so it crosses owner
        # boundaries. A scoped (per-user viewer) caller is refused; operators
        # and the loopback dev flow (both → _caller_owner_id None) see it.
        if _caller_owner_id(request) is not None:
            raise HTTPException(status_code=404, detail="not found")
        snapshot = await _gather_fairshare(cfg)
        return templates.TemplateResponse(
            request, "fairshare.html",
            {
                **snapshot,
                "active_page": "fairshare",
                **_refresh_context(request, cfg),
            },
        )

    # ── /builds (P1.6.d) ─────────────────────────────────────────────────────

    @app.get("/builds", response_class=HTMLResponse)
    async def builds(
        request: Request,
        page: int = Query(1, ge=1),
        page_size: int = Query(32, ge=1, le=128),
        status: str | None = Query(None),
    ) -> HTMLResponse:
        snapshot = await asyncio.to_thread(
            _gather_builds, cfg,
            page=page, page_size=page_size, status=status,
        )
        prev_url = str(
            request.url.include_query_params(
                page=max(1, page - 1), page_size=page_size,
            ),
        )
        next_url = str(
            request.url.include_query_params(
                page=page + 1, page_size=page_size,
            ),
        )
        return templates.TemplateResponse(
            request, "builds.html",
            {
                **snapshot,
                "prev_url": prev_url,
                "next_url": next_url,
                "active_page": "builds",
                **_refresh_context(request, cfg),
            },
        )

    @app.get("/builds/{plan_id}", response_class=HTMLResponse)
    async def build_detail(request: Request, plan_id: str) -> HTMLResponse:
        snapshot = await asyncio.to_thread(_gather_build_detail, cfg, plan_id)
        if snapshot is None:
            raise HTTPException(
                status_code=404, detail=f"build plan {plan_id!r} not found",
            )
        return templates.TemplateResponse(
            request, "build_detail.html",
            {
                **snapshot,
                "active_page": "builds",
                # Detail pages don't auto-refresh — inspect-an-artifact
                # view; consistent with /rollouts/<id>.
                "refresh_s": 0,
            },
        )

    # ── /api/build/* (P1.6.f cluster-RPC) ────────────────────────────────────
    #
    # Operator-facing JSON API for ``xrlenv build apply --connect-host``.
    # POST kicks off a build in the background asyncio task pool;
    # GET polls the persisted snapshot. Auth = operator-role bearer
    # token when ``cfg.token_store`` is wired.

    _build_tasks: dict[str, asyncio.Task[Any]] = {}

    def _require_operator(request: Request) -> None:
        _require_role(request, allowed_roles={"operator"})

    def _require_read_role(request: Request) -> None:
        # Read (GET) access: any authenticated identity. A per-user ``consumer``
        # token (the one a user already puts in their .env to submit jobs) now
        # also opens the admin, read-only and owner-scoped to their own jobs —
        # so a user needs just one token. ``viewer`` is the watch-only role for
        # people who don't submit; ``operator`` additionally gets writes + the
        # global (un-scoped) view. Owner scoping is applied downstream by
        # ``_caller_owner_id`` (operator → see-all, everyone else → own owner).
        _require_role(request, allowed_roles=_ADMIN_READ_ROLES)

    def _require_role(
        request: Request, *, allowed_roles: set[str],
    ) -> None:
        """B7.3 admin auth: gate ``request`` on one of ``allowed_roles``.

        Two transports accepted:

        - ``Authorization: Bearer <token>`` — the CLI / programmatic path.
          ``xrlenv build apply --connect-host ... --operator-token``
          posts with a bearer header; we verify against
          :class:`TokenStore` and require ``identity.role`` to be in
          ``allowed_roles``.
        - **Session cookie** (``xrlenv_admin_session``) — the browser path
          (B7.4). ``POST /login`` verifies a pasted token and drops it into
          the cookie; here we re-verify it exactly like the bearer path. We
          deliberately no longer honor ``Authorization: Basic``: the browser
          caches basic-auth creds per realm and replays them on every request
          with no app-controllable logout, so an operator who signed in as one
          token could never switch to another. The cookie is logout-able
          (``POST /logout`` clears it), and ignoring any stale cached basic
          header means logout is authoritative.

        Read (GET) routes accept ``consumer`` / ``viewer`` / ``operator``;
        write routes are ``operator``-only. Owner scoping is applied
        downstream by ``_caller_owner_id`` (operator → see-all, everyone
        else → their own owner).

        When ``cfg.token_store`` is absent or empty, the helper no-ops —
        loopback-only dev flow keeps working (the bind guard already
        refuses unauthenticated public binds).
        """
        store = cfg.token_store
        if store is not None:
            # Match the gRPC interceptor (auth_interceptor.py): re-stat the
            # secret files and rebuild the store so per-user tokens issued
            # AFTER the control plane started authenticate here too, without a
            # restart. The admin HTTP path makes no gRPC RPC of its own, so
            # without this it would only ever see the startup snapshot — a
            # freshly-issued ``consumer`` token would be rejected at the login
            # prompt until some unrelated RPC happened to reload the shared
            # store. Reload BEFORE the ``is_empty`` check so a store that was
            # empty at boot still picks up the first token issued at runtime.
            store.maybe_reload()
        if store is None or store.is_empty:
            # No-auth / loopback dev flow: no authenticated tenant, so the
            # caller is treated as admin (sees every owner). Multi-user
            # scoping engages only once tokens are issued.
            request.state.identity = None
            return
        header = request.headers.get("authorization") or ""
        lowered = header.lower()
        if lowered.startswith("bearer "):
            token = header[len("Bearer "):].strip()
            identity = store.verify(token)
            if identity is None:
                raise HTTPException(
                    status_code=401,
                    detail="invalid bearer token",
                    headers=_admin_auth_challenge_headers(),
                )
            if identity.role not in allowed_roles:
                raise HTTPException(
                    status_code=403,
                    detail=(
                        f"role {identity.role!r} cannot access this route "
                        f"(needs one of {sorted(allowed_roles)})"
                    ),
                )
            request.state.identity = identity
            return
        # Browser path: a signed-in session carries the token in the
        # ``xrlenv_admin_session`` HttpOnly cookie (set by ``POST /login``).
        # Re-verify it every request so a revoked token stops working at once.
        # We check the cookie even when an ``Authorization`` header is present
        # but isn't a recognized bearer — a browser that cached basic-auth
        # creds from the pre-B7.4 build keeps sending ``Authorization: Basic``;
        # ignoring it (and preferring the cookie) is what makes the cookie
        # logout authoritative for those stuck sessions.
        cookie_token = request.cookies.get(_SESSION_COOKIE)
        if cookie_token:
            identity = store.verify(cookie_token)
            if identity is None:
                raise HTTPException(
                    status_code=401,
                    detail="session expired or revoked; sign in again",
                    headers=_admin_auth_challenge_headers(),
                )
            if identity.role not in allowed_roles:
                raise HTTPException(
                    status_code=403,
                    detail=(
                        f"role {identity.role!r} cannot access this route "
                        f"(needs one of {sorted(allowed_roles)})"
                    ),
                )
            request.state.identity = identity
            return
        raise HTTPException(
            status_code=401,
            detail="sign in to access the admin panel",
            headers=_admin_auth_challenge_headers(),
        )

    @app.post("/api/build/apply")
    async def api_build_apply(request: Request) -> JSONResponse:
        _require_operator(request)
        if cfg.build_coordinator is None:
            return JSONResponse(
                status_code=503,
                content={"error": "build coordinator not wired"},
            )
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="invalid JSON body")  # noqa: B904
        from xrlenv.control.build_plan import BuildPlan

        plan_raw = body.get("plan")
        if not isinstance(plan_raw, dict):
            raise HTTPException(
                status_code=400,
                detail="body.plan must be a mapping",
            )
        try:
            plan = BuildPlan.model_validate(plan_raw)
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"plan failed validation: {exc}",
            ) from exc
        force = bool(body.get("force", False))
        dry_run = bool(body.get("dry_run", False))
        # Audit P1.6.g-M1 fix (2026-05-05): the cluster path was
        # silently dropping ``eager`` so cluster operators asking for
        # eager prebuild semantics got the opportunistic default.
        # Read it from the body now and thread it through both the
        # dry-run path (to surface InsufficientCapacity at plan time)
        # and the background apply task.
        eager = bool(body.get("eager", False))
        fill_missing = bool(body.get("fill_missing", False))
        skip_if_present = bool(body.get("skip_if_present", False))
        # ``xrlenv build push`` — build git/tarball entries AND push them to the
        # registry each ref encodes (build-once fleet-wide). The coordinator
        # rejects push + fill_missing.
        push = bool(body.get("push", False))
        applied_by = str(body.get("applied_by") or "operator-token")
        # Per-invocation coordinator fan-out override — the dynamic knob
        # that replaces the import-time XRLENV_BUILD_CONCURRENCY env +
        # control-plane restart. ``None`` → the process default.
        concurrency_raw = body.get("concurrency")
        concurrency: int | None = None
        if concurrency_raw is not None:
            try:
                concurrency = int(concurrency_raw)
            except (TypeError, ValueError):
                raise HTTPException(  # noqa: B904
                    status_code=400,
                    detail="body.concurrency must be a positive integer",
                )
            if concurrency < 1:
                raise HTTPException(
                    status_code=400,
                    detail="body.concurrency must be >= 1",
                )
        if fill_missing and (force or eager):
            raise HTTPException(
                status_code=400,
                detail=(
                    "fill_missing is mutually exclusive with force "
                    "and eager — pick one apply mode"
                ),
            )

        coordinator = cfg.build_coordinator

        if dry_run:
            try:
                outcome = await coordinator.apply(
                    plan, dry_run=True, eager=eager,
                    fill_missing=fill_missing,
                    applied_by=applied_by,
                    skip_if_present=skip_if_present,
                    concurrency=concurrency,
                    push=push,
                )
            except Exception as exc:
                raise HTTPException(
                    status_code=400, detail=str(exc),
                ) from exc
            placement = outcome.placement
            return JSONResponse({
                "plan_id": outcome.plan_id,
                "status": outcome.status,
                "deferred": outcome.deferred,
                "placement": (
                    [
                        {
                            "image_ref": a.image_ref,
                            "node_id": a.node_id,
                            "benchmark": a.benchmark,
                            "size_bytes": a.size_bytes,
                        }
                        for a in placement.assignments
                    ]
                    if placement is not None else []
                ),
            })

        # Non-dry-run: kick off the apply in a background asyncio task
        # so the HTTP request returns immediately (long builds would
        # otherwise time out at proxies / the operator can't see
        # progress). The task updates state.db as it runs; the CLI
        # polls GET /api/build/plans/<plan_id>.
        from xrlenv.control.build_plan import compute_plan_id

        plan_id = compute_plan_id(plan)

        # Idempotency layer 1+2 short-circuits before we spawn the task.
        existing = (
            await asyncio.to_thread(_get_persisted_plan, cfg, plan_id)
        )
        if existing is not None:
            if existing.status == "in_flight" and plan_id in _build_tasks:
                return JSONResponse(
                    status_code=202,
                    content={"plan_id": plan_id, "status": "in_flight"},
                )
            if (
                existing.status == "completed"
                and not force
                and not fill_missing
            ):
                # ``--fill-missing`` is the operator's "the cluster
                # drifted from the completed plan; reconcile what's
                # missing" verb. Short-circuiting here would prevent
                # exactly that: a completed plan whose images were
                # later evicted (or never on disk on a node added
                # since) could not be brought back into alignment.
                # Audit M2 (2026-05-12) — caught in the post-merge
                # admin-route review.
                return JSONResponse({
                    "plan_id": plan_id,
                    "status": "no_op_already_completed",
                })
            # Existing plan was non-terminal-to-us (partial_failure /
            # cancelled / superseded) — the coordinator will pick it
            # back up and re-dispatch. Flip the status row to
            # ``in_flight`` synchronously here, BEFORE returning 202,
            # so the CLI poller's first ``GET /api/build/plans/<id>``
            # doesn't read the stale terminal status (e.g.
            # ``partial_failure`` from a prior crashed run) and exit
            # early. The coordinator's apply task will flip again to
            # in_flight as part of its existing flow — a redundant
            # write, harmless. Also purge the stale assignment rows
            # so per_status counts start from zero.
            await asyncio.to_thread(
                _flip_existing_plan_to_in_flight, cfg, plan_id,
            )

        async def _run() -> None:
            try:
                await coordinator.apply(
                    plan, force=force, eager=eager,
                    fill_missing=fill_missing,
                    # Admin already gates concurrency via the
                    # ``_build_tasks`` short-circuit above; the
                    # coordinator's own in_flight check (which is
                    # intended for direct LocalRuntime callers) would
                    # otherwise reject our just-pre-flipped row. See
                    # the 2026-05-12 hang ("plan_id=de4716... status
                    # cancelled → re-apply → 184s of zero rows")
                    # diagnosis.
                    bypass_in_flight_check=True,
                    dry_run=False,
                    applied_by=applied_by,
                    skip_if_present=skip_if_present,
                    concurrency=concurrency,
                    push=push,
                )
            except Exception as exc:
                LOGGER.exception(
                    "build coordinator apply raised for plan_id=%s",
                    plan_id,
                )
                # Persist a partial_failure plan record so the
                # /api/build/plans/<plan_id> endpoint and the
                # /builds admin panel surface the run rather than
                # returning 404 forever (the apply raised before
                # record_build_plan, so without this the operator
                # has no in-cluster trace of what went wrong).
                try:
                    await asyncio.to_thread(
                        _persist_failed_plan,
                        cfg, plan_id, plan, applied_by, str(exc),
                    )
                except Exception:
                    LOGGER.exception(
                        "failed to persist error record for plan_id=%s",
                        plan_id,
                    )

        task = asyncio.create_task(_run(), name=f"build-apply-{plan_id[:12]}")
        _build_tasks[plan_id] = task

        def _drop_when_done(_t: asyncio.Task[Any]) -> None:
            _build_tasks.pop(plan_id, None)

        task.add_done_callback(_drop_when_done)
        return JSONResponse(
            status_code=202,
            content={"plan_id": plan_id, "status": "in_flight"},
        )

    @app.get("/api/build/plans/{plan_id}")
    async def api_build_plan_status(
        plan_id: str, request: Request,
    ) -> JSONResponse:
        _require_operator(request)
        snapshot = await asyncio.to_thread(_gather_build_plan, cfg, plan_id)
        if snapshot is None:
            raise HTTPException(
                status_code=404, detail=f"plan_id {plan_id!r} not found",
            )
        return JSONResponse(snapshot)

    @app.get("/api/scratch/active-digests")
    async def api_scratch_active_digests(request: Request) -> JSONResponse:
        """The scratch repos active runs reference — the scratch-registry GC's
        exemption set. Point ``deploy/registry/scratch_registry_gc.py --exempt-url`` at
        this so the GC never reclaims a build-on-demand image an in-flight
        rollout still uses (notes/scratch-registry-build-on-demand.md)."""
        _require_operator(request)

        def _gather() -> list[str]:
            from xrlenv.control.scratch_gc import active_scratch_repos
            if not cfg.state_db.exists():
                return []
            store = SqliteStateStore(cfg.state_db, read_only=True)
            try:
                pairs = [(sb.image, sb.status) for sb in store.list_sandboxes()]
            finally:
                store.close()
            return sorted(active_scratch_repos(pairs))

        return JSONResponse({"repos": await asyncio.to_thread(_gather)})

    @app.post("/api/build/calibrate")
    async def api_build_calibrate(request: Request) -> JSONResponse:
        """Sub-slice 3 (F5) — operator-driven cluster size probe.

        Walks every connected node's ``report_images()`` snapshot,
        cross-references each entry's ``image_ref`` against the
        operator-supplied plan, and returns a per-image_ref
        ``cluster_max_size_bytes`` map. Image_refs that no node has
        materialized yet land in ``unmeasured`` and keep their
        operator-supplied ``size_hint_bytes`` when the CLI writes the
        calibrated YAML.

        **Layer-sharing-aware sizing.** When the node-side
        ``report_images()`` surfaces ``shared_size_bytes`` per image
        (Docker daemon's ``GET /system/df``), calibrate writes the
        **unique** per-image footprint
        (``size_bytes - shared_size_bytes``) instead of the legacy
        ``size_bytes``. The unique number is the incremental disk a
        node pays to cache this image when its base layers are
        already present from a sibling image — the metric FFD
        actually needs to pack many images on the same node. Per-
        image ``size_bytes`` (used by the pre-2026-05 calibrate)
        double-counts shared layers and produces over-reservation
        in plans where many images share a common base
        (swebench-verified's Python base, terminal-bench-2's harbor
        runtime, etc.). When the backend doesn't surface
        ``shared_size_bytes`` (older daemons, the in-memory test
        backend), calibrate falls back to ``size_bytes`` — same
        behavior as before, just a less accurate hint.

        The aggregation takes the **max** unique size across nodes
        (safest for FFD: padding for layer-version drift across nodes
        that might have slightly different bases). The per-node
        ``report_images`` call is the same one the admin /images
        page already uses; no new spec-21 command needed.
        """
        _require_operator(request)
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(  # noqa: B904
                status_code=400, detail="invalid JSON body",
            )
        plan_raw = body.get("plan")
        if not isinstance(plan_raw, dict):
            raise HTTPException(
                status_code=400, detail="body.plan must be a mapping",
            )
        from xrlenv.control.build_plan import BuildPlan

        try:
            plan = BuildPlan.model_validate(plan_raw)
        except Exception as exc:
            raise HTTPException(  # noqa: B904
                status_code=400,
                detail=f"plan failed validation: {exc}",
            )
        wanted_refs = {e.image_ref for e in plan.entries}
        if not wanted_refs:
            return JSONResponse({
                "calibrated": {}, "unmeasured": [],
                "nodes_queried": 0, "nodes_unreachable": [],
            })
        # Map each plan ref's registry-agnostic form back to the plan
        # ref(s) it represents, so a registry-qualified node tag
        # (``<host:port>/<repo>:<tag>`` after a pull) still resolves to
        # the plan's bare ``image_ref``. One normalized form can credit
        # more than one plan ref (a plan could list both the bare and a
        # qualified spelling of the same image), so the value is a set.
        wanted_by_norm: dict[str, set[str]] = {}
        for ref in wanted_refs:
            wanted_by_norm.setdefault(
                _registry_agnostic_ref(ref), set(),
            ).add(ref)
        # Repo-path index (host + tag + digest stripped) for the digest-pull
        # fallback below. Only entries whose repo maps to EXACTLY ONE plan ref
        # are eligible — an unambiguous repo can be credited from a node image
        # that was pulled by digest (so it carries an ``@sha256`` / untagged ref
        # the tag-preserving ``wanted_by_norm`` match can't reach).
        by_repo: dict[str, set[str]] = {}
        for ref in wanted_refs:
            by_repo.setdefault(_repo_path(ref), set()).add(ref)
        unambiguous_by_repo = {
            repo: next(iter(refs)) for repo, refs in by_repo.items()
            if len(refs) == 1
        }

        # Connected nodes: state.db NodeRegistry mirror, status='connected'.
        def _connected_node_ids() -> list[str]:
            if not cfg.state_db.exists():
                return []
            store = SqliteStateStore(cfg.state_db, read_only=True)
            try:
                return [
                    n.node_id for n in store.list_nodes()
                    if n.status == "connected"
                ]
            finally:
                store.close()

        node_ids = await asyncio.to_thread(_connected_node_ids)
        if cfg.node_lookup is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "admin server has no node_lookup wired; "
                    "calibrate needs cluster reachability"
                ),
            )

        # Per-image-ref running max + per-node unreachable list.
        max_size: dict[str, int] = {}
        unreachable: list[dict[str, str]] = []
        nodes_queried = 0
        # Manifest-digest → max unique-size across nodes, populated for EVERY
        # reported image that carries a ``RepoDigests`` digest (not just matched
        # ones). Feeds the post-loop digest-match fallback below: a plan ref the
        # tag/repo-path matchers miss (its image is held digest-pinned / untagged
        # and its repo is shared by many tags) is attributed by resolving the
        # plan ref to its digest and looking it up here.
        digest_sizes: dict[str, int] = {}

        for node_id in node_ids:
            transport = cfg.node_lookup(node_id)
            if transport is None:
                unreachable.append({
                    "node_id": node_id,
                    "error": "no live transport (node disconnected?)",
                })
                continue
            report_fn = getattr(transport, "report_images", None)
            if report_fn is None:
                unreachable.append({
                    "node_id": node_id,
                    "error": "node transport missing report_images",
                })
                continue
            try:
                # Calibrate is the one caller that needs per-image
                # SharedSize (unique = size - shared). Everywhere else
                # (the /images view, fill-missing inventory) takes the
                # cheap default that skips the slow ``docker system df``.
                report = await report_fn(include_shared_size=True)
            except Exception as exc:
                unreachable.append({
                    "node_id": node_id,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                continue
            nodes_queried += 1
            for img in getattr(report, "images", ()):
                # Prefer ``unique = size - shared`` when the node surfaces
                # layer-sharing; fall back to ``size_bytes`` otherwise. The
                # unique number is what FFD actually needs to reserve for
                # incremental cache cost; the raw ``size_bytes`` over-counts
                # shared layers across sibling images. See ``api_build_calibrate``
                # docstring for the operator-facing rationale. Computed BEFORE
                # matching because both the plan-ref crediting AND the
                # digest-index accrual below need it.
                shared = getattr(img, "shared_size_bytes", None)
                if isinstance(shared, int) and shared >= 0:
                    measured = max(0, int(img.size_bytes) - shared)
                else:
                    measured = int(img.size_bytes)
                # Index this image's unique size by its manifest digest, for the
                # digest-match fallback after the node loop. Done for every image
                # with a digest — even ones this node can't attribute by tag —
                # because a plan ref's own resolved digest is what reunites them.
                img_dig = _manifest_digest(getattr(img, "digest", None))
                if img_dig is not None:
                    prior_d = digest_sizes.get(img_dig)
                    if prior_d is None or measured > prior_d:
                        digest_sizes[img_dig] = measured
                # Match registry-agnostically: the plan carries bare
                # refs but a node reports the registry-qualified tag it
                # pulled. Credit every plan ref the normalized tag maps
                # to, keying ``max_size`` by the *plan* ref so the CLI's
                # ``calibrated[image_ref]`` lookup resolves. (Before this,
                # a strict ``img.name not in wanted_refs`` left every
                # pulled-from-registry image ``unmeasured`` — calibrate
                # reported "0 measured" while the admin /images page
                # plainly listed the image. Sibling failure mode to the
                # 2026-05-12 unique==0 bug below.)
                plan_refs = wanted_by_norm.get(
                    _registry_agnostic_ref(img.name),
                )
                if not plan_refs and not _has_explicit_tag(img.name):
                    # Digest-pull fallback: the control plane digest-pins
                    # ``:tag`` → ``@sha256:...`` (invariant 4), so a node holds
                    # the image under an ``@sha256`` / untagged ref that the
                    # tag-preserving match above misses — even though the image
                    # is plainly present. Credit the plan ref by repo path when
                    # that repo is unambiguous (one plan ref). Was: every
                    # digest-pulled image reported ``unmeasured`` (e.g. 7
                    # measured / 195 unmeasured despite 90+ images cached).
                    #
                    # Guard (``not _has_explicit_tag``): fire ONLY for an
                    # untagged / ``@sha256`` node ref — a real digest pull. A
                    # node image carrying a *different* explicit tag of the same
                    # repo (e.g. a stale ``ns/img:20251031`` while the plan pins
                    # ``ns/img:20260403``) is NOT the plan's image; crediting it
                    # attributed the wrong — often 0-byte, fully-shared — size to
                    # the plan ref (the 2026-07-17 tb2.1 mixed-tag over-credit:
                    # 6 newer-tag entries wrongly "cluster-reported", 3 as 0).
                    # The digest-match fallback after this loop handles the
                    # AMBIGUOUS-repo case this can't (many tags of one repo).
                    one = unambiguous_by_repo.get(_repo_path(img.name))
                    if one is not None:
                        plan_refs = {one}
                if not plan_refs:
                    continue
                # ``is None`` (not ``> 0``) is load-bearing: thin
                # task images on a fat shared base can legitimately
                # have ``unique == 0`` (every layer they reference
                # is already present on the node from a sibling).
                # A strict ``measured > prior`` check with ``prior``
                # defaulting to 0 would silently drop these refs
                # from the calibrated set, even though the node
                # clearly reported them — the failure mode the
                # 2026-05-12 calibrate-shows-9-unmeasured-despite-
                # admin-page-listing-them bug surfaced.
                for plan_ref in plan_refs:
                    prior = max_size.get(plan_ref)
                    if prior is None or measured > prior:
                        max_size[plan_ref] = measured

        # ── Digest-match fallback ───────────────────────────────────────────
        # Refs the tag/repo-path matchers left unmeasured may still be present
        # on a node under a digest-pinned (untagged) ref. The tag match can't
        # reach them (untagged) and the repo-path fallback is disabled when many
        # tags share ONE repository (SWE-bench: 113 tags of
        # ``d3j8x8q7/swe-bench-202605`` → ``unambiguous_by_repo`` empty). Digest
        # is the canonical image identity, so resolve each still-unmeasured plan
        # ref to its manifest digest and look it up in the digests the nodes
        # reported. Best-effort: a ref whose digest can't be resolved (registry
        # unreachable / resolver disabled) or that no node holds stays
        # unmeasured — exactly the prior behavior, never a regression.
        still_unmeasured = wanted_refs - set(max_size.keys())
        if still_unmeasured and digest_sizes:
            from xrlenv.control.registry_resolver import resolver_from_env

            resolver = resolver_from_env()
            if resolver is not None:
                sem = asyncio.Semaphore(8)

                async def _resolve_digest(ref: str) -> tuple[str, str | None]:
                    async with sem:
                        try:
                            return ref, _manifest_digest(
                                await resolver.resolve(ref),
                            )
                        except Exception:
                            return ref, None  # unresolvable → stays unmeasured

                resolved = await asyncio.gather(
                    *(_resolve_digest(r) for r in sorted(still_unmeasured)),
                )
                refs_by_digest: dict[str, set[str]] = {}
                for ref, dig in resolved:
                    if dig is not None and dig in digest_sizes:
                        refs_by_digest.setdefault(dig, set()).add(ref)
                # Same-digest dedup: refs sharing a manifest digest are the SAME
                # physical image. Mark ALL measured (cluster-reported), but count
                # the unique footprint ONCE — a byte-identical duplicate adds
                # zero incremental disk (consistent with the unique-size metric,
                # which already scores a fully-shared image as 0). The measured
                # size lands on the lexicographically-first ref; the rest get 0,
                # so downstream FFD never double-counts one image's storage.
                for dig, refs in refs_by_digest.items():
                    ordered = sorted(refs)
                    max_size[ordered[0]] = digest_sizes[dig]
                    for dup in ordered[1:]:
                        max_size[dup] = 0

        unmeasured = sorted(wanted_refs - set(max_size.keys()))
        return JSONResponse({
            "calibrated": max_size,
            "unmeasured": unmeasured,
            "nodes_queried": nodes_queried,
            "nodes_unreachable": unreachable,
        })

    @app.post("/api/image/evict")
    async def api_image_evict(request: Request) -> JSONResponse:
        """Operator-driven cluster-wide node-cache eviction.

        Fans one :class:`EvictImageCommand` out to every connected node;
        each node matches ``image_ref`` **registry-agnostically** against
        the tags it actually holds (so a bare ref matches the
        registry-qualified tag a node pulled) and removes the matching
        image(s), so the next acquire re-pulls fresh from the registry.

        The escape hatch for the mutable-tag staleness problem: after a
        rebuild + re-push under the *same* tag, a node never re-pulls on
        its own (``ensure_present`` short-circuits on local presence).
        In-use / pinned images are skipped unless ``force`` so a live
        rollout is never disrupted.

        Status is 200 even on partial errors — the operator-visible
        state is the per-node JSON body, not the HTTP code.
        """
        _require_operator(request)
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(  # noqa: B904
                status_code=400, detail="invalid JSON body",
            )
        image_ref = body.get("image_ref")
        if not isinstance(image_ref, str) or not image_ref:
            raise HTTPException(
                status_code=400,
                detail="body.image_ref must be a non-empty string",
            )
        force = bool(body.get("force", False))

        def _connected_node_ids() -> list[str]:
            if not cfg.state_db.exists():
                return []
            store = SqliteStateStore(cfg.state_db, read_only=True)
            try:
                return [
                    n.node_id for n in store.list_nodes()
                    if n.status == "connected"
                ]
            finally:
                store.close()

        node_ids = await asyncio.to_thread(_connected_node_ids)
        if cfg.node_lookup is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "admin server has no node_lookup wired; "
                    "evict needs cluster reachability"
                ),
            )

        results: list[dict[str, Any]] = []
        nodes_evicted = 0
        total_reclaimed = 0
        for node_id in node_ids:
            transport = cfg.node_lookup(node_id)
            evict_fn = getattr(transport, "evict_image", None) if transport else None
            if evict_fn is None:
                results.append({
                    "node_id": node_id,
                    "status": "unreachable",
                    "error": (
                        "no live transport (node disconnected?)"
                        if transport is None
                        else "node transport missing evict_image"
                    ),
                })
                continue
            try:
                outcome = await evict_fn(image_ref=image_ref, force=force)
            except Exception as exc:
                results.append({
                    "node_id": node_id,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                })
                continue
            entry: dict[str, Any] = {
                "node_id": node_id,
                "status": outcome.status,
                "reclaimed_bytes": int(outcome.reclaimed_bytes),
                "removed": list(outcome.removed),
            }
            if outcome.detail:
                entry["error"] = outcome.detail
            results.append(entry)
            if outcome.status == "evicted":
                nodes_evicted += 1
                total_reclaimed += int(outcome.reclaimed_bytes)

        return JSONResponse({
            "image_ref": image_ref,
            "force": force,
            "nodes_queried": len(node_ids),
            "nodes_evicted": nodes_evicted,
            "total_reclaimed_bytes": total_reclaimed,
            "results": results,
        })

    @app.post("/api/build/cancel")
    async def api_build_cancel(request: Request) -> JSONResponse:
        """Operator-driven cluster-side cancel of an in-flight plan.

        Resolves ``plan_id`` (full id or unique prefix) and:

        1. Marks the plan ``cancelled`` so the next admission /
           idempotency check short-circuits.
        2. Marks every ``pending`` assignment ``cancelled`` (they
           were waiting in the coordinator's queue and never
           reached the wire).
        3. Sends one ``CancelBuildImageCommand`` to each node that
           has a ``building`` assignment; on success marks the
           assignment ``cancelled``, on failure records the
           per-(node, image) error.
        4. Cancels the in-process apply task so the coordinator's
           ``apply()`` loop unwinds. Without this the running
           ``await build_image_fn(...)`` would still be parked
           against the BuildImageCommand reply that the cancel
           caused the node to ship — the reply does come through
           correctly, but the apply's per-image-ref loop only
           transitions the next step once it returns. We cancel
           the task to short-circuit any subsequent in-flight work
           (eager mode would otherwise launch the next batch).

        Returns a summary dict with cancelled_count + per-node
        errors (if any). Status is 200 even on partial errors —
        the operator-visible state is the JSON body, not the HTTP
        code.
        """
        _require_operator(request)
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(  # noqa: B904
                status_code=400, detail="invalid JSON body",
            )
        plan_id_or_prefix = body.get("plan_id")
        if not isinstance(plan_id_or_prefix, str) or not plan_id_or_prefix:
            raise HTTPException(
                status_code=400,
                detail="body.plan_id must be a non-empty string",
            )

        plan = await asyncio.to_thread(
            _resolve_plan_for_admin, cfg, plan_id_or_prefix,
        )
        if plan is None:
            raise HTTPException(
                status_code=404,
                detail=f"plan_id {plan_id_or_prefix!r} not found",
            )

        from xrlenv.control.state import SqliteStateStore

        def _reconcile_nonterminal(plan_id: str) -> int:
            """Mark any ``building`` / ``pending`` / ``registered`` row
            ``cancelled``. Catches rows the apply task flipped to
            ``building`` during the cancel race (after our snapshot, before
            the task was stopped) and ``registered`` overflow rows the main
            flow doesn't otherwise touch — without this a cancelled plan can
            show "N building" forever. Terminal rows (done/failed/cancelled/
            superseded) are left as-is."""
            store = SqliteStateStore(cfg.state_db)
            try:
                n = 0
                for a in store.list_assignments(plan_id):
                    if a.status in ("building", "pending", "registered"):
                        store.update_assignment_status(
                            plan_id=plan_id, node_id=a.node_id,
                            image_ref=a.image_ref, status="cancelled",
                            error="cancelled by operator (reconciled)",
                        )
                        n += 1
                return n
            finally:
                store.close()

        if plan.status in ("completed", "superseded"):
            return JSONResponse({
                "plan_id": plan.plan_id,
                "status": plan.status,
                "cancelled_count": 0,
                "errors": [],
                "note": (
                    f"plan already terminal ({plan.status}); "
                    "no cluster-side action taken"
                ),
            })
        if plan.status == "cancelled":
            # Already cancelled — but an earlier cancel could have left
            # orphaned non-terminal rows from the dispatch race. Re-running
            # cancel reconciles them rather than no-op'ing, so a plan stuck
            # showing "N building" under a cancelled status can be cleaned.
            reconciled = await asyncio.to_thread(
                _reconcile_nonterminal, plan.plan_id,
            )
            return JSONResponse({
                "plan_id": plan.plan_id,
                "status": "cancelled",
                "cancelled_count": reconciled,
                "errors": [],
                "note": (
                    f"plan already cancelled; reconciled {reconciled} "
                    "orphaned non-terminal row(s)"
                ),
            })

        # 1. Stop the in-process apply task FIRST and wait for it to
        # unwind, BEFORE snapshotting. This ordering is what closes the
        # cancel race: a dispatch parked on the build-concurrency
        # semaphore could otherwise flip a fresh row to ``building`` and
        # send work to a node *after* our snapshot but *before* we stop
        # the task — that node-side build/pull would then never receive a
        # CancelBuildImageCommand. Stopping first makes the ``building``
        # set stable; any row the task left ``building`` as it unwound is
        # captured by the snapshot below and gets a real cancel RPC. The
        # nodes' in-flight BuildImage replies still land on the
        # coordinator's already-awaited command_id futures.
        task = _build_tasks.pop(plan.plan_id, None)
        if task is not None and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError, TimeoutError):
                await asyncio.wait_for(task, timeout=10.0)

        # 2. Mark the plan cancelled + mark every pending assignment
        # cancelled (those never reached the wire), and snapshot the now-
        # stable set of ``building`` rows for CancelBuildImage fan-out.
        # The apply task is already stopped, so nothing can race a new row
        # into ``building`` behind this snapshot.

        def _do_state_updates() -> tuple[int, list[dict[str, Any]]]:
            store = SqliteStateStore(cfg.state_db)
            try:
                store.update_build_plan_status(plan.plan_id, "cancelled")
                assignments = store.list_assignments(plan.plan_id)
                count = 0
                # Mark pending immediately (they never reached the wire).
                for a in assignments:
                    if a.status == "pending":
                        store.update_assignment_status(
                            plan_id=plan.plan_id, node_id=a.node_id,
                            image_ref=a.image_ref, status="cancelled",
                            error="cancelled by operator before dispatch",
                        )
                        count += 1
                return count, [
                    {"node_id": a.node_id, "image_ref": a.image_ref}
                    for a in assignments if a.status == "building"
                ]
            finally:
                store.close()

        cancelled_count, in_flight = await asyncio.to_thread(_do_state_updates)
        errors: list[dict[str, Any]] = []

        # Helper: persist a per-(node, image) row in either the
        # success or failure shape. Bound default args bind loop
        # vars at definition time so the closure references the
        # iteration's values rather than the last loop value (B023).
        def _persist_row_status(
            *, node_id: str, image_ref: str,
            status: BuildAssignmentStatus, error: str,
        ) -> None:
            store = SqliteStateStore(cfg.state_db)
            try:
                store.update_assignment_status(
                    plan_id=plan.plan_id, node_id=node_id,
                    image_ref=image_ref, status=status, error=error,
                )
            finally:
                store.close()

        # 3. Dispatch CancelBuildImageCommand for each in-flight pair
        # IN PARALLEL. The old sequential loop sent one cancel at a
        # time, each without a control-plane timeout — with hundreds
        # of in-flight images the HTTP handler exceeded the CLI's
        # 60 s timeout before finishing. ``asyncio.gather`` with a
        # per-call ``wait_for`` ceiling fixes both latency and the
        # unbounded-wait hazard.
        _CANCEL_PER_IMAGE_TIMEOUT_S = 30.0

        async def _cancel_one(
            entry: dict[str, Any],
        ) -> tuple[bool, dict[str, Any] | None]:
            """Returns ``(success, error_dict_or_None)``."""
            node_id = entry["node_id"]
            image_ref = entry["image_ref"]
            transport = (
                cfg.node_lookup(node_id)
                if cfg.node_lookup is not None else None
            )
            if transport is None:
                err = (
                    "no live transport for node (node disconnected?); "
                    "the build will fail on its own when the node "
                    "reconnects or times out"
                )
                await asyncio.to_thread(
                    _persist_row_status, node_id=node_id,
                    image_ref=image_ref, status="failed",
                    error=f"cancel could not be issued: {err}",
                )
                return False, {
                    "node_id": node_id, "image_ref": image_ref,
                    "error": err,
                }
            cancel_fn = getattr(transport, "cancel_build_image", None)
            if cancel_fn is None:
                err = (
                    "node transport missing cancel_build_image; "
                    "this control plane is newer than the node — "
                    "upgrade xrlenv on the node and retry"
                )
                await asyncio.to_thread(
                    _persist_row_status, node_id=node_id,
                    image_ref=image_ref, status="failed",
                    error=f"cancel could not be issued: {err}",
                )
                return False, {
                    "node_id": node_id, "image_ref": image_ref,
                    "error": err,
                }
            try:
                # Give the transport its OWN timeout so a no-reply node
                # takes ``_send_and_wait``'s clean timeout branch (pops the
                # pending entry + flags the node's command-timeout health
                # marker). The outer ``wait_for`` is a slightly larger
                # backstop for a hang that isn't a missing reply, so the
                # transport timeout normally fires first.
                status, error = await asyncio.wait_for(
                    cancel_fn(
                        image_ref=image_ref,
                        timeout_s=_CANCEL_PER_IMAGE_TIMEOUT_S,
                    ),
                    timeout=_CANCEL_PER_IMAGE_TIMEOUT_S + 5.0,
                )
            except TimeoutError:
                err = (
                    f"cancel_build_image timed out after "
                    f"{_CANCEL_PER_IMAGE_TIMEOUT_S:.0f}s"
                )
                await asyncio.to_thread(
                    _persist_row_status, node_id=node_id,
                    image_ref=image_ref, status="failed",
                    error=f"cancel dispatch timed out: {err}",
                )
                return False, {
                    "node_id": node_id, "image_ref": image_ref,
                    "error": err,
                }
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
                await asyncio.to_thread(
                    _persist_row_status, node_id=node_id,
                    image_ref=image_ref, status="failed",
                    error=f"cancel dispatch raised: {err}",
                )
                return False, {
                    "node_id": node_id, "image_ref": image_ref,
                    "error": err,
                }
            if status != "ok":
                err = error or status
                await asyncio.to_thread(
                    _persist_row_status, node_id=node_id,
                    image_ref=image_ref, status="failed",
                    error=f"cancel reply was {status!r}: {err}",
                )
                return False, {
                    "node_id": node_id, "image_ref": image_ref,
                    "error": err,
                }
            await asyncio.to_thread(
                _persist_row_status, node_id=node_id,
                image_ref=image_ref, status="cancelled",
                error="cancelled by operator",
            )
            return True, None

        results = await asyncio.gather(
            *(_cancel_one(entry) for entry in in_flight),
            return_exceptions=True,
        )
        for r in results:
            if isinstance(r, BaseException):
                LOGGER.exception("cancel_one raised unexpectedly: %s", r)
                continue
            ok, err_dict = r
            if ok:
                cancelled_count += 1
            elif err_dict is not None:
                errors.append(err_dict)

        # 4. Reconcile sweep: mark any leftover non-terminal row
        # (``registered`` overflow the fan-out doesn't touch, or a
        # ``building`` row whose cancel RPC failed) cancelled, so the plan
        # never shows "N building" under a cancelled status. With the apply
        # task already stopped (step 1) this is a safety net, not the
        # primary mechanism.
        reconciled = await asyncio.to_thread(
            _reconcile_nonterminal, plan.plan_id,
        )
        cancelled_count += reconciled

        return JSONResponse({
            "plan_id": plan.plan_id,
            "status": "cancelled",
            "cancelled_count": cancelled_count,
            "errors": errors,
        })

    # ── /images ──────────────────────────────────────────────────────────────

    async def _render_images(
        request: Request,
        *,
        view: str,
        q: str | None,
        tier: str | None,
        pressure: str | None,
        pinned: str | None,
        sort: str | None,
        page: int,
        page_size: int | None,
        include: str | None = None,
        show_intermediate: int = 0,
        show_external: int = 0,
    ) -> HTMLResponse:
        effective_page_size = _resolve_image_page_size(page_size)
        # Catalog (image-coverage) view supports the same xrlenv-only /
        # +intermediate / +foreign include filter as the per-node detail
        # page. Cache (node-storage) view shows raw on-disk truth — no
        # ownership filter there since operators want the actual disk
        # pressure including intermediates.
        if view == "images":
            include_filter, hide_intermediate, xrlenv_only = (
                _resolve_image_include(
                    include,
                    show_intermediate=show_intermediate,
                    show_external=show_external,
                )
            )
        else:
            include_filter, hide_intermediate, xrlenv_only = (
                "all", False, False,
            )
        snapshot = await _gather_images(
            cfg,
            request=request,
            view=view,
            q=q,
            tier=tier,
            pressure=pressure,
            pinned=pinned,
            sort=sort,
            page=page,
            page_size=effective_page_size,
            include_filter=include_filter,
            hide_intermediate=hide_intermediate,
            xrlenv_only=xrlenv_only,
            snapshot=image_snapshot,
        )
        active_page = "images_cache" if view == "nodes" else "images_catalog"
        # Both /images/cache and /images/catalog default to a 60 s
        # auto-refresh: per-node disk + cluster-wide image coverage
        # change at human-operator timescales (build apply, eviction
        # pressure), so the 5-second cadence the rollouts page uses
        # would burn page renders without surfacing meaningful change.
        return templates.TemplateResponse(
            request, "images.html",
            {
                **snapshot,
                "active_page": active_page,
                **_refresh_context(request, cfg),
            },
        )

    @app.get("/images", response_class=HTMLResponse)
    async def images(
        request: Request,
        view: str = Query("nodes", description="Cluster table: nodes or images"),
        q: str | None = Query(None, description="Substring filter"),
        tier: str | None = Query(None, description="Filter by image tier"),
        pressure: str | None = Query(None, description="Filter node pressure"),
        pinned: str | None = Query(None, description="Filter pinned state"),
        sort: str | None = Query(None, description="Sort key"),
        page: int = Query(1, ge=1, description="One-indexed page"),
        page_size: int | None = Query(
            None, ge=1, le=_MAX_IMAGE_PAGE_SIZE,
            description="Rows per page",
        ),
    ) -> HTMLResponse:
        if "view" in request.query_params and view in {"nodes", "images"}:
            warnings.warn(
                "?view=nodes|images on /images is deprecated; use "
                "/images/cache or /images/catalog before the next phase boundary.",
                DeprecationWarning,
                stacklevel=2,
            )
        return await _render_images(
            request,
            view=view,
            q=q,
            tier=tier,
            pressure=pressure,
            pinned=pinned,
            sort=sort,
            page=page,
            page_size=page_size,
        )

    @app.get("/images/cache", response_class=HTMLResponse)
    async def images_cache(
        request: Request,
        q: str | None = Query(None, description="Substring filter"),
        tier: str | None = Query(None, description="Filter by image tier"),
        pressure: str | None = Query(None, description="Filter node pressure"),
        pinned: str | None = Query(None, description="Filter pinned state"),
        sort: str | None = Query(None, description="Sort key"),
        page: int = Query(1, ge=1, description="One-indexed page"),
        page_size: int | None = Query(
            None, ge=1, le=_MAX_IMAGE_PAGE_SIZE,
            description="Rows per page",
        ),
    ) -> HTMLResponse:
        return await _render_images(
            request,
            view="nodes",
            q=q,
            tier=tier,
            pressure=pressure,
            pinned=pinned,
            sort=sort,
            page=page,
            page_size=page_size,
        )

    @app.get("/images/catalog", response_class=HTMLResponse)
    async def images_catalog(
        request: Request,
        q: str | None = Query(None, description="Substring filter"),
        tier: str | None = Query(None, description="Filter by image tier"),
        pinned: str | None = Query(None, description="Filter pinned state"),
        sort: str | None = Query(None, description="Sort key"),
        page: int = Query(1, ge=1, description="One-indexed page"),
        page_size: int | None = Query(
            None, ge=1, le=_MAX_IMAGE_PAGE_SIZE,
            description="Rows per page",
        ),
        include: str | None = Query(
            None,
            description=(
                "Image classes to include in the catalog: default "
                "(xrlenv only), intermediates, foreign, all"
            ),
        ),
    ) -> HTMLResponse:
        return await _render_images(
            request,
            view="images",
            q=q,
            tier=tier,
            pressure=None,
            pinned=pinned,
            sort=sort,
            page=page,
            page_size=page_size,
            include=include,
        )

    @app.get("/images/nodes/{node_id}", response_class=HTMLResponse)
    async def image_node_detail(
        request: Request,
        node_id: str,
        q: str | None = Query(None, description="Substring filter"),
        tier: str | None = Query(None, description="Filter by image tier"),
        pinned: str | None = Query(None, description="Filter pinned state"),
        sort: str | None = Query(None, description="Sort key"),
        page: int = Query(1, ge=1, description="One-indexed page"),
        page_size: int | None = Query(
            None, ge=1, le=_MAX_IMAGE_PAGE_SIZE,
            description="Rows per page",
        ),
        include: str | None = Query(
            None,
            description="Extra image classes to include in the node-detail view",
        ),
        # Legacy checkbox params accepted for existing URLs. The visible
        # form now submits the four-state ``include`` dropdown instead.
        show_intermediate: int = Query(
            0, ge=0, le=1,
            description="Show xrlenv build intermediates (default off)",
        ),
        show_external: int = Query(
            0, ge=0, le=1,
            description="Show images xrlenv didn't build (default off)",
        ),
    ) -> HTMLResponse:
        effective_page_size = _resolve_image_page_size(page_size)
        include_filter, hide_intermediate, xrlenv_only = _resolve_image_include(
            include,
            show_intermediate=show_intermediate,
            show_external=show_external,
        )
        snapshot = await _gather_image_node_detail(
            cfg,
            request=request,
            node_id=node_id,
            q=q,
            tier=tier,
            pinned=pinned,
            sort=sort,
            page=page,
            page_size=effective_page_size,
            include_filter=include_filter,
            hide_intermediate=hide_intermediate,
            xrlenv_only=xrlenv_only,
        )
        return templates.TemplateResponse(
            request, "image_node_detail.html",
            {**snapshot, "active_page": "images_cache", "refresh_s": 0},
        )

    @app.get("/images/image", response_class=HTMLResponse)
    async def image_detail(
        request: Request,
        ref: str = Query(..., description="Image reference"),
    ) -> HTMLResponse:
        snapshot = await _gather_image_detail(
            cfg, request=request, image_ref=ref, snapshot=image_snapshot,
        )
        return templates.TemplateResponse(
            request, "image_detail.html",
            {**snapshot, "active_page": "images_catalog", "refresh_s": 0},
        )

    # ── /health ──────────────────────────────────────────────────────────────

    @app.get("/health", response_class=HTMLResponse)
    async def health(request: Request) -> HTMLResponse:
        snapshot = await _gather_health(cfg)
        return templates.TemplateResponse(
            request, "health.html",
            {
                **snapshot,
                "active_page": "health",
                **_refresh_context(request, cfg),
            },
        )

    # ── /rollouts/{id} ───────────────────────────────────────────────────────

    @app.get("/rollouts/{rollout_id}", response_class=HTMLResponse)
    async def rollout_detail(
        request: Request,
        rollout_id: str,
        find: str | None = Query(None, description="Highlight steps containing this substring"),
    ) -> HTMLResponse:
        snapshot = await _gather_rollout_detail(
            cfg, rollout_id, cache=cache, fetch_fn=_fetch_trajectory,
        )
        if snapshot is None or _owner_forbidden(
            _caller_owner_id(request),
            await asyncio.to_thread(_rollout_owner_blocking, cfg, rollout_id),
        ):
            raise HTTPException(status_code=404, detail=f"rollout {rollout_id} not found")
        # In-page search (spec 17 phase-0): tag steps that contain ``find`` so
        # the template can highlight + count matches. Cheap server-side scan
        # over the already-cached trajectory body — no extra fetch.
        match_indices: list[int] = []
        if find and snapshot["trajectory"] is not None:
            needle = find.lower()
            for step in snapshot["trajectory"].steps:
                blob = json.dumps(
                    {"action": step.action, "obs": step.obs, "info": step.info},
                    default=str,
                ).lower()
                if needle in blob:
                    match_indices.append(step.index)
        # Rollout-detail is always auto-refresh=OFF. It's an inspect-
        # an-artifact view; refresh would reset scroll/search/step
        # state. Operators that want a live view of an in-flight
        # rollout watch the rollouts list (which still auto-refreshes
        # at the configured interval) and click into the detail page
        # when something looks worth investigating.
        return templates.TemplateResponse(
            request, "rollout_detail.html",
            {
                **snapshot,
                "find": find,
                "match_indices": match_indices,
                "active_page": "rollouts",
                "refresh_s": 0,
            },
        )

    # ── /rollouts/{id}/download ──────────────────────────────────────────────

    @app.get("/rollouts/{rollout_id}/download")
    async def rollout_download(
        request: Request, rollout_id: str,
    ) -> PlainTextResponse:
        """Spec-17 phase-0 'Download as jsonl' — streams the normalized
        trajectory body. The native sink format download (raw) needs the
        sink-aware reader (phase 1); for platform-jsonl the two are the
        same so we serve the on-disk jsonl bytes directly when present.
        """
        if _owner_forbidden(
            _caller_owner_id(request),
            await asyncio.to_thread(_rollout_owner_blocking, cfg, rollout_id),
        ):
            raise HTTPException(status_code=404, detail=f"rollout {rollout_id} not found")
        body = await asyncio.to_thread(_download_blocking, cfg, rollout_id)
        if body is None:
            raise HTTPException(status_code=404, detail=f"rollout {rollout_id} not found")
        return PlainTextResponse(
            body,
            media_type="application/x-ndjson",
            headers={
                "content-disposition": (
                    f'attachment; filename="{rollout_id}.jsonl"'
                ),
            },
        )

    # ── /rollouts/{id}/verifier/{path} ───────────────────────────────────────

    @app.get("/rollouts/{rollout_id}/verifier/{file_path:path}")
    async def rollout_verifier_file(
        request: Request, rollout_id: str, file_path: str,
    ) -> FileResponse:
        """Stream a single file from ``<run_dir>/verifier/``.

        The rollout-detail page inlines small text files; this endpoint
        backs the "Download" link for binaries and large text files.
        Path traversal is guarded by resolving + checking that the
        resolved path stays under the verifier root.
        """
        if _owner_forbidden(
            _caller_owner_id(request),
            await asyncio.to_thread(_rollout_owner_blocking, cfg, rollout_id),
        ):
            raise HTTPException(
                status_code=404, detail=f"rollout {rollout_id} not found"
            )
        run_dir = await asyncio.to_thread(
            _run_dir_for_rollout, cfg, rollout_id,
        )
        if run_dir is None:
            raise HTTPException(
                status_code=404, detail=f"rollout {rollout_id} not found"
            )
        verifier_root = (run_dir / "verifier").resolve()
        if not verifier_root.is_dir():
            raise HTTPException(
                status_code=404,
                detail=f"rollout {rollout_id} has no verifier output",
            )
        target = (verifier_root / file_path).resolve()
        # Path-traversal guard: the resolved target must be under
        # ``verifier_root``. ``Path.is_relative_to`` is the modern
        # idiom (3.9+); we use ``str.startswith`` on the resolved
        # paths for explicit clarity.
        if (
            str(target) != str(verifier_root)
            and not str(target).startswith(str(verifier_root) + "/")
        ):
            raise HTTPException(status_code=400, detail="path traversal denied")
        if not target.is_file():
            raise HTTPException(
                status_code=404, detail=f"verifier file {file_path} not found"
            )
        # ``application/octet-stream`` is safest; the browser offers
        # download instead of trying to render unknown extensions.
        return FileResponse(
            path=target,
            media_type="application/octet-stream",
            filename=Path(file_path).name,
        )

    # ── /rollouts/{id}/coordinator.log ───────────────────────────────────────

    @app.get("/rollouts/{rollout_id}/coordinator.log")
    async def rollout_coordinator_log(
        request: Request, rollout_id: str,
    ) -> FileResponse:
        """Stream the rollout's full ``coordinator.log`` (JSON-lines).

        The rollout-detail page inlines a tail of this file; this route
        backs a "Download full log" link for the cases where the tail
        clipped early lines (image-pull, scheduler decisions) the user
        wants to see. Plain text so the browser displays it inline by
        default.
        """
        if _owner_forbidden(
            _caller_owner_id(request),
            await asyncio.to_thread(_rollout_owner_blocking, cfg, rollout_id),
        ):
            raise HTTPException(
                status_code=404, detail=f"rollout {rollout_id} not found"
            )
        run_dir = await asyncio.to_thread(
            _run_dir_for_rollout, cfg, rollout_id,
        )
        if run_dir is None:
            raise HTTPException(
                status_code=404, detail=f"rollout {rollout_id} not found"
            )
        log_path = run_dir / "coordinator.log"
        if not log_path.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"rollout {rollout_id} has no coordinator.log",
            )
        return FileResponse(
            path=log_path,
            media_type="text/plain; charset=utf-8",
            filename=f"{rollout_id}.coordinator.log",
        )

    # ── /api/* — JSON convenience for tools / future SSE clients ─────────────

    @app.get("/api/overview")
    async def api_overview() -> JSONResponse:
        return JSONResponse(await _gather_overview(cfg, started_at))

    @app.get("/api/rollouts")
    async def api_rollouts(
        request: Request,
        status: str | None = None,
        template: str | None = None,
        since: str | None = None,
        page: int = Query(1, ge=1),
        page_size: int | None = Query(None, ge=1, le=_MAX_ROLLOUT_PAGE_SIZE),
    ) -> JSONResponse:
        effective_page_size = _resolve_rollout_page_size(
            page_size, cfg.rollout_page_size,
        )
        rollout_page = await _gather_rollouts(
            cfg,
            status=status,
            template=template,
            since=since,
            page=page,
            page_size=effective_page_size,
            owner_id=_caller_owner_id(request),
        )
        records = rollout_page.records[:rollout_page.page_size]
        has_next = rollout_page.has_next or len(rollout_page.records) > rollout_page.page_size
        return JSONResponse(
            {
                "records": [
                    _rollout_dict(r)
                    for r in records
                ],
                "page": rollout_page.page,
                "page_size": rollout_page.page_size,
                "has_next": has_next,
            }
        )

    return app


# ──────────────────────────────────────────────────────────────────────────────
# View-data gathering — runs off the event loop via asyncio.to_thread
# ──────────────────────────────────────────────────────────────────────────────


async def _gather_overview(cfg: AdminServerConfig, started_at: float) -> dict[str, Any]:
    return await asyncio.to_thread(_overview_blocking, cfg, started_at)


def _cluster_info(cfg: AdminServerConfig) -> dict[str, Any]:
    """The optional cluster-info banner (control-plane endpoint + registries).
    Each value is shown only when set; ``any_set`` lets the template skip the
    whole section when nothing is configured."""
    info: dict[str, Any] = {
        "control_plane_endpoint": cfg.control_plane_endpoint,
        "registry_mirror": cfg.registry_mirror,
        "private_registry": cfg.private_registry,
    }
    info["any_set"] = any(info.values())
    return info


def _overview_blocking(cfg: AdminServerConfig, started_at: float) -> dict[str, Any]:
    if not cfg.state_db.exists():
        return {
            "node_count": 0, "node_active": 0,
            "node_connected": 0, "node_rostered": 0, "nodes_lost": 0,
            "container_count": 0, "rollout_running": 0,
            "rollout_finished_1h": 0, "rollout_failed_1h": 0,
            "rollout_capacity_rejected_1h": 0,
            "uptime_s": time.time() - started_at,
            "state_db_present": False,
            "cluster_info": _cluster_info(cfg),
        }
    store = SqliteStateStore(cfg.state_db, read_only=True)
    try:
        rollouts = store.list_rollouts()
        sandboxes = store.list_sandboxes()
        # P1.7.B.3 case-2/3 rollouts (docker-py drop-in path) — count
        # them alongside case-1 sandbox-driven rollouts so the
        # operator-facing "rollouts running" / "finished in last
        # hour" / "failed in last hour" don't underreport when the
        # cluster is running raw harnesses (the dominant audience
        # under the slim pivot). Previously the overview queried
        # only ``list_rollouts()`` (case-1) and showed 0s while
        # /rollouts/raw plainly had running rows.
        #
        # These are bounded aggregate queries, NOT a full-table load:
        # tallying 168k raw rows in Python here was the read that pinned
        # the WAL open and stalled the control plane ([[wal-runaway-cp-stall]]).
        raw_status_counts = store.count_raw_rollouts_by_status()
        raw_active_nodes = store.active_raw_node_ids()
        cutoff = time.time() - 3600.0
        raw_finished_1h = store.count_raw_rollouts_finished_since(
            cutoff, ("released",),
        )
        raw_failed_1h = store.count_raw_rollouts_finished_since(
            cutoff, ("failed", "cancelled"),
        )
        # Backpressure declines are categorized separately (spec 13): they are
        # NOT failures (the acquire never ran; the consumer typically retries),
        # so they stay OUT of ``raw_failed_1h`` and get their own tile.
        raw_capacity_rejected_1h = store.count_raw_rollouts_finished_since(
            cutoff, ("capacity_rejected",),
        )
        # NodeRegistry's persistent shadow — gives us the count of
        # *connected* nodes, including idle ones that aren't currently
        # running a sandbox. Without this, a fresh cluster (or any
        # period between rollouts) shows "0 / 0 nodes (active / known)"
        # even though the Nodes page lists them as connected. Mirrors
        # the same union ``_nodes_blocking`` does for the page rows
        # so the overview count + the Nodes-page row count agree.
        registry_node_ids: set[str] = {n.node_id for n in store.list_nodes()}
        # Currently-connected subset, so the overview can flag rostered nodes
        # that have dropped off. Without this the overview stayed green through
        # a full-fleet loss (2026-08-21) because it only counted rollouts/
        # containers, never node liveness.
        connected_node_ids: set[str] = {
            n.node_id for n in store.list_nodes(status="connected")
        }
    finally:
        store.close()

    nodes = _load_nodes_yaml_lazy(cfg.nodes_yaml or Path("nodes.yaml"))
    nodes_active: set[str] = {sb.node_id for sb in sandboxes}
    # A node hosting an in-flight raw rollout is active too —
    # without this the overview showed "0 active" while case-2/3
    # rollouts were plainly visible on /rollouts/raw.
    nodes_active |= raw_active_nodes

    rostered_ids: set[str] = {
        str(n["id"]) for n in nodes if isinstance(n.get("id"), str)
    }
    nodes_total: set[str] = (
        nodes_active | rostered_ids | registry_node_ids
    )

    finished_1h = sum(
        1 for r in rollouts
        if r.status == RolloutStatus.FINISHED and r.last_touched_at >= cutoff
    )
    finished_1h += raw_finished_1h
    failed_1h = sum(
        1 for r in rollouts
        if r.status == RolloutStatus.FAILED and r.last_touched_at >= cutoff
    )
    failed_1h += raw_failed_1h
    running = sum(1 for r in rollouts if r.status == RolloutStatus.RUNNING)
    # Acquiring counts as in-flight: the AcquireContainerCommand has
    # been dispatched but the node hasn't ack'd yet — the operator
    # cares that *something* is happening, not which phase.
    running += raw_status_counts.get("acquiring", 0)
    running += raw_status_counts.get("running", 0)
    # "containers" = jobs with a live container *right now*. A raw
    # rollout in ``acquiring`` is still cold-pulling its image — no
    # container yet — so only ``running`` counts; each case-1 sandbox
    # row maps to a live container. The gap between this and
    # ``rollout_running`` (which also counts ``acquiring``) is the
    # cold-pull backlog the operator wants to see.
    container_count = len(sandboxes) + raw_status_counts.get("running", 0)
    # Rostered nodes that are NOT currently connected — "lost" (was connected)
    # or "absent" (never connected). Any nonzero value is an operator alarm the
    # overview must not hide behind green rollout tiles.
    nodes_lost = len(rostered_ids - connected_node_ids)
    return {
        "node_count": len(nodes_total),
        "node_active": len(nodes_active),
        "node_connected": len(connected_node_ids),
        "node_rostered": len(rostered_ids),
        "nodes_lost": nodes_lost,
        "container_count": container_count,
        "rollout_running": running,
        "rollout_finished_1h": finished_1h,
        "rollout_failed_1h": failed_1h,
        "rollout_capacity_rejected_1h": raw_capacity_rejected_1h,
        "uptime_s": time.time() - started_at,
        "state_db_present": True,
        "cluster_info": _cluster_info(cfg),
    }


async def _gather_nodes(cfg: AdminServerConfig) -> list[dict[str, Any]]:
    return await asyncio.to_thread(_nodes_blocking, cfg)


def _nodes_blocking(cfg: AdminServerConfig) -> list[dict[str, Any]]:
    """Build the admin ``/nodes`` row list.

    Unions three sources so a node shows up the moment it's relevant:

    - ``state.list_nodes()`` — :class:`NodeRegistry`'s persistent shadow,
      so a freshly-attached but idle node shows up immediately (without
      this, the page said "No nodes recorded yet" until the first
      rollout landed and wrote a sandbox row).
    - ``nodes.yaml`` — operator-rostered nodes (so an expected node
      that hasn't connected still surfaces as "absent" / unrostered=False).
    - ``state.list_sandboxes()`` — covers the pre-Slice-4 case where a
      node had sandboxes recorded against it before the registry table
      existed (or for an in-process runtime that doesn't write to the
      registry table).

    Mirrors the union the ``xrlenv nodes`` CLI does, so the two views
    don't disagree on what nodes exist.
    """
    rostered = _load_nodes_yaml_lazy(cfg.nodes_yaml or Path("nodes.yaml"))
    rostered_by_id: dict[str, dict[str, Any]] = {}
    for entry in rostered:
        if isinstance(entry.get("id"), str):
            rostered_by_id[entry["id"]] = entry
    sandboxes_by_node: dict[str, int] = {}
    nodes_by_id: dict[str, Any] = {}
    if cfg.state_db.exists():
        store = SqliteStateStore(cfg.state_db, read_only=True)
        try:
            for sb in store.list_sandboxes():
                sandboxes_by_node[sb.node_id] = sandboxes_by_node.get(sb.node_id, 0) + 1
            for n in store.list_nodes():
                nodes_by_id[n.node_id] = n
        finally:
            store.close()
    seen = set(sandboxes_by_node) | set(rostered_by_id) | set(nodes_by_id)
    now = time.time()
    rows: list[dict[str, Any]] = []
    for nid in sorted(seen):
        roster = rostered_by_id.get(nid)
        live = nodes_by_id.get(nid)
        if live is not None:
            status = live.status
            last_seen_age_s: float | None = max(0.0, now - live.last_seen_at)
        else:
            status = "absent"
            last_seen_age_s = None
        rows.append(
            {
                "id": nid,
                "status": status,
                "last_seen_age_s": last_seen_age_s,
                "rostered": roster is not None,
                "cloud": (roster or {}).get("cloud"),
                "expected_address": (
                    (roster or {}).get("expected_address")
                    or (roster or {}).get("address")
                ),
                # P6 step-2c (observability) — advertised CPU-isolation
                # capability + last-known pinnable-CPU counts (absent / pre-P6
                # rows read false / 0 / 0). Nothing schedules on these.
                "isolation_capable": bool(
                    getattr(live, "isolation_capable", False),
                ),
                "pinned_cpus_free": int(getattr(live, "pinned_cpus_free", 0)),
                "pinned_cpus_total": int(getattr(live, "pinned_cpus_total", 0)),
            }
        )
    return rows


def _node_distribution_blocking(
    cfg: AdminServerConfig, node_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """All-time raw-rollout count per node, for the /nodes distribution figure.

    Built from a single ``GROUP BY node_id`` aggregate. Bars cover every node
    in the current roster (0 included, so an idle node reads as imbalance);
    rollouts on nodes no longer rostered, and not-yet-assigned ones, are folded
    into reconciling notes rather than bars. Uniformity is summarised by the
    coefficient of variation over the rostered nodes (0% == perfectly even).
    """
    counts: dict[str | None, int] = {}
    if cfg.state_db.exists():
        store = SqliteStateStore(cfg.state_db, read_only=True)
        try:
            counts = store.count_raw_rollouts_by_node()
        finally:
            store.close()

    known_set = {str(r["id"]) for r in node_rows}
    unassigned = int(counts.get(None, 0))
    assigned = {str(nid): int(c) for nid, c in counts.items() if nid is not None}
    # Bars cover the CURRENT fleet only: every rostered node (0 included, so an
    # idle node reads as 0 = imbalance). Rollouts on nodes NO LONGER in the
    # roster — reboot-orphaned IP-derived node_ids that still carry history —
    # are folded into the ``off_roster`` reconciling note below rather than
    # drawn as per-node bars. An all-time figure would otherwise be dominated by
    # the previous fleet's dead ids after every reboot (node_id is IP-derived),
    # drowning out the live nodes at ~0%. Fall back to every node with history
    # when nothing is rostered yet, so the figure isn't empty on a bare CP.
    display_ids = known_set if known_set else set(assigned)
    per_node = {nid: assigned.get(nid, 0) for nid in display_ids}

    display_total = sum(per_node.values())
    max_count = max(per_node.values()) if per_node else 0
    entries = [
        {
            "node": nid,
            "count": per_node[nid],
            # Displayed bars are all rostered (or, with no roster yet, treated as
            # present); the off-roster "(gone)" case is the note, not a bar.
            "rostered": (nid in known_set) if known_set else True,
            "pct": (per_node[nid] / display_total * 100.0) if display_total else 0.0,
            "bar_pct": (per_node[nid] / max_count * 100.0) if max_count else 0.0,
        }
        for nid in sorted(per_node, key=lambda n: (-per_node[n], n))
    ]
    # Uniformity is judged over the displayed (rostered) nodes — the cluster you
    # actually balance.
    stat_values = list(per_node.values())
    n = len(stat_values)
    mean = sum(stat_values) / n if n else 0.0
    cv_pct = (statistics.pstdev(stat_values) / mean * 100.0) if (n and mean) else 0.0
    off_roster = (
        sum(c for nid, c in assigned.items() if nid not in known_set)
        if known_set
        else 0
    )
    return {
        "entries": entries,
        "total": display_total,
        "node_count": n,
        "mean": mean,
        # Where the mean sits on the bar scale (max == 100%), so the figure can
        # draw a reference line to eyeball spread around it.
        "mean_bar_pct": (mean / max_count * 100.0) if max_count else 0.0,
        "min": min(stat_values) if stat_values else 0,
        "max": max(stat_values) if stat_values else 0,
        "cv_pct": cv_pct,
        "off_roster": off_roster,
        "unassigned": unassigned,
    }


# Terminal raw-rollout statuses surfaced as their own column on /users.
_USER_TERMINAL_STATUSES = ("released", "failed", "cancelled", "reaped")
# In-flight statuses rolled into the "active" column.
_USER_ACTIVE_STATUSES = ("acquiring", "running")
# Backpressure/pacing declines (scheduler never placed the acquire). NOT a
# rollout outcome: shown as an informational "paced" column and EXCLUDED from
# the success-rate denominator, so a paced-then-retried run isn't scored as a
# partial failure. See ``capacity_rejected`` in xrlenv.control.state.
_USER_PACED_STATUSES = ("capacity_rejected",)


def _users_blocking(cfg: AdminServerConfig) -> dict[str, Any]:
    """Per-owner raw-rollout scoreboard for the operator-only /users page.

    One row per ``owner_id`` with total / released / failed / cancelled /
    reaped / active counts and ``released ÷ total`` as the success rate, from a
    single ``GROUP BY owner_id, status`` aggregate. Raw rollouts only — the
    ``released`` / ``reaped`` vocabulary is the raw-container plane's; case-1
    gym rollouts use a different status set and a different page.
    """
    agg: dict[str, dict[str, int]] = {}
    span: tuple[float | None, float | None] = (None, None)
    inception: float | None = None
    if cfg.state_db.exists():
        store = SqliteStateStore(cfg.state_db, read_only=True)
        try:
            # Cumulative per-owner/status = live raw_rollouts + the durable
            # lifetime tally of GC'd rows, read in ONE query so the retention
            # janitor can't move a row between two reads and double/under-count
            # it (audit M1). The tally accrues from when lifetime tracking was
            # enabled — rows pruned before that are not backfilled (audit H4).
            agg = store.aggregate_raw_rollouts_all_time_by_owner_status()
            # created_at window of the rows STILL in raw_rollouts — totals are
            # cumulative, but per-rollout detail only exists for this window.
            span = store.raw_rollouts_created_span()
            # The boundary below which pre-feature rollouts are NOT counted
            # (audit H4) — surfaced so the "cumulative since X" claim is concrete.
            inception = store.lifetime_inception_ts()
        finally:
            store.close()

    rows: list[dict[str, Any]] = []
    totals: dict[str, Any] = {
        k: 0 for k in ("total", "active", "paced", *_USER_TERMINAL_STATUSES)
    }
    for owner in sorted(agg):
        by_status = agg[owner]
        # ``paced`` (capacity_rejected) is a scheduler decline, not an outcome —
        # keep it OUT of ``total`` so the success rate reflects real attempts
        # (a paced-then-retried task is scored once, on its successful retry).
        paced = sum(by_status.get(s, 0) for s in _USER_PACED_STATUSES)
        total = sum(by_status.values()) - paced
        active = sum(by_status.get(s, 0) for s in _USER_ACTIVE_STATUSES)
        row: dict[str, Any] = {
            "owner": owner,
            "total": total,
            "active": active,
            "paced": paced,
            "success_pct": (
                by_status.get("released", 0) / total * 100.0 if total else None
            ),
        }
        for status in _USER_TERMINAL_STATUSES:
            row[status] = by_status.get(status, 0)
        rows.append(row)
        totals["total"] += total
        totals["active"] += active
        totals["paced"] += paced
        for status in _USER_TERMINAL_STATUSES:
            totals[status] += by_status.get(status, 0)

    rows.sort(key=lambda r: (-r["total"], r["owner"]))
    totals["success_pct"] = (
        totals["released"] / totals["total"] * 100.0 if totals["total"] else None
    )
    span_start, span_end = span
    return {
        "rows": rows,
        "totals": totals,
        # Retention window of the underlying raw_rollouts rows (None when empty).
        "span_start": _iso(span_start) if span_start is not None else None,
        "span_end": _iso(span_end) if span_end is not None else None,
        # Boundary below which pre-feature rollouts are not counted (H4).
        "inception": _iso(inception) if inception is not None else None,
    }


# ── Multi-user scoping (Slice B) ──────────────────────────────────────────────


def _caller_owner_id(request: Request) -> str | None:
    """The tenant the request is scoped to, or ``None`` for an admin view.

    Admin (sees every owner): the loopback / no-auth dev flow (no identity
    stashed) and any ``operator``-role token. Everyone else (a per-user
    ``viewer`` token) is scoped to their own ``owner_id``. The identity is
    stashed on ``request.state`` by ``_require_role`` during auth; absent it
    (loopback bypass, which never runs the auth helper) the caller is admin.
    """
    identity = getattr(request.state, "identity", None)
    if identity is None:
        return None
    if getattr(identity, "role", None) == "operator":
        return None
    return getattr(identity, "owner_id", None)


def _owner_forbidden(caller_owner: str | None, record_owner: str | None) -> bool:
    """True when a scoped caller may not see a record owned by another tenant.

    ``caller_owner is None`` is the admin view — never forbidden. A scoped
    caller may see only records whose owner matches theirs; a record with an
    unknown owner (``None``) is treated as not-theirs.
    """
    if caller_owner is None:
        return False
    return record_owner != caller_owner


def _rollout_owner_blocking(
    cfg: AdminServerConfig, rollout_id: str,
) -> str | None:
    """Resolve a rollout's ``owner_id`` by id across both the gym/step and
    raw (case-2/3) tables. ``None`` when the id is unknown.

    Used to owner-gate the per-rollout file/artifact/log routes that fetch by
    run-dir rather than by record, so a scoped user can't read another
    tenant's artifacts by guessing an id.
    """
    if not cfg.state_db.exists():
        return None
    store = SqliteStateStore(cfg.state_db, read_only=True)
    try:
        try:
            return store.get_rollout(rollout_id).owner_id
        except KeyError:
            pass
        get_raw = getattr(store, "get_raw_rollout", None)
        if get_raw is not None:
            raw = get_raw(rollout_id)
            if raw is not None:
                return raw.owner_id
        return None
    finally:
        store.close()


async def _gather_rollouts(
    cfg: AdminServerConfig,
    *,
    status: str | None,
    template: str | None,
    since: str | None,
    page: int,
    page_size: int,
    owner_id: str | None = None,
) -> RolloutPage:
    return await asyncio.to_thread(
        _rollouts_blocking, cfg, status, template, since, page, page_size,
        owner_id,
    )


def _rollouts_blocking(
    cfg: AdminServerConfig,
    status: str | None,
    template: str | None,
    since: str | None,
    page: int,
    page_size: int,
    owner_id: str | None = None,
) -> RolloutPage:
    page = max(1, page)
    page_size = max(1, min(page_size, _MAX_ROLLOUT_PAGE_SIZE))
    if not cfg.state_db.exists():
        return RolloutPage(
            records=[], page=page, page_size=page_size, has_next=False,
        )
    since_s = parse_duration_lazy(since) if since else None
    created_after = time.time() - since_s if since_s is not None else None
    store = SqliteStateStore(cfg.state_db, read_only=True)
    try:
        records, has_next = store.list_rollouts_page(
            status=status or None,
            template=template or None,
            created_after=created_after,
            owner_id=owner_id,
            limit=page_size,
            offset=(page - 1) * page_size,
        )
    finally:
        store.close()
    return RolloutPage(
        records=records, page=page, page_size=page_size, has_next=has_next,
    )


def _rollout_dict(r: RolloutRecord) -> dict[str, Any]:
    return {
        "rollout_id": r.rollout_id,
        "template": r.template,
        "status": r.status.value,
        "reason": r.reason,
        "node_id": r.node_id,
        "task_key": r.task_key,
        "group_id": r.group_id,
        "step_count": len(r.steps),
        "final_reward": r.final_reward,
        "created_at": r.created_at,
    }


_COORDINATOR_START_EVENT = "rollout.start"
_COORDINATOR_TERMINAL_EVENTS = frozenset(
    {"rollout.finish", "rollout.truncate", "rollout.cancel", "rollout.fail"}
)


def _rollout_durations_blocking(
    cfg: AdminServerConfig, records: list[Any], now: float,
) -> dict[str, dict[str, Any]]:
    return {
        r.rollout_id: _rollout_duration_snapshot(cfg, r, now)
        for r in records
    }


# ──────────────────────────────────────────────────────────────────────────────
# P1.7.B.3 — raw rollouts (case-2/3 evaluation harness tracking)
# ──────────────────────────────────────────────────────────────────────────────


# Mirrors ``xrlenv.control.state.RawRolloutStatus`` literal values.
# Pulled out as a tuple so the admin route's ``raw_statuses``
# dropdown can render them without importing typing.Literal at
# runtime.
_RAW_ROLLOUT_STATUSES: tuple[str, ...] = (
    "acquiring", "running", "released", "cancelled", "failed", "reaped",
    "capacity_rejected",
)

# Per-file artifact preview cap. Files larger than this serve as
# octet-stream download (no inline render) so the browser doesn't
# choke on a multi-GB log. Common case (swebench's run_instance.log,
# report.json, test_output.txt) is well under this limit.
_ARTIFACT_PREVIEW_MAX_BYTES: int = 1 * 1024 * 1024  # 1 MiB


def _resolve_artifact_file(
    artifact_path: str, file_path: str,
) -> Path | None:
    """Resolve ``<artifact_path>/<file_path>`` with traversal guard.

    Returns the resolved Path iff:
    - The parent ``artifact_path`` exists and is a directory.
    - The resolved target is a regular file.
    - The resolved target is under the parent (no ``..`` escapes,
      no symlinks pointing outside).

    Returns ``None`` for missing / not-a-file. Permissions are
    checked at read time, not here, so a permission-denied path
    surfaces with a clean 403 from the caller.
    """
    parent = Path(artifact_path).resolve(strict=False)
    if not parent.is_dir():
        return None
    try:
        target = (parent / file_path).resolve(strict=False)
    except OSError:
        return None
    # Path-traversal guard: target must be under parent.
    try:
        target.relative_to(parent)
    except ValueError:
        return None
    if not target.is_file():
        return None
    return target


def _read_artifact_file(
    path: Path,
) -> tuple[str, str | None, int]:
    """Read an artifact file for inline preview.

    Returns ``(kind, body, size)`` where:
    - ``kind == "text"``: file is UTF-8 decodable AND under the
      preview cap. ``body`` is the decoded text (truncated to the
      cap with a banner on overflow). Caller serves as
      ``text/plain``.
    - ``kind == "binary"``: file is not UTF-8 decodable OR too
      large for inline preview. ``body`` is ``None``; caller
      streams as ``application/octet-stream``.
    - ``kind == "denied"``: ``os.access`` says the admin process
      can't read the file; caller serves 403.

    The text/binary heuristic is "tried to decode the first chunk
    as UTF-8; failed → binary." Cheap + good enough for log files +
    report.json + patch.diff.
    """
    if not os.access(path, os.R_OK):
        return "denied", None, 0
    try:
        size = path.stat().st_size
    except OSError:
        return "denied", None, 0
    if size > _ARTIFACT_PREVIEW_MAX_BYTES:
        # Oversize: try to detect text vs binary on a small head
        # sample so we can still inline a head excerpt for text.
        with path.open("rb") as fh:
            head = fh.read(min(_ARTIFACT_PREVIEW_MAX_BYTES, 65536))
        try:
            head_text = head.decode("utf-8")
        except UnicodeDecodeError:
            return "binary", None, size
        # Inline a head excerpt with a banner.
        banner = (
            f"# xrlenv admin: truncated to first {len(head)} bytes "
            f"(file is {size} bytes; full content available via "
            f"the raw download link below or by SSH).\n"
            f"# ---\n"
        )
        return "text", banner + head_text, size
    # Under cap: try the full read.
    try:
        body = path.read_bytes()
    except OSError:
        return "denied", None, size
    try:
        return "text", body.decode("utf-8"), size
    except UnicodeDecodeError:
        return "binary", None, size


def _parse_since(spec: str | None) -> float | None:
    """Parse a ``since`` query-param duration spec (e.g. ``"5m"`` /
    ``"2h"`` / ``"1d"``) into a unix-epoch ``since_after`` cutoff.

    Returns ``None`` on missing/empty/invalid input — the caller's
    list query then has no time filter (graceful degrade for stale
    URLs / typos).
    """
    if not spec:
        return None
    try:
        seconds = parse_duration_lazy(spec)
    except Exception:
        LOGGER.warning("admin: ignoring bad since=%r", spec)
        return None
    return time.time() - seconds


def _list_raw_rollouts_paginated_blocking(
    cfg: AdminServerConfig,
    raw_status: str | None,
    since_after: float | None,
    task_key: str | None,
    group_id: str | None,
    limit: int,
    offset: int,
    owner_id: str | None = None,
) -> tuple[list[Any], int]:
    """Read a page of raw rollouts plus the total count for the
    same filter set. Best-effort: missing DB / older StateStore
    that doesn't expose ``count_raw_rollouts`` returns
    ``([], 0)``. ``task_key`` / ``group_id`` filters are forwarded
    as kwargs when the underlying store supports them; older
    stores predating those kwargs treat the filter as a no-op
    via TypeError fallback."""
    if not cfg.state_db.exists():
        return [], 0
    store = SqliteStateStore(cfg.state_db, read_only=True)
    list_fn = getattr(store, "list_raw_rollouts", None)
    count_fn = getattr(store, "count_raw_rollouts", None)
    if list_fn is None:
        return [], 0
    status_arg: str | None = None
    if raw_status:
        if raw_status in _RAW_ROLLOUT_STATUSES:
            status_arg = raw_status
        else:
            LOGGER.warning(
                "admin: ignoring raw_status=%r (not in %s)",
                raw_status, _RAW_ROLLOUT_STATUSES,
            )
    try:
        rows = list(list_fn(
            status=status_arg, since_after=since_after,
            task_key=task_key, group_id=group_id, owner_id=owner_id,
            limit=limit, offset=offset,
        ))
    except TypeError:
        # Older StateStore predating task_key/group_id/owner_id kwargs:
        # fall back to the un-filtered call so the page still renders.
        # Filters become a no-op rather than 500-ing — but a scoped
        # (non-admin) caller must still never see another tenant's rows,
        # so re-apply the owner filter in Python on the way out.
        LOGGER.info(
            "admin: list_raw_rollouts predates task_key/group_id/owner_id; "
            "filtering applied in-process",
        )
        rows = list(list_fn(
            status=status_arg, since_after=since_after,
            limit=limit, offset=offset,
        ))
        if owner_id is not None:
            rows = [r for r in rows if getattr(r, "owner_id", "default") == owner_id]
    except Exception:
        LOGGER.exception("admin: list_raw_rollouts failed")
        return [], 0
    total = 0
    if count_fn is not None:
        try:
            total = int(count_fn(
                status=status_arg, since_after=since_after,
                task_key=task_key, group_id=group_id, owner_id=owner_id,
            ))
        except TypeError:
            try:
                total = int(count_fn(
                    status=status_arg, since_after=since_after,
                ))
            except Exception:
                LOGGER.exception("admin: count_raw_rollouts failed")
        except Exception:
            LOGGER.exception("admin: count_raw_rollouts failed")
    return rows, total


def _list_raw_rollouts_blocking(
    cfg: AdminServerConfig, limit: int,
    raw_status: str | None = None,
) -> list[Any]:
    """Read recent raw rollouts (newest-first) from the StateStore.

    Best-effort: when the state DB is absent or the StateStore
    predates the ``list_raw_rollouts`` method, return an empty
    list so the rollouts page still renders. Invalid raw_status
    values (not in :data:`_RAW_ROLLOUT_STATUSES`) get logged and
    treated as no filter — better than 500-ing on a bookmarked
    URL with a stale status string.
    """
    if not cfg.state_db.exists():
        return []
    store = SqliteStateStore(cfg.state_db, read_only=True)
    list_fn = getattr(store, "list_raw_rollouts", None)
    if list_fn is None:
        return []
    status_arg: str | None = None
    if raw_status:
        if raw_status in _RAW_ROLLOUT_STATUSES:
            status_arg = raw_status
        else:
            LOGGER.warning(
                "admin: ignoring raw_status=%r (not in %s)",
                raw_status, _RAW_ROLLOUT_STATUSES,
            )
    try:
        return list(list_fn(limit=limit, status=status_arg))
    except Exception:
        LOGGER.exception("admin: list_raw_rollouts failed")
        return []


def _get_raw_rollout_blocking(
    cfg: AdminServerConfig, rollout_id: str,
) -> Any | None:
    if not cfg.state_db.exists():
        return None
    store = SqliteStateStore(cfg.state_db, read_only=True)
    get_fn = getattr(store, "get_raw_rollout", None)
    if get_fn is None:
        return None
    try:
        return get_fn(rollout_id)
    except Exception:
        LOGGER.exception(
            "admin: get_raw_rollout(%s) failed", rollout_id,
        )
        return None


def _resolve_artifact_path(
    path_str: str,
) -> tuple[str, list[dict[str, Any]]]:
    """Best-effort filesystem read of the consumer-recorded artifact
    path. Returns ``(status, listing)`` where ``status`` is one of
    ``"missing" | "denied" | "not_a_dir" | "ok"``.

    Lists immediate children only — no recursion, no traversal — to
    keep the attack surface minimal. Operators wanting deeper
    inspection SSH to the consumer host directly (or share the
    artifact tree on a filesystem the admin host can read).
    """
    from pathlib import Path

    p = Path(path_str)
    try:
        if not p.exists():
            return "missing", []
    except OSError:
        # PermissionError on the parent dir at e.g. ``stat()``.
        return "denied", []
    if not p.is_dir():
        return "not_a_dir", []
    if not os.access(p, os.R_OK | os.X_OK):
        return "denied", []
    listing: list[dict[str, Any]] = []
    try:
        for entry in sorted(p.iterdir(), key=lambda e: e.name):
            if entry.is_dir():
                listing.append({"name": entry.name, "kind": "dir"})
            elif entry.is_file():
                try:
                    size = entry.stat().st_size
                except OSError:
                    size = -1
                listing.append({
                    "name": entry.name, "kind": "file",
                    "size_bytes": size,
                })
    except PermissionError:
        return "denied", []
    return "ok", listing


def _iso(epoch: float) -> str:
    """Render a UTC ISO-8601 timestamp from a unix epoch."""
    import datetime
    return datetime.datetime.fromtimestamp(
        epoch, tz=datetime.UTC,
    ).strftime("%Y-%m-%d %H:%M:%S UTC")


def _rollout_duration_snapshot(
    cfg: AdminServerConfig, record: Any, now: float,
) -> dict[str, Any]:
    # State-store is the source of truth for ``live`` (= "is the rollout
    # in a non-terminal status?"). The coordinator.log is a per-rollout
    # event journal — useful for accurate start/end timestamps, but it
    # can lag behind state.db: a rollout force-sealed via the
    # startup-sweep, the SQL cleanup script, or a future admin RPC
    # writes to state.db without emitting a ``rollout.cancel`` event
    # to the log. Pre-fix the snapshot trusted the log's "no terminal
    # event yet" reading, so a row with ``status='cancelled'`` rendered
    # as "30.6 min live" — a confusing UX bug operators reported during
    # the multi-VM smoke recovery.
    state_terminal = bool(record.status.is_terminal)
    bounds = _coordinator_log_lifecycle_bounds(cfg, record.rollout_id)
    if bounds is not None:
        start_ts, terminal_ts = bounds
        if state_terminal:
            # Trust state-store on liveness; prefer the log's terminal
            # ts when the coordinator wrote one, else fall back to
            # ``last_touched_at`` (which the state-store updates on
            # every status transition).
            end_ts = terminal_ts if terminal_ts is not None else record.last_touched_at
            return {
                "seconds": max(0.0, end_ts - start_ts),
                "live": False,
                "source": "coordinator.log+state",
            }
        end_ts = terminal_ts if terminal_ts is not None else now
        return {
            "seconds": max(0.0, end_ts - start_ts),
            "live": terminal_ts is None,
            "source": "coordinator.log",
        }

    # Fallback for rollouts that failed before the sink opened, in-memory test
    # fixtures, or old run dirs without per-rollout coordinator logs.
    end_ts = record.last_touched_at if state_terminal else now
    return {
        "seconds": max(0.0, end_ts - record.created_at),
        "live": not state_terminal,
        "source": "state timestamps",
    }


def _coordinator_log_lifecycle_bounds(
    cfg: AdminServerConfig, rollout_id: str,
) -> tuple[float, float | None] | None:
    run_dir = _run_dir_for_rollout(cfg, rollout_id)
    if run_dir is None:
        return None
    log_path = run_dir / "coordinator.log"
    if not log_path.is_file():
        return None

    start_ts: float | None = None
    terminal_ts: float | None = None
    try:
        with log_path.open("r", encoding="utf-8") as fp:
            for line in fp:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                event = record.get("event")
                ts = _parse_coordinator_log_ts(record.get("ts"))
                if ts is None:
                    continue
                if event == _COORDINATOR_START_EVENT and start_ts is None:
                    start_ts = ts
                elif event in _COORDINATOR_TERMINAL_EVENTS:
                    terminal_ts = ts
    except OSError:
        return None
    if start_ts is None:
        return None
    return start_ts, terminal_ts


def _parse_coordinator_log_ts(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        text = value[:-1] + "+00:00" if value.endswith("Z") else value
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


async def _gather_fairshare(cfg: AdminServerConfig) -> dict[str, Any]:
    return await asyncio.to_thread(_fairshare_blocking, cfg)


def _fairshare_blocking(cfg: AdminServerConfig) -> dict[str, Any]:
    """Read the live fair-share policy + per-owner usage for the admin tab."""
    empty = {"enabled": False, "capacity_basis": None, "floor": 1, "rows": []}
    if not cfg.state_db.exists():
        return empty
    store = SqliteStateStore(cfg.state_db, read_only=True)
    try:
        get_policy = getattr(store, "get_fairness_policy", None)
        get_counts = getattr(store, "running_counts_by_owner", None)
        if get_policy is None or get_counts is None:
            return empty
        policy = get_policy()
        counts = get_counts()
    finally:
        store.close()
    active = set(counts) | set(policy.overrides)
    rows: list[dict[str, Any]] = []
    for owner in sorted(active):
        ov = policy.overrides.get(owner)
        rows.append({
            "owner": owner,
            "running": counts.get(owner, 0),
            "cap": policy.cap_for(owner, active) if policy.enabled else None,
            "owner_cap": ov.hard_cap if ov is not None else None,
            "uncapped": ov.uncapped if ov is not None else False,
            "blocked": ov.blocked if ov is not None else False,
        })
    return {
        "enabled": policy.enabled,
        "capacity_basis": policy.capacity_basis,
        "floor": policy.floor,
        "rows": rows,
    }


async def _gather_sandboxes(
    cfg: AdminServerConfig,
    *,
    node: str | None,
    template: str | None,
    owner_id: str | None = None,
) -> list[dict[str, Any]]:
    return await asyncio.to_thread(
        _sandboxes_blocking, cfg, node, template, owner_id,
    )


def _sandboxes_blocking(
    cfg: AdminServerConfig,
    node: str | None,
    template: str | None,
    owner_id: str | None = None,
) -> list[dict[str, Any]]:
    if not cfg.state_db.exists():
        return []
    store = SqliteStateStore(cfg.state_db, read_only=True)
    try:
        sandboxes = store.list_sandboxes()
        # SandboxRecord carries no owner of its own, so when the caller is
        # scoped (non-admin) we join each sandbox to its rollout's owner_id
        # via rollout_id. A sandbox with no rollout_id (or an unknown one)
        # has no resolvable owner and is hidden from scoped callers.
        owner_cache: dict[str, str | None] = {}

        def _owner_of(rollout_id: str | None) -> str | None:
            if rollout_id is None:
                return None
            if rollout_id not in owner_cache:
                resolved: str | None
                try:
                    resolved = store.get_rollout(rollout_id).owner_id
                except KeyError:
                    get_raw = getattr(store, "get_raw_rollout", None)
                    raw = get_raw(rollout_id) if get_raw is not None else None
                    resolved = raw.owner_id if raw is not None else None
                owner_cache[rollout_id] = resolved
            return owner_cache[rollout_id]

        rows: list[dict[str, Any]] = []
        for sb in sandboxes:
            if node and sb.node_id != node:
                continue
            if template and sb.template != template:
                continue
            if owner_id is not None and _owner_of(sb.rollout_id) != owner_id:
                continue
            rows.append(
                {
                    "sandbox_id": sb.sandbox_id,
                    "node_id": sb.node_id,
                    "template": sb.template,
                    "image": sb.image,
                    "rollout_id": sb.rollout_id,
                    "backend": sb.backend,
                    "status": sb.status,
                    "created_at": sb.created_at,
                }
            )
    finally:
        store.close()
    rows.sort(key=lambda r: r["created_at"], reverse=True)
    return rows


async def _gather_images(
    cfg: AdminServerConfig,
    *,
    request: Request,
    view: str = "nodes",
    q: str | None = None,
    tier: str | None = None,
    pressure: str | None = None,
    pinned: str | None = None,
    sort: str | None = None,
    page: int = 1,
    page_size: int = _DEFAULT_IMAGE_PAGE_SIZE,
    include_filter: ImageIncludeFilter = "all",
    hide_intermediate: bool = False,
    xrlenv_only: bool = False,
    snapshot: _ImageRowsSnapshot | None = None,
) -> dict[str, Any]:
    """B7.6 (P1.2.c) — cluster-wide image cache snapshot.

    Fans out a ``ReportImagesCommand`` to every connected node via
    ``cfg.node_lookup`` (the embedded-runtime callback) and aggregates
    two per-tier views: ``totals_summed`` (every node·image instance
    counted, the right answer for eviction-pressure questions) and
    ``totals_distinct`` (deduped by image name, the right answer for
    "what tasks does the cluster cover?"). The data is fetched
    on-demand on each render — image-cache state doesn't move every
    second so the simpler "no caching" approach is fine and avoids a
    new state-store table for what is essentially live diagnostic
    data.

    When ``cfg.node_lookup`` is unwired (standalone admin process,
    no live runtime), the page renders an explanatory hint instead
    of partial data.
    """
    view = _normalize_image_view(view)
    tier = _normalize_image_tier(tier)
    pressure = _normalize_image_pressure(pressure)
    pinned = _normalize_image_pinned(pinned)
    form_action = "/images/cache" if view == "nodes" else "/images/catalog"
    if cfg.node_lookup is None:
        return {
            "node_lookup_wired": False,
            "rows": [],
            "totals_summed": _empty_image_totals(),
            "totals_distinct": _empty_image_totals(),
            "summary": _empty_image_summary(),
            "risks": [],
            "view": view,
            "q": q,
            "tier": tier,
            "pressure": pressure,
            "pinned": pinned,
            "sort": sort,
            "node_page": _empty_page(page=page, page_size=page_size),
            "image_page": _empty_page(page=page, page_size=page_size),
            "page_size_options": _IMAGE_PAGE_SIZE_OPTIONS,
            "tier_options": _IMAGE_TIERS,
            "pressure_options": ("critical", "watch", "unreachable", "ok"),
            "nodes_view_url": "/images/cache",
            "images_view_url": "/images/catalog",
            "images_form_action": form_action,
            "include_filter": include_filter,
            "image_include_options": _IMAGE_INCLUDE_OPTIONS,
            "hide_intermediate": hide_intermediate,
            "xrlenv_only": xrlenv_only,
            "ownership_total": 0,
            "ownership_visible": 0,
            "ownership_breakdown": {
                "xrlenv_final": 0,
                "xrlenv_intermediate": 0,
                "external": 0,
            },
            "image_snapshot_age_s": 0.0,
            "image_snapshot_refreshing": False,
        }
    node_ids = await asyncio.to_thread(_list_known_node_ids, cfg)
    if snapshot is not None:
        rows, snapshot_age_s, snapshot_refreshing = await snapshot.rows_for(
            cfg, node_ids,
        )
    else:
        rows = await _fetch_image_rows_for_nodes(cfg, node_ids)
        snapshot_age_s, snapshot_refreshing = 0.0, False
    totals_summed, totals_distinct = _aggregate_image_totals(rows)
    node_rows = _shape_image_node_rows(rows)
    # Catalog (image-coverage) shape consumes the per-node ``images``
    # lists. Apply the ownership filter at shape time so the
    # aggregate cards (distinct_images, copied_bytes, pinned_bytes)
    # all reflect the include filter consistently with the row table.
    image_rows_unfiltered = _shape_distinct_image_rows(rows)
    if hide_intermediate or xrlenv_only:
        image_rows = _shape_distinct_image_rows(
            rows,
            ownership_filter=lambda img: _image_passes_ownership(
                img, hide_intermediate=hide_intermediate,
                xrlenv_only=xrlenv_only,
            ),
        )
    else:
        image_rows = image_rows_unfiltered
    ownership_total = len(image_rows_unfiltered)
    ownership_visible = len(image_rows)
    # Cluster-wide breakdown of distinct images by owner — operators
    # need this to debug why the include filter "does nothing": images
    # built before commit 12237ea (admin labeling) classify as
    # ``external`` because they carry no ``org.xrlenv.owned`` label, so
    # the default xrlenv-only filter drops them all and operators see
    # the dropdown change nothing visible.
    ownership_breakdown = _ownership_breakdown(rows)
    summary = _image_summary(rows, node_rows, image_rows)
    filtered_node_rows = _filter_image_node_rows(
        node_rows, q=q, tier=tier, pressure=pressure, pinned=pinned,
    )
    filtered_image_rows = _filter_distinct_image_rows(
        image_rows, q=q, tier=tier, pinned=pinned,
    )
    node_sort = sort if view == "nodes" else None
    image_sort = sort if view == "images" else None
    filtered_node_rows = _sort_image_node_rows(filtered_node_rows, node_sort)
    filtered_image_rows = _sort_distinct_image_rows(filtered_image_rows, image_sort)
    node_page = _page_items(filtered_node_rows, page=page, page_size=page_size)
    image_page = _page_items(filtered_image_rows, page=page, page_size=page_size)
    _attach_image_page_urls(request, node_page)
    _attach_image_page_urls(request, image_page)
    return {
        "node_lookup_wired": True,
        "rows": rows,
        "totals_summed": totals_summed,
        "totals_distinct": totals_distinct,
        "summary": summary,
        "risks": _image_risks(summary, node_rows),
        "view": view,
        "q": q,
        "tier": tier,
        "pressure": pressure,
        "pinned": pinned,
        "sort": sort or ("pressure" if view == "nodes" else "operational"),
        "node_page": node_page,
        "image_page": image_page,
        "page_size_options": _IMAGE_PAGE_SIZE_OPTIONS,
        "tier_options": _IMAGE_TIERS,
        "pressure_options": ("critical", "watch", "unreachable", "ok"),
        "nodes_view_url": "/images/cache",
        "images_view_url": "/images/catalog",
        "images_form_action": form_action,
        "include_filter": include_filter,
        "image_include_options": _IMAGE_INCLUDE_OPTIONS,
        "hide_intermediate": hide_intermediate,
        "xrlenv_only": xrlenv_only,
        "ownership_total": ownership_total,
        "ownership_visible": ownership_visible,
        "ownership_breakdown": ownership_breakdown,
        "image_snapshot_age_s": snapshot_age_s,
        "image_snapshot_refreshing": snapshot_refreshing,
    }


async def _gather_image_node_detail(
    cfg: AdminServerConfig,
    *,
    request: Request,
    node_id: str,
    q: str | None = None,
    tier: str | None = None,
    pinned: str | None = None,
    sort: str | None = None,
    page: int = 1,
    page_size: int = _DEFAULT_IMAGE_PAGE_SIZE,
    include_filter: ImageIncludeFilter = "default",
    hide_intermediate: bool = True,
    xrlenv_only: bool = True,
) -> dict[str, Any]:
    tier = _normalize_image_tier(tier)
    pinned = _normalize_image_pinned(pinned)
    if cfg.node_lookup is None:
        return {
            "node_lookup_wired": False,
            "node_id": node_id,
            "row": _unreachable_image_row(node_id, "no live node-lookup wired"),
            "image_page": _empty_page(page=page, page_size=page_size),
            "q": q,
            "tier": tier,
            "pinned": pinned,
            "sort": sort or "operational",
            "include_filter": include_filter,
            "image_include_options": _IMAGE_INCLUDE_OPTIONS,
            "hide_intermediate": hide_intermediate,
            "xrlenv_only": xrlenv_only,
            "ownership_total": 0,
            "ownership_visible": 0,
            "page_size_options": _IMAGE_PAGE_SIZE_OPTIONS,
            "tier_options": _IMAGE_TIERS,
        }
    known = await asyncio.to_thread(_list_known_node_ids, cfg)
    if node_id not in known:
        raise HTTPException(status_code=404, detail=f"node {node_id} not connected")
    row = await _fetch_image_row_for(cfg, node_id)
    image_rows = _shape_single_node_image_rows(row)
    ownership_total = len(image_rows)
    image_rows = _filter_image_rows_by_ownership(
        image_rows, hide_intermediate=hide_intermediate, xrlenv_only=xrlenv_only,
    )
    ownership_visible = len(image_rows)
    image_rows = _filter_single_node_image_rows(
        image_rows, q=q, tier=tier, pinned=pinned,
    )
    image_rows = _sort_single_node_image_rows(image_rows, sort)
    image_page = _page_items(image_rows, page=page, page_size=page_size)
    _attach_image_page_urls(request, image_page)
    return {
        "node_lookup_wired": True,
        "node_id": node_id,
        "row": _shape_image_node_rows([row])[0],
        "image_page": image_page,
        "q": q,
        "tier": tier,
        "pinned": pinned,
        "sort": sort or "operational",
        "include_filter": include_filter,
        "image_include_options": _IMAGE_INCLUDE_OPTIONS,
        "hide_intermediate": hide_intermediate,
        "xrlenv_only": xrlenv_only,
        "ownership_total": ownership_total,
        "ownership_visible": ownership_visible,
        "page_size_options": _IMAGE_PAGE_SIZE_OPTIONS,
        "tier_options": _IMAGE_TIERS,
    }


def _filter_image_rows_by_ownership(
    rows: list[dict[str, Any]],
    *,
    hide_intermediate: bool,
    xrlenv_only: bool,
) -> list[dict[str, Any]]:
    """Apply the B7.6 admin default-on filters.

    ``hide_intermediate=True`` drops rows tagged ``xrlenv_intermediate``
    (build byproducts). ``xrlenv_only=True`` drops rows tagged
    ``external`` (operator-foreign images xrlenv didn't build).
    Both default-on; toggling either off restores those rows.
    """
    out = rows
    if hide_intermediate:
        out = [row for row in out if row.get("owner") != "xrlenv_intermediate"]
    if xrlenv_only:
        out = [row for row in out if row.get("owner") != "external"]
    return out


async def _gather_image_detail(
    cfg: AdminServerConfig,
    *,
    request: Request,
    image_ref: str,
    snapshot: _ImageRowsSnapshot | None = None,
) -> dict[str, Any]:
    if cfg.node_lookup is None:
        return {
            "node_lookup_wired": False,
            "image_ref": image_ref,
            "image": None,
            "instances": [],
        }
    node_ids = await asyncio.to_thread(_list_known_node_ids, cfg)
    if snapshot is not None:
        rows, _age, _refreshing = await snapshot.rows_for(cfg, node_ids)
    else:
        rows = await _fetch_image_rows_for_nodes(cfg, node_ids)
    image_rows = _shape_distinct_image_rows(rows)
    by_name = {row["name"]: row for row in image_rows}
    image = by_name.get(image_ref)
    if image is None:
        raise HTTPException(status_code=404, detail=f"image {image_ref!r} not found")
    instances: list[dict[str, Any]] = []
    for row in rows:
        if not row.get("reachable"):
            continue
        shaped_node = _shape_image_node_rows([row])[0]
        for img in row.get("images", []):
            if img["name"] != image_ref:
                continue
            instances.append({
                "node_id": row["node_id"],
                "node_detail_url": shaped_node["detail_url"],
                "tier": img["tier"],
                "tier_rank": _tier_rank(img["tier"]),
                "size_bytes": img["size_bytes"],
                "in_use_count": img["in_use_count"],
                "pinned": img["pinned"],
                "last_used_at": img["last_used_at"],
                "free_disk_bytes": row["free_disk_bytes"],
                "pressure": shaped_node["pressure"],
                "pressure_label": shaped_node["pressure_label"],
            })
    instances.sort(key=lambda i: (-i["tier_rank"], i["node_id"]))
    return {
        "node_lookup_wired": True,
        "image_ref": image_ref,
        "image": image,
        "instances": instances,
        "request": request,
    }


def _normalize_image_view(view: str) -> str:
    if view not in {"nodes", "images"}:
        raise HTTPException(status_code=422, detail="view must be 'nodes' or 'images'")
    return view


def _normalize_image_tier(tier: str | None) -> str | None:
    if tier in {None, ""}:
        return None
    if tier not in _IMAGE_TIERS:
        raise HTTPException(
            status_code=422,
            detail=f"tier must be one of {', '.join(_IMAGE_TIERS)}",
        )
    return tier


def _normalize_image_pressure(pressure: str | None) -> str | None:
    if pressure in {None, ""}:
        return None
    allowed = {"critical", "watch", "unreachable", "ok"}
    if pressure not in allowed:
        raise HTTPException(
            status_code=422,
            detail=f"pressure must be one of {', '.join(sorted(allowed))}",
        )
    return pressure


def _normalize_image_pinned(pinned: str | None) -> str | None:
    if pinned in {None, ""}:
        return None
    allowed = {"yes", "no"}
    if pinned not in allowed:
        raise HTTPException(status_code=422, detail="pinned must be yes or no")
    return pinned


async def _fetch_image_rows_for_nodes(
    cfg: AdminServerConfig, node_ids: list[str],
) -> list[dict[str, Any]]:
    sem = asyncio.Semaphore(32)
    # Backstop above the transport's own ``report_images`` timeout
    # (XRLENV_REPORT_IMAGES_TIMEOUT_S, default 25 s) so the transport
    # ceiling fires first — a slow ``docker system df`` then surfaces a
    # clean "report_images failed: … timed out" row instead of this
    # wait_for cancelling the RPC (which pops the pending future and
    # leaves the node's late reply logging "unknown command_id"). The old
    # flat 5 s ceiling tripped on every node once a real catalog landed.
    backstop_s = float(
        os.environ.get("XRLENV_REPORT_IMAGES_TIMEOUT_S", "60"),
    ) + 5.0

    async def _one(node_id: str) -> dict[str, Any]:
        async with sem:
            try:
                return await asyncio.wait_for(
                    _fetch_image_row_for(cfg, node_id),
                    timeout=backstop_s,
                )
            except TimeoutError:
                return _unreachable_image_row(node_id, "report_images timed out")

    return await asyncio.gather(*(_one(nid) for nid in node_ids))


# Default freshness window for the cluster image-rows snapshot. Beyond this,
# the next /images render serves the cached rows but kicks off a background
# refresh. Kept well under the page's 60 s auto-refresh so an open page stays
# roughly one refresh-cycle fresh. Tunable via XRLENV_IMAGE_SNAPSHOT_TTL_S.
_IMAGE_SNAPSHOT_TTL_S = float(os.environ.get("XRLENV_IMAGE_SNAPSHOT_TTL_S", "15"))


class _ImageRowsSnapshot:
    """Stale-while-revalidate cache of the cluster image-rows fan-out.

    The /images pages fan out a ``report_images`` RPC to every node on each
    render. Under a heavy ``build apply`` that fan-out can take tens of
    seconds, so this cache decouples page latency from node latency:

    - When a snapshot exists for the current node set it is served
      **immediately**, tagged with its age; once the snapshot is older than
      ``ttl_s`` a background refresh is kicked off so the next render sees
      fresh data. The page never blocks on the fan-out while any snapshot
      exists.
    - The first render (cold cache) — or a render after the node set changes —
      blocks on one fan-out, single-flighted via ``_cold_lock`` so concurrent
      renders don't stampede.

    One instance per admin app (created in :func:`create_app`); it is
    request-triggered, so it does no work when nobody is viewing the page
    (no perpetual background task, which also matters because the server runs
    uvicorn with ``lifespan="off"``).
    """

    def __init__(self, *, ttl_s: float = _IMAGE_SNAPSHOT_TTL_S) -> None:
        self._ttl_s = ttl_s
        self._rows: list[dict[str, Any]] | None = None
        self._node_key: tuple[str, ...] = ()
        self._at: float = 0.0
        self._refreshing = False
        self._refresh_task: asyncio.Task[None] | None = None
        self._cold_lock = asyncio.Lock()

    async def rows_for(
        self, cfg: AdminServerConfig, node_ids: list[str],
    ) -> tuple[list[dict[str, Any]], float, bool]:
        """Return ``(rows, age_s, refreshing)`` for ``node_ids``.

        Serves the cached snapshot instantly when one exists for this node
        set; blocks only on a cold cache or a node-set change.
        """
        key = tuple(sorted(node_ids))
        if self._rows is not None and self._node_key == key:
            age = max(0.0, time.monotonic() - self._at)
            if age > self._ttl_s and not self._refreshing:
                # No ``await`` between this check and the flag set below, so
                # two concurrent renders can't both spawn a refresh.
                self._refreshing = True
                self._refresh_task = asyncio.create_task(
                    self._refresh(cfg, key), name="image-snapshot-refresh",
                )
            return self._rows, age, self._refreshing
        # Cold cache or node set changed → block on one fan-out, single-flight.
        async with self._cold_lock:
            if self._rows is not None and self._node_key == key:
                return (
                    self._rows,
                    max(0.0, time.monotonic() - self._at),
                    self._refreshing,
                )
            rows = await _fetch_image_rows_for_nodes(cfg, list(key))
            self._rows, self._node_key, self._at = rows, key, time.monotonic()
            return rows, 0.0, False

    async def _refresh(
        self, cfg: AdminServerConfig, key: tuple[str, ...],
    ) -> None:
        try:
            rows = await _fetch_image_rows_for_nodes(cfg, list(key))
            self._rows, self._node_key, self._at = rows, key, time.monotonic()
        except Exception:
            LOGGER.exception("image snapshot background refresh failed")
        finally:
            self._refreshing = False


def _empty_page(*, page: int, page_size: int) -> dict[str, Any]:
    return {
        "rows": [],
        "total": 0,
        "page": page,
        "page_size": page_size,
        "page_count": 1,
        "has_prev": page > 1,
        "has_next": False,
        "first_index": 0,
        "last_index": 0,
        "prev_page_url": "",
        "next_page_url": "",
    }


def _page_items(
    rows: list[dict[str, Any]], *, page: int, page_size: int,
) -> dict[str, Any]:
    total = len(rows)
    page_count = max(1, (total + page_size - 1) // page_size)
    start = (page - 1) * page_size
    end = start + page_size
    visible = rows[start:end]
    return {
        "rows": visible,
        "total": total,
        "page": page,
        "page_size": page_size,
        "page_count": page_count,
        "has_prev": page > 1,
        "has_next": end < total,
        "first_index": start + 1 if visible else 0,
        "last_index": start + len(visible) if visible else 0,
        "prev_page_url": "",
        "next_page_url": "",
    }


def _attach_image_page_urls(request: Request, page_data: dict[str, Any]) -> None:
    page_data["prev_page_url"] = str(
        request.url.include_query_params(
            page=max(1, int(page_data["page"]) - 1),
            page_size=page_data["page_size"],
        )
    )
    page_data["next_page_url"] = str(
        request.url.include_query_params(
            page=int(page_data["page"]) + 1,
            page_size=page_data["page_size"],
        )
    )


def _shape_image_node_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    shaped: list[dict[str, Any]] = []
    for row in rows:
        cold = row["histogram"].get("cold", {"count": 0, "bytes": 0})
        pinned_tier = row["histogram"].get("pinned", {"count": 0, "bytes": 0})
        pressure, label, rank = _node_image_pressure(row)
        shaped.append({
            "node_id": row["node_id"],
            "reachable": row["reachable"],
            "error": row["error"],
            "free_disk_bytes": row["free_disk_bytes"],
            "disk_total_bytes": row.get("disk_total_bytes", 0),
            "disk_free_bytes": row.get("disk_free_bytes", row["free_disk_bytes"]),
            "disk_used_bytes": max(
                0, row.get("disk_total_bytes", 0)
                - row.get("disk_free_bytes", row["free_disk_bytes"]),
            ),
            "histogram": row["histogram"],
            "pinned": row["pinned"],
            "pinned_count": len(row["pinned"]),
            "images": row["images"],
            "total_count": row["total_count"],
            "total_bytes": row["total_bytes"],
            "cold_bytes": cold["bytes"],
            "pinned_tier_bytes": pinned_tier["bytes"],
            "pressure": pressure,
            "pressure_label": label,
            "pressure_rank": rank,
            "detail_url": f"/images/nodes/{quote(row['node_id'], safe='')}",
        })
    return shaped


def _shape_single_node_image_rows(row: dict[str, Any]) -> list[dict[str, Any]]:
    if not row.get("reachable"):
        return []
    shaped: list[dict[str, Any]] = []
    for img in row.get("images", []):
        shaped.append({
            "name": img["name"],
            "tier": img["tier"],
            "tier_rank": _tier_rank(img["tier"]),
            "size_bytes": img["size_bytes"],
            "in_use_count": img["in_use_count"],
            "pinned": img["pinned"],
            "last_used_at": img["last_used_at"],
            "owner": img.get("owner", "external"),
            "detail_url": f"/images/image?ref={quote(img['name'], safe='')}",
        })
    return shaped


def _ownership_breakdown(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Count distinct image refs per ownership class across the cluster.

    The admin /images/catalog dropdown is label-driven, so operators
    whose images were built before commit ``12237ea`` (which added the
    ``org.xrlenv.owned=true`` build labels) see every image classified
    as ``external``. The breakdown surfaces that mismatch directly so
    "the filter does nothing" turns into "your images aren't labeled —
    rebuild" rather than a silent UX cliff.

    Distinct-image counts (not per-node-instance) so re-replication
    doesn't inflate the chip.
    """
    seen: dict[str, str] = {}
    for row in rows:
        if not row.get("reachable"):
            continue
        for img in row.get("images", []):
            name = img.get("name")
            owner = img.get("owner") or "external"
            if not isinstance(name, str):
                continue
            # First-seen wins; an image is classified once.
            seen.setdefault(name, owner)
    counts = {"xrlenv_final": 0, "xrlenv_intermediate": 0, "external": 0}
    for owner in seen.values():
        counts[owner] = counts.get(owner, 0) + 1
    return counts


def _image_passes_ownership(
    img: dict[str, Any], *, hide_intermediate: bool, xrlenv_only: bool,
) -> bool:
    """Catalog-side counterpart to :func:`_filter_image_rows_by_ownership`.

    The per-node detail page applies its filter on already-shaped
    rows; the catalog aggregator works pre-shape (over per-node
    images) so the distinct-image counts reflect the include
    filter exactly.
    """
    owner = img.get("owner")
    if hide_intermediate and owner == "xrlenv_intermediate":
        return False
    return not (xrlenv_only and owner == "external")


def _shape_distinct_image_rows(
    rows: list[dict[str, Any]],
    *,
    ownership_filter: Callable[[dict[str, Any]], bool] | None = None,
) -> list[dict[str, Any]]:
    by_name: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not row.get("reachable"):
            continue
        for img in row.get("images", []):
            if ownership_filter is not None and not ownership_filter(img):
                continue
            name = img["name"]
            entry = by_name.setdefault(
                name,
                {
                    "name": name,
                    "node_ids": set(),
                    "node_count": 0,
                    "instance_count": 0,
                    "total_bytes": 0,
                    "max_size_bytes": 0,
                    "best_tier": img["tier"],
                    "tier_rank": _tier_rank(img["tier"]),
                    "tier_counts": {tier: 0 for tier in _IMAGE_TIERS},
                    "in_use_count": 0,
                    "pinned_count": 0,
                    "last_used_at": None,
                    "detail_url": f"/images/image?ref={quote(name, safe='')}",
                },
            )
            entry["node_ids"].add(row["node_id"])
            entry["instance_count"] += 1
            entry["total_bytes"] += img["size_bytes"]
            entry["max_size_bytes"] = max(entry["max_size_bytes"], img["size_bytes"])
            img_rank = _tier_rank(img["tier"])
            if img_rank > entry["tier_rank"]:
                entry["best_tier"] = img["tier"]
                entry["tier_rank"] = img_rank
            entry["tier_counts"][img["tier"]] = entry["tier_counts"].get(img["tier"], 0) + 1
            entry["in_use_count"] += img["in_use_count"]
            if img["pinned"]:
                entry["pinned_count"] += 1
            if img["last_used_at"] is not None:
                prior = entry["last_used_at"]
                entry["last_used_at"] = (
                    img["last_used_at"] if prior is None else max(prior, img["last_used_at"])
                )
    out: list[dict[str, Any]] = []
    for entry in by_name.values():
        entry["node_ids"] = sorted(entry["node_ids"])
        entry["node_count"] = len(entry["node_ids"])
        out.append(entry)
    return out


def _filter_image_node_rows(
    rows: list[dict[str, Any]],
    *,
    q: str | None,
    tier: str | None,
    pressure: str | None,
    pinned: str | None,
) -> list[dict[str, Any]]:
    needle = q.strip().lower() if q else ""
    out = rows
    if needle:
        out = [row for row in out if needle in row["node_id"].lower()]
    if tier:
        out = [
            row for row in out
            if row["histogram"].get(tier, {"count": 0})["count"] > 0
        ]
    if pressure:
        out = [row for row in out if row["pressure"] == pressure]
    if pinned == "yes":
        out = [row for row in out if row["pinned_count"] > 0]
    elif pinned == "no":
        out = [row for row in out if row["pinned_count"] == 0]
    return out


def _filter_distinct_image_rows(
    rows: list[dict[str, Any]],
    *,
    q: str | None,
    tier: str | None,
    pinned: str | None,
) -> list[dict[str, Any]]:
    needle = q.strip().lower() if q else ""
    out = rows
    if needle:
        out = [row for row in out if needle in row["name"].lower()]
    if tier:
        out = [row for row in out if row["tier_counts"].get(tier, 0) > 0]
    if pinned == "yes":
        out = [row for row in out if row["pinned_count"] > 0]
    elif pinned == "no":
        out = [row for row in out if row["pinned_count"] == 0]
    return out


def _filter_single_node_image_rows(
    rows: list[dict[str, Any]],
    *,
    q: str | None,
    tier: str | None,
    pinned: str | None,
) -> list[dict[str, Any]]:
    needle = q.strip().lower() if q else ""
    out = rows
    if needle:
        out = [row for row in out if needle in row["name"].lower()]
    if tier:
        out = [row for row in out if row["tier"] == tier]
    if pinned == "yes":
        out = [row for row in out if row["pinned"]]
    elif pinned == "no":
        out = [row for row in out if not row["pinned"]]
    return out


def _sort_image_node_rows(
    rows: list[dict[str, Any]], sort: str | None,
) -> list[dict[str, Any]]:
    key = sort or "pressure"
    if key not in {"pressure", "node", "free", "size", "images", "cold", "pinned"}:
        raise HTTPException(
            status_code=422,
            detail="sort must be pressure, node, free, size, images, cold, or pinned",
        )
    if key == "node":
        return sorted(rows, key=lambda row: row["node_id"])
    if key == "free":
        return sorted(rows, key=lambda row: (row["free_disk_bytes"], row["node_id"]))
    if key == "size":
        return sorted(rows, key=lambda row: (-row["total_bytes"], row["node_id"]))
    if key == "images":
        return sorted(rows, key=lambda row: (-row["total_count"], row["node_id"]))
    if key == "cold":
        return sorted(rows, key=lambda row: (-row["cold_bytes"], row["node_id"]))
    if key == "pinned":
        return sorted(rows, key=lambda row: (-row["pinned_count"], row["node_id"]))
    return sorted(
        rows,
        key=lambda row: (
            -row["pressure_rank"],
            row["free_disk_bytes"] if row["reachable"] else 0,
            -row["total_bytes"],
            row["node_id"],
        ),
    )


def _sort_distinct_image_rows(
    rows: list[dict[str, Any]], sort: str | None,
) -> list[dict[str, Any]]:
    key = sort or "operational"
    allowed = {"operational", "coverage", "size", "name", "tier", "pinned"}
    if key not in allowed:
        raise HTTPException(
            status_code=422,
            detail=f"sort must be one of {', '.join(sorted(allowed))}",
        )
    if key == "name":
        return sorted(rows, key=lambda row: row["name"])
    if key == "coverage":
        return sorted(rows, key=lambda row: (-row["node_count"], -row["total_bytes"], row["name"]))
    if key == "size":
        return sorted(rows, key=lambda row: (-row["total_bytes"], row["name"]))
    if key == "tier":
        return sorted(rows, key=lambda row: (-row["tier_rank"], row["name"]))
    if key == "pinned":
        return sorted(rows, key=lambda row: (-row["pinned_count"], -row["total_bytes"], row["name"]))
    return sorted(
        rows,
        key=lambda row: (
            -row["tier_rank"],
            -row["in_use_count"],
            -row["pinned_count"],
            -row["total_bytes"],
            row["name"],
        ),
    )


def _sort_single_node_image_rows(
    rows: list[dict[str, Any]], sort: str | None,
) -> list[dict[str, Any]]:
    key = sort or "operational"
    allowed = {"operational", "size", "name", "tier", "pinned", "in_use"}
    if key not in allowed:
        raise HTTPException(
            status_code=422,
            detail=f"sort must be one of {', '.join(sorted(allowed))}",
        )
    if key == "name":
        return sorted(rows, key=lambda row: row["name"])
    if key == "size":
        return sorted(rows, key=lambda row: (-row["size_bytes"], row["name"]))
    if key == "tier":
        return sorted(rows, key=lambda row: (-row["tier_rank"], row["name"]))
    if key == "pinned":
        return sorted(rows, key=lambda row: (not row["pinned"], row["name"]))
    if key == "in_use":
        return sorted(rows, key=lambda row: (-row["in_use_count"], row["name"]))
    return sorted(
        rows,
        key=lambda row: (
            -row["tier_rank"],
            -row["in_use_count"],
            not row["pinned"],
            -row["size_bytes"],
            row["name"],
        ),
    )


def _image_summary(
    rows: list[dict[str, Any]],
    node_rows: list[dict[str, Any]],
    image_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    reachable = [row for row in rows if row.get("reachable")]
    pinned_instances = 0
    pinned_bytes = 0
    for row in reachable:
        for img in row.get("images", []):
            if img.get("pinned"):
                pinned_instances += 1
                pinned_bytes += int(img.get("size_bytes", 0))
    lowest = min(
        (row for row in node_rows if row.get("reachable")),
        key=lambda row: row["free_disk_bytes"],
        default=None,
    )
    cold_bytes = sum(row["cold_bytes"] for row in node_rows if row.get("reachable"))
    total_bytes = sum(row["total_bytes"] for row in rows if row.get("reachable"))
    total_instances = sum(row["total_count"] for row in rows if row.get("reachable"))
    return {
        "connected_nodes": len(rows),
        "reachable_nodes": len(reachable),
        "unreachable_nodes": len(rows) - len(reachable),
        "total_instances": total_instances,
        "distinct_images": len(image_rows),
        "total_bytes": total_bytes,
        "cold_bytes": cold_bytes,
        "pinned_instances": pinned_instances,
        "pinned_bytes": pinned_bytes,
        "lowest_free_disk_node": lowest,
    }


def _empty_image_summary() -> dict[str, Any]:
    return {
        "connected_nodes": 0,
        "reachable_nodes": 0,
        "unreachable_nodes": 0,
        "total_instances": 0,
        "distinct_images": 0,
        "total_bytes": 0,
        "cold_bytes": 0,
        "pinned_instances": 0,
        "pinned_bytes": 0,
        "lowest_free_disk_node": None,
    }


def _image_risks(
    summary: dict[str, Any], node_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    risks: list[dict[str, str]] = []
    if summary["unreachable_nodes"]:
        risks.append({
            "level": "bad",
            "text": f"{summary['unreachable_nodes']} connected node(s) did not report images",
        })
    low_free = [
        row for row in node_rows
        if row["pressure"] in {"critical", "watch"} and row.get("reachable")
    ]
    if low_free:
        preview = ", ".join(row["node_id"] for row in low_free[:3])
        suffix = "..." if len(low_free) > 3 else ""
        risks.append({
            "level": "warn" if all(row["pressure"] == "watch" for row in low_free) else "bad",
            "text": f"{len(low_free)} node(s) low on free disk: {preview}{suffix}",
        })
    if summary["total_bytes"] and summary["cold_bytes"] / summary["total_bytes"] >= 0.5:
        risks.append({
            "level": "warn",
            "text": "Cold images account for at least half of cached bytes",
        })
    if (
        summary["pinned_bytes"]
        and summary["total_bytes"]
        and summary["pinned_bytes"] / summary["total_bytes"] >= 0.5
    ):
        risks.append({
            "level": "warn",
            "text": "Pinned images account for at least half of cached bytes",
        })
    return risks


def _node_image_pressure(row: dict[str, Any]) -> tuple[str, str, int]:
    if not row.get("reachable"):
        return "unreachable", "unreachable", 4
    free = int(row.get("free_disk_bytes", 0))
    if free < _IMAGE_FREE_DISK_CRITICAL_BYTES:
        return "critical", "critical free disk", 3
    if free < _IMAGE_FREE_DISK_WARN_BYTES:
        return "watch", "watch free disk", 2
    return "ok", "ok", 1


def _tier_rank(tier: str) -> int:
    return {"in_use": 3, "pinned": 2, "recently_used": 1, "cold": 0}.get(tier, -1)


def _list_known_node_ids(cfg: AdminServerConfig) -> list[str]:
    """Currently-connected node IDs only.

    The /images panel fans out a live ``ReportImagesCommand`` per node;
    rostered-but-never-connected entries from ``nodes.yaml`` and
    historical ``status='lost'`` rows from the state store have no
    live transport, so including them just produces a row of
    "no live transport for this node" noise (operator-reported,
    2026-05-04). Filter to ``status='connected'`` and drop the
    nodes.yaml union — the absent/lost picture lives on the /nodes
    view, not here.
    """
    if not cfg.state_db.exists():
        return []
    store = SqliteStateStore(cfg.state_db, read_only=True)
    try:
        return sorted(n.node_id for n in store.list_nodes(status="connected"))
    finally:
        store.close()


# Live-state ``ImageTier`` vocabulary (xrlenv.node.image_cache.ImageTier).
# Documented in docs/deployment/images.md "Tier vocabulary". Note: the
# orthogonal ``EvictionTier`` (final/stub_runtime/base) is a *build-cost*
# classification used by the per-node eviction policy and never appears
# in the report stream — pre-2026-05 versions of this view conflated
# the two and rendered three permanently-empty rows.
_IMAGE_TIERS: tuple[str, ...] = ("in_use", "pinned", "recently_used", "cold")


async def _fetch_image_row_for(
    cfg: AdminServerConfig, node_id: str,
) -> dict[str, Any]:
    """Fetch + summarise one node's image cache snapshot.

    Each row carries: node id, transport-reachable flag, free-disk
    bytes, per-tier histogram (count + bytes), pinned set, and the
    raw per-image entries for the drill-down table.
    """
    transport = cfg.node_lookup(node_id) if cfg.node_lookup else None
    if transport is None:
        return _unreachable_image_row(node_id, "no live transport for this node")
    # Fresh disk capacity for the used/total columns. ``disk_state`` is the
    # heartbeat-cached ``(free, total)`` (the same source /capacity uses, ~5 s
    # fresh) — used here so the page can show ``used + free = total`` and the
    # operator can see that "images size" (a per-image sum that double-counts
    # shared layers) is NOT the disk used. statvfs-derived; no daemon call.
    disk_free = 0
    disk_total = 0
    disk_state = getattr(transport, "disk_state", None)
    if callable(disk_state):
        try:
            disk_free, disk_total = (int(x) for x in disk_state())
        except Exception:
            disk_free, disk_total = 0, 0
    try:
        report = await transport.report_images()
    except Exception as exc:
        return _unreachable_image_row(node_id, f"report_images failed: {exc}")
    histogram: dict[str, dict[str, int]] = {
        tier: {"count": 0, "bytes": 0} for tier in _IMAGE_TIERS
    }
    images = []
    total_bytes = 0
    for img in report.images:
        images.append(
            {
                "name": img.name,
                "tier": img.tier,
                "size_bytes": img.size_bytes,
                "in_use_count": img.in_use_count,
                "last_used_at": img.last_used_at,
                "pinned": img.pinned,
                "owner": img.owner,
            }
        )
        bucket = histogram.setdefault(img.tier, {"count": 0, "bytes": 0})
        bucket["count"] += 1
        bucket["bytes"] += img.size_bytes
        total_bytes += img.size_bytes
    images.sort(
        key=lambda img: (
            -_tier_rank(img["tier"]),
            -int(img["in_use_count"]),
            not bool(img["pinned"]),
            -int(img["size_bytes"]),
            img["name"],
        )
    )
    total_count = len(report.images)
    return {
        "node_id": node_id,
        "reachable": True,
        "error": None,
        "free_disk_bytes": report.free_disk_bytes,
        # Heartbeat disk capacity (0 when the node hasn't reported one yet);
        # ``disk_free_bytes`` prefers the fresh heartbeat free, falling back
        # to the report's statvfs free so used = total - free reconciles.
        "disk_total_bytes": disk_total,
        "disk_free_bytes": disk_free or report.free_disk_bytes,
        "histogram": histogram,
        "pinned": list(report.pinned),
        "images": images,
        "total_count": total_count,
        "total_bytes": total_bytes,
    }


def _unreachable_image_row(node_id: str, error: str) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "reachable": False,
        "error": error,
        **_empty_image_row_fields(),
    }


def _empty_image_row_fields() -> dict[str, Any]:
    return {
        "free_disk_bytes": 0,
        "disk_total_bytes": 0,
        "disk_free_bytes": 0,
        "histogram": {tier: {"count": 0, "bytes": 0} for tier in _IMAGE_TIERS},
        "pinned": [],
        "images": [],
        "total_count": 0,
        "total_bytes": 0,
    }


def _empty_image_totals() -> dict[str, dict[str, int]]:
    return {tier: {"count": 0, "bytes": 0} for tier in _IMAGE_TIERS}


def _aggregate_image_totals(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    """Two complementary cluster-wide rollups.

    ``totals_summed`` — every node·image instance counted. The same
    8 GB image present on five nodes contributes ``5`` to ``count``
    and ``40 GB`` to ``bytes``. Right answer for "how much disk is
    images burning across the cluster?" and "where's the eviction
    pressure?".

    ``totals_distinct`` — deduplicated by image name. Same image on
    five nodes contributes ``1`` to ``count`` and one image's bytes
    (the max observed across nodes — they should all agree, but if
    they don't we'd rather not undercount). Right answer for "what
    tasks does the cluster cover?". When an image's tier disagrees
    across nodes (one node has it ``in_use`` while another has it
    ``cold``), the distinct view records it under the *highest*
    tier observed, ranked
    ``in_use > pinned > recently_used > cold`` — the operator
    typically cares about "is anyone using this image right now?",
    not "is it inert somewhere?".
    """
    summed = _empty_image_totals()
    # Track distinct images: image name → (best-tier seen, max size_bytes).
    tier_rank = {"in_use": 3, "pinned": 2, "recently_used": 1, "cold": 0}
    seen: dict[str, tuple[str, int]] = {}
    for row in rows:
        if not row.get("reachable"):
            continue
        for tier, counts in row.get("histogram", {}).items():
            bucket = summed.setdefault(tier, {"count": 0, "bytes": 0})
            bucket["count"] += counts.get("count", 0)
            bucket["bytes"] += counts.get("bytes", 0)
        for img in row.get("images", []):
            name = img["name"]
            tier = img["tier"]
            size = int(img.get("size_bytes", 0))
            prior = seen.get(name)
            if prior is None or tier_rank.get(tier, -1) > tier_rank.get(prior[0], -1):
                seen[name] = (tier, max(size, prior[1] if prior else 0))
            elif size > prior[1]:
                seen[name] = (prior[0], size)
    distinct = _empty_image_totals()
    for tier, size in seen.values():
        bucket = distinct.setdefault(tier, {"count": 0, "bytes": 0})
        bucket["count"] += 1
        bucket["bytes"] += size
    return summed, distinct


async def _gather_capacity(cfg: AdminServerConfig) -> dict[str, Any]:
    return await asyncio.to_thread(_capacity_blocking, cfg)


def _summarize_plan_json(plan_json: str) -> dict[str, Any]:
    """Extract the operator-readable bits from a stored plan JSON.

    Returns ``{"benchmarks": [...], "summary": "<one-line>"}`` so the
    /builds page can show "what was actually being built" instead of
    only the opaque sha256 plan_id. Best-effort: malformed plans
    surface as an empty list + literal "unknown" summary so the row
    still renders.
    """
    try:
        raw = json.loads(plan_json)
    except (ValueError, TypeError):
        return {"benchmarks": [], "summary": "(unreadable plan)"}
    benchmarks = []
    summary_parts: list[str] = []
    for b in raw.get("benchmarks") or []:
        if not isinstance(b, dict):
            continue
        name = str(b.get("name", "?"))
        sel = b.get("selection") or {}
        if isinstance(sel, dict):
            if sel.get("smoke"):
                sel_label = "--smoke"
            elif sel.get("all"):
                sel_label = "--all"
            elif sel.get("instances"):
                instances = sel.get("instances") or []
                count = len(instances) if isinstance(instances, list) else 0
                sel_label = f"--instances ({count})"
            else:
                sel_label = "?"
        else:
            sel_label = "?"
        build_path = b.get("build_path") or ""
        benchmarks.append({
            "name": name,
            "selection_label": sel_label,
            "build_path": build_path,
        })
        summary_parts.append(f"{name} {sel_label}")
    return {
        "benchmarks": benchmarks,
        "summary": " + ".join(summary_parts) if summary_parts else "(no benchmarks)",
    }


def _format_relative_time(applied_at: float, *, now: float) -> str:
    """Render ``applied_at`` as ``"<N units> ago"`` (e.g. ``"5 min ago"``).

    Mirrors the admin's other relative-time renders so the operator
    doesn't have to convert raw Unix epochs in their head."""
    delta = now - applied_at
    if delta < 60:
        return f"{int(delta)}s ago"
    if delta < 3600:
        return f"{int(delta / 60)} min ago"
    if delta < 86400:
        return f"{int(delta / 3600)}h ago"
    return f"{int(delta / 86400)}d ago"


def _format_absolute_time(applied_at: float) -> str:
    """``YYYY-MM-DD HH:MM`` in local time."""
    import datetime as _dt

    return _dt.datetime.fromtimestamp(applied_at).strftime("%Y-%m-%d %H:%M")


def _gather_builds(
    cfg: AdminServerConfig,
    *,
    page: int = 1,
    page_size: int = 32,
    status: str | None = None,
) -> dict[str, Any]:
    """Build-plans list view (paginated, mirrors /rollouts shape).

    Each row carries the operator-readable summary (which benchmark
    + which selection) extracted from the persisted plan_json so the
    page is self-explanatory without click-through. Detail page at
    /builds/<plan_id> renders the full per-assignment breakdown.
    """
    if not cfg.state_db.exists():
        return {
            "plans": [], "state_db_missing": True,
            "page": page, "page_size": page_size, "total": 0,
            "has_prev": False, "has_next": False, "first_index": 0,
            "last_index": 0, "status_filter": status,
            "page_size_options": [16, 32, 64, 128],
            "status_options": [
                "in_flight", "completed", "partial_failure",
                "cancelled", "superseded",
            ],
        }
    from xrlenv.control.state import SqliteStateStore

    now = time.time()
    state = SqliteStateStore(cfg.state_db, read_only=True)
    try:
        if status:
            plans_raw = state.list_build_plans(status=cast(Any, status))
        else:
            plans_raw = state.list_build_plans()
        total = len(plans_raw)
        start = max(0, (page - 1) * page_size)
        end = start + page_size
        plans: list[dict[str, Any]] = []
        for p in plans_raw[start:end]:
            rows = state.list_assignments(p.plan_id)
            per_status: dict[str, int] = {}
            for r in rows:
                per_status[r.status] = per_status.get(r.status, 0) + 1
            done = per_status.get("done", 0)
            failed = per_status.get("failed", 0)
            building = per_status.get("building", 0)
            pending = per_status.get("pending", 0)
            registered = per_status.get("registered", 0)
            evicted = per_status.get("evicted", 0)
            summary = _summarize_plan_json(p.plan_json)
            plans.append({
                "plan_id": p.plan_id,
                "plan_id_short": p.plan_id[:12],
                "status": p.status,
                "applied_at": p.applied_at,
                "applied_at_abs": _format_absolute_time(p.applied_at),
                "applied_at_rel": _format_relative_time(p.applied_at, now=now),
                "applied_by": p.applied_by,
                "assignment_count": len(rows),
                "done": done,
                "failed": failed,
                "building": building,
                "pending": pending,
                "registered": registered,
                "evicted": evicted,
                "summary": summary["summary"] or "(unparseable plan)",
                "benchmarks": summary["benchmarks"],
                "name": p.name or "(unnamed)",
                "detail_url": f"/builds/{p.plan_id}",
            })
    finally:
        state.close()
    has_prev = page > 1
    has_next = end < total
    return {
        "plans": plans,
        "state_db_missing": False,
        "page": page,
        "page_size": page_size,
        "total": total,
        "has_prev": has_prev,
        "has_next": has_next,
        "first_index": start + 1 if plans else 0,
        "last_index": start + len(plans),
        "status_filter": status,
        "page_size_options": [16, 32, 64, 128],
        "status_options": [
            "in_flight", "completed", "partial_failure",
            "cancelled", "superseded",
        ],
    }


def _gather_build_detail(
    cfg: AdminServerConfig, plan_id: str,
) -> dict[str, Any] | None:
    """Detail view at /builds/<plan_id>: full plan summary + per-
    assignment table.

    Distinct from :func:`_gather_build_plan` (the JSON-API shape) —
    that one returns a flat structure for the CLI poller; this adds
    the parsed-plan summary + per-node bucketing the HTML page wants.
    """
    if not cfg.state_db.exists():
        return None
    from xrlenv.control.state import SqliteStateStore

    state = SqliteStateStore(cfg.state_db, read_only=True)
    try:
        plan = state.get_build_plan(plan_id)
        if plan is None:
            return None
        rows = state.list_assignments(plan_id)
    finally:
        state.close()

    per_status: dict[str, int] = {}
    for r in rows:
        per_status[r.status] = per_status.get(r.status, 0) + 1
    summary = _summarize_plan_json(plan.plan_json)

    # Bucket assignments by node so the page can render one section
    # per node — matches the "where am I building" mental model.
    by_node: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_node.setdefault(r.node_id, []).append({
            "image_ref": r.image_ref,
            "benchmark": r.benchmark,
            "status": r.status,
            "started_at": r.started_at,
            "completed_at": r.completed_at,
            "error": r.error,
        })
    nodes = [
        {
            "node_id": nid,
            "rows": sorted(rows_, key=lambda x: x["image_ref"]),
            "row_count": len(rows_),
            "done": sum(1 for x in rows_ if x["status"] == "done"),
            "failed": sum(1 for x in rows_ if x["status"] == "failed"),
            "registered": sum(1 for x in rows_ if x["status"] == "registered"),
            "evicted": sum(1 for x in rows_ if x["status"] == "evicted"),
        }
        for nid, rows_ in sorted(by_node.items())
    ]

    return {
        "plan_id": plan.plan_id,
        "plan_id_short": plan.plan_id[:12],
        "status": plan.status,
        "applied_at": plan.applied_at,
        "applied_at_abs": _format_absolute_time(plan.applied_at),
        "applied_at_rel": _format_relative_time(
            plan.applied_at, now=time.time(),
        ),
        "applied_by": plan.applied_by,
        "name": plan.name or "(unnamed)",
        "plan_json_pretty": _format_plan_json(plan.plan_json),
        "summary": summary["summary"] or "(unparseable plan)",
        "benchmarks": summary["benchmarks"],
        "assignment_count": len(rows),
        "per_status": per_status,
        "nodes": nodes,
    }


def _format_plan_json(plan_json: str) -> str:
    """Pretty-print the stored canonical plan for the detail page's
    ``<details>`` block. Best-effort; falls back to the raw string."""
    try:
        return json.dumps(json.loads(plan_json), indent=2, sort_keys=True)
    except (ValueError, TypeError):
        return plan_json


def _get_persisted_plan(cfg: AdminServerConfig, plan_id: str) -> Any:
    """Cheap lookup used by ``POST /api/build/apply`` to short-circuit
    on already-completed plans before spawning the background task."""
    if not cfg.state_db.exists():
        return None
    from xrlenv.control.state import SqliteStateStore

    state = SqliteStateStore(cfg.state_db, read_only=True)
    try:
        return state.get_build_plan(plan_id)
    finally:
        state.close()


def _resolve_plan_for_admin(
    cfg: AdminServerConfig, plan_id_or_prefix: str,
) -> Any:
    """Resolve a plan_id (full or unique prefix, ≥4 chars) to a
    BuildPlanRecord. Mirrors the CLI's ``_resolve_plan_id`` so the
    admin /api/build/cancel endpoint accepts the same 12-char short id
    the admin /builds panel surfaces. Returns ``None`` when no plan
    matches; raises ``HTTPException(409)`` when the prefix is
    ambiguous (the operator must paste more chars).

    The "≥4 char prefix" guard is enforced before consulting state.db
    so a 2-char typo is rejected with a clear UX message even on a
    fresh control plane (no state.db yet) — matches the CLI helper's
    behavior.
    """
    # Fast syntactic guard: under-4-char inputs are rejected up front
    # (the operator may have just typed too few chars). Full SHA-256
    # plan_ids are 64 chars and the admin /builds short id is 12,
    # so any legitimate input clears the bar.
    if len(plan_id_or_prefix) < 4:
        raise HTTPException(
            status_code=400,
            detail=(
                "plan_id prefix must be at least 4 chars "
                "(use the full id or the 12-char short id from "
                "the /builds page)"
            ),
        )
    if not cfg.state_db.exists():
        return None
    from xrlenv.control.state import SqliteStateStore

    state = SqliteStateStore(cfg.state_db, read_only=True)
    try:
        # Exact match first — defensive, so a full plan_id that is
        # ALSO a prefix of other plans never errors with "ambiguous."
        exact = state.get_build_plan(plan_id_or_prefix)
        if exact is not None:
            return exact
        candidates = [
            p for p in state.list_build_plans()
            if p.plan_id.startswith(plan_id_or_prefix)
        ]
        if not candidates:
            return None
        if len(candidates) > 1:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"plan_id prefix {plan_id_or_prefix!r} matches "
                    f"{len(candidates)} plans; add more chars"
                ),
            )
        return candidates[0]
    finally:
        state.close()


def _flip_existing_plan_to_in_flight(
    cfg: AdminServerConfig, plan_id: str,
) -> None:
    """Synchronously flip an existing plan row to ``in_flight`` +
    purge its stale assignments BEFORE returning 202 from
    ``/api/build/apply``. Without this, the CLI poller's first GET
    sees the prior terminal status (e.g. ``partial_failure`` from a
    crashed run, ``cancelled`` from an operator cancel) and exits
    before the new background apply has had time to do its own flip.

    Called only when the admin endpoint has already determined this
    is NOT a real concurrent apply (i.e. ``plan_id not in
    _build_tasks``), so it's safe to purge unconditionally. The
    matching ``bypass_in_flight_check=True`` on ``coordinator.apply``
    is what prevents the coordinator from then re-rejecting on the
    fresh ``in_flight`` row this function just wrote — 2026-05-12
    bug regression: an orphan ``in_flight`` (from a prior crashed
    process) OR a ``cancelled`` plan would otherwise hang the
    re-apply forever (coordinator rejects → admin _run returns
    without raising → plan stays ``in_flight`` with zero rows →
    poller polls indefinitely).
    """
    from xrlenv.control.state import SqliteStateStore

    state = SqliteStateStore(cfg.state_db)
    try:
        existing = state.get_build_plan(plan_id)
        if existing is None:
            return
        state.update_build_plan_status(plan_id, "in_flight")
        state.delete_assignments(plan_id)
    finally:
        state.close()


def _persist_failed_plan(
    cfg: AdminServerConfig, plan_id: str, plan: Any,
    applied_by: str, error_msg: str,
) -> None:
    """Record a ``partial_failure`` plan row for an apply that raised
    before reaching ``record_build_plan``. Without this, the
    /api/build/plans/<plan_id> endpoint returns 404 indefinitely and
    operators have no in-cluster trace of the failure (only the
    server log knows). The error message itself goes to the log;
    the persisted row carries enough for the admin /builds panel
    and the CLI poller to converge on a terminal status.

    ``error_msg`` is logged but not stored on the plan record (the
    schema doesn't carry a plan-level error field today). Operators
    inspect the cluster log to recover the underlying exception.
    """
    LOGGER.error(
        "build apply error to persist for plan_id=%s: %s",
        plan_id, error_msg,
    )
    from xrlenv.control.state import SqliteStateStore

    state = SqliteStateStore(cfg.state_db)
    try:
        plan_json = plan.model_dump_json(exclude_none=True)
        state.record_build_plan(
            plan_id=plan_id, applied_by=applied_by, plan_json=plan_json,
            name=getattr(plan, "name", None),
        )
        state.update_build_plan_status(plan_id, "partial_failure")
    finally:
        state.close()


def _gather_build_plan(
    cfg: AdminServerConfig, plan_id: str,
) -> dict[str, Any] | None:
    """JSON shape ``GET /api/build/plans/<plan_id>`` returns to the CLI."""
    if not cfg.state_db.exists():
        return None
    from xrlenv.control.state import SqliteStateStore

    state = SqliteStateStore(cfg.state_db, read_only=True)
    try:
        plan = state.get_build_plan(plan_id)
        if plan is None:
            return None
        rows = state.list_assignments(plan_id)
    finally:
        state.close()

    per_status: dict[str, int] = {}
    for r in rows:
        per_status[r.status] = per_status.get(r.status, 0) + 1

    return {
        "plan_id": plan.plan_id,
        "status": plan.status,
        "applied_at": plan.applied_at,
        "applied_by": plan.applied_by,
        "name": plan.name,
        "assignment_count": len(rows),
        "per_status": per_status,
        "assignments": [
            {
                "node_id": r.node_id,
                "image_ref": r.image_ref,
                "benchmark": r.benchmark,
                "status": r.status,
                "started_at": r.started_at,
                "completed_at": r.completed_at,
                "error": r.error,
            }
            for r in rows
        ],
    }


# Raw-rollout statuses that hold capacity. Everything else is terminal
# (``released`` / ``cancelled`` / ``failed`` / ``reaped`` /
# ``capacity_rejected``) and has given its footprint back.
_LIVE_RAW_STATUSES: tuple[RawRolloutStatus, ...] = ("acquiring", "running")

# Cap on how many distinct workload profiles the matrix renders. Profiles are
# ranked by live container count, so the ones actually filling the cluster are
# the ones shown; the tail is summarised in ``profiles_omitted``.
_MAX_CAPACITY_PROFILES = 12


def _node_hardware_live(cfg: AdminServerConfig, node_id: str) -> Any:
    """Real :class:`HardwareInfo` for ``node_id``, or ``None``.

    Reads it off the live node transport the same way the scheduler does
    (``node_lookup`` → ``NodeTransport.hardware()``), so the matrix is
    computed from the *same* numbers placement uses — including the
    heartbeat-reconciled ``disk_bytes``. Returns ``None`` when the admin
    runs without cluster reachability (``node_lookup`` unwired) or the node
    is disconnected; the caller renders "unknown" rather than inventing a
    profile.
    """
    lookup = cfg.node_lookup
    if lookup is None:
        return None
    try:
        transport = lookup(node_id)
    except Exception:
        return None
    probe = getattr(transport, "hardware", None)
    if probe is None:
        return None
    try:
        hw = probe()
    except Exception:
        return None
    return hw if isinstance(hw, HardwareInfo) else None


def _resource_spec_from_json(raw: str | None) -> Any:
    """Parse a stored ``effective_resources_json`` into a ``ResourceSpec``.

    ``None`` (a rollout sealed before placement, e.g. ``capacity_rejected``)
    and malformed JSON both yield ``None`` — such a row held no footprint.
    """
    if not raw:
        return None
    from xrlenv.backends.base import ResourceSpec
    try:
        return ResourceSpec.model_validate_json(raw)
    except Exception:
        return None


def _profile_key(spec: Any) -> tuple[float, int, int]:
    """Group footprints by the three axes the estimator packs against."""
    return (
        float(spec.cpu_request),
        int(spec.mem_request_bytes),
        int(spec.disk_request_bytes),
    )


def _profile_label(key: tuple[float, int, int]) -> str:
    cpu, mem, disk = key
    cpu_txt = f"{cpu:g}"
    return f"{cpu_txt} cpu / {mem / 1024 ** 3:g} GiB / {disk / 1024 ** 3:g} GiB disk"


def _capacity_blocking(cfg: AdminServerConfig) -> dict[str, Any]:
    """Spec-13 capacity matrix — what the scheduler would actually admit.

    Three properties this view must have, each learned the hard way when a
    node advertising the wrong disk capped the cluster at ~28 % of its real
    capacity and this page showed nothing at all:

    1. **Real hardware.** Cells are computed from each node's live
       ``HardwareInfo`` (``node_lookup`` → the same transport the scheduler
       reads), not a fabricated profile. A node whose hardware can't be read
       is listed as ``hardware_known=False`` and gets no cells — an honest
       gap beats a plausible-looking fiction.
    2. **Raw containers are first-class.** Workload profiles are derived from
       the footprints of *live raw rollouts* as well as registered template
       manifests, so a cluster running only raw containers (case 2/3 — every
       benchmark harness) is no longer a blank page.
    3. **Live load, so ``remaining`` is real.** Each node's running raw
       containers and sandboxes are charged against its capacity, and the
       binding axis is reported per cell. ``remaining == 0`` with
       ``binding='disk:sandbox_writable'`` while cpu/mem sit idle is exactly
       the signal that was missing.

    The estimator + synthetic manifests are the ones the raw-container
    coordinator uses, so the numbers here match the placement decision rather
    than approximating it.
    """
    from xrlenv.control.capacity import NodeProfile, StaticCapacityEstimator
    from xrlenv.control.raw_container_service import _synthetic_manifest_for_raw
    from xrlenv.control.template_catalog import TemplateCatalog

    computed_at = time.time()
    rostered = _load_nodes_yaml_lazy(cfg.nodes_yaml or Path("nodes.yaml"))

    sandboxes: list[Any] = []
    live_raw: list[Any] = []
    connected: list[str] = []
    if cfg.state_db.exists():
        store = SqliteStateStore(cfg.state_db, read_only=True)
        try:
            sandboxes = store.list_sandboxes()
            for status in _LIVE_RAW_STATUSES:
                live_raw.extend(store.list_raw_rollouts(status=status))
            connected = [n.node_id for n in store.list_nodes(status="connected")]
        finally:
            store.close()

    # ── Node set: rostered, plus connected, plus whoever carries load ───────
    node_ids: list[str] = []
    for candidate in (
        [e["id"] for e in rostered if isinstance(e.get("id"), str)],
        connected,
        [sb.node_id for sb in sandboxes],
        [r.node_id for r in live_raw if r.node_id],
    ):
        for nid in candidate:
            if nid not in node_ids:
                node_ids.append(nid)
    node_ids.sort()

    # ── Live load per node, charged with the footprint it was placed on ─────
    #
    # ``running`` mirrors the estimator's ``(template_id, ResourceSpec)``
    # shape so ``fits`` / ``capacity_remaining`` see exactly what the
    # scheduler's own load accounting sees.
    load: dict[str, list[tuple[str, Any]]] = {nid: [] for nid in node_ids}
    counts: dict[str, int] = dict.fromkeys(node_ids, 0)
    profile_counts: Counter[tuple[float, int, int]] = Counter()
    profile_specs: dict[tuple[float, int, int], Any] = {}
    unpriced = 0

    for record in live_raw:
        nid = record.node_id
        spec = _resource_spec_from_json(record.effective_resources_json)
        if nid is None or spec is None:
            # ``acquiring`` before placement lands: no node, no footprint yet.
            unpriced += 1
            continue
        load.setdefault(nid, []).append((record.image, spec))
        counts[nid] = counts.get(nid, 0) + 1
        key = _profile_key(spec)
        profile_counts[key] += 1
        profile_specs.setdefault(key, spec)

    for sb in sandboxes:
        counts[sb.node_id] = counts.get(sb.node_id, 0) + 1

    # ── Workload profiles: live raw footprints first, then templates ────────
    catalog = TemplateCatalog()
    pkg_templates = Path(__file__).resolve().parent.parent / "templates"
    if pkg_templates.exists():
        catalog.register_dir(pkg_templates)
    manifests = catalog.list()

    profiles: list[tuple[str, str, Any]] = []   # (kind, label, manifest)
    for key, _count in profile_counts.most_common(_MAX_CAPACITY_PROFILES):
        profiles.append((
            "live",
            _profile_label(key),
            _synthetic_manifest_for_raw("live-workload", profile_specs[key]),
        ))
    profiles_omitted = max(0, len(profile_counts) - _MAX_CAPACITY_PROFILES)
    for manifest in manifests:
        profiles.append(("template", manifest.name, manifest))

    # ── The matrix ──────────────────────────────────────────────────────────
    estimator = StaticCapacityEstimator()
    nodes_out: list[dict[str, Any]] = []
    for nid in node_ids:
        hw = _node_hardware_live(cfg, nid)
        running = load.get(nid, [])
        row: dict[str, Any] = {
            "id": nid,
            "hardware_known": hw is not None,
            "vcpus": hw.vcpus if hw else None,
            "mem_bytes": hw.mem_bytes if hw else None,
            "disk_bytes": hw.disk_bytes if hw else None,
            "active": counts.get(nid, 0),
            "used_cpu": round(sum(s.cpu_request for _, s in running), 2),
            "used_mem_bytes": sum(s.mem_request_bytes for _, s in running),
            "used_disk_bytes": sum(s.disk_request_bytes for _, s in running),
            "cells": [],
        }
        if hw is not None:
            profile = NodeProfile(
                node_id=nid, hardware=hw, backends=("docker",),
            )
            for kind, label, manifest in profiles:
                cell = estimator.capacity(profile, manifest)
                left = estimator.capacity_remaining(profile, running, manifest)
                row["cells"].append({
                    "kind": kind,
                    "label": label,
                    "max_concurrent": cell.max_concurrent,
                    "remaining": left.max_concurrent,
                    "remaining_binding": left.binding_constraint,
                    "cpu_cap": cell.cpu_cap,
                    "mem_cap": cell.mem_cap,
                    "disk_cap": cell.disk_cap,
                    "binding_constraint": cell.binding_constraint,
                })
        nodes_out.append(row)

    cluster_total = {
        label: sum(
            c["max_concurrent"]
            for n in nodes_out for c in n["cells"] if c["label"] == label
        )
        for _kind, label, _m in profiles
    }

    return {
        "nodes": nodes_out,
        "profiles": [{"kind": k, "label": lb} for k, lb, _ in profiles],
        "cluster_total": cluster_total,
        "profiles_omitted": profiles_omitted,
        "unpriced_live": unpriced,
        "live_containers": sum(counts.values()),
        "hardware_unknown": [n["id"] for n in nodes_out if not n["hardware_known"]],
        "node_lookup_wired": cfg.node_lookup is not None,
        "matrix_present": bool(
            profiles and any(n["hardware_known"] for n in nodes_out)
        ),
        "computed_at": computed_at,
        "computed_at_label": datetime.fromtimestamp(computed_at).strftime("%H:%M:%S"),
    }
async def _gather_health(cfg: AdminServerConfig) -> dict[str, Any]:
    return await asyncio.to_thread(_health_blocking, cfg)


def _health_blocking(cfg: AdminServerConfig) -> dict[str, Any]:
    """Spec-13 Cluster-health view, reworked for Stage 1 of the
    admission/capacity design (notes/admission-stage-1-observability.md).

    The centrepiece is a per-node signal table — docker-run p95, docker
    error/timeout counts, create-gate contention, heartbeat age — fed by
    the Stage-1 ``node_health`` mirror. Triage sections (long-running /
    queued sessions, failure rate) cover both case-1 sandboxes and case-2/3 raw
    rollouts,
    so the page is informative for raw-container workloads (the dominant
    audience), not just template rollouts.
    """
    evaluated_at = time.time()
    label = datetime.fromtimestamp(evaluated_at).strftime("%H:%M:%S")
    if not cfg.state_db.exists():
        return {
            "nodes": [], "long_running": [],
            "long_running_threshold_label": (
                f"{_HEALTH_LONG_RUNNING_AGE_S / 3600.0:.1f}h"
            ),
            "failure_rate_high": [],
            "all_clear": True, "check_count": _HEALTH_CHECK_COUNT,
            "evaluated_at": evaluated_at, "evaluated_at_label": label,
        }

    now = evaluated_at
    long_running_cutoff = now - _HEALTH_LONG_RUNNING_AGE_S
    failure_cutoff = now - 30 * 60.0
    store = SqliteStateStore(cfg.state_db, read_only=True)
    try:
        node_rows = store.list_nodes(status="connected")
        node_health = store.list_node_health()
        node_aimd = store.list_node_aimd_limits()
        rollouts = store.list_rollouts()
        sandboxes = store.list_sandboxes()
        # Bounded aggregates instead of loading every raw row — the full
        # load here was O(168k) per render and pinned the WAL open
        # ([[wal-runaway-cp-stall]]). Each of these is a GROUP BY / indexed
        # COUNT / small-subset query.
        raw_running_by_node = store.running_raw_counts_by_node()
        raw_long_running = store.list_long_running_raw(long_running_cutoff)
        raw_started_30m = store.count_raw_rollouts_created_since(failure_cutoff)
        raw_failed_30m = store.count_raw_rollouts_finished_since(
            failure_cutoff, ("failed", "cancelled"),
        )
    finally:
        store.close()

    # Running-container count per node — case-1 sandbox + case-2/3 raw.
    running_by_node: dict[str, int] = {}
    for sb in sandboxes:
        running_by_node[sb.node_id] = running_by_node.get(sb.node_id, 0) + 1
    for node_id, n in raw_running_by_node.items():
        running_by_node[node_id] = running_by_node.get(node_id, 0) + n

    # Per-node signal table — the Stage-1 centrepiece.
    nodes: list[dict[str, Any]] = []
    for rec in node_rows:
        raw_health = node_health.get(rec.node_id)
        h: dict[str, Any] = {}
        if raw_health:
            try:
                h = json.loads(raw_health)
            except (ValueError, TypeError):
                h = {}
        age_s = max(0.0, now - rec.last_seen_at)
        p95 = float(h.get("create_p95_ms", 0.0))
        errors = int(h.get("docker_error_count", 0))
        healthy = (
            age_s <= _HEALTH_HEARTBEAT_STALE_S
            and p95 <= _HEALTH_CREATE_P95_HIGH_MS
            and errors == 0
        )
        nodes.append({
            "node_id": rec.node_id,
            "running": running_by_node.get(rec.node_id, 0),
            "create_p95_ms": p95,
            "create_count": int(h.get("create_count", 0)),
            "docker_error_count": errors,
            "docker_timeout_count": int(h.get("docker_timeout_count", 0)),
            "create_inflight": int(h.get("create_inflight", 0)),
            "create_queued": int(h.get("create_queued", 0)),
            "heartbeat_age_s": age_s,
            "has_health": bool(h),
            "healthy": healthy,
            # Stage-3 — the AIMD adaptive admission limit; None when
            # adaptive admission is off or the node hasn't been ticked.
            "adaptive_limit": node_aimd.get(rec.node_id),
        })

    # Long-running & queued sessions: case-1 sandboxes + non-terminal raw
    # rollouts alive beyond a coarse ceiling. This is an age heuristic for
    # triage, NOT a "stuck"/failure signal — long-horizon rollouts and
    # persistent substrate containers (e.g. WebArena) legitimately run for
    # hours, and a raw rollout still ``acquiring`` is parked in the admission
    # queue waiting for capacity (backpressure), not hung. The ``state`` column
    # spells out which case each row is so the operator isn't left guessing.
    threshold_s = _HEALTH_LONG_RUNNING_AGE_S

    def _age_label(age_s: float) -> str:
        return f"{age_s / 3600.0:.1f}h" if age_s >= 3600.0 else f"{age_s:.0f}s"

    long_running: list[dict[str, Any]] = []
    for sb in sandboxes:
        age = now - sb.created_at
        if age > threshold_s:
            long_running.append({
                "kind": "sandbox", "id": sb.sandbox_id, "state": "alive",
                "node_id": sb.node_id, "label": sb.template,
                "age_s": age, "age_label": _age_label(age),
            })
    # ``raw_long_running`` is already the acquiring/running subset older
    # than ``threshold_s`` (the query filtered ``created_at < now -
    # _HEALTH_LONG_RUNNING_AGE_S``), so no per-row age gate is needed here.
    for rr in raw_long_running:
        age = now - rr.created_at
        state = (
            "queued — awaiting capacity"
            if rr.status == "acquiring" else "running"
        )
        long_running.append({
            "kind": "raw", "id": rr.rollout_id, "state": state,
            "node_id": rr.node_id or "?", "label": rr.image,
            "age_s": age, "age_label": _age_label(age),
        })

    # Failure rate over the last 30 min — case-1 per template, plus a
    # single aggregate row for case-2/3 raw rollouts. ``failure_cutoff``
    # (== now - 30 min) is shared with the raw aggregates computed above.
    failure_rate_high: list[dict[str, Any]] = []
    started_by: dict[str, int] = {}
    failed_by: dict[str, int] = {}
    for r in rollouts:
        if r.created_at < failure_cutoff:
            continue
        started_by[r.template] = started_by.get(r.template, 0) + 1
        if r.status == RolloutStatus.FAILED:
            failed_by[r.template] = failed_by.get(r.template, 0) + 1
    for workload, started in started_by.items():
        failed = failed_by.get(workload, 0)
        rate = failed / max(started, 1)
        if rate >= 0.25 and started >= 4:
            failure_rate_high.append({
                "workload": workload, "started": started,
                "failed": failed, "rate": rate,
            })
    # Bounded aggregates (computed in the store block above) instead of a
    # full-table Python tally over every raw row.
    raw_started = raw_started_30m
    raw_failed = raw_failed_30m
    if raw_started >= 4:
        raw_rate = raw_failed / max(raw_started, 1)
        if raw_rate >= 0.25:
            failure_rate_high.append({
                "workload": "raw containers (all)", "started": raw_started,
                "failed": raw_failed, "rate": raw_rate,
            })

    # Long-running/queued sessions are informational, not health failures, so
    # they deliberately do NOT flip ``all_clear`` — otherwise a healthy cluster
    # running a persistent substrate container would read "Findings" forever.
    all_clear = (
        not failure_rate_high
        and all(n["healthy"] for n in nodes)
    )
    return {
        "nodes": nodes, "long_running": long_running,
        "long_running_threshold_label": _age_label(threshold_s),
        "failure_rate_high": failure_rate_high,
        "all_clear": all_clear, "check_count": _HEALTH_CHECK_COUNT,
        "evaluated_at": evaluated_at, "evaluated_at_label": label,
    }


async def _gather_rollout_detail(
    cfg: AdminServerConfig,
    rollout_id: str,
    *,
    cache: TrajectoryCache,
    fetch_fn: Any,
) -> dict[str, Any] | None:
    """Build the rollout-detail snapshot. The trajectory body comes through
    the spec-17 cache so distributed multi-host setups don't pay a fresh
    bidi fetch per click."""
    base = await asyncio.to_thread(_rollout_detail_blocking, cfg, rollout_id)
    if base is None:
        return None
    trajectory: Trajectory | None
    # Cache only terminal rollouts. The spec-17 cache assumes the
    # body is immutable once written — true for sealed trajectories,
    # NOT true for in-flight rollouts where ``record_step`` keeps
    # appending to ``trajectory.jsonl``. Caching a partial body
    # would lock in a stale step count for the cache TTL even after
    # the live file grows. Terminal statuses (``finished``, ``failed``,
    # ``truncated``, ``cancelled``) are immutable per spec 00 invariant 3.
    # Audit M1 (2026-05-01).
    is_terminal = base["record"]["status"] in (
        "finished", "failed", "truncated", "cancelled",
    )
    try:
        if is_terminal:
            trajectory = await cache.get(rollout_id, fetch_fn)
        else:
            # Non-terminal: read directly without caching. The local
            # ``trajectory.jsonl`` is the freshest source.
            trajectory = await fetch_fn(rollout_id)
    except FileNotFoundError:
        trajectory = None
    except Exception as exc:
        # Spec-17 ReplayUnavailable: distributed-mode fetch failed (node
        # unreachable, sink=none). Surface a hint in the rendered page
        # instead of 500-ing the whole rollout-detail view.
        LOGGER.info(
            "trajectory fetch failed for rollout=%s: %s", rollout_id, exc,
        )
        trajectory = None
    base["trajectory"] = trajectory
    base["step_count_on_disk"] = (
        len(trajectory.steps) if trajectory else None
    )
    base["verifier_files"] = await asyncio.to_thread(
        _gather_verifier_artifacts, cfg, rollout_id,
    )
    coord_log = await asyncio.to_thread(
        _read_coordinator_log_tail, cfg, rollout_id,
    )
    if coord_log is None:
        base["coordinator_log_tail"] = None
        base["coordinator_log_truncated_head"] = False
        base["coordinator_log_total_bytes"] = 0
    else:
        text, truncated_head, total_bytes = coord_log
        base["coordinator_log_tail"] = text
        base["coordinator_log_truncated_head"] = truncated_head
        base["coordinator_log_total_bytes"] = total_bytes
    return base


# Inline-preview cap on individual verifier files. Files larger than
# this render as "(file is N bytes — open the link to view)" and the
# operator clicks through to the raw download endpoint to read the
# whole thing. 64 KiB is enough for typical pytest test.log + ctrf.json
# tails without bloating the rendered HTML.
_VERIFIER_INLINE_PREVIEW_BYTES = 64 * 1024

# Inline-tail cap on coordinator.log. Same 64 KiB budget — typical
# rollouts emit ~10-50 JSON-lines averaging 200-500 bytes each, well
# under the cap; long-running rollouts with many image-cache /
# scheduler events get the head clipped, with a "Download full log"
# link rendering when truncation kicks in.
_COORDINATOR_LOG_INLINE_TAIL_BYTES = 64 * 1024

# File extensions we treat as text and inline as ``<pre>`` content in
# the rollout-detail page. Anything else renders as a download link
# only — image/binary/etc. blobs aren't inlined.
_VERIFIER_TEXT_SUFFIXES = frozenset(
    {".txt", ".log", ".json", ".html", ".md", ".csv", ".tsv", ".xml", ".yaml", ".yml"}
)


def _gather_verifier_artifacts(
    cfg: AdminServerConfig, rollout_id: str,
) -> list[dict[str, Any]] | None:
    """Walk ``<run_dir>/verifier/`` and produce a per-file summary.

    The platform's coordinator (D12 stage 1 follow-up) extracts the
    sandbox's ``/logs/verifier/`` into ``<run_dir>/verifier/`` for any
    benchmark that follows harbor's verifier-output convention. This
    helper surfaces those files in the admin panel so an operator can
    distinguish a genuine agent failure (test.log shows assertion
    errors) from a verifier misfire (no test.log, or empty reward.txt)
    without shelling into the run dir.

    Returns ``None`` when no run dir exists (rollout was cancelled
    before any artifact was written), or an empty list when the run
    dir exists but has no ``verifier/`` subdirectory (benchmarks
    that don't follow the harbor convention — e.g. hello-shell —
    just won't show this section).
    """
    run_dir = _run_dir_for_rollout(cfg, rollout_id)
    if run_dir is None:
        return None
    verifier_root = run_dir / "verifier"
    if not verifier_root.is_dir():
        return []

    out: list[dict[str, Any]] = []
    for entry in sorted(verifier_root.rglob("*")):
        if not entry.is_file():
            continue
        rel = entry.relative_to(verifier_root).as_posix()
        size = entry.stat().st_size
        suffix = entry.suffix.lower()
        is_text = suffix in _VERIFIER_TEXT_SUFFIXES
        preview: str | None = None
        truncated = False
        if is_text:
            try:
                # Read up to the cap+1; if the file is bigger we mark
                # truncated so the template can show a "Download for
                # full file" hint.
                with entry.open("rb") as fp:
                    raw = fp.read(_VERIFIER_INLINE_PREVIEW_BYTES + 1)
                if len(raw) > _VERIFIER_INLINE_PREVIEW_BYTES:
                    raw = raw[:_VERIFIER_INLINE_PREVIEW_BYTES]
                    truncated = True
                preview = raw.decode("utf-8", errors="replace")
            except OSError as exc:
                preview = f"<read failed: {exc}>"
        out.append(
            {
                "rel_path": rel,
                "size_bytes": size,
                "is_text": is_text,
                "preview": preview,
                "truncated": truncated,
            }
        )
    return out


def _read_coordinator_log_tail(
    cfg: AdminServerConfig, rollout_id: str,
) -> tuple[str, bool, int] | None:
    """Read up to the last :data:`_COORDINATOR_LOG_INLINE_TAIL_BYTES`
    bytes of the rollout's ``coordinator.log``.

    Returns ``(text, truncated_at_head, total_bytes)`` so the template
    can decide whether to surface a "Download full log" link, or
    ``None`` when the log file doesn't exist (rollout hasn't started,
    sink wasn't configured, or run dir was pruned).

    The lifecycle log is JSON-lines; we slice on a byte boundary and
    discard any partial leading line so the rendered tail doesn't
    show truncated JSON. The trailing line is always whole because
    the sink writes a newline-terminated record per event.
    """
    run_dir = _run_dir_for_rollout(cfg, rollout_id)
    if run_dir is None:
        return None
    log_path = run_dir / "coordinator.log"
    if not log_path.is_file():
        return None
    total_bytes = log_path.stat().st_size
    if total_bytes == 0:
        return ("", False, 0)
    cap = _COORDINATOR_LOG_INLINE_TAIL_BYTES
    truncated_head = total_bytes > cap
    with log_path.open("rb") as fp:
        if truncated_head:
            fp.seek(total_bytes - cap)
        raw = fp.read()
    text = raw.decode("utf-8", errors="replace")
    if truncated_head:
        # Drop the partial leading line so the first record on screen
        # is well-formed JSON; the user can still see the missing
        # prefix via the download link.
        nl = text.find("\n")
        if nl >= 0:
            text = text[nl + 1 :]
    return (text, truncated_head, total_bytes)


def _run_dir_for_rollout(
    cfg: AdminServerConfig, rollout_id: str,
) -> Path | None:
    """Return ``<runs_root>/<date>/<rollout_id>/`` for any sealed or
    in-flight rollout that wrote artifacts. Mirrors the lookup the
    PlatformJsonlSink does, but free of the sink's open-rollout map
    (the admin server runs in a separate process from the
    coordinator)."""
    if not cfg.runs_root.exists():
        return None
    for date_dir in sorted(cfg.runs_root.iterdir()):
        candidate = date_dir / rollout_id
        if candidate.is_dir():
            return candidate
    return None


def _rollout_detail_blocking(
    cfg: AdminServerConfig, rollout_id: str
) -> dict[str, Any] | None:
    """Snapshot the rollout's StateStore record + recent events. The
    trajectory body is fetched separately via the spec-17 cache from
    :py:func:`_gather_rollout_detail`."""
    if not cfg.state_db.exists():
        return None
    store = SqliteStateStore(cfg.state_db, read_only=True)
    try:
        try:
            record = store.get_rollout(rollout_id)
        except KeyError:
            return None
        events = [e for e in store.events_since(0) if e.rollout_id == rollout_id]
    finally:
        store.close()

    return {
        "record": _rollout_dict(record),
        "metadata": record.metadata,
        "events": [
            {"seq": e.seq, "ts": e.ts, "kind": e.kind, "payload": e.payload}
            for e in events
        ],
        "trajectory": None,  # filled by _gather_rollout_detail via cache
        "step_count_on_disk": None,
    }


# ──────────────────────────────────────────────────────────────────────────────
# uvicorn lifecycle wrapper
# ──────────────────────────────────────────────────────────────────────────────


class AdminServer(BaseModel):
    """Lifecycle wrapper around uvicorn for the FastAPI admin app.

    Mirrors :class:`MetricsServer`'s start/stop surface so ``LocalRuntime``
    and ``DistributedRuntime`` can compose the server into the existing
    startup pipeline.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    config: AdminServerConfig
    _server: SkipValidation[uvicorn.Server | None] = None
    _thread: SkipValidation[threading.Thread | None] = None
    # Sibling-loopback (dual-stack) server. When the operator binds on
    # a loopback host (127.0.0.1 / ::1 / localhost), we ALSO bind on
    # the sibling-family loopback so VS Code's port-forward — which
    # may resolve ``localhost`` to whichever family the remote OS
    # prefers (typically ::1 first on Linux) — reaches the panel
    # regardless of the family it chose. Tolerant of the sibling
    # bind failing (IPv6 disabled, dual-stack restrictions): we log
    # a warning and keep the primary listener.
    _sibling_server: SkipValidation[uvicorn.Server | None] = None
    _sibling_thread: SkipValidation[threading.Thread | None] = None

    @property
    def host(self) -> str:
        return self.config.host

    @property
    def port(self) -> int:
        # uvicorn only sets ``Server.servers`` once ``startup()`` has bound the
        # listeners; on a fresh or not-yet-started server (uvicorn ≥0.46) the
        # attribute is absent, so read it defensively. When it isn't there yet
        # we fall back to the configured port — correct for a fixed port, and
        # the read-back simply isn't available until the bind completes for a
        # ``port=0`` (kernel-assigned) bind.
        server = self._server
        servers = getattr(server, "servers", None) if server is not None else None
        if servers:
            sockets = servers[0].sockets
            if sockets:
                return int(sockets[0].getsockname()[1])
        return self.config.port

    def start(self) -> None:
        """Bind ``host:port`` and start serving in a daemon thread.

        uvicorn runs its own asyncio loop in the thread so it doesn't
        compete with the control-plane's main loop. On loopback binds
        a second uvicorn instance is launched on the sibling-family
        loopback (IPv4 ↔ IPv6) so the panel is reachable regardless
        of which family the connecting client resolves ``localhost``
        to — closes the VS Code Remote port-forward IPv4/IPv6
        mismatch.
        """
        if self._server is not None:
            return
        _enforce_bind_guard(self.config)
        app = build_admin_app(self.config)
        ucfg = uvicorn.Config(
            app,
            host=self.config.host,
            port=self.config.port,
            log_config=None,
            access_log=False,
            lifespan="off",
        )
        server = uvicorn.Server(ucfg)

        def _serve(srv: uvicorn.Server) -> None:
            # uvicorn ≥0.46's ``capture_signals`` context manager only works
            # on the main thread; from our daemon thread we just call run()
            # which skips signal capture when no main-thread loop is set.
            import asyncio as _asyncio

            _asyncio.run(srv.serve())

        thread = threading.Thread(
            target=_serve, args=(server,),
            name="xrlenv-admin", daemon=True,
        )
        thread.start()
        # Wait briefly for the socket to bind so callers + tests can read
        # back the actual port (when port=0 the kernel picks one).
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if server.started:
                break
            time.sleep(0.05)
        self._server = server
        self._thread = thread
        LOGGER.info(
            "admin server listening on http://%s:%d", self.host, self.port,
        )

        # ── Sibling-loopback bind for dual-stack reachability ──────────
        sibling_host = _sibling_loopback(self.host)
        if sibling_host is not None:
            sibling_port = self.port  # kernel-picked port if config.port == 0.
            sibling_ucfg = uvicorn.Config(
                app,
                host=sibling_host, port=sibling_port,
                log_config=None, access_log=False, lifespan="off",
            )
            sibling_server = uvicorn.Server(sibling_ucfg)
            sibling_thread = threading.Thread(
                target=_serve, args=(sibling_server,),
                name="xrlenv-admin-sibling", daemon=True,
            )
            sibling_thread.start()
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if sibling_server.started:
                    break
                time.sleep(0.05)
            if sibling_server.started:
                self._sibling_server = sibling_server
                self._sibling_thread = sibling_thread
                LOGGER.info(
                    "admin server: also listening on sibling-loopback "
                    "http://%s:%d (dual-stack reachability for VS Code "
                    "port-forwarding etc.)",
                    _format_host_for_url(sibling_host), sibling_port,
                )
            else:
                LOGGER.warning(
                    "admin server: sibling-loopback bind on %s:%d "
                    "didn't come up within 5s — panel reachable only "
                    "via %s. Likely the host's IPv6 stack is disabled "
                    "or restricted; not fatal.",
                    sibling_host, sibling_port, self.host,
                )
        # Operator-UX gotcha worth surfacing on every startup: modern
        # browsers auto-upgrade ``http://localhost`` to https, which
        # uvicorn (HTTP-only) can't satisfy → silent connection
        # failure that surfaces as a generic "Internal Server Error"
        # in the browser, with nothing in the server log to point at.
        # Bites operators doing VS Code remote port-forwarding (which
        # always surfaces ``localhost:<port>`` in the "Open in
        # Browser" links). 127.0.0.1 is exempt from the upgrade in
        # most browsers.
        host_lower = self.host.lower()
        if host_lower in ("127.0.0.1", "localhost", "0.0.0.0", "::1", "::"):
            LOGGER.info(
                "admin server: use http://127.0.0.1:%d (NOT "
                "http://localhost:%d — Chrome/Safari auto-upgrade "
                "localhost to https, which this server doesn't speak; "
                "the failure surfaces as a generic 'Internal Server "
                "Error' / 'ERR_EMPTY_RESPONSE' with nothing in the log).",
                self.port, self.port,
            )

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._server = None
        self._thread = None
        # Shut down the sibling-loopback listener too, if we brought one up.
        if self._sibling_server is not None:
            self._sibling_server.should_exit = True
            if self._sibling_thread is not None:
                self._sibling_thread.join(timeout=5.0)
            self._sibling_server = None
            self._sibling_thread = None


def _ensure_status(records: Iterable[Any], status: str) -> list[Any]:
    return [r for r in records if r.status.value == status]


def _download_blocking(cfg: AdminServerConfig, rollout_id: str) -> str | None:
    """Read the on-disk trajectory.jsonl bytes for an attachment download.

    Returns ``None`` when the run dir is missing (already pruned by the
    janitor or never written for sink=none); the caller maps that to 404.
    """
    if not cfg.runs_root.exists():
        return None
    for date_dir in sorted(cfg.runs_root.iterdir()):
        candidate = date_dir / rollout_id / "trajectory.jsonl"
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")
    return None


__all__ = [
    "AdminBindError",
    "AdminServer",
    "AdminServerConfig",
    "build_admin_app",
]
