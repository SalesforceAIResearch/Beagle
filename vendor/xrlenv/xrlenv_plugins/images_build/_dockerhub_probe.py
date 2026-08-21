"""Docker Hub size-probe helper shared by per-benchmark plan generators.

Build-plan generators (``swebench_verified``, ``terminal_bench_2``)
probe Docker Hub's v2 manifest API to learn each image's compressed
layer size so the FFD bin-packer reserves a realistic amount of disk
per entry. Two failure modes bit operators previously and are the
reason this helper exists:

1. **Rate limits.** Unauthenticated requests against
   ``hub.docker.com/v2`` are throttled at ~100 / 6 h per source IP.
   A 500-instance ``--all`` run blows through the budget around entry
   #100; every subsequent probe silently falls back to the
   generator's conservative default size_hint (typically much larger
   than reality), which over-reserves disk in FFD and rejects the
   plan at apply time.
2. **No visibility.** The old probe path returned ``None`` on every
   failure and the generator quietly used the heuristic. Operators
   only learned something was wrong 30 min later when
   ``xrlenv build apply`` reported ``InsufficientCapacity`` against
   an apparently-reasonable plan.

This module fixes both:

- ``probe_image_size`` accepts a Docker Hub Personal Access Token
  via ``DOCKERHUB_USER`` + ``DOCKERHUB_TOKEN`` env vars. When set,
  the helper exchanges them for a JWT once and attaches
  ``Authorization: Bearer <jwt>`` to every probe. Business / Team
  / Pro accounts get their plan's higher (or unlimited) rate cap.
- ``probe_image_size`` increments a process-wide counter on each
  call. ``announce_auth_status`` prints a banner before the loop so
  operators know whether the run is authenticated. The first
  failure surfaces a loud stderr warning with the HTTP status and
  response body so operators see *immediately* that fallbacks have
  started. ``print_probe_summary`` prints final stats — fall back
  count, suspected cause — so the operator knows whether to trust
  the size hints or set ``DOCKERHUB_TOKEN`` and re-run.

The helper is stdlib-only and re-entrant: tests call
``reset_probe_state()`` between cases to clear the JWT cache + stats.

**Thread-safety.** ``probe_image_size`` is safe to call from multiple
threads concurrently. The JWT exchange is guarded by a double-checked
lock so concurrent first-callers exchange exactly once, and every
mutation of the ``_STATS`` counters + ``_FIRST_FAILURE_REPORTED``
flag is performed under ``_STATS_LOCK``. The generator's
``--max-workers`` driver chooses the pool size; the helper just
honors whatever is asked. (This is consistent with the project rule
that concurrency policy lives at the driver level — xrlenv core
carries no locks; this is a driver-side helper under
``xrlenv_plugins/``.)
"""

from __future__ import annotations

import json
import os
import sys
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field

# Env-var names operators set in their ``.env`` or shell.
USER_ENV = "DOCKERHUB_USER"
TOKEN_ENV = "DOCKERHUB_TOKEN"

# Module-level state. ``_HUB_JWT_CACHE`` is set after a successful
# user/pass→JWT exchange; ``None`` means we haven't tried yet (or
# tried and failed — see ``_AUTH_TRIED``). ``_STATS`` accumulates
# probe outcomes for the current generator run.
_HUB_JWT_CACHE: str | None = None
_AUTH_TRIED: bool = False
_FIRST_FAILURE_REPORTED: bool = False

# Guards the JWT exchange (double-checked locking) and stat-counter
# updates so the helper is safe under any pool size the driver picks.
_AUTH_LOCK = threading.Lock()
_STATS_LOCK = threading.Lock()


@dataclass
class ProbeStats:
    """Counts of probe outcomes since the last ``reset_probe_state()``.

    ``ok`` and ``failed`` are the two terminal buckets;
    ``authenticated`` reflects whether the JWT exchange succeeded
    (i.e. whether the operator's account-tier rate cap applies).
    Generators read these to emit a summary line at end-of-run.
    """

    ok: int = 0
    failed: int = 0
    authenticated: bool = False
    first_failure_status: int | None = None
    first_failure_body: str = ""
    failure_image_refs: list[str] = field(default_factory=list)


_STATS = ProbeStats()


def reset_probe_state() -> None:
    """Reset the JWT cache + stats. Tests call this between cases;
    generators don't need to call it explicitly."""
    global _HUB_JWT_CACHE, _AUTH_TRIED, _FIRST_FAILURE_REPORTED, _STATS
    _HUB_JWT_CACHE = None
    _AUTH_TRIED = False
    _FIRST_FAILURE_REPORTED = False
    _STATS = ProbeStats()


def _hub_jwt() -> str | None:
    """Exchange ``$DOCKERHUB_USER`` + ``$DOCKERHUB_TOKEN`` for a Hub
    JWT. Returns ``None`` when either env var is missing or the
    exchange fails (probes then continue unauth). The result is
    cached so the exchange runs at most once per process. Safe under
    concurrent first-callers via double-checked locking.
    """
    global _HUB_JWT_CACHE, _AUTH_TRIED
    # Fast path: most calls hit this after the first thread has
    # populated the cache. A racy read of ``_AUTH_TRIED == True``
    # against a stale ``None`` cache is harmless (probe falls back
    # to unauth for that one call); the slow path locks.
    if _AUTH_TRIED:
        return _HUB_JWT_CACHE
    with _AUTH_LOCK:
        if _AUTH_TRIED:
            return _HUB_JWT_CACHE
        _AUTH_TRIED = True

        user = os.environ.get(USER_ENV)
        token = os.environ.get(TOKEN_ENV)
        if not user or not token:
            return None
        req = urllib.request.Request(
            "https://hub.docker.com/v2/users/login/",
            data=json.dumps({"username": user, "password": token}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                if resp.status != 200:
                    print(
                        f"WARN: Docker Hub auth exchange returned HTTP "
                        f"{resp.status}; probes will fall back to "
                        f"unauthenticated requests "
                        f"(rate-limited ~100/6h).",
                        file=sys.stderr,
                    )
                    return None
                payload = json.loads(resp.read())
        except (urllib.error.URLError, urllib.error.HTTPError,
                OSError, ValueError) as exc:
            print(
                f"WARN: Docker Hub auth exchange failed "
                f"({type(exc).__name__}: {exc}); probes will fall "
                f"back to unauthenticated requests "
                f"(rate-limited ~100/6h).",
                file=sys.stderr,
            )
            return None
        jwt = payload.get("token")
        if not isinstance(jwt, str):
            return None
        _HUB_JWT_CACHE = jwt
        _STATS.authenticated = True
        return jwt


def announce_auth_status() -> None:
    """Print a one-line banner naming the auth state. Generators call
    this once before the per-entry probe loop so the operator knows
    upfront whether the run is rate-limited.
    """
    jwt = _hub_jwt()
    if jwt is not None:
        print(
            f"Docker Hub probes: authenticated as "
            f"${USER_ENV}={os.environ.get(USER_ENV)!r} "
            f"(account-tier rate cap applies).",
            file=sys.stderr,
        )
    else:
        print(
            f"Docker Hub probes: unauthenticated — rate-limited at "
            f"~100 / 6h per source IP. Set ${USER_ENV} + ${TOKEN_ENV} "
            f"(a Docker Hub Personal Access Token) to lift the limit "
            f"to your account's tier cap.",
            file=sys.stderr,
        )


def probe_image_size(
    repo: str,
    tag: str,
    *,
    timeout_s: float = 10.0,
) -> int | None:
    """Probe ``repo:tag`` for its compressed image size on Docker Hub.

    Returns the size in bytes on success, ``None`` on any failure
    (network error, non-200 HTTP, missing ``images`` field, etc.).
    The first failure in a run prints a loud stderr warning with the
    HTTP status and response body so operators learn immediately
    that fallbacks have started — the prior behavior (silent
    ``None``) hid rate-limit cliffs.
    """
    global _FIRST_FAILURE_REPORTED

    url = f"https://hub.docker.com/v2/repositories/{repo}/tags/{tag}"
    headers: dict[str, str] = {}
    jwt = _hub_jwt()
    if jwt is not None:
        headers["Authorization"] = f"Bearer {jwt}"

    req = urllib.request.Request(url, headers=headers)
    status: int | None = None
    body: bytes = b""
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            status = resp.status
            body = resp.read()
            if status != 200:
                _record_failure(f"{repo}:{tag}", status, body)
                return None
            data = json.loads(body)
    except urllib.error.HTTPError as exc:
        # 429 (Too Many Requests) / 401 (token expired) etc. land
        # here; we want the status code in the loud message.
        try:
            body = exc.read()
        except Exception:
            body = b""
        _record_failure(f"{repo}:{tag}", exc.code, body)
        return None
    except (urllib.error.URLError, OSError, ValueError) as exc:
        _record_failure(f"{repo}:{tag}", None, str(exc).encode())
        return None

    images = data.get("images") or []
    if not images:
        _record_failure(f"{repo}:{tag}", status, body)
        return None
    size = images[0].get("size")
    if isinstance(size, int) and size > 0:
        with _STATS_LOCK:
            _STATS.ok += 1
        return size
    _record_failure(f"{repo}:{tag}", status, body)
    return None


def _record_failure(image_ref: str, status: int | None, body: bytes) -> None:
    """Increment counters + emit a one-time loud warning. Holds
    ``_STATS_LOCK`` across all stat updates + the first-failure flag
    flip so concurrent workers can't both decide they're the "first"
    failure and both print the WARN banner.
    """
    global _FIRST_FAILURE_REPORTED
    with _STATS_LOCK:
        _STATS.failed += 1
        _STATS.failure_image_refs.append(image_ref)
        if _FIRST_FAILURE_REPORTED:
            return
        _FIRST_FAILURE_REPORTED = True
        _STATS.first_failure_status = status
        body_snippet = body[:240].decode("utf-8", errors="replace")
        _STATS.first_failure_body = body_snippet
        auth_note = (
            "(authenticated; failure is unexpected — check the response "
            "body)"
            if _STATS.authenticated
            else (
                f"(unauthenticated; ~100 / 6h limit. Set ${USER_ENV} + "
                f"${TOKEN_ENV} to lift it.)"
            )
        )
        print(
            f"\nWARN: Docker Hub probe failed for {image_ref}: "
            f"HTTP {status if status is not None else 'network-error'} "
            f"{auth_note}\n"
            f"      body: {body_snippet!r}\n"
            f"      Subsequent failures will be counted silently; see "
            f"the end-of-run summary for the total.\n",
            file=sys.stderr,
        )


def get_probe_stats() -> ProbeStats:
    """Snapshot of the current run's probe stats (for tests + the
    end-of-run summary printer)."""
    return _STATS


def print_probe_summary(default_size_hint_bytes: int) -> None:
    """Print a final summary after the per-entry loop. Generators
    call this from ``main()`` once they're done iterating.

    ``default_size_hint_bytes`` is the heuristic fallback the
    generator uses when a probe returns ``None``; we name it in the
    summary so the operator can judge how much over-reservation the
    fallback caused (e.g. "300 entries at the 2.5 GiB default
    reserve 750 GiB more than reality").
    """
    total = _STATS.ok + _STATS.failed
    if total == 0:
        return
    if _STATS.failed == 0:
        print(
            f"Docker Hub probes: {_STATS.ok}/{total} succeeded "
            f"(every entry has a registry-probed size_hint).",
            file=sys.stderr,
        )
        return
    fallback_gib = (
        _STATS.failed * default_size_hint_bytes / 1024 ** 3
    )
    cause = (
        "authenticated probe still failing — inspect the first-failure "
        "body above for the upstream error"
        if _STATS.authenticated
        else (
            f"likely Docker Hub rate-limit. Set ${USER_ENV} + "
            f"${TOKEN_ENV} (a Docker Hub PAT) and re-run to get "
            f"accurate sizes for all entries."
        )
    )
    print(
        f"\nWARN: Docker Hub probes: {_STATS.ok}/{total} succeeded, "
        f"{_STATS.failed} fell back to the "
        f"{default_size_hint_bytes / 1024**3:.1f} GiB heuristic "
        f"({fallback_gib:.0f} GiB of over-reservation in FFD "
        f"bin-packing).\n      Cause: {cause}\n",
        file=sys.stderr,
    )
