"""Transport types — how the agent's source code reaches the container.

Two concrete shapes:

- ``BindMount``: a host path is mounted into the container. Cheap, fast
  iteration; only viable when the host and container share a filesystem
  (i.e. local docker).
- ``GitClone``: the container clones a repo at run-time. Required for
  remote/cloud runtimes where the host filesystem isn't reachable.

The ``Transport`` union lets adapters declare what they need without
knowing which runtime will materialize it. ``LocalDockerRuntime`` accepts
``BindMount`` directly via ``acquire(mounts=...)``; ``GitClone`` is
materialized later via ``exec`` (a `git clone ...` command run inside the
already-running container).
"""
from __future__ import annotations

import random
import re
import shlex
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from beagle.rollout.runtime.runtime import ExecResult

# A ref that is all-hex and >= 7 chars is treated as a commit SHA (``git clone
# --branch`` rejects commit SHAs — see :func:`git_clone_argv`). Branch/tag names
# that happen to be all-hex are pathological and not supported via this path.
_SHA_RE = re.compile(r"[0-9a-fA-F]{7,40}\Z")


@dataclass(frozen=True)
class BindMount:
    """Bind-mount a host path into the container (``docker run -v``).

    ``host_path`` is an absolute path on the host. ``container_path`` is
    where it should appear inside the container. ``read_only=True`` is
    the safe default — we don't want agents writing back into our
    project tree.
    """

    host_path: Path
    container_path: str
    read_only: bool = True


@dataclass(frozen=True)
class GitClone:
    """Clone a git repo into the container at start-of-run.

    Used by remote runtimes where the host's filesystem isn't reachable
    from inside the container, so a bind mount is impossible. The
    runtime is responsible for materializing this (typically via
    ``git clone --depth 1 --branch <ref>`` inside the container).

    ``token_env`` names an environment variable (e.g. ``"GH_TOKEN"``)
    whose value is injected into the HTTPS URL at clone time as
    ``https://x-access-token:<token>@host/...``. The wrapper argv that
    ``subprocess.run`` / ``docker exec`` carries on the host holds only
    the variable name (``$GH_TOKEN``), not the value — the shell expands
    it at exec time inside the container. The spawned ``git clone``
    process itself has the expanded URL in its argv and is visible via
    ``ps`` *inside the container* for the duration of the clone; for
    process-list secrecy switch to a git credential helper / ``GIT_ASKPASS``
    flow (not done here). See :func:`git_clone_argv`. If ``None``, the
    clone is unauthenticated (public repos).
    """

    repo_url: str
    ref: str
    container_path: str
    token_env: str | None = None


Transport = BindMount | GitClone


def git_clone_argv(spec: GitClone) -> list[str]:
    """Argv for materializing a ``GitClone`` via ``rt.exec(handle, ...)``.

    Shallow-fetches the requested ref into ``container_path``. Callers run this
    *after* ``acquire`` (clones can only happen inside a running container).

    Two shapes, because ``git clone --branch`` accepts a **branch or tag** but
    **rejects a commit SHA** (``"Could not find remote branch <sha>"``):

    * branch/tag ref → ``git clone --depth 1 --branch <ref>`` (fast path);
    * commit-SHA ref → ``git init`` + ``git fetch --depth 1 origin <sha>`` +
      ``git checkout FETCH_HEAD`` (fetch-by-commit; GitHub serves reachable SHAs).
      Pin baselines with a **full** 40-char SHA — abbreviated SHAs may be rejected
      by the server's want-protocol.

    When ``token_env`` is set the URL is rewritten with the token via shell
    expansion at exec time, so the *wrapper* argv holds only the variable name
    (``$GH_TOKEN``); the spawned git process still carries the expanded URL in its
    argv (visible via ``ps`` inside the container). For process-list secrecy use a
    git credential helper / ``GIT_ASKPASS`` flow.
    """
    # ``repo_url`` is arg $1, ref is $2, container_path is $3. Rewrite $1 with the
    # token when present; otherwise pass it through.
    rewrite = (
        f'repo_url=$(echo "$1" | sed "s|https://|https://x-access-token:${spec.token_env}@|")'
        if spec.token_env
        else 'repo_url="$1"'
    )

    if not spec.ref:
        # No ref → clone the repo's *default* branch (don't pass --branch, and don't
        # guess "HEAD"/"main" — ``git clone --branch HEAD`` fails on the remote).
        if spec.token_env:
            script = f'{rewrite} && git clone --depth 1 "$repo_url" "$2"'
            return ["sh", "-c", script, "--", spec.repo_url, spec.container_path]
        return ["git", "clone", "--depth", "1", spec.repo_url, spec.container_path]

    if _SHA_RE.fullmatch(spec.ref):
        script = (
            f"{rewrite} && "
            'git init -q "$3" && cd "$3" && '
            'git remote add origin "$repo_url" && '
            'git fetch -q --depth 1 origin "$2" && '
            "git checkout -q FETCH_HEAD"
        )
        return ["sh", "-c", script, "--", spec.repo_url, spec.ref, spec.container_path]

    if spec.token_env:
        script = f'{rewrite} && git clone --depth 1 --branch "$2" "$repo_url" "$3"'
        return ["sh", "-c", script, "--", spec.repo_url, spec.ref, spec.container_path]

    return [
        "git", "clone", "--depth", "1",
        "--branch", spec.ref, spec.repo_url, spec.container_path,
    ]


#: Substrings (lowercased) that mark a git-clone failure as TRANSIENT — a retry may clear it. The
#: load-bearing case: many trials cloning the same private repo at once trip GitHub's per-token auth
#: throttle, surfacing as a spurious ``401`` / ``invalid username or token`` / ``RPC failed`` mid
#: ref-negotiation; a short backoff clears it. :meth:`install` already verifies the token is *set*, so a
#: set-but-rejected token is overwhelmingly throttle (not a bad value) — and a genuinely bad token just
#: exhausts the bounded retries to the same terminal failure, only later. A DEFINITIVE failure (repo/ref
#: not found, unauthorized on a valid repo) matches none of these and fails fast.
_TRANSIENT_CLONE_SIGNS: tuple[str, ...] = (
    "rpc failed",
    "http 401", "http 429", "http 500", "http 502", "http 503", "http 504",
    "returned error: 401", "returned error: 429", "returned error: 5",
    "invalid username or token",
    "expected flush after ref listing",
    "could not read from remote repository",
    "unable to access",
    "failed to connect", "connection reset", "connection timed out", "timed out",
    "early eof", "gnutls_handshake", "ssl_read", "ssl_write",
)


def _is_transient_clone_error(stderr: str) -> bool:
    """Whether a git-clone ``stderr`` looks transient (retryable) vs definitive. See
    :data:`_TRANSIENT_CLONE_SIGNS`."""
    s = (stderr or "").lower()
    return any(sign in s for sign in _TRANSIENT_CLONE_SIGNS)


def clone_with_retry(
    runtime: Any,
    handle: Any,
    spec: GitClone,
    *,
    env: dict[str, str] | None = None,
    timeout: float = 300.0,
    attempts: int = 4,
    base_delay: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
    jitter: Callable[[], float] | None = None,
) -> ExecResult:
    """Exec ``git_clone_argv(spec)`` via ``runtime``, retrying a **transient** failure with exponential
    backoff + jitter. Returns the last :class:`ExecResult` (caller inspects ``.ok``); never raises.

    Concurrent trials cloning the same private repo can trip GitHub's per-token auth throttle (a spurious
    ``401`` during ref negotiation); a short *randomized* backoff clears it, and the jitter spreads a
    thundering herd of simultaneous clones so they don't all retry in lockstep. A definitive failure
    (repo/ref missing — :func:`_is_transient_clone_error` is ``False``) fails fast with no retry. Between
    attempts the partial checkout is removed so the retry's ``clone``/``init`` starts on a clean dir
    (git refuses a non-empty target). ``sleep``/``jitter`` are injectable for tests."""
    argv = git_clone_argv(spec)
    rng = jitter or (lambda: random.uniform(0.0, base_delay))
    result = runtime.exec(handle, argv, timeout=timeout, env=env)
    attempt = 1
    while not result.ok and attempt < attempts and _is_transient_clone_error(result.stderr):
        # Clear any partial checkout so the retry's clone/init starts clean (git refuses a non-empty dir).
        runtime.exec(handle, ["sh", "-c", f"rm -rf {shlex.quote(spec.container_path)}"], timeout=120)
        sleep(base_delay * (2 ** (attempt - 1)) + rng())
        result = runtime.exec(handle, argv, timeout=timeout, env=env)
        attempt += 1
    return result


__all__ = ["BindMount", "GitClone", "Transport", "clone_with_retry", "git_clone_argv"]
