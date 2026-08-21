"""Node-side multi-service compose project runner (multi-service step 2b).

Brings up a **CP-vetted** compose document (§3.5) on the node's docker daemon via
the real ``docker compose`` CLI — the faithful substrate for TerminalWorld tasks
that ship sidecars + a private network + `depends_on`/healthcheck ordering
(``docker compose up`` does all of that for free). Harbor's agent then execs
solve.sh into the project's ``main`` service over the existing container-scoped
exec wire, unchanged.

Why the CLI and not docker-py: docker-py has no compose support, and the whole
point of this path is to delegate orchestration (networks, static IPs,
`depends_on`, healthchecks) to ``docker compose`` verbatim rather than
re-implement it (CLAUDE.md: don't reinvent benchmark-side wheels).

This module is the pure orchestration core: it shells out through an **injected**
async runner (``run``) and resolves container IDs from ``docker compose ps``, so
its command sequencing, ps-parsing, and teardown are unit-testable with a fake
runner — no docker required. The manager wiring (image-cache ensure, create/
destroy gates, record tracking for GC) layers on top in the node agent.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import tempfile
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from xrlenv.errors import XRLEnvError

# Default per-project ``up --wait`` ceiling. A compose stack pulls N images then
# waits for healthchecks; generous because sidecars (a DB, a mock server) can take
# tens of seconds to report healthy. The caller (node command) can override.
DEFAULT_UP_TIMEOUT_S = 600.0
DEFAULT_DOWN_TIMEOUT_S = 180.0
DEFAULT_MAIN_SERVICE = "main"


class ComposeError(XRLEnvError):
    """A ``docker compose`` invocation failed. Carries the failing argv tail +
    captured output so the operator sees *why* (a healthcheck that never passed,
    an image that won't pull) rather than a bare non-zero exit."""


@dataclass(frozen=True)
class ShellResult:
    """Result of one shelled command."""

    rc: int
    stdout: str
    stderr: str


# An injected runner: ``await run(argv, timeout_s=...) -> ShellResult``. Never
# raises on a non-zero exit — the caller decides what a bad rc means.
Runner = Callable[..., Awaitable[ShellResult]]
# Optional async image-ensure hook: ``await ensure_image(ref)`` (image cache).
ImageEnsurer = Callable[[str], Awaitable[None]]


@dataclass(frozen=True)
class ComposeProjectRecord:
    """A brought-up compose project. ``member_container_ids`` includes ``main``;
    the node tracks all of them ↔ rollout_id for scoping + GC (a crashed consumer
    must have its *whole* project reaped, not just ``main``)."""

    project_name: str
    project_dir: str
    main_container_id: str
    main_container_name: str
    service_container_ids: dict[str, str] = field(default_factory=dict)

    @property
    def member_container_ids(self) -> tuple[str, ...]:
        return tuple(self.service_container_ids.values())


async def _default_run(argv: Sequence[str], *, timeout_s: float | None = None) -> ShellResult:
    """Real subprocess runner (production default). Captures stdout/stderr.

    On timeout, kills the **whole process group** — the command runs in a fresh
    session (``start_new_session=True``, so its pid is the group leader), and we
    ``killpg`` it so any ``docker compose`` helper child is reaped too, not left
    orphaned to race the caller's cleanup. Falls back to a direct ``proc.kill()``
    if the group signal can't be sent (already-exited)."""
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        stdin=asyncio.subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except TimeoutError:
        _kill_process_group(proc)
        await proc.wait()
        raise ComposeError(
            f"`{' '.join(argv[:3])} …` timed out after {timeout_s}s",
        ) from None
    except asyncio.CancelledError:
        # audit H10 — a CANCELLED up/down must NOT leave the `docker compose` subprocess
        # running: cancelling ``communicate()`` only stops us reading, the process keeps going
        # and can create/remove containers AFTER the caller's rollback — leaking an unowned
        # stack that races the cleanup. Kill the whole process group + reap it before propagating
        # so any subsequent teardown runs against a settled node. (Was: only TimeoutError killed
        # the group; cancellation let the subprocess outlive rollback — audit H10.)
        _kill_process_group(proc)
        with suppress(BaseException):
            await proc.wait()
        raise
    return ShellResult(
        rc=proc.returncode or 0,
        stdout=out.decode(errors="replace") if out else "",
        stderr=err.decode(errors="replace") if err else "",
    )


def _kill_process_group(proc: asyncio.subprocess.Process) -> None:
    """SIGKILL the subprocess's whole process group (it's a session leader via
    ``start_new_session``); fall back to killing just the process if the group
    signal can't be sent (already-exited / no such group)."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        with suppress(ProcessLookupError):
            proc.kill()


class ComposeProjectRunner:
    """Runs one compose project up / down against the node's daemon.

    ``run`` defaults to a real subprocess; tests inject a fake. ``ensure_image``
    (optional) pulls each image via the node's image cache before ``up`` so
    ``docker compose`` never triggers its own uncached pull on the hot path.
    ``compose_bin`` is the compose entrypoint argv (default ``docker compose``).
    ``root_dir`` is where per-project temp dirs are created (default the system
    temp)."""

    def __init__(
        self,
        *,
        run: Runner | None = None,
        ensure_image: ImageEnsurer | None = None,
        compose_bin: Sequence[str] = ("docker", "compose"),
        root_dir: str | Path | None = None,
    ) -> None:
        self._run: Runner = run or _default_run
        self._ensure_image = ensure_image
        self._compose_bin = list(compose_bin)
        self._root_dir = str(root_dir) if root_dir is not None else None

    # ── public API ────────────────────────────────────────────────────────────

    async def up(
        self,
        *,
        project_name: str,
        compose_yaml: str,
        images: Sequence[str] = (),
        main_service: str = DEFAULT_MAIN_SERVICE,
        up_timeout_s: float = DEFAULT_UP_TIMEOUT_S,
        wait: bool = True,
    ) -> ComposeProjectRecord:
        """Write the compose doc, ensure images, ``up -d --wait``, resolve
        container IDs. On any failure the partial project is torn down before the
        error propagates so a failed acquire never leaks containers."""
        if not project_name:
            raise ComposeError("compose up: empty project_name")
        project_dir = self._make_project_dir(project_name)
        compose_file = str(Path(project_dir) / "docker-compose.yaml")
        Path(compose_file).write_text(compose_yaml)

        try:
            for ref in images:
                if self._ensure_image is not None:
                    await self._ensure_image(ref)

            up_argv = [
                *self._base_argv(project_name, project_dir, compose_file),
                "up", "-d", "--remove-orphans",
            ]
            if wait:
                up_argv.append("--wait")
            result = await self._run(up_argv, timeout_s=up_timeout_s)
            if result.rc != 0:
                logs = await self._safe_logs(project_name, project_dir, compose_file)
                raise ComposeError(
                    f"`docker compose up` failed for project {project_name!r} "
                    f"(exit={result.rc}): {_tail(result.stderr or result.stdout)}"
                    f"{logs}",
                )

            services = await self._resolve_services(
                project_name, project_dir, compose_file,
            )
            service_ids = {svc: cid for svc, cid, _ in services}
            if main_service not in service_ids:
                raise ComposeError(
                    f"compose project {project_name!r} has no {main_service!r} "
                    f"service after up (services: {sorted(service_ids)}). Harbor "
                    f"execs the agent into {main_service!r}; the rewritten compose "
                    f"must define it.",
                )
            main_id = service_ids[main_service]
            main_name = next(
                (name for svc, _, name in services if svc == main_service), "",
            ) or main_id
            return ComposeProjectRecord(
                project_name=project_name,
                project_dir=project_dir,
                main_container_id=main_id,
                main_container_name=main_name,
                service_container_ids=service_ids,
            )
        except BaseException:
            # Teardown the partial project; never leak on a failed acquire.
            await self._safe_down(project_name, project_dir, compose_file)
            raise

    async def down(
        self,
        *,
        project_name: str,
        project_dir: str,
        timeout_s: float = DEFAULT_DOWN_TIMEOUT_S,
    ) -> None:
        """``down -v --remove-orphans`` the whole project — **strict**.

        Unlike the best-effort teardown in the failed-``up`` path
        (:meth:`_safe_down`), an explicit destroy must be node-*confirmed*:
        capacity is released only on a confirmed destroy (spec-00 invariant 2). So
        a non-zero / timed-out ``docker compose down`` **raises** ``ComposeError``
        rather than reporting success while containers may still be running — the
        caller keeps the project registered and its capacity reserved until a
        retry (or the raw-GC reaper) confirms teardown. Idempotent for the benign
        case: ``docker compose down`` on an already-absent project returns 0. Only
        on success is the temp project dir removed (a failure keeps it so a retry
        can re-issue ``down`` with the same compose file)."""
        compose_file = str(Path(project_dir) / "docker-compose.yaml")
        argv = [
            *self._base_argv(project_name, project_dir, compose_file),
            "down", "-v", "--remove-orphans",
        ]
        result = await self._run(argv, timeout_s=timeout_s)
        if result.rc != 0:
            raise ComposeError(
                f"`docker compose down` failed for project {project_name!r} "
                f"(exit={result.rc}): {_tail(result.stderr or result.stdout)}. "
                f"Project left registered — capacity is released only on a "
                f"confirmed teardown.",
            )
        shutil.rmtree(project_dir, ignore_errors=True)

    # ── internals ─────────────────────────────────────────────────────────────

    def _make_project_dir(self, project_name: str) -> str:
        return tempfile.mkdtemp(
            prefix=f"xrlenv-compose-{project_name}-", dir=self._root_dir,
        )

    def _base_argv(
        self, project_name: str, project_dir: str, compose_file: str,
    ) -> list[str]:
        return [
            *self._compose_bin,
            "-p", project_name,
            "--project-directory", project_dir,
            "-f", compose_file,
        ]

    async def _resolve_services(
        self, project_name: str, project_dir: str, compose_file: str,
    ) -> list[tuple[str, str, str]]:
        """Return ``[(service, full_container_id, name), …]`` for the project.

        ``docker compose ps --format json`` reports the **short** (12-char) id, but
        the raw-GC node-truth diff compares the CP's session container id against
        the node's ``list_raw_containers``, which reports the **full** 64-char
        ``container.id``. A short/full mismatch would classify ``main`` as *both* a
        node-only orphan and ``coordinator_only`` (§2.3 of the step-3 plan). So we
        discover the containers via ``ps`` then upgrade to full ids (+ names) in one
        ``docker inspect``, keyed by the compose service label — order-independent."""
        ps_argv = [
            *self._base_argv(project_name, project_dir, compose_file),
            "ps", "--format", "json", "-a",
        ]
        ps = await self._run(ps_argv, timeout_s=60.0)
        if ps.rc != 0:
            raise ComposeError(
                f"`docker compose ps` failed for {project_name!r} "
                f"(exit={ps.rc}): {_tail(ps.stderr)}",
            )
        short_ids = [
            str(entry.get("ID") or entry.get("Id"))
            for entry in _parse_ps_json(ps.stdout)
            if entry.get("ID") or entry.get("Id")
        ]
        if not short_ids:
            raise ComposeError(
                f"compose project {project_name!r}: `ps` returned no containers "
                f"(stdout: {_tail(ps.stdout)})",
            )
        # ``|`` is a safe separator: hex ids, ``/name``, and compose service names
        # ([a-zA-Z0-9._-]) never contain it. One line per inspected container.
        tmpl = (
            '{{.Id}}|{{.Name}}|'
            '{{index .Config.Labels "com.docker.compose.service"}}'
        )
        insp = await self._run(
            ["docker", "inspect", "-f", tmpl, *short_ids], timeout_s=60.0,
        )
        if insp.rc != 0:
            raise ComposeError(
                f"`docker inspect` for full ids failed for {project_name!r} "
                f"(exit={insp.rc}): {_tail(insp.stderr)}",
            )
        services: list[tuple[str, str, str]] = []
        for line in insp.stdout.splitlines():
            parts = line.strip().split("|")
            if len(parts) != 3:
                continue
            full_id, name, service = parts
            if full_id and service:
                services.append((service, full_id, name.lstrip("/")))
        if not services:
            raise ComposeError(
                f"compose project {project_name!r}: could not resolve full "
                f"container ids (inspect stdout: {_tail(insp.stdout)})",
            )
        return services

    async def _safe_logs(
        self, project_name: str, project_dir: str, compose_file: str,
    ) -> str:
        """Capture a short tail of the project's logs for an ``up`` failure
        diagnostic. Never raises — diagnostics must not mask the real error."""
        try:
            argv = [
                *self._base_argv(project_name, project_dir, compose_file),
                "logs", "--no-color", "--tail", "40",
            ]
            result = await self._run(argv, timeout_s=30.0)
            body = _tail(result.stdout or result.stderr, limit=1500)
            return f"\n--- compose logs (tail) ---\n{body}" if body else ""
        except Exception:
            return ""

    async def _safe_down(
        self,
        project_name: str,
        project_dir: str,
        compose_file: str,
        *,
        timeout_s: float = DEFAULT_DOWN_TIMEOUT_S,
    ) -> None:
        """Best-effort ``down`` + temp-dir removal; swallows errors so teardown is
        idempotent and never re-raises into a caller already handling failure."""
        try:
            argv = [
                *self._base_argv(project_name, project_dir, compose_file),
                "down", "-v", "--remove-orphans",
            ]
            await self._run(argv, timeout_s=timeout_s)
        except Exception:
            pass
        finally:
            shutil.rmtree(project_dir, ignore_errors=True)


def _parse_ps_json(stdout: str) -> list[dict[str, Any]]:
    """Parse ``docker compose ps --format json`` output, tolerating both the
    JSON-array form and the newline-delimited-objects form."""
    text = stdout.strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [e for e in parsed if isinstance(e, dict)]
        if isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        pass
    entries: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            entries.append(obj)
    return entries


def _tail(text: str | None, limit: int = 800) -> str:
    if not text:
        return ""
    text = text.strip()
    return text if len(text) <= limit else "…" + text[-limit:]
