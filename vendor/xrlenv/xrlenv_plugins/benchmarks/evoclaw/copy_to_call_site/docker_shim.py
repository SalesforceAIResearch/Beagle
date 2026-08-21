"""In-process ``docker`` CLI interceptor → xrlenv compat client.

EvoClaw drives Docker through raw ``subprocess`` calls to the ``docker`` CLI
(see SHIM-SURFACE.md). xrlenv's ``from_env()`` docker-py drop-in cannot intercept
those. This module monkeypatches ``subprocess.run`` / ``subprocess.Popen`` so any
argv whose program is ``docker`` is routed to an xrlenv (or real docker-py)
client instead of the real binary; everything else passes through untouched.

Because the compat client's cluster sessions are bound to the creating process
(``xrlenv/compat/docker_client.py``), this only works **in-process** — install it
in the same Python process that runs EvoClaw's orchestrator (see
``run_e2e_xrlenv.py``). Container identity is tracked here by EvoClaw's
``--name`` in an in-process ``name → Container`` registry.

Covered surface and per-flag decisions are specified in SHIM-SURFACE.md. Anything
outside that surface fails loud rather than silently dropping a cap.
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import shlex
import subprocess
import tarfile
import threading
import time
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("xrlenv.evoclaw.docker_shim")

# ---- module state, set by install() ----------------------------------------
_CLIENT: Any = None
_NAME_PREFIX: str = ""
_LABELS: dict[str, str] = {}
_REGISTRY: dict[str, Any] = {}  # EvoClaw's --name -> docker-py Container
# Per-container output binds: `-v host:ctr` whose host side was EMPTY at create
# (an output dir the container fills, e.g. the evaluator's `-v output_dir:/output`).
# A real bind is live; here we approximate by copying ctr -> host after each exec
# so the host sees what the container wrote. Non-empty host binds are input-only.
_OUTPUT_BINDS: dict[str, list[tuple[str, str]]] = {}
_REAL_RUN = subprocess.run
_REAL_POPEN = subprocess.Popen
_INSTALLED = False
_LOCK = threading.RLock()

# ---- fleet reservation (opt-in) --------------------------------------------
# EvoClaw schedules a FLEET of containers per task: one long-lived agent
# container (the first `docker run`, the base image) plus one or more heavier
# per-milestone evaluation containers (later `docker run`s, `--cpus 16`). Under
# a naive scheduler the greedy admission of many agents starves the evals. When
# fleet reservation is enabled (both footprint values supplied to install()),
# the shim declares the task's PEAK footprint to xrlenv via three generic
# Docker labels on the FIRST container so the control plane reserves the whole
# footprint on one node up front; later containers carry only the fleet_id and
# draw from that reservation. This is the ENTIRE EvoClaw-side surface of the
# feature — xrlenv-core stays consumer-agnostic (it only reads these labels).
#
# The three label keys are xrlenv's generic contract (see xrlenv
# compat.metadata + spec 21); they are NOT EvoClaw-specific.
_LABEL_FLEET_ID = "xrlenv.fleet_id"
_LABEL_FLEET_CPU_REQUEST = "xrlenv.fleet_cpu_request"
_LABEL_FLEET_MEM_REQUEST = "xrlenv.fleet_mem_request"

# fleet_id for this process's task (each sweep task is one process — agent +
# its evals share it), or None when fleet reservation is disabled.
_FLEET_ID: str | None = None
# The declared PEAK footprint the fleet reserves (task-level, NOT any single
# container's own --cpus/--memory). Supplied explicitly by the operator — the
# shim invents no default.
_FLEET_CPU_REQUEST: float | None = None
_FLEET_MEM_REQUEST_BYTES: int | None = None
# Set true once the fleet-opening (first) container has been launched, so every
# later container in this process is a companion (fleet_id only). Guarded by
# _LOCK so the opener is unambiguous even if the first evals race.
_FLEET_OPENED: bool = False


class DockerShimError(RuntimeError):
    """A docker invocation outside the covered surface (fails loud)."""


# ---- transient cluster-loss retry ------------------------------------------
# A control-plane restart / node drop surfaces as an xrlenv error (rehydrated
# from gRPC UNAVAILABLE) mid-call. We retry the cluster call in place with
# bounded backoff (outlasting a typical CP restart); if that's exhausted we
# re-raise as ConnectionError — an OSError subclass, which EvoClaw's own
# evaluator classifies as transient (orchestrator.py) and re-runs the whole
# eval against a fresh container. Tuned via install() from the wrapper's
# --cluster-retries / --cluster-retry-base-s / --mem-per-cpu-gb flags.
_RETRY_ATTEMPTS = 4
_RETRY_BASE_S = 3.0
# Memory (GiB) to request per CPU for an undeclared container (0 = cluster
# default). Set by install(); see _effective_mem.
_MEM_PER_CPU_GB = 2.0
# xrlenv.errors names that mean "cluster blipped, the endpoint may recover".
_TRANSIENT_TYPES = frozenset({
    "ControlPlaneLost", "NodeLost", "NodeCommandTimeout",
    "SessionExpired", "SessionDegraded",
})
# Message markers for raw gRPC / transport errors not wrapped in those types.
_TRANSIENT_MARKERS = (
    "Cancelling all calls", "Connection refused", "UNAVAILABLE",
    "failed to connect", "disconnected before reply", "Connection reset",
)


def _is_transient_cluster_error(exc: BaseException) -> bool:
    """True if ``exc`` looks like a recoverable CP/node blip (by xrlenv error
    type name or transport message) — matched without importing xrlenv so unit
    tests can exercise it with plain exceptions."""
    if _TRANSIENT_TYPES & {c.__name__ for c in type(exc).__mro__}:
        return True
    msg = str(exc)
    return any(m in msg for m in _TRANSIENT_MARKERS)


def _with_retry(what: str, fn: Any) -> Any:
    """Call ``fn()``, retrying transient cluster errors with exponential backoff.

    Non-transient errors propagate immediately (unchanged). On exhaustion, raise
    ``ConnectionError`` so EvoClaw's own eval retry (which treats OSError as
    transient) re-runs the evaluation from a fresh container.
    """
    last: BaseException | None = None
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            return fn()
        except Exception as exc:
            if not _is_transient_cluster_error(exc):
                raise
            last = exc
            if attempt == _RETRY_ATTEMPTS - 1:
                break
            delay = _RETRY_BASE_S * (2 ** attempt)
            LOGGER.warning(
                "transient cluster error on %s (attempt %d/%d): %s; retrying in %.0fs",
                what, attempt + 1, _RETRY_ATTEMPTS, exc, delay,
            )
            time.sleep(delay)
    raise ConnectionError(
        f"transient cluster loss on {what} after {_RETRY_ATTEMPTS} attempt(s): {last}"
    ) from last


# ---- install / uninstall ----------------------------------------------------
def install(
    *,
    client: Any = None,
    name_prefix: str = "",
    labels: dict[str, str] | None = None,
    cluster_retries: int = 4,
    cluster_retry_base_s: float = 3.0,
    container_mem_per_cpu_gb: float = 2.0,
    fleet_id: str | None = None,
    fleet_cpu_request: float | None = None,
    fleet_mem_request_bytes: int | None = None,
) -> None:
    """Patch subprocess so ``docker`` argv routes to ``client``.

    Args:
        client: a docker-py-compatible client (``xrlenv.from_env()`` or the real
            ``docker.from_env()`` for local tests, or a fake in unit tests). If
            None, built from the environment via :func:`client_from_env`.
        name_prefix: prepended to EvoClaw's ``--name`` for the *actual* container
            name, so concurrent rollouts don't collide on a fixed name. EvoClaw
            still refers to its own unprefixed name; the registry maps it.
        labels: default labels merged onto every ``docker run`` (e.g. task_key).
        fleet_id: enable fleet reservation for this process's task. When set,
            the FIRST container gets ``xrlenv.fleet_id`` + the footprint labels
            (fleet opener) and every later container gets ``xrlenv.fleet_id``
            only (companion). ``None`` disables fleet reservation entirely
            (legacy: each container admitted independently).
        fleet_cpu_request / fleet_mem_request_bytes: the fleet's declared PEAK
            footprint (task-level cpu / memory). REQUIRED together with
            ``fleet_id`` — the shim invents no default. Passing ``fleet_id``
            without both is a ``DockerShimError`` (fail loud, no silent
            partial footprint).
    """
    global _CLIENT, _NAME_PREFIX, _LABELS, _REAL_RUN, _REAL_POPEN, _INSTALLED
    global _RETRY_ATTEMPTS, _RETRY_BASE_S, _MEM_PER_CPU_GB
    global _FLEET_ID, _FLEET_CPU_REQUEST, _FLEET_MEM_REQUEST_BYTES, _FLEET_OPENED
    if fleet_id is not None and (
        fleet_cpu_request is None or fleet_mem_request_bytes is None
    ):
        raise DockerShimError(
            "docker_shim.install: fleet_id set but the footprint is incomplete "
            f"(fleet_cpu_request={fleet_cpu_request!r}, "
            f"fleet_mem_request_bytes={fleet_mem_request_bytes!r}); a fleet "
            "must declare BOTH cpu and memory. No silent default.",
        )
    with _LOCK:
        if _INSTALLED:
            raise DockerShimError("docker_shim already installed")
        _CLIENT = client if client is not None else client_from_env()
        _NAME_PREFIX = name_prefix
        _LABELS = dict(labels or {})
        _RETRY_ATTEMPTS = max(1, int(cluster_retries))
        _RETRY_BASE_S = max(0.0, float(cluster_retry_base_s))
        _MEM_PER_CPU_GB = max(0.0, float(container_mem_per_cpu_gb))
        _FLEET_ID = fleet_id
        _FLEET_CPU_REQUEST = fleet_cpu_request
        _FLEET_MEM_REQUEST_BYTES = fleet_mem_request_bytes
        _FLEET_OPENED = False
        _REAL_RUN = subprocess.run
        _REAL_POPEN = subprocess.Popen
        subprocess.run = _patched_run
        subprocess.Popen = _PatchedPopen
        _INSTALLED = True
        LOGGER.info(
            "docker_shim installed (name_prefix=%r, fleet=%s)",
            name_prefix,
            (
                f"{fleet_id} cpu={fleet_cpu_request} "
                f"mem_bytes={fleet_mem_request_bytes}"
                if fleet_id is not None else "off"
            ),
        )


def uninstall() -> None:
    """Restore the real subprocess functions and forget the registry."""
    global _INSTALLED, _FLEET_ID, _FLEET_CPU_REQUEST
    global _FLEET_MEM_REQUEST_BYTES, _FLEET_OPENED
    with _LOCK:
        subprocess.run = _REAL_RUN
        subprocess.Popen = _REAL_POPEN
        _REGISTRY.clear()
        _OUTPUT_BINDS.clear()
        _FLEET_ID = None
        _FLEET_CPU_REQUEST = None
        _FLEET_MEM_REQUEST_BYTES = None
        _FLEET_OPENED = False
        _INSTALLED = False
        LOGGER.info("docker_shim uninstalled")


def _fleet_labels_for_next_container() -> dict[str, str]:
    """The fleet labels to stamp on the NEXT container launched by this process.

    Returns ``{}`` when fleet reservation is disabled. Otherwise, under
    ``_LOCK``: the FIRST call returns the full opener declaration
    (``xrlenv.fleet_id`` + ``xrlenv.fleet_cpu_request`` +
    ``xrlenv.fleet_mem_request``) and flips ``_FLEET_OPENED``; every later call
    returns just ``{xrlenv.fleet_id: ...}`` (companion). The check-and-set is
    atomic so exactly one container opens the fleet even if the first evals
    race the agent (they don't in EvoClaw's flow — the agent container is
    created first — but the lock makes it robust regardless).
    """
    global _FLEET_OPENED
    if _FLEET_ID is None:
        return {}
    with _LOCK:
        if not _FLEET_OPENED:
            _FLEET_OPENED = True
            # Opener: declare the whole task footprint. cpu is a float string,
            # mem is an integer byte string — the exact shapes xrlenv's
            # _parse_fleet_labels expects.
            return {
                _LABEL_FLEET_ID: _FLEET_ID,
                _LABEL_FLEET_CPU_REQUEST: repr(float(_FLEET_CPU_REQUEST or 0.0)),
                _LABEL_FLEET_MEM_REQUEST: str(int(_FLEET_MEM_REQUEST_BYTES or 0)),
            }
        # Companion: only the fleet_id; the footprint was declared by the opener.
        return {_LABEL_FLEET_ID: _FLEET_ID}


def cleanup_containers() -> None:
    """Force-remove every container still in the registry.

    A safety net: EvoClaw removes its own containers (agent via
    ``--remove-container``, eval + golden in their own ``finally`` blocks), but a
    crash/interrupt before that teardown would otherwise leak cluster containers
    until the watchdog reaps them. Idempotent; empties the registry.
    """
    with _LOCK:
        names = list(_REGISTRY)
    for name in names:
        c = _REGISTRY.pop(name, None)
        if c is None:
            continue
        try:
            c.remove(force=True)
            LOGGER.info("cleanup: removed leaked container %s", name)
        except Exception as exc:
            LOGGER.debug("cleanup remove %s: %s", name, exc)


def client() -> Any:
    """The installed client (for host-side helpers like golden extraction)."""
    if _CLIENT is None:
        raise DockerShimError("docker_shim not installed")
    return _CLIENT


def client_from_env() -> Any:
    """Build an xrlenv docker-py-compat client from XRLENV_* env vars."""
    import xrlenv  # local import: only needed when not injected

    host = os.environ.get("XRLENV_GRPC_HOST")
    if not host:
        raise DockerShimError(
            "XRLENV_GRPC_HOST is required to build the xrlenv client; set "
            "XRLENV_GRPC_HOST/PORT/CONSUMER_TOKEN or pass client= explicitly.",
        )
    return xrlenv.from_env()


# ---- docker detection -------------------------------------------------------
def _docker_argv(args: Any, shell: bool) -> list[str] | None:
    """Return the docker token list if ``args`` is a docker invocation, else None."""
    if shell and isinstance(args, str):
        toks = shlex.split(args)
    elif isinstance(args, (list, tuple)) and args:
        toks = [str(a) for a in args]
    else:
        return None
    if toks and os.path.basename(toks[0]) == "docker":
        return toks
    return None


# ---- patched subprocess.run -------------------------------------------------
def _patched_run(args: Any = None, **kw: Any) -> subprocess.CompletedProcess[Any]:
    toks = _docker_argv(args, bool(kw.get("shell")))
    if toks is None:
        return _REAL_RUN(args, **kw)

    text = bool(kw.get("text") or kw.get("universal_newlines") or kw.get("encoding"))
    stdout_target = kw.get("stdout")
    check = bool(kw.get("check"))
    rc, out, err = _dispatch(toks[1:], stdout_target=stdout_target)

    # If caller asked stdout to go to a file object, we already wrote bytes there.
    wrote_file = hasattr(stdout_target, "write") and stdout_target not in (
        subprocess.PIPE,
        subprocess.STDOUT,
        subprocess.DEVNULL,
    )
    cp_out: Any = b"" if wrote_file else out
    cp_err: Any = err
    if text:
        cp_out = "" if wrote_file else out.decode(errors="replace")
        cp_err = err.decode(errors="replace")
    result = subprocess.CompletedProcess(toks, rc, cp_out, cp_err)
    if check and rc != 0:
        raise subprocess.CalledProcessError(rc, toks, cp_out, cp_err)
    return result


# ---- patched subprocess.Popen (streaming exec) ------------------------------
class _PatchedPopen:
    """Stands in for ``subprocess.Popen(["docker", ...])``.

    Only the attributes EvoClaw's ``_execute_with_streaming`` uses are
    implemented: ``stdout``/``stderr`` (line-readable, real OS pipes so reader
    threads block then drain), ``wait(timeout)``, ``kill()``, ``returncode``.
    Non-docker Popen calls fall through to the real class.
    """

    def __new__(cls, args: Any = None, **kw: Any) -> Any:
        toks = _docker_argv(args, bool(kw.get("shell")))
        if toks is None:
            return _REAL_POPEN(args, **kw)
        self = object.__new__(cls)
        self._init_docker(toks, **kw)
        return self

    def __init__(self, *_args: Any, **_kw: Any) -> None:
        # __new__ already initialised the docker path; real-Popen fall-through
        # returns a different type so this never runs for it.
        pass

    def _init_docker(self, toks: list[str], **kw: Any) -> None:
        text = bool(kw.get("text") or kw.get("universal_newlines") or kw.get("encoding"))
        r_out, w_out = os.pipe()
        r_err, w_err = os.pipe()
        mode = "r" if text else "rb"
        self.stdout = os.fdopen(r_out, mode, buffering=1 if text else -1)
        self.stderr = os.fdopen(r_err, mode, buffering=1 if text else -1)
        self.returncode: int | None = None
        self._toks = toks
        self._text = text
        self._done = threading.Event()

        def _worker() -> None:
            rc, out, err = 1, b"", b""
            try:
                rc, out, err = _dispatch(toks[1:], stdout_target=None)
            except Exception as exc:
                err = f"docker_shim error: {exc}".encode()
                rc = 1
            finally:
                for fd, data in ((w_out, out), (w_err, err)):
                    with contextlib.suppress(OSError):
                        os.write(fd, data.encode() if isinstance(data, str) else data)
                    os.close(fd)
                self.returncode = rc
                self._done.set()

        self._thread = threading.Thread(target=_worker, name="docker-shim-exec", daemon=True)
        self._thread.start()

    def wait(self, timeout: float | None = None) -> int:
        if not self._done.wait(timeout=timeout):
            raise subprocess.TimeoutExpired(self._toks, timeout or 0)
        self._thread.join()
        assert self.returncode is not None
        return self.returncode

    def kill(self) -> None:
        # Best-effort: the underlying exec is not externally cancellable here;
        # mark done so readers unblock. (Oracle exec is short-lived.)
        self.returncode = self.returncode if self.returncode is not None else -9

    def poll(self) -> int | None:
        return self.returncode

    def __enter__(self) -> _PatchedPopen:
        return self

    def __exit__(self, *exc: Any) -> None:
        with contextlib.suppress(Exception):
            self.wait()


# ---- dispatch ---------------------------------------------------------------
def _dispatch(rest: list[str], *, stdout_target: Any) -> tuple[int, bytes, bytes]:
    """Route ``docker <rest...>`` to the client. Returns (rc, stdout, stderr)."""
    if not rest:
        return 1, b"", b"docker: no subcommand\n"
    sub = rest[0]
    args = rest[1:]
    if sub == "image" and args and args[0] == "inspect":
        return _image_inspect(args[1:])
    handler = {
        "run": _run,
        "exec": _exec,
        "cp": _cp,
        "rm": _rm,
        "stop": _stop,
        "start": _start,
        "kill": _kill,
        "inspect": _inspect,
        "ps": _ps,
        "images": _images,
    }.get(sub)
    if handler is None:
        # build / rmi / pull / tag / ... — out of v1 scope: fail loud.
        msg = f"docker_shim: '{sub}' is not covered (SHIM-SURFACE.md v1 scope)\n"
        LOGGER.error(msg.strip())
        return 127, b"", msg.encode()
    if sub == "exec":
        return _exec(args, stdout_target=stdout_target)
    return handler(args)


def _lookup(name: str) -> Any:
    c = _REGISTRY.get(name)
    if c is None:
        raise DockerShimError(
            f"docker_shim: no container named {name!r} in registry "
            f"(known: {sorted(_REGISTRY)})",
        )
    return c


# ---- run --------------------------------------------------------------------
def _run(args: list[str]) -> tuple[int, bytes, bytes]:
    detached = False
    name: str | None = None
    workdir: str | None = None
    env: dict[str, str] = {}
    cap_add: list[str] = []
    nano_cpus: int | None = None
    mem_bytes: int | None = None
    binds: list[tuple[str, str, bool]] = []  # (host, ctr, read_only)
    i = 0
    n = len(args)
    image: str | None = None
    command: list[str] = []
    while i < n:
        a = args[i]
        if a in ("-d", "--detach"):
            detached = True
            i += 1
        elif a in ("--init", "--rm"):
            i += 1  # lifecycle: session model owns this
        elif a == "--name":
            name = args[i + 1]
            i += 2
        elif a in ("-w", "--workdir"):
            workdir = args[i + 1]
            i += 2
        elif a in ("-e", "--env"):
            k, _, v = args[i + 1].partition("=")
            env[k] = v
            i += 2
        elif a in ("-v", "--volume"):
            # host bind mount: a cluster node has no host path, so we approximate.
            # host:ctr[:mode] — put existing host content in (input); a read-write
            # mount (no `:ro`) is also synced ctr->host after each exec (output).
            spec = args[i + 1].split(":")
            if len(spec) >= 2:
                read_only = len(spec) >= 3 and "ro" in spec[2].split(",")
                binds.append((spec[0], spec[1], read_only))
            i += 2
        elif a == "--cpus":
            nano_cpus = int(float(args[i + 1]) * 1e9)
            i += 2
        elif a in ("--memory", "-m"):
            mem_bytes = _parse_mem(args[i + 1])
            i += 2
        elif a.startswith("--memory="):
            mem_bytes = _parse_mem(a.split("=", 1)[1])
            i += 1
        elif a.startswith("--cap-add="):
            cap_add.append(a.split("=", 1)[1])
            i += 1
        elif a == "--cap-add":
            cap_add.append(args[i + 1])
            i += 2
        elif a == "--ulimit":
            LOGGER.debug("ignoring --ulimit %s", args[i + 1])
            i += 2
        elif a == "--sysctl":
            LOGGER.debug("ignoring --sysctl %s", args[i + 1])
            i += 2
        elif a.startswith("--add-host"):
            i += 1 if "=" in a else 2  # ignore host-gateway
        elif a == "--network":
            LOGGER.debug("ignoring --network %s", args[i + 1])
            i += 2
        elif a.startswith("-"):
            raise DockerShimError(f"docker_shim: unhandled `run` flag {a!r}")
        else:
            image = a
            command = args[i + 1:]
            break
    if image is None:
        return 1, b"", b"docker run: no image\n"

    # Fleet reservation (opt-in): the first container of this process opens the
    # fleet (footprint labels); later ones are companions (fleet_id only). No-op
    # ({}) when fleet reservation is disabled, so the labels are exactly _LABELS.
    _run_labels = dict(_LABELS)
    _run_labels.update(_fleet_labels_for_next_container())
    run_kwargs: dict[str, Any] = {
        "environment": env or None,
        "labels": _run_labels,
        "working_dir": workdir,
    }
    if cap_add:
        run_kwargs["cap_add"] = cap_add
    if nano_cpus:
        run_kwargs["nano_cpus"] = nano_cpus
    # Memory: EvoClaw declares none ("use host memory freely"), but xrlenv caps an
    # undeclared container at a small default (4 GiB) — memory-heavy suites
    # (element-web's 16-worker jest) OOM (exit 137). Forward an explicit limit:
    # EvoClaw's --memory if set, else scale with the CPU request to match the
    # workload (default 2 GiB/CPU; tune EVOCLAW_CONTAINER_MEM_PER_CPU_GB, 0=off).
    mem = _effective_mem(mem_bytes, nano_cpus)
    if mem:
        run_kwargs["mem_limit"] = mem

    # Docker-in-Docker via Sysbox (opt-in): request the sysbox-runc OCI runtime
    # so `testcontainers`-based suites find an inner Docker runtime — e.g.
    # element-web's E2E tests spin up Dendrite/Pinecone Matrix homeservers as
    # nested containers (without it: "Could not find a working container runtime
    # strategy"). This REPLACES the old host-socket bind (a container-escape
    # stop-gap): sysbox gives each container its OWN user namespace + inner
    # dockerd, so nothing on the node is shared into the container and inner root
    # is not host root. Requires a Sysbox node pool (xrlenv_plugins/sysbox/) with
    # sysbox-runc in nodes.yaml policy.allowed_runtimes. Read from
    # EVOCLAW_CONTAINER_RUNTIME — the batch driver (run_all_xrlenv.py) sets this
    # per-worker ONLY for milestones passed via --sysbox-milestone, so DinD tasks
    # route to the small sysbox pool while everything else stays on the runc pool.
    # Unset = docker's default runc.
    _runtime = os.environ.get("EVOCLAW_CONTAINER_RUNTIME", "").strip()
    if _runtime:
        run_kwargs["runtime"] = _runtime

    # Sysbox: provide binds as REAL mounts instead of the copy-emulation below.
    # `docker cp` / `put_archive` INTO a sysbox container corrupts its /etc
    # (openat "file exists" → EvoClaw's fakeroot provisioning fails; see the
    # xrlenv sysbox runbook). Real bind mounts are unaffected. The mounts flow
    # via docker-py volumes → host_config Binds → xrlenv acquire_container(binds=)
    # → a real node bind, gated by the node's allowed_host_paths policy (which
    # must include the host-path prefix). Scoped to sysbox only — the runc path
    # keeps the byte-for-byte copy emulation. Real rw mounts also write straight
    # to the host, so no ctr→host output-sync is registered.
    _sysbox = _runtime == "sysbox-runc"
    if _sysbox and binds:
        _vols = run_kwargs.setdefault("volumes", {})
        for host, ctr, ro in binds:
            _vols[host] = {"bind": ctr, "mode": "ro" if ro else "rw"}
            # Sysbox can't uid-shift bind mounts on a network FS (our /shared-fs):
            # idmapped mounts don't apply there, so the container sees the tree
            # as `nobody` and writes as an unmapped subuid → a normally-owned
            # host dir rejects writes ("Permission denied"). Make each WRITABLE
            # bind world-writable so the mapped uid can write. Read-only binds
            # (e.g. /golden) don't need it — 0644/0755 is world-readable. Files
            # the container creates end up owned by the sysbox subuid; the host
            # can still READ them (0644) for result collection but needs root to
            # delete them (dev-pool cleanup caveat).
            if not ro:
                _chmod_world_writable(host)

    if detached:
        prefixed = f"{_NAME_PREFIX}{name}" if name else None
        container = _with_retry(
            f"run {name or image}",
            lambda: _CLIENT.containers.run(
                image, command or None, name=prefixed, detach=True, **_clean(run_kwargs)
            ),
        )
        if name:
            _REGISTRY[name] = container
        for host, ctr, ro in binds:
            if _sysbox:
                continue  # real mount (above) — no copy-in, no output-sync
            try:
                hp = Path(host)
                if hp.is_file() or (hp.is_dir() and any(hp.iterdir())):
                    _put_path(container, hp, ctr)  # existing host content = input
                else:
                    container.exec_run(["mkdir", "-p", ctr])
                # A read-write mount (no :ro) can be written by the container —
                # sync ctr -> host after each exec so the host sees it (e.g.
                # the evaluator's /output test reports). :ro mounts are input-only.
                if not ro and name:
                    _OUTPUT_BINDS.setdefault(name, []).append((host, ctr))
            except Exception as exc:
                LOGGER.warning("bind %s -> %s failed: %s", host, ctr, exc)
        cid = getattr(container, "id", "") or ""
        return 0, (cid + "\n").encode(), b""

    # one-shot: run to completion, auto-remove, return logs
    try:
        logs = _CLIENT.containers.run(
            image, command or None, detach=False, remove=True, **_clean(run_kwargs)
        )
        out = logs if isinstance(logs, bytes) else str(logs).encode()
        return 0, out, b""
    except Exception as exc:
        rc = int(getattr(exc, "exit_status", 1) or 1)
        err = getattr(exc, "stderr", None)
        if err is None:
            err = str(exc)
        return rc, b"", err if isinstance(err, bytes) else str(err).encode()


def _clean(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


def _parse_mem(s: str) -> int:
    """Parse a docker ``--memory`` value ('4g', '512m', '4294967296') to bytes."""
    s = s.strip().lower()
    for suf, mult in (("gb", 1024**3), ("g", 1024**3), ("mb", 1024**2),
                      ("m", 1024**2), ("kb", 1024), ("k", 1024), ("b", 1)):
        if s.endswith(suf):
            return int(float(s[: -len(suf)]) * mult)
    return int(float(s))


def _effective_mem(declared: int | None, nano_cpus: int | None) -> int | None:
    """Memory limit to request on acquire: the harness's ``--memory`` if it set
    one, else scale with the CPU request so memory-heavy test suites don't inherit
    xrlenv's small undeclared-container default and OOM. None → cluster default."""
    if declared:
        return declared
    if not nano_cpus:
        return None
    if _MEM_PER_CPU_GB <= 0:
        return None
    return int((nano_cpus / 1e9) * _MEM_PER_CPU_GB * (1024**3))


def _chmod_world_writable(host: str) -> None:
    """Make a writable host bind dir (and existing contents) world-writable so a
    sysbox container can write to it on a network FS (/shared-fs) where sysbox can't
    uid-shift the mount. Best-effort; recurses so pre-existing files the
    container must modify are writable too. Only called on the sysbox path."""
    try:
        p = Path(host)
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)
        os.chmod(p, 0o777)
        if p.is_dir():
            for sub in p.rglob("*"):
                with contextlib.suppress(OSError):
                    os.chmod(sub, 0o777)
    except OSError as exc:
        LOGGER.warning("sysbox bind chmod %s failed: %s", host, exc)


# name -> "uid:gid" resolution cache, keyed by (container_id, name). See
# _resolve_sysbox_user.
_UID_CACHE: dict[tuple[str, str], str] = {}


def _resolve_sysbox_user(container: Any, name: str) -> str:
    """Translate a `docker exec --user <NAME>` into `--user <uid>:<gid>` for
    sysbox containers.

    Under sysbox-runc, a user added to /etc/passwd at RUNTIME (EvoClaw's init
    appends `fakeroot` after container start) is NOT resolvable by
    `docker exec --user <name>` — the runtime resolves the name against a
    passwd view that predates the exec-time append, so it fails with
    "unable to find user <name>: no matching entries in passwd file". But
    `--user <uid>:<gid>` (numeric) works AND still resolves to the name inside
    the container (whoami/id are correct). A plain exec (getent) DOES see the
    appended entry, so we resolve the name -> numeric there and pass numeric.
    Result cached per (container, name). Falls back to the name on any failure
    (no worse than today). Only called on the sysbox path."""
    if not name or name.replace(":", "").isdigit():
        return name  # already numeric (uid or uid:gid)
    # Guard against anything that isn't a plain user name.
    if not all(c.isalnum() or c in "_-." for c in name):
        return name
    cid = getattr(container, "id", "") or name
    key = (cid, name)
    cached = _UID_CACHE.get(key)
    if cached is not None:
        return cached
    numeric = name
    try:
        # getent first; fall back to a raw /etc/passwd scan for minimal images.
        res = container.exec_run(
            ["sh", "-c", f"getent passwd {name} 2>/dev/null || grep '^{name}:' /etc/passwd"],
            demux=False,
        )
        out = getattr(res, "output", res[1] if isinstance(res, tuple) else b"") or b""
        text = out.decode(errors="replace") if isinstance(out, bytes) else str(out)
        line = text.strip().splitlines()
        if line:
            parts = line[0].split(":")
            if len(parts) >= 4 and parts[2].isdigit() and parts[3].isdigit():
                numeric = f"{parts[2]}:{parts[3]}"
    except Exception as exc:
        LOGGER.debug("sysbox user resolve %s failed: %s", name, exc)
    _UID_CACHE[key] = numeric
    return numeric


# ---- exec -------------------------------------------------------------------
def _exec(args: list[str], *, stdout_target: Any = None) -> tuple[int, bytes, bytes]:
    user: str | None = None
    workdir: str | None = None
    env: dict[str, str] = {}
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("--user", "-u"):
            user = args[i + 1]
            i += 2
        elif a in ("-w", "--workdir"):
            workdir = args[i + 1]
            i += 2
        elif a in ("-e", "--env"):
            k, _, v = args[i + 1].partition("=")
            env[k] = v
            i += 2
        elif a in ("-i", "-t", "-it", "--interactive", "--tty", "--detach", "-d"):
            i += 1
        elif a.startswith("-"):
            raise DockerShimError(f"docker_shim: unhandled `exec` flag {a!r}")
        else:
            break
    name = args[i]
    command = args[i + 1:]
    try:
        container = _lookup(name)
    except DockerShimError:
        # The container isn't in the registry — almost always because it was
        # already torn down (EvoClaw's background watcher thread keeps polling
        # `docker exec <c> git rev-parse <tag>` while the main thread removes
        # the agent container at trial end). Real `docker exec` on a missing
        # container returns a non-zero rc with "No such container", NOT a hard
        # crash — mirror that so the watcher race is benign instead of killing
        # the run. (Same faithfulness fix as _cp_from's refused-transfer path.)
        return 1, b"", f"Error: No such container: {name}\n".encode()
    exec_user = user or ""
    # Sysbox: --user <name> can't resolve a runtime-added passwd entry; translate
    # to numeric uid:gid (see _resolve_sysbox_user). Runc path is byte-for-byte.
    if exec_user and os.environ.get("EVOCLAW_CONTAINER_RUNTIME", "").strip() == "sysbox-runc":
        exec_user = _resolve_sysbox_user(container, exec_user)
    try:
        res = _with_retry(
            f"exec {name}",
            lambda: container.exec_run(
                command, user=exec_user, workdir=workdir, environment=env or None, demux=True
            ),
        )
    except ConnectionError:
        # Cluster-loss: _with_retry exhausted its retries and raised ConnectionError
        # (an OSError) so EvoClaw's own eval-retry re-runs the eval on a fresh
        # container. Let it propagate — don't mask a genuine node loss.
        raise
    except Exception as exc:
        # A non-cluster-loss node exec error — e.g. a docker-py demux stream
        # corruption ("N is not a valid stream") that survived the node-side
        # resync-retry — must NOT crash the caller. EvoClaw's tag-watcher thread
        # polls `git rev-parse <tag>` through here; a raised error kills that
        # thread and the runner then hangs on "Waiting..." indefinitely (holding
        # the container). Real `docker exec` returns a non-zero rc, it doesn't
        # raise — mirror that so the watcher treats it as "tag not found", survives,
        # and retries next poll. Same faithfulness pattern as the No-such-container
        # (above) and refused-transfer (_cp_from) paths.
        LOGGER.warning(
            "docker_shim: exec %s raised a non-cluster error (%s); returning a "
            "docker-style non-zero so the caller (e.g. EvoClaw's tag-watcher) "
            "survives instead of dying", name, exc,
        )
        return 1, b"", f"docker_shim exec error: {exc}\n".encode()
    exit_code = getattr(res, "exit_code", res[0] if isinstance(res, tuple) else 0)
    output = getattr(res, "output", res[1] if isinstance(res, tuple) else (b"", b""))
    out, err = output if isinstance(output, tuple) else (output, b"")
    out = out or b""
    err = err or b""
    # Approximate the bind: copy each output volume ctr -> host so the host sees
    # what this exec wrote (e.g. the evaluator's /output test reports).
    for host, ctr in _OUTPUT_BINDS.get(name, []):
        _sync_out(container, ctr, Path(host))
    # git-archive-style stdout-to-file: caller passed stdout=<file>
    if hasattr(stdout_target, "write") and stdout_target not in (
        subprocess.PIPE, subprocess.STDOUT, subprocess.DEVNULL,
    ):
        stdout_target.write(out)
        out = b""
    return int(exit_code or 0), out, err


# ---- cp ---------------------------------------------------------------------
def _cp(args: list[str]) -> tuple[int, bytes, bytes]:
    paths = [a for a in args if not a.startswith("-")]
    if len(paths) != 2:
        return 1, b"", b"docker cp: expected SRC DST\n"
    src, dst = paths
    if ":" in src and src.split(":", 1)[0] in _REGISTRY:
        name, cpath = src.split(":", 1)
        return _cp_from(name, cpath, Path(dst))
    if ":" in dst and dst.split(":", 1)[0] in _REGISTRY:
        name, cpath = dst.split(":", 1)
        return _cp_to(Path(src), name, cpath)
    raise DockerShimError(f"docker_shim: cp endpoints not in registry: {src} {dst}")


def _put_path(container: Any, local: Path, cpath: str) -> None:
    """``docker cp``-equivalent: place ``local`` (file or dir) at ``cpath``."""
    parent = os.path.dirname(cpath.rstrip("/")) or "/"
    arcname = os.path.basename(cpath.rstrip("/"))
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        tf.add(str(local), arcname=arcname)
    container.exec_run(["mkdir", "-p", parent])
    container.put_archive(parent, buf.getvalue())


def _cp_to(local: Path, name: str, cpath: str) -> tuple[int, bytes, bytes]:
    _put_path(_lookup(name), local, cpath)
    return 0, b"", b""


def _sync_out(container: Any, cpath: str, host: Path) -> None:
    """Copy the *contents* of container ``cpath`` (a dir) into host dir ``host``.

    Used to approximate a live bind mount for output volumes: after an exec, the
    container's writes under ``cpath`` are mirrored to ``host``. Best-effort.
    """
    try:
        bits, _ = container.get_archive(cpath)
    except Exception as exc:
        LOGGER.debug("sync_out %s: %s", cpath, exc)
        return
    raw = b"".join(bits)
    host.mkdir(parents=True, exist_ok=True)
    root = host.resolve()
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:*") as tf:
        for m in tf.getmembers():
            rel = m.name.split("/", 1)[1] if "/" in m.name else ""  # strip leading dir
            if not rel:
                continue
            dest = (host / rel).resolve()
            if root not in (dest, *dest.parents):
                continue  # refuse to write outside host
            if m.isdir():
                dest.mkdir(parents=True, exist_ok=True)
            elif m.isfile():
                dest.parent.mkdir(parents=True, exist_ok=True)
                src = tf.extractfile(m)
                if src is not None:
                    dest.write_bytes(src.read())


def _cp_from(name: str, cpath: str, local: Path) -> tuple[int, bytes, bytes]:
    container = _lookup(name)
    try:
        bits, _stat = container.get_archive(cpath)
        raw = b"".join(bits)
    except Exception as exc:
        # xrlenv may REFUSE an over-cap get_archive at the transport
        # (a whole-/testbed copy exceeds the control-plane relay cap,
        # surfaced as ArchiveTooLarge). Treat it as a CONTAINED failure
        # of THIS copy only: return a non-zero docker-cp result so
        # EvoClaw's cleanup logs a warning and the eval/grading
        # continues, rather than letting a raw exception escape
        # subprocess.run. (EVOCLAW_COPY_TESTBED opts into this copy;
        # see run_e2e_xrlenv._configure_testbed_copy.)
        msg = f"docker cp {name}:{cpath}: transfer refused/failed: {exc}\n"
        LOGGER.warning(msg.strip())
        return 1, b"", msg.encode()
    local.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:*") as tf:
        members = tf.getmembers()
        # docker cp NAME:/a/b dst → dst is the file/dir; strip the leading arc name
        for m in members:
            stripped = m.name.split("/", 1)[1] if "/" in m.name else ""
            target = local if not stripped else local / stripped
            if m.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif m.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                f = tf.extractfile(m)
                if f is not None:
                    target.write_bytes(f.read())
    return 0, b"", b""


# ---- lifecycle / probes -----------------------------------------------------
def _rm(args: list[str]) -> tuple[int, bytes, bytes]:
    names = [a for a in args if not a.startswith("-")]
    for name in names:
        c = _REGISTRY.pop(name, None)
        if c is not None:
            try:
                c.remove(force=True)
            except Exception as exc:
                LOGGER.debug("remove %s: %s", name, exc)
    return 0, b"", b""


def _stop(args: list[str]) -> tuple[int, bytes, bytes]:
    for name in [a for a in args if not a.startswith("-")]:
        c = _REGISTRY.get(name)
        if c is not None:
            try:
                c.stop()
            except Exception as exc:
                LOGGER.debug("stop %s: %s", name, exc)
    return 0, b"", b""


def _start(args: list[str]) -> tuple[int, bytes, bytes]:
    # We force the `run` path for create, so `start` is the already-exists branch.
    for name in [a for a in args if not a.startswith("-")]:
        c = _REGISTRY.get(name)
        if c is not None:
            try:
                c.start()
            except Exception as exc:
                LOGGER.debug("start %s: %s", name, exc)
    return 0, b"", b""


def _kill(args: list[str]) -> tuple[int, bytes, bytes]:
    return _rm([a for a in args if not a.startswith("-")])


def _inspect(args: list[str]) -> tuple[int, bytes, bytes]:
    # docker inspect -f '{{.State.Running}}' NAME  → true/false from registry.
    fmt = None
    names = []
    i = 0
    while i < len(args):
        if args[i] in ("-f", "--format"):
            fmt = args[i + 1]
            i += 2
        else:
            names.append(args[i])
            i += 1
    if not names:
        return 1, b"", b"docker inspect: no target\n"
    name = names[0]
    present = name in _REGISTRY
    if fmt and "State.Running" in fmt:
        return (0, b"true\n", b"") if present else (1, b"", b"")
    return (0, b"[{}]\n", b"") if present else (1, b"", b"Error: No such object\n")


def _ps(args: list[str]) -> tuple[int, bytes, bytes]:
    # docker ps -a --format '{{.Names}}' --filter name=^NAME$  → NAME if known.
    want = None
    i = 0
    while i < len(args):
        if args[i] == "--filter":
            val = args[i + 1]
            if val.startswith("name="):
                want = val[len("name="):].strip("^$")
                i += 2
                continue
        i += 1
    if want and want in _REGISTRY:
        return 0, (want + "\n").encode(), b""
    return 0, b"", b""


def _images(_args: list[str]) -> tuple[int, bytes, bytes]:
    # docker images -q REF — existence probe. Return a placeholder id so the
    # caller treats the image as present; the real pull happens on `run` via
    # ensure_image_present. (Documented simplification — SHIM-SURFACE.md.)
    return 0, b"xrlenv-shim-present\n", b""


def _image_inspect(_args: list[str]) -> tuple[int, bytes, bytes]:
    # docker image inspect REF — succeed; real pull is on `run`.
    return 0, b"[{}]\n", b""
