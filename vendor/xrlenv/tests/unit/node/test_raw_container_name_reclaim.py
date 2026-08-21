"""Unit tests for the raw-acquire 409 name-reclaim + docker-error
classification (prod "container name already in use" + AIMD-collapse fix).

Two behaviours are locked in here:

1. ``_is_node_health_error`` — only daemon-saturation faults (timeouts,
   transport errors, 5xx) feed the AIMD admission limiter. A 409 name
   conflict / 404 missing image is a *request* fault and must NOT drive
   the node's admission limit down.
2. ``RawContainerManager`` reclaims an orphaned container name on a 409
   and retries the create once — but only when the holder is an
   xrlenv-managed raw container; a foreign container is never removed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import docker.errors
import pytest
import requests.exceptions
from xrlenv.errors import XRLEnvError
from xrlenv.node.raw_container import (
    RawContainerManager,
    _is_name_conflict,
    _is_node_health_error,
)

# ──────────────────────────────────────────────────────────────────────────────
# docker.errors.APIError factory — status_code is derived from .response
# ──────────────────────────────────────────────────────────────────────────────


class _Resp:
    """Minimal stand-in for ``requests.Response`` — docker-py's
    ``APIError.__str__`` reads ``.status_code``, ``.url`` and ``.reason``
    when formatting a 4xx/5xx."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        self.url = "http+docker://localhost/v1.54/containers/create"
        self.reason = "Conflict" if status_code == 409 else "Error"


def _api_error(
    status: int, message: str, *, explanation: str | None = None,
) -> docker.errors.APIError:
    # docker-py's APIError.__str__ rewrites the message for a 4xx/5xx as
    # "{status} {Client,Server} Error for {url}: {reason}" and appends the
    # ``explanation`` in parens — so daemon detail (e.g. "already in use")
    # reaches ``str(exc)`` via ``explanation``, NOT the bare message.
    return docker.errors.APIError(message, response=_Resp(status),
                                  explanation=explanation)


def _name_conflict_error(name: str, holder_id: str) -> docker.errors.APIError:
    return _api_error(
        409,
        "409 Client Error",
        explanation=(
            f'Conflict. The container name "/{name}" is already in use by '
            f'container "{holder_id}". You have to remove (or rename) that '
            "container to be able to reuse that name."
        ),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Classification helpers
# ──────────────────────────────────────────────────────────────────────────────


def test_409_name_conflict_is_not_a_health_error() -> None:
    exc = _name_conflict_error("7687821c__sanitize-git-repo__s2", "797521ca")
    assert _is_name_conflict(exc) is True
    assert _is_node_health_error(exc) is False


def test_404_image_not_found_is_not_a_health_error() -> None:
    exc = _api_error(404, "manifest unknown: image not found")
    assert _is_node_health_error(exc) is False


def test_400_bad_request_is_not_a_health_error() -> None:
    assert _is_node_health_error(_api_error(400, "bad parameter")) is False


def test_500_daemon_error_is_a_health_error() -> None:
    assert _is_node_health_error(_api_error(500, "server error")) is True


def test_timeout_is_a_health_error() -> None:
    assert _is_node_health_error(requests.exceptions.ReadTimeout("timed out")) is True


def test_connection_error_is_a_health_error() -> None:
    assert _is_node_health_error(
        requests.exceptions.ConnectionError("connection refused"),
    ) is True


def test_statusless_transport_error_defaults_to_health() -> None:
    """A docker exception with no HTTP status (couldn't even reach the
    daemon) defaults to a health fault — over-throttle, don't ignore."""
    assert _is_node_health_error(docker.errors.DockerException("broken pipe")) is True


def test_non_conflict_409ish_message_without_status_is_not_a_conflict() -> None:
    assert _is_name_conflict(docker.errors.DockerException("already in use")) is False


# ──────────────────────────────────────────────────────────────────────────────
# Minimal docker fake that raises a real APIError on a name collision
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class _FakeContainer:
    id: str
    name: str
    labels: dict[str, str]
    removed: bool = False
    # Any exception to raise from ``remove`` — a docker APIError, a
    # requests timeout, or asyncio.CancelledError (the destroy-health tests).
    raise_on_remove: BaseException | None = None

    def remove(self, *, force: bool = False) -> None:
        if self.raise_on_remove is not None:
            raise self.raise_on_remove
        self.removed = True


class _FakeImages:
    def get(self, image: str) -> Any:
        return object()  # always "present"; the reclaim path is image-agnostic


class _FakeContainers:
    """Keyed by container id (like docker), with a name->id index so
    ``run(name=...)`` can raise a real 409 ``APIError`` on collision and
    ``get(id_or_name)`` resolves either (docker-py accepts both)."""

    def __init__(self) -> None:
        self._by_id: dict[str, _FakeContainer] = {}
        self._name_to_id: dict[str, str] = {}
        self._next = 0
        self.run_calls = 0

    def _register(self, c: _FakeContainer) -> None:
        self._by_id[c.id] = c
        self._name_to_id[c.name] = c.id

    # seed a pre-existing container holding a name
    def seed(self, name: str, *, labels: dict[str, str]) -> _FakeContainer:
        self._next += 1
        c = _FakeContainer(id=f"holder-{self._next:04d}", name=name, labels=labels)
        self._register(c)
        return c

    def run(self, *, name: str | None = None, labels: dict[str, str],
            **_: Any) -> _FakeContainer:
        self.run_calls += 1
        if name is not None and name in self._name_to_id:
            holder = self._by_id[self._name_to_id[name]]
            if not holder.removed:
                raise _name_conflict_error(name, holder.id)
        self._next += 1
        cid = f"new-{self._next:04d}"
        cname = name or cid
        c = _FakeContainer(id=cid, name=cname, labels=labels)
        self._register(c)
        return c

    def get(self, key: str) -> _FakeContainer:
        c = self._by_id.get(key)
        if c is None and key in self._name_to_id:
            c = self._by_id.get(self._name_to_id[key])
        if c is None or c.removed:
            raise docker.errors.NotFound(f"no such container: {key}")
        return c

    def list(self, **_: Any) -> list[_FakeContainer]:
        # core-ledger reconcile on manager init enumerates raw containers.
        return [
            c for c in self._by_id.values()
            if not c.removed and c.labels.get("xrlenv.session_kind") == "raw"
        ]


class _FakeAPI:
    pass


class _FakeDockerClient:
    def __init__(self) -> None:
        self.images = _FakeImages()
        self.containers = _FakeContainers()
        self.api = _FakeAPI()


_RAW_LABELS = {"xrlenv.session_kind": "raw", "xrlenv.rollout_id": "old"}


@pytest.mark.asyncio
async def test_409_against_xrlenv_orphan_is_reclaimed_and_retried() -> None:
    """A name held by a stale xrlenv container is reclaimed (force-removed)
    and the create retried once — the acquire succeeds."""
    client = _FakeDockerClient()
    name = "7687821c__sanitize-git-repo__s2"
    holder = client.containers.seed(name, labels=dict(_RAW_LABELS))
    mgr = RawContainerManager(docker_client=client)

    record = await mgr.acquire(rollout_id="r-new", image="busybox:1", name=name)

    assert holder.removed is True          # stale orphan was reclaimed
    assert record.container_name == name   # new container took the name
    assert client.containers.run_calls == 2  # initial 409 + one retry


@pytest.mark.asyncio
async def test_409_against_foreign_container_is_not_reclaimed() -> None:
    """A name held by a NON-xrlenv container must never be removed; the
    409 surfaces as an XRLEnvError and the foreign container is untouched."""
    client = _FakeDockerClient()
    name = "some-unmanaged-container"
    foreign = client.containers.seed(name, labels={"com.example": "theirs"})
    mgr = RawContainerManager(docker_client=client)

    with pytest.raises(XRLEnvError) as ei:
        await mgr.acquire(rollout_id="r-new", image="busybox:1", name=name)

    assert "already in use" in str(ei.value)
    assert foreign.removed is False          # foreign container untouched
    assert client.containers.run_calls == 1  # no retry attempted


@pytest.mark.asyncio
async def test_reclaim_failure_surfaces_original_409() -> None:
    """If the orphan can't be removed (docker error on remove), the
    original 409 is surfaced rather than retried into a second failure."""
    client = _FakeDockerClient()
    name = "7687821c__stuck__s0"
    holder = client.containers.seed(name, labels=dict(_RAW_LABELS))
    holder.raise_on_remove = docker.errors.APIError("remove failed")
    mgr = RawContainerManager(docker_client=client)

    with pytest.raises(XRLEnvError) as ei:
        await mgr.acquire(rollout_id="r-new", image="busybox:1", name=name)

    assert "already in use" in str(ei.value)
    assert client.containers.run_calls == 1  # remove failed → no retry


# ──────────────────────────────────────────────────────────────────────────────
# Destroy-path health (open item 1): a node whose destroys fail/hang must feed
# the AIMD health signal, so a saturated node is throttled — not (pre-fix)
# promoted because only the create path was instrumented.
# ──────────────────────────────────────────────────────────────────────────────


async def _acquire_one(mgr: RawContainerManager) -> Any:
    return await mgr.acquire(rollout_id="r1", image="busybox:1")


@pytest.mark.asyncio
async def test_destroy_remove_timeout_records_health() -> None:
    client = _FakeDockerClient()
    mgr = RawContainerManager(docker_client=client)
    rec = await _acquire_one(mgr)
    client.containers.get(rec.container_id).raise_on_remove = (
        requests.exceptions.ReadTimeout("timed out")
    )

    await mgr.destroy(rollout_id="r1", container_id=rec.container_id)

    snap = mgr._health.snapshot()
    assert snap.docker_error_count == 1
    assert snap.docker_timeout_count == 1  # timeout-class fault


@pytest.mark.asyncio
async def test_destroy_remove_5xx_records_health_non_timeout() -> None:
    client = _FakeDockerClient()
    mgr = RawContainerManager(docker_client=client)
    rec = await _acquire_one(mgr)
    client.containers.get(rec.container_id).raise_on_remove = _api_error(
        500, "500 Server Error",
    )

    await mgr.destroy(rollout_id="r1", container_id=rec.container_id)

    snap = mgr._health.snapshot()
    assert snap.docker_error_count == 1
    assert snap.docker_timeout_count == 0  # 5xx is an error, not a timeout


@pytest.mark.asyncio
async def test_destroy_remove_404_does_not_record_health() -> None:
    """A 4xx on remove (already-gone) is not a saturation signal."""
    client = _FakeDockerClient()
    mgr = RawContainerManager(docker_client=client)
    rec = await _acquire_one(mgr)
    client.containers.get(rec.container_id).raise_on_remove = _api_error(
        404, "404 Client Error",
    )

    await mgr.destroy(rollout_id="r1", container_id=rec.container_id)

    assert mgr._health.snapshot().docker_error_count == 0


@pytest.mark.asyncio
async def test_destroy_cancelled_records_timeout_health_and_reraises() -> None:
    """A destroy abandoned mid-remove (control wire ceiling / shutdown) is a
    saturation signal — the daemon couldn't finish in time."""
    client = _FakeDockerClient()
    mgr = RawContainerManager(docker_client=client)
    rec = await _acquire_one(mgr)
    import asyncio
    client.containers.get(rec.container_id).raise_on_remove = asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await mgr.destroy(rollout_id="r1", container_id=rec.container_id)

    snap = mgr._health.snapshot()
    assert snap.docker_error_count == 1
    assert snap.docker_timeout_count == 1


@pytest.mark.asyncio
async def test_clean_destroy_records_no_health() -> None:
    client = _FakeDockerClient()
    mgr = RawContainerManager(docker_client=client)
    rec = await _acquire_one(mgr)

    await mgr.destroy(rollout_id="r1", container_id=rec.container_id)

    assert mgr._health.snapshot().docker_error_count == 0


@pytest.mark.asyncio
async def test_force_destroy_remove_timeout_records_health() -> None:
    client = _FakeDockerClient()
    mgr = RawContainerManager(docker_client=client)
    rec = await _acquire_one(mgr)
    client.containers.get(rec.container_id).raise_on_remove = (
        requests.exceptions.ReadTimeout("timed out")
    )

    await mgr.force_destroy(container_id=rec.container_id)

    assert mgr._health.snapshot().docker_timeout_count == 1
