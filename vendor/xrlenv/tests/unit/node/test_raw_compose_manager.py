"""RawContainerManager multi-service compose integration (P1.7.C.2 step 2b-2).

Verifies the manager wiring around the (separately-tested) ComposeProjectRunner:
every member container is registered ↔ rollout_id so the existing
container-scoped exec/archive/ownership path addresses main + sidecars; images
are refcounted in the cache for the project's lifetime; and destroy downs the
WHOLE project + deregisters + releases. Uses an injected fake runner + fake image
cache — no docker.
"""
from __future__ import annotations

from typing import cast

import pytest
from xrlenv.errors import XRLEnvError
from xrlenv.node.raw_compose import ComposeProjectRecord, ComposeProjectRunner
from xrlenv.node.raw_container import RawContainerManager


class FakeComposeRunner:
    def __init__(
        self, record: ComposeProjectRecord, *, down_error: Exception | None = None,
        up_error: Exception | None = None,
    ) -> None:
        self._record = record
        self._down_error = down_error
        self._up_error = up_error
        self.up_calls: list[dict] = []
        self.down_calls: list[dict] = []

    async def up(self, **kwargs) -> ComposeProjectRecord:
        self.up_calls.append(kwargs)
        if self._up_error is not None:
            raise self._up_error
        return self._record

    async def down(self, **kwargs) -> None:
        self.down_calls.append(kwargs)
        if self._down_error is not None:
            raise self._down_error


class FakeImageCache:
    def __init__(self) -> None:
        self.acquired: list[str] = []
        self.released: list[str] = []

    def acquire(self, ref: str) -> None:
        self.acquired.append(ref)

    def release(self, ref: str) -> None:
        self.released.append(ref)

    async def ensure_present(self, ref: str) -> None:  # pragma: no cover
        pass


class _FakeContainer:
    def __init__(self, cid: str, *, labels: dict[str, str] | None = None,
                 remove_error: Exception | None = None) -> None:
        self.id = cid
        self.labels = labels or {}
        self.removed = False
        self._remove_error = remove_error

    def remove(self, force: bool = False, v: bool = False) -> None:
        if self._remove_error is not None:
            raise self._remove_error
        self.removed = True


class _FakeNetwork:
    def __init__(self, name: str) -> None:
        self.name = name
        self.removed = False

    def remove(self) -> None:
        self.removed = True


class _FakeVolume:
    def __init__(self, name: str, *, remove_error: Exception | None = None) -> None:
        self.name = name
        self.removed = False
        self._remove_error = remove_error

    def remove(self, force: bool = False) -> None:
        if self._remove_error is not None:
            raise self._remove_error
        self.removed = True


def _labels_match(container_labels: dict[str, str], label_filter: object) -> bool:
    # docker's list filter with a list of "k=v" ANDs them; match that semantics.
    if not isinstance(label_filter, list):
        return True
    wanted = [f.split("=", 1) for f in label_filter]
    return all(container_labels.get(k) == v for k, v in wanted)


class FakeDockerClient:
    """Minimal docker client for the H10 fail-closed, ownership-scoped teardown.

    ``list`` returns only NON-removed members (so a re-list after ``remove`` reflects the
    teardown), honors the label AND-filter, and can inject a list failure."""

    def __init__(self, members: list[_FakeContainer] | None = None,
                 networks: list[_FakeNetwork] | None = None,
                 volumes: list[_FakeVolume] | None = None,
                 list_error: Exception | None = None) -> None:
        self._members = members or []
        self._networks = networks or []
        self._volumes = volumes or []
        self._list_error = list_error
        outer = self

        class _Containers:
            def list(self, *, all: bool = False, filters: dict | None = None) -> list:
                if outer._list_error is not None:
                    raise outer._list_error
                live = [m for m in outer._members if not m.removed]
                lf = (filters or {}).get("label")
                return [m for m in live if _labels_match(m.labels, lf)]
        class _Networks:
            def list(self, *, names: list | None = None, filters: dict | None = None) -> list:
                return [n for n in outer._networks if not n.removed]   # re-list reflects removal
        class _Volumes:
            def list(self, *, filters: dict | None = None) -> list:
                return [v for v in outer._volumes if not v.removed]    # re-list reflects removal
        self.containers = _Containers()
        self.networks = _Networks()
        self.volumes = _Volumes()


def _member(cid: str, *, project: str = "ghost", rollout_id: str = "r1",
            remove_error: Exception | None = None) -> _FakeContainer:
    return _FakeContainer(
        cid,
        labels={"com.docker.compose.project": project, "xrlenv.rollout_id": rollout_id},
        remove_error=remove_error,
    )


def _record() -> ComposeProjectRecord:
    return ComposeProjectRecord(
        project_name="proj",
        project_dir="/tmp/proj",
        main_container_id="cid_main",
        main_container_name="proj-main",
        service_container_ids={"main": "cid_main", "postgres": "cid_pg"},
    )


def _manager(runner: FakeComposeRunner, cache: FakeImageCache | None = None):
    return RawContainerManager(
        docker_client=object(),  # compose path never touches it
        image_cache=cache,
        compose_runner=cast(ComposeProjectRunner, runner),  # duck-typed fake
    )


@pytest.fixture()
def runner() -> FakeComposeRunner:
    return FakeComposeRunner(_record())


async def test_acquire_registers_all_members_for_scoping(runner: FakeComposeRunner) -> None:
    cache = FakeImageCache()
    mgr = _manager(runner, cache)
    rec = await mgr.acquire_compose_project(
        rollout_id="r1",
        project_name="proj",
        compose_yaml="services:\n  main: {}\n",
        images=["ns/app:main", "postgres:14"],
        main_service="main",
    )
    assert rec.main_container_id == "cid_main"
    # every member registered under the rollout so exec/archive scoping addresses
    # main AND sidecars (the ownership check reads _records).
    assert set(mgr._records) == {"cid_main", "cid_pg"}
    assert mgr._records["cid_main"].rollout_id == "r1"
    assert mgr._records["cid_pg"].rollout_id == "r1"
    # main keeps its friendly name; sidecars fall back to their id.
    assert mgr._records["cid_main"].container_name == "proj-main"
    # ownership check passes for the owner, fails for another rollout.
    mgr._assert_owner("r1", "cid_main")
    with pytest.raises(XRLEnvError):
        mgr._assert_owner("other", "cid_main")


async def test_acquire_refcounts_images(runner: FakeComposeRunner) -> None:
    cache = FakeImageCache()
    mgr = _manager(runner, cache)
    await mgr.acquire_compose_project(
        rollout_id="r1", project_name="proj",
        compose_yaml="x", images=["ns/app:main", "postgres:14"],
    )
    assert sorted(cache.acquired) == ["ns/app:main", "postgres:14"]
    # the runner was asked to bring the project up with those images
    assert runner.up_calls[0]["images"] == ("ns/app:main", "postgres:14")


async def test_acquire_rejects_duplicate_project_name(runner: FakeComposeRunner) -> None:
    # audit H10: a second acquire with a live project's name is rejected BEFORE `up`, so it can't
    # clobber the first rollout's ownership (or mutate its resources via `up`).
    mgr = _manager(runner)
    await mgr.acquire_compose_project(rollout_id="r1", project_name="proj", compose_yaml="x")
    assert len(runner.up_calls) == 1
    with pytest.raises(XRLEnvError, match="already active"):
        await mgr.acquire_compose_project(rollout_id="r2", project_name="proj", compose_yaml="x")
    assert len(runner.up_calls) == 1                       # no second `up`
    assert mgr._compose_projects["proj"].rollout_id == "r1"  # first rollout still owns it


async def test_acquire_rolls_back_name_reservation_on_up_failure() -> None:
    # a failed `up` must release the name reservation so a later acquire of the same name isn't
    # permanently blocked by a stale reservation.
    from xrlenv.node.raw_compose import ComposeError
    runner = FakeComposeRunner(_record(), up_error=ComposeError("up boom"))
    mgr = _manager(runner)
    with pytest.raises(ComposeError, match="up boom"):
        await mgr.acquire_compose_project(rollout_id="r1", project_name="proj", compose_yaml="x")
    assert "proj" not in mgr._reserving_projects          # reservation rolled back
    assert "proj" not in mgr._compose_projects


async def test_destroy_downs_project_and_deregisters(runner: FakeComposeRunner) -> None:
    cache = FakeImageCache()
    mgr = _manager(runner, cache)
    await mgr.acquire_compose_project(
        rollout_id="r1", project_name="proj",
        compose_yaml="x", images=["ns/app:main"],
    )
    await mgr.destroy_compose_project(rollout_id="r1", project_name="proj")
    # downed the whole project by name + dir
    assert runner.down_calls == [{"project_name": "proj", "project_dir": "/tmp/proj"}]
    # every member deregistered, project state cleared, image released
    assert mgr._records == {}
    assert mgr._compose_projects == {}
    assert cache.released == ["ns/app:main"]


async def test_destroy_rejects_wrong_owner(runner: FakeComposeRunner) -> None:
    mgr = _manager(runner)
    await mgr.acquire_compose_project(
        rollout_id="r1", project_name="proj", compose_yaml="x",
    )
    with pytest.raises(XRLEnvError, match="does not own"):
        await mgr.destroy_compose_project(rollout_id="intruder", project_name="proj")
    # nothing torn down on the rejected call
    assert runner.down_calls == []
    assert "proj" in mgr._compose_projects


async def test_destroy_unregistered_project_reaps_and_verifies(
    runner: FakeComposeRunner,
) -> None:
    # audit H10: an unregistered project (a completed down whose reply timed out, or a node-agent
    # restart that wiped memory-only ownership) reaps surviving members by the project+rollout
    # labels, VERIFIES none remain, and only then reports success (confirmed absence) so the
    # coordinator's node-confirmed capacity release proceeds.
    survivor = _member("cid_pg")
    net = _FakeNetwork("ghost_default")
    vol = _FakeVolume("ghost_pgdata")   # a NAMED volume `docker rm -v` wouldn't remove
    mgr = RawContainerManager(
        docker_client=FakeDockerClient(members=[survivor], networks=[net], volumes=[vol]),
        compose_runner=cast(ComposeProjectRunner, runner),
    )
    await mgr.destroy_compose_project(rollout_id="r1", project_name="ghost")  # no raise
    assert survivor.removed is True     # surviving sidecar reaped (with volumes)
    assert vol.removed is True          # named project volume pruned (audit H10 disk cleanup)
    assert net.removed is True          # project network pruned
    assert runner.down_calls == []      # no compose-file down (there's no state/dir)


async def test_destroy_unregistered_project_noop_when_nothing_survives(
    runner: FakeComposeRunner,
) -> None:
    # the timeout-retry case: the down already completed, so nothing carries the label.
    mgr = RawContainerManager(
        docker_client=FakeDockerClient(members=[], networks=[]),
        compose_runner=cast(ComposeProjectRunner, runner),
    )
    await mgr.destroy_compose_project(rollout_id="r1", project_name="ghost")  # no raise


async def test_destroy_unregistered_fails_closed_on_list_error(
    runner: FakeComposeRunner,
) -> None:
    # audit H10: a Docker LIST failure means we cannot confirm absence — RAISE (fail closed) so
    # the coordinator retains capacity, NOT a false confirmed-absence.
    import docker.errors
    mgr = RawContainerManager(
        docker_client=FakeDockerClient(list_error=docker.errors.DockerException("daemon down")),
        compose_runner=cast(ComposeProjectRunner, runner),
    )
    with pytest.raises(docker.errors.DockerException):
        await mgr.destroy_compose_project(rollout_id="r1", project_name="ghost")


async def test_destroy_unregistered_fails_closed_when_member_survives(
    runner: FakeComposeRunner,
) -> None:
    # a member whose removal fails stays present on the re-list — RAISE (fail closed) rather than
    # report the project torn down while a container is still running.
    import docker.errors
    stuck = _member("cid_stuck", remove_error=docker.errors.APIError("cannot remove"))
    mgr = RawContainerManager(
        docker_client=FakeDockerClient(members=[stuck]),
        compose_runner=cast(ComposeProjectRunner, runner),
    )
    with pytest.raises(docker.errors.APIError):
        await mgr.destroy_compose_project(rollout_id="r1", project_name="ghost")
    assert stuck.removed is False


async def test_destroy_unregistered_is_ownership_scoped(
    runner: FakeComposeRunner,
) -> None:
    # audit H10: the reap is keyed on BOTH project AND rollout_id, so a project-NAME collision
    # from a DIFFERENT rollout is never removed.
    ours = _member("cid_ours", project="ghost", rollout_id="r1")
    theirs = _member("cid_theirs", project="ghost", rollout_id="OTHER-ROLLOUT")
    mgr = RawContainerManager(
        docker_client=FakeDockerClient(members=[ours, theirs]),
        compose_runner=cast(ComposeProjectRunner, runner),
    )
    await mgr.destroy_compose_project(rollout_id="r1", project_name="ghost")
    assert ours.removed is True         # our member reaped
    assert theirs.removed is False      # another rollout's same-named project left untouched


async def test_acquire_cancelled_after_up_tears_down_unregistered_stack() -> None:
    # audit H10: a cancellation AFTER `up` succeeded but BEFORE registration would otherwise leak
    # a LIVE stack with no owner (invisible to destroy, holding capacity). The cancellation-safe
    # rollback tears the just-created stack down and releases the name reservation.
    import asyncio

    gate = asyncio.Event()

    class _BlockingRunner(FakeComposeRunner):
        async def up(self, **kwargs):  # type: ignore[no-untyped-def]
            self.up_calls.append(kwargs)
            await gate.wait()          # hold inside `up` so we can arrange the post-up cancel
            return self._record

    runner = _BlockingRunner(_record())
    mgr = _manager(runner)
    task = asyncio.create_task(
        mgr.acquire_compose_project(rollout_id="r1", project_name="proj", compose_yaml="x"),
    )
    for _ in range(1000):                         # let it reserve the name + block inside `up`
        await asyncio.sleep(0)
        if runner.up_calls:
            break
    # Now `up` is in flight and the reservation lock is FREE again. Hold it so that when `up`
    # returns, the registration `async with self._lock` blocks — the exact post-up window.
    await mgr._lock.acquire()
    gate.set()                                     # `up` returns → task advances to registration
    await asyncio.sleep(0)                          # settle at the blocked registration lock
    task.cancel()
    mgr._lock.release()                            # let the rollback path acquire the lock
    with pytest.raises(asyncio.CancelledError):
        await task
    assert runner.up_calls and runner.down_calls  # `up` happened; rollback `down` tore it back down
    assert "proj" not in mgr._reserving_projects   # reservation released (finally)
    assert "proj" not in mgr._compose_projects     # never registered


async def test_recovery_prune_survivor_logs_but_does_not_fail_closed(
    runner: FakeComposeRunner, caplog: pytest.LogCaptureFixture,
) -> None:
    # audit H10: durable DISK cleanup is decoupled from the capacity release. A NAMED volume whose
    # removal fails must NOT fail-close the teardown (its containers are already gone → capacity
    # is releasable), but the survivor is RE-VERIFIED and LOGGED so the disk/reuse uncertainty is
    # surfaced, not silently swallowed.
    import logging

    import docker.errors
    survivor = _member("cid_pg", project="ghost", rollout_id="r1")
    stuck_vol = _FakeVolume("ghost_data", remove_error=docker.errors.APIError("volume busy"))
    mgr = RawContainerManager(
        docker_client=FakeDockerClient(members=[survivor], volumes=[stuck_vol]),
        compose_runner=cast(ComposeProjectRunner, runner),
    )
    with caplog.at_level(logging.WARNING):
        await mgr.destroy_compose_project(rollout_id="r1", project_name="ghost")  # no raise
    assert survivor.removed is True          # capacity-relevant container reap succeeded
    assert stuck_vol.removed is False        # the volume genuinely survived
    assert any("still present after prune" in r.message and "volume" in r.message
               for r in caplog.records)      # the uncertainty is surfaced, not swallowed


async def test_recovery_destroy_defers_when_acquire_owns_name(
    runner: FakeComposeRunner,
) -> None:
    # audit H10: if a NEW same-name acquire is in flight (holds the name reservation), the recovery
    # must NOT reap concurrently — its member removal + name-scoped resource cleanup would race the
    # acquire's `docker compose up` reconciliation. It DEFERS (raises a retryable error) so the
    # coordinator retains capacity and retries once the acquire has released the name.
    survivor = _member("cid_old", project="ghost", rollout_id="r1")
    vol = _FakeVolume("ghost_pgdata")
    net = _FakeNetwork("ghost_default")
    mgr = RawContainerManager(
        docker_client=FakeDockerClient(members=[survivor], networks=[net], volumes=[vol]),
        compose_runner=cast(ComposeProjectRunner, runner),
    )
    mgr._reserving_projects.add("ghost")   # a concurrent acquire owns the name

    with pytest.raises(XRLEnvError, match="DEFERRING"):
        await mgr.destroy_compose_project(rollout_id="r1", project_name="ghost")
    # nothing reaped (fully serialized — the reap is deferred, not run concurrently)…
    assert survivor.removed is False
    assert vol.removed is False
    assert net.removed is False
    assert "ghost" in mgr._reserving_projects  # the acquire's reservation left intact


async def test_leaked_disk_prune_is_queued_and_retried_on_next_teardown(
    runner: FakeComposeRunner,
) -> None:
    # audit H10: a named volume whose removal fails during teardown is QUEUED and RETRIED on the
    # next compose teardown (a bounded GC/retry, no periodic task, no hot-path change) — not left
    # to leak silently until manual operator cleanup.
    import docker.errors
    survivor = _member("cid_pg", project="ghost", rollout_id="r1")
    stuck_vol = _FakeVolume("ghost_data", remove_error=docker.errors.APIError("volume busy"))
    client = FakeDockerClient(members=[survivor], volumes=[stuck_vol])
    mgr = RawContainerManager(
        docker_client=client, compose_runner=cast(ComposeProjectRunner, runner),
    )
    await mgr.destroy_compose_project(rollout_id="r1", project_name="ghost")
    assert stuck_vol.removed is False              # the volume genuinely leaked…
    assert "ghost" in mgr._pending_resource_prune  # …and was QUEUED for retry

    # the volume becomes removable (the transient docker hiccup cleared); the next unrelated
    # teardown retries the queued prune and clears it.
    stuck_vol._remove_error = None
    await mgr.destroy_compose_project(rollout_id="r2", project_name="other")  # nothing to reap
    assert stuck_vol.removed is True               # retry pruned the previously-leaked volume
    assert "ghost" not in mgr._pending_resource_prune


async def test_recovery_destroy_reaps_and_prunes_when_name_free(
    runner: FakeComposeRunner,
) -> None:
    # the common case: no concurrent acquire → recovery reserves the name, reaps the old rollout's
    # members AND prunes the project-name-scoped volumes/network (it holds the reservation, so no
    # new owner can exist), then releases the reservation.
    survivor = _member("cid_old", project="ghost", rollout_id="r1")
    vol = _FakeVolume("ghost_pgdata")
    net = _FakeNetwork("ghost_default")
    mgr = RawContainerManager(
        docker_client=FakeDockerClient(members=[survivor], networks=[net], volumes=[vol]),
        compose_runner=cast(ComposeProjectRunner, runner),
    )
    await mgr.destroy_compose_project(rollout_id="r1", project_name="ghost")
    assert survivor.removed is True
    assert vol.removed is True
    assert net.removed is True
    assert "ghost" not in mgr._reserving_projects   # reservation released


async def test_failed_down_retains_state_and_refcounts() -> None:
    # P1 audit: a failed/timeout `docker compose down` must NOT report success —
    # the project stays registered + capacity held (invariant 2) until a confirmed
    # teardown. The manager propagates the error and clears NOTHING.
    from xrlenv.node.raw_compose import ComposeError

    runner = FakeComposeRunner(_record(), down_error=ComposeError("daemon wedged"))
    cache = FakeImageCache()
    mgr = _manager(runner, cache)
    await mgr.acquire_compose_project(
        rollout_id="r1", project_name="proj",
        compose_yaml="x", images=["ns/app:main"],
    )
    with pytest.raises(ComposeError, match="daemon wedged"):
        await mgr.destroy_compose_project(rollout_id="r1", project_name="proj")
    # members still registered, project still tracked, images NOT released
    assert set(mgr._records) == {"cid_main", "cid_pg"}
    assert "proj" in mgr._compose_projects
    assert cache.released == []


async def test_acquire_forwards_timeout_and_main_service(runner: FakeComposeRunner) -> None:
    mgr = _manager(runner)
    await mgr.acquire_compose_project(
        rollout_id="r1", project_name="proj", compose_yaml="x",
        main_service="app", up_timeout_s=42.0,
    )
    call = runner.up_calls[0]
    assert call["main_service"] == "app"
    assert call["up_timeout_s"] == 42.0


# ── P6 step-3c — compose cgroup_parent injection (capable nodes only) ──────────


class _NoopCgroupWriter:
    def ensure_group(self, path: str) -> None: ...
    def write_cpuset_cpus(self, path: str, value: str) -> None: ...
    def remove_group(self, path: str) -> None: ...


def _capable_compose_manager(runner: FakeComposeRunner) -> RawContainerManager:
    mgr = RawContainerManager(
        docker_client=object(),  # best-effort docker calls fail gracefully
        compose_runner=cast(ComposeProjectRunner, runner),
        isolation_selftest=lambda: True,
        cgroup_writer=_NoopCgroupWriter(),  # structural _CgroupWriter Protocol
    )
    assert mgr.isolation_capable() is True  # NodeHello populates the capability cache
    return mgr


async def test_capable_node_injects_cgroup_parent_into_runc_compose_services(
    runner: FakeComposeRunner,
) -> None:
    import yaml

    mgr = _capable_compose_manager(runner)
    await mgr.acquire_compose_project(
        rollout_id="r1", project_name="proj",
        compose_yaml=(
            "services:\n"
            "  main:\n    image: app\n"
            "  sb:\n    image: dind\n    runtime: sysbox-runc\n"
        ),
        images=["app"],
    )
    sent = yaml.safe_load(runner.up_calls[-1]["compose_yaml"])
    assert sent["services"]["main"]["cgroup_parent"] == "/xrlenv-shared"  # runc
    assert "cgroup_parent" not in sent["services"]["sb"]                  # sysbox left alone


async def test_non_capable_node_does_not_inject_cgroup_parent(
    runner: FakeComposeRunner,
) -> None:
    # A plain manager never populates the capability cache → no shared parent →
    # the compose document reaches the runner unchanged.
    mgr = _manager(runner)
    original = "services:\n  main:\n    image: app\n"
    await mgr.acquire_compose_project(
        rollout_id="r1", project_name="proj", compose_yaml=original, images=["app"],
    )
    assert runner.up_calls[-1]["compose_yaml"] == original
