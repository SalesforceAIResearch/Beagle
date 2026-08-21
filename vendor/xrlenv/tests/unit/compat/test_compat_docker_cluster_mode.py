"""P1.7.B foundation — tests for ``xrlenv.compat.docker_client``
cluster mode.

Uses a fake ``Client`` (no live control plane). Verifies that
docker-py's high-level managers (``client.containers.create`` →
``container.start`` → ``container.remove``) route through the
xrlenv ``Client.acquire_container`` / ``ClusterContainerSession``
surface, AND that calls we haven't yet wired raise a clean
``NotImplementedError`` rather than failing on uninitialized
parent state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from xrlenv.client.container_session import ClusterContainerSession
from xrlenv.compat import docker_client as _docker_client_mod
from xrlenv.compat.docker_client import (
    ClusterContainerControl,
    XrlenvAPIClient,
    XrlenvDockerClient,
    from_env,
)
from xrlenv.control.service import RawAcquireResult


@pytest.fixture(autouse=True)
def _reset_kwarg_warn_dedup() -> Any:
    """The "ignoring docker-py kwargs"/platform warnings dedup once-per-process via a
    module-level set. Clear it around each test so a test that ASSERTS the warning
    fires isn't silenced by an earlier test that already tripped the same signature."""
    _docker_client_mod._KWARG_WARN_SEEN.clear()
    yield
    _docker_client_mod._KWARG_WARN_SEEN.clear()

# ──────────────────────────────────────────────────────────────────────────────
# Fakes
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class _FakeClient:
    """Minimal ``Client`` stand-in. Records calls so tests can
    assert routing + arguments. Returns a real
    ``ClusterContainerSession`` wrapping a fake transport so the
    drop-in's session.destroy() etc. work."""

    next_acquire: RawAcquireResult = field(
        default_factory=lambda: RawAcquireResult(
            rollout_id="r-1", container_id="c-1",
            container_name="cname-1", node_id="node-A",
        ),
    )
    acquire_calls: list[dict] = field(default_factory=list)
    transport: Any = None

    def __post_init__(self) -> None:
        if self.transport is None:
            self.transport = _FakeTransport()

    async def acquire_container(self, **kwargs: Any) -> ClusterContainerSession:
        self.acquire_calls.append(kwargs)
        return ClusterContainerSession(self.transport, self.next_acquire)


@dataclass
class _FakeTransport:
    destroy_calls: list[dict] = field(default_factory=list)
    exec_calls: list[dict] = field(default_factory=list)
    put_archive_calls: list[dict] = field(default_factory=list)
    get_archive_calls: list[dict] = field(default_factory=list)
    next_exec_result: dict[str, Any] = field(default_factory=lambda: {
        "exit_code": 0, "stdout": b"hi\n", "stderr": b"",
        "timed_out": False,
    })
    next_get_archive: bytes = b"<tar bytes>"
    # P1.7.B streaming: chunks the exec_stream iterator should
    # yield. Each must be a RawExecChunk-shaped object; tests
    # build their own.
    next_exec_stream_chunks: list[Any] = field(default_factory=list)

    async def container_exec(self, **kwargs: Any) -> Any:
        self.exec_calls.append(kwargs)
        return dict(self.next_exec_result)

    async def container_put_archive(self, **kwargs: Any) -> None:
        self.put_archive_calls.append(kwargs)

    async def container_get_archive(self, **kwargs: Any) -> bytes:
        self.get_archive_calls.append(kwargs)
        return self.next_get_archive

    def container_exec_stream(self, **kwargs: Any) -> Any:
        chunks = list(self.next_exec_stream_chunks)

        async def _gen() -> Any:
            for c in chunks:
                yield c
        return _gen()

    async def destroy_container(self, **kwargs: Any) -> None:
        self.destroy_calls.append(kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# Cluster control + drop-in construction
# ──────────────────────────────────────────────────────────────────────────────


def test_from_env_with_client_kwarg_constructs_cluster_drop_in() -> None:
    """``xrlenv.from_env(client=...)`` returns a drop-in in
    cluster mode. The high-level XrlenvDockerClient + low-level
    XrlenvAPIClient both carry the cluster control."""
    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]

    assert isinstance(client, XrlenvDockerClient)
    assert isinstance(client.api, XrlenvAPIClient)
    assert client.api._control.mode == "cluster"  # type: ignore[attr-defined]


def test_cluster_control_session_registry() -> None:
    """register / get / drop session lifecycle on the control."""
    fake = _FakeClient()
    ctrl = ClusterContainerControl(client=fake)  # type: ignore[arg-type]
    session = ClusterContainerSession(
        _FakeTransport(),
        RawAcquireResult(
            rollout_id="r-A", container_id="cid-A",
            container_name="n-A", node_id="node-A",
        ),
    )
    ctrl.register_session(session, image="busybox:1")
    assert ctrl.get_session("cid-A") is session
    assert ctrl.get_image("cid-A") == "busybox:1"
    ctrl.drop_session("cid-A")
    import docker.errors
    with pytest.raises(docker.errors.NotFound):
        ctrl.get_session("cid-A")


# ──────────────────────────────────────────────────────────────────────────────
# exec_inspect — Pid / Running contract for a timeout watchdog (audit M2)
# ──────────────────────────────────────────────────────────────────────────────


def test_exec_inspect_completed_carries_exitcode_and_pid() -> None:
    api = from_env(client=_FakeClient()).api  # type: ignore[arg-type]
    api._exec_results["e-done"] = {"exit_code": 3}  # type: ignore[attr-defined]
    info = api.exec_inspect("e-done")
    assert info["ExitCode"] == 3
    assert info["Running"] is False
    assert info["Pid"] == 0  # cluster exec has no host PID; 0 = a real docker finished exec


def test_exec_inspect_in_flight_reports_running_not_notfound() -> None:
    # swebench's exec_run_with_timeout inspects a still-streaming exec to read ["Pid"]
    # before killing it on a test timeout; NotFound there crashes the watchdog cleanup.
    api = from_env(client=_FakeClient()).api  # type: ignore[arg-type]
    api._exec_streaming.add("e-live")  # type: ignore[attr-defined]
    info = api.exec_inspect("e-live")
    assert info["Running"] is True
    assert info["Pid"] == 0


def test_exec_inspect_unknown_exec_still_raises_notfound() -> None:
    api = from_env(client=_FakeClient()).api  # type: ignore[arg-type]
    import docker.errors
    with pytest.raises(docker.errors.NotFound):
        api.exec_inspect("e-never-existed")


# ──────────────────────────────────────────────────────────────────────────────
# create_container → acquire_container
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class _RaisingClient:
    """A ``Client`` stand-in whose acquire raises a chosen exception (audit M8)."""

    exc: BaseException

    async def acquire_container(self, **_kwargs: Any) -> Any:
        raise self.exc


def test_create_container_records_infra_failure_at_acquire() -> None:
    # audit M8: an infra-transient exception raised at acquire is stamped into the drop-in's
    # STRUCTURED record (keyed by the rollout displayed_name) BEFORE it propagates — so a
    # consumer reads the real type even after the harness wraps/swallows it. It is still
    # re-raised (not swallowed here).
    from xrlenv.compat.metadata import rollout_metadata
    from xrlenv.errors import NodeLost

    client = from_env(client=_RaisingClient(NodeLost("node dropped")))  # type: ignore[arg-type]
    with rollout_metadata(displayed_name="inst-a"), pytest.raises(NodeLost):
        client.api.create_container("busybox:latest", command=["true"])
    # popped exactly once; the real infra type, not a wrapper/log guess.
    assert client.api.take_infra_failure("inst-a") == "NodeLost"
    assert client.api.take_infra_failure("inst-a") is None


def test_create_container_records_pin_capacity_subclass() -> None:
    # audit M8: a PinCapacityExhausted (subclass of CapacityExhausted) is caught by the base
    # type and recorded under its real name — the consumer's _InfraFailure retries regardless.
    from xrlenv.compat.metadata import rollout_metadata
    from xrlenv.errors import PinCapacityExhausted

    client = from_env(client=_RaisingClient(PinCapacityExhausted("no free core")))  # type: ignore[arg-type]
    with rollout_metadata(displayed_name="inst-b"), pytest.raises(PinCapacityExhausted):
        client.api.create_container("busybox:latest", command=["true"])
    assert client.api.take_infra_failure("inst-b") == "PinCapacityExhausted"


def test_create_container_does_not_record_non_infra_exception() -> None:
    # a genuine non-infra error at acquire must NOT be recorded as infra (no false retry).
    from xrlenv.compat.metadata import rollout_metadata

    client = from_env(client=_RaisingClient(ValueError("bad image ref")))  # type: ignore[arg-type]
    with rollout_metadata(displayed_name="inst-c"), pytest.raises(ValueError):
        client.api.create_container("busybox:latest", command=["true"])
    assert client.api.take_infra_failure("inst-c") is None


def test_create_container_forwards_cpu_isolation_from_rollout_metadata() -> None:
    # P2/P6: the drop-in forwards ``rollout_metadata(cpu_isolation=...)`` straight to
    # ``acquire_container`` so a docker-py-drop-in harness (swebench) can pin a SPECIFIC
    # task's container to whole cores per-task (the OpenMP-oversubscription fix).
    from xrlenv.backends.base import CpuIsolation
    from xrlenv.compat.metadata import rollout_metadata

    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]
    with rollout_metadata(displayed_name="inst-pin", cpu_isolation=CpuIsolation.BEST_EFFORT):
        client.api.create_container("busybox:latest", command=["true"])
    assert fake.acquire_calls[0]["cpu_isolation"] is CpuIsolation.BEST_EFFORT


def test_create_container_cpu_isolation_defaults_off() -> None:
    # No hint (or no rollout_metadata context) → acquire gets OFF: unchanged, no pinning.
    from xrlenv.backends.base import CpuIsolation
    from xrlenv.compat.metadata import rollout_metadata

    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]
    with rollout_metadata(displayed_name="inst-nopin"):
        client.api.create_container("busybox:latest", command=["true"])
    assert fake.acquire_calls[0]["cpu_isolation"] is CpuIsolation.OFF

    fake2 = _FakeClient()
    client2 = from_env(client=fake2)  # type: ignore[arg-type]
    client2.api.create_container("busybox:latest", command=["true"])   # no context at all
    assert fake2.acquire_calls[0]["cpu_isolation"] is CpuIsolation.OFF


def test_create_container_infra_without_displayed_name_is_not_recorded() -> None:
    # no rollout metadata (no correlation key) -> still raises, but nothing to key the record
    # on, so no orphan entry accumulates.
    from xrlenv.errors import NodeLost

    client = from_env(client=_RaisingClient(NodeLost("x")))  # type: ignore[arg-type]
    with pytest.raises(NodeLost):
        client.api.create_container("busybox:latest", command=["true"])
    assert client.api.take_infra_failure("inst-a") is None


def test_infra_kind_extracts_type_and_wire_prefix() -> None:
    # audit M8: _infra_kind recovers the concrete kind — the exception type name normally, and
    # from the CP's structured "remote <op> <Kind>:" wire prefix (EVERY op shape) when it was
    # flattened to a bare XRLEnvError across the node-control hop.
    from xrlenv.compat.docker_client import _infra_kind
    from xrlenv.errors import NodeLost, XRLEnvError

    assert _infra_kind(NodeLost("x")) == "NodeLost"
    # all three production wire shapes (command / stream / get_archive):
    assert _infra_kind(
        XRLEnvError("node node-A: remote command CapacityExhausted: pool full"),
    ) == "CapacityExhausted"
    assert _infra_kind(
        XRLEnvError("remote stream NodeLost: node dropped mid-eval"),   # swebench /eval.sh path
    ) == "NodeLost"
    assert _infra_kind(
        XRLEnvError("node node-A: remote get_archive NodeLost: link lost"),
    ) == "NodeLost"
    assert _infra_kind(XRLEnvError("just a plain cluster error")) == "XRLEnvError"
    assert _infra_kind(ValueError("v")) == "ValueError"


def test_infra_kind_not_spoofed_by_embedded_prose() -> None:
    # audit Low: the prefix is anchored to the EXACT CP shape (^ or "node <id>: "), so even a
    # VALID op keyword embedded mid-prose can't spoof a recovered kind.
    from xrlenv.compat.docker_client import _infra_kind
    from xrlenv.errors import XRLEnvError

    # a VALID op keyword ("remote command NodeLost:") but preceded by ordinary prose (not
    # "node <id>: " and not ^) must NOT be recovered.
    assert _infra_kind(
        XRLEnvError("assertion failed: the log said remote command NodeLost: earlier"),
    ) == "XRLEnvError"
    # a made-up op keyword likewise doesn't match.
    assert _infra_kind(
        XRLEnvError("node n-1: remote frobnicate NodeLost: made up"),
    ) == "XRLEnvError"
    # audit Low: a full CP-shaped clause embedded MID-message (e.g. inside a non-retryable
    # "gRPC error UNKNOWN: ..." string) must NOT be recovered — the match is anchored at start.
    assert _infra_kind(
        XRLEnvError("gRPC error UNKNOWN: worker said node fake: remote command NodeLost: boom"),
    ) == "XRLEnvError"
    # audit Low: the node-id slot is a SINGLE token, so multi-token "arbitrary prose" stuffed
    # into it (``node <words with spaces>: remote command NodeLost:``) can NOT spoof a kind.
    assert _infra_kind(
        XRLEnvError("node arbitrary prose here: remote command NodeLost: boom"),
    ) == "XRLEnvError"
    # but the exact CP shapes (message STARTS with the prefix) DO still match — incl. a
    # single-token node id that is a hostname / IP:
    assert _infra_kind(XRLEnvError("node n-1: remote command NodeLost: real")) == "NodeLost"
    assert _infra_kind(XRLEnvError("node internal-ip: remote get_archive NodeLost: real")) == "NodeLost"
    assert _infra_kind(XRLEnvError("remote stream NodeLost: real")) == "NodeLost"


def test_create_container_records_wire_flattened_kind() -> None:
    # audit M8 gap 1: a node-side infra failure flattened to a bare XRLEnvError (concrete kind
    # only in the "remote command <Kind>:" message) is still recorded under its real kind.
    from xrlenv.compat.metadata import rollout_metadata
    from xrlenv.errors import XRLEnvError

    flat = XRLEnvError("node node-A: remote command NodeLost: heartbeat lost")
    client = from_env(client=_RaisingClient(flat))  # type: ignore[arg-type]
    with rollout_metadata(displayed_name="inst-w"), pytest.raises(XRLEnvError):
        client.api.create_container("busybox:latest", command=["true"])
    assert client.api.take_infra_failure("inst-w") == "NodeLost"


class _RaisingExecTransport(_FakeTransport):
    async def container_exec(self, **_kwargs: Any) -> Any:
        from xrlenv.errors import NodeLost
        raise NodeLost("dropped mid-exec")


def test_exec_records_post_acquire_infra_failure() -> None:
    # audit M8 gap 2: a NodeLost during POST-acquire exec is recorded too — correlated via the
    # container->rollout map built at acquire, so it works even though the exec runs OUTSIDE the
    # consumer's rollout_metadata context (swebench's watchdog thread).
    from xrlenv.compat.metadata import rollout_metadata
    from xrlenv.errors import NodeLost

    fake = _FakeClient(transport=_RaisingExecTransport())
    client = from_env(client=fake)  # type: ignore[arg-type]
    with rollout_metadata(displayed_name="inst-e"):
        client.api.create_container("busybox:latest", command=["true"])   # builds the map
    info = client.api.exec_create("c-1", ["run"])                          # OUTSIDE metadata
    with pytest.raises(NodeLost):
        client.api.exec_start(info["Id"])
    assert client.api.take_infra_failure("inst-e") == "NodeLost"


class _CancelStreamTransport(_FakeTransport):
    def container_exec_stream(self, **_kwargs: Any) -> Any:
        async def _gen() -> Any:
            import asyncio as _asyncio
            raise _asyncio.CancelledError()
            yield None  # unreachable — makes this an async generator
        return _gen()


def test_streaming_cancellation_surfaces_as_error_not_clean_exhaustion() -> None:
    # audit Low: a CancelledError (BaseException) in the drain must reach the consumer as an
    # error — NOT be swallowed into a clean StopIteration with no evidence.
    import asyncio as _asyncio

    from xrlenv.compat.metadata import rollout_metadata

    fake = _FakeClient(transport=_CancelStreamTransport())
    client = from_env(client=fake)  # type: ignore[arg-type]
    with rollout_metadata(displayed_name="inst-x"):
        client.api.create_container("busybox:latest", command=["true"])
    info = client.api.exec_create("c-1", ["/eval.sh"])
    with pytest.raises(_asyncio.CancelledError):
        list(client.api.exec_start(info["Id"], stream=True))


class _RaisingArchiveTransport(_FakeTransport):
    async def container_get_archive(self, **_kwargs: Any) -> Any:
        from xrlenv.errors import NodeLost
        raise NodeLost("dropped mid-archive")


def test_get_archive_records_post_acquire_infra_failure() -> None:
    # audit M8 gap 2: the archive boundary records too.
    from xrlenv.compat.metadata import rollout_metadata
    from xrlenv.errors import NodeLost

    fake = _FakeClient(transport=_RaisingArchiveTransport())
    client = from_env(client=fake)  # type: ignore[arg-type]
    with rollout_metadata(displayed_name="inst-g"):
        client.api.create_container("busybox:latest", command=["true"])
    with pytest.raises(NodeLost):
        client.api.get_archive("c-1", "/logs")
    assert client.api.take_infra_failure("inst-g") == "NodeLost"


class _RaisingStreamTransport(_FakeTransport):
    def container_exec_stream(self, **_kwargs: Any) -> Any:
        async def _gen() -> Any:
            from xrlenv.errors import NodeLost
            raise NodeLost("stream dropped mid-eval")
            yield None  # unreachable — makes this an async generator
        return _gen()


def test_streaming_exec_records_post_acquire_infra_failure() -> None:
    # audit M8 (the PRINCIPAL path): swebench runs /eval.sh via exec_start(stream=True). A
    # NodeLost on the streaming drain must be recorded at the mechanism boundary too — the
    # earlier fix only covered batched exec/archive.
    from xrlenv.compat.metadata import rollout_metadata
    from xrlenv.errors import NodeLost

    fake = _FakeClient(transport=_RaisingStreamTransport())
    client = from_env(client=fake)  # type: ignore[arg-type]
    with rollout_metadata(displayed_name="inst-s"):
        client.api.create_container("busybox:latest", command=["true"])
    info = client.api.exec_create("c-1", ["/eval.sh"])
    with pytest.raises(NodeLost):
        list(client.api.exec_start(info["Id"], stream=True))   # draining raises
    assert client.api.take_infra_failure("inst-s") == "NodeLost"


class _RaisingDestroyTransport(_FakeTransport):
    async def destroy_container(self, **_kwargs: Any) -> None:
        from xrlenv.errors import NodeLost
        raise NodeLost("teardown lost")


def test_failed_destroy_forgets_map_without_polluting_eval_evidence() -> None:
    # audit M8 last-write-wins: a swallowed NodeLost during TEARDOWN must NOT be recorded into
    # the eval-evidence channel (recording it would overwrite the eval outcome the adapter acts
    # on). It must still forget the container->rollout map (cleanup runs in a finally).
    from xrlenv.compat.metadata import rollout_metadata
    from xrlenv.errors import NodeLost

    fake = _FakeClient(transport=_RaisingDestroyTransport())
    client = from_env(client=fake)  # type: ignore[arg-type]
    with rollout_metadata(displayed_name="inst-t"):
        client.api.create_container("busybox:latest", command=["true"])
    assert "c-1" in client.api._container_rollout  # type: ignore[attr-defined]
    with pytest.raises(NodeLost):
        client.api.remove_container("c-1", force=True)
    assert "c-1" not in client.api._container_rollout  # forgotten despite failure  # type: ignore[attr-defined]
    # Teardown did NOT record into the eval channel.
    assert client.api.take_infra_failure("inst-t") is None


def test_teardown_failure_does_not_overwrite_eval_evidence() -> None:
    # audit M8 last-write-wins (the reproduced case): a retryable EVAL NodeLost followed by a
    # swallowed TEARDOWN failure must keep the eval evidence — the adapter still sees NodeLost.
    from xrlenv.compat.metadata import rollout_metadata
    from xrlenv.errors import NodeLost

    fake = _FakeClient(transport=_RaisingExecTransport())   # exec raises NodeLost (eval stage)
    client = from_env(client=fake)  # type: ignore[arg-type]
    with rollout_metadata(displayed_name="inst-o"):
        client.api.create_container("busybox:latest", command=["true"])
    info = client.api.exec_create("c-1", ["/eval.sh"])
    with pytest.raises(NodeLost):
        client.api.exec_start(info["Id"])              # records eval NodeLost

    # Now teardown ALSO fails — must NOT overwrite the eval NodeLost.
    async def _boom_destroy(**_k: Any) -> None:
        from xrlenv.errors import XRLEnvError
        raise XRLEnvError("teardown transport blip")
    fake.transport.destroy_container = _boom_destroy   # type: ignore[attr-defined]
    with pytest.raises(Exception):  # noqa: B017 — teardown re-raises; identity not asserted
        client.api.remove_container("c-1", force=True)

    assert client.api.take_infra_failure("inst-o") == "NodeLost"  # eval evidence preserved


def test_infra_failures_record_is_bounded() -> None:
    # audit M8 Low: the record dict is bounded — a consumer that never pops can't grow it
    # without limit. Record past the cap and confirm size stays capped.
    from xrlenv.compat.docker_client import _MAX_INFRA_RECORDS

    client = from_env(client=_FakeClient())  # type: ignore[arg-type]
    for i in range(_MAX_INFRA_RECORDS + 50):
        client.api._record_infra_failure(f"k{i}", "NodeLost")  # type: ignore[attr-defined]
    assert len(client.api._infra_failures) <= _MAX_INFRA_RECORDS  # type: ignore[attr-defined]


def test_destroy_forgets_container_rollout_map() -> None:
    # audit M8 Low: the container->rollout association is lifecycle-bounded — dropped on remove.
    from xrlenv.compat.metadata import rollout_metadata

    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]
    with rollout_metadata(displayed_name="inst-d"):
        client.api.create_container("busybox:latest", command=["true"])
    assert "c-1" in client.api._container_rollout  # type: ignore[attr-defined]
    client.api.remove_container("c-1", force=True)
    assert "c-1" not in client.api._container_rollout  # type: ignore[attr-defined]


def test_create_container_acquires_via_client() -> None:
    """The drop-in's create_container call routes to
    ``Client.acquire_container`` and returns the docker-API-shaped
    ``{"Id": <container_id>}`` dict."""
    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]

    result = client.api.create_container(
        "busybox:latest",
        command=["sleep", "infinity"],
        labels={"my-label": "abc"},
    )

    assert result == {"Id": "c-1", "Warnings": []}
    assert len(fake.acquire_calls) == 1
    call = fake.acquire_calls[0]
    assert call["image"] == "busybox:latest"
    assert call["command"] == ["sleep", "infinity"]
    assert call["labels"] == {"my-label": "abc"}


def test_create_container_normalises_string_command() -> None:
    """docker-py accepts ``command="echo hi"`` (string); we split
    on whitespace before forwarding to acquire_container which
    expects a list."""
    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]

    client.api.create_container("busybox:1", command="echo hi")
    assert fake.acquire_calls[0]["command"] == ["echo", "hi"]


def test_create_container_warns_on_unsupported_kwargs(caplog: pytest.LogCaptureFixture) -> None:
    """Cluster-mode ``create_container`` previously silently swallowed
    docker-py kwargs (``volumes``, ``working_dir``, ``platform``, ...)
    that the raw-container path doesn't forward. The warning makes the
    integration gap visible to operators so a harness depending on
    e.g. ``volumes=...`` doesn't fail mysteriously at runtime.
    ``entrypoint`` is NOT in this set anymore — it has explicit
    proto-level forwarding now (test in
    ``test_create_container_forwards_entrypoint_to_acquire``)."""
    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]

    with caplog.at_level("WARNING", logger="xrlenv.compat.docker_client"):
        client.api.create_container(
            "busybox:1",
            command=["sleep", "infinity"],
            volumes={"/host": {"bind": "/ctr", "mode": "ro"}},
            working_dir="/work",
            platform="linux/amd64",
        )

    msgs = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("create_container" in m for m in msgs), msgs
    combined = "\n".join(msgs)
    for expected in ("volumes", "working_dir", "platform"):
        assert expected in combined, f"missing {expected!r} in warning: {combined!r}"
    # entrypoint is explicit now — must NOT appear in the unused warning.
    assert "entrypoint" not in combined, (
        f"entrypoint is a supported kwarg now and should not warn: {combined!r}"
    )


def test_kwarg_warnings_dedup_across_repeated_creates(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A harness that creates hundreds of containers with the SAME unsupported kwargs
    must warn ONCE, not once per create (the swebench-log-flood fix). The unused-kwargs
    and platform-reject warnings each dedup per unique signature per process."""
    with caplog.at_level("WARNING", logger="xrlenv.compat.docker_client"):
        _docker_client_mod._warn_unused_kwargs("create_container", {"volumes": {}})
        _docker_client_mod._warn_unused_kwargs("create_container", {"volumes": {}})
        _docker_client_mod._reject_platform_kwarg("linux/amd64")
        _docker_client_mod._reject_platform_kwarg("linux/amd64")
    warns = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warns) == 2, [r.getMessage() for r in warns]   # one per signature, not 4
    # a DIFFERENT signature still warns (dedup is per-signature, not global-mute).
    with caplog.at_level("WARNING", logger="xrlenv.compat.docker_client"):
        caplog.clear()
        _docker_client_mod._warn_unused_kwargs("create_container", {"working_dir": "/w"})
    assert len([r for r in caplog.records if r.levelname == "WARNING"]) == 1


def test_create_container_forwards_user_to_acquire() -> None:
    """``user="root"`` (the swebench grader's DOCKER_USER constant)
    flows through to ``acquire_container(user=...)``. Without this
    forwarding, the grader's containers ran as whatever USER the
    image declared — usually root anyway, but the explicit kwarg
    documents the harness's intent and would matter on non-root
    eval images."""
    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]

    client.api.create_container(
        "busybox:1", command=["sleep", "infinity"], user="root",
    )
    assert fake.acquire_calls[-1]["user"] == "root"

    # Empty / None user → not on the wire (image's USER directive wins).
    client.api.create_container("busybox:1", command=["sleep", "infinity"])
    assert fake.acquire_calls[-1]["user"] is None
    client.api.create_container("busybox:1", command=["sleep", "infinity"], user="")
    assert fake.acquire_calls[-1]["user"] is None


def test_create_container_extracts_cap_add_from_host_config() -> None:
    """docker-py's high-level ``containers.create(cap_add=[...])``
    bundles into ``host_config["CapAdd"]`` before reaching the
    low-level API. The cluster path extracts it back out and
    forwards as ``acquire_container(cap_add=...)`` so SWE-bench
    repos that need e.g. ``SYS_PTRACE`` for debugger tests get the
    capability — silently dropping it caused wrong grader verdicts
    (patch correct, test setup wrong)."""
    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]

    client.api.create_container(
        "busybox:1",
        command=["sleep", "infinity"],
        host_config={"CapAdd": ["SYS_PTRACE", "NET_ADMIN"]},
    )
    assert fake.acquire_calls[-1]["cap_add"] == ["SYS_PTRACE", "NET_ADMIN"]

    # Empty CapAdd → None on the wire (default capabilities).
    client.api.create_container(
        "busybox:1",
        command=["sleep", "infinity"],
        host_config={"CapAdd": []},
    )
    assert fake.acquire_calls[-1]["cap_add"] is None


def test_create_container_forwards_runtime_from_host_config_and_kwarg() -> None:
    """§5.4 — docker-py maps ``containers.run(runtime="sysbox-runc")`` and
    ``HostConfig(Runtime=...)`` onto ``host_config["Runtime"]``. The cluster
    path extracts it and forwards as ``acquire_container(container_runtime=...)``
    so a DinD/systemd task opts into Sysbox. Also honors a top-level ``runtime=``
    kwarg; HostConfig.Runtime wins when both are present."""
    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]

    # HostConfig.Runtime (the docker-py containers.run path).
    client.api.create_container(
        "dind:1", command=["sleep", "infinity"],
        host_config={"Runtime": "sysbox-runc"},
    )
    assert fake.acquire_calls[-1]["container_runtime"] == "sysbox-runc"

    # Top-level runtime= kwarg (low-level api.create_container callers).
    client.api.create_container(
        "dind:1", command=["sleep", "infinity"], runtime="sysbox-runc",
    )
    assert fake.acquire_calls[-1]["container_runtime"] == "sysbox-runc"

    # HostConfig.Runtime wins over the top-level kwarg when both are set.
    client.api.create_container(
        "dind:1", command=["sleep", "infinity"],
        runtime="runc", host_config={"Runtime": "sysbox-runc"},
    )
    assert fake.acquire_calls[-1]["container_runtime"] == "sysbox-runc"

    # No runtime → None (the ordinary runc path is unchanged).
    client.api.create_container("busybox:1", command=["sleep", "infinity"])
    assert fake.acquire_calls[-1]["container_runtime"] is None


def test_create_container_forwards_devices_from_host_config() -> None:
    """docker-py's high-level ``containers.create(devices=[...])``
    bundles the device list into ``host_config["Devices"]`` as a list
    of dicts. The cluster path round-trips back to CLI-style spec
    strings and forwards via ``acquire_container(devices=...)`` so
    SCUBA-style nested-VM benchmarks (``/dev/kvm``) work without
    operator config. Default policy allows /dev/kvm + /dev/net/tun
    + /dev/fuse out of the box."""
    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]

    # Common case: docker-py-bundled dict form.
    client.api.create_container(
        "busybox:1",
        command=["sleep", "infinity"],
        host_config={
            "Devices": [{
                "PathOnHost": "/dev/kvm",
                "PathInContainer": "/dev/kvm",
                "CgroupPermissions": "rwm",
            }],
        },
    )
    assert fake.acquire_calls[-1]["devices"] == ["/dev/kvm:rwm"]

    # Raw-string form (caller bypassed the high-level manager).
    client.api.create_container(
        "busybox:1",
        command=["sleep", "infinity"],
        host_config={"Devices": ["/dev/fuse"]},
    )
    assert fake.acquire_calls[-1]["devices"] == ["/dev/fuse"]

    # Empty / missing → None on the wire.
    client.api.create_container(
        "busybox:1",
        command=["sleep", "infinity"],
        host_config={"Devices": []},
    )
    assert fake.acquire_calls[-1]["devices"] is None


def test_create_container_forwards_unlisted_device_to_wire() -> None:
    """Drop-in does NOT validate Level-1 device kwargs against
    DEFAULT_POLICY (audit M1): the cluster policy may extend
    ``allowed_devices`` and the drop-in can't see that. Unlisted
    devices flow to the wire; the control-plane coordinator is the
    sole authoritative validator. Tests for the coordinator-side
    rejection live in ``test_raw_container_coordinator.py``."""
    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]

    client.api.create_container(
        "busybox:1",
        command=["sleep", "infinity"],
        host_config={"Devices": [{
            "PathOnHost": "/dev/sda",
            "PathInContainer": "/dev/sda",
            "CgroupPermissions": "rwm",
        }]},
    )
    # Forwarded — control plane decides.
    assert fake.acquire_calls[-1]["devices"] == ["/dev/sda:rwm"]


def test_create_container_forwards_privileged_to_wire() -> None:
    """Drop-in does NOT pre-reject Level-2 ``Privileged=True`` (audit
    M1): the operator's ``allow_privileged: true`` opt-in lives on the
    control plane, invisible to the drop-in. The flag flows to the
    wire; the coordinator validates against the cluster policy."""
    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]

    client.api.create_container(
        "busybox:1",
        command=["sleep", "infinity"],
        host_config={"Privileged": True},
    )
    assert fake.acquire_calls[-1]["privileged"] is True


def test_create_container_forwards_network_mode_host_to_wire() -> None:
    """Level-2 ``NetworkMode=host`` flows through; coordinator decides
    against ``allow_host_network``."""
    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]

    client.api.create_container(
        "busybox:1",
        command=["sleep", "infinity"],
        host_config={"NetworkMode": "host"},
    )
    assert fake.acquire_calls[-1]["network_mode"] == "host"


def test_create_container_forwards_binds_to_wire() -> None:
    """Level-2 ``Binds`` flow through; coordinator decides against
    ``allowed_host_paths``."""
    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]

    client.api.create_container(
        "busybox:1",
        command=["sleep", "infinity"],
        host_config={"Binds": ["/mnt/datasets:/data:ro"]},
    )
    assert fake.acquire_calls[-1]["binds"] == ["/mnt/datasets:/data:ro"]


def test_create_container_rejects_pid_mode_host() -> None:
    """Level-3 namespace escape — no policy override available, so the
    drop-in fast-fails locally (control plane would reject anyway)."""
    from xrlenv.control.kwargs_policy import KwargsPolicyViolation

    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]

    with pytest.raises(KwargsPolicyViolation) as ei:
        client.api.create_container(
            "busybox:1",
            command=["sleep", "infinity"],
            host_config={"PidMode": "host"},
        )
    msg = str(ei.value)
    assert "pid_mode" in msg
    assert "level 3" in msg
    # Drop-in fast-fails BEFORE the acquire RPC.
    assert fake.acquire_calls == []


def test_create_container_rejects_ipc_mode_host() -> None:
    """Level-3 IPC namespace escape — drop-in fast-fails."""
    from xrlenv.control.kwargs_policy import KwargsPolicyViolation

    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]

    with pytest.raises(KwargsPolicyViolation):
        client.api.create_container(
            "busybox:1",
            command=["sleep", "infinity"],
            host_config={"IpcMode": "host"},
        )
    assert fake.acquire_calls == []


def test_create_container_rejects_network_mode_container() -> None:
    """Level-3 ``NetworkMode=container:other`` — drop-in fast-fails.
    Distinct from Level-2 ``NetworkMode=host`` which is deferred to
    the control plane."""
    from xrlenv.control.kwargs_policy import KwargsPolicyViolation

    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]

    with pytest.raises(KwargsPolicyViolation) as ei:
        client.api.create_container(
            "busybox:1",
            command=["sleep", "infinity"],
            host_config={"NetworkMode": "container:other"},
        )
    assert "network_mode" in str(ei.value)
    assert fake.acquire_calls == []


def test_create_container_level_3_only_short_circuits_l1_l2() -> None:
    """When a request mixes Level-1/2/3 kwargs, the drop-in raises
    ONLY for the Level-3 violations (the others flow to the wire if
    the request retries without the L3). This preserves the audit-M1
    invariant: drop-in never rejects on Level 1 / Level 2."""
    from xrlenv.control.kwargs_policy import KwargsPolicyViolation

    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]

    with pytest.raises(KwargsPolicyViolation) as ei:
        client.api.create_container(
            "busybox:1",
            command=["sleep", "infinity"],
            host_config={
                "Devices": [{
                    "PathOnHost": "/dev/sda",
                    "PathInContainer": "/dev/sda",
                    "CgroupPermissions": "rwm",
                }],
                "Privileged": True,
                "PidMode": "host",  # Level 3 — only this is in the error.
            },
        )
    msg = str(ei.value)
    assert "pid_mode" in msg
    # Level 1 (/dev/sda) and Level 2 (privileged) are NOT pre-rejected
    # at the drop-in — they'd flow to the wire if pid_mode were absent.
    assert "/dev/sda" not in msg
    assert "privileged" not in msg


def test_create_container_forwards_allowed_cap_and_device_together() -> None:
    """SCUBA-style: ``cap_add=[NET_ADMIN]`` + ``devices=[/dev/kvm]``
    flow through cleanly with default policy. Regression guard for the
    real-world unblock that motivated issue #6's Option D."""
    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]

    client.api.create_container(
        "busybox:1",
        command=["sleep", "infinity"],
        host_config={
            "CapAdd": ["NET_ADMIN"],
            "Devices": [{
                "PathOnHost": "/dev/kvm",
                "PathInContainer": "/dev/kvm",
                "CgroupPermissions": "rwm",
            }],
        },
    )
    assert fake.acquire_calls[-1]["cap_add"] == ["NET_ADMIN"]
    assert fake.acquire_calls[-1]["devices"] == ["/dev/kvm:rwm"]


def test_create_container_host_config_warning_excludes_consumed_fields(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The unused-kwargs warning consumes ``Devices``, ``Privileged``,
    ``NetworkMode``, ``PidMode``, ``IpcMode``, ``CgroupParent``, ``Binds``,
    and (P0a) the CPU/memory resource keys. Only host_config entries the
    cluster path genuinely doesn't honor surface in the warning."""
    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]

    with caplog.at_level("WARNING", logger="xrlenv.compat.docker_client"):
        client.api.create_container(
            "busybox:1",
            command=["sleep", "infinity"],
            host_config={
                "Devices": [{
                    "PathOnHost": "/dev/kvm",
                    "PathInContainer": "/dev/kvm",
                    "CgroupPermissions": "rwm",
                }],
                "NetworkMode": "bridge",
                "Memory": 4_000_000_000,   # P0a — now consumed
                "OomKillDisable": True,    # genuinely not wired
            },
        )
    msgs = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    combined = "\n".join(msgs)
    # Honored fields don't appear in the warning.
    assert "Devices" not in combined
    assert "NetworkMode" not in combined
    # P0a — Memory is now consumed (effective ResourceSpec), not dropped.
    assert "Memory" not in combined
    # Genuinely-unwired fields still appear.
    assert "OomKillDisable" in combined


def test_create_container_rejects_platform_with_rationale(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``platform`` is operator-controlled at deploy time — each
    node's docker daemon serves a single architecture and the
    scheduler routes to whichever node matches. A consumer-side
    hint can't change that. The cluster path emits a TARGETED
    warning (distinct from the generic _warn_unused_kwargs) so
    operators understand this is intentional rejection, not a
    pending wire-up. See issue #6 for the design rationale."""
    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]

    with caplog.at_level("WARNING", logger="xrlenv.compat.docker_client"):
        client.api.create_container(
            "busybox:1",
            command=["sleep", "infinity"],
            platform="linux/x86_64",
        )

    # The targeted message is present (mentions "operator-controlled"
    # + "intentional rejection") and the value is included for the
    # operator to see what was rejected.
    msgs = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    targeted = [m for m in msgs if "platform=" in m and "operator-controlled" in m]
    assert targeted, (
        f"expected targeted platform-rejection warning, got: {msgs!r}"
    )
    assert "'linux/x86_64'" in targeted[0]
    assert "intentional" in targeted[0]
    assert "issue #6" in targeted[0]

    # platform is NOT in _warn_unused_kwargs's "ignoring docker-py
    # kwargs" generic message — that would falsely imply it's a
    # pending wire-up.
    generic_drop = [
        m for m in msgs
        if "ignoring docker-py kwargs" in m and "platform" in m
    ]
    assert not generic_drop, (
        f"platform should not appear in the generic _warn_unused_kwargs "
        f"message — it's intentionally rejected, not 'not yet wired'. "
        f"Got: {generic_drop!r}"
    )


def test_create_container_forwards_entrypoint_to_acquire() -> None:
    """``entrypoint`` flows through the cluster path to
    ``acquire_container(entrypoint=...)`` so the node can pass it to
    ``docker run --entrypoint``. The docker CLI's ``--entrypoint ""``
    "clear ENTRYPOINT" idiom is preserved as the single-element list
    ``[""]`` (proto3 ``repeated`` collapses unset and empty-list, so
    we need a non-empty list to carry the intent)."""
    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]

    # docker-py accepts entrypoint as a string or a list. Both forms
    # should reach acquire_container as a list.
    client.api.create_container(
        "busybox:1",
        command=["sleep", "infinity"],
        entrypoint=["/bin/bash", "-lc"],
    )
    assert fake.acquire_calls[-1]["entrypoint"] == ["/bin/bash", "-lc"]

    client.api.create_container(
        "busybox:1", command=["sleep", "infinity"], entrypoint="/bin/sh",
    )
    assert fake.acquire_calls[-1]["entrypoint"] == ["/bin/sh"]

    # The "clear it" idiom — empty-string OR empty-list, normalized
    # to [""] so proto3 can carry it.
    client.api.create_container(
        "busybox:1", command=["sleep", "infinity"], entrypoint="",
    )
    assert fake.acquire_calls[-1]["entrypoint"] == [""]

    # ``entrypoint=None`` (or unset) → None on the wire; image
    # default is preserved.
    client.api.create_container("busybox:1", command=["sleep", "infinity"])
    assert fake.acquire_calls[-1]["entrypoint"] is None


def test_create_container_quiet_kwargs_do_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    """``detach`` is the only kwarg the audience routinely passes
    that has no cluster-side effect (acquire_container forces it
    on); we suppress the warning for it to keep noise down."""
    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]

    with caplog.at_level("WARNING", logger="xrlenv.compat.docker_client"):
        client.api.create_container(
            "busybox:1", command=["sleep", "infinity"], detach=True,
        )

    msgs = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert not any("create_container" in m for m in msgs), (
        f"unexpected warning: {msgs!r}"
    )


def test_create_container_promotes_xrlenv_task_key_label_to_kwarg() -> None:
    """Operators pass ``xrlenv.task_key`` as a docker label (the
    conventional channel from coding-bench / swebench harnesses).
    Cluster mode pops it off ``labels`` and promotes it to
    ``acquire_container(task_key=...)`` — the scheduler's
    anti-affinity reads the AcquireContainerRequest field, not the
    label dict, so this lifting is load-bearing for fairness."""
    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]

    client.api.create_container(
        "busybox:1",
        command=["sleep", "infinity"],
        labels={"xrlenv.task_key": "astropy__astropy-7166", "other": "x"},
    )

    call = fake.acquire_calls[0]
    assert call["task_key"] == "astropy__astropy-7166"
    # The task_key entry is popped from labels (the scheduler kwarg
    # owns it now); unrelated operator labels remain.
    assert call["labels"] == {"other": "x"}


def test_create_container_no_task_key_when_label_absent() -> None:
    """Backwards-compatible: without the reserved label, task_key is
    None and acquire_container falls through to its default
    anti-affinity behavior (no key = no spread)."""
    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]

    client.api.create_container(
        "busybox:1", command=["sleep", "infinity"], labels={"other": "x"},
    )

    assert fake.acquire_calls[0]["task_key"] is None
    assert fake.acquire_calls[0]["labels"] == {"other": "x"}


def test_create_container_task_key_only_label_clears_labels_arg() -> None:
    """If ``xrlenv.task_key`` is the *only* label, popping it leaves
    an empty dict; we send ``labels=None`` to acquire_container so we
    don't waste a no-op proto map entry."""
    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]

    client.api.create_container(
        "busybox:1",
        command=["sleep", "infinity"],
        labels={"xrlenv.task_key": "tb2/oracle"},
    )

    call = fake.acquire_calls[0]
    assert call["task_key"] == "tb2/oracle"
    assert call["labels"] is None


def test_start_is_a_no_op_in_cluster_mode() -> None:
    """``container.start()`` is a no-op — acquire already
    spawned + detached the container. Returns None per docker-py
    contract."""
    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]

    assert client.api.start("c-1") is None


def test_stop_is_a_no_op_in_cluster_mode() -> None:
    """``container.stop()`` no-op — the harness usually does
    stop+remove; remove does the actual ``docker rm -f``."""
    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]

    assert client.api.stop("c-1") is None


def test_remove_container_destroys_via_session() -> None:
    """``container.remove()`` routes to the session's destroy.
    After remove, the control's session map no longer has the
    container."""
    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]
    client.api.create_container("busybox:1")  # registers session

    client.api.remove_container("c-1", force=True)

    assert len(fake.transport.destroy_calls) == 1
    assert fake.transport.destroy_calls[0]["rollout_id"] == "r-1"
    # Session is dropped from the control map.
    import docker.errors
    with pytest.raises(docker.errors.NotFound):
        client.api._cluster_control.get_session("c-1")  # type: ignore[attr-defined]


def test_remove_container_idempotent_on_unknown_id() -> None:
    """Removing a container the drop-in doesn't know about is a
    silent no-op (matches docker-py's tolerance for already-
    removed containers)."""
    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]

    # No prior create_container — nothing in the session map.
    assert client.api.remove_container("ghost-id", force=True) is None


def test_inspect_container_returns_synthetic_state() -> None:
    """Inspect returns a docker-API-shaped dict with the labels
    + image + state we know from the session."""
    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]
    client.api.create_container("busybox:1")

    info = client.api.inspect_container("c-1")
    assert info["Id"] == "c-1"
    assert info["Config"]["Image"] == "c-1" or info["Config"]["Image"]
    assert info["Config"]["Labels"]["xrlenv.rollout_id"] == "r-1"
    assert info["State"]["Running"] is True


# ──────────────────────────────────────────────────────────────────────────────
# Unwired calls raise a clean NotImplementedError
# ──────────────────────────────────────────────────────────────────────────────


def test_unwired_api_method_raises_not_implemented() -> None:
    """Methods not in ``_CLUSTER_OVERRIDES`` raise
    NotImplementedError with a clear message instead of failing
    on uninitialized parent state."""
    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]

    with pytest.raises(NotImplementedError, match="not yet implement"):
        client.api.networks(filters={"name": "anything"})


def test_unwired_method_message_lists_supported_methods() -> None:
    """The error message names what IS supported so callers
    know what's wired vs not."""
    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]

    with pytest.raises(NotImplementedError) as exc_info:
        client.api.diff("c-1")
    msg = str(exc_info.value)
    assert "create_container" in msg
    assert "remove_container" in msg


def test_inherited_mixin_methods_also_raise_not_implemented() -> None:
    """Audit Cluster-Dropin-M1 closure. ``exec_create``,
    ``exec_start``, ``put_archive``, ``inspect_image``, ``pull``
    are all inherited from docker.APIClient's mixins
    (ExecApiMixin / ImageApiMixin / ContainerApiMixin). Earlier
    these would have run against uninitialized parent state and
    failed with confusing AttributeErrors. The
    ``_install_cluster_safety_net`` shadows them at __init__
    time so they raise the same clean NotImplementedError as
    truly-missing methods."""
    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]

    # These are inherited mixin methods we haven't yet wired —
    # exec_*, put_archive, get_archive, image ops all landed in
    # P1.7.B as real overrides. The unwired tail is mostly
    # other-mixin methods (build, networks, ...) we don't expect
    # cluster-mode harnesses to need.
    for inherited_method, args in [
        ("build", ()),
        ("networks", ()),
        ("volumes", ()),
        ("attach", ("c-1",)),
        ("commit", ("c-1",)),
    ]:
        with pytest.raises(NotImplementedError, match=inherited_method):
            getattr(client.api, inherited_method)(*args)


def test_local_mode_inherited_methods_not_shadowed() -> None:
    """The safety net only fires in cluster mode. Local-mode
    drop-in keeps full inherited surface alive (it IS a real
    docker.APIClient with super().__init__ called)."""
    # Default = local mode.
    client = from_env()
    # Inherited methods exist normally — no NotImplementedError.
    # We don't actually call them (would need a docker daemon)
    # but verify they're callable bound methods, not stubs.
    method = client.api.exec_create
    assert callable(method)
    assert "_cluster_stub_" not in getattr(method, "__name__", "")


# ──────────────────────────────────────────────────────────────────────────────
# P1.7.B exec triple — batched exec via api.exec_create + exec_start + exec_inspect
# ──────────────────────────────────────────────────────────────────────────────


def test_exec_create_returns_synthetic_id_and_stages() -> None:
    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]
    client.api.create_container("busybox:1")

    info = client.api.exec_create("c-1", ["echo", "hi"])

    assert info["Id"].startswith("xrlenv-exec-")
    # No transport call yet — exec_start does the real work.
    assert fake.transport.exec_calls == []  # type: ignore[attr-defined]


def test_exec_create_normalises_string_command() -> None:
    """docker-py accepts ``cmd="echo hi"`` (string); we wrap in
    ``sh -c`` so the shell-string semantics match docker-py's
    own behaviour for string commands."""
    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]
    client.api.create_container("busybox:1")

    info = client.api.exec_create("c-1", "echo hi && true")
    client.api.exec_start(info["Id"], demux=True)

    assert fake.transport.exec_calls[0]["cmd"] == [  # type: ignore[attr-defined]
        "sh", "-c", "echo hi && true",
    ]


def test_exec_start_batched_runs_and_returns_demuxed_bytes() -> None:
    fake = _FakeClient()
    fake.transport.next_exec_result = {  # type: ignore[attr-defined]
        "exit_code": 7, "stdout": b"out\n",
        "stderr": b"err\n", "timed_out": False,
    }
    client = from_env(client=fake)  # type: ignore[arg-type]
    client.api.create_container("busybox:1")
    info = client.api.exec_create("c-1", ["echo", "hi"])

    out, err = client.api.exec_start(info["Id"], demux=True)
    assert out == b"out\n"
    assert err == b"err\n"

    inspect = client.api.exec_inspect(info["Id"])
    assert inspect["ExitCode"] == 7
    assert inspect["Running"] is False


def test_exec_start_handles_raw_exec_result_dataclass() -> None:
    """Regression: the real gRPC + in-process transports return a
    ``RawExecResult`` dataclass (not a dict). ``exec_start``
    must normalise either shape; calling ``.get()`` on a
    dataclass would AttributeError. The streaming terminator
    stores a dict already; this keeps ``_exec_results`` shape-
    consistent so ``exec_inspect`` works the same on both
    paths."""
    from xrlenv.control.service import RawExecResult

    @dataclass
    class _DataclassReturningTransport(_FakeTransport):
        async def container_exec(self, **_kwargs: Any) -> Any:  # type: ignore[override]
            return RawExecResult(
                exit_code=3, stdout=b"hello\n",
                stderr=b"warn\n", timed_out=False,
            )

    fake = _FakeClient(transport=_DataclassReturningTransport())
    client = from_env(client=fake)  # type: ignore[arg-type]
    client.api.create_container("busybox:1")
    info = client.api.exec_create("c-1", ["echo", "hi"])

    # Batched + demux'd path round-trips bytes correctly.
    out, err = client.api.exec_start(info["Id"], demux=True)
    assert out == b"hello\n"
    assert err == b"warn\n"

    # exec_inspect reads the normalised dict — no AttributeError.
    inspect = client.api.exec_inspect(info["Id"])
    assert inspect["ExitCode"] == 3
    assert inspect["Running"] is False


def test_exec_start_combined_returns_concatenated_bytes() -> None:
    """``demux=False`` (the default) returns concatenated
    stdout+stderr per docker-py's contract."""
    fake = _FakeClient()
    fake.transport.next_exec_result = {  # type: ignore[attr-defined]
        "exit_code": 0, "stdout": b"out\n",
        "stderr": b"err\n", "timed_out": False,
    }
    client = from_env(client=fake)  # type: ignore[arg-type]
    client.api.create_container("busybox:1")
    info = client.api.exec_create("c-1", ["echo", "hi"])

    combined = client.api.exec_start(info["Id"])
    assert combined == b"out\nerr\n"


def test_exec_start_stream_yields_demuxed_chunks_via_sync_iterator() -> None:
    """``api.exec_start(exec_id, stream=True, demux=True)``
    returns a sync iterator yielding ``(stdout, stderr)``
    tuples per chunk. Heartbeat chunks (empty stdout AND
    stderr, done=False) are filtered. Terminator (done=True)
    ends iteration; exit_code becomes available via
    exec_inspect afterward. Bridges from the async
    ``ClusterContainerSession.exec_stream`` via a daemon
    thread + queue.Queue."""
    from xrlenv.control.service import RawExecChunk
    fake = _FakeClient()
    fake.transport.next_exec_stream_chunks = [  # type: ignore[attr-defined]
        RawExecChunk(stdout=b"a\n", stderr=b"", done=False,
                     exit_code=0, timed_out=False),
        # Heartbeat — should be filtered out.
        RawExecChunk(stdout=b"", stderr=b"", done=False,
                     exit_code=0, timed_out=False),
        RawExecChunk(stdout=b"", stderr=b"oops\n", done=False,
                     exit_code=0, timed_out=False),
        RawExecChunk(stdout=b"c\n", stderr=b"", done=False,
                     exit_code=0, timed_out=False),
        RawExecChunk(stdout=b"", stderr=b"", done=True,
                     exit_code=7, timed_out=False),
    ]
    client = from_env(client=fake)  # type: ignore[arg-type]
    client.api.create_container("busybox:1")
    info = client.api.exec_create("c-1", ["bash", "-c", "echo a; echo c"])

    chunks = list(client.api.exec_start(
        info["Id"], stream=True, demux=True,
    ))

    # 3 real chunks (heartbeat filtered, terminator not yielded).
    assert chunks == [
        (b"a\n", b""),
        (b"", b"oops\n"),
        (b"c\n", b""),
    ]
    # exec_inspect now sees the final exit code from the
    # terminator chunk.
    assert client.api.exec_inspect(info["Id"])["ExitCode"] == 7


def test_exec_start_stream_combined_yields_concatenated_bytes() -> None:
    """``demux=False`` combines stdout+stderr per docker-py
    contract."""
    from xrlenv.control.service import RawExecChunk
    fake = _FakeClient()
    fake.transport.next_exec_stream_chunks = [  # type: ignore[attr-defined]
        RawExecChunk(stdout=b"out\n", stderr=b"err\n",
                     done=False, exit_code=0, timed_out=False),
        RawExecChunk(stdout=b"", stderr=b"", done=True,
                     exit_code=0, timed_out=False),
    ]
    client = from_env(client=fake)  # type: ignore[arg-type]
    client.api.create_container("busybox:1")
    info = client.api.exec_create("c-1", ["echo", "hi"])

    chunks = list(client.api.exec_start(info["Id"], stream=True))
    assert chunks == [b"out\nerr\n"]


def test_exec_start_stream_propagates_async_drain_errors() -> None:
    """If the async ``exec_stream`` raises (gRPC blip, node
    disconnect, chunk-timeout), the sync iterator surfaces the
    exception on the next ``__next__`` call rather than silently
    truncating."""
    fake = _FakeClient()

    # Override the fake transport's container_exec_stream to
    # produce an async generator that raises.
    def _bad_stream(**kwargs: Any) -> Any:
        async def _gen() -> Any:
            raise RuntimeError("transient gRPC blip")
            yield  # unreachable; makes this an async generator
        return _gen()
    fake.transport.container_exec_stream = _bad_stream  # type: ignore[method-assign]

    client = from_env(client=fake)  # type: ignore[arg-type]
    client.api.create_container("busybox:1")
    info = client.api.exec_create("c-1", ["echo", "hi"])

    iterator = client.api.exec_start(info["Id"], stream=True)
    with pytest.raises(RuntimeError, match="transient gRPC blip"):
        next(iterator)


def test_exec_inspect_unknown_id_raises_notfound() -> None:
    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]

    import docker.errors
    with pytest.raises(docker.errors.NotFound):
        client.api.exec_inspect("never-staged")


# ──────────────────────────────────────────────────────────────────────────────
# P1.7.B archives — api.put_archive + api.get_archive
# ──────────────────────────────────────────────────────────────────────────────


def test_put_archive_routes_to_session() -> None:
    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]
    client.api.create_container("busybox:1")

    ok = client.api.put_archive("c-1", "/tmp", b"<tar bytes>")
    assert ok is True
    assert len(fake.transport.put_archive_calls) == 1  # type: ignore[attr-defined]
    call = fake.transport.put_archive_calls[0]  # type: ignore[attr-defined]
    assert call["target_dir"] == "/tmp"
    assert call["tarball"] == b"<tar bytes>"


def test_get_archive_returns_iter_and_stat() -> None:
    fake = _FakeClient()
    fake.transport.next_get_archive = b"<received tar>"  # type: ignore[attr-defined]
    client = from_env(client=fake)  # type: ignore[arg-type]
    client.api.create_container("busybox:1")

    chunks_iter, stat = client.api.get_archive("c-1", "/logs/artifacts")

    chunks = list(chunks_iter)
    assert b"".join(chunks) == b"<received tar>"
    assert stat["name"] == "/logs/artifacts"


# ──────────────────────────────────────────────────────────────────────────────
# P1.7.B image ops — operator-pre-pulled contract
# ──────────────────────────────────────────────────────────────────────────────


def test_pull_is_no_op_with_synthetic_response() -> None:
    """``api.pull`` returns a successful no-op response per the
    operator-pre-pulled contract. Real pull happens via the
    image-cache layer (P1.6.g) at the operator's request, not
    in-band on the consumer side."""
    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]

    result = client.api.pull("busybox", tag="latest")
    assert "no-op" in result
    assert "busybox:latest" in result


def test_pull_stream_returns_iter() -> None:
    """``stream=True`` returns an iterator of progress dicts
    (matches docker-py shape minimally)."""
    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]

    progress = list(client.api.pull("busybox", stream=True))
    assert len(progress) == 1
    assert "no-op" in progress[0]["status"]


def test_inspect_image_returns_synthetic_image() -> None:
    """The harness's "is image local" probe always succeeds in
    cluster mode. Real check happens at acquire_container time."""
    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]

    info = client.api.inspect_image("busybox:latest")
    assert info["RepoTags"] == ["busybox:latest"]
    assert info["Id"].startswith("sha256:")


def test_images_returns_empty_list() -> None:
    """Consumer-side host has no docker images of its own."""
    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]

    assert client.api.images() == []
    assert client.api.images(name="busybox", all=True) == []


def test_containers_lists_registered_sessions() -> None:
    """``api.containers`` in cluster mode enumerates the still-alive
    containers this drop-in instance created (tracked by
    ``ClusterContainerControl._sessions``). swebench's grader uses
    this for its leak-detection summary line (``run_id in name``
    filter) — returning ``[]`` would hide real leaks; returning the
    full Docker daemon listing isn't reachable from the consumer.
    The per-client-instance registry is the meaningful middle:
    every entry is genuinely "still alive from this client's
    perspective" since destroy() drops the session.

    Before this implementation the method raised NotImplementedError
    via the safety-net _stub, truncating swebench's
    make_run_report mid-aggregate."""
    # Empty client → empty list (no containers created yet).
    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]
    assert client.api.containers() == []
    assert client.api.containers(all=True) == []
    assert list(client.containers.list(all=True)) == []

    # After creating two containers via the cluster path, both
    # appear in the listing as docker-API-shaped dicts.
    client.api.create_container(
        "busybox:latest",
        command=["sleep", "infinity"],
        name="grade-instance-A",
    )
    fake.next_acquire = RawAcquireResult(
        rollout_id="r-2", container_id="c-2",
        container_name="grade-instance-B", node_id="node-A",
    )
    client.api.create_container(
        "busybox:latest",
        command=["sleep", "infinity"],
        name="grade-instance-B",
    )

    listed = client.api.containers(all=True)
    assert len(listed) == 2
    by_id = {row["Id"]: row for row in listed}
    assert "c-1" in by_id and "c-2" in by_id
    assert by_id["c-1"]["Image"] == "busybox:latest"
    assert by_id["c-1"]["State"] == "running"
    # Names follow docker-py's leading-slash convention so high-level
    # Container.name (which strips the slash) renders cleanly.
    assert by_id["c-1"]["Names"] == ["/cname-1"]
    assert by_id["c-2"]["Names"] == ["/grade-instance-B"]

    # Destroying drops the session — the row disappears from
    # subsequent listings (the leak-detection contract).
    container_c1 = client.containers.get("c-1")
    container_c1.remove(force=True)
    listed_after = client.api.containers(all=True)
    assert len(listed_after) == 1
    assert listed_after[0]["Id"] == "c-2"

    # High-level docker-py path mirrors: client.containers.list() wraps
    # each dict in a Container model.
    high_level = list(client.containers.list(all=True))
    assert len(high_level) == 1
    assert high_level[0].id == "c-2"
    # Container.name strips the leading slash.
    assert high_level[0].name == "grade-instance-B"


def test_remove_image_is_no_op() -> None:
    """Image lifetime owned by the per-node LRU cache;
    consumer-side ``remove_image`` doesn't unilaterally evict."""
    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]

    assert client.api.remove_image("busybox", force=True) is None


def test_history_returns_empty_list() -> None:
    """swebench uses image.history() for find_dependent_images
    cleanup; empty history → no dependents → no-op cleanup."""
    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]

    assert client.api.history("busybox") == []


# ──────────────────────────────────────────────────────────────────────────────
# High-level docker-py manager surface (audit Cluster-Dropin-M3 closure)
#
# These tests exercise ``client.containers.create`` /
# ``client.containers.get`` / ``container.exec_run`` /
# ``container.put_archive`` / ``container.remove`` — the path
# harnesses actually take. Earlier the high-level managers
# AttributeError'd before reaching our api.* overrides because
# ``client.api._version`` wasn't set; cluster init now populates
# the minimum APIClient state managers consult.
# ──────────────────────────────────────────────────────────────────────────────


def test_high_level_containers_create_routes_through_overrides() -> None:
    """``client.containers.create(image, command)`` now reaches
    our api.create_container override (no more AttributeError on
    ``api._version``). Returns a docker-py Container wrapper
    whose ``id`` matches what acquire_container assigned."""
    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]

    container = client.containers.create(
        "busybox:latest", command=["sleep", "infinity"],
    )
    assert container.id == "c-1"
    assert len(fake.acquire_calls) == 1
    assert fake.acquire_calls[0]["image"] == "busybox:latest"


def test_high_level_containers_get_returns_container_object() -> None:
    """``client.containers.get(container_id)`` traverses
    api.inspect_container and wraps as a Container."""
    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]
    client.containers.create("busybox:1")  # registers session

    container = client.containers.get("c-1")
    assert container.id == "c-1"
    assert container.attrs["Config"]["Image"] == "busybox:1"


def test_high_level_container_exec_run_routes_through_api() -> None:
    """``container.exec_run(cmd)`` calls api.exec_create +
    exec_start (batched). Returns ``(exit_code, output)`` per
    docker-py contract."""
    fake = _FakeClient()
    fake.transport.next_exec_result = {  # type: ignore[attr-defined]
        "exit_code": 0, "stdout": b"hi from exec\n",
        "stderr": b"", "timed_out": False,
    }
    client = from_env(client=fake)  # type: ignore[arg-type]
    container = client.containers.create("busybox:1")

    exit_code, output = container.exec_run(["echo", "hi"])
    assert exit_code == 0
    assert b"hi from exec" in output


def test_high_level_container_put_archive_round_trip() -> None:
    """``container.put_archive`` and ``container.get_archive``
    both work via the high-level wrapper."""
    fake = _FakeClient()
    fake.transport.next_get_archive = b"<received tar>"  # type: ignore[attr-defined]
    client = from_env(client=fake)  # type: ignore[arg-type]
    container = client.containers.create("busybox:1")

    ok = container.put_archive("/tmp", b"<sent tar>")
    assert ok is True
    chunks_iter, _stat = container.get_archive("/logs")
    assert b"".join(chunks_iter) == b"<received tar>"


def test_high_level_container_remove_destroys_session() -> None:
    """``container.remove(force=True)`` flows through
    api.remove_container → SDK session.destroy."""
    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]
    container = client.containers.create("busybox:1")

    container.remove(force=True)
    assert len(fake.transport.destroy_calls) == 1


def test_high_level_full_lifecycle_flow() -> None:
    """End-to-end via the high-level manager surface — what an
    unmodified docker-py harness sees:

        client = xrlenv.from_env(client=...)
        container = client.containers.create(image, command)
        container.put_archive("/tmp", tar_bytes)
        ec, out = container.exec_run(["cat", "/tmp/file"])
        container.remove(force=True)
    """
    fake = _FakeClient()
    fake.transport.next_exec_result = {  # type: ignore[attr-defined]
        "exit_code": 0, "stdout": b"file contents\n",
        "stderr": b"", "timed_out": False,
    }
    client = from_env(client=fake)  # type: ignore[arg-type]
    container = client.containers.create(
        "busybox:1", command=["sleep", "infinity"],
    )
    assert container.put_archive("/tmp", b"<tar>") is True
    exit_code, output = container.exec_run(["cat", "/tmp/file"])
    container.remove(force=True)

    assert exit_code == 0
    assert b"file contents" in output
    assert len(fake.acquire_calls) == 1
    assert len(fake.transport.put_archive_calls) == 1  # type: ignore[attr-defined]
    assert len(fake.transport.exec_calls) == 1  # type: ignore[attr-defined]
    assert len(fake.transport.destroy_calls) == 1


def test_cluster_remove_force_overrides_caller_default() -> None:
    """Audit Cluster-Dropin-M2 closure. Cluster ``stop()`` is a
    no-op so the container is still running when ``remove()`` is
    called; we always force regardless of caller's flag.
    swebench's ``container.stop(); container.remove(force=True)``
    flow is unaffected; a generic ``container.stop();
    container.remove()`` (no force) Just Works under cluster mode
    where it would error against a real daemon."""
    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]
    client.api.create_container("busybox:1")

    # Caller passes default force=False (or no force flag) —
    # cluster mode still forces.
    client.api.remove_container("c-1")
    assert len(fake.transport.destroy_calls) == 1
    assert fake.transport.destroy_calls[0]["force"] is True


# ──────────────────────────────────────────────────────────────────────────────
# P0a — resource host_config becomes a scheduling input
# (cluster-resource-isolation-plan)
# ──────────────────────────────────────────────────────────────────────────────


def test_create_container_nano_cpus_becomes_cpu_limit() -> None:
    """A harness CPU cap (``NanoCpus``) reaches acquire as an effective
    cpu_limit rather than being silently dropped."""
    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]
    client.api.create_container(
        "busybox:1",
        command=["sleep", "infinity"],
        host_config={"NanoCpus": 2_000_000_000},  # 2 CPU
    )
    assert fake.acquire_calls[-1]["cpu_limit"] == pytest.approx(2.0)
    assert fake.acquire_calls[-1]["mem_limit_bytes"] is None


def test_create_container_cpu_quota_period_becomes_cpu_limit() -> None:
    """``CpuQuota`` + ``CpuPeriod`` resolve to the same effective limit."""
    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]
    client.api.create_container(
        "busybox:1",
        command=["sleep", "infinity"],
        host_config={"CpuQuota": 150_000, "CpuPeriod": 100_000},  # 1.5 CPU
    )
    assert fake.acquire_calls[-1]["cpu_limit"] == pytest.approx(1.5)


def test_create_container_memory_becomes_mem_limit() -> None:
    """A harness ``Memory`` cap reaches acquire as mem_limit_bytes."""
    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]
    client.api.create_container(
        "busybox:1",
        command=["sleep", "infinity"],
        host_config={"Memory": 8 * 1024 * 1024 * 1024},
    )
    assert fake.acquire_calls[-1]["mem_limit_bytes"] == 8 * 1024 * 1024 * 1024


def test_create_container_no_resources_passes_none() -> None:
    """No resource host_config → acquire gets None (raw-container default
    budget applies control-plane-side)."""
    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]
    client.api.create_container("busybox:1", command=["sleep", "infinity"])
    assert fake.acquire_calls[-1]["cpu_limit"] is None
    assert fake.acquire_calls[-1]["mem_limit_bytes"] is None


def test_create_container_rejects_cpu_shares() -> None:
    """CpuShares is a relative weight, not a hard cap — hard error, never
    a silent drop."""
    from xrlenv.compat.docker_client import ClusterResourceKwargError

    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]
    with pytest.raises(ClusterResourceKwargError) as ei:
        client.api.create_container(
            "busybox:1",
            command=["sleep", "infinity"],
            host_config={"CpuShares": 512},
        )
    msg = str(ei.value)
    assert "CpuShares" in msg
    # Four-part message: requested / reason / action.
    assert "requested:" in msg and "reason:" in msg and "action:" in msg
    assert fake.acquire_calls == []


def test_create_container_rejects_memory_reservation() -> None:
    """MemoryReservation is a soft limit — hard error in cluster mode."""
    from xrlenv.compat.docker_client import ClusterResourceKwargError

    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]
    with pytest.raises(ClusterResourceKwargError):
        client.api.create_container(
            "busybox:1",
            command=["sleep", "infinity"],
            host_config={"MemoryReservation": 1 * 1024 * 1024 * 1024},
        )
    assert fake.acquire_calls == []


def test_create_container_runtime_limits_flow_to_acquire() -> None:
    """P0b — RuntimeLimits host_config keys (pids / shm / tmpfs /
    read-only) are extracted and forwarded as a RuntimeLimits object,
    not dropped and not hard-errored."""
    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]
    client.api.create_container(
        "busybox:1",
        command=["sleep", "infinity"],
        host_config={
            "PidsLimit": 4096,
            "ShmSize": 67108864,
            "Tmpfs": {"/run": "size=64m"},
            "ReadonlyRootfs": True,
        },
    )
    rl = fake.acquire_calls[-1]["runtime_limits"]
    assert rl is not None
    assert rl.pids_limit == 4096
    assert rl.shm_size_bytes == 67108864
    assert rl.tmpfs == {"/run": "size=64m"}
    assert rl.readonly_rootfs is True


def test_create_container_no_runtime_limits_passes_none() -> None:
    """P0b — no RuntimeLimits host_config → runtime_limits is None
    (the node then applies no pids/shm/tmpfs/read-only constraint)."""
    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]
    client.api.create_container("busybox:1", command=["sleep", "infinity"])
    assert fake.acquire_calls[-1]["runtime_limits"] is None


def test_create_container_rejects_cpuset_cpus() -> None:
    """cpuset_cpus is cluster-owned (Level-3) — the harness must not pin
    cores; the node assigns them."""
    from xrlenv.control.kwargs_policy import KwargsPolicyViolation

    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]
    with pytest.raises(KwargsPolicyViolation) as ei:
        client.api.create_container(
            "busybox:1",
            command=["sleep", "infinity"],
            host_config={"CpusetCpus": "0-3"},
        )
    msg = str(ei.value)
    assert "cpuset_cpus" in msg
    assert "level 3" in msg
    assert fake.acquire_calls == []


def test_create_container_rejects_ambiguous_memory_swap() -> None:
    """MemorySwap != Memory is ambiguous (daemon-dependent) — hard error."""
    from xrlenv.compat.docker_client import ClusterResourceKwargError

    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]
    with pytest.raises(ClusterResourceKwargError) as ei:
        client.api.create_container(
            "busybox:1",
            command=["sleep", "infinity"],
            host_config={"Memory": 4_000_000_000, "MemorySwap": -1},
        )
    assert "MemorySwap" in str(ei.value)
    assert fake.acquire_calls == []


def test_create_container_memory_swap_equal_to_memory_ok() -> None:
    """MemorySwap == Memory means swap disabled — unambiguous, accepted;
    Memory still flows through as the effective limit."""
    fake = _FakeClient()
    client = from_env(client=fake)  # type: ignore[arg-type]
    client.api.create_container(
        "busybox:1",
        command=["sleep", "infinity"],
        host_config={"Memory": 4_000_000_000, "MemorySwap": 4_000_000_000},
    )
    assert fake.acquire_calls[-1]["mem_limit_bytes"] == 4_000_000_000


# ──────────────────────────────────────────────────────────────────────────────
# exec_start(detach=True) contained no-op (audit M2) + stream cleanup (audit M9)
# ──────────────────────────────────────────────────────────────────────────────


def test_exec_start_detach_watchdog_kill_is_contained_noop() -> None:
    # swebench's exec_run_with_timeout does container.exec_run("kill -TERM <pid>",
    # detach=True) on a test timeout — a STAGED kill exec. It must be a contained no-op
    # returning the empty detached shape, and docker-py's exec_run then calls exec_inspect
    # for the ExitCode -> must resolve.
    api = from_env(client=_FakeClient()).api  # type: ignore[arg-type]
    api._exec_pending["e-kill"] = {"cmd": ["sh", "-c", "kill -TERM 0"], "container_id": "c"}  # type: ignore[attr-defined]
    out = api.exec_start("e-kill", detach=True)
    assert out == b""
    info = api.exec_inspect("e-kill")
    assert info["ExitCode"] == 0 and info["Running"] is False


def test_exec_start_detach_unknown_id_raises_notfound() -> None:
    # audit M2: an UNKNOWN (never-staged) detached id must NOT be faked as a success.
    api = from_env(client=_FakeClient()).api  # type: ignore[arg-type]
    import docker.errors
    with pytest.raises(docker.errors.NotFound):
        api.exec_start("e-never-staged", detach=True)


def test_exec_start_detach_non_kill_command_is_not_supported() -> None:
    # audit M2: a legitimate detached command that ISN'T the watchdog kill must fail loudly,
    # not be silently discarded + reported as success.
    api = from_env(client=_FakeClient()).api  # type: ignore[arg-type]
    api._exec_pending["e-cp"] = {"cmd": ["cp", "a", "b"], "container_id": "c"}  # type: ignore[attr-defined]
    with pytest.raises(NotImplementedError, match="timeout-watchdog kill"):
        api.exec_start("e-cp", detach=True)


def test_exec_start_detach_command_merely_containing_kill_is_rejected() -> None:
    # audit Low: only the EXACT staged watchdog list ``["sh", "-c", "kill -TERM 0"]`` passes as
    # the no-op. A command that merely CONTAINS a `kill` token (echo kill), a different pid, a
    # whitespace variant, or even the bare ``["kill", "-TERM", "0"]`` list (swebench never
    # stages that — it passes a string, which exec_create wraps as ``sh -c``) must all fail.
    api = from_env(client=_FakeClient()).api  # type: ignore[arg-type]
    for bad in (["sh", "-c", "echo kill"], ["sh", "-c", "echo kill > /tmp/m"],
                ["kill", "-9", "1"], ["killall", "python"],
                ["sh", "-c", "kill -TERM 1"],       # nonzero pid — not our Pid-0 watchdog
                ["sh", "-c", "kill  -TERM  0"],      # whitespace variant — not the exact stage
                ["kill", "-TERM", "0"]):             # pure-list form — upstream never stages it
        api._exec_pending["e-x"] = {"cmd": bad, "container_id": "c"}  # type: ignore[attr-defined]
        with pytest.raises(NotImplementedError, match="timeout-watchdog kill"):
            api.exec_start("e-x", detach=True)


def _bare_iterator(on_close: object) -> object:
    # build a _SyncStreamIterator WITHOUT __init__ (no drain thread) to unit-test __next__'s
    # terminal paths in isolation.
    import queue as _queue
    import threading as _threading

    from xrlenv.compat.docker_client import _SyncStreamIterator
    it = _SyncStreamIterator.__new__(_SyncStreamIterator)  # type: ignore[call-overload]
    it._demux = False
    it._closed = False
    it._close_lock = _threading.Lock()
    it._on_close = on_close
    it._on_terminator = lambda _c: None
    it._queue = _queue.Queue()
    return it


def test_stream_iterator_close_is_idempotent_across_producer_and_consumer() -> None:
    # audit M9: producer (drain finally) AND consumer (__next__) both call _close on the
    # same stream; on_close must fire EXACTLY once (locked check-and-set), or a double
    # discard/stash would race. Two _close() calls -> one on_close.
    fired: list[int] = []
    it = _bare_iterator(lambda: fired.append(1))
    it._close()  # type: ignore[attr-defined]
    it._close()  # type: ignore[attr-defined]  # e.g. the other thread
    assert fired == [1]


def test_stream_iterator_clears_active_marker_on_error() -> None:
    # audit M9: a stream that raises before its terminator must still fire on_close, else
    # _exec_streaming keeps the exec Running forever.
    fired: list[bool] = []
    it = _bare_iterator(lambda: fired.append(True))
    it._queue.put(RuntimeError("stream blew up mid-drain"))  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError):
        next(it)  # type: ignore[arg-type]
    assert fired == [True]


def test_stream_iterator_clears_active_marker_on_sentinel_and_terminator() -> None:
    from xrlenv.compat.docker_client import _STREAM_SENTINEL

    fired: list[str] = []
    it = _bare_iterator(lambda: fired.append("sentinel"))
    it._queue.put(_STREAM_SENTINEL)  # type: ignore[attr-defined]
    with pytest.raises(StopIteration):
        next(it)  # type: ignore[arg-type]

    it2 = _bare_iterator(lambda: fired.append("terminator"))
    it2._queue.put({"done": True, "stdout": b"", "stderr": b"", "exit_code": 0})  # type: ignore[attr-defined]
    with pytest.raises(StopIteration):
        next(it2)  # type: ignore[arg-type]
    assert fired == ["sentinel", "terminator"]
