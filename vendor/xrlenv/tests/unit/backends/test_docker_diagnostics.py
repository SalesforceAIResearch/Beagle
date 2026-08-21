"""Unit tests for the Docker backend's failure-diagnostics helper +
sandbox-host-dir permission setup.

The Docker backend itself is exercised in docker-marked integration tests
that require a live daemon. The diagnostics helper added here is pure
SDK-method orchestration with a few error paths, so it's worth a unit
test independent of any container runtime.
"""

from __future__ import annotations

import asyncio
import os
import stat
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from xrlenv.backends.base import (
    ResourceSpec,
    SandboxHandle,
    TemplateRef,
)
from xrlenv.backends.docker import (
    DockerBackend,
    DockerBackendConfig,
    _collect_failure_diagnostics,
)


def _container(
    *,
    status: str = "exited",
    exit_code: int = 1,
    error: str = "",
    logs: bytes = b"",
) -> MagicMock:
    """Build a Docker SDK ``Container`` mock matching the real attribute shape."""
    c = MagicMock()
    c.attrs = {"State": {"Status": status, "ExitCode": exit_code, "Error": error}}
    c.logs.return_value = logs
    return c


def test_includes_status_exit_code_and_logs() -> None:
    container = _container(
        status="exited", exit_code=1,
        logs=b"ModuleNotFoundError: No module named 'aiohttp'\n",
    )
    summary = _collect_failure_diagnostics(container)
    assert "container.status=exited" in summary
    assert "exit_code=1" in summary
    assert "ModuleNotFoundError" in summary


def test_includes_oci_error_when_present() -> None:
    container = _container(
        error="OCI runtime create failed: container_linux.go:380: starting "
              "container process caused: exec: \"python\": executable file not "
              "found in $PATH",
    )
    summary = _collect_failure_diagnostics(container)
    assert "oci_error=" in summary
    assert "executable file not found" in summary


def test_handles_empty_logs() -> None:
    container = _container(logs=b"")
    summary = _collect_failure_diagnostics(container)
    assert "stdout/stderr=<empty>" in summary


def test_truncates_huge_log_output() -> None:
    container = _container(logs=b"x" * 5000)
    summary = _collect_failure_diagnostics(container)
    # Hard cap at 1500 chars in the stdout/stderr field — keeps the
    # raised TimeoutError message readable in the journal.
    stdout_field = summary.split("stdout/stderr=", 1)[1]
    assert len(stdout_field) <= 1500


def test_tolerates_reload_failure() -> None:
    container = MagicMock()
    container.reload.side_effect = RuntimeError("daemon unreachable")
    container.logs.return_value = b"some output"
    summary = _collect_failure_diagnostics(container)
    assert "reload_failed=RuntimeError" in summary
    assert "some output" in summary


def test_tolerates_logs_failure() -> None:
    container = _container(logs=b"")
    container.logs.side_effect = RuntimeError("can't fetch logs")
    summary = _collect_failure_diagnostics(container)
    # Reload still works, so we get the State fields.
    assert "container.status=exited" in summary
    assert "logs_failed=RuntimeError" in summary


def test_handles_str_logs_output() -> None:
    """Older docker-py versions can return ``str`` instead of ``bytes`` from
    ``logs()`` when ``decode=True`` is the daemon default. The helper
    should round-trip either."""
    container = _container()
    container.logs.return_value = "stub bound /run/xrlenv/stub.sock"
    summary = _collect_failure_diagnostics(container)
    assert "stub bound" in summary


def test_collapses_newlines_for_one_line_summary() -> None:
    container = _container(logs=b"line one\nline two\nline three\n")
    summary = _collect_failure_diagnostics(container)
    assert "\n" not in summary
    assert "line one | line two | line three" in summary


# ──────────────────────────────────────────────────────────────────────────────
# DockerBackend.create() — host-side run-dir permission setup.
#
# Real container creation is excluded from the unit suite (requires Docker).
# The relevant pre-mkdir + chmod logic happens before any docker SDK call,
# so we patch out the SDK and exercise just that slice.
# ──────────────────────────────────────────────────────────────────────────────


def _template() -> TemplateRef:
    return TemplateRef(name="hello-shell", image="xrlenv/hello-shell:0.1")


def _resources() -> ResourceSpec:
    return ResourceSpec(
        cpu_request=0.25, cpu_limit=1.0,
        mem_request_bytes=64_000_000, mem_limit_bytes=128_000_000,
        disk_request_bytes=64_000_000,
    )


def _put_archive_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> tuple[DockerBackend, MagicMock, MagicMock, SandboxHandle]:
    """Build a DockerBackend whose docker-py client is fully mocked.

    Returns ``(backend, fake_client, fake_api, sandbox_handle)``. Each
    ``api.exec_create`` returns a fresh handle id; ``api.exec_inspect``
    returns the exit-code dict the test pre-seeded onto
    ``fake_api.exec_inspect_queue`` (popped FIFO per call).
    """
    cfg = DockerBackendConfig(
        runs_root=tmp_path / "runs",
        xrlenv_pkg_path=tmp_path / "xrlenv",
        stub_transport="tcp",
    )

    fake_client = MagicMock()
    fake_api = MagicMock()
    fake_client.api = fake_api
    fake_container = MagicMock()
    fake_container.id = "cid-1"
    fake_client.containers.get.return_value = fake_container

    # exec_create returns sequential handle ids; exec_inspect pops from a
    # pre-seeded queue so tests can script multiple successive exec
    # calls (wipe → mkdir).
    counter = {"n": 0}
    inspect_queue: list[dict[str, int]] = []

    def _exec_create(*_a: Any, **_kw: Any) -> dict[str, str]:
        counter["n"] += 1
        return {"Id": f"exec-{counter['n']}"}

    def _exec_inspect(_id: str) -> dict[str, int]:
        if not inspect_queue:
            return {"ExitCode": 0}
        return inspect_queue.pop(0)

    fake_api.exec_create.side_effect = _exec_create
    fake_api.exec_start.return_value = b""
    fake_api.exec_inspect.side_effect = _exec_inspect
    # Tests can append to this queue via the ``exec_inspect_queue``
    # attribute (kept distinct from MagicMock auto-attribute access).
    fake_api.exec_inspect_queue = inspect_queue

    monkeypatch.setattr(
        "xrlenv.backends.docker.docker.from_env",
        lambda **_: fake_client,
    )
    backend = DockerBackend(cfg)
    handle = SandboxHandle(
        id="sb-1", backend="docker", backend_ref="cid-1",
        stub_endpoint="tcp://127.0.0.1:0",
    )
    return backend, fake_client, fake_api, handle


def test_put_archive_clean_target_runs_root_wipe_then_mkdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audit H1 follow-up: ``put_archive(clean_target=True)`` MUST run
    ``rm -rf <target>`` and ``mkdir -p <target>`` BOTH as user=root
    via ``docker exec --user root``, in that order, before extracting
    the tarball. Pinning the exec-time user="root" kwarg is what
    makes the wipe robust against an image whose default user can't
    remove root-owned residue.
    """
    backend, _client, fake_api, handle = _put_archive_setup(tmp_path, monkeypatch)

    asyncio.run(backend.put_archive(handle, "/tests", b"\x1f\x8b\x08\x00fake", clean_target=True))

    exec_calls = fake_api.exec_create.call_args_list
    # Two exec_create calls: rm -rf, then mkdir -p. Both as user=root.
    assert len(exec_calls) == 2
    rm_call = exec_calls[0]
    mkdir_call = exec_calls[1]
    assert rm_call.args[1] == ["rm", "-rf", "/tests"]
    assert rm_call.kwargs.get("user") == "root"
    assert mkdir_call.args[1] == ["mkdir", "-p", "/tests"]
    assert mkdir_call.kwargs.get("user") == "root"


def test_put_archive_fails_closed_when_wipe_returns_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audit H1 follow-up: when the wipe ``rm -rf`` exec returns a
    non-zero exit code, ``put_archive`` MUST raise rather than
    silently proceed to extract. Otherwise an agent that left
    immutable residue under the target dir could survive into the
    verifier phase."""
    backend, _client, fake_api, handle = _put_archive_setup(tmp_path, monkeypatch)
    fake_api.exec_inspect_queue.append({"ExitCode": 1})  # wipe fails

    with pytest.raises(OSError, match=r"wipe step failed"):
        asyncio.run(
            backend.put_archive(handle, "/tests", b"\x1f\x8b\x08\x00fake", clean_target=True),
        )


def test_put_archive_without_clean_target_skips_wipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default ``clean_target=False`` (or omitted) skips the wipe and
    only runs the mkdir + put_archive — preserves the existing
    contract for callers that don't need the H1 wipe."""
    backend, _client, fake_api, handle = _put_archive_setup(tmp_path, monkeypatch)

    asyncio.run(backend.put_archive(handle, "/tests", b"\x1f\x8b\x08\x00fake"))

    exec_calls = fake_api.exec_create.call_args_list
    # Just the mkdir; no rm.
    assert len(exec_calls) == 1
    assert exec_calls[0].args[1] == ["mkdir", "-p", "/tests"]


def test_create_makes_uds_run_dir_writable_for_container_uid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-sandbox host run dir must be mode 0o777 so the container's
    ``USER sandbox`` (uid 1000) can ``bind()`` ``stub.sock`` inside the
    bind-mount. Without this, the stub crashes at startup with
    ``[Errno 13] Permission denied`` on Linux nodes — see
    xrlenv/backends/docker.py for the full rationale.
    """
    runs_root = tmp_path / "runs"
    cfg = DockerBackendConfig(
        runs_root=runs_root,
        xrlenv_pkg_path=tmp_path / "xrlenv",
        stub_transport="uds",
    )

    # Skip the real Docker client at construction.
    fake_client = MagicMock()
    fake_container = MagicMock()
    fake_container.id = "fake-container-id"
    fake_client.containers.run.return_value = fake_container
    monkeypatch.setattr(
        "xrlenv.backends.docker.docker.from_env",
        lambda **_: fake_client,
    )
    backend = DockerBackend(cfg)

    # Pretend the stub came up so we don't wait the full 30s timeout.
    async def _fake_ready(**_kw: object) -> str:
        return "/run/xrlenv/stub.sock"

    monkeypatch.setattr(backend, "_await_stub_ready", _fake_ready)

    handle = asyncio.run(
        backend.create(
            template=_template(),
            resources=_resources(),
            network_policy="open",
        ),
    )
    sb_dir = runs_root / handle.id
    assert sb_dir.is_dir()
    mode = stat.S_IMODE(sb_dir.stat().st_mode)
    assert mode == 0o777, (
        f"sandbox host dir {sb_dir} has mode {oct(mode)}; expected 0o777 "
        "so the container's sandbox user can bind() stub.sock"
    )


def test_create_tcp_transport_skips_run_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The chmod path is uds-only — tcp doesn't need a host-side dir at all,
    so we shouldn't create it (avoids littering ``runs_root`` with empty
    dirs that the destroy path would have to clean up)."""
    runs_root = tmp_path / "runs"
    cfg = DockerBackendConfig(
        runs_root=runs_root,
        xrlenv_pkg_path=tmp_path / "xrlenv",
        stub_transport="tcp",
    )

    fake_client = MagicMock()
    fake_container = MagicMock()
    fake_container.id = "fake-container-id"
    fake_client.containers.run.return_value = fake_container
    monkeypatch.setattr(
        "xrlenv.backends.docker.docker.from_env",
        lambda **_: fake_client,
    )
    backend = DockerBackend(cfg)

    async def _fake_ready(**_kw: object) -> str:
        return "127.0.0.1:49100"

    monkeypatch.setattr(backend, "_await_stub_ready", _fake_ready)

    handle = asyncio.run(
        backend.create(
            template=_template(),
            resources=_resources(),
            network_policy="open",
        ),
    )
    sb_dir = runs_root / handle.id
    assert not sb_dir.exists(), (
        f"tcp transport unexpectedly created {sb_dir}; only uds should"
    )


def test_create_chmod_survives_existing_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``mkdir(parents=True, exist_ok=True)`` is a no-op when the dir
    already exists at the wrong mode. The chmod must still apply so a
    re-run after a half-finished previous attempt doesn't inherit
    a broken 0o755."""
    runs_root = tmp_path / "runs"
    cfg = DockerBackendConfig(
        runs_root=runs_root,
        xrlenv_pkg_path=tmp_path / "xrlenv",
        stub_transport="uds",
    )

    fake_client = MagicMock()
    fake_container = MagicMock()
    fake_container.id = "fake-container-id"
    fake_client.containers.run.return_value = fake_container
    monkeypatch.setattr(
        "xrlenv.backends.docker.docker.from_env",
        lambda **_: fake_client,
    )
    backend = DockerBackend(cfg)

    # Pre-create the dir at the wrong mode and have create() reuse it.
    forced_id = "fixed_sb_id_for_test"
    stale = runs_root / forced_id
    stale.mkdir(parents=True)
    stale.chmod(0o700)

    with patch("xrlenv.backends.docker.uuid.uuid4") as fake_uuid:
        fake_uuid.return_value.hex = forced_id

        async def _fake_ready(**_kw: object) -> str:
            return "/run/xrlenv/stub.sock"

        monkeypatch.setattr(backend, "_await_stub_ready", _fake_ready)

        asyncio.run(
            backend.create(
                template=_template(),
                resources=_resources(),
                network_policy="open",
            ),
        )

    mode = stat.S_IMODE(stale.stat().st_mode)
    assert mode == 0o777, f"chmod missed a stale dir; got {oct(mode)}"


# Quietly skip the perm tests on platforms where chmod 0o777 is meaningless
# (Windows). Linux + Darwin both honor POSIX chmod, which covers our
# supported phase-0 hosts.
if os.name != "posix":  # pragma: no cover
    pytest.skip("posix-only chmod assertions", allow_module_level=True)


# ──────────────────────────────────────────────────────────────────────────────
# DockerBackend.create() — xrlenv_plugins/ bind-mount.
#
# When DockerBackendConfig.xrlenv_plugins_path is set, the plug-in tree must
# be bind-mounted at /opt/xrlenv-pkg/xrlenv_plugins read-only. When None,
# the mount must be absent so non-plugin installs don't pay for it.
# ──────────────────────────────────────────────────────────────────────────────


def test_create_mounts_xrlenv_plugins_when_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs_root = tmp_path / "runs"
    plugins_root = tmp_path / "xrlenv_plugins"
    plugins_root.mkdir()
    cfg = DockerBackendConfig(
        runs_root=runs_root,
        xrlenv_pkg_path=tmp_path / "xrlenv",
        xrlenv_plugins_path=plugins_root,
        stub_transport="tcp",
    )

    fake_client = MagicMock()
    fake_container = MagicMock()
    fake_container.id = "fake-container-id"
    fake_client.containers.run.return_value = fake_container
    monkeypatch.setattr(
        "xrlenv.backends.docker.docker.from_env",
        lambda **_: fake_client,
    )
    backend = DockerBackend(cfg)

    async def _fake_ready(**_kw: object) -> str:
        return "127.0.0.1:49100"

    monkeypatch.setattr(backend, "_await_stub_ready", _fake_ready)

    asyncio.run(
        backend.create(
            template=_template(),
            resources=_resources(),
            network_policy="open",
        ),
    )

    assert fake_client.containers.run.called
    volumes = fake_client.containers.run.call_args.kwargs["volumes"]
    assert str(plugins_root) in volumes, (
        f"xrlenv_plugins host path {plugins_root} not mounted; got {volumes}"
    )
    assert volumes[str(plugins_root)] == {
        "bind": "/opt/xrlenv-pkg/xrlenv_plugins",
        "mode": "ro",
    }


def test_create_omits_xrlenv_plugins_mount_when_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs_root = tmp_path / "runs"
    cfg = DockerBackendConfig(
        runs_root=runs_root,
        xrlenv_pkg_path=tmp_path / "xrlenv",
        xrlenv_plugins_path=None,
        stub_transport="tcp",
    )

    fake_client = MagicMock()
    fake_container = MagicMock()
    fake_container.id = "fake-container-id"
    fake_client.containers.run.return_value = fake_container
    monkeypatch.setattr(
        "xrlenv.backends.docker.docker.from_env",
        lambda **_: fake_client,
    )
    backend = DockerBackend(cfg)

    async def _fake_ready(**_kw: object) -> str:
        return "127.0.0.1:49100"

    monkeypatch.setattr(backend, "_await_stub_ready", _fake_ready)

    asyncio.run(
        backend.create(
            template=_template(),
            resources=_resources(),
            network_policy="open",
        ),
    )

    volumes = fake_client.containers.run.call_args.kwargs["volumes"]
    bind_targets = {spec["bind"] for spec in volumes.values()}
    assert "/opt/xrlenv-pkg/xrlenv_plugins" not in bind_targets, (
        f"plug-in mount appeared without xrlenv_plugins_path; volumes={volumes}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# resolve_image_digest — buildx local-only RepoDigests trap
# ──────────────────────────────────────────────────────────────────────────────


def test_resolve_image_digest_returns_real_registry_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Registry-resolved images have a RepoDigests entry whose digest
    is the manifest hash (different from the local content ``Id``).
    The resolver must return that digest so the catalog can pin
    ``image:tag`` → ``image@sha256:...``.
    """
    cfg = DockerBackendConfig(
        runs_root=tmp_path / "runs",
        xrlenv_pkg_path=tmp_path / "xrlenv",
        stub_transport="tcp",
    )
    fake_client = MagicMock()
    fake_image = MagicMock()
    # Real-registry case: Id (config hash) ≠ RepoDigests' digest (manifest hash).
    fake_image.attrs = {
        "Id": "sha256:" + "a" * 64,
        "RepoDigests": [
            "registry.example.com/foo@sha256:" + "b" * 64,
        ],
    }
    fake_client.images.get.return_value = fake_image
    monkeypatch.setattr(
        "xrlenv.backends.docker.docker.from_env",
        lambda **_: fake_client,
    )
    backend = DockerBackend(cfg)

    digest = backend.resolve_image_digest("registry.example.com/foo:1.0")
    assert digest == "sha256:" + "b" * 64


def test_resolve_image_digest_skips_buildx_local_only_repodigests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recent Docker (buildx as default builder) pre-populates
    ``RepoDigests`` for locally-built images with an entry whose digest
    is just the image's content ``Id``. That digest is **not**
    registry-resolvable — pinning to it produces an
    ``image@sha256:<id>`` ref no registry can resolve, and any
    ``ensure_present`` on a different host (e.g. a remote node) falls
    through to ``docker pull`` with ``pull access denied``.

    Symptom that surfaced this: 2026-04-30 multi-VM tb2 acceptance
    smoke. The laptop's locally-built ``terminal-bench-2/dna-insert:0.1``
    had ``RepoDigests = ["...@sha256:b2d642..."]`` matching its ``Id``.
    The catalog pinned the resolved tag to ``image@sha256:b2d642...``
    and the GCP node failed at sandbox-create time with
    ``pull access denied for terminal-bench-2/dna-insert: repository
    does not exist or may require 'docker login'``.

    Fix: detect ``RepoDigests`` entries where digest == Id and skip
    them, falling through to the unpinned (None) outcome so the
    catalog leaves the manifest's tag-form image alone.
    """
    cfg = DockerBackendConfig(
        runs_root=tmp_path / "runs",
        xrlenv_pkg_path=tmp_path / "xrlenv",
        stub_transport="tcp",
    )
    fake_client = MagicMock()
    fake_image = MagicMock()
    # Buildx local-only case: Id == RepoDigests' digest.
    local_digest = "sha256:" + "c" * 64
    fake_image.attrs = {
        "Id": local_digest,
        "RepoDigests": [f"terminal-bench-2/dna-insert@{local_digest}"],
    }
    fake_client.images.get.return_value = fake_image
    monkeypatch.setattr(
        "xrlenv.backends.docker.docker.from_env",
        lambda **_: fake_client,
    )
    backend = DockerBackend(cfg)

    digest = backend.resolve_image_digest("terminal-bench-2/dna-insert:0.1")
    assert digest is None, (
        f"expected None for buildx local-only RepoDigest (digest==Id); got {digest!r}"
    )


def test_resolve_image_digest_prefers_real_over_local_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When both a local-only and a registry-resolved RepoDigests
    entry are present (e.g. an image was built locally AND pulled
    from a registry under the same tag at different times), the
    resolver must pick the registry one — that's the cross-host
    pin we actually want.
    """
    cfg = DockerBackendConfig(
        runs_root=tmp_path / "runs",
        xrlenv_pkg_path=tmp_path / "xrlenv",
        stub_transport="tcp",
    )
    fake_client = MagicMock()
    fake_image = MagicMock()
    local_id = "sha256:" + "d" * 64
    real_digest = "sha256:" + "e" * 64
    fake_image.attrs = {
        "Id": local_id,
        # Order matters: skip the local-only one even when it appears first.
        "RepoDigests": [
            f"foo@{local_id}",
            f"foo@{real_digest}",
        ],
    }
    fake_client.images.get.return_value = fake_image
    monkeypatch.setattr(
        "xrlenv.backends.docker.docker.from_env",
        lambda **_: fake_client,
    )
    backend = DockerBackend(cfg)

    digest = backend.resolve_image_digest("foo:0.1")
    assert digest == real_digest


# ──────────────────────────────────────────────────────────────────────────────
# PR #16 regression — docker-py HTTP timeout is set to 600 s
# ──────────────────────────────────────────────────────────────────────────────


def test_docker_backend_pins_from_env_timeout_to_600s(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PR #16 regression: when DockerBackend builds its own docker-py
    client (i.e. no explicit ``client=`` was passed), it must call
    ``docker.from_env`` with ``timeout=DOCKER_CLIENT_HTTP_TIMEOUT_S``
    (600 s) so cold pulls don't get aborted by docker-py's 60 s
    default while the daemon is still streaming layers.

    Captures the actual kwargs passed to ``from_env`` rather than
    asserting the resulting client's ``.timeout`` attribute — the
    latter goes through docker-py's APIClient construction, which
    we don't want to depend on the shape of.
    """
    from xrlenv.backends.docker import DOCKER_CLIENT_HTTP_TIMEOUT_S

    captured: dict[str, Any] = {}
    fake_client = MagicMock(name="docker-client")

    def _fake_from_env(**kwargs: Any) -> MagicMock:
        captured.update(kwargs)
        return fake_client

    monkeypatch.setattr(
        "xrlenv.backends.docker.docker.from_env", _fake_from_env,
    )
    cfg = DockerBackendConfig(
        runs_root=tmp_path / "runs",
        xrlenv_pkg_path=Path("/dev/null"),
        xrlenv_plugins_path=Path("/dev/null"),
    )
    DockerBackend(cfg)

    assert "timeout" in captured, (
        "DockerBackend must pass an explicit ``timeout=`` to docker.from_env "
        "so the docker-py HTTP layer doesn't fall back to the 60 s default "
        "and abort cold pulls of large images mid-stream (PR #16)."
    )
    assert captured["timeout"] == DOCKER_CLIENT_HTTP_TIMEOUT_S


def test_docker_backend_preserves_caller_supplied_client_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The timeout pinning must only apply to the
    ``DockerBackend``-constructed client. A caller that injects its
    own (e.g. an integration test pointing at a remote daemon, a
    custom auth path) keeps full control over the timeout — we
    don't second-guess and don't call ``from_env`` at all.
    """
    sentinel_client = MagicMock(name="caller-supplied")

    def _exploding_from_env(**_: Any) -> MagicMock:
        raise AssertionError(
            "DockerBackend must NOT call docker.from_env when the "
            "caller supplied its own client",
        )

    monkeypatch.setattr(
        "xrlenv.backends.docker.docker.from_env", _exploding_from_env,
    )
    cfg = DockerBackendConfig(
        runs_root=tmp_path / "runs",
        xrlenv_pkg_path=Path("/dev/null"),
        xrlenv_plugins_path=Path("/dev/null"),
    )
    backend = DockerBackend(cfg, client=sentinel_client)
    assert backend.docker_client is sentinel_client


def test_docker_client_http_timeout_matches_image_cache_pull_timeout() -> None:
    """Cross-file invariant: the docker-py HTTP socket timeout and
    the image cache's default pull deadline have to march together,
    otherwise a cold pull is bounded by the wrong layer's ceiling
    and the failure surfaces with a confusing error class
    (urllib3 ReadTimeout vs the cache's TimeoutError). The
    docstring on each constant references the other; this test
    pins the relationship in code so a future refactor that bumps
    one without the other fails loudly here.
    """
    from xrlenv.backends.docker import DOCKER_CLIENT_HTTP_TIMEOUT_S
    from xrlenv.node.image_cache import ImageCacheConfig

    assert (
        ImageCacheConfig().default_pull_timeout_s
        <= DOCKER_CLIENT_HTTP_TIMEOUT_S
    ), (
        "DOCKER_CLIENT_HTTP_TIMEOUT_S must be >= "
        "ImageCacheConfig.default_pull_timeout_s — if the docker-py "
        "HTTP layer's timeout is tighter than the cache's pull "
        "deadline, cold pulls abort at the urllib3 layer before "
        "the cache layer even gets a chance to surface a clean "
        "TimeoutError. See PR #16."
    )


async def test_gc_containerd_content_debounced_and_single_flight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The containerd content prune is debounced + single-flight: a burst
    of concurrently finishing pulls coalesces to one ``ctr content prune``
    (not a per-pull lock storm), and further prunes are skipped until the
    debounce interval elapses. Regression for the daemon-saturation
    incident where per-pull pruning at high pull concurrency starved
    ``docker ps`` / ``docker system df`` on the containerd metadata lock.
    """
    import time as _time

    import xrlenv.backends.docker as dmod
    from xrlenv.backends.docker import _CONTENT_GC_MIN_INTERVAL_S

    cfg = DockerBackendConfig(
        runs_root=tmp_path / "runs",
        xrlenv_pkg_path=tmp_path / "xrlenv",
    )
    backend = DockerBackend(cfg, client=MagicMock())

    calls = {"n": 0}

    def _fake_run(*_a: Any, **_k: Any) -> MagicMock:
        calls["n"] += 1
        return MagicMock(returncode=0)

    monkeypatch.setattr(dmod.subprocess, "run", _fake_run)

    # A burst of 16 concurrently-finishing pulls coalesces to one prune.
    await asyncio.gather(
        *(backend._gc_containerd_content() for _ in range(16))
    )
    assert calls["n"] == 1

    # Within the debounce interval, further prunes are skipped.
    await backend._gc_containerd_content()
    assert calls["n"] == 1

    # After the interval elapses, a prune runs again.
    backend._last_content_gc_monotonic = (
        _time.monotonic() - _CONTENT_GC_MIN_INTERVAL_S - 1.0
    )
    await backend._gc_containerd_content()
    assert calls["n"] == 2


async def test_disk_usage_caches_docker_root_dir_no_root_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """free_disk_bytes resolves DockerRootDir once and reuses it, so a
    later 'docker info' hiccup under load can't silently fall back to '/'
    (the small root fs) and make the evictor thrash. Regression for the
    /images 'free disk 18 GiB' vs df '118 GiB' mismatch on a node whose
    data-root is a separate EBS volume mounted at /opt/sagemaker.
    """
    from types import SimpleNamespace

    import xrlenv.backends.docker as dmod

    cfg = DockerBackendConfig(
        runs_root=tmp_path / "runs",
        xrlenv_pkg_path=tmp_path / "xrlenv",
    )
    client = MagicMock()
    info_calls = {"n": 0}

    def _info() -> dict[str, str]:
        info_calls["n"] += 1
        if info_calls["n"] == 1:
            return {"DockerRootDir": "/opt/sagemaker/docker/data-root"}
        raise RuntimeError("docker info hiccup under load")

    client.info.side_effect = _info
    backend = DockerBackend(cfg, client=client)

    seen_paths: list[str] = []

    def _fake_disk_usage(path: str) -> Any:
        seen_paths.append(path)
        return SimpleNamespace(
            total=500 * 1024**3, used=382 * 1024**3, free=118 * 1024**3,
        )

    monkeypatch.setattr(dmod.shutil, "disk_usage", _fake_disk_usage)

    first = await backend.free_disk_bytes()
    second = await backend.free_disk_bytes()  # info() raises on this 2nd call

    # Resolved once + cached: the 2nd call neither re-queried info() nor
    # fell back to '/'.
    assert info_calls["n"] == 1
    assert seen_paths == [
        "/opt/sagemaker/docker/data-root",
        "/opt/sagemaker/docker/data-root",
    ]
    assert first == second == 118 * 1024**3


async def test_list_images_skips_system_df_unless_requested(
    tmp_path: Path,
) -> None:
    """``list_images`` only calls the expensive ``docker system df``
    (SharedSize) when ``include_shared_size=True`` — the hot path
    (eviction, adaptive stats, /images) leaves it off so it's a cheap
    ``images.list``."""
    client = MagicMock()
    client.images.list.return_value = []
    df_calls = {"n": 0}

    def _df() -> dict[str, Any]:
        df_calls["n"] += 1
        return {"Images": []}

    client.df.side_effect = _df
    cfg = DockerBackendConfig(
        runs_root=tmp_path / "runs",
        xrlenv_pkg_path=tmp_path / "xrlenv",
    )
    backend = DockerBackend(cfg, client=client)

    await backend.list_images()                       # default: cheap
    assert df_calls["n"] == 0
    await backend.list_images(include_shared_size=True)  # calibrate path
    assert df_calls["n"] == 1


# ──────────────────────────────────────────────────────────────────────────────
# remove_image: 409 Conflict (image held by a container) → typed ImageInUse
# ──────────────────────────────────────────────────────────────────────────────


async def test_remove_image_409_conflict_raises_image_in_use() -> None:
    import requests
    from docker.errors import APIError
    from xrlenv.backends.base import ImageInUse

    resp = requests.Response()
    resp.status_code = 409
    client = MagicMock()
    client.images.remove.side_effect = APIError("conflict", response=resp)
    backend = DockerBackend.__new__(DockerBackend)
    backend._client = client  # type: ignore[attr-defined]

    # The 409 ("container is using its referenced image") becomes a typed,
    # expected signal so eviction skips quietly with force=False intact.
    with pytest.raises(ImageInUse):
        await backend.remove_image("x:1")


async def test_remove_image_non_409_apierror_propagates() -> None:
    import requests
    from docker.errors import APIError

    resp = requests.Response()
    resp.status_code = 500
    client = MagicMock()
    client.images.remove.side_effect = APIError("boom", response=resp)
    backend = DockerBackend.__new__(DockerBackend)
    backend._client = client  # type: ignore[attr-defined]

    # A real failure must NOT be swallowed as ImageInUse.
    with pytest.raises(APIError):
        await backend.remove_image("x:1")
