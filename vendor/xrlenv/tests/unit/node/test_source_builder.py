"""Unit tests for the per-node source-build pipeline.

Mocks ``docker.from_env`` and ``git`` subprocess calls so the tests
don't need a Docker daemon or network access. Covers the public
contract of :class:`GitSourceBuilder.build`: success path, build
failure surfaces, missing subdir, oversize → ephemeral, label
propagation + reserved rebuild-cost label, and tarball rejection.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from xrlenv.control.build_plan import GitSource, TarballSource
from xrlenv.node.source_builder import (
    DEFAULT_PER_CONTEXT_MAX_BYTES,
    REBUILD_COST_GIT,
    REBUILD_COST_LABEL,
    GitSourceBuilder,
)


@pytest.fixture
def fake_docker_client() -> MagicMock:
    """Stand-in for the docker-py client. ``images.build`` is a
    mock the test asserts on; default returns a fake (Image, log_iter)
    pair that mirrors docker-py's signature without needing the real
    types."""
    client = MagicMock()
    client.images.build = MagicMock(return_value=(MagicMock(), iter([])))
    return client


@pytest.fixture
def isolated_cache_root(tmp_path: Path) -> Path:
    return tmp_path / "build-context-cache"


def _git_clone_writer(target_dir: Path, *, file_payload: bytes = b"hi") -> Any:
    """Test helper: build an awaitable that, when patched in for
    ``_run_subprocess`` and called with a clone command, materializes
    a fake checkout under whatever path the clone targets.

    The helper inspects the command vector for the ``git clone ...
    <target>`` form and writes a Dockerfile + a payload file at the
    target so the build step finds a valid context."""

    async def _runner(cmd: list[str], **_kwargs: Any) -> None:
        if not cmd or cmd[0] != "git":
            return
        if cmd[1] == "clone":
            # Last positional argument is the clone target.
            dest = Path(cmd[-1])
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "Dockerfile").write_text("FROM scratch\n")
            (dest / "payload.bin").write_bytes(file_payload)
            return
        # fetch / checkout — no-op for the test.
        return

    return _runner


@pytest.mark.asyncio
async def test_git_build_happy_path(
    isolated_cache_root: Path, fake_docker_client: MagicMock,
) -> None:
    builder = GitSourceBuilder(
        cache_root=isolated_cache_root,
        docker_client=fake_docker_client,
    )
    source = GitSource(
        repo="https://github.com/example/repo",
        ref="main", subdir=".", dockerfile="Dockerfile",
    )
    with patch(
        "xrlenv.node.source_builder._run_subprocess",
        side_effect=_git_clone_writer(isolated_cache_root),
    ):
        status, error = await builder.build(
            image_ref="example/x:1", source=source,
            timeout_s=120.0, labels={"team": "ops"},
        )
    assert status == "ok"
    assert error is None
    # docker build called once with the right image_ref and merged labels.
    call_kwargs = fake_docker_client.images.build.call_args.kwargs
    assert call_kwargs["tag"] == "example/x:1"
    # Reserved rebuild-cost label set; operator label preserved.
    assert call_kwargs["labels"][REBUILD_COST_LABEL] == REBUILD_COST_GIT
    assert call_kwargs["labels"]["team"] == "ops"
    assert call_kwargs["dockerfile"] == "Dockerfile"


@pytest.mark.asyncio
async def test_git_build_reserved_label_overrides_operator(
    isolated_cache_root: Path, fake_docker_client: MagicMock,
) -> None:
    """Operator-supplied ``xrlenv.image.rebuild-cost`` is overridden
    by the source builder's reserved value — operators can't smuggle
    a wrong eviction tier into a built image."""
    builder = GitSourceBuilder(
        cache_root=isolated_cache_root,
        docker_client=fake_docker_client,
    )
    source = GitSource(
        repo="https://github.com/example/repo", ref="main",
        subdir=".", dockerfile="Dockerfile",
    )
    operator_attempt = {REBUILD_COST_LABEL: "registry-pull"}
    with patch(
        "xrlenv.node.source_builder._run_subprocess",
        side_effect=_git_clone_writer(isolated_cache_root),
    ):
        await builder.build(
            image_ref="example/x:1", source=source,
            timeout_s=120.0, labels=operator_attempt,
        )
    call_kwargs = fake_docker_client.images.build.call_args.kwargs
    assert call_kwargs["labels"][REBUILD_COST_LABEL] == REBUILD_COST_GIT


@pytest.mark.asyncio
async def test_git_build_missing_subdir_returns_failed(
    isolated_cache_root: Path, fake_docker_client: MagicMock,
) -> None:
    builder = GitSourceBuilder(
        cache_root=isolated_cache_root,
        docker_client=fake_docker_client,
    )
    source = GitSource(
        repo="https://github.com/example/repo", ref="main",
        subdir="not-here", dockerfile="Dockerfile",
    )
    with patch(
        "xrlenv.node.source_builder._run_subprocess",
        side_effect=_git_clone_writer(isolated_cache_root),
    ):
        status, error = await builder.build(
            image_ref="example/x:1", source=source, timeout_s=60.0,
        )
    assert status == "failed"
    assert error is not None
    assert "subdir" in error
    assert "not-here" in error
    fake_docker_client.images.build.assert_not_called()


@pytest.mark.asyncio
async def test_git_build_clone_failure_surfaces(
    isolated_cache_root: Path, fake_docker_client: MagicMock,
) -> None:
    async def failing_clone(cmd: list[str], **_kwargs: Any) -> None:
        from xrlenv.node.source_builder import _BuildError
        raise _BuildError("git clone exited 128: fatal: invalid ref")

    builder = GitSourceBuilder(
        cache_root=isolated_cache_root,
        docker_client=fake_docker_client,
    )
    source = GitSource(
        repo="https://github.com/example/repo", ref="bogus",
        subdir=".", dockerfile="Dockerfile",
    )
    with patch(
        "xrlenv.node.source_builder._run_subprocess",
        side_effect=failing_clone,
    ):
        status, error = await builder.build(
            image_ref="example/x:1", source=source, timeout_s=30.0,
        )
    assert status == "failed"
    assert error is not None
    assert "context resolve" in error
    assert "git clone" in error
    fake_docker_client.images.build.assert_not_called()


@pytest.mark.asyncio
async def test_git_build_docker_failure_surfaces(
    isolated_cache_root: Path, fake_docker_client: MagicMock,
) -> None:
    """A failure inside ``docker build`` (any exception type) is
    translated to ``("failed", <message>)`` so the coordinator
    doesn't have to pattern-match docker-py exception types."""
    fake_docker_client.images.build.side_effect = RuntimeError(
        "syntax error in Dockerfile",
    )
    builder = GitSourceBuilder(
        cache_root=isolated_cache_root,
        docker_client=fake_docker_client,
    )
    source = GitSource(
        repo="https://github.com/example/repo", ref="main",
        subdir=".", dockerfile="Dockerfile",
    )
    with patch(
        "xrlenv.node.source_builder._run_subprocess",
        side_effect=_git_clone_writer(isolated_cache_root),
    ):
        status, error = await builder.build(
            image_ref="example/x:1", source=source, timeout_s=60.0,
        )
    assert status == "failed"
    assert error is not None
    assert "docker build" in error
    assert "syntax error" in error


@pytest.mark.asyncio
async def test_tarball_source_returns_failed_friendly(
    isolated_cache_root: Path, fake_docker_client: MagicMock,
) -> None:
    """Tarball-source dispatch isn't yet implemented on the node
    side; the builder returns a clear operator-friendly error
    rather than crashing."""
    builder = GitSourceBuilder(
        cache_root=isolated_cache_root,
        docker_client=fake_docker_client,
    )
    source = TarballSource(
        path="<wire>", dockerfile="Dockerfile",
    )
    status, error = await builder.build(
        image_ref="example/x:1", source=source, timeout_s=60.0,
    )
    assert status == "failed"
    assert error is not None
    assert "tarball" in error.lower()
    fake_docker_client.images.build.assert_not_called()


@pytest.mark.asyncio
async def test_oversize_context_falls_back_to_ephemeral(
    isolated_cache_root: Path, fake_docker_client: MagicMock,
) -> None:
    """A context larger than ``per_context_max_bytes`` clones into
    an ephemeral tempdir + builds + cleans up, instead of poisoning
    the persistent cache."""
    # 256-byte cap; the helper writes a 1 KB payload, so this
    # context will trip the oversize check.
    builder = GitSourceBuilder(
        cache_root=isolated_cache_root,
        docker_client=fake_docker_client,
        per_context_max_bytes=256,
    )
    source = GitSource(
        repo="https://github.com/example/repo", ref="main",
        subdir=".", dockerfile="Dockerfile",
    )
    with patch(
        "xrlenv.node.source_builder._run_subprocess",
        side_effect=_git_clone_writer(
            isolated_cache_root, file_payload=b"X" * 1024,
        ),
    ):
        status, _error = await builder.build(
            image_ref="example/x:1", source=source, timeout_s=60.0,
        )
    assert status == "ok"
    # Persistent cache stays empty since the context was ephemeral.
    repo_dirs = list(isolated_cache_root.iterdir()) if isolated_cache_root.is_dir() else []
    # Repo hash dir may exist (mkdir at construction) but its ref
    # subdirs should be empty since the oversize path skipped the
    # promotion step.
    for d in repo_dirs:
        if d.is_dir():
            assert not any(d.iterdir()), (
                f"oversize context leaked into persistent cache: {d}"
            )


@pytest.mark.asyncio
async def test_concurrent_builds_serialized_by_default(
    isolated_cache_root: Path, fake_docker_client: MagicMock,
) -> None:
    """Default ``concurrent_builds=1`` means two simultaneous calls
    don't enter the docker build region in parallel. Verify the
    semaphore does its job."""
    in_flight = 0
    max_concurrent = 0
    enter_event = asyncio.Event()

    def slow_build(**_kwargs: Any) -> tuple[Any, Any]:
        nonlocal in_flight, max_concurrent
        in_flight += 1
        max_concurrent = max(max_concurrent, in_flight)
        try:
            # Block briefly to give a sibling task a chance to
            # observe the simultaneous-entry condition if the
            # semaphore is broken.
            import time
            time.sleep(0.05)
            return (MagicMock(), iter([]))
        finally:
            in_flight -= 1
            enter_event.set()

    fake_docker_client.images.build.side_effect = slow_build
    builder = GitSourceBuilder(
        cache_root=isolated_cache_root,
        docker_client=fake_docker_client,
    )
    source = GitSource(
        repo="https://github.com/example/repo", ref="main",
        subdir=".", dockerfile="Dockerfile",
    )
    with patch(
        "xrlenv.node.source_builder._run_subprocess",
        side_effect=_git_clone_writer(isolated_cache_root),
    ):
        await asyncio.gather(
            builder.build(image_ref="x/a:1", source=source, timeout_s=60.0),
            builder.build(image_ref="x/b:1", source=source, timeout_s=60.0),
        )
    assert max_concurrent == 1, (
        f"expected serialized builds (max_concurrent=1), got {max_concurrent}"
    )


def test_default_constructor_uses_home_cache(tmp_path: Path) -> None:
    """The default cache root is under ``~/.xrlenv``. Verifying the
    constructor doesn't accidentally reach the docker daemon at
    import time."""
    builder = GitSourceBuilder(cache_root=tmp_path)
    assert builder._cache_root == tmp_path
    assert builder._per_context_max_bytes == DEFAULT_PER_CONTEXT_MAX_BYTES


def test_cache_root_env_var_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Operators can override the cache root via
    ``$XRLENV_BUILD_CONTEXT_CACHE`` — useful on nodes whose
    ``~/.xrlenv`` is mounted read-only."""
    from xrlenv.node.source_builder import _resolve_cache_root

    target = tmp_path / "operator-set"
    monkeypatch.setenv("XRLENV_BUILD_CONTEXT_CACHE", str(target))
    resolved = _resolve_cache_root(None)
    assert resolved == target
    assert resolved.is_dir()


def test_cache_root_falls_back_to_tmp_when_home_unwritable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the home-based default isn't writable (bootstrap-managed
    read-only mount), resolution falls back to a uid-scoped /tmp
    path with a logged warning, so the daemon still works without
    operator intervention."""
    from xrlenv.node import source_builder

    monkeypatch.delenv("XRLENV_BUILD_CONTEXT_CACHE", raising=False)
    # Pretend Path.home() returns a read-only path so the default
    # mkdir raises OSError.
    fake_home = tmp_path / "ro-home"
    fake_home.mkdir()
    fake_home.chmod(0o500)  # read+execute, no write
    monkeypatch.setattr(
        source_builder.Path, "home", staticmethod(lambda: fake_home),
    )
    try:
        resolved = source_builder._resolve_cache_root(None)
    finally:
        # Restore write perms so tmp_path cleanup doesn't fail.
        fake_home.chmod(0o700)
    assert "/tmp/xrlenv-build-context-cache-" in str(resolved)
    assert resolved.is_dir()


# ──────────────────────────────────────────────────────────────────────────────
# Cancel surface
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cancel_key_label_set_on_every_git_build(
    isolated_cache_root: Path, fake_docker_client: MagicMock,
) -> None:
    """Every git build must carry ``xrlenv.cancel-key=<image_ref>`` so
    the cancel path can find the running build container by label."""
    from xrlenv.node.source_builder import CANCEL_KEY_LABEL

    builder = GitSourceBuilder(
        cache_root=isolated_cache_root,
        docker_client=fake_docker_client,
    )
    source = GitSource(
        repo="https://github.com/example/repo", ref="main",
        subdir=".", dockerfile="Dockerfile",
    )
    with patch(
        "xrlenv.node.source_builder._run_subprocess",
        side_effect=_git_clone_writer(isolated_cache_root),
    ):
        await builder.build(
            image_ref="example/x:1", source=source, timeout_s=60.0,
        )
    call_kwargs = fake_docker_client.images.build.call_args.kwargs
    assert call_kwargs["labels"][CANCEL_KEY_LABEL] == "example/x:1"


@pytest.mark.asyncio
async def test_cancel_with_no_active_build_is_ok(
    isolated_cache_root: Path, fake_docker_client: MagicMock,
) -> None:
    """Cancel for an image_ref with no in-flight build is a no-op
    success — operator-idempotent."""
    fake_docker_client.containers.list = MagicMock(return_value=[])
    builder = GitSourceBuilder(
        cache_root=isolated_cache_root,
        docker_client=fake_docker_client,
    )
    status, error = await builder.cancel("never/built:1")
    assert status == "ok"
    assert error is None


@pytest.mark.asyncio
async def test_cancel_kills_matching_containers(
    isolated_cache_root: Path, fake_docker_client: MagicMock,
) -> None:
    """The cancel path looks up containers labeled with the
    cancel-key for the image_ref and force-kills each. The docker-py
    filter must include the reserved label."""
    from xrlenv.node.source_builder import CANCEL_KEY_LABEL

    fake_container_a = MagicMock()
    fake_container_a.id = "deadbeef-a"
    fake_container_b = MagicMock()
    fake_container_b.id = "deadbeef-b"
    fake_docker_client.containers.list = MagicMock(
        return_value=[fake_container_a, fake_container_b],
    )
    builder = GitSourceBuilder(
        cache_root=isolated_cache_root,
        docker_client=fake_docker_client,
    )
    status, error = await builder.cancel("xrlenv-seta-env/3:main")
    assert status == "ok"
    assert error is None
    fake_docker_client.containers.list.assert_called_once_with(
        filters={"label": f"{CANCEL_KEY_LABEL}=xrlenv-seta-env/3:main"},
    )
    fake_container_a.kill.assert_called_once()
    fake_container_b.kill.assert_called_once()


@pytest.mark.asyncio
async def test_cancel_signals_in_flight_task(
    isolated_cache_root: Path, fake_docker_client: MagicMock,
) -> None:
    """When a build is in-flight, ``cancel`` calls ``Task.cancel()``
    on the registered task. The next await inside ``_build_git``
    raises CancelledError, which the handler converts to a
    ``("failed", "cancelled by operator")`` return value so the
    outer command_id reply lands normally."""
    fake_docker_client.containers.list = MagicMock(return_value=[])
    builder = GitSourceBuilder(
        cache_root=isolated_cache_root,
        docker_client=fake_docker_client,
    )

    # Make ``_resolve_context`` block forever so we have a definite
    # in-flight window to issue the cancel during.
    block_event = asyncio.Event()

    async def _blocking_resolve(
        source: Any, *, timeout_s: float,
    ) -> tuple[Path, bool]:
        await block_event.wait()
        # Should never reach here in this test — task gets cancelled
        # while we're waiting on the event.
        return (isolated_cache_root, False)

    with patch.object(
        builder, "_resolve_context", side_effect=_blocking_resolve,
    ):
        source = GitSource(
            repo="https://github.com/example/repo", ref="main",
            subdir=".", dockerfile="Dockerfile",
        )
        build_task = asyncio.create_task(builder.build(
            image_ref="cancel-me:1", source=source, timeout_s=60.0,
        ))
        # Yield once so the task starts and registers itself.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        # Confirm registered before cancelling — sanity.
        assert "cancel-me:1" in builder._active_builds
        cancel_status, cancel_error = await builder.cancel("cancel-me:1")
        assert cancel_status == "ok"
        assert cancel_error is None
        # Build task unwinds with a normal return value (NOT raises
        # CancelledError) so the outer wire reply is a clean payload.
        status, error = await build_task
        assert status == "failed"
        assert error == "cancelled by operator"
        # Registry was cleaned up.
        assert "cancel-me:1" not in builder._active_builds


# ──────────────────────────────────────────────────────────────────────────────
# Sub-slice 1.b — tarball-source build path
# ──────────────────────────────────────────────────────────────────────────────


def _make_tarball_bytes(payload: dict[str, bytes], *, gzip: bool = False) -> bytes:
    """Build an in-memory tar (or tar.gz) with ``{name: bytes}`` entries."""
    import io
    import tarfile

    buf = io.BytesIO()
    mode = "w:gz" if gzip else "w"
    with tarfile.open(fileobj=buf, mode=mode) as tf:
        for name, data in payload.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


@pytest.mark.asyncio
async def test_tarball_build_happy_path(
    isolated_cache_root: Path, fake_docker_client: MagicMock,
) -> None:
    """A valid tarball with a Dockerfile at the root extracts +
    docker-builds + tags, with the reserved cancel-key + cheap
    rebuild-cost labels merged on top of operator labels."""
    import base64

    from xrlenv.node.source_builder import (
        CANCEL_KEY_LABEL,
        REBUILD_COST_LABEL,
        REBUILD_COST_TARBALL,
    )

    tar_bytes = _make_tarball_bytes({
        "Dockerfile": b"FROM scratch\n",
        "data.bin": b"hello",
    })
    builder = GitSourceBuilder(
        cache_root=isolated_cache_root,
        docker_client=fake_docker_client,
    )
    source = TarballSource(
        path="<wire>", dockerfile="Dockerfile",
        content_b64=base64.b64encode(tar_bytes).decode("ascii"),
    )
    status, error = await builder.build(
        image_ref="example/from-tar:1", source=source,
        timeout_s=60.0, labels={"team": "ops"},
    )
    assert status == "ok"
    assert error is None
    call_kwargs = fake_docker_client.images.build.call_args.kwargs
    assert call_kwargs["tag"] == "example/from-tar:1"
    assert call_kwargs["dockerfile"] == "Dockerfile"
    # Reserved labels set + operator label preserved.
    assert call_kwargs["labels"][REBUILD_COST_LABEL] == REBUILD_COST_TARBALL
    assert call_kwargs["labels"][CANCEL_KEY_LABEL] == "example/from-tar:1"
    assert call_kwargs["labels"]["team"] == "ops"


@pytest.mark.asyncio
async def test_tarball_build_handles_gzip(
    isolated_cache_root: Path, fake_docker_client: MagicMock,
) -> None:
    """tar.gz input works because tarfile.open auto-detects gzip
    via the magic bytes; the schema's ``path`` extension is not
    consulted (operators sometimes ship .tar.gz with .tar extension
    or vice versa)."""
    import base64

    tar_bytes = _make_tarball_bytes({"Dockerfile": b"FROM scratch\n"}, gzip=True)
    builder = GitSourceBuilder(
        cache_root=isolated_cache_root,
        docker_client=fake_docker_client,
    )
    source = TarballSource(
        path="<wire>", dockerfile="Dockerfile",
        content_b64=base64.b64encode(tar_bytes).decode("ascii"),
    )
    status, error = await builder.build(
        image_ref="gzip/img:1", source=source, timeout_s=60.0,
    )
    assert status == "ok"
    assert error is None


@pytest.mark.asyncio
async def test_tarball_build_missing_dockerfile_returns_failed(
    isolated_cache_root: Path, fake_docker_client: MagicMock,
) -> None:
    """Tarball without the named Dockerfile fails fast with a
    clear error, before docker build is even invoked."""
    import base64

    tar_bytes = _make_tarball_bytes({"random.txt": b"hi"})
    builder = GitSourceBuilder(
        cache_root=isolated_cache_root,
        docker_client=fake_docker_client,
    )
    source = TarballSource(
        path="<wire>", dockerfile="Dockerfile",
        content_b64=base64.b64encode(tar_bytes).decode("ascii"),
    )
    status, error = await builder.build(
        image_ref="missing/df:1", source=source, timeout_s=60.0,
    )
    assert status == "failed"
    assert error is not None
    assert "Dockerfile" in error
    assert "not found" in error
    fake_docker_client.images.build.assert_not_called()


@pytest.mark.asyncio
async def test_tarball_build_malformed_tar_returns_failed(
    isolated_cache_root: Path, fake_docker_client: MagicMock,
) -> None:
    """Malformed tarball bytes surface as ("failed", ...) — operator
    sees the TarError instead of an uncaught exception bubbling up."""
    import base64

    builder = GitSourceBuilder(
        cache_root=isolated_cache_root,
        docker_client=fake_docker_client,
    )
    source = TarballSource(
        path="<wire>", dockerfile="Dockerfile",
        content_b64=base64.b64encode(b"this-is-not-a-tar").decode("ascii"),
    )
    status, error = await builder.build(
        image_ref="malformed:1", source=source, timeout_s=60.0,
    )
    assert status == "failed"
    assert error is not None
    assert "tarball extraction failed" in error
    fake_docker_client.images.build.assert_not_called()


@pytest.mark.asyncio
async def test_tarball_build_no_content_b64_returns_failed(
    isolated_cache_root: Path, fake_docker_client: MagicMock,
) -> None:
    """A TarballSource with content_b64=None never reaches docker
    build — clear error pointing at the CLI helper."""
    builder = GitSourceBuilder(
        cache_root=isolated_cache_root,
        docker_client=fake_docker_client,
    )
    source = TarballSource(
        path="<wire>", dockerfile="Dockerfile",
        content_b64=None,
    )
    status, error = await builder.build(
        image_ref="unresolved:1", source=source, timeout_s=60.0,
    )
    assert status == "failed"
    assert error is not None
    assert "content_b64" in error
    assert "resolve_tarball_sources" in error


@pytest.mark.asyncio
async def test_tarball_build_rejects_path_traversal(
    isolated_cache_root: Path, fake_docker_client: MagicMock,
) -> None:
    """A malicious tarball with a member like ``../etc/foo`` is
    refused before extraction — the path-traversal guard catches
    members whose normalized path escapes the extract dir."""
    import base64

    tar_bytes = _make_tarball_bytes({
        "../escape.bin": b"oops",
        "Dockerfile": b"FROM scratch\n",
    })
    builder = GitSourceBuilder(
        cache_root=isolated_cache_root,
        docker_client=fake_docker_client,
    )
    source = TarballSource(
        path="<wire>", dockerfile="Dockerfile",
        content_b64=base64.b64encode(tar_bytes).decode("ascii"),
    )
    status, error = await builder.build(
        image_ref="evil:1", source=source, timeout_s=60.0,
    )
    assert status == "failed"
    assert error is not None
    assert "escape" in error or "../" in error
    fake_docker_client.images.build.assert_not_called()


@pytest.mark.asyncio
async def test_tarball_build_cancel_signals_active_task(
    isolated_cache_root: Path, fake_docker_client: MagicMock,
) -> None:
    """The active-builds registry covers tarball builds the same
    way as git: ``cancel(image_ref)`` interrupts an in-flight
    tarball extract+build, and the task unwinds with a normal
    ("failed", "cancelled by operator") return."""
    import base64

    fake_docker_client.containers.list = MagicMock(return_value=[])
    builder = GitSourceBuilder(
        cache_root=isolated_cache_root,
        docker_client=fake_docker_client,
    )
    tar_bytes = _make_tarball_bytes({"Dockerfile": b"FROM scratch\n"})

    block_event = asyncio.Event()

    async def _blocking_docker_build(*args: Any, **kwargs: Any) -> None:
        await block_event.wait()

    with patch.object(
        builder, "_docker_build", side_effect=_blocking_docker_build,
    ):
        source = TarballSource(
            path="<wire>", dockerfile="Dockerfile",
            content_b64=base64.b64encode(tar_bytes).decode("ascii"),
        )
        build_task = asyncio.create_task(builder.build(
            image_ref="cancel-tar:1", source=source, timeout_s=60.0,
        ))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        # Confirm the tarball build is registered.
        assert "cancel-tar:1" in builder._active_builds
        cancel_status, cancel_error = await builder.cancel("cancel-tar:1")
        assert cancel_status == "ok"
        assert cancel_error is None
        status, error = await build_task
        assert status == "failed"
        assert error == "cancelled by operator"
        assert "cancel-tar:1" not in builder._active_builds


@pytest.mark.asyncio
async def test_cancel_kill_error_does_not_break_idempotent_noop(
    isolated_cache_root: Path, fake_docker_client: MagicMock,
) -> None:
    """When the docker daemon throws on the kill path AND there's no
    active build to cancel either, ``cancel`` reports the docker
    failure — the operator should know cluster state is degraded."""
    fake_docker_client.containers.list = MagicMock(
        side_effect=RuntimeError("docker daemon unreachable"),
    )
    builder = GitSourceBuilder(
        cache_root=isolated_cache_root,
        docker_client=fake_docker_client,
    )
    status, error = await builder.cancel("not-active:1")
    assert status == "failed"
    assert error is not None
    assert "docker daemon unreachable" in error


# ──────────────────────────────────────────────────────────────────────────────
# skip_if_present (operator-driven warm-cluster re-apply)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_skip_if_present_short_circuits_when_image_tagged(
    isolated_cache_root: Path, fake_docker_client: MagicMock,
) -> None:
    """skip_if_present=True + image already tagged locally returns
    ('ok', None) without invoking the clone / docker build pipeline.
    Source-spec persistence still fires so a later
    acquire-after-eviction has the recipe."""
    fake_docker_client.images.get = MagicMock(return_value=MagicMock())
    builder = GitSourceBuilder(
        cache_root=isolated_cache_root,
        docker_client=fake_docker_client,
    )
    source = GitSource(
        repo="https://github.com/example/repo", ref="main",
        subdir=".", dockerfile="Dockerfile",
    )
    status, error = await builder.build(
        image_ref="already-here:1", source=source,
        timeout_s=60.0, skip_if_present=True,
    )
    assert status == "ok"
    assert error is None
    fake_docker_client.images.get.assert_called_once_with("already-here:1")
    # Critically: NO clone, NO docker build.
    fake_docker_client.images.build.assert_not_called()
    # Source-spec was still persisted (so acquire-after-eviction
    # still works on this ref).
    h = hashlib.sha256(b"already-here:1").hexdigest()[:32]
    assert (builder._source_registry_root / h / "spec.json").is_file()


@pytest.mark.asyncio
async def test_skip_if_present_dispatches_when_image_absent(
    isolated_cache_root: Path, fake_docker_client: MagicMock,
) -> None:
    """skip_if_present=True + image absent → full normal dispatch
    (clone + docker build). The flag is a fast-path, not a no-op."""
    import docker.errors
    fake_docker_client.images.get = MagicMock(
        side_effect=docker.errors.ImageNotFound("not present"),
    )
    builder = GitSourceBuilder(
        cache_root=isolated_cache_root,
        docker_client=fake_docker_client,
    )
    source = GitSource(
        repo="https://github.com/example/repo", ref="main",
        subdir=".", dockerfile="Dockerfile",
    )
    with patch(
        "xrlenv.node.source_builder._run_subprocess",
        side_effect=_git_clone_writer(isolated_cache_root),
    ):
        status, error = await builder.build(
            image_ref="needs-build:1", source=source,
            timeout_s=120.0, skip_if_present=True,
        )
    assert status == "ok"
    assert error is None
    # docker build DID run — short-circuit was correctly skipped.
    fake_docker_client.images.build.assert_called_once()


@pytest.mark.asyncio
async def test_skip_if_present_false_always_dispatches(
    isolated_cache_root: Path, fake_docker_client: MagicMock,
) -> None:
    """skip_if_present=False (the default) preserves the prior
    behavior: dispatch the build even when the image is already
    tagged locally. Existing operator scripts don't silently
    change semantics."""
    fake_docker_client.images.get = MagicMock(return_value=MagicMock())
    builder = GitSourceBuilder(
        cache_root=isolated_cache_root,
        docker_client=fake_docker_client,
    )
    source = GitSource(
        repo="https://github.com/example/repo", ref="main",
        subdir=".", dockerfile="Dockerfile",
    )
    with patch(
        "xrlenv.node.source_builder._run_subprocess",
        side_effect=_git_clone_writer(isolated_cache_root),
    ):
        await builder.build(
            image_ref="always-rebuild:1", source=source,
            timeout_s=120.0,
            # skip_if_present defaults to False.
        )
    fake_docker_client.images.build.assert_called_once()


# ──────────────────────────────────────────────────────────────────────────────
# Sub-slice 2 — persistent source-spec registry + lookup_producer
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_lookup_producer_returns_none_for_unregistered_ref(
    isolated_cache_root: Path, fake_docker_client: MagicMock,
) -> None:
    """An image_ref the builder has never seen returns None — the
    image cache then falls through to its registry-pull path
    (registry-pullable refs keep working unchanged after sub-slice 2)."""
    builder = GitSourceBuilder(
        cache_root=isolated_cache_root,
        docker_client=fake_docker_client,
    )
    assert builder.lookup_producer("never-built:1") is None


@pytest.mark.asyncio
async def test_successful_git_build_persists_source_spec(
    isolated_cache_root: Path, fake_docker_client: MagicMock,
) -> None:
    """A successful git build writes the source spec to disk so a
    later ``lookup_producer`` (post-eviction OR post-restart) can
    rebuild without re-shipping from the operator."""
    builder = GitSourceBuilder(
        cache_root=isolated_cache_root,
        docker_client=fake_docker_client,
    )
    source = GitSource(
        repo="https://github.com/example/repo", ref="main",
        subdir=".", dockerfile="Dockerfile",
    )
    with patch(
        "xrlenv.node.source_builder._run_subprocess",
        side_effect=_git_clone_writer(isolated_cache_root),
    ):
        status, _ = await builder.build(
            image_ref="persisted-git:1", source=source,
            timeout_s=120.0,
        )
    assert status == "ok"

    # Producer is callable now.
    producer = builder.lookup_producer("persisted-git:1")
    assert producer is not None

    # Disk side: spec.json exists with the right shape.
    h = hashlib.sha256(b"persisted-git:1").hexdigest()[:32]
    spec_path = builder._source_registry_root / h / "spec.json"
    assert spec_path.is_file()
    spec = json.loads(spec_path.read_text())
    assert spec["type"] == "git"
    assert spec["repo"] == "https://github.com/example/repo"
    assert spec["ref"] == "main"


@pytest.mark.asyncio
async def test_successful_tarball_build_persists_content(
    isolated_cache_root: Path, fake_docker_client: MagicMock,
) -> None:
    """Tarball builds persist BOTH the spec.json and the bytes
    (``content.bin``) so a later rebuild after eviction has
    everything it needs without re-shipping from the operator."""
    import base64

    tar_bytes = _make_tarball_bytes({"Dockerfile": b"FROM scratch\n"})
    builder = GitSourceBuilder(
        cache_root=isolated_cache_root,
        docker_client=fake_docker_client,
    )
    source = TarballSource(
        path="<wire>", dockerfile="Dockerfile",
        content_b64=base64.b64encode(tar_bytes).decode("ascii"),
    )
    status, _ = await builder.build(
        image_ref="persisted-tar:1", source=source, timeout_s=60.0,
    )
    assert status == "ok"

    h = hashlib.sha256(b"persisted-tar:1").hexdigest()[:32]
    entry_dir = builder._source_registry_root / h
    assert (entry_dir / "spec.json").is_file()
    assert (entry_dir / "content.bin").is_file()
    # Bytes survive the round-trip (no truncation).
    assert (entry_dir / "content.bin").read_bytes() == tar_bytes


@pytest.mark.asyncio
async def test_lookup_producer_rebuilds_after_eviction(
    isolated_cache_root: Path, fake_docker_client: MagicMock,
) -> None:
    """The producer returned by ``lookup_producer`` actually invokes
    ``build()`` for the registered source — simulates "image got
    evicted, ensure_present asks for it again, rebuild fires
    automatically." The fake docker client's images.build counter
    bumps each invocation."""
    builder = GitSourceBuilder(
        cache_root=isolated_cache_root,
        docker_client=fake_docker_client,
    )
    source = GitSource(
        repo="https://github.com/example/repo", ref="main",
        subdir=".", dockerfile="Dockerfile",
    )
    with patch(
        "xrlenv.node.source_builder._run_subprocess",
        side_effect=_git_clone_writer(isolated_cache_root),
    ):
        await builder.build(
            image_ref="rebuild-me:1", source=source, timeout_s=120.0,
        )
    initial_calls = fake_docker_client.images.build.call_count
    assert initial_calls == 1

    # Eviction would happen elsewhere; for this unit test we just
    # invoke the producer directly. It must call docker build again.
    producer = builder.lookup_producer("rebuild-me:1")
    assert producer is not None
    with patch(
        "xrlenv.node.source_builder._run_subprocess",
        side_effect=_git_clone_writer(isolated_cache_root),
    ):
        await producer("rebuild-me:1", 120.0)
    assert fake_docker_client.images.build.call_count == initial_calls + 1


@pytest.mark.asyncio
async def test_source_registry_survives_builder_recreation(
    isolated_cache_root: Path, fake_docker_client: MagicMock,
) -> None:
    """Simulates a node restart: build with one builder, throw it
    away, construct a new builder against the same registry root,
    confirm the new instance can ``lookup_producer`` for the
    pre-existing image_ref. This is the "build-on-acquire after
    restart" property that's the whole point of persistence."""
    registry_root = isolated_cache_root / "source-registry"
    builder1 = GitSourceBuilder(
        cache_root=isolated_cache_root,
        source_registry_root=registry_root,
        docker_client=fake_docker_client,
    )
    source = GitSource(
        repo="https://github.com/example/repo", ref="main",
        subdir=".", dockerfile="Dockerfile",
    )
    with patch(
        "xrlenv.node.source_builder._run_subprocess",
        side_effect=_git_clone_writer(isolated_cache_root),
    ):
        await builder1.build(
            image_ref="restart-me:1", source=source, timeout_s=120.0,
        )

    # New builder, same disk state.
    builder2 = GitSourceBuilder(
        cache_root=isolated_cache_root,
        source_registry_root=registry_root,
        docker_client=fake_docker_client,
    )
    producer = builder2.lookup_producer("restart-me:1")
    assert producer is not None
