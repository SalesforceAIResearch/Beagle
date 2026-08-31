"""Unit tests for ``DockerBackend.disk_monitor_path()`` (ff354a1).

``disk_monitor_path()`` exposes the resolved docker data-root path for use
by ``DiskIoSampler``. It must:
- Return ``None`` before ``docker info`` has succeeded (so the sampler
  doesn't accidentally bind to the small root-fs ``"/"``).
- Return the resolved path once ``docker info`` succeeds and it's cached.
- Call ``_resolve_docker_root_dir()`` when the path is not yet cached,
  and return ``None`` if the daemon fails (daemon not up yet → retry later).
- Return the cached path immediately on subsequent calls without
  re-querying the daemon.

These tests use a ``MagicMock`` Docker client (same pattern as the
existing ``test_docker_diagnostics.py`` tests) so no live daemon is needed.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from xrlenv.backends.docker import DockerBackend, DockerBackendConfig


def _make_backend(tmp_path: Path, client: MagicMock) -> DockerBackend:
    cfg = DockerBackendConfig(
        runs_root=tmp_path / "runs",
        xrlenv_pkg_path=tmp_path / "xrlenv",
    )
    return DockerBackend(cfg, client=client)  # type: ignore[arg-type]


# ── returns None before docker info succeeds ─────────────────────────────────


def test_disk_monitor_path_none_when_info_fails(tmp_path: Path) -> None:
    """When ``docker info`` raises, ``disk_monitor_path`` returns ``None``
    (daemon not yet up) so the sampler defers binding until a later tick."""
    client = MagicMock()
    client.info.side_effect = RuntimeError("docker not ready")
    backend = _make_backend(tmp_path, client)

    result = backend.disk_monitor_path()

    assert result is None


def test_disk_monitor_path_none_when_info_returns_empty_root(
    tmp_path: Path,
) -> None:
    """When ``docker info`` returns a falsey/missing ``DockerRootDir``,
    ``disk_monitor_path`` returns ``None`` (treats it as unresolved)."""
    client = MagicMock()
    client.info.return_value = {}  # no DockerRootDir key
    backend = _make_backend(tmp_path, client)

    result = backend.disk_monitor_path()

    assert result is None


# ── returns path once docker info succeeds ────────────────────────────────────


def test_disk_monitor_path_returns_resolved_path_on_success(
    tmp_path: Path,
) -> None:
    """When ``docker info`` returns a valid ``DockerRootDir``,
    ``disk_monitor_path`` returns that path."""
    client = MagicMock()
    client.info.return_value = {"DockerRootDir": "/opt/sagemaker/docker"}
    backend = _make_backend(tmp_path, client)

    result = backend.disk_monitor_path()

    assert result == "/opt/sagemaker/docker"


def test_disk_monitor_path_uses_cached_value_without_re_querying_daemon(
    tmp_path: Path,
) -> None:
    """Once the path is resolved, subsequent calls must not re-query
    ``docker info`` — the cache must be reused."""
    client = MagicMock()
    client.info.return_value = {"DockerRootDir": "/opt/sagemaker/docker"}
    backend = _make_backend(tmp_path, client)

    first = backend.disk_monitor_path()
    second = backend.disk_monitor_path()

    assert first == second == "/opt/sagemaker/docker"
    # docker info must have been called at most once (the second call hit cache).
    assert client.info.call_count == 1


def test_disk_monitor_path_reflects_already_cached_root(
    tmp_path: Path,
) -> None:
    """If ``_docker_root_dir`` is already populated (e.g. after a prior
    ``free_disk_bytes`` call resolved it), ``disk_monitor_path`` returns
    it immediately without calling info again."""
    client = MagicMock()
    # Pre-populate the cache as ``free_disk_bytes`` would.
    backend = _make_backend(tmp_path, client)
    backend._docker_root_dir = "/pre/cached/path"

    result = backend.disk_monitor_path()

    assert result == "/pre/cached/path"
    # info() must not have been called since the cache was already warm.
    client.info.assert_not_called()


# ── sampler retry: provider returns None first, path later ────────────────────


def test_disk_monitor_path_retries_after_initial_failure(
    tmp_path: Path,
) -> None:
    """First call: daemon not up → None. Second call: daemon up → path.
    This mirrors the DiskIoSampler's lazy-retry path_provider contract."""
    client = MagicMock()
    info_calls: list[int] = []

    def _info() -> dict[str, str]:
        info_calls.append(1)
        if len(info_calls) == 1:
            raise RuntimeError("daemon not ready yet")
        return {"DockerRootDir": "/data-root"}

    client.info.side_effect = _info
    backend = _make_backend(tmp_path, client)

    first = backend.disk_monitor_path()
    second = backend.disk_monitor_path()

    assert first is None
    assert second == "/data-root"
    assert len(info_calls) == 2
