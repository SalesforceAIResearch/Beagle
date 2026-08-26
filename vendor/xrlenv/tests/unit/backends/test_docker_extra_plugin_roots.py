"""D22 — DockerBackend.create() bind-mounts extra plug-in roots.

Pre-D22 the docker backend mounted the single in-tree ``xrlenv_plugins/``
sibling next to the imported ``xrlenv`` package. External plug-ins
discovered via ``XRLENV_TEMPLATE_DIRS`` or ``xrlenv.benchmarks``
entry-points lived outside that mount, so the in-sandbox stub's first
``env_setup`` raised ``ModuleNotFoundError``.

These tests pin the post-D22 contract: ``DockerBackendConfig.extra_plugin_roots``
adds one indexed read-only mount per entry, and the resulting PYTHONPATH
layers each mount target onto the in-tree mount.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pytest
from xrlenv.backends.base import ResourceSpec, TemplateRef
from xrlenv.backends.docker import DockerBackend, DockerBackendConfig


def _resources() -> ResourceSpec:
    return ResourceSpec(
        cpu_request=0.5, cpu_limit=1.0,
        mem_request_bytes=128_000_000, mem_limit_bytes=256_000_000,
        disk_request_bytes=1_000_000_000,
    )


def _template() -> TemplateRef:
    return TemplateRef(name="t", image="alpine:latest")


def _mock_client_capturing_run_kwargs() -> tuple[Any, dict[str, Any]]:
    """Build a fake ``docker.DockerClient`` that records the kwargs it
    would have passed to ``containers.run`` and returns a mock container.
    Tests then assert against the captured kwargs without touching a
    real Docker daemon.
    """
    captured: dict[str, Any] = {}

    class _FakeContainer:
        id: ClassVar[str] = "fake-cid"
        short_id: ClassVar[str] = "fake"
        attrs: ClassVar[dict[str, Any]] = {"NetworkSettings": {"Ports": {}}}
        labels: ClassVar[dict[str, str]] = {}

        def reload(self) -> None: ...

    class _FakeContainersAPI:
        def run(self, **kwargs: Any) -> _FakeContainer:
            captured.update(kwargs)
            return _FakeContainer()

    class _FakeClient:
        containers = _FakeContainersAPI()

    return _FakeClient(), captured


@pytest.fixture
def host_paths(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Layout for a backend that has the in-tree mount + 2 extra roots."""
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    xrlenv_pkg = tmp_path / "xrlenv"
    xrlenv_pkg.mkdir()
    in_tree_plugins = tmp_path / "xrlenv_plugins"
    in_tree_plugins.mkdir()
    extra_a = tmp_path / "extra_a"
    extra_a.mkdir()
    extra_b = tmp_path / "extra_b"
    extra_b.mkdir()
    return runs_root, xrlenv_pkg, in_tree_plugins, extra_a


async def _create_and_capture(
    *,
    extra_plugin_roots: tuple[Path, ...],
    runs_root: Path,
    xrlenv_pkg: Path,
    in_tree_plugins: Path,
) -> dict[str, Any]:
    client, captured = _mock_client_capturing_run_kwargs()
    backend = DockerBackend(
        DockerBackendConfig(
            runs_root=runs_root,
            xrlenv_pkg_path=xrlenv_pkg,
            xrlenv_plugins_path=in_tree_plugins,
            extra_plugin_roots=extra_plugin_roots,
            stub_transport="tcp",  # avoid uds chmod path on test boxes
            stub_startup_timeout_s=0.01,
        ),
        client=client,
    )
    # ``create()`` waits for the stub to come up; we suppress that path
    # by intercepting the wait helper. The volumes/PYTHONPATH assembly
    # we care about happens BEFORE the wait, so a short-circuit is fine.
    backend._await_stub_ready = _ready_noop  # type: ignore[assignment]
    try:
        await backend.create(
            template=_template(),
            resources=_resources(),
            network_policy="none",
        )
    except Exception as exc:
        # The fake client returns a mock container; downstream code that
        # walks Docker introspection may raise. The kwargs we want were
        # captured at the start of ``containers.run``, so we don't care
        # about the post-call failure as long as something *was* captured.
        if not captured:
            raise AssertionError(
                f"containers.run was never called; create() failed early: {exc!r}"
            ) from exc
    return captured


async def _ready_noop(*_a: Any, **_kw: Any) -> None:
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Backwards compat — empty extra_plugin_roots looks like pre-D22.
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_extra_roots_volumes_match_pre_d22_shape(
    host_paths: tuple[Path, Path, Path, Path],
) -> None:
    """With ``extra_plugin_roots=()`` the volumes dict carries the same
    two binds the pre-D22 backend produced (xrlenv + xrlenv_plugins),
    and PYTHONPATH stays at the original ``/opt/xrlenv-pkg``."""
    runs_root, xrlenv_pkg, in_tree_plugins, _ = host_paths
    captured = await _create_and_capture(
        extra_plugin_roots=(),
        runs_root=runs_root,
        xrlenv_pkg=xrlenv_pkg,
        in_tree_plugins=in_tree_plugins,
    )
    volumes = captured["volumes"]
    assert volumes == {
        str(xrlenv_pkg): {"bind": "/opt/xrlenv-pkg/xrlenv", "mode": "ro"},
        str(in_tree_plugins): {
            "bind": "/opt/xrlenv-pkg/xrlenv_plugins", "mode": "ro",
        },
    }
    assert captured["environment"]["PYTHONPATH"] == "/opt/xrlenv-pkg"


# ──────────────────────────────────────────────────────────────────────────────
# D22 happy path — one or more extras land at indexed prefixes.
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_single_extra_root_mounts_at_indexed_prefix(
    host_paths: tuple[Path, Path, Path, Path],
) -> None:
    runs_root, xrlenv_pkg, in_tree_plugins, extra_a = host_paths
    captured = await _create_and_capture(
        extra_plugin_roots=(extra_a,),
        runs_root=runs_root,
        xrlenv_pkg=xrlenv_pkg,
        in_tree_plugins=in_tree_plugins,
    )
    volumes = captured["volumes"]
    assert volumes[str(extra_a)] == {
        "bind": "/opt/xrlenv-extras/0", "mode": "ro",
    }
    assert captured["environment"]["PYTHONPATH"] == (
        "/opt/xrlenv-pkg:/opt/xrlenv-extras/0"
    )


@pytest.mark.asyncio
async def test_multiple_extra_roots_use_stable_indexed_prefixes(
    host_paths: tuple[Path, Path, Path, Path],
) -> None:
    """Two same-basename roots can't collide because the container
    target is the positional index, not the host basename."""
    runs_root, xrlenv_pkg, in_tree_plugins, extra_a = host_paths
    extra_b = host_paths[0].parent / "extra_b"
    captured = await _create_and_capture(
        extra_plugin_roots=(extra_a, extra_b),
        runs_root=runs_root,
        xrlenv_pkg=xrlenv_pkg,
        in_tree_plugins=in_tree_plugins,
    )
    volumes = captured["volumes"]
    assert volumes[str(extra_a)]["bind"] == "/opt/xrlenv-extras/0"
    assert volumes[str(extra_b)]["bind"] == "/opt/xrlenv-extras/1"
    assert volumes[str(extra_a)]["mode"] == "ro"
    assert volumes[str(extra_b)]["mode"] == "ro"
    assert captured["environment"]["PYTHONPATH"] == (
        "/opt/xrlenv-pkg:/opt/xrlenv-extras/0:/opt/xrlenv-extras/1"
    )


@pytest.mark.asyncio
async def test_extra_roots_preserve_in_tree_mount(
    host_paths: tuple[Path, Path, Path, Path],
) -> None:
    """The pre-D22 in-tree xrlenv_plugins mount stays alongside the new
    extras — they don't replace it."""
    runs_root, xrlenv_pkg, in_tree_plugins, extra_a = host_paths
    captured = await _create_and_capture(
        extra_plugin_roots=(extra_a,),
        runs_root=runs_root,
        xrlenv_pkg=xrlenv_pkg,
        in_tree_plugins=in_tree_plugins,
    )
    volumes = captured["volumes"]
    assert volumes[str(in_tree_plugins)] == {
        "bind": "/opt/xrlenv-pkg/xrlenv_plugins", "mode": "ro",
    }


# ──────────────────────────────────────────────────────────────────────────────
# D22 acceptance test — PEP-420 namespace-package merging across roots
# ──────────────────────────────────────────────────────────────────────────────
#
# The volume-dict + PYTHONPATH tests above pin the *config shape* the
# backend produces. The acceptance contract D22 promised was actually
# semantic: with two roots on PYTHONPATH, ``import
# xrlenv_plugins.benchmarks.<name>.adapter`` from each root must
# resolve. The end-to-end check (smoke against a real Docker daemon)
# is exercised out-of-band against a live daemon, not in CI.
#
# This test exercises the same Python-import semantics in-process, so
# CI catches a regression without needing Docker. It builds two
# disjoint plug-in-root trees and confirms that adding both to
# ``sys.path`` lets imports from both resolve under the same
# ``xrlenv_plugins.benchmarks`` namespace package.


@pytest.fixture
def two_external_plugin_roots(tmp_path: Path) -> tuple[Path, Path]:
    """Build two independent plug-in roots, each contributing a
    different ``xrlenv_plugins/benchmarks/<name>/adapter.py``.

    Layout for each root::

        <root>/xrlenv_plugins/benchmarks/<name>/__init__.py
        <root>/xrlenv_plugins/benchmarks/<name>/adapter.py
    """
    def _build(root: Path, bench_name: str, marker: str) -> Path:
        leaf = root / "xrlenv_plugins" / "benchmarks" / bench_name
        leaf.mkdir(parents=True)
        (leaf / "__init__.py").write_text("", encoding="utf-8")
        (leaf / "adapter.py").write_text(
            f'BENCH_NAME = "{bench_name}"\nMARKER = "{marker}"\n',
            encoding="utf-8",
        )
        # No __init__.py at the namespace levels — PEP-420 requires
        # their absence for the merging to work.
        return root

    a = _build(tmp_path / "ext_a", "ext_alpha", "alpha")
    b = _build(tmp_path / "ext_b", "ext_beta", "beta")
    return a, b


def test_namespace_packages_merge_across_two_roots(
    two_external_plugin_roots: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D22 acceptance: with two plug-in roots on PYTHONPATH simulating
    the indexed ``/opt/xrlenv-extras/<idx>`` mounts, ``import
    xrlenv_plugins.benchmarks.<name>.adapter`` resolves for a benchmark
    from EITHER root. PEP-420 namespace-package merging is what makes
    the multi-mount strategy work; this test pins it.
    """
    import importlib
    import sys

    root_a, root_b = two_external_plugin_roots

    # Drop any cached import of xrlenv_plugins.* so we observe a fresh
    # resolution against the new sys.path. Tests in this suite that ran
    # earlier may have populated the module cache.
    for mod in list(sys.modules):
        if mod == "xrlenv_plugins" or mod.startswith("xrlenv_plugins."):
            monkeypatch.delitem(sys.modules, mod, raising=False)

    monkeypatch.syspath_prepend(str(root_b))
    monkeypatch.syspath_prepend(str(root_a))

    # Import from root A.
    alpha = importlib.import_module(
        "xrlenv_plugins.benchmarks.ext_alpha.adapter",
    )
    assert alpha.BENCH_NAME == "ext_alpha"
    assert alpha.MARKER == "alpha"

    # Import from root B — succeeds via the same merged namespace
    # package despite living at a different on-disk root. Pre-D22, only
    # one root reached PYTHONPATH so this import would raise
    # ``ModuleNotFoundError``.
    beta = importlib.import_module(
        "xrlenv_plugins.benchmarks.ext_beta.adapter",
    )
    assert beta.BENCH_NAME == "ext_beta"
    assert beta.MARKER == "beta"


def test_namespace_package_levels_have_no_init_py() -> None:
    """The merging in the test above only works because ``xrlenv_plugins/``
    and ``xrlenv_plugins/benchmarks/`` are PEP-420 namespace packages
    (no ``__init__.py``). If a contributor accidentally adds one,
    every external plug-in disappears from import resolution. Pin the
    invariant.
    """
    repo_root = Path(__file__).resolve().parents[3]
    # In-tree namespace levels.
    for ns in (
        repo_root / "xrlenv_plugins" / "__init__.py",
        repo_root / "xrlenv_plugins" / "benchmarks" / "__init__.py",
    ):
        assert not ns.exists(), (
            f"PEP-420 namespace levels must NOT carry __init__.py — "
            f"found one at {ns} which will break external-plug-in "
            f"namespace-package merging (D22). Delete it."
        )
