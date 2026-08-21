"""Tests for plug-in manifest discovery + bind-mount path discovery.

Plug-ins live at ``<repo-root>/xrlenv_plugins/<category>/<name>/`` with
a single ``manifest.yaml`` at the plug-in root. The platform must:

- discover those manifest files at runtime startup so operators don't
  enumerate them via ``--template-dirs``;
- expose the plug-ins root path so the Docker backend can bind-mount
  it alongside ``xrlenv`` (closes the "adapter module not importable
  inside the sandbox" failure mode without requiring per-image pip
  installs).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from xrlenv.control.template_discovery import (
    ENTRY_POINT_GROUP,
    TEMPLATE_DIRS_ENV_VAR,
    DiscoveredManifest,
    extra_template_dirs_from_env,
    find_entry_point_manifest_files,
    find_external_template_dir_manifests,
    find_plugin_manifest_files,
    find_plugin_root,
)


def _layout_plugin_dirs(repo_root: Path) -> None:
    (repo_root / "xrlenv").mkdir()
    plugins = repo_root / "xrlenv_plugins"
    tb2 = plugins / "benchmarks" / "terminal_bench_2"
    tb2.mkdir(parents=True)
    (tb2 / "manifest.yaml").write_text("name: t\n")
    another = plugins / "benchmarks" / "another"
    another.mkdir(parents=True)
    (another / "manifest.yaml").write_text("name: a\n")
    # A nested manifest.yaml inside a plug-in's own fixture / tasks tree
    # must NOT be returned (the two-level glob pins discovery to the
    # plug-in root).
    (tb2 / "tasks" / "deeper").mkdir(parents=True)
    (tb2 / "tasks" / "deeper" / "manifest.yaml").write_text("name: nope\n")


def test_find_plugin_manifest_files_returns_one_per_plugin(
    tmp_path: Path,
) -> None:
    _layout_plugin_dirs(tmp_path)
    result = find_plugin_manifest_files(tmp_path / "xrlenv")
    plugin_names = sorted(p.parent.name for p in result)
    assert plugin_names == ["another", "terminal_bench_2"]


def test_find_plugin_manifest_files_ignores_nested_manifests(
    tmp_path: Path,
) -> None:
    _layout_plugin_dirs(tmp_path)
    result = find_plugin_manifest_files(tmp_path / "xrlenv")
    # The deep tasks/deeper/manifest.yaml must not appear.
    assert all(p.parent.parent.name == "benchmarks" for p in result), (
        f"discovery returned a manifest at unexpected depth: {result}"
    )


def test_find_plugin_manifest_files_returns_empty_when_no_plugins(
    tmp_path: Path,
) -> None:
    (tmp_path / "xrlenv").mkdir()  # platform present, no plug-ins root
    assert find_plugin_manifest_files(tmp_path / "xrlenv") == []


def test_find_plugin_root_returns_path_when_present(tmp_path: Path) -> None:
    _layout_plugin_dirs(tmp_path)
    root = find_plugin_root(tmp_path / "xrlenv")
    assert root == tmp_path / "xrlenv_plugins"


def test_find_plugin_root_returns_none_when_absent(tmp_path: Path) -> None:
    (tmp_path / "xrlenv").mkdir()
    assert find_plugin_root(tmp_path / "xrlenv") is None


# Note: pre-P1.7.D this file carried a
# ``test_real_repo_layout_finds_terminal_bench_2`` end-to-end check
# that pinned the discovery helper against the in-tree
# ``xrlenv_plugins/benchmarks/terminal_bench_2/`` plug-in. Under the
# slim pivot there are no in-tree benchmark plug-ins; the discovery
# property is fully covered by the synthesized-fixture tests above
# + the entry-point discovery tests + the external-package-skeleton
# tests below.


# ── B11.1: XRLENV_TEMPLATE_DIRS env var ──────────────────────────────────────


def test_extra_template_dirs_from_env_unset_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No env var → empty list. Pin both the explicit-unset and the
    explicit-empty-string cases (treated identically)."""
    import os

    monkeypatch.delenv(TEMPLATE_DIRS_ENV_VAR, raising=False)
    assert extra_template_dirs_from_env() == []
    monkeypatch.setenv(TEMPLATE_DIRS_ENV_VAR, "")
    assert extra_template_dirs_from_env() == []
    # Whitespace-only entries are also dropped.
    monkeypatch.setenv(TEMPLATE_DIRS_ENV_VAR, os.pathsep.join(["", "  "]))
    assert extra_template_dirs_from_env() == []


def test_extra_template_dirs_from_env_resolves_existing_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Existing dirs are resolved to absolute paths; non-existent
    entries are dropped (with a warning logged); ordering is preserved.
    """
    import os

    a = tmp_path / "external_a"
    a.mkdir()
    b = tmp_path / "external_b"
    b.mkdir()
    bogus = tmp_path / "does_not_exist"

    raw = os.pathsep.join([str(a), str(bogus), str(b)])
    monkeypatch.setenv(TEMPLATE_DIRS_ENV_VAR, raw)

    result = extra_template_dirs_from_env()
    assert result == [a.resolve(), b.resolve()]


def test_extra_template_dirs_from_env_expands_user_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``~/foo`` is expanded via :func:`Path.expanduser` so operators
    can write user-relative paths in the env var."""
    monkeypatch.setenv("HOME", str(tmp_path))
    target = tmp_path / "tilde_dir"
    target.mkdir()

    monkeypatch.setenv(TEMPLATE_DIRS_ENV_VAR, "~/tilde_dir")
    result = extra_template_dirs_from_env()
    assert result == [target.resolve()]


# ── B11.2: entry-points discovery ────────────────────────────────────────────


def test_find_entry_point_manifest_files_no_eps_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No entry points in the group → empty list, no error."""
    import importlib.metadata

    class _EmptyEPs:
        def __iter__(self) -> object:
            return iter([])

    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda *_a, **_kw: _EmptyEPs(),
    )
    assert find_entry_point_manifest_files() == []


def _fake_entry_point(name: str, loaded: object) -> object:
    """Build a stub object that quacks like ``importlib.metadata.EntryPoint``
    for our purposes: has a ``.name`` attribute and a ``.load()`` method.
    """
    class _EP:
        def __init__(self) -> None:
            self.name = name

        def load(self) -> object:
            return loaded

    return _EP()


def test_find_entry_point_manifest_files_callable_returning_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An entry point that loads to a callable returning a single
    :class:`Path` is registered."""
    import importlib.metadata

    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text("name: external\n")

    def _loader() -> Path:
        return manifest_path

    fake_ep = _fake_entry_point("external_bench", _loader)
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda *_a, **_kw: [fake_ep],
    )

    result = find_entry_point_manifest_files()
    assert [m.manifest_path for m in result] == [manifest_path]


def test_find_entry_point_manifest_files_callable_returning_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A multi-manifest plug-in returns a list — every entry registers."""
    import importlib.metadata

    a = tmp_path / "a.yaml"
    a.write_text("name: a\n")
    b = tmp_path / "b.yaml"
    b.write_text("name: b\n")

    def _loader() -> list[Path]:
        return [a, b]

    fake_ep = _fake_entry_point("multi", _loader)
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda *_a, **_kw: [fake_ep],
    )

    result = find_entry_point_manifest_files()
    assert sorted(m.manifest_path for m in result) == sorted([a, b])


def test_find_entry_point_manifest_files_one_failing_does_not_block_others(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If one entry point's loader raises, the OTHER plug-ins must
    still register. Pin the per-plug-in failure isolation contract.
    """
    import importlib.metadata

    healthy_path = tmp_path / "ok.yaml"
    healthy_path.write_text("name: ok\n")

    def _broken() -> Path:
        raise RuntimeError("vendor bug")

    def _ok() -> Path:
        return healthy_path

    eps = [
        _fake_entry_point("broken", _broken),
        _fake_entry_point("ok", _ok),
    ]
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda *_a, **_kw: eps,
    )

    result = find_entry_point_manifest_files()
    assert [m.manifest_path for m in result] == [healthy_path]


def test_find_entry_point_manifest_files_skips_non_callable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Entry-point loads to something that isn't callable (operator
    misconfigured the pyproject.toml entry-point target) → log + skip,
    no crash.
    """
    import importlib.metadata

    fake_ep = _fake_entry_point("bad", "not-a-callable")
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda *_a, **_kw: [fake_ep],
    )
    assert find_entry_point_manifest_files() == []


def test_find_entry_point_manifest_files_skips_non_existent_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A loader that returns a path that doesn't exist on disk is
    logged and skipped — we never register a manifest the catalog
    can't read."""
    import importlib.metadata

    bogus = tmp_path / "nope.yaml"  # never created

    def _loader() -> Path:
        return bogus

    fake_ep = _fake_entry_point("bogus", _loader)
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda *_a, **_kw: [fake_ep],
    )
    assert find_entry_point_manifest_files() == []


def test_entry_point_group_constant_is_xrlenv_benchmarks() -> None:
    """Pin the group name — it's wire format for plug-in authors and
    a renaming would silently break every external pip package."""
    assert ENTRY_POINT_GROUP == "xrlenv.benchmarks"


def test_template_dirs_env_var_constant_is_xrlenv_template_dirs() -> None:
    """Pin the env var name — same wire-format reasoning as above."""
    assert TEMPLATE_DIRS_ENV_VAR == "XRLENV_TEMPLATE_DIRS"


# ── B11.4: external-package skeleton self-validates ──────────────────────────


def test_external_package_skeleton_manifest_is_readable() -> None:
    """The shipped skeleton at ``templates/external_benchmark_package/``
    is a copy-paste source for plug-in authors. If the skeleton's
    own ``manifest.yaml`` ever drifts out of valid spec-06 shape,
    every plug-in author who copies it inherits the bug. Pin the
    skeleton's manifest as YAML-valid + carrying the documented
    fields.
    """
    import yaml

    repo_root = Path(__file__).resolve().parents[3]
    manifest_path = (
        repo_root
        / "templates"
        / "external_benchmark_package"
        / "xrlenv_plugins"
        / "benchmarks"
        / "example_bench"
        / "manifest.yaml"
    )
    assert manifest_path.is_file(), (
        f"external-package skeleton missing its manifest at {manifest_path}"
    )
    data = yaml.safe_load(manifest_path.read_text())
    # Spec-06 required fields. Only assert presence — values are
    # placeholders the plug-in author replaces.
    for key in ("name", "version", "image", "resources", "env_adapter", "reward"):
        assert key in data, (
            f"skeleton manifest missing required spec-06 field {key!r}: {data}"
        )


def test_external_package_skeleton_pyproject_declares_entry_point() -> None:
    """The skeleton's ``pyproject.toml`` must declare the
    ``xrlenv.benchmarks`` entry-point group — that's the whole point
    of the skeleton. A missing / renamed group would silently produce
    a plug-in that never registers."""
    repo_root = Path(__file__).resolve().parents[3]
    pyproject = (
        repo_root
        / "templates"
        / "external_benchmark_package"
        / "pyproject.toml"
    )
    body = pyproject.read_text()
    assert '[project.entry-points."xrlenv.benchmarks"]' in body, (
        "skeleton pyproject.toml is missing the xrlenv.benchmarks entry-point group"
    )
    # And the entry-point must point at the plugin.py callable shape
    # the docs describe.
    assert "plugin:plugin_manifests" in body, (
        "skeleton pyproject.toml entry-point should target the "
        "plugin_manifests callable in plugin.py"
    )


# ── D22: plug-in root resolution + system-path guard ────────────────────────


def test_find_entry_point_manifest_resolves_plugin_root_canonical_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Manifests under the canonical ``xrlenv_plugins/<cat>/<name>/``
    layout get a ``plugin_root`` set to the parent of ``xrlenv_plugins``
    (i.e. the directory you'd put on PYTHONPATH).
    """
    import importlib.metadata

    pkg_root = tmp_path / "external"
    leaf = pkg_root / "xrlenv_plugins" / "benchmarks" / "foo"
    leaf.mkdir(parents=True)
    manifest_path = leaf / "manifest.yaml"
    manifest_path.write_text("name: foo\n")

    def _loader() -> Path:
        return manifest_path

    fake_ep = _fake_entry_point("foo", _loader)
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda *_a, **_kw: [fake_ep],
    )

    result = find_entry_point_manifest_files()
    assert len(result) == 1
    assert result[0] == DiscoveredManifest(
        manifest_path=manifest_path.resolve(),
        plugin_root=pkg_root.resolve(),
    )


def test_find_entry_point_manifest_no_xrlenv_plugins_ancestor_returns_none_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """Manifest dropped at an arbitrary path with no ``xrlenv_plugins/``
    ancestor still registers, but ``plugin_root`` is None and the
    discovery layer logs a warning (the adapter must reach the sandbox
    via image-bundled code or another platform-injected mount)."""
    import importlib.metadata
    import logging

    manifest_path = tmp_path / "loose" / "manifest.yaml"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("name: loose\n")

    def _loader() -> Path:
        return manifest_path

    fake_ep = _fake_entry_point("loose", _loader)
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda *_a, **_kw: [fake_ep],
    )

    caplog.set_level(logging.WARNING, logger="xrlenv.control.template_discovery")
    result = find_entry_point_manifest_files()
    assert len(result) == 1
    assert result[0].plugin_root is None
    assert any(
        "is not under an xrlenv_plugins/ ancestor" in r.message
        for r in caplog.records
    )


def test_find_external_template_dir_manifests_walks_env_var_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B11.1 + D22 — XRLENV_TEMPLATE_DIRS dirs are walked for
    ``manifest.yaml`` files (recursively, so depth doesn't matter), and
    each manifest carries its plug-in root.
    """
    pkg_root = tmp_path / "ext"
    leaf = pkg_root / "xrlenv_plugins" / "benchmarks" / "bar"
    leaf.mkdir(parents=True)
    manifest_path = leaf / "manifest.yaml"
    manifest_path.write_text("name: bar\n")

    monkeypatch.setenv(TEMPLATE_DIRS_ENV_VAR, str(pkg_root))

    result = find_external_template_dir_manifests()
    assert len(result) == 1
    assert result[0].manifest_path == manifest_path.resolve()
    assert result[0].plugin_root == pkg_root.resolve()


def test_find_external_template_dir_manifests_unset_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No env var → empty list; runtime starts up cleanly with
    no external plug-ins configured."""
    monkeypatch.delenv(TEMPLATE_DIRS_ENV_VAR, raising=False)
    assert find_external_template_dir_manifests() == []


def test_plugin_root_drops_forbidden_system_paths(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A manifest whose plug-in root would resolve under a forbidden
    system prefix (``/etc``, ``/proc``, etc.) is treated as
    plugin_root=None with a warning. Narrower than the spec-19 mount
    allowlist (which would deny ``/home`` and break the dev workflow);
    here we only catch genuinely-system paths.

    Uses ``/dev`` (always-present unix-system dir) and fabricates a
    manifest path that doesn't exist on disk — :func:`Path.resolve`
    happily constructs the path object without validating existence,
    so we can exercise the guard without mocking.
    """
    import logging

    from xrlenv.control.template_discovery import _resolve_plugin_root

    fabricated = Path("/dev/xrlenv_plugins/benchmarks/x/manifest.yaml")
    caplog.set_level(logging.WARNING, logger="xrlenv.control.template_discovery")
    assert _resolve_plugin_root(fabricated) is None
    assert any(
        "forbidden system prefix" in r.message
        for r in caplog.records
    )


def test_plugin_root_drops_root_path_specifically(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A manifest at ``/xrlenv_plugins/benchmarks/x/manifest.yaml``
    would land plug-in root at ``/`` — exact-match guard fires (root
    is technically a prefix of every path, so the forbidden-prefix
    check uses exact-match-only semantics for ``Path("/")``).
    """
    import logging

    from xrlenv.control.template_discovery import _resolve_plugin_root

    fabricated = Path("/xrlenv_plugins/benchmarks/x/manifest.yaml")
    caplog.set_level(logging.WARNING, logger="xrlenv.control.template_discovery")
    assert _resolve_plugin_root(fabricated) is None
    assert any(
        "forbidden system prefix" in r.message
        for r in caplog.records
    )
