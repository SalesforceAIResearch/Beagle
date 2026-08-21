"""Smoke tests for the example_bench plug-in skeleton.

Replace with your benchmark's real test suite. The shipped tests
verify the discovery contract — the entry-point loads, the manifest
file resolves to a readable path, and the adapter class is
importable. They do NOT exercise the harness because the skeleton's
adapter raises ``NotImplementedError``.
"""

from __future__ import annotations

from pathlib import Path


def test_plugin_manifests_returns_existing_yaml() -> None:
    """The B11.2 entry-point callable must return a Path that points
    at a real ``manifest.yaml`` file on disk."""
    from xrlenv_plugins.benchmarks.example_bench.plugin import (
        plugin_manifests,
    )

    result = plugin_manifests()
    paths = [result] if isinstance(result, Path) else list(result)
    assert paths, "plugin_manifests() returned an empty result"
    for path in paths:
        assert path.is_file(), f"manifest path {path} does not exist"
        assert path.name == "manifest.yaml"


def test_adapter_class_is_importable() -> None:
    """The class named in the manifest's ``env_adapter.class_name``
    must be importable. Pin this so a typo in the manifest /
    adapter file surfaces at test time, not at first rollout."""
    from xrlenv_plugins.benchmarks.example_bench.adapter import (
        ExampleBenchEnvAdapter,
    )

    assert ExampleBenchEnvAdapter.__name__ == "ExampleBenchEnvAdapter"
