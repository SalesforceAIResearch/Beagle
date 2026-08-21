"""echo_bench pre-flight tests.

Cheap checks that don't run a real rollout — those need Docker. The
tests here pin:

- The package's manifest is on disk and parses as YAML.
- The entry-point callable returns the manifest path.
- The adapter classes import.
- The resolver enumerates 3 instances and resolves each one.

End-to-end validation lives in
``examples/echo_smoke.py`` (manual run, requires Docker).
"""

from __future__ import annotations

from pathlib import Path

import pytest


def test_manifest_yaml_parses() -> None:
    """The shipped manifest is valid YAML and has the spec-06 fields."""
    import yaml

    pkg_root = Path(__file__).resolve().parents[1]
    manifest = pkg_root / "xrlenv_plugins" / "benchmarks" / "echo_bench" / "manifest.yaml"
    assert manifest.is_file()
    data = yaml.safe_load(manifest.read_text())
    for key in ("name", "version", "instances", "env_adapter", "reward"):
        assert key in data, f"manifest missing required field {key!r}"
    assert data["name"] == "echo-bench"
    assert data["reward"]["mode"] == "in_sandbox_final"


def test_plugin_entry_point_returns_manifest_path() -> None:
    """The entry-point callable returns the manifest.yaml path. This
    is what xrlenv's discovery layer calls at runtime startup."""
    from xrlenv_plugins.benchmarks.echo_bench.plugin import plugin_manifests

    path = plugin_manifests()
    assert path.is_file()
    assert path.name == "manifest.yaml"


def test_adapter_classes_import() -> None:
    """The adapter module imports + exposes both the resolver and the
    EnvAdapter. Catches typos in import paths the manifest references."""
    from xrlenv_plugins.benchmarks.echo_bench.adapter import (
        EchoBenchEnvAdapter,
        EchoBenchInstanceResolver,
    )

    assert EchoBenchEnvAdapter is not None
    assert EchoBenchInstanceResolver is not None


def test_resolver_enumerates_three_instances() -> None:
    """The resolver advertises exactly the three instances documented
    in the README. Pin so the README and the code stay aligned."""
    from xrlenv.control.instance_resolver import InstanceResolverDecl
    from xrlenv_plugins.benchmarks.echo_bench.adapter import (
        EchoBenchInstanceResolver,
    )

    decl = InstanceResolverDecl.model_validate({
        "module": "xrlenv_plugins.benchmarks.echo_bench.adapter",
        "class": "EchoBenchInstanceResolver",
    })
    resolver = EchoBenchInstanceResolver(decl)
    assert sorted(resolver.enumerate_instances()) == [
        "echo-hello", "echo-multiline", "echo-symbols",
    ]


def test_resolve_known_instance_carries_target_and_image() -> None:
    """Each ``resolve()`` call wires the instance's target string into
    init_params and emits the per-instance image tag the build script
    produces. Pin the wiring contract."""
    from xrlenv.control.instance_resolver import InstanceResolverDecl
    from xrlenv_plugins.benchmarks.echo_bench.adapter import (
        EchoBenchInstanceResolver,
    )

    resolver = EchoBenchInstanceResolver(
        InstanceResolverDecl.model_validate({
            "module": "xrlenv_plugins.benchmarks.echo_bench.adapter",
            "class": "EchoBenchInstanceResolver",
        }),
    )
    inst = resolver.resolve("echo-hello")
    assert inst.image == "echo-bench/echo-hello:0.1"
    assert inst.init_params["target"] == "Hello, world!"
    assert inst.init_params["instance_id"] == "echo-hello"
    # Verifier-asset upload — the in-sandbox grader script ships
    # via D12 stage 1, not baked into the image.
    assert len(inst.verifier_uploads) == 1
    upload = inst.verifier_uploads[0]
    assert upload.target_dir == "/opt/xrlenv"
    assert isinstance(upload.tarball, bytes) and len(upload.tarball) > 0


def test_resolve_unknown_instance_raises() -> None:
    """A typo in the instance_id surfaces as a clean KeyError with the
    known set in the message — pin the operator-debugging contract."""
    from xrlenv.control.instance_resolver import InstanceResolverDecl
    from xrlenv_plugins.benchmarks.echo_bench.adapter import (
        EchoBenchInstanceResolver,
    )

    resolver = EchoBenchInstanceResolver(
        InstanceResolverDecl.model_validate({
            "module": "xrlenv_plugins.benchmarks.echo_bench.adapter",
            "class": "EchoBenchInstanceResolver",
        }),
    )
    with pytest.raises(KeyError, match="unknown instance_id"):
        resolver.resolve("does-not-exist")
