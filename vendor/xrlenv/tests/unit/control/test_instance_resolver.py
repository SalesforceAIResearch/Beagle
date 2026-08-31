"""Tests for the Slice 9 Pattern A instance resolver (spec 06).

Earlier drafts of this file imported ``SWEBenchInstanceResolver`` from
the in-tree ``xrlenv.envs.swebench`` scaffold. That scaffold has been
deleted as part of the externalize-first refactor (benchmark plug-ins
live under ``xrlenv_plugins/`` now). Tests that need a concrete
resolver use the inline :class:`_StubResolver` defined here so the
platform's Pattern-A wiring stays exercised without a per-benchmark
dependency.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml
from xrlenv.backends.base import MountSpec, ResourceSpec
from xrlenv.control.instance_resolver import (
    InstanceResolver,
    InstanceResolverDecl,
    InstanceResolverImportError,
    ResolvedInstance,
    apply_to_manifest,
    index_path_relative_to,
    load_resolver,
)
from xrlenv.control.template_catalog import (
    EnvAdapterDecl,
    RewardContract,
    TemplateCatalog,
    TemplateManifest,
    load_manifest,
)

# ──────────────────────────────────────────────────────────────────────────────
# Inline resolver for tests — exercises the same protocol benchmark
# plug-ins implement, without dragging in any specific plug-in.
# ──────────────────────────────────────────────────────────────────────────────


class _StubResolver(InstanceResolver):
    """In-test InstanceResolver. Reads a JSON ``index_path`` mapping
    ``instance_id`` → resolved-instance fields; no benchmark coupling."""

    def __init__(self, decl: InstanceResolverDecl) -> None:
        self._decl = decl
        path = decl.index_path
        if path is None:
            self._index: dict[str, dict[str, Any]] = {}
        else:
            self._index = json.loads(Path(path).read_text(encoding="utf-8"))

    def resolve(self, instance_id: str) -> ResolvedInstance:
        if instance_id not in self._index:
            raise KeyError(f"{instance_id} not in index")
        entry = self._index[instance_id]
        resources: ResourceSpec | None = None
        if "cpus" in entry:
            cpu = float(entry["cpus"])
            resources = ResourceSpec(
                cpu_request=cpu, cpu_limit=cpu,
                mem_request_bytes=int(entry.get("mem_gb", 1) * 1024**3),
                mem_limit_bytes=int(entry.get("mem_gb", 1) * 1024**3),
                disk_request_bytes=int(entry.get("disk_gb", 1) * 1024**3),
            )
        network = entry.get("network")
        return ResolvedInstance(
            instance_id=instance_id,
            image=entry.get("image"),
            resources=resources,
            network=network,
            init_params={
                k: v for k, v in entry.items()
                if k not in {"image", "cpus", "mem_gb", "disk_gb", "network"}
            },
        )

    def enumerate_instances(self) -> list[str]:
        return list(self._index)


# ──────────────────────────────────────────────────────────────────────────────
# InstanceResolverDecl + manifest loader
# ──────────────────────────────────────────────────────────────────────────────


def test_decl_accepts_class_alias_for_class_name() -> None:
    decl = InstanceResolverDecl.model_validate(
        {"module": "x.y", "class": "Z", "index_path": "./foo"},
    )
    assert decl.module == "x.y"
    assert decl.class_name == "Z"
    assert decl.index_path == "./foo"


def test_manifest_loader_parses_instances_block(tmp_path: Path) -> None:
    p = tmp_path / "manifest.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "name": "t",
                "image": "im/t:1",
                "env_adapter": {"module": "m", "class": "C"},
                "reward": {"mode": "env_step"},
                "instances": {
                    "module": "xrlenv_plugins.benchmarks.terminal_bench_2.adapter",
                    "class": "TerminalBench2InstanceResolver",
                    "index_path": "./tasks/",
                },
            }
        )
    )
    manifest = load_manifest(p)
    assert manifest.instances is not None
    assert manifest.instances.module == "xrlenv_plugins.benchmarks.terminal_bench_2.adapter"
    assert manifest.instances.class_name == "TerminalBench2InstanceResolver"


def test_manifest_loader_rejects_non_mapping_instances(tmp_path: Path) -> None:
    p = tmp_path / "manifest.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "name": "t", "image": "im/t:1",
                "env_adapter": {"module": "m", "class": "C"},
                "reward": {"mode": "env_step"},
                "instances": ["not", "a", "mapping"],
            }
        )
    )
    from xrlenv.errors import ManifestInvalid

    with pytest.raises(ManifestInvalid):
        load_manifest(p)


# ──────────────────────────────────────────────────────────────────────────────
# load_resolver — generic loader (uses the inline _StubResolver above
# rather than any benchmark-specific class).
# ──────────────────────────────────────────────────────────────────────────────


def test_load_resolver_imports_inline_stub_resolver(tmp_path: Path) -> None:
    index = tmp_path / "i.json"
    index.write_text(json.dumps({"a-1": {"image": "im/a:1"}}))
    decl = InstanceResolverDecl(
        module=__name__,                     # this test module
        **{"class": "_StubResolver"},
        index_path=str(index),
    )
    resolver = load_resolver(decl)
    assert isinstance(resolver, _StubResolver)
    assert resolver.enumerate_instances() == ["a-1"]


def test_load_resolver_unknown_module_raises_typed_error() -> None:
    decl = InstanceResolverDecl(
        module="xrlenv.does_not_exist", **{"class": "X"},
    )
    with pytest.raises(InstanceResolverImportError, match="could not import"):
        load_resolver(decl)


def test_load_resolver_unknown_class_raises_typed_error() -> None:
    decl = InstanceResolverDecl(
        module=__name__, **{"class": "NonExistent"},
    )
    with pytest.raises(InstanceResolverImportError, match="no attribute"):
        load_resolver(decl)


# ──────────────────────────────────────────────────────────────────────────────
# apply_to_manifest
# ──────────────────────────────────────────────────────────────────────────────


def _manifest() -> TemplateManifest:
    return TemplateManifest(
        name="outer", version="0.1", digest="sha256:o", image="im/outer:1",
        resources=ResourceSpec(
            cpu_request=2.0, cpu_limit=2.0,
            mem_request_bytes=4 * 1024**3, mem_limit_bytes=4 * 1024**3,
            disk_request_bytes=10 * 1024**3,
            mounts=(MountSpec(host_path="/dev/null", sandbox_path="/x"),),
        ),
        env_adapter=EnvAdapterDecl(
            module="m", class_name="C", init_params={"shared": "yes"},
        ),
        reward=RewardContract(mode="env_step"),
    )


def test_apply_to_manifest_overlays_image_and_resources() -> None:
    base = _manifest()
    resolved = ResolvedInstance(
        instance_id="i-1",
        image="im/instance:1",
        resources=ResourceSpec(
            cpu_request=8.0, cpu_limit=8.0,
            mem_request_bytes=16 * 1024**3, mem_limit_bytes=16 * 1024**3,
            disk_request_bytes=50 * 1024**3,
        ),
        init_params={"shared": "no", "instance_id": "i-1"},
    )
    effective = apply_to_manifest(base, resolved)
    assert effective.image == "im/instance:1"
    assert effective.resources.cpu_request == 8.0
    assert effective.resources.disk_request_bytes == 50 * 1024**3
    assert effective.env_adapter.init_params == {"shared": "no", "instance_id": "i-1"}
    assert base.image == "im/outer:1"
    assert base.resources.cpu_request == 2.0


def test_apply_to_manifest_mounts_extend_outer() -> None:
    base = _manifest()
    extra_mount = MountSpec(host_path="/dev/zero", sandbox_path="/data")
    resolved = ResolvedInstance(instance_id="i-1", mounts=(extra_mount,))
    effective = apply_to_manifest(base, resolved)
    assert len(effective.resources.mounts) == 2
    assert effective.resources.mounts[1].host_path == "/dev/zero"


def test_apply_to_manifest_noop_when_resolved_empty() -> None:
    base = _manifest()
    effective = apply_to_manifest(base, ResolvedInstance(instance_id="i-1"))
    assert effective is base  # exact same object — no copy


def test_resolved_instance_network_field_validates_against_literal() -> None:
    """Audit-driven: like ``StartRolloutRequest.network`` and
    ``TemplatePolicy.network``, ``ResolvedInstance.network`` is typed
    ``NetworkPolicy`` so a resolver implementation can't slip
    ``"nonee"`` past pydantic and silently fail-open to bridge
    networking inside the Docker backend."""
    from pydantic import ValidationError

    # Valid literal: round-trips cleanly.
    ok = ResolvedInstance(instance_id="i-1", network="none")
    assert ok.network == "none"

    # Invalid literal: pydantic rejects at construction.
    with pytest.raises(ValidationError):
        ResolvedInstance(instance_id="i-1", network="nonee")  # type: ignore[arg-type]


# ──────────────────────────────────────────────────────────────────────────────
# index_path_relative_to
# ──────────────────────────────────────────────────────────────────────────────


def test_index_path_relative_to_handles_absolute_and_relative(tmp_path: Path) -> None:
    decl_rel = InstanceResolverDecl(
        module="m", **{"class": "C"}, index_path="./foo/bar.json",
    )
    decl_abs = InstanceResolverDecl(
        module="m", **{"class": "C"},
        index_path=str(tmp_path / "absolute.json"),
    )
    decl_none = InstanceResolverDecl(module="m", **{"class": "C"})

    assert index_path_relative_to(decl_rel, tmp_path) == (
        (tmp_path / "foo/bar.json").resolve()
    )
    assert index_path_relative_to(decl_abs, tmp_path) == tmp_path / "absolute.json"
    assert index_path_relative_to(decl_none, tmp_path) is None


# ──────────────────────────────────────────────────────────────────────────────
# Resolver behaviour — uses the inline _StubResolver as the concrete
# implementation under test. Real benchmark plug-ins (e.g.
# xrlenv_plugins.benchmarks.terminal_bench_2.adapter.TerminalBench2InstanceResolver
# in Slice 9b) ship their own per-plug-in tests.
# ──────────────────────────────────────────────────────────────────────────────


def test_resolver_resolves_known_instance(tmp_path: Path) -> None:
    idx = tmp_path / "i.json"
    idx.write_text(json.dumps({
        "django__django-11099": {
            "image": "xrlenv/instance:django-11099",
            "cpus": 4.0, "mem_gb": 8.0, "disk_gb": 30.0,
            "test_command": "pytest -x",
        },
    }))
    resolver = _StubResolver(
        InstanceResolverDecl(
            module="x", **{"class": "C"}, index_path=str(idx),
        ),
    )
    resolved = resolver.resolve("django__django-11099")
    assert resolved.image == "xrlenv/instance:django-11099"
    assert resolved.resources.cpu_request == 4.0
    assert resolved.init_params["test_command"] == "pytest -x"


def test_resolver_unknown_instance_raises_keyerror(tmp_path: Path) -> None:
    idx = tmp_path / "i.json"
    idx.write_text(json.dumps({"a-1": {"image": "im/a:1"}}))
    resolver = _StubResolver(
        InstanceResolverDecl(
            module="x", **{"class": "C"}, index_path=str(idx),
        ),
    )
    with pytest.raises(KeyError, match="not in index"):
        resolver.resolve("never-existed")


# ──────────────────────────────────────────────────────────────────────────────
# Coordinator integration: instance overlay before placement
# ──────────────────────────────────────────────────────────────────────────────


def test_coordinator_applies_resolver_overlay_before_placement(
    tmp_path: Path,
) -> None:
    """When init carries instance_id and the manifest has an instances
    block, the coordinator overlays the resolver's image onto the
    effective manifest passed to the scheduler."""
    from xrlenv.control.coordinator import RolloutCoordinator
    from xrlenv.control.scheduler import Placement
    from xrlenv.control.state import InMemoryStateStore

    idx = tmp_path / "i.json"
    idx.write_text(json.dumps({
        "i-A": {"image": "xrlenv/inst-A:1", "cpus": 2.0, "mem_gb": 4.0, "disk_gb": 8.0},
    }))
    base = _manifest().model_copy(
        update={
            "instances": InstanceResolverDecl(
                module=__name__,
                **{"class": "_StubResolver"},
                index_path=str(idx),
            )
        }
    )

    catalog = TemplateCatalog()
    catalog.register(base)
    sched = MagicMock()
    sched.place.return_value = Placement(
        node=MagicMock(node_id="n1"), backend="docker", score=1,
    )
    coord = RolloutCoordinator(
        catalog=catalog, scheduler=sched, state=InMemoryStateStore(),
    )
    effective, resolved_network, _uploads = coord._maybe_resolve_instance(
        base, {"instance_id": "i-A"},
    )
    assert effective.image == "xrlenv/inst-A:1"
    assert effective.resources.cpu_request == 2.0
    # The stub resolver doesn't supply a per-task network; the
    # coordinator's caller takes the request-level value.
    assert resolved_network is None
    # Cache: a second call with the same template doesn't re-import.
    effective2, _, _ = coord._maybe_resolve_instance(
        base, {"instance_id": "i-A"},
    )
    assert effective2.image == effective.image


def test_coordinator_skips_overlay_when_no_instance_id() -> None:
    """Pattern-A template invoked without an instance_id falls back to
    the outer manifest's defaults."""
    from xrlenv.control.coordinator import RolloutCoordinator
    from xrlenv.control.state import InMemoryStateStore

    base = _manifest().model_copy(
        update={
            "instances": InstanceResolverDecl(
                module=__name__,
                **{"class": "_StubResolver"},
            )
        }
    )
    coord = RolloutCoordinator(
        catalog=TemplateCatalog(),
        scheduler=MagicMock(),
        state=InMemoryStateStore(),
    )
    effective, resolved_network, _ = coord._maybe_resolve_instance(base, {})
    assert effective is base
    assert resolved_network is None


def test_coordinator_resolved_network_returned_alongside_manifest(tmp_path: Path) -> None:
    """Pattern A: when the resolver supplies ``network`` for a task,
    ``_maybe_resolve_instance`` returns it as the second tuple
    element so ``start_rollout`` can layer it over the request /
    DEFAULT_NETWORK fallback. Coordinator-level test that the wire
    actually carries the resolver value out."""
    from xrlenv.control.coordinator import RolloutCoordinator
    from xrlenv.control.state import InMemoryStateStore

    index = tmp_path / "index.json"
    index.write_text(json.dumps({
        "i-hermetic": {"image": "im/x:1", "cpus": 1, "network": "none"},
        "i-default": {"image": "im/x:1", "cpus": 1},  # no network
    }))
    base = _manifest().model_copy(
        update={
            "instances": InstanceResolverDecl(
                module=__name__,
                **{"class": "_StubResolver"},
                index_path=str(index),
            )
        }
    )
    catalog = TemplateCatalog()
    catalog.register(base)
    coord = RolloutCoordinator(
        catalog=catalog,
        scheduler=MagicMock(),
        state=InMemoryStateStore(),
    )
    _, hermetic, _ = coord._maybe_resolve_instance(base, {"instance_id": "i-hermetic"})
    _, default, _ = coord._maybe_resolve_instance(base, {"instance_id": "i-default"})
    assert hermetic == "none"
    assert default is None
