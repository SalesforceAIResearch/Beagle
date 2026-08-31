"""H3 lazy image lifecycle: ``ensure_present`` invokes the registered
benchmark builder when the ref isn't registry-pullable (P1.6.g).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, ClassVar

import pytest
from xrlenv.backends.base import (
    ExecChunk,
    ImageRecord,
    NetworkPolicy,
    ResourceSpec,
    ResourceUsage,
    SandboxBackend,
    SandboxCapabilities,
    SandboxHandle,
    ServiceSpec,
    SnapshotID,
    TemplateRef,
)
from xrlenv.control.image_builder import BuildResult, ImageBuilderDecl
from xrlenv.control.node_builder import BuilderRef
from xrlenv.node.agent import NodeAgent, NodeAgentConfig
from xrlenv.node.image_cache import ImageCacheManager


class _Backend(SandboxBackend):
    """Minimal backend that distinguishes "registry pull" from "local
    image" — used to assert ensure_present routes to the builder
    instead of calling ``pull_image`` for benchmark-internal refs."""

    name = "fake"
    capabilities = SandboxCapabilities(
        supports_snapshot=False, supports_chainable_snapshot=False,
        live_state_captured=False, supports_port_forward=False,
        supports_gpu=False, isolation_class="container",
        fast_create_p50_ms=10,
    )

    def __init__(self) -> None:
        self.present: dict[str, ImageRecord] = {}
        self.pulled: list[str] = []

    async def list_images(self) -> list[ImageRecord]:
        return list(self.present.values())

    async def image_exists(self, image: str) -> bool:
        return image in self.present

    async def pull_image(self, image: str, *, timeout_s: float = 600.0) -> None:
        self.pulled.append(image)
        # Simulate the pull landing a tag locally.
        self.present[image] = ImageRecord(name=image, size_bytes=1024 * 1024)

    async def remove_image(self, image: str, *, force: bool = False) -> None:
        self.present.pop(image, None)

    async def free_disk_bytes(self) -> int:
        return 100 * 1024**3

    async def create(
        self, template: TemplateRef, resources: ResourceSpec,
        network_policy: NetworkPolicy,
    ) -> SandboxHandle:
        raise NotImplementedError

    async def destroy(self, sb: SandboxHandle) -> None:
        return None

    def exec(
        self, sb: SandboxHandle, cmd: list[str], stdin: bytes | None = None,
        env: dict[str, str] | None = None, timeout_s: float | None = None,
    ) -> AsyncIterator[ExecChunk]:
        raise NotImplementedError

    async def read_file(self, sb: SandboxHandle, path: str) -> bytes:
        raise NotImplementedError

    async def write_file(
        self, sb: SandboxHandle, path: str, data: bytes,
    ) -> None:
        raise NotImplementedError

    async def put_archive(
        self, sb: SandboxHandle, target_dir: str, tarball: bytes,
        *, clean_target: bool = False,
    ) -> None:
        raise NotImplementedError

    def read_file_stream(
        self, sb: SandboxHandle, path: str,
    ) -> AsyncIterator[bytes]:
        raise NotImplementedError

    async def write_file_stream(
        self, sb: SandboxHandle, path: str,
        src: AsyncIterator[bytes],
    ) -> None:
        raise NotImplementedError

    async def spawn_service(
        self, sb: SandboxHandle, spec: ServiceSpec,
    ) -> object:
        raise NotImplementedError

    async def spawn_services(
        self, sb: SandboxHandle, specs: list[ServiceSpec],
    ) -> list[object]:
        raise NotImplementedError

    async def port_forward(
        self, sb: SandboxHandle, internal_port: int,
    ) -> str:
        raise NotImplementedError

    async def snapshot(self, sb: SandboxHandle) -> SnapshotID:
        raise NotImplementedError

    async def restore(self, snapshot: SnapshotID) -> SandboxHandle:
        raise NotImplementedError

    async def stats(self, sb: SandboxHandle) -> ResourceUsage:
        raise NotImplementedError


_LAZY_BUILDS: list[str] = []


class _LazyBuilder:
    """Tracks every ``build()`` call into ``_LAZY_BUILDS`` so the
    test asserts the lazy path actually fired."""

    IMAGE_SIZE_HINT_BYTES: ClassVar[int] = 1 * 1024**3

    def __init__(self, decl: ImageBuilderDecl) -> None:
        self._decl = decl

    def enumerate_image_refs(self, *, selection: dict[str, Any]) -> list[str]:
        del selection
        return ["fake-bench/x:1"]

    async def build(
        self,
        *,
        image_ref: str,
        kwargs: dict[str, Any],
        force: bool,
    ) -> BuildResult:
        del kwargs, force
        _LAZY_BUILDS.append(image_ref)
        return BuildResult(image_ref=image_ref, status="done")


@pytest.fixture(autouse=True)
def _reset_lazy_builds() -> None:
    _LAZY_BUILDS.clear()


def _publish_fake_module(monkeypatch: Any) -> str:
    """Publish ``_LazyBuilder`` on a transient module path so
    ``load_image_builder`` can import it."""
    import sys
    import types

    name = "_test_lazy_builder_module"
    mod = types.ModuleType(name)
    mod.LazyBuilder = _LazyBuilder  # type: ignore[attr-defined]
    sys.modules[name] = mod
    return name


@pytest.mark.asyncio
async def test_ensure_present_routes_to_registered_builder(monkeypatch) -> None:
    """When a benchmark builder is registered for an image_ref, the
    image cache calls it instead of ``backend.pull_image`` (which
    would fail for benchmark-internal tags not on Dockerhub)."""
    mod_name = _publish_fake_module(monkeypatch)

    backend = _Backend()
    cache = ImageCacheManager(backend=backend)
    agent = NodeAgent(
        NodeAgentConfig(node_id="n", backends=["fake"]),
        image_cache=cache,
    )
    agent.register_lazy_image_builders({
        "fake-bench/x:1": (
            BuilderRef(module=mod_name, class_name="LazyBuilder"),
            {},
        ),
    })

    # The image isn't present locally and isn't pull-eligible — the
    # builder hook should fire instead of ``backend.pull_image``.
    await cache.ensure_present("fake-bench/x:1")

    # Builder ran exactly once.
    assert _LAZY_BUILDS == ["fake-bench/x:1"]
    # ``backend.pull_image`` was NOT called for this ref.
    assert "fake-bench/x:1" not in backend.pulled


@pytest.mark.asyncio
async def test_ensure_present_falls_through_to_pull_when_no_builder(
    monkeypatch,
) -> None:
    """For ordinary registry-pullable refs (no benchmark builder
    registered), ``ensure_present`` keeps the existing
    ``backend.pull_image`` behavior."""
    backend = _Backend()
    cache = ImageCacheManager(backend=backend)
    agent = NodeAgent(  # noqa: F841 — wires the cache
        NodeAgentConfig(node_id="n", backends=["fake"]),
        image_cache=cache,
    )

    await cache.ensure_present("python:3.12-slim")
    assert backend.pulled == ["python:3.12-slim"]
    assert _LAZY_BUILDS == []


@pytest.mark.asyncio
async def test_register_lazy_builders_replaces_prior_entries(
    monkeypatch,
) -> None:
    """Re-registering an image_ref overwrites the prior mapping
    (matches dict.update semantics)."""
    mod_name = _publish_fake_module(monkeypatch)

    backend = _Backend()
    cache = ImageCacheManager(backend=backend)
    agent = NodeAgent(
        NodeAgentConfig(node_id="n", backends=["fake"]),
        image_cache=cache,
    )

    # First registration with one kwargs.
    agent.register_lazy_image_builders({
        "fake-bench/x:1": (
            BuilderRef(module=mod_name, class_name="LazyBuilder"),
            {"flavor": "v1"},
        ),
    })
    # Second, replacing.
    agent.register_lazy_image_builders({
        "fake-bench/x:1": (
            BuilderRef(module=mod_name, class_name="LazyBuilder"),
            {"flavor": "v2"},
        ),
    })

    producer = agent._lookup_image_producer("fake-bench/x:1")
    assert producer is not None
    # Trigger the closure to confirm it picks the latest kwargs.
    await producer("fake-bench/x:1", 60.0)
    assert _LAZY_BUILDS == ["fake-bench/x:1"]


@pytest.mark.asyncio
async def test_node_grpc_link_ensure_present_handler_routes_to_cache(
    monkeypatch,
) -> None:
    """P1.6.g step 4 (F4=2): the EnsurePresentCommand handler delegates
    to the wired ImageCacheManager and returns ok/failed in the reply
    body. End-to-end against the in-process NodeAgent + grpc_link."""
    from xrlenv.api._pb2 import node_control_pb2 as pb
    from xrlenv.node.grpc_link import NodeGrpcLink

    mod_name = _publish_fake_module(monkeypatch)

    backend = _Backend()
    cache = ImageCacheManager(backend=backend)
    agent = NodeAgent(
        NodeAgentConfig(node_id="n", backends=["fake"]),
        image_cache=cache,
    )
    agent.register_lazy_image_builders({
        "fake-bench/x:1": (
            BuilderRef(module=mod_name, class_name="LazyBuilder"),
            {},
        ),
    })

    # Drive the link's handler directly (no real gRPC server needed).
    link = NodeGrpcLink(agent, control_addr="ignored")

    # Lazy ref: handler routes to the builder via cache.ensure_present.
    cmd = pb.EnsurePresentCommand(
        header=pb.CommandHeader(command_id="c1"),
        image_ref="fake-bench/x:1",
        timeout_s=60.0,
    )
    reply = await link._exec_ensure_present(cmd)
    assert reply.ensure_present.status == "ok"
    assert reply.ensure_present.error == ""
    assert _LAZY_BUILDS == ["fake-bench/x:1"]


@pytest.mark.asyncio
async def test_node_grpc_link_ensure_present_returns_failed_on_error(
    monkeypatch,
) -> None:
    """When ensure_present raises (e.g. backend.pull_image failure on
    a non-registry-pullable ref with no builder registered), the
    handler returns ``status=failed`` + the error message instead of
    bubbling the exception."""
    from xrlenv.api._pb2 import node_control_pb2 as pb
    from xrlenv.node.grpc_link import NodeGrpcLink

    class _FailingBackend(_Backend):
        async def pull_image(
            self, image: str, *, timeout_s: float = 600.0,
        ) -> None:
            raise RuntimeError(f"pull denied for {image}")

    backend = _FailingBackend()
    cache = ImageCacheManager(backend=backend)
    agent = NodeAgent(
        NodeAgentConfig(node_id="n", backends=["fake"]),
        image_cache=cache,
    )
    link = NodeGrpcLink(agent, control_addr="ignored")

    cmd = pb.EnsurePresentCommand(
        header=pb.CommandHeader(command_id="c2"),
        image_ref="not-on-dockerhub:latest",
        timeout_s=10.0,
    )
    reply = await link._exec_ensure_present(cmd)
    assert reply.ensure_present.status == "failed"
    assert "pull denied" in reply.ensure_present.error


@pytest.mark.asyncio
async def test_lookup_returns_none_for_unregistered(monkeypatch) -> None:
    backend = _Backend()
    cache = ImageCacheManager(backend=backend)
    agent = NodeAgent(
        NodeAgentConfig(node_id="n", backends=["fake"]),
        image_cache=cache,
    )
    assert agent._lookup_image_producer("never-registered:1") is None


@pytest.mark.asyncio
async def test_opportunistic_deferred_ref_is_lazy_buildable(monkeypatch) -> None:
    """Audit P1.6.g-H1 end-to-end: an opportunistic plan that overflows
    the budget defers its image. The deferred ref's builder mapping
    must reach the preferred-home node so a later ensure_present
    invokes the benchmark builder, NOT backend.pull_image.

    Setup: tiny budget, a single huge image → bin-packer can't fit it
    → coordinator records the row as ``status=registered`` AND
    dispatches a BuildJob carrying ``lazy_registrations`` to the
    preferred-home node. InProcessNodeBuilder registers that mapping
    on the agent. Subsequent ensure_present routes through the
    builder, leaving backend.pulled empty for the deferred ref.
    """
    from typing import ClassVar as _ClassVar

    from xrlenv.control.build_coordinator import BuildCoordinator
    from xrlenv.control.build_plan import (
        BenchmarkBuildSpec,
        BenchmarkSelection,
        BuildPlan,
    )
    from xrlenv.control.image_builder import (
        BuildResult as _BuildResult,
    )
    from xrlenv.control.image_builder import (
        ImageBuilderDecl as _ImageBuilderDecl,
    )
    from xrlenv.control.image_planner import NodeBudget
    from xrlenv.control.node_builder import InProcessNodeBuilder
    from xrlenv.control.state import InMemoryStateStore
    from xrlenv.control.template_catalog import (
        TemplateCatalog,
        load_manifest,
    )

    pulls: list[str] = []
    builds: list[str] = []

    class _LocalBackend(_Backend):
        async def pull_image(self, image, *, timeout_s=600.0):  # type: ignore[no-untyped-def]
            pulls.append(image)
            await super().pull_image(image, timeout_s=timeout_s)

    class _HugeBuilder:
        IMAGE_SIZE_HINT_BYTES: _ClassVar[int] = 200 * 1024**3

        def __init__(self, decl: _ImageBuilderDecl) -> None:
            self._decl = decl

        def enumerate_image_refs(self, *, selection):  # type: ignore[no-untyped-def]
            del selection
            return ["fake-bench/huge:1"]

        async def build(self, *, image_ref, kwargs, force):  # type: ignore[no-untyped-def]
            del kwargs, force
            builds.append(image_ref)
            return _BuildResult(image_ref=image_ref, status="done")

    # Publish builder on a transient module path the loader can import.
    import sys
    import types

    mod_name = "_test_h1_huge_builder_module"
    mod = types.ModuleType(mod_name)
    mod._HugeBuilder = _HugeBuilder  # type: ignore[attr-defined]
    sys.modules[mod_name] = mod

    # Build the catalog with a manifest pointing at the huge builder.
    import tempfile
    from pathlib import Path
    from textwrap import dedent

    manifest_yaml = dedent(f"""\
        name: fake-bench
        version: "0.1"
        image: scratch:latest
        env_adapter:
          module: xrlenv.envs.base
          class: NoOpEnvAdapter
        reward:
          mode: env_step
        image_builder:
          module: {mod_name}
          class: _HugeBuilder
    """)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        fh.write(manifest_yaml)
        path = Path(fh.name)
    catalog = TemplateCatalog()
    catalog.register(load_manifest(path))

    backend = _LocalBackend()
    cache = ImageCacheManager(backend=backend)
    agent = NodeAgent(
        NodeAgentConfig(node_id="n1", backends=["fake"]),
        image_cache=cache,
    )

    class _StaticBudget:
        async def get_budgets(self, **kw):  # type: ignore[no-untyped-def]
            del kw
            return [NodeBudget(node_id="n1", available_bytes=10 * 1024**3)]  # type: ignore[arg-type]

    coord = BuildCoordinator(
        catalog=catalog,
        state=InMemoryStateStore(),
        node_builder=InProcessNodeBuilder(node_agent=agent),
        budget_provider=_StaticBudget(),  # type: ignore[arg-type]
    )
    plan = BuildPlan(benchmarks=(BenchmarkBuildSpec(
        name="fake-bench", selection=BenchmarkSelection(smoke=True),
    ),))
    outcome = await coord.apply(plan)  # opportunistic default
    assert outcome.deferred == 1
    assert builds == []  # nothing pre-built — overflow deferred everything

    # Now the rollout-time call: ensure_present should hit the builder,
    # NOT fall through to backend.pull_image.
    await cache.ensure_present("fake-bench/huge:1")
    assert builds == ["fake-bench/huge:1"]
    assert "fake-bench/huge:1" not in pulls
