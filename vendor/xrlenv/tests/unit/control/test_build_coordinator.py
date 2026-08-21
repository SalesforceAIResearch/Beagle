"""build_coordinator end-to-end against an in-process node-builder
backed by a fake :class:`BenchmarkImageBuilder` (no real Docker).

P1.6.c will add a real-Docker integration alongside the existing
build-task-images.sh smoke; this file exercises the dispatch /
state-store / planner integration the coordinator coordinates.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest
from xrlenv.control.build_coordinator import BuildCoordinator
from xrlenv.control.build_plan import (
    BenchmarkBuildSpec,
    BenchmarkSelection,
    BuildEntry,
    BuildPlan,
    EntryPlacement,
    LocalSource,
)
from xrlenv.control.image_builder import BuildResult, ImageBuilderDecl
from xrlenv.control.image_planner import NodeBudget
from xrlenv.control.node_builder import BuildJob, InProcessNodeBuilder
from xrlenv.control.state import BuildAssignmentRecord, InMemoryStateStore
from xrlenv.control.template_catalog import TemplateCatalog

# ──────────────────────────────────────────────────────────────────────────────
# Fake BenchmarkImageBuilder — in-test, no Docker
# ──────────────────────────────────────────────────────────────────────────────


_FAKE_REGISTRY: dict[str, list[str]] = {}


class FakeBuilder:
    """Test double implementing the BenchmarkImageBuilder Protocol.

    Each instance records every ``build()`` call into the global
    ``_FAKE_REGISTRY`` keyed by the ``decl.module`` so tests can assert
    on what was built without needing a real Docker daemon.
    """

    IMAGE_SIZE_HINT_BYTES: ClassVar[int] = 1 * 1024**3
    SMOKE_REFS: ClassVar[tuple[str, ...]] = ("fake-bench/a:1", "fake-bench/b:1")

    def __init__(self, decl: ImageBuilderDecl) -> None:
        self._decl = decl
        _FAKE_REGISTRY.setdefault(decl.module, [])

    def enumerate_image_refs(self, *, selection: dict[str, Any]) -> list[str]:
        if selection.get("smoke"):
            return list(self.SMOKE_REFS)
        if "instances" in selection:
            return [f"fake-bench/{i}:1" for i in selection["instances"]]
        if selection.get("all"):
            return [*self.SMOKE_REFS, "fake-bench/c:1"]
        raise ValueError("invalid selection")

    async def build(
        self,
        *,
        image_ref: str,
        kwargs: dict[str, Any],
        force: bool,
    ) -> BuildResult:
        _FAKE_REGISTRY[self._decl.module].append(image_ref)
        if image_ref.endswith("fail:1"):
            return BuildResult(
                image_ref=image_ref, status="failed",
                error="oops, fake failure",
            )
        return BuildResult(image_ref=image_ref, status="done")


class FakeBuilderHuge:
    """Same Protocol, but every image is 200 GiB → triggers
    InsufficientCapacity in tests with small node budgets."""

    IMAGE_SIZE_HINT_BYTES: ClassVar[int] = 200 * 1024**3

    def __init__(self, decl: ImageBuilderDecl) -> None:
        self._decl = decl

    def enumerate_image_refs(self, *, selection: dict[str, Any]) -> list[str]:
        return ["fake-bench/huge:1"]

    async def build(self, **kw: Any) -> BuildResult:
        return BuildResult(image_ref="fake-bench/huge:1", status="done")


# ──────────────────────────────────────────────────────────────────────────────
# Test harness — register a fake manifest in a TemplateCatalog
# ──────────────────────────────────────────────────────────────────────────────


def _build_fake_catalog(monkeypatch, builder_cls: type) -> TemplateCatalog:
    """Build a catalog with one fake-bench manifest pointing at the
    given builder class. Uses monkeypatch to publish the class on a
    transient module path the loader can import."""
    import sys
    import types

    mod_name = f"_test_image_builders.{builder_cls.__name__}"
    mod = types.ModuleType(mod_name)
    setattr(mod, builder_cls.__name__, builder_cls)
    sys.modules[mod_name] = mod
    monkeypatch.setattr(
        "_test_image_builders.{cls}", mod, raising=False,
    ) if False else None

    from textwrap import dedent

    from xrlenv.control.template_catalog import load_manifest

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
          class: {builder_cls.__name__}
    """)
    catalog = TemplateCatalog()
    import tempfile
    from pathlib import Path

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        fh.write(manifest_yaml)
        path = Path(fh.name)
    catalog.register(load_manifest(path))
    return catalog


# ──────────────────────────────────────────────────────────────────────────────
# Coordinator end-to-end tests
# ──────────────────────────────────────────────────────────────────────────────


class _StaticBudgetProvider:
    def __init__(self, budgets: list[NodeBudget]) -> None:
        self._budgets = budgets

    async def get_budgets(
        self, *, reserved_runtime_gb: int, buffer_gb: int,
        cap_per_node_gb: int | None,
    ) -> list[NodeBudget]:
        return list(self._budgets)


@pytest.mark.asyncio
async def test_coordinator_apply_smoke_succeeds(monkeypatch) -> None:
    _FAKE_REGISTRY.clear()
    catalog = _build_fake_catalog(monkeypatch, FakeBuilder)
    state = InMemoryStateStore()
    coordinator = BuildCoordinator(
        catalog=catalog,
        state=state,
        node_builder=InProcessNodeBuilder(),
        budget_provider=_StaticBudgetProvider([
            NodeBudget(node_id="n1", available_bytes=100 * 1024**3),  # type: ignore[arg-type]
        ]),
    )
    plan = BuildPlan(
        replication=1,
        benchmarks=(BenchmarkBuildSpec(
            name="fake-bench",
            selection=BenchmarkSelection(smoke=True),
        ),),
    )
    outcome = await coordinator.apply(plan)
    assert outcome.status == "completed"
    assert outcome.successes == 2
    assert outcome.failures == 0
    # Both smoke images persisted as 'done'.
    rows = state.list_assignments(outcome.plan_id)
    assert all(r.status == "done" for r in rows)
    assert {r.image_ref for r in rows} == set(FakeBuilder.SMOKE_REFS)
    # Plan record carries the terminal status.
    plan_record = state.get_build_plan(outcome.plan_id)
    assert plan_record is not None
    assert plan_record.status == "completed"


@pytest.mark.asyncio
async def test_coordinator_dry_run_does_not_persist(monkeypatch) -> None:
    _FAKE_REGISTRY.clear()
    catalog = _build_fake_catalog(monkeypatch, FakeBuilder)
    state = InMemoryStateStore()
    coordinator = BuildCoordinator(
        catalog=catalog,
        state=state,
        node_builder=InProcessNodeBuilder(),
        budget_provider=_StaticBudgetProvider([
            NodeBudget(node_id="n1", available_bytes=100 * 1024**3),  # type: ignore[arg-type]
        ]),
    )
    plan = BuildPlan(benchmarks=(BenchmarkBuildSpec(
        name="fake-bench", selection=BenchmarkSelection(smoke=True),
    ),))
    outcome = await coordinator.apply(plan, dry_run=True)
    assert outcome.status == "dry_run"
    assert outcome.placement is not None
    assert len(outcome.placement.assignments) == 2
    # Nothing persisted.
    assert state.list_assignments(outcome.plan_id) == []
    assert state.get_build_plan(outcome.plan_id) is None
    # Builder was NOT actually invoked.
    assert _FAKE_REGISTRY == {} or all(not v for v in _FAKE_REGISTRY.values())


@pytest.mark.asyncio
async def test_coordinator_idempotent_reapply_returns_no_op(monkeypatch) -> None:
    _FAKE_REGISTRY.clear()
    catalog = _build_fake_catalog(monkeypatch, FakeBuilder)
    state = InMemoryStateStore()
    coordinator = BuildCoordinator(
        catalog=catalog,
        state=state,
        node_builder=InProcessNodeBuilder(),
        budget_provider=_StaticBudgetProvider([
            NodeBudget(node_id="n1", available_bytes=100 * 1024**3),  # type: ignore[arg-type]
        ]),
    )
    plan = BuildPlan(benchmarks=(BenchmarkBuildSpec(
        name="fake-bench", selection=BenchmarkSelection(smoke=True),
    ),))
    first = await coordinator.apply(plan)
    assert first.status == "completed"
    second = await coordinator.apply(plan)
    assert second.status == "no_op_already_completed"
    # First apply built 2; second built 0.
    built_count = sum(len(v) for v in _FAKE_REGISTRY.values())
    assert built_count == 2


@pytest.mark.asyncio
async def test_coordinator_partial_failure_marks_plan_partial(monkeypatch) -> None:
    """One image's builder returns status='failed' → plan transitions to partial_failure."""
    _FAKE_REGISTRY.clear()
    catalog = _build_fake_catalog(monkeypatch, FakeBuilder)
    state = InMemoryStateStore()
    coordinator = BuildCoordinator(
        catalog=catalog,
        state=state,
        node_builder=InProcessNodeBuilder(),
        budget_provider=_StaticBudgetProvider([
            NodeBudget(node_id="n1", available_bytes=100 * 1024**3),  # type: ignore[arg-type]
        ]),
    )
    plan = BuildPlan(benchmarks=(BenchmarkBuildSpec(
        name="fake-bench",
        # Force a 'fail:1' instance which the fake builder fails on.
        selection=BenchmarkSelection(instances=("a", "fail")),
    ),))
    outcome = await coordinator.apply(plan)
    assert outcome.status == "partial_failure"
    assert outcome.successes == 1
    assert outcome.failures == 1
    rows = state.list_assignments(outcome.plan_id)
    by_status = {r.image_ref: r.status for r in rows}
    assert by_status == {
        "fake-bench/a:1": "done",
        "fake-bench/fail:1": "failed",
    }


@pytest.mark.asyncio
async def test_coordinator_rejects_benchmark_without_image_builder(
    monkeypatch,
) -> None:
    """Plan that references a manifest lacking ``image_builder:`` →
    ManifestInvalid before any dispatch."""
    from textwrap import dedent

    from xrlenv.control.template_catalog import TemplateCatalog, load_manifest
    from xrlenv.errors import ManifestInvalid

    minimal = dedent(
        """\
        name: no-builder
        version: "0.1"
        image: scratch:latest
        env_adapter:
          module: xrlenv.envs.base
          class: NoOpEnvAdapter
        reward:
          mode: env_step
        """,
    )
    import tempfile
    from pathlib import Path

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        fh.write(minimal)
        path = Path(fh.name)
    catalog = TemplateCatalog()
    catalog.register(load_manifest(path))

    coordinator = BuildCoordinator(
        catalog=catalog,
        state=InMemoryStateStore(),
        node_builder=InProcessNodeBuilder(),
        budget_provider=_StaticBudgetProvider([]),
    )
    plan = BuildPlan(benchmarks=(BenchmarkBuildSpec(
        name="no-builder", selection=BenchmarkSelection(smoke=True),
    ),))
    with pytest.raises(ManifestInvalid, match="doesn't declare an image_builder"):
        await coordinator.apply(plan)


@pytest.mark.asyncio
async def test_coordinator_propagates_insufficient_capacity_in_eager_mode(
    monkeypatch,
) -> None:
    """In eager mode (P1.6.b/c semantics), the bin-packer raises
    InsufficientCapacity if any image doesn't fit. Default
    opportunistic mode (P1.6.g) handles overflow gracefully — see
    test_coordinator_opportunistic_defers_overflow."""
    catalog = _build_fake_catalog(monkeypatch, FakeBuilderHuge)
    coordinator = BuildCoordinator(
        catalog=catalog,
        state=InMemoryStateStore(),
        node_builder=InProcessNodeBuilder(),
        budget_provider=_StaticBudgetProvider([
            NodeBudget(node_id="n1", available_bytes=10 * 1024**3),  # type: ignore[arg-type]
        ]),
    )
    plan = BuildPlan(benchmarks=(BenchmarkBuildSpec(
        name="fake-bench", selection=BenchmarkSelection(smoke=True),
    ),))
    from xrlenv.control.image_planner import InsufficientCapacity

    with pytest.raises(InsufficientCapacity):
        await coordinator.apply(plan, eager=True)


@pytest.mark.asyncio
async def test_coordinator_opportunistic_defers_overflow(monkeypatch) -> None:
    """Default opportunistic mode (P1.6.g, F2=2): when an image
    can't fit the budget, it's recorded as ``status=registered``
    instead of raising. Pre-built rows still complete normally."""
    _FAKE_REGISTRY.clear()
    catalog = _build_fake_catalog(monkeypatch, FakeBuilderHuge)
    state = InMemoryStateStore()
    coordinator = BuildCoordinator(
        catalog=catalog,
        state=state,
        node_builder=InProcessNodeBuilder(),
        budget_provider=_StaticBudgetProvider([
            NodeBudget(node_id="n1", available_bytes=10 * 1024**3),  # type: ignore[arg-type]
        ]),
    )
    plan = BuildPlan(benchmarks=(BenchmarkBuildSpec(
        name="fake-bench", selection=BenchmarkSelection(smoke=True),
    ),))
    outcome = await coordinator.apply(plan)  # eager=False default
    assert outcome.status == "completed"
    assert outcome.deferred == 1  # the huge image overflowed budget
    rows = state.list_assignments(outcome.plan_id)
    assert len(rows) == 1
    assert rows[0].status == "registered"
    assert rows[0].node_id == "n1"  # preferred-home recorded


@pytest.mark.asyncio
async def test_coordinator_opportunistic_registers_lazy_builders_for_deferred(
    monkeypatch,
) -> None:
    """Audit P1.6.g-H1 fix: when opportunistic apply defers an image
    (cluster ran out of budget), the BuildJob dispatched to the
    deferred row's preferred-home node MUST carry the ref in
    ``lazy_registrations`` so the node can populate its lazy-builder
    map. Otherwise a later ``ensure_present`` falls through to
    ``backend.pull_image`` for a benchmark-internal tag and fails."""
    _FAKE_REGISTRY.clear()
    catalog = _build_fake_catalog(monkeypatch, FakeBuilderHuge)
    state = InMemoryStateStore()

    # Capture every BuildJob that reaches the node-builder so we can
    # assert lazy_registrations carries the deferred ref.
    captured_jobs: list[BuildJob] = []

    class _CapturingNodeBuilder:
        async def execute(self, job):  # type: ignore[no-untyped-def]
            captured_jobs.append(job)
            return
            yield  # pragma: no cover — make this an async generator

    coordinator = BuildCoordinator(
        catalog=catalog,
        state=state,
        node_builder=_CapturingNodeBuilder(),  # type: ignore[arg-type]
        budget_provider=_StaticBudgetProvider([
            NodeBudget(node_id="n1", available_bytes=10 * 1024**3),  # type: ignore[arg-type]
        ]),
    )
    plan = BuildPlan(benchmarks=(BenchmarkBuildSpec(
        name="fake-bench", selection=BenchmarkSelection(smoke=True),
    ),))
    outcome = await coordinator.apply(plan)  # opportunistic default
    assert outcome.deferred == 1
    # Even though the placement was empty (huge image overflowed
    # budget), n1 still received a BuildJob — carrying the deferred
    # ref in lazy_registrations.
    assert len(captured_jobs) == 1
    job = captured_jobs[0]
    assert job.node_id == "n1"
    assert tuple(a.image_ref for a in job.assignments) == ()
    lazy_refs = tuple(a.image_ref for a in job.lazy_registrations)
    assert lazy_refs == ("fake-bench/huge:1",)
    # Builder mapping has to be present so the node-side handler can
    # populate the lazy_builders dict.
    assert "fake-bench" in job.builder_per_benchmark


@pytest.mark.asyncio
async def test_coordinator_broadcasts_lazy_registrations_to_all_budget_nodes(
    monkeypatch,
) -> None:
    """Audit P1.6.g-H2 fix: deferred refs land in EVERY backend-capable
    node's BuildJob.lazy_registrations, not just the row's preferred_home.

    Otherwise the scheduler — which doesn't read the build snapshot —
    can route a rollout to a non-preferred node, and that node has no
    builder mapping, so ensure_present falls through to
    ``backend.pull_image``. Broadcasting trades ~200 B per ref of memory
    on each node for correctness regardless of placement. Locality is
    recovered separately by the scheduler-side preferred_home routing.
    """
    _FAKE_REGISTRY.clear()
    catalog = _build_fake_catalog(monkeypatch, FakeBuilderHuge)
    state = InMemoryStateStore()
    captured_jobs: list[BuildJob] = []

    class _CapturingNodeBuilder:
        async def execute(self, job):  # type: ignore[no-untyped-def]
            captured_jobs.append(job)
            return
            yield  # pragma: no cover

    coordinator = BuildCoordinator(
        catalog=catalog,
        state=state,
        node_builder=_CapturingNodeBuilder(),  # type: ignore[arg-type]
        budget_provider=_StaticBudgetProvider([
            NodeBudget(node_id="n1", available_bytes=10 * 1024**3),  # type: ignore[arg-type]
            NodeBudget(node_id="n2", available_bytes=10 * 1024**3),  # type: ignore[arg-type]
        ]),
    )
    plan = BuildPlan(benchmarks=(BenchmarkBuildSpec(
        name="fake-bench", selection=BenchmarkSelection(smoke=True),
    ),))
    outcome = await coordinator.apply(plan)
    assert outcome.deferred == 1

    # Both nodes received a job, both carry the deferred ref in
    # lazy_registrations even though only one is the planner's
    # preferred_home for that row.
    assert len(captured_jobs) == 2
    by_node = {j.node_id: j for j in captured_jobs}
    assert set(by_node.keys()) == {"n1", "n2"}
    for nid, job in by_node.items():
        lazy_refs = tuple(a.image_ref for a in job.lazy_registrations)
        assert lazy_refs == ("fake-bench/huge:1",), (
            f"node {nid} missing the broadcast: {lazy_refs}"
        )


@pytest.mark.asyncio
async def test_coordinator_force_rebuilds_completed_plan(monkeypatch) -> None:
    """Audit M1 fix: --force on a completed plan re-dispatches the
    builds rather than returning no_op_already_completed."""
    _FAKE_REGISTRY.clear()
    catalog = _build_fake_catalog(monkeypatch, FakeBuilder)
    state = InMemoryStateStore()
    coordinator = BuildCoordinator(
        catalog=catalog,
        state=state,
        node_builder=InProcessNodeBuilder(),
        budget_provider=_StaticBudgetProvider([
            NodeBudget(node_id="n1", available_bytes=100 * 1024**3),  # type: ignore[arg-type]
        ]),
    )
    plan = BuildPlan(benchmarks=(BenchmarkBuildSpec(
        name="fake-bench", selection=BenchmarkSelection(smoke=True),
    ),))
    first = await coordinator.apply(plan)
    assert first.status == "completed"
    built_after_first = sum(len(v) for v in _FAKE_REGISTRY.values())
    assert built_after_first == 2

    # Without force: no-op (preserves the existing M1 idempotency).
    second = await coordinator.apply(plan, force=False)
    assert second.status == "no_op_already_completed"
    assert sum(len(v) for v in _FAKE_REGISTRY.values()) == built_after_first

    # With force: bypasses the no-op; both images dispatched again.
    third = await coordinator.apply(plan, force=True)
    assert third.status == "completed"
    assert sum(len(v) for v in _FAKE_REGISTRY.values()) == built_after_first + 2


@pytest.mark.asyncio
async def test_coordinator_partial_failure_residual_only_retry(monkeypatch) -> None:
    """Audit M2 fix: re-applying a partial_failure plan dispatches
    only the failed rows and preserves done timestamps."""
    _FAKE_REGISTRY.clear()
    catalog = _build_fake_catalog(monkeypatch, FakeBuilder)
    state = InMemoryStateStore()
    coordinator = BuildCoordinator(
        catalog=catalog,
        state=state,
        node_builder=InProcessNodeBuilder(),
        budget_provider=_StaticBudgetProvider([
            NodeBudget(node_id="n1", available_bytes=100 * 1024**3),  # type: ignore[arg-type]
        ]),
    )
    plan = BuildPlan(benchmarks=(BenchmarkBuildSpec(
        name="fake-bench",
        selection=BenchmarkSelection(instances=("a", "fail")),
    ),))
    first = await coordinator.apply(plan)
    assert first.status == "partial_failure"
    rows_first = {r.image_ref: r for r in state.list_assignments(first.plan_id)}
    done_completed_at = rows_first["fake-bench/a:1"].completed_at
    assert done_completed_at is not None
    builds_after_first = sum(len(v) for v in _FAKE_REGISTRY.values())
    assert builds_after_first == 2

    # Re-apply (force=False) → only the failed row should re-dispatch;
    # the done row's timestamp must be preserved.
    second = await coordinator.apply(plan)
    assert second.status == "partial_failure"  # 'fail:1' still fails
    rows_second = {r.image_ref: r for r in state.list_assignments(second.plan_id)}
    assert rows_second["fake-bench/a:1"].status == "done"
    assert rows_second["fake-bench/a:1"].completed_at == done_completed_at
    # The fail row WAS dispatched (counter went up by exactly 1).
    builds_after_second = sum(len(v) for v in _FAKE_REGISTRY.values())
    assert builds_after_second == builds_after_first + 1


@pytest.mark.asyncio
async def test_coordinator_residual_retry_handles_replanned_node(monkeypatch) -> None:
    """Audit P1.6-M2a fix: when a failed image replans onto a
    *different* node between applies (e.g. because node disk filled
    up), the residual retry must materialize the new (node_id,
    image_ref) row before the per-node task tries to mark it
    ``building`` — otherwise the dispatch raises KeyError and the
    plan stays stuck in_flight.

    Setup: two nodes, both with budget for one 1 GiB image. First
    apply lands fail:1 on n1 (where it fails); a:1 on n2 (succeeds).
    Then we drain n1's budget so retry-replanning forces fail:1 onto
    n2 — that's the new (node_id, image_ref) pair the residual_only
    branch must record.
    """
    _FAKE_REGISTRY.clear()
    catalog = _build_fake_catalog(monkeypatch, FakeBuilder)
    state = InMemoryStateStore()

    class _MutableBudgetProvider:
        def __init__(self) -> None:
            # Both nodes have room initially.
            self._budgets = [
                NodeBudget(node_id="n1", available_bytes=2 * 1024**3),  # type: ignore[arg-type]
                NodeBudget(node_id="n2", available_bytes=2 * 1024**3),  # type: ignore[arg-type]
            ]

        async def get_budgets(
            self, *, reserved_runtime_gb: int, buffer_gb: int,
            cap_per_node_gb: int | None,
        ) -> list[NodeBudget]:
            return list(self._budgets)

        def evacuate_node(self, node_id: str) -> None:
            """Simulate a node filling up between applies."""
            self._budgets = [
                NodeBudget(
                    node_id=b.node_id,
                    available_bytes=0 if b.node_id == node_id else b.available_bytes,
                )
                for b in self._budgets
            ]

    budgets = _MutableBudgetProvider()
    coordinator = BuildCoordinator(
        catalog=catalog,
        state=state,
        node_builder=InProcessNodeBuilder(),
        budget_provider=budgets,
    )

    # First apply: a + fail across two nodes (FFD spreads them).
    plan = BuildPlan(benchmarks=(BenchmarkBuildSpec(
        name="fake-bench",
        selection=BenchmarkSelection(instances=("a", "fail")),
    ),))
    first = await coordinator.apply(plan)
    assert first.status == "partial_failure"
    rows_first = state.list_assignments(first.plan_id)
    failed_first = [r for r in rows_first if r.status == "failed"]
    assert len(failed_first) == 1
    failed_node = failed_first[0].node_id
    other_node = "n2" if failed_node == "n1" else "n1"

    # Drain the failed node so retry-replanning has to move fail:1
    # to the other node.
    budgets.evacuate_node(failed_node)

    second = await coordinator.apply(plan)
    # Without the M2a fix, this raises KeyError mid-dispatch and the
    # plan stays in_flight. With the fix, the new (other_node,
    # fail:1) row gets recorded before ``building`` is set, the
    # build runs (and fails again, since the FakeBuilder is
    # deterministic on ``fail:1``), and the plan transitions to
    # partial_failure.
    assert second.status == "partial_failure"
    rows_second = {
        (r.node_id, r.image_ref): r
        for r in state.list_assignments(second.plan_id)
    }
    # The new replanned row exists at the other node.
    assert (other_node, "fake-bench/fail:1") in rows_second
    # Original done row preserved.
    a_row = next(
        r for k, r in rows_second.items()
        if r.image_ref == "fake-bench/a:1" and r.status == "done"
    )
    assert a_row.completed_at is not None


@pytest.mark.asyncio
async def test_coordinator_dispatches_per_node_in_parallel(monkeypatch) -> None:
    """Audit M3 fix: per-node dispatch overlaps. Two slow node jobs
    that each take ~50 ms should finish in <90 ms wall-clock if
    they actually run in parallel; serial execution would take ~100 ms."""
    import asyncio
    import time

    from xrlenv.control.image_planner import NodeBudget

    # Hand-roll a catalog: two benchmarks (well, one benchmark with
    # two nodes) — easier to drive the budget provider to two nodes.
    _FAKE_REGISTRY.clear()
    catalog = _build_fake_catalog(monkeypatch, FakeBuilder)
    state = InMemoryStateStore()

    class _SlowNodeBuilder:
        """Each node task sleeps 50ms before yielding 'done'."""

        async def execute(self, job: BuildJob):
            await asyncio.sleep(0.05)
            for a in job.assignments:
                yield BuildResult(image_ref=a.image_ref, status="done")

    coordinator = BuildCoordinator(
        catalog=catalog,
        state=state,
        node_builder=_SlowNodeBuilder(),
        budget_provider=_StaticBudgetProvider([
            NodeBudget(node_id="n1", available_bytes=100 * 1024**3),  # type: ignore[arg-type]
            NodeBudget(node_id="n2", available_bytes=100 * 1024**3),  # type: ignore[arg-type]
        ]),
    )
    plan = BuildPlan(benchmarks=(BenchmarkBuildSpec(
        # Use ``all`` to land 3 images so FFD spreads them across both
        # nodes; that gives us two distinct per-node jobs.
        name="fake-bench", selection=BenchmarkSelection(all=True),
    ),))
    started = time.monotonic()
    outcome = await coordinator.apply(plan)
    elapsed = time.monotonic() - started

    assert outcome.status == "completed"
    # Two node jobs x 50ms each. Serial = 100+ms; parallel = 50-90ms.
    # Use a comfortable threshold.
    assert elapsed < 0.09, (
        f"per-node dispatch appears serial: elapsed={elapsed:.3f}s"
    )


@pytest.mark.asyncio
async def test_grpc_node_builder_dispatches_via_transport_build_images() -> None:
    """GrpcNodeBuilder calls transport.build_images(...) and yields the
    returned BuildResult rows."""
    from xrlenv.control.image_planner import PlanAssignment
    from xrlenv.control.node_builder import BuilderRef, GrpcNodeBuilder

    captured: dict[str, Any] = {}

    class _FakeTransport:
        async def build_images(
            self, *, assignments, builder_per_benchmark,
            kwargs_per_benchmark, force, lazy_registrations=None,
        ):
            captured["assignments"] = assignments
            captured["force"] = force
            captured["lazy_registrations"] = lazy_registrations or []
            return [
                BuildResult(image_ref=a.image_ref, status="done")
                for a in assignments
            ]

    def lookup(node_id: str):
        return _FakeTransport() if node_id == "n-good" else None

    job = BuildJob(
        node_id="n-good",
        assignments=(
            PlanAssignment(
                image_ref="x:1", node_id="n-good",  # type: ignore[arg-type]
                benchmark="b", size_bytes=1,
            ),
        ),
        builder_per_benchmark={
            "b": BuilderRef(module="m", class_name="C"),
        },
        force=True,
    )
    builder = GrpcNodeBuilder(node_lookup=lookup)
    results = []
    async for r in builder.execute(job):
        results.append(r)
    assert len(results) == 1
    assert results[0].status == "done"
    assert captured["force"] is True


@pytest.mark.asyncio
async def test_grpc_node_builder_marks_unknown_node_failed() -> None:
    """When the lookup returns None (no live transport), every
    assignment yields a 'failed' BuildResult so the coordinator can
    update the per-row status without crashing."""
    from xrlenv.control.image_planner import PlanAssignment
    from xrlenv.control.node_builder import BuilderRef, GrpcNodeBuilder

    job = BuildJob(
        node_id="n-missing",
        assignments=(
            PlanAssignment(
                image_ref="x:1", node_id="n-missing",  # type: ignore[arg-type]
                benchmark="b", size_bytes=1,
            ),
            PlanAssignment(
                image_ref="x:2", node_id="n-missing",  # type: ignore[arg-type]
                benchmark="b", size_bytes=1,
            ),
        ),
        builder_per_benchmark={
            "b": BuilderRef(module="m", class_name="C"),
        },
    )
    builder = GrpcNodeBuilder(node_lookup=lambda _: None)
    results = []
    async for r in builder.execute(job):
        results.append(r)
    assert len(results) == 2
    assert all(r.status == "failed" for r in results)
    assert all("no live transport" in (r.error or "") for r in results)


@pytest.mark.asyncio
async def test_inprocess_node_builder_streams_results() -> None:
    """The InProcessNodeBuilder yields one BuildResult per assignment."""
    import sys
    import types

    mod_name = "_test_image_builders.streaming_fake"
    mod = types.ModuleType(mod_name)
    mod.FakeBuilder = FakeBuilder  # type: ignore[attr-defined]
    sys.modules[mod_name] = mod

    from xrlenv.control.image_planner import PlanAssignment
    from xrlenv.control.node_builder import BuilderRef

    job = BuildJob(
        node_id="n1",
        assignments=(
            PlanAssignment(
                image_ref="fake-bench/a:1", node_id="n1",  # type: ignore[arg-type]
                benchmark="fake-bench", size_bytes=1024,
            ),
            PlanAssignment(
                image_ref="fake-bench/b:1", node_id="n1",  # type: ignore[arg-type]
                benchmark="fake-bench", size_bytes=1024,
            ),
        ),
        builder_per_benchmark={
            "fake-bench": BuilderRef(module=mod_name, class_name="FakeBuilder"),
        },
        force=False,
    )
    builder = InProcessNodeBuilder()
    results = []
    async for r in builder.execute(job):
        results.append(r)
    assert len(results) == 2
    assert all(r.status == "done" for r in results)


# ──────────────────────────────────────────────────────────────────────────────
# Per-image-ref dispatch
# ──────────────────────────────────────────────────────────────────────────────


def _entries_plan(
    *,
    image_refs: list[str] | None = None,
    size_hint_bytes: int = 1 * 1024**3,
    size_hint_source: str = "registry-probe",
    preferred_home_count: int = 1,
    pinned: bool = False,
) -> BuildPlan:
    from xrlenv.control.build_plan import (
        BuildEntry,
        EntryPlacement,
        RegistrySource,
    )

    refs = image_refs if image_refs is not None else ["alex/a:1", "alex/b:1"]
    return BuildPlan(entries=tuple(
        BuildEntry(
            image_ref=ref,
            context_source=RegistrySource(),
            placement=EntryPlacement(
                preferred_home_count=preferred_home_count,
                size_hint_bytes=size_hint_bytes,
                size_hint_source=size_hint_source,  # type: ignore[arg-type]
            ),
            pinned=pinned,
        )
        for ref in refs
    ))


def _make_coordinator(
    *,
    ensure_present_fn: Any,
    build_image_fn: Any = None,
    nodes: list[NodeBudget] | None = None,
    inventory: dict[str, set[str]] | None = None,
) -> tuple[BuildCoordinator, InMemoryStateStore]:
    state = InMemoryStateStore()

    class _StaticInventoryProvider:
        def __init__(self, snapshot: dict[str, set[str]]) -> None:
            self._snapshot = {k: set(v) for k, v in snapshot.items()}

        async def get_inventory(self) -> dict[str, set[str]]:
            return {k: set(v) for k, v in self._snapshot.items()}

    coordinator = BuildCoordinator(
        catalog=TemplateCatalog(),
        state=state,
        node_builder=InProcessNodeBuilder(),
        budget_provider=_StaticBudgetProvider(
            nodes or [NodeBudget(  # type: ignore[arg-type]
                node_id="n1", available_bytes=100 * 1024**3,
            )],
        ),
        ensure_present_fn=ensure_present_fn,
        build_image_fn=build_image_fn,
        inventory_provider=(
            _StaticInventoryProvider(inventory)
            if inventory is not None else None
        ),
    )
    return coordinator, state


@pytest.mark.asyncio
async def test_per_image_ref_apply_dispatches_via_ensure_present_fn() -> None:
    calls: list[tuple[str, str]] = []

    async def fake_ensure(
        node_id: str, image_ref: str, timeout_s: float,
    ) -> tuple[str, str | None]:
        calls.append((node_id, image_ref))
        return ("ok", None)

    coordinator, state = _make_coordinator(ensure_present_fn=fake_ensure)
    plan = _entries_plan(image_refs=["a:1", "b:1", "c:1"])
    outcome = await coordinator.apply(plan)

    assert outcome.status == "completed"
    assert outcome.successes == 3
    assert outcome.failures == 0
    assert {ref for _, ref in calls} == {"a:1", "b:1", "c:1"}
    rows = state.list_assignments(outcome.plan_id)
    assert all(r.status == "done" for r in rows)
    # All rows tagged with the synthetic per-image-ref benchmark.
    from xrlenv.control.build_coordinator import PER_IMAGE_REF_BENCHMARK_TAG
    assert all(r.benchmark == PER_IMAGE_REF_BENCHMARK_TAG for r in rows)


@pytest.mark.asyncio
async def test_per_image_ref_apply_marks_per_image_failures() -> None:
    async def flaky_ensure(
        node_id: str, image_ref: str, timeout_s: float,
    ) -> tuple[str, str | None]:
        if image_ref == "broken:1":
            return ("failed", "image not found")
        return ("ok", None)

    coordinator, state = _make_coordinator(ensure_present_fn=flaky_ensure)
    plan = _entries_plan(image_refs=["good:1", "broken:1"])
    outcome = await coordinator.apply(plan)

    assert outcome.status == "partial_failure"
    assert outcome.successes == 1
    assert outcome.failures == 1
    error_lines = list(outcome.error_summary)
    assert any("broken:1" in line and "image not found" in line for line in error_lines)
    rows = {r.image_ref: r for r in state.list_assignments(outcome.plan_id)}
    assert rows["good:1"].status == "done"
    assert rows["broken:1"].status == "failed"
    assert rows["broken:1"].error == "image not found"


@pytest.mark.asyncio
async def test_per_image_ref_apply_dry_run_does_not_persist() -> None:
    async def fake_ensure(*_args: Any) -> tuple[str, str | None]:
        return ("ok", None)

    coordinator, state = _make_coordinator(ensure_present_fn=fake_ensure)
    plan = _entries_plan()
    outcome = await coordinator.apply(plan, dry_run=True)

    assert outcome.status == "dry_run"
    assert outcome.placement is not None
    assert state.get_build_plan(outcome.plan_id) is None


@pytest.mark.asyncio
async def test_per_image_ref_apply_idempotent_on_completed_plan() -> None:
    call_count = 0

    async def counting_ensure(
        node_id: str, image_ref: str, timeout_s: float,
    ) -> tuple[str, str | None]:
        nonlocal call_count
        call_count += 1
        return ("ok", None)

    coordinator, _state = _make_coordinator(ensure_present_fn=counting_ensure)
    plan = _entries_plan(image_refs=["a:1", "b:1"])
    first = await coordinator.apply(plan)
    assert first.status == "completed"
    first_call_count = call_count

    second = await coordinator.apply(plan)
    assert second.status == "no_op_already_completed"
    # No additional dispatches on the second apply.
    assert call_count == first_call_count


@pytest.mark.asyncio
async def test_per_image_ref_rejects_git_when_no_build_image_fn() -> None:
    """A coordinator without a wired ``build_image_fn`` rejects
    git-source plans up front with a clear error rather than
    silently dispatching the build through the wrong path."""
    from xrlenv.control.build_plan import (
        BuildEntry,
        EntryPlacement,
        GitSource,
    )
    from xrlenv.errors import ManifestInvalid

    async def fake_ensure(*_args: Any) -> tuple[str, str | None]:
        return ("ok", None)

    # Note: only ensure_present_fn wired — no build_image_fn.
    coordinator, _ = _make_coordinator(ensure_present_fn=fake_ensure)
    plan = BuildPlan(entries=(
        BuildEntry(
            image_ref="xrlenv-seta-env/0:main",
            context_source=GitSource(
                repo="https://github.com/example/repo",
                ref="main", subdir="path", dockerfile="Dockerfile",
            ),
            placement=EntryPlacement(size_hint_bytes=1024),
        ),
    ))
    with pytest.raises(ManifestInvalid) as excinfo:
        await coordinator.apply(plan)
    msg = str(excinfo.value)
    assert "build_image_fn" in msg
    assert "git" in msg.lower()


@pytest.mark.asyncio
async def test_per_image_ref_rejects_unresolved_tarball() -> None:
    """Tarball entries that reach the coordinator without their
    bytes resolved (no ``content_b64``) are rejected with a clear
    error pointing at the CLI's ``resolve_tarball_sources`` helper.

    Sub-slice 1.b: the operator's CLI populates ``content_b64`` from
    operator-local disk before apply. A direct programmatic caller
    that skipped that step shouldn't silently succeed with a wire
    payload missing its bytes."""
    from xrlenv.control.build_plan import (
        BuildEntry,
        EntryPlacement,
        TarballSource,
    )
    from xrlenv.errors import ManifestInvalid

    async def fake_ensure(*_args: Any) -> tuple[str, str | None]:
        return ("ok", None)

    coordinator, _ = _make_coordinator(ensure_present_fn=fake_ensure)
    plan = BuildPlan(entries=(
        BuildEntry(
            image_ref="my-org/private-task:v3",
            context_source=TarballSource(
                path="./contexts/private-task.tar.gz",
                dockerfile="Dockerfile",
                # NOTE: no content_b64 — simulates a caller that
                # built a plan in-memory and skipped the CLI helper.
            ),
            placement=EntryPlacement(size_hint_bytes=1024),
        ),
    ))
    with pytest.raises(ManifestInvalid) as excinfo:
        await coordinator.apply(plan)
    msg = str(excinfo.value)
    assert "resolve_tarball_sources" in msg
    assert "my-org/private-task:v3" in msg


@pytest.mark.asyncio
async def test_per_image_ref_pin_budget_rejects_when_pinned_exceeds_node_budget() -> None:
    """Sub-slice 3 (F4): hard reject at apply time when the sum of
    pinned entries' size_hint_bytes exceeds any node's available
    bytes. Conservative bound — if the pinned total > one node's
    budget, FFD might land them all there and leave no room for
    non-pinned rollouts.
    """
    from xrlenv.control.build_plan import (
        BuildEntry,
        EntryPlacement,
        RegistrySource,
    )
    from xrlenv.errors import ManifestInvalid

    async def fake_ensure(*_args: Any) -> tuple[str, str | None]:
        return ("ok", None)

    # Single small node; two big pinned entries that together
    # overflow it.
    coordinator, _ = _make_coordinator(
        ensure_present_fn=fake_ensure,
        nodes=[NodeBudget(  # type: ignore[arg-type]
            node_id="small-node", available_bytes=10 * 1024**3,
        )],
    )
    plan = BuildPlan(entries=(
        BuildEntry(
            image_ref="big-pin/a:1",
            context_source=RegistrySource(),
            placement=EntryPlacement(size_hint_bytes=8 * 1024**3),
            pinned=True,
        ),
        BuildEntry(
            image_ref="big-pin/b:1",
            context_source=RegistrySource(),
            placement=EntryPlacement(size_hint_bytes=8 * 1024**3),
            pinned=True,
        ),
    ))
    with pytest.raises(ManifestInvalid) as excinfo:
        await coordinator.apply(plan)
    msg = str(excinfo.value)
    assert "pin-budget over-commit" in msg
    assert "small-node" in msg
    # Names the over-by amount so the operator can act.
    assert "over by" in msg


@pytest.mark.asyncio
async def test_per_image_ref_pin_budget_passes_when_pinned_fits() -> None:
    """A plan whose pinned entries fit each node's budget proceeds
    normally — pin-budget guard doesn't false-positive when the
    operator is within their declared cap."""
    from xrlenv.control.build_plan import (
        BuildEntry,
        EntryPlacement,
        RegistrySource,
    )

    async def fake_ensure(*_args: Any) -> tuple[str, str | None]:
        return ("ok", None)

    coordinator, _ = _make_coordinator(
        ensure_present_fn=fake_ensure,
        nodes=[NodeBudget(  # type: ignore[arg-type]
            node_id="big-node", available_bytes=100 * 1024**3,
        )],
    )
    plan = BuildPlan(entries=(
        BuildEntry(
            image_ref="small-pin/a:1",
            context_source=RegistrySource(),
            placement=EntryPlacement(size_hint_bytes=2 * 1024**3),
            pinned=True,
        ),
        BuildEntry(
            image_ref="small-pin/b:1",
            context_source=RegistrySource(),
            placement=EntryPlacement(size_hint_bytes=2 * 1024**3),
            pinned=True,
        ),
    ))
    outcome = await coordinator.apply(plan)
    assert outcome.status == "completed"
    assert outcome.successes == 2


@pytest.mark.asyncio
async def test_per_image_ref_pin_budget_ignores_unpinned_overflow() -> None:
    """Pin-budget is about pinned entries only. Non-pinned entries
    that overflow a node's budget are FFD's problem to surface
    (different error path: ``InsufficientCapacity``)."""
    from xrlenv.control.build_plan import (
        BuildEntry,
        EntryPlacement,
        RegistrySource,
    )

    async def fake_ensure(*_args: Any) -> tuple[str, str | None]:
        return ("ok", None)

    coordinator, _ = _make_coordinator(
        ensure_present_fn=fake_ensure,
        nodes=[NodeBudget(  # type: ignore[arg-type]
            node_id="small-node", available_bytes=10 * 1024**3,
        )],
    )
    plan = BuildPlan(entries=(
        BuildEntry(
            image_ref="big-unpinned/a:1",
            context_source=RegistrySource(),
            # cluster-reported = already on-disk (x1), so this test exercises
            # the deferral mechanism on real sizes, decoupled from the
            # registry-probe→on-disk inflation (tested separately).
            placement=EntryPlacement(
                size_hint_bytes=8 * 1024**3, size_hint_source="cluster-reported",
            ),
            pinned=False,
        ),
        BuildEntry(
            image_ref="big-unpinned/b:1",
            context_source=RegistrySource(),
            placement=EntryPlacement(
                size_hint_bytes=8 * 1024**3, size_hint_source="cluster-reported",
            ),
            pinned=False,
        ),
    ))
    # Pin-budget guard does NOT fire (no entries are pinned).
    # FFD's own InsufficientCapacity does, on a different code path —
    # but only under ``eager=True``. Default opportunistic mode
    # defers the overflow rows to lazy-pull on first acquire.
    from xrlenv.control.image_planner import InsufficientCapacity
    with pytest.raises(InsufficientCapacity):
        await coordinator.apply(plan, eager=True)

    # Default opportunistic mode: one image fits (placed), the other
    # is deferred as ``registered`` against its preferred-home node.
    # The runtime's acquire-time ``ensure_present`` pulls the deferred
    # one on first rollout (LRU evicts the other if needed).
    outcome = await coordinator.apply(plan)
    assert outcome.status == "completed"
    assert outcome.deferred == 1


@pytest.mark.asyncio
async def test_per_image_ref_opportunistic_defers_overflow_registry() -> None:
    """Per-image-ref opportunistic mode (default) places what fits +
    records the rest as ``status="registered"`` so the runtime
    acquire-time ``ensure_present`` pulls on demand. Registry-source
    entries need no extra wiring — the node pulls from Docker Hub
    when a rollout lands on a node that doesn't have the image cached.
    """
    from xrlenv.control.build_plan import (
        BuildEntry,
        EntryPlacement,
        RegistrySource,
    )

    pulls: list[tuple[str, str]] = []

    async def fake_ensure(
        node_id: str, image_ref: str, _t: float,
    ) -> tuple[str, str | None]:
        pulls.append((node_id, image_ref))
        return ("ok", None)

    # One node with room for ~1.5 of the 1 GiB images.
    coordinator, store = _make_coordinator(
        ensure_present_fn=fake_ensure,
        nodes=[NodeBudget(  # type: ignore[arg-type]
            node_id="n1", available_bytes=1_500_000_000,
        )],
    )
    # cluster-reported = on-disk (x1) so the 1-fits-2-defer math is about the
    # deferral mechanism, not the registry-probe→on-disk inflation.
    plan = BuildPlan(entries=(
        BuildEntry(
            image_ref="ns/a:1",
            context_source=RegistrySource(),
            placement=EntryPlacement(
                size_hint_bytes=1_000_000_000, size_hint_source="cluster-reported",
            ),
        ),
        BuildEntry(
            image_ref="ns/b:1",
            context_source=RegistrySource(),
            placement=EntryPlacement(
                size_hint_bytes=1_000_000_000, size_hint_source="cluster-reported",
            ),
        ),
        BuildEntry(
            image_ref="ns/c:1",
            context_source=RegistrySource(),
            placement=EntryPlacement(
                size_hint_bytes=1_000_000_000, size_hint_source="cluster-reported",
            ),
        ),
    ))
    outcome = await coordinator.apply(plan)  # opportunistic default

    assert outcome.status == "completed"
    # 1 placed (synchronously pulled) + 2 deferred (lazy on acquire).
    assert outcome.deferred == 2
    assert len(pulls) == 1

    # All three rows persisted; placed → done, deferred → registered.
    rows = sorted(
        store.list_assignments(outcome.plan_id),
        key=lambda r: r.image_ref,
    )
    assert len(rows) == 3
    placed_refs = {r.image_ref for r in rows if r.status == "done"}
    registered_refs = {r.image_ref for r in rows if r.status == "registered"}
    assert len(placed_refs) == 1
    assert len(registered_refs) == 2
    assert placed_refs.union(registered_refs) == {
        "ns/a:1", "ns/b:1", "ns/c:1",
    }
    # Every registered row points at the only available node as its
    # preferred home — image-affinity scheduling still has a sensible
    # default target even though the image isn't cached anywhere yet.
    for r in rows:
        if r.status == "registered":
            assert r.node_id == "n1"
    # Sanity: terminal plan is not partial_failure (deferred ≠ failed).
    plan_record = store.get_build_plan(outcome.plan_id)
    assert plan_record is not None
    assert plan_record.status == "completed"


@pytest.mark.asyncio
async def test_per_image_ref_opportunistic_rejects_deferred_git_source() -> None:
    """Lazy-build for git/tarball sources needs per-node source-spec
    broadcast that the per-image-ref dispatcher doesn't carry yet.
    Until that's wired, deferring a git entry would silently fail
    at runtime (the node has no spec to rebuild from), so reject
    upfront with a clear remediation path."""
    from xrlenv.control.build_plan import (
        BuildEntry,
        EntryPlacement,
        GitSource,
        RegistrySource,
    )

    coordinator, _ = _make_coordinator(
        ensure_present_fn=lambda *_a: _ok(),  # type: ignore[arg-type, return-value]
        build_image_fn=lambda *_a: _ok(),     # type: ignore[arg-type, return-value]
        nodes=[NodeBudget(  # type: ignore[arg-type]
            node_id="n1", available_bytes=1_500_000_000,
        )],
    )
    plan = BuildPlan(entries=(
        BuildEntry(
            image_ref="ns/registry:1",
            context_source=RegistrySource(),
            placement=EntryPlacement(size_hint_bytes=1_000_000_000),
        ),
        BuildEntry(
            image_ref="ns/git-too-big:1",
            context_source=GitSource(
                repo="https://example.com/r.git", ref="main",
            ),
            placement=EntryPlacement(size_hint_bytes=1_000_000_000),
        ),
    ))
    from xrlenv.errors import ManifestInvalid

    with pytest.raises(ManifestInvalid) as exc:
        await coordinator.apply(plan)
    # Message must name the offending ref + the remediation paths.
    assert "ns/git-too-big:1" in str(exc.value)
    assert "--eager" in str(exc.value)

    # ``eager=True`` surfaces the underlying FFD reject instead.
    from xrlenv.control.image_planner import InsufficientCapacity
    with pytest.raises(InsufficientCapacity):
        await coordinator.apply(plan, eager=True)


def _ok() -> Any:
    """Async no-op returning ``("ok", None)`` — used to satisfy the
    dispatcher Protocol where the body doesn't matter for the test."""
    async def _wrap() -> tuple[str, str | None]:
        return ("ok", None)
    return _wrap()


@pytest.mark.asyncio
async def test_apply_threads_skip_if_present_to_build_image_fn() -> None:
    """``apply(skip_if_present=True)`` passes the flag through to
    ``build_image_fn`` for source-build entries. Default
    ``skip_if_present=False`` preserves prior behavior (no flag set
    on the fn call). ``force=True`` overrides ``skip_if_present``
    (forced rebuilds always dispatch — operator-explicit semantics)."""
    from xrlenv.control.build_plan import (
        BuildEntry,
        EntryPlacement,
        GitSource,
    )

    captured: list[bool] = []

    async def fake_build(
        node_id: str, image_ref: str, source: Any,
        timeout_s: float, labels: dict[str, str],
        skip_if_present: bool = False,
    ) -> tuple[str, str | None]:
        captured.append(skip_if_present)
        return ("ok", None)

    async def _unused_ensure(*_args: Any) -> tuple[str, str | None]:
        raise AssertionError("ensure_present_fn should not be reached")

    coordinator, _ = _make_coordinator(
        ensure_present_fn=_unused_ensure,
        build_image_fn=fake_build,
    )
    plan = BuildPlan(entries=(
        BuildEntry(
            image_ref="git/source-build:1",
            context_source=GitSource(
                repo="https://github.com/example/repo", ref="main",
                subdir=".", dockerfile="Dockerfile",
            ),
            placement=EntryPlacement(size_hint_bytes=1024),
        ),
    ))

    # Default — skip_if_present is False.
    captured.clear()
    await coordinator.apply(plan)
    assert captured == [False]

    # skip_if_present=True (without force) propagates as True.
    plan2 = BuildPlan(entries=(
        BuildEntry(
            image_ref="git/source-build-2:1",
            context_source=GitSource(
                repo="https://github.com/example/repo", ref="main",
                subdir=".", dockerfile="Dockerfile",
            ),
            placement=EntryPlacement(size_hint_bytes=1024),
        ),
    ))
    captured.clear()
    await coordinator.apply(plan2, skip_if_present=True)
    assert captured == [True]

    # force=True overrides skip_if_present: forced rebuilds always
    # dispatch (operator-explicit semantics). Use a third fresh
    # plan to avoid the completed-idempotency short-circuit firing
    # against plan2 above.
    plan3 = BuildPlan(entries=(
        BuildEntry(
            image_ref="git/source-build-3:1",
            context_source=GitSource(
                repo="https://github.com/example/repo", ref="main",
                subdir=".", dockerfile="Dockerfile",
            ),
            placement=EntryPlacement(size_hint_bytes=1024),
        ),
    ))
    captured.clear()
    await coordinator.apply(plan3, force=True, skip_if_present=True)
    assert captured == [False], (
        f"force=True must override skip_if_present (forced rebuilds "
        f"always dispatch); got captured={captured!r}"
    )


@pytest.mark.asyncio
async def test_per_image_ref_apply_does_not_overwrite_cancelled_status() -> None:
    """Regression for the cancel-race smoke-found bug: when an
    operator cancel flips the plan to ``cancelled`` between the
    dispatch loop and the apply's terminal-status update, the
    apply MUST NOT overwrite it back to ``partial_failure`` /
    ``completed``.

    Simulates the race by having the ensure_present fn call
    update_build_plan_status to ``cancelled`` after recording
    its failure. The fake represents an operator-cancel handler
    racing the apply task — same effect.
    """
    from xrlenv.control.build_plan import (
        BuildEntry,
        EntryPlacement,
        RegistrySource,
    )

    captured_state: dict[str, Any] = {}

    coordinator, state = _make_coordinator(
        ensure_present_fn=lambda *args: None,  # placeholder; rebound below
    )

    async def racy_ensure(
        node_id: str, image_ref: str, timeout_s: float,
    ) -> tuple[str, str | None]:
        # Simulate the registry-404 path: assignment fails, AND
        # before the apply's final flip runs, a concurrent cancel
        # has already flipped plan status to ``cancelled``.
        state.update_build_plan_status(
            captured_state["plan_id"], "cancelled",
        )
        return ("failed", "pull access denied for fake-image: 404")

    coordinator._ensure_present_fn = racy_ensure   # type: ignore[assignment]

    plan = BuildPlan(entries=(
        BuildEntry(
            image_ref="fake/missing:1",
            context_source=RegistrySource(),
            placement=EntryPlacement(size_hint_bytes=1024),
        ),
    ))
    # Pre-compute plan_id so racy_ensure can reference it.
    from xrlenv.control.build_plan import compute_plan_id
    captured_state["plan_id"] = compute_plan_id(plan)

    outcome = await coordinator.apply(plan)

    # The outcome reflects what apply() computed locally
    # (partial_failure — 1 failure), but the PERSISTED status
    # must be ``cancelled`` because the racy cancel won.
    assert outcome.failures == 1
    plan_rec = state.get_build_plan(outcome.plan_id)
    assert plan_rec is not None
    assert plan_rec.status == "cancelled", (
        f"expected state.db to still hold 'cancelled' after apply "
        f"finished; got {plan_rec.status!r}. Apply's terminal-status "
        f"update overwrote the cancel — the CAS guard is missing."
    )
    # Apply's reported outcome status also reflects the persisted
    # truth (not the locally-computed partial_failure).
    assert outcome.status == "cancelled"


@pytest.mark.asyncio
async def test_per_image_ref_dispatches_resolved_tarball_via_build_image_fn() -> None:
    """A tarball entry with ``content_b64`` populated routes through
    ``build_image_fn`` (the same path git uses) and the dispatched
    source carries the bytes inline."""
    import base64

    from xrlenv.control.build_plan import (
        BuildEntry,
        EntryPlacement,
        TarballSource,
    )

    captured: list[Any] = []

    async def fake_build(
        node_id: str, image_ref: str, source: Any,
        timeout_s: float, labels: dict[str, str],
        skip_if_present: bool = False,
    ) -> tuple[str, str | None]:
        captured.append((image_ref, source))
        return ("ok", None)

    async def _unused_ensure(*_args: Any) -> tuple[str, str | None]:
        raise AssertionError("ensure_present_fn should not be reached")

    coordinator, _ = _make_coordinator(
        ensure_present_fn=_unused_ensure, build_image_fn=fake_build,
    )
    fake_bytes = b"<fake-tarball-bytes>"
    plan = BuildPlan(entries=(
        BuildEntry(
            image_ref="my-org/from-tarball:v1",
            context_source=TarballSource(
                path="./contexts/private-task.tar.gz",
                dockerfile="Dockerfile",
                content_b64=base64.b64encode(fake_bytes).decode("ascii"),
            ),
            placement=EntryPlacement(size_hint_bytes=1024),
        ),
    ))
    outcome = await coordinator.apply(plan)
    assert outcome.status == "completed"
    assert outcome.successes == 1
    assert outcome.failures == 0
    assert len(captured) == 1
    image_ref, dispatched_source = captured[0]
    assert image_ref == "my-org/from-tarball:v1"
    assert isinstance(dispatched_source, TarballSource)
    # Bytes round-trip end-to-end via the schema.
    assert base64.b64decode(dispatched_source.content_b64) == fake_bytes


@pytest.mark.asyncio
async def test_per_image_ref_dispatches_git_via_build_image_fn() -> None:
    """A coordinator with both ensure_present_fn (registry) and
    build_image_fn (git) wired routes each entry to the right
    dispatcher based on context_source.type."""
    from xrlenv.control.build_plan import (
        BuildEntry,
        EntryPlacement,
        GitSource,
        RegistrySource,
    )

    ensure_calls: list[str] = []
    build_calls: list[tuple[str, dict[str, str]]] = []

    async def fake_ensure(
        node_id: str, image_ref: str, timeout_s: float,
    ) -> tuple[str, str | None]:
        ensure_calls.append(image_ref)
        return ("ok", None)

    async def fake_build(
        node_id: str, image_ref: str, source: Any,
        timeout_s: float, labels: dict[str, str],
        skip_if_present: bool = False,
    ) -> tuple[str, str | None]:
        build_calls.append((image_ref, dict(labels)))
        return ("ok", None)

    coordinator, _ = _make_coordinator(
        ensure_present_fn=fake_ensure,
        build_image_fn=fake_build,
    )
    plan = BuildPlan(entries=(
        BuildEntry(
            image_ref="alex/from-registry:1",
            context_source=RegistrySource(),
            placement=EntryPlacement(
                size_hint_bytes=1024, size_hint_source="registry-probe",
            ),
        ),
        BuildEntry(
            image_ref="xrlenv-seta-env/0:main",
            context_source=GitSource(
                repo="https://github.com/example/repo", ref="main",
                subdir="path", dockerfile="Dockerfile",
            ),
            placement=EntryPlacement(size_hint_bytes=1024),
            labels={"xrlenv.benchmark": "seta-env"},
        ),
    ))
    outcome = await coordinator.apply(plan)
    assert outcome.status == "completed"
    assert outcome.successes == 2
    assert outcome.failures == 0
    # Each entry dispatched exactly once via its source-appropriate fn.
    assert ensure_calls == ["alex/from-registry:1"]
    assert len(build_calls) == 1
    assert build_calls[0][0] == "xrlenv-seta-env/0:main"
    # Operator labels propagate to the build_image_fn.
    assert build_calls[0][1] == {"xrlenv.benchmark": "seta-env"}


@pytest.mark.asyncio
async def test_per_image_ref_apply_without_dispatcher_raises() -> None:
    from xrlenv.errors import ManifestInvalid

    state = InMemoryStateStore()
    coordinator = BuildCoordinator(
        catalog=TemplateCatalog(),
        state=state,
        node_builder=InProcessNodeBuilder(),
        budget_provider=_StaticBudgetProvider([
            NodeBudget(node_id="n1", available_bytes=100 * 1024**3),  # type: ignore[arg-type]
        ]),
        # No ensure_present_fn wired.
    )
    plan = _entries_plan()
    with pytest.raises(ManifestInvalid, match="ensure_present_fn"):
        await coordinator.apply(plan)


@pytest.mark.asyncio
async def test_per_image_ref_apply_propagates_dispatcher_exception() -> None:
    async def crashing_ensure(
        node_id: str, image_ref: str, timeout_s: float,
    ) -> tuple[str, str | None]:
        raise RuntimeError("transport gone")

    coordinator, state = _make_coordinator(ensure_present_fn=crashing_ensure)
    plan = _entries_plan(image_refs=["a:1"])
    outcome = await coordinator.apply(plan)
    assert outcome.status == "partial_failure"
    rows = state.list_assignments(outcome.plan_id)
    assert rows[0].status == "failed"
    assert "RuntimeError" in (rows[0].error or "")
    assert "transport gone" in (rows[0].error or "")


@pytest.mark.asyncio
async def test_per_image_ref_apply_replication_spans_multiple_nodes() -> None:
    calls_per_node: dict[str, list[str]] = {"n1": [], "n2": []}

    async def fake_ensure(
        node_id: str, image_ref: str, timeout_s: float,
    ) -> tuple[str, str | None]:
        calls_per_node[node_id].append(image_ref)
        return ("ok", None)

    coordinator, _ = _make_coordinator(
        ensure_present_fn=fake_ensure,
        nodes=[
            NodeBudget(node_id="n1", available_bytes=100 * 1024**3),  # type: ignore[arg-type]
            NodeBudget(node_id="n2", available_bytes=100 * 1024**3),  # type: ignore[arg-type]
        ],
    )
    plan = _entries_plan(image_refs=["a:1"], preferred_home_count=2)
    outcome = await coordinator.apply(plan)
    assert outcome.status == "completed"
    assert outcome.successes == 2
    # Image landed on both nodes.
    assert calls_per_node["n1"] == ["a:1"]
    assert calls_per_node["n2"] == ["a:1"]


# ──────────────────────────────────────────────────────────────────────────────
# bypass_in_flight_check — admin uses this so a re-apply of an
# existing plan_id (whatever status: cancelled / partial_failure /
# orphan in_flight from a crashed prior process) actually proceeds
# instead of getting bounced by the coordinator's idempotency check.
# Regression for the 2026-05-12 hang.
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_per_image_ref_bypass_in_flight_check_re_applies_in_flight_row() -> None:
    """An ``in_flight`` plan_id with no live admin task (e.g. xrlenv up
    restarted mid-apply) gets re-applied when bypass is set. Without
    bypass, coordinator rejects with ``rejected_in_flight``."""
    async def fake_ensure(
        node_id: str, image_ref: str, timeout_s: float,
    ) -> tuple[str, str | None]:
        return ("ok", None)

    coordinator, state = _make_coordinator(ensure_present_fn=fake_ensure)
    plan = _entries_plan(image_refs=["a:1"])

    # Pre-seed an orphan in_flight plan row (no assignments yet, like
    # what _flip_existing_plan_to_in_flight produces).
    from xrlenv.control.build_plan import compute_plan_id
    plan_id = compute_plan_id(plan)
    state.record_build_plan(
        plan_id=plan_id, applied_by="x",
        plan_json=plan.model_dump_json(exclude_none=True),
        name=plan.name,
    )
    state.update_build_plan_status(plan_id, "in_flight")

    # Without bypass → rejected_in_flight.
    out_rejected = await coordinator.apply(plan)
    assert out_rejected.status == "rejected_in_flight"

    # With bypass → proceeds, flips to in_flight, dispatches, completes.
    out_ok = await coordinator.apply(plan, bypass_in_flight_check=True)
    assert out_ok.status == "completed"
    assert out_ok.successes == 1
    rows = state.list_assignments(plan_id)
    assert len(rows) == 1
    assert rows[0].status == "done"


@pytest.mark.asyncio
async def test_per_image_ref_bypass_in_flight_check_re_applies_cancelled() -> None:
    """A ``cancelled`` plan_id re-applied (with admin's typical flow:
    _flip pre-sets in_flight + purges assignments, then coordinator
    runs with bypass) should proceed and dispatch the entries —
    matching the operator's intent that re-applying a cancelled
    plan revives it."""
    async def fake_ensure(
        node_id: str, image_ref: str, timeout_s: float,
    ) -> tuple[str, str | None]:
        return ("ok", None)

    coordinator, state = _make_coordinator(ensure_present_fn=fake_ensure)
    plan = _entries_plan(image_refs=["a:1", "b:1"])

    # Pre-seed a cancelled plan row. ``_flip`` would set this to
    # ``in_flight`` + delete assignments before our coordinator
    # call; simulate that pre-step here.
    from xrlenv.control.build_plan import compute_plan_id
    plan_id = compute_plan_id(plan)
    state.record_build_plan(
        plan_id=plan_id, applied_by="x",
        plan_json=plan.model_dump_json(exclude_none=True),
        name=plan.name,
    )
    state.update_build_plan_status(plan_id, "in_flight")
    state.delete_assignments(plan_id)

    out = await coordinator.apply(plan, bypass_in_flight_check=True)
    assert out.status == "completed"
    assert out.successes == 2


# ──────────────────────────────────────────────────────────────────────────────
# --fill-missing apply mode (2026-05-12)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_per_image_ref_fill_missing_no_op_when_all_cached() -> None:
    """--fill-missing on a plan whose every entry is already on at
    least one node: no dispatch, all rows persist as ``done`` against
    the node that has them. The operator's "do not build if present"
    intent — covers the use case of re-applying after a completed
    plan to verify nothing has drifted."""
    dispatch_calls: list[tuple[str, str]] = []

    async def fake_ensure(
        node_id: str, image_ref: str, timeout_s: float,
    ) -> tuple[str, str | None]:
        dispatch_calls.append((node_id, image_ref))
        return ("ok", None)

    coordinator, state = _make_coordinator(
        ensure_present_fn=fake_ensure,
        nodes=[
            NodeBudget(node_id="n1", available_bytes=100 * 1024**3),  # type: ignore[arg-type]
            NodeBudget(node_id="n2", available_bytes=100 * 1024**3),  # type: ignore[arg-type]
        ],
        inventory={
            "a:1": {"n1"},
            "b:1": {"n2"},
            "c:1": {"n1", "n2"},
        },
    )
    plan = _entries_plan(image_refs=["a:1", "b:1", "c:1"])
    outcome = await coordinator.apply(plan, fill_missing=True)

    assert outcome.status == "completed"
    assert outcome.failures == 0
    # No dispatch — every entry was already cached.
    assert dispatch_calls == []
    # Three rows persisted, all ``done``, anchored at the node that
    # actually has each image. ``c:1`` is on both; first-alphabetical
    # ``n1`` wins.
    rows = sorted(
        state.list_assignments(outcome.plan_id),
        key=lambda r: r.image_ref,
    )
    assert len(rows) == 3
    by_ref = {r.image_ref: r for r in rows}
    assert by_ref["a:1"].status == "done"
    assert by_ref["a:1"].node_id == "n1"
    assert by_ref["b:1"].node_id == "n2"
    assert by_ref["c:1"].node_id == "n1"


@pytest.mark.asyncio
async def test_per_image_ref_fill_missing_dispatches_only_missing() -> None:
    """--fill-missing on a plan where one entry is absent from the
    cluster: only that one entry triggers ensure_present. The
    other two re-anchor without dispatch — the use case the
    operator described after a partial_failure where a single
    transient pull fails out of N."""
    dispatch_calls: list[tuple[str, str]] = []

    async def fake_ensure(
        node_id: str, image_ref: str, timeout_s: float,
    ) -> tuple[str, str | None]:
        dispatch_calls.append((node_id, image_ref))
        return ("ok", None)

    coordinator, state = _make_coordinator(
        ensure_present_fn=fake_ensure,
        nodes=[
            NodeBudget(node_id="n1", available_bytes=100 * 1024**3),  # type: ignore[arg-type]
            NodeBudget(node_id="n2", available_bytes=100 * 1024**3),  # type: ignore[arg-type]
        ],
        # "b:1" was meant to be on the cluster but isn't (partial pull
        # failed last time / got evicted / etc).
        inventory={
            "a:1": {"n1"},
            "c:1": {"n2"},
        },
    )
    plan = _entries_plan(image_refs=["a:1", "b:1", "c:1"])
    outcome = await coordinator.apply(plan, fill_missing=True)

    assert outcome.status == "completed"
    # 2 already-present + 1 newly-dispatched + 0 deferred.
    assert outcome.successes == 3
    assert outcome.failures == 0
    # Exactly one dispatch — for ``b:1``.
    assert len(dispatch_calls) == 1
    assert dispatch_calls[0][1] == "b:1"

    rows = sorted(
        state.list_assignments(outcome.plan_id),
        key=lambda r: r.image_ref,
    )
    statuses = {r.image_ref: r.status for r in rows}
    assert statuses == {"a:1": "done", "b:1": "done", "c:1": "done"}


@pytest.mark.asyncio
async def test_per_image_ref_fill_missing_reanchors_drifted_assignment() -> None:
    """--fill-missing rewrites the assignment row to point at the
    node that ACTUALLY has the image, not whichever node a prior
    apply happened to FFD-place it on. Operator did some manual
    docker work or a prior apply targeted a different node;
    fill_missing makes the row match reality so the image-affinity
    scheduler routes rollouts to the right place."""
    dispatch_calls: list[tuple[str, str]] = []

    async def fake_ensure(
        node_id: str, image_ref: str, timeout_s: float,
    ) -> tuple[str, str | None]:
        dispatch_calls.append((node_id, image_ref))
        return ("ok", None)

    coordinator, state = _make_coordinator(
        ensure_present_fn=fake_ensure,
        nodes=[
            NodeBudget(node_id="n1", available_bytes=100 * 1024**3),  # type: ignore[arg-type]
            NodeBudget(node_id="n2", available_bytes=100 * 1024**3),  # type: ignore[arg-type]
        ],
        # Plan would have FFD-placed on n1 (alphabetical), but reality
        # is the image lives on n2.
        inventory={"a:1": {"n2"}},
    )
    plan = _entries_plan(image_refs=["a:1"])
    outcome = await coordinator.apply(plan, fill_missing=True)

    assert outcome.status == "completed"
    # No dispatch — image was already on n2.
    assert dispatch_calls == []
    rows = state.list_assignments(outcome.plan_id)
    assert len(rows) == 1
    # The row is anchored at n2 (reality), not whichever node FFD
    # would have picked.
    assert rows[0].node_id == "n2"
    assert rows[0].image_ref == "a:1"
    assert rows[0].status == "done"


@pytest.mark.asyncio
async def test_per_image_ref_fill_missing_rejects_without_inventory_provider() -> None:
    """LocalRuntime doesn't wire an inventory provider (single-host
    plans don't need fill-missing). Apply with fill_missing=True
    must raise ManifestInvalid with a clear remediation path."""
    async def fake_ensure(
        node_id: str, image_ref: str, timeout_s: float,
    ) -> tuple[str, str | None]:
        return ("ok", None)

    coordinator, _ = _make_coordinator(
        ensure_present_fn=fake_ensure,
        # NB: NOT passing inventory={...} → inventory_provider stays None.
    )
    plan = _entries_plan(image_refs=["a:1"])
    from xrlenv.errors import ManifestInvalid

    with pytest.raises(ManifestInvalid) as exc:
        await coordinator.apply(plan, fill_missing=True)
    assert "ClusterInventoryProvider" in str(exc.value)


@pytest.mark.asyncio
async def test_per_image_ref_fill_missing_works_on_partial_failure_plan() -> None:
    """The most operationally-relevant case: previous apply landed
    in ``partial_failure`` (1 transient pull failure out of N).
    A naive re-apply without --force would also be partial_failure
    semantics here today (fall-through to fresh-apply), wasting all
    the cache-hit dispatch RPCs. --fill-missing skips the
    already-present ones and re-dispatches only the failed one."""
    dispatch_calls: list[tuple[str, str]] = []

    async def fake_ensure(
        node_id: str, image_ref: str, timeout_s: float,
    ) -> tuple[str, str | None]:
        dispatch_calls.append((node_id, image_ref))
        return ("ok", None)

    coordinator, state = _make_coordinator(
        ensure_present_fn=fake_ensure,
        nodes=[
            NodeBudget(node_id="n1", available_bytes=100 * 1024**3),  # type: ignore[arg-type]
        ],
        inventory={"a:1": {"n1"}, "b:1": {"n1"}},  # "c:1" still missing
    )
    plan = _entries_plan(image_refs=["a:1", "b:1", "c:1"])

    # Step 1: pre-seed a partial_failure plan record + assignment rows
    # (as if the first apply had landed with c:1 as the failed entry).
    from xrlenv.control.build_plan import compute_plan_id
    plan_id = compute_plan_id(plan)
    state.record_build_plan(
        plan_id=plan_id, applied_by="operator-cli",
        plan_json=plan.model_dump_json(exclude_none=True),
        name=plan.name,
    )
    state.update_build_plan_status(plan_id, "partial_failure")
    from xrlenv.control.build_coordinator import PER_IMAGE_REF_BENCHMARK_TAG
    for ref, status in [("a:1", "done"), ("b:1", "done"), ("c:1", "failed")]:
        state.record_assignment(BuildAssignmentRecord(
            plan_id=plan_id, node_id="n1",
            image_ref=ref, benchmark=PER_IMAGE_REF_BENCHMARK_TAG,
            status=status,
        ))

    # Step 2: re-apply with --fill-missing. Should dispatch only c:1.
    outcome = await coordinator.apply(plan, fill_missing=True)

    assert outcome.status == "completed"
    assert outcome.failures == 0
    assert len(dispatch_calls) == 1
    assert dispatch_calls[0][1] == "c:1"
    rows = sorted(
        state.list_assignments(outcome.plan_id),
        key=lambda r: r.image_ref,
    )
    statuses = {r.image_ref: r.status for r in rows}
    assert statuses == {"a:1": "done", "b:1": "done", "c:1": "done"}


@pytest.mark.asyncio
async def test_per_image_ref_apply_concurrency_caps_in_flight_dispatch() -> None:
    """``apply(concurrency=N)`` bounds simultaneous ensure_present_fn calls."""
    import asyncio

    state_box = {"inflight": 0, "max_inflight": 0}

    async def tracking_ensure(
        node_id: str, image_ref: str, timeout_s: float,
    ) -> tuple[str, str | None]:
        state_box["inflight"] += 1
        state_box["max_inflight"] = max(
            state_box["max_inflight"], state_box["inflight"],
        )
        try:
            await asyncio.sleep(0.05)
        finally:
            state_box["inflight"] -= 1
        return ("ok", None)

    coordinator, _ = _make_coordinator(
        ensure_present_fn=tracking_ensure,
        nodes=[NodeBudget(  # type: ignore[arg-type]
            node_id="n1", available_bytes=1000 * 1024**3,
        )],
    )
    plan = _entries_plan(image_refs=[f"img/{i}:1" for i in range(6)])
    outcome = await coordinator.apply(plan, concurrency=2)

    assert outcome.status == "completed"
    assert outcome.successes == 6
    assert state_box["max_inflight"] == 2


@pytest.mark.asyncio
async def test_per_image_ref_apply_concurrency_defaults_admit_small_plan() -> None:
    """Without ``concurrency`` the default fan-out runs a small plan wide."""
    import asyncio

    state_box = {"inflight": 0, "max_inflight": 0}

    async def tracking_ensure(*_args: Any) -> tuple[str, str | None]:
        state_box["inflight"] += 1
        state_box["max_inflight"] = max(
            state_box["max_inflight"], state_box["inflight"],
        )
        try:
            await asyncio.sleep(0.02)
        finally:
            state_box["inflight"] -= 1
        return ("ok", None)

    coordinator, _ = _make_coordinator(
        ensure_present_fn=tracking_ensure,
        nodes=[NodeBudget(  # type: ignore[arg-type]
            node_id="n1", available_bytes=1000 * 1024**3,
        )],
    )
    plan = _entries_plan(image_refs=[f"img/{i}:1" for i in range(5)])
    outcome = await coordinator.apply(plan)  # default fan-out (>= 32)

    assert outcome.status == "completed"
    assert state_box["max_inflight"] == 5


# ──────────────────────────────────────────────────────────────────────────────
# Disk-aware build dispatch (pace dispatch against each node's free disk so a
# heavy plan doesn't overrun the node's eviction reserve)
# ──────────────────────────────────────────────────────────────────────────────


class _FakeAssignment:
    def __init__(self, image_ref: str) -> None:
        self.image_ref = image_ref


class _FakeEntry:
    # Mirror the real BuildEntry shape: size_hint_bytes lives on the nested
    # EntryPlacement, NOT on the entry directly. Build a real EntryPlacement so
    # this test breaks if the field ever moves/renames — the earlier flat fake
    # gave false confidence and let an AttributeError reach production.
    def __init__(self, size_hint_bytes: int) -> None:
        self.placement = EntryPlacement(size_hint_bytes=size_hint_bytes)


def _bare_coordinator() -> BuildCoordinator:
    # The disk-gate helpers only touch ``_free_disk_fn``; bypass the heavy
    # constructor so the unit tests stay focused on the pacing logic.
    return BuildCoordinator.__new__(BuildCoordinator)


def test_dispatch_watermark_scales_with_largest_image() -> None:
    from xrlenv.control.build_coordinator import DISPATCH_DISK_HEADROOM_FACTOR

    coord = _bare_coordinator()

    async def _fd(_node_id: str) -> tuple[int, int] | None:
        return (0, 0)

    coord._free_disk_fn = _fd
    assignments = [_FakeAssignment("a"), _FakeAssignment("b")]
    entry_by_ref = {
        "a": _FakeEntry(2 * 1024**3),
        "b": _FakeEntry(5 * 1024**3),  # largest
    }
    wm = coord._dispatch_watermark_bytes(assignments, entry_by_ref)
    assert wm == int(5 * 1024**3 * DISPATCH_DISK_HEADROOM_FACTOR)


def test_dispatch_watermark_zero_without_probe() -> None:
    coord = _bare_coordinator()
    coord._free_disk_fn = None  # gate disabled
    assert coord._dispatch_watermark_bytes(
        [_FakeAssignment("a")], {"a": _FakeEntry(9 * 1024**3)},
    ) == 0


async def test_await_disk_headroom_noop_when_disabled() -> None:
    coord = _bare_coordinator()
    coord._free_disk_fn = None
    # Returns immediately even with a positive watermark (gate disabled).
    await coord._await_disk_headroom("n", 10 * 1024**3)


async def test_await_disk_headroom_returns_when_already_free() -> None:
    coord = _bare_coordinator()

    async def _fd(_node_id: str) -> tuple[int, int] | None:
        return (100 * 1024**3, 500 * 1024**3)

    coord._free_disk_fn = _fd
    await coord._await_disk_headroom("n", 50 * 1024**3)  # 100 >= 50


async def test_await_disk_headroom_waits_then_proceeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import xrlenv.control.build_coordinator as bc

    calls = {"n": 0}

    async def _fd(_node_id: str) -> tuple[int, int] | None:
        calls["n"] += 1
        # Low for the first two polls, then recovers (eviction freed space).
        free = 10 * 1024**3 if calls["n"] < 3 else 100 * 1024**3
        return (free, 500 * 1024**3)

    async def _no_sleep(_s: float) -> None:
        return None

    monkeypatch.setattr(bc.asyncio, "sleep", _no_sleep)
    coord = _bare_coordinator()
    coord._free_disk_fn = _fd
    await coord._await_disk_headroom("n", 50 * 1024**3)
    assert calls["n"] == 3  # polled until free climbed past the watermark


async def test_await_disk_headroom_times_out_and_proceeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import xrlenv.control.build_coordinator as bc

    async def _fd(_node_id: str) -> tuple[int, int] | None:
        return (1 * 1024**3, 500 * 1024**3)  # never enough

    async def _no_sleep(_s: float) -> None:
        return None

    monkeypatch.setattr(bc.asyncio, "sleep", _no_sleep)
    monkeypatch.setattr(bc, "DISPATCH_DISK_WAIT_TIMEOUT_S", 0.0)
    coord = _bare_coordinator()
    coord._free_disk_fn = _fd
    # Must return (not hang) despite free always below the watermark.
    import asyncio
    await asyncio.wait_for(
        coord._await_disk_headroom("n", 50 * 1024**3), timeout=2.0,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Size-hint → on-disk inflation (compressed registry-probe sizes must be
# reserved as on-disk, or FFD over-packs and the build ENOSPCs / defers nothing)
# ──────────────────────────────────────────────────────────────────────────────


def test_expected_on_disk_bytes_inflates_compressed_sources() -> None:
    # Defaults: registry-probe/heuristic x3.0 (compressed → on-disk),
    # cluster-reported x1.0 (already measured on-disk). No module reload here —
    # reloading image_planner swaps its class identities and breaks the rest of
    # the suite (isinstance / shared-type assumptions).
    from xrlenv.control.image_planner import expected_on_disk_bytes

    gib = 1024**3
    assert expected_on_disk_bytes(gib, "registry-probe") == 3 * gib
    assert expected_on_disk_bytes(gib, "heuristic") == 3 * gib
    assert expected_on_disk_bytes(gib, "cluster-reported") == gib


@pytest.mark.asyncio
async def test_registry_probe_hints_inflated_so_overflow_defers() -> None:
    # 12 GiB node, 6 images x 1 GiB *compressed* registry-probe hints. Raw they
    # all fit (6 GiB <= 12); but on-disk is ~3x, so only 4 fit (12/3) and 2 must
    # defer. Pre-fix the packer used the raw 1 GiB and deferred NOTHING, then the
    # build ENOSPC'd.
    async def fake_ensure(
        node_id: str, image_ref: str, timeout_s: float,
    ) -> tuple[str, str | None]:
        return ("ok", None)

    coordinator, state = _make_coordinator(
        ensure_present_fn=fake_ensure,
        nodes=[NodeBudget(node_id="n1", available_bytes=12 * 1024**3)],
    )
    plan = _entries_plan(
        image_refs=[f"img:{i}" for i in range(6)],
        size_hint_bytes=1 * 1024**3,
        size_hint_source="registry-probe",
    )
    outcome = await coordinator.apply(plan)
    assert outcome.deferred == 2  # 4 fit at 3 GiB each; 2 overflow
    rows = {r.image_ref: r for r in state.list_assignments(outcome.plan_id)}
    assert sum(1 for r in rows.values() if r.status == "registered") == 2


@pytest.mark.asyncio
async def test_cluster_reported_hints_not_inflated() -> None:
    # Same plan/budget but cluster-reported (already on-disk) → 1x → all 6 fit,
    # nothing deferred. Proves the inflation is source-aware, not blanket.
    async def fake_ensure(
        node_id: str, image_ref: str, timeout_s: float,
    ) -> tuple[str, str | None]:
        return ("ok", None)

    coordinator, _state = _make_coordinator(
        ensure_present_fn=fake_ensure,
        nodes=[NodeBudget(node_id="n1", available_bytes=12 * 1024**3)],
    )
    plan = _entries_plan(
        image_refs=[f"img:{i}" for i in range(6)],
        size_hint_bytes=1 * 1024**3,
        size_hint_source="cluster-reported",
    )
    outcome = await coordinator.apply(plan)
    assert outcome.deferred == 0


# ──────────────────────────────────────────────────────────────────────────────
# Wire timeout ≠ failure: a timed-out dispatch is recorded ``registered``
# (lazy), not ``failed`` — the node may still complete the pull.
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_dispatch_timeout_records_registered_not_failed() -> None:
    async def ensure_with_timeout(
        node_id: str, image_ref: str, timeout_s: float,
    ) -> tuple[str, str | None]:
        if image_ref == "slow:1":
            return ("timeout", "node x: command y timed out after 60.0s")
        return ("ok", None)

    coordinator, state = _make_coordinator(ensure_present_fn=ensure_with_timeout)
    plan = _entries_plan(image_refs=["good:1", "slow:1"])
    outcome = await coordinator.apply(plan)

    rows = {r.image_ref: r for r in state.list_assignments(outcome.plan_id)}
    assert rows["good:1"].status == "done"
    # The timed-out one is registered (lazy), NOT failed — and carries no error.
    assert rows["slow:1"].status == "registered"
    assert rows["slow:1"].error is None
    assert outcome.successes == 1
    assert outcome.failures == 0  # a wire timeout is not a failure


@pytest.mark.asyncio
async def test_coordinator_rejects_local_source_with_clear_message() -> None:
    """``local`` context sources are build-host-only (they docker-build a path in
    place). Feeding one to the cluster build-apply path — which ships sources to
    nodes that may not share the path — rejects with a ManifestInvalid pointing
    at build_and_push_images.py, instead of silently dropping the entry at the
    per-source partition."""
    from xrlenv.errors import ManifestInvalid

    async def _never(*a: Any, **k: Any) -> tuple[str, str | None]:
        raise AssertionError("dispatch must not run for a rejected plan")

    coordinator, _state = _make_coordinator(ensure_present_fn=_never)
    plan = BuildPlan(entries=(BuildEntry(
        image_ref="turing-tb2/abs-mex-service:main",
        context_source=LocalSource(
            path="/path/to/data",
            shared_fs="hyperpod",
        ),
        placement=EntryPlacement(size_hint_bytes=1, size_hint_source="heuristic"),
    ),))
    # Raises before any dispatch — _never (ensure_present_fn) is never awaited.
    with pytest.raises(ManifestInvalid, match="build-host-only"):
        await coordinator.apply(plan)
