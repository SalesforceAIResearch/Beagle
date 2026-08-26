"""Tests for ``xrlenv build push`` — slices 2b + 3 (e7df646).

Covers:
- ``_shard_for_push`` pure function (every-assigned-once, size-balance,
  exceeds-budget-still-assigns, empty-nodes-in-result filtered)
- ``apply(push=True)`` via coordinator: oversized plan succeeds, no-nodes raises
- ``_qualify_image_ref`` + ``_registry_qualify_plan`` helper logic
- ``cmd_build_push`` guard-rails (no connect_host → rc=2; non-source entries → rc=2)
- ``cmd_build_push`` happy path: _build_apply_via_admin called with push=True
- Admin /api/build/apply push param threads through to coordinator.apply
- LocalRuntime build_push_fn wiring
"""

from __future__ import annotations

import io
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ──────────────────────────────────────────────────────────────────────────────
# 1. _shard_for_push — pure function tests
# ──────────────────────────────────────────────────────────────────────────────


def _make_image(ref: str, size_bytes: int):
    from xrlenv.control.image_planner import ImageToPlace

    return ImageToPlace(image_ref=ref, size_bytes=size_bytes)  # type: ignore[arg-type]


def _make_node(node_id: str, available_bytes: int = 100 * 1024**3):
    from xrlenv.control.image_planner import NodeBudget

    return NodeBudget(node_id=node_id, available_bytes=available_bytes)  # type: ignore[arg-type]


def test_shard_for_push_every_image_assigned_exactly_once() -> None:
    """Every image appears in exactly one assignment across all nodes."""
    from xrlenv.control.build_coordinator import _shard_for_push

    images = [
        _make_image("reg/a:1", 1 * 1024**3),
        _make_image("reg/b:1", 2 * 1024**3),
        _make_image("reg/c:1", 3 * 1024**3),
        _make_image("reg/d:1", 500 * 1024**2),
    ]
    nodes = [_make_node("n1"), _make_node("n2"), _make_node("n3")]

    result = _shard_for_push(images, nodes)

    assigned_refs = [a.image_ref for a in result.assignments]
    expected_refs = {img.image_ref for img in images}
    # Every ref appears exactly once.
    assert sorted(assigned_refs) == sorted(expected_refs), (
        f"Expected every image assigned once; got: {assigned_refs}"
    )
    assert len(assigned_refs) == len(set(assigned_refs)), (
        "Duplicate assignments found"
    )


def test_shard_for_push_size_balance_big_images_spread() -> None:
    """LPT-greedy: the three biggest images should NOT all land on the same
    node when 3 nodes are available."""
    from xrlenv.control.build_coordinator import _shard_for_push

    # Three equal-sized big images, three nodes → one per node.
    GiB = 1024**3
    images = [
        _make_image("reg/big-a:1", 10 * GiB),
        _make_image("reg/big-b:1", 10 * GiB),
        _make_image("reg/big-c:1", 10 * GiB),
    ]
    nodes = [_make_node("n1"), _make_node("n2"), _make_node("n3")]

    result = _shard_for_push(images, nodes)

    # Each node should get exactly one of the three equal images.
    by_node = result.assignments_by_node
    assert set(by_node.keys()) == {"n1", "n2", "n3"}, (
        "All three nodes should get work"
    )
    for nid, assignments in by_node.items():
        assert len(assignments) == 1, (
            f"Node {nid} got {len(assignments)} images, expected 1"
        )


def test_shard_for_push_lopsided_sizes_stay_balanced() -> None:
    """With lopsided sizes, the two biggest images go to different nodes
    (the greedy assigns the first big one, then the second to the now-lighter
    node)."""
    from xrlenv.control.build_coordinator import _shard_for_push

    GiB = 1024**3
    images = [
        _make_image("reg/giant:1", 100 * GiB),
        _make_image("reg/large:1", 50 * GiB),
        _make_image("reg/small:1", 1 * GiB),
    ]
    nodes = [_make_node("n1"), _make_node("n2")]

    result = _shard_for_push(images, nodes)

    by_node = result.assignments_by_node
    # Giant (100G) goes to one node; large (50G) to the other.
    giant_node = next(
        nid for nid, asgns in by_node.items()
        if any(a.image_ref == "reg/giant:1" for a in asgns)
    )
    large_node = next(
        nid for nid, asgns in by_node.items()
        if any(a.image_ref == "reg/large:1" for a in asgns)
    )
    assert giant_node != large_node, (
        "Giant + large images should land on different nodes for balance"
    )


def test_shard_for_push_exceeds_every_node_disk_still_assigns_all() -> None:
    """A plan whose total size far exceeds every node's available_bytes still
    assigns ALL images (no FFD rejection — this is the core regression guard)."""
    from xrlenv.control.build_coordinator import _shard_for_push

    # Each node has 1 GiB; each image is 100 GiB — impossible for FFD.
    GiB = 1024**3
    images = [_make_image(f"reg/huge-{i}:1", 100 * GiB) for i in range(5)]
    nodes = [_make_node("n1", 1 * GiB), _make_node("n2", 1 * GiB)]

    result = _shard_for_push(images, nodes)

    assert len(result.assignments) == 5, (
        "All 5 images must be assigned even though total size >> node budgets"
    )
    assigned_refs = {a.image_ref for a in result.assignments}
    assert assigned_refs == {img.image_ref for img in images}


def test_shard_for_push_assignments_by_node_excludes_empty_nodes() -> None:
    """``assignments_by_node`` only contains nodes that actually got work.
    If images < nodes, empty nodes must not appear in the dict."""
    from xrlenv.control.build_coordinator import _shard_for_push

    images = [_make_image("reg/only-one:1", 1 * 1024**3)]
    nodes = [_make_node("n1"), _make_node("n2"), _make_node("n3")]

    result = _shard_for_push(images, nodes)

    # Only 1 image → exactly 1 node gets work.
    assert len(result.assignments_by_node) == 1, (
        f"Expected 1 node in assignments_by_node; got: {list(result.assignments_by_node)}"
    )
    # That one node has the image.
    only_node_assignments = next(iter(result.assignments_by_node.values()))
    assert only_node_assignments[0].image_ref == "reg/only-one:1"


# ──────────────────────────────────────────────────────────────────────────────
# 2. apply(push=True) — sharding + no-nodes guard
# ──────────────────────────────────────────────────────────────────────────────


def _make_push_coordinator_for_sharding(
    *,
    nodes,
    build_push_fn: Any = None,
):
    """Re-uses the test-double pattern from the existing suite."""
    from xrlenv.control.build_coordinator import BuildCoordinator
    from xrlenv.control.node_builder import InProcessNodeBuilder
    from xrlenv.control.state import InMemoryStateStore
    from xrlenv.control.template_catalog import TemplateCatalog

    state = InMemoryStateStore()

    class _StaticBudgetProvider:
        def __init__(self, budgets) -> None:
            self._budgets = budgets

        async def get_budgets(self, **_kw):
            return list(self._budgets)

    async def _dummy_ensure(node_id, image_ref, timeout_s):
        return ("ok", None)

    async def _ok_push(node_id, image_ref, source, timeout_s, labels):
        return ("ok", None, f"{image_ref}@sha256:pushed")

    coordinator = BuildCoordinator(
        catalog=TemplateCatalog(),
        state=state,
        node_builder=InProcessNodeBuilder(),
        budget_provider=_StaticBudgetProvider(nodes),
        ensure_present_fn=_dummy_ensure,
        build_push_fn=build_push_fn or _ok_push,
    )
    return coordinator, state


@pytest.mark.asyncio
async def test_push_apply_plan_exceeding_node_disk_succeeds() -> None:
    """Regression guard: apply(push=True) with a git-source plan whose total
    size far exceeds node available_bytes must still succeed — no FFD rejection.
    This proves push bypasses the FFD fit constraint that build apply has."""
    from xrlenv.control.build_plan import BuildEntry, BuildPlan, EntryPlacement, GitSource

    GiB = 1024**3
    # Each node has 1 GiB; each image is 100 GiB.
    nodes = [_make_node("n1", 1 * GiB), _make_node("n2", 1 * GiB)]
    coordinator, _state = _make_push_coordinator_for_sharding(nodes=nodes)

    plan = BuildPlan(entries=tuple(
        BuildEntry(
            image_ref=f"reg/env-{i}:v1",
            context_source=GitSource(
                repo="https://github.com/example/repo",
                ref="main", subdir=".", dockerfile="Dockerfile",
            ),
            placement=EntryPlacement(size_hint_bytes=100 * GiB),
        )
        for i in range(4)
    ))

    outcome = await coordinator.apply(plan, push=True)

    assert outcome.status == "completed", (
        f"Expected completed; got {outcome.status} — push must not FFD-reject"
    )
    assert outcome.successes == 4
    assert outcome.failures == 0


@pytest.mark.asyncio
async def test_push_apply_no_connected_nodes_raises_manifest_invalid() -> None:
    """apply(push=True) with an empty node list raises ManifestInvalid
    ('no nodes are connected')."""
    from xrlenv.control.build_plan import BuildEntry, BuildPlan, EntryPlacement, GitSource
    from xrlenv.errors import ManifestInvalid

    coordinator, _ = _make_push_coordinator_for_sharding(nodes=[])

    plan = BuildPlan(entries=(
        BuildEntry(
            image_ref="reg/env:v1",
            context_source=GitSource(
                repo="https://github.com/example/repo",
                ref="main", subdir=".", dockerfile="Dockerfile",
            ),
            placement=EntryPlacement(size_hint_bytes=1 * 1024**3),
        ),
    ))

    with pytest.raises(ManifestInvalid) as excinfo:
        await coordinator.apply(plan, push=True)

    msg = str(excinfo.value)
    assert "no nodes" in msg.lower() or "connected" in msg.lower(), (
        f"Expected 'no nodes'/'connected' in error; got: {msg!r}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# 3. CLI helpers + cmd_build_push
# ──────────────────────────────────────────────────────────────────────────────


class TestQualifyImageRef:
    """Unit tests for ``_qualify_image_ref``."""

    def test_bare_ref_gets_registry_prepended(self) -> None:
        from xrlenv.cli.commands import _qualify_image_ref

        assert _qualify_image_ref("hb__x", "reg:5011") == "reg:5011/hb__x"

    def test_nested_bare_ref_gets_registry_prepended(self) -> None:
        from xrlenv.cli.commands import _qualify_image_ref

        assert (
            _qualify_image_ref("team/env:v1", "localhost:5011")
            == "localhost:5011/team/env:v1"
        )

    def test_already_qualified_with_colon_port_unchanged(self) -> None:
        """First segment contains ':', so it's already a registry host."""
        from xrlenv.cli.commands import _qualify_image_ref

        ref = "reg:5011/hb__x"
        assert _qualify_image_ref(ref, "other:5011") == ref

    def test_already_qualified_with_dot_unchanged(self) -> None:
        """First segment contains '.', so it's already a FQDN registry."""
        from xrlenv.cli.commands import _qualify_image_ref

        ref = "reg.example.com/repo/img:tag"
        assert _qualify_image_ref(ref, "reg:5011") == ref

    def test_localhost_prefix_unchanged(self) -> None:
        """First segment is 'localhost'."""
        from xrlenv.cli.commands import _qualify_image_ref

        ref = "localhost/myimage:1"
        assert _qualify_image_ref(ref, "reg:5011") == ref

    def test_localhost_with_port_first_segment_unchanged(self) -> None:
        """'localhost:5000/repo' — first segment is 'localhost:5000' (has ':')."""
        from xrlenv.cli.commands import _qualify_image_ref

        ref = "localhost:5000/repo:tag"
        assert _qualify_image_ref(ref, "reg:5011") == ref

    def test_single_component_ref_with_tag_colon_is_qualified(
        self,
    ) -> None:
        """A slash-less ref like ``env-a:v1`` is a bare ``repo:tag`` — the ``:``
        is a tag separator, not a host port — so it IS registry-qualified.
        Docker only applies the first-segment host heuristic when the ref
        contains a ``/``; without one there is no registry component to detect.
        """
        from xrlenv.cli.commands import _qualify_image_ref

        assert _qualify_image_ref("env-a:v1", "reg:5011") == "reg:5011/env-a:v1"


class TestRegistryQualifyPlan:
    """Unit tests for ``_registry_qualify_plan``."""

    def test_qualifies_all_entries(self) -> None:
        """Bare refs get qualified — including slash-less ``repo:tag`` refs,
        whose ``:`` is a tag separator (the host heuristic only applies to a
        first segment when the ref contains a ``/``)."""
        from xrlenv.cli.commands import _registry_qualify_plan
        from xrlenv.control.build_plan import (
            BuildEntry,
            BuildPlan,
            EntryPlacement,
            GitSource,
        )

        # Use bare refs without colons — the expected convention for
        # images that need to be qualified against a registry.
        plan = BuildPlan(entries=(
            BuildEntry(
                image_ref="hb__env-a",
                context_source=GitSource(
                    repo="https://github.com/x/r", ref="main",
                    subdir=".", dockerfile="Dockerfile",
                ),
                placement=EntryPlacement(size_hint_bytes=1 * 1024**3),
            ),
            BuildEntry(
                image_ref="hb__env-b",
                context_source=GitSource(
                    repo="https://github.com/x/r", ref="main",
                    subdir=".", dockerfile="Dockerfile",
                ),
                placement=EntryPlacement(size_hint_bytes=1 * 1024**3),
            ),
        ))

        qualified = _registry_qualify_plan(plan, "reg:5011")

        refs = [e.image_ref for e in qualified.entries]
        assert refs == ["reg:5011/hb__env-a", "reg:5011/hb__env-b"]

    def test_already_qualified_refs_unchanged(self) -> None:
        from xrlenv.cli.commands import _registry_qualify_plan
        from xrlenv.control.build_plan import (
            BuildEntry,
            BuildPlan,
            EntryPlacement,
            GitSource,
        )

        plan = BuildPlan(entries=(
            BuildEntry(
                image_ref="reg.example.com/env:v1",
                context_source=GitSource(
                    repo="https://github.com/x/r", ref="main",
                    subdir=".", dockerfile="Dockerfile",
                ),
                placement=EntryPlacement(size_hint_bytes=1 * 1024**3),
            ),
        ))

        qualified = _registry_qualify_plan(plan, "reg:5011")

        assert qualified.entries[0].image_ref == "reg.example.com/env:v1"


def _write_git_source_plan(tmp_path: Path, image_refs: list[str] | None = None) -> Path:
    """Write a minimal git-source BuildPlan YAML to tmp_path and return its path.

    Note on ref naming: bare refs (no registry prefix) must NOT contain a colon
    in the name/tag, because ``_qualify_image_ref`` checks ``":" in first_segment``
    and would leave them unqualified (treating the tag colon as a host:port indicator).
    Use underscores or dashes without a tag suffix for bare refs in tests.
    """
    refs = image_refs or ["bare-ref-v1"]
    lines = ["version: 1", "entries:"]
    for ref in refs:
        lines += [
            f"  - image_ref: {ref}",
            "    context_source:",
            "      type: git",
            "      repo: https://github.com/example/repo",
            "      ref: main",
            "    placement:",
            "      size_hint_bytes: 1073741824",
        ]
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text("\n".join(lines) + "\n")
    return plan_path


class TestCmdBuildPush:
    """Integration tests for ``cmd_build_push``."""

    def test_no_connect_host_returns_2(self, tmp_path: Path) -> None:
        """connect_host=None must print an error and return 2."""
        from xrlenv.cli.commands import cmd_build_push

        plan_path = _write_git_source_plan(tmp_path)
        out = io.StringIO()
        rc = cmd_build_push(
            plan_path=plan_path,
            registry="reg:5011",
            out=out,
            connect_host=None,
        )
        assert rc == 2
        body = out.getvalue()
        assert "connect-host" in body or "connect_host" in body or "error" in body

    def test_registry_source_entry_rejected(self, tmp_path: Path) -> None:
        """A plan with a RegistrySource entry must be rejected with 'only accepts
        git/tarball' and return 2."""
        from xrlenv.cli.commands import cmd_build_push

        # Write a plan with a registry-source entry.
        yaml_content = textwrap.dedent("""\
            version: 1
            entries:
              - image_ref: reg.example.com/prebuilt:v1
                context_source:
                  type: registry
                placement:
                  size_hint_bytes: 1073741824
        """)
        plan_path = tmp_path / "registry_plan.yaml"
        plan_path.write_text(yaml_content)

        out = io.StringIO()
        rc = cmd_build_push(
            plan_path=plan_path,
            registry="reg:5011",
            out=out,
            connect_host="admin.example.com",
        )
        assert rc == 2
        body = out.getvalue()
        assert "git/tarball" in body or "registry" in body.lower()

    def test_local_source_entry_rejected(self, tmp_path: Path) -> None:
        """A plan with a LocalSource entry must also be rejected."""
        from xrlenv.cli.commands import cmd_build_push

        yaml_content = textwrap.dedent("""\
            version: 1
            entries:
              - image_ref: local-built:v1
                context_source:
                  type: local
                  path: /shared/fs/context
                  shared_fs: hyperpod
                placement:
                  size_hint_bytes: 1073741824
        """)
        plan_path = tmp_path / "local_plan.yaml"
        plan_path.write_text(yaml_content)

        out = io.StringIO()
        rc = cmd_build_push(
            plan_path=plan_path,
            registry="reg:5011",
            out=out,
            connect_host="admin.example.com",
        )
        assert rc == 2
        body = out.getvalue()
        assert "git/tarball" in body or "error" in body

    def test_happy_path_calls_build_apply_via_admin_with_push_true(
        self, tmp_path: Path,
    ) -> None:
        """Happy path: cmd_build_push calls _build_apply_via_admin with
        push=True, the registry-qualified plan, and the correct connect_host."""
        from xrlenv.cli import commands as _cli_mod

        plan_path = _write_git_source_plan(tmp_path, image_refs=["bare-ref-v1"])

        captured: dict[str, Any] = {}

        def _fake_build_apply_via_admin(**kwargs: Any) -> int:
            captured.update(kwargs)
            return 0

        out = io.StringIO()
        with patch.object(_cli_mod, "_build_apply_via_admin", _fake_build_apply_via_admin):
            rc = _cli_mod.cmd_build_push(
                plan_path=plan_path,
                registry="reg:5011",
                out=out,
                connect_host="admin.example.com",
                connect_port=9090,
            )

        assert rc == 0
        assert captured.get("push") is True, (
            f"Expected push=True in _build_apply_via_admin kwargs; got: {captured}"
        )
        assert captured.get("host") == "admin.example.com"
        assert captured.get("port") == 9090
        # All image_refs in the plan must be registry-qualified.
        plan_sent = captured.get("plan")
        assert plan_sent is not None
        for entry in plan_sent.entries:
            assert entry.image_ref.startswith("reg:5011/"), (
                f"image_ref not qualified: {entry.image_ref!r}"
            )

    def test_happy_path_eager_is_false(self, tmp_path: Path) -> None:
        """cmd_build_push passes eager=False to _build_apply_via_admin
        (push path never uses eager FFD placement)."""
        from xrlenv.cli import commands as _cli_mod

        plan_path = _write_git_source_plan(tmp_path)
        captured: dict[str, Any] = {}

        def _fake(**kwargs: Any) -> int:
            captured.update(kwargs)
            return 0

        with patch.object(_cli_mod, "_build_apply_via_admin", _fake):
            _cli_mod.cmd_build_push(
                plan_path=plan_path,
                registry="reg:5011",
                out=io.StringIO(),
                connect_host="admin.example.com",
            )

        assert captured.get("eager") is False, (
            "push path must pass eager=False to avoid FFD rejection"
        )


# ──────────────────────────────────────────────────────────────────────────────
# 4. Admin /api/build/apply push param → coordinator.apply(push=...)
# ──────────────────────────────────────────────────────────────────────────────


def test_api_build_apply_threads_push_to_coordinator(tmp_path: Path) -> None:
    """``push`` in the request body must reach ``coordinator.apply(push=True)``
    on the dry-run path (where the result is returned synchronously)."""
    from fastapi.testclient import TestClient
    from xrlenv.admin.server import AdminServerConfig, build_admin_app
    from xrlenv.control.image_planner import PlacementResult

    captured: dict[str, Any] = {}

    class _FakeCoordinator:
        async def apply(self, plan, **kw):
            captured.update(kw)
            from xrlenv.control.build_coordinator import BuildOutcome

            return BuildOutcome(
                plan_id="push-test-1234", status="dry_run",
                placement=PlacementResult(
                    assignments=(), assignments_by_node={},
                ),
            )

    cfg = AdminServerConfig(
        state_db=tmp_path / "state.db",
        runs_root=tmp_path / "runs",
        port=0,
        build_coordinator=_FakeCoordinator(),
    )
    client = TestClient(build_admin_app(cfg))

    r = client.post("/api/build/apply", json={
        "plan": {
            "version": 1,
            "entries": [{
                "image_ref": "reg:5011/env:v1",
                "context_source": {"type": "registry"},
                "placement": {"size_hint_bytes": 1_000_000_000},
            }],
        },
        "dry_run": True,
        "push": True,
    })

    assert r.status_code == 200
    assert captured.get("push") is True, (
        f"Expected push=True to reach coordinator.apply; captured kwargs: {captured}"
    )


def test_api_build_apply_push_defaults_to_false(tmp_path: Path) -> None:
    """When ``push`` is omitted from the request body, coordinator.apply gets
    push=False (the existing default) — no accidental push mode."""
    from fastapi.testclient import TestClient
    from xrlenv.admin.server import AdminServerConfig, build_admin_app
    from xrlenv.control.image_planner import PlacementResult

    captured: dict[str, Any] = {}

    class _FakeCoordinator:
        async def apply(self, plan, **kw):
            captured.update(kw)
            from xrlenv.control.build_coordinator import BuildOutcome

            return BuildOutcome(
                plan_id="no-push-test-1234", status="dry_run",
                placement=PlacementResult(
                    assignments=(), assignments_by_node={},
                ),
            )

    cfg = AdminServerConfig(
        state_db=tmp_path / "state.db",
        runs_root=tmp_path / "runs",
        port=0,
        build_coordinator=_FakeCoordinator(),
    )
    client = TestClient(build_admin_app(cfg))

    r = client.post("/api/build/apply", json={
        "plan": {
            "version": 1,
            "entries": [{
                "image_ref": "reg:5011/env:v1",
                "context_source": {"type": "registry"},
                "placement": {"size_hint_bytes": 1_000_000_000},
            }],
        },
        "dry_run": True,
        # "push" intentionally omitted
    })

    assert r.status_code == 200
    assert captured.get("push") is False, (
        f"Expected push=False when omitted; got: {captured.get('push')!r}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# 5. LocalRuntime build_push_fn wiring
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_apply_push_false_calls_build_image_fn_not_build_push_fn(
    tmp_path: Path,
) -> None:
    """Seam regression guard: apply(push=False) routes through build_image_fn and
    never touches build_push_fn — confirms the two dispatch paths are isolated.

    This is the critical invariant that prevents build apply from accidentally
    acquiring push-mode semantics when the push feature is wired in."""
    from xrlenv.control.build_plan import BuildEntry, BuildPlan, EntryPlacement, GitSource

    build_image_calls: list[tuple] = []
    build_push_calls: list[tuple] = []

    async def _record_build_image(
        node_id, image_ref, source, timeout_s, labels, skip_if_present,
    ):
        build_image_calls.append((node_id, image_ref))
        return ("ok", None)

    async def _record_build_push(node_id, image_ref, source, timeout_s, **kw):
        build_push_calls.append((node_id, image_ref))
        return ("ok", None, f"{image_ref}@sha256:pushed")

    coordinator, _ = _make_push_coordinator_for_sharding(
        nodes=[_make_node("n1")],
        build_push_fn=_record_build_push,
    )
    # Manually wire build_image_fn so both dispatchers are present.
    coordinator._build_image_fn = _record_build_image  # type: ignore[assignment]

    plan = BuildPlan(entries=(
        BuildEntry(
            image_ref="reg/env:v1",
            context_source=GitSource(
                repo="https://github.com/example/repo",
                ref="main", subdir=".", dockerfile="Dockerfile",
            ),
            placement=EntryPlacement(size_hint_bytes=1 * 1024**3),
        ),
    ))

    outcome = await coordinator.apply(plan, push=False)

    assert outcome.status == "completed", f"Expected completed; got {outcome.status}"
    assert len(build_image_calls) == 1, (
        "build_image_fn must be called once for push=False apply"
    )
    assert len(build_push_calls) == 0, (
        "build_push_fn must NEVER be called for push=False apply — "
        "seam isolation violated"
    )


def test_build_local_runtime_wires_build_push_fn(tmp_path: Path) -> None:
    """``build_local_runtime`` must wire ``_local_build_push`` as the
    coordinator's ``build_push_fn`` so the local path can drive push mode."""
    from xrlenv.control.runtime import LocalRuntime, build_local_runtime

    with patch("xrlenv.control.runtime.DockerBackend") as MockDockerBackend:
        MockDockerBackend.return_value = MagicMock(name="docker")
        runtime = build_local_runtime(runs_root=tmp_path / "runs")

    assert isinstance(runtime, LocalRuntime)
    assert runtime.build_coordinator._build_push_fn is not None, (
        "build_push_fn must be wired on the coordinator by build_local_runtime"
    )
