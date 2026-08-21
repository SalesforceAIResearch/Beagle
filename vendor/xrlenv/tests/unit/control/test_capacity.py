"""Tests for the static capacity estimator (spec 10)."""

from __future__ import annotations

import pytest
from xrlenv.backends.base import ResourceSpec
from xrlenv.control.capacity import (
    BackendOverhead,
    HeadroomConfig,
    NodeProfile,
    StaticCapacityEstimator,
)
from xrlenv.control.template_catalog import (
    EnvAdapterDecl,
    RewardContract,
    TemplateManifest,
)
from xrlenv.node.hw_probe import HardwareInfo


def _hw(
    *,
    vcpus: int = 8,
    mem_gb: int = 32,
    disk_gb: int = 200,
    has_gpu: bool = False,
) -> HardwareInfo:
    return HardwareInfo(
        vcpus=vcpus,
        mem_bytes=mem_gb * 1024**3,
        disk_bytes=disk_gb * 1024**3,
        has_kvm=True,
        has_gpu=has_gpu,
        gpu_model=None,
        kernel_version="6.0.0",
        platform="linux",
    )


def _node(
    node_id: str = "node-A",
    *,
    backends: tuple[str, ...] = ("docker",),
    **hw_kwargs: object,
) -> NodeProfile:
    return NodeProfile(node_id=node_id, hardware=_hw(**hw_kwargs), backends=backends)  # type: ignore[arg-type]


def _manifest(
    name: str = "swebench-base",
    *,
    backend: str = "docker",
    cpu_request: float = 1.0,
    mem_gb: int = 4,
    disk_gb: int = 8,
    gpu_required: bool = False,
) -> TemplateManifest:
    return TemplateManifest(
        name=name,
        version="0.1",
        digest=f"sha256:{name}",
        image=f"im/{name}:0.1",

        resources=ResourceSpec(
            cpu_request=cpu_request,
            cpu_limit=cpu_request,
            mem_request_bytes=mem_gb * 1024**3,
            mem_limit_bytes=mem_gb * 1024**3,
            disk_request_bytes=disk_gb * 1024**3,
            gpu_required=gpu_required,
        ),
        env_adapter=EnvAdapterDecl(module="m", class_name="C"),
        reward=RewardContract(mode="env_step"),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Capacity math
# ──────────────────────────────────────────────────────────────────────────────


def test_capacity_zero_when_backend_missing() -> None:
    node = _node(backends=("cubesandbox",))
    cell = StaticCapacityEstimator().capacity(node, _manifest(backend="docker"))
    assert cell.max_concurrent == 0
    assert cell.binding_constraint == "backend_missing"


def test_capacity_zero_when_gpu_missing() -> None:
    node = _node(has_gpu=False)
    cell = StaticCapacityEstimator().capacity(node, _manifest(gpu_required=True))
    assert cell.max_concurrent == 0
    assert cell.binding_constraint == "gpu"


def test_cpu_bound_template() -> None:
    # 8 vcpus * 0.9 headroom = 7.2 usable; per-sandbox = 4.0 + 0.05 overhead = 4.05.
    # → cpu_cap = floor(7.2 / 4.05) = 1
    node = _node(vcpus=8, mem_gb=64, disk_gb=400)
    cell = StaticCapacityEstimator().capacity(
        node, _manifest(cpu_request=4.0, mem_gb=2, disk_gb=4)
    )
    assert cell.cpu_cap == 1
    assert cell.binding_constraint == "cpu"
    assert cell.max_concurrent == 1


def test_mem_bound_template() -> None:
    # 16 GiB * 0.85 ≈ 13.6 GiB; per-sandbox = 8 GiB + 50 MiB overhead.
    # → mem_cap = floor(13.6 / 8.05) = 1
    node = _node(vcpus=32, mem_gb=16, disk_gb=400)
    cell = StaticCapacityEstimator().capacity(
        node, _manifest(cpu_request=0.5, mem_gb=8, disk_gb=4)
    )
    assert cell.binding_constraint == "mem"
    assert cell.mem_cap == 1


def test_disk_bound_template() -> None:
    # Slice 6 multi-pool: 50 GiB - 5 GiB OS reserve = 45 GiB usable;
    # default image-cache pool fraction is 0.5 → image_cache pool gets
    # 22 GiB, sandbox_writable pool gets 23 GiB; per-sandbox 20 GiB →
    # 1 fits on the sandbox-writable pool.
    node = _node(vcpus=32, mem_gb=64, disk_gb=50)
    cell = StaticCapacityEstimator().capacity(
        node, _manifest(cpu_request=0.5, mem_gb=2, disk_gb=20)
    )
    assert cell.disk_cap == 1
    assert cell.binding_constraint == "disk:sandbox_writable"


def test_min_of_three_caps_wins() -> None:
    # cpu:5, mem:3, disk:7 → max = 3, binding = mem.
    node = _node(vcpus=16, mem_gb=12, disk_gb=200)
    cell = StaticCapacityEstimator().capacity(
        node, _manifest(cpu_request=2.0, mem_gb=3, disk_gb=20)
    )
    assert cell.binding_constraint == "mem"
    assert cell.max_concurrent == cell.mem_cap


# ──────────────────────────────────────────────────────────────────────────────
# fits() — placement decisions
# ──────────────────────────────────────────────────────────────────────────────


def test_fits_true_on_empty_node() -> None:
    est = StaticCapacityEstimator()
    node = _node(vcpus=8, mem_gb=16, disk_gb=200)
    m = _manifest(cpu_request=1.0, mem_gb=2, disk_gb=4)
    assert est.fits(node, running=[], candidate=m)


def test_fits_false_when_cpu_already_committed() -> None:
    est = StaticCapacityEstimator()
    node = _node(vcpus=4, mem_gb=64, disk_gb=400)
    m = _manifest(cpu_request=2.0, mem_gb=2, disk_gb=4)
    # Two of m already running → 4.0 cpu committed; node has only 3.6 usable.
    running = [(m.name, m.resources), (m.name, m.resources)]
    assert est.fits(node, running=running, candidate=m) is False


def test_fits_respects_per_task_cap() -> None:
    est = StaticCapacityEstimator()
    node = _node()
    m = _manifest(cpu_request=0.25, mem_gb=1, disk_gb=2)
    assert (
        est.fits(
            node,
            running=[],
            candidate=m,
            task_key="instance-x",
            task_count_on_node=4,
            max_runs_per_task=4,
        )
        is False
    )
    assert est.fits(
        node,
        running=[],
        candidate=m,
        task_key="instance-x",
        task_count_on_node=3,
        max_runs_per_task=4,
    )


def test_fits_false_when_backend_missing() -> None:
    est = StaticCapacityEstimator()
    node = _node(backends=("cubesandbox",))
    m = _manifest()
    # Default backend is "docker"; node only supports cubesandbox →
    # capability filter rejects.
    assert est.fits(node, running=[], candidate=m) is False


def test_fits_false_when_gpu_required() -> None:
    est = StaticCapacityEstimator()
    node = _node(has_gpu=False)
    m = _manifest(gpu_required=True)
    assert est.fits(node, running=[], candidate=m) is False


# ──────────────────────────────────────────────────────────────────────────────
# Mixed-template packing
# ──────────────────────────────────────────────────────────────────────────────


def test_fits_accounts_for_mixed_templates() -> None:
    est = StaticCapacityEstimator()
    node = _node(vcpus=8, mem_gb=16, disk_gb=400)
    a = _manifest("a", cpu_request=2.0, mem_gb=4)
    b = _manifest("b", cpu_request=1.0, mem_gb=2, disk_gb=4)
    # a using 4.0 cpu of 7.2 usable; b candidate 1.0 + 0.05 overhead → fits twice.
    running = [(a.name, a.resources), (a.name, a.resources)]
    assert est.fits(node, running=running, candidate=b)


def test_fits_accounts_for_per_task_resource_overlay() -> None:
    """Slice 9b — Pattern A: ``running`` carries each sandbox's
    *effective* (post-resolver-overlay) resources, not the outer
    manifest's defaults. A heavy per-task instance pinned to a node
    blocks a light candidate from over-admitting.

    Without this, two instances of the same outer template at heavy
    per-task cost (e.g. 3 CPU each) would be charged the outer
    manifest's 0.5 CPU and the scheduler would happily over-admit a
    third.
    """
    est = StaticCapacityEstimator()
    node = _node(vcpus=8, mem_gb=16, disk_gb=400)
    candidate = _manifest("tb2", cpu_request=0.5, mem_gb=1)  # outer "default"
    # Two heavy per-task instances already running on the node.
    heavy = ResourceSpec(
        cpu_request=3.0, cpu_limit=3.0,
        mem_request_bytes=4 * 1024**3, mem_limit_bytes=4 * 1024**3,
        disk_request_bytes=4 * 1024**3,
    )
    running = [("tb2", heavy), ("tb2", heavy)]
    # Outer-manifest accounting would say "1.0 CPU committed → 6.2 free
    # after headroom → fits". Effective accounting says "6.0 CPU
    # committed → 1.2 free → can squeeze a 0.5 + 0.05 overhead".
    # Bump candidate so it doesn't fit.
    big_candidate = _manifest("tb2", cpu_request=2.0, mem_gb=2)
    assert est.fits(node, running=running, candidate=big_candidate) is False
    # Light candidate still fits (slack remains).
    assert est.fits(node, running=running, candidate=candidate)


# ──────────────────────────────────────────────────────────────────────────────
# Headroom + matrix
# ──────────────────────────────────────────────────────────────────────────────


def test_custom_headroom_changes_caps() -> None:
    node = _node(vcpus=8, mem_gb=16, disk_gb=200)
    m = _manifest(cpu_request=1.0, mem_gb=1, disk_gb=4)
    relaxed = StaticCapacityEstimator(headroom=HeadroomConfig(cpu_fraction=0.0, mem_fraction=0.0))
    strict = StaticCapacityEstimator(headroom=HeadroomConfig(cpu_fraction=0.5, mem_fraction=0.5))
    assert relaxed.capacity(node, m).max_concurrent > strict.capacity(node, m).max_concurrent


def test_matrix_returns_one_cell_per_pair() -> None:
    """matrix() iterates (node, template) for a single backend per
    call. The default backend is ``docker``; pass ``backend=`` to
    project the matrix for a different runtime."""
    nodes = [_node("a"), _node("b", backends=("cubesandbox",))]
    manifests = [_manifest("t1"), _manifest("t2")]
    # Default (docker): node b doesn't support docker → all b cells 0.
    cells = StaticCapacityEstimator().matrix(nodes, manifests)
    assert len(cells) == 4
    keyed = {(c.node_id, c.template): c for c in cells}
    assert keyed[("a", "t1")].max_concurrent > 0
    assert keyed[("b", "t1")].max_concurrent == 0  # no docker on b
    assert keyed[("b", "t2")].max_concurrent == 0  # same — backend is per-call

    # Project the matrix for cubesandbox: now b's cells are non-zero.
    cube_cells = StaticCapacityEstimator().matrix(
        nodes, manifests, backend="cubesandbox",
    )
    cube_keyed = {(c.node_id, c.template): c for c in cube_cells}
    assert cube_keyed[("a", "t1")].max_concurrent == 0  # a doesn't have cube
    assert cube_keyed[("b", "t1")].max_concurrent > 0
    assert cube_keyed[("b", "t2")].max_concurrent > 0


def test_overhead_override() -> None:
    # Make Docker overhead dwarf any sane sandbox so capacity drops to zero.
    est = StaticCapacityEstimator(
        backend_overhead={
            "docker": BackendOverhead(cpu_per_sandbox=64.0, mem_bytes_per_sandbox=0),
        }
    )
    cell = est.capacity(_node(vcpus=8, mem_gb=64, disk_gb=400), _manifest())
    assert cell.cpu_cap == 0
    assert cell.binding_constraint == "cpu"


# ──────────────────────────────────────────────────────────────────────────────
# Edge cases
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("cpu_req", [0.0, -1.0])
def test_zero_or_negative_cpu_request_treated_as_unbounded(cpu_req: float) -> None:
    est = StaticCapacityEstimator()
    node = _node()
    m = _manifest(cpu_request=cpu_req, mem_gb=1, disk_gb=4)
    cell = est.capacity(node, m)
    assert cell.cpu_cap >= 1000  # effectively unbounded
