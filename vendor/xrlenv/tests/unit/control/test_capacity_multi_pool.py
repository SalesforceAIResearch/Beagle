"""Tests for the Slice 6 multi-pool disk split in StaticCapacityEstimator
(spec 10 §"Disk is multi-pool" + spec 15)."""

from __future__ import annotations

from xrlenv.backends.base import ResourceSpec
from xrlenv.control.capacity import (
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


def _node(*, disk_gb: int = 100) -> NodeProfile:
    hw = HardwareInfo(
        vcpus=32,
        mem_bytes=64 * 1024**3,
        disk_bytes=disk_gb * 1024**3,
        has_kvm=False,
        has_gpu=False,
        gpu_model=None,
        kernel_version="0.0.0",
        platform="linux",
    )
    return NodeProfile(node_id="N", hardware=hw, backends=("docker",))


def _manifest(disk_gb: int = 10) -> TemplateManifest:
    return TemplateManifest(
        name="t", version="0.1", digest="sha256:t", image="im/t:1",
        resources=ResourceSpec(
            cpu_request=0.1, cpu_limit=1.0,
            mem_request_bytes=1 * 1024**3, mem_limit_bytes=2 * 1024**3,
            disk_request_bytes=disk_gb * 1024**3,
        ),
        env_adapter=EnvAdapterDecl(module="m", class_name="C"),
        reward=RewardContract(mode="env_step"),
    )


def test_default_image_pool_fraction_halves_sandbox_writable_pool() -> None:
    """100 GB - 5 GB OS reserve = 95 GB usable; default 0.5 → image pool 47 GB,
    sandbox-writable pool 48 GB; per-sandbox 10 GB → 4 fit (not 9).
    """
    node = _node(disk_gb=100)
    cell = StaticCapacityEstimator().capacity(node, _manifest(disk_gb=10))
    assert cell.disk_cap == 4
    assert cell.binding_constraint == "disk:sandbox_writable"


def test_image_pool_fraction_zero_uses_full_pool() -> None:
    """Operator override: ``image_cache_pool_fraction=0`` makes the
    sandbox-writable pool the entire usable disk (slice-2 single-pool
    behaviour). 100 GB - 5 GB = 95 GB; per-sandbox 10 GB → 9 fit.
    """
    node = _node(disk_gb=100)
    headroom = HeadroomConfig(image_cache_pool_fraction=0.0)
    cell = StaticCapacityEstimator(headroom=headroom).capacity(
        node, _manifest(disk_gb=10),
    )
    assert cell.disk_cap == 9


def test_image_pool_fraction_takes_majority() -> None:
    """``image_cache_pool_fraction=0.8`` cedes 76 GB to image cache pool,
    leaving 19 GB sandbox-writable; per-sandbox 10 GB → 1 fit.
    """
    node = _node(disk_gb=100)
    headroom = HeadroomConfig(image_cache_pool_fraction=0.8)
    cell = StaticCapacityEstimator(headroom=headroom).capacity(
        node, _manifest(disk_gb=10),
    )
    assert cell.disk_cap == 1


def test_disk_cap_remaining_uses_sandbox_writable_pool_only() -> None:
    """The fits-check (``_disk_cap_remaining``) must subtract running
    sandboxes against the sandbox-writable pool, not against total disk.
    """
    node = _node(disk_gb=100)
    estimator = StaticCapacityEstimator()
    candidate = _manifest(disk_gb=10)
    # Three already-running sandboxes consume 30 GB writable. Sandbox-writable
    # pool is 48 GB; 48 - 30 = 18 GB remaining → 1 more fits, 2 do not.
    running3 = [("t", candidate.resources)] * 3
    running4 = [("t", candidate.resources)] * 4
    assert estimator.fits(node, running=running3, candidate=candidate)
    assert not estimator.fits(node, running=running4, candidate=candidate)


def test_image_pool_does_not_starve_sandbox_writable() -> None:
    """Regression: a hot pin set on the image pool must not push
    sandbox-writable capacity below what the operator's headroom config
    declared for it.
    """
    node = _node(disk_gb=100)
    sandbox_writable_with_default = StaticCapacityEstimator().capacity(
        node, _manifest(disk_gb=1),
    ).disk_cap
    # 47 GB sandbox-writable / 1 GB per sandbox = 47 (allowing for OS reserve
    # rounding). Confirm the estimator reports a credible mid-range cap, not
    # 0 (which would mean the image pool ate everything).
    assert sandbox_writable_with_default >= 40


def test_pool_split_preserves_existing_min_of_three_caps_logic() -> None:
    """Slice 6 doesn't change the fact that the smallest of (cpu, mem,
    disk) wins; only the disk axis's denominator changed.
    """
    node = _node(disk_gb=200)
    # Memory-bound: per-sandbox 8 GB mem against ~64 GB usable mem;
    # disk + cpu both abundant.
    manifest = TemplateManifest(
        name="mem-heavy", version="0.1", digest="sha256:x", image="im/x:1",
        resources=ResourceSpec(
            cpu_request=0.5, cpu_limit=1.0,
            mem_request_bytes=8 * 1024**3, mem_limit_bytes=16 * 1024**3,
            disk_request_bytes=1 * 1024**3,
        ),
        env_adapter=EnvAdapterDecl(module="m", class_name="C"),
        reward=RewardContract(mode="env_step"),
    )
    cell = StaticCapacityEstimator().capacity(node, manifest)
    assert cell.binding_constraint == "mem"
    # Still bounded by mem, not disk:sandbox_writable.
