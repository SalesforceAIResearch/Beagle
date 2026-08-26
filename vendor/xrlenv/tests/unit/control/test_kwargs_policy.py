"""Unit tests for the cluster docker-kwarg policy module (issue #6).

Covers the four-tier validation matrix end-to-end:

* Level 0 — standard caps and image-shape kwargs pass through silently.
* Level 1 — devices / elevated caps default-allowed; operator overrides
  via ``allowed_devices`` / ``denied_caps`` work in both directions.
* Level 2 — host network / privileged / host binds default-reject with
  operator-friendly hint; ``allow_*`` flips honored.
* Level 3 — pid_mode=host, ipc_mode=host, cgroup_parent, cpuset_cpus,
  cpuset_mems, network_mode=container:* reject with no policy override.
* Level 4 — platform + non-host userns_mode reject with architectural
  rationale.

Also asserts ``validate_kwargs`` returns the full rejection set (not
first-only) so the operator-facing error names every problem in one go.
"""

from __future__ import annotations

import pytest
from xrlenv.control.kwargs_policy import (
    DEFAULT_POLICY,
    KwargsPolicy,
    KwargsPolicyViolation,
    KwargsRejection,
    validate_kwargs,
)

# ─── Level 0 (always allowed) ────────────────────────────────────────────────


def test_empty_kwargs_pass() -> None:
    """No kwargs supplied — clean."""
    assert validate_kwargs() == []


def test_level_0_caps_pass() -> None:
    """Standard caps in docker's default set never trip cap_add validation,
    even when the operator hasn't set up an explicit allowlist."""
    rejections = validate_kwargs(cap_add=["SYS_PTRACE", "NET_RAW", "CHOWN"])
    assert rejections == []


def test_level_0_caps_pass_with_cap_prefix() -> None:
    """``CAP_NET_RAW`` and ``NET_RAW`` are the same capability — validator
    normalizes."""
    rejections = validate_kwargs(cap_add=["CAP_SYS_PTRACE", "cap_chown"])
    assert rejections == []


# ─── Level 1 (operator-configurable) ────────────────────────────────────────


def test_default_allows_kvm_device() -> None:
    """SCUBA-style benchmarks need /dev/kvm; it's in the default allowlist."""
    rejections = validate_kwargs(devices=["/dev/kvm"])
    assert rejections == []


def test_default_allows_kvm_with_mount_spec() -> None:
    """docker-py accepts ``/dev/kvm:/dev/kvm:rwm`` — the validator should
    strip past the first colon to check the host device."""
    rejections = validate_kwargs(devices=["/dev/kvm:/dev/kvm:rwm"])
    assert rejections == []


def test_unlisted_device_rejected() -> None:
    """Device not in ``allowed_devices`` rejects with an operator-facing
    hint pointing at the yaml stanza to edit."""
    rejections = validate_kwargs(devices=["/dev/sda"])
    assert len(rejections) == 1
    r = rejections[0]
    assert r.kwarg == "devices"
    assert r.level == 1
    assert "/dev/sda" in r.reason
    assert r.hint is not None
    assert "allowed_devices" in r.hint


def test_device_allowlist_extended_by_policy() -> None:
    """Operator extends ``allowed_devices`` → previously-rejected device
    now passes."""
    policy = KwargsPolicy(
        allowed_devices=("/dev/kvm", "/dev/net/tun", "/dev/fuse", "/dev/dri/card0"),
    )
    assert validate_kwargs(devices=["/dev/dri/card0"], policy=policy) == []


def test_device_allowlist_restricted_by_policy() -> None:
    """Operator narrows ``allowed_devices`` → /dev/kvm rejects."""
    policy = KwargsPolicy(allowed_devices=())
    rejections = validate_kwargs(devices=["/dev/kvm"], policy=policy)
    assert len(rejections) == 1
    assert rejections[0].kwarg == "devices"


def test_default_allows_net_admin() -> None:
    """SCUBA-style benchmarks need NET_ADMIN; default allows."""
    rejections = validate_kwargs(cap_add=["NET_ADMIN"])
    assert rejections == []


def test_denied_cap_rejects() -> None:
    """Operator's ``denied_caps`` rejects matching kwargs with hint."""
    policy = KwargsPolicy(denied_caps=("SYS_MODULE",))
    rejections = validate_kwargs(cap_add=["NET_ADMIN", "SYS_MODULE"], policy=policy)
    assert len(rejections) == 1
    r = rejections[0]
    assert r.kwarg == "cap_add"
    assert r.level == 1
    assert "SYS_MODULE" in r.reason
    assert r.hint is not None
    assert "denied_caps" in r.hint


def test_denied_caps_normalize_cap_prefix() -> None:
    """``denied_caps: [CAP_SYS_MODULE]`` and ``[SYS_MODULE]`` are equivalent."""
    policy = KwargsPolicy(denied_caps=("CAP_SYS_MODULE",))
    rejections = validate_kwargs(cap_add=["sys_module"], policy=policy)
    assert len(rejections) == 1


# ─── Level 2 (default-reject; operator opt-in) ───────────────────────────────


def test_privileged_default_rejects() -> None:
    rejections = validate_kwargs(privileged=True)
    assert len(rejections) == 1
    r = rejections[0]
    assert r.kwarg == "privileged"
    assert r.level == 2
    assert r.hint is not None
    assert "allow_privileged" in r.hint


def test_privileged_false_passes() -> None:
    """``privileged=False`` is the docker-py default; never trip."""
    assert validate_kwargs(privileged=False) == []


def test_privileged_with_opt_in_passes() -> None:
    policy = KwargsPolicy(allow_privileged=True)
    assert validate_kwargs(privileged=True, policy=policy) == []


def test_network_mode_host_default_rejects() -> None:
    rejections = validate_kwargs(network_mode="host")
    assert len(rejections) == 1
    r = rejections[0]
    assert r.kwarg == "network_mode"
    assert r.level == 2
    assert r.hint is not None
    assert "allow_host_network" in r.hint


def test_network_mode_host_with_opt_in_passes() -> None:
    policy = KwargsPolicy(allow_host_network=True)
    assert validate_kwargs(network_mode="host", policy=policy) == []


def test_network_mode_bridge_passes() -> None:
    """Default bridge mode + ``none`` always allowed regardless of policy."""
    assert validate_kwargs(network_mode="bridge") == []
    assert validate_kwargs(network_mode="none") == []
    assert validate_kwargs(network_mode="default") == []


def test_binds_default_rejects() -> None:
    """Bind mount of host path that's not in ``allowed_host_paths`` rejects."""
    rejections = validate_kwargs(binds=["/var/data:/data:ro"])
    assert len(rejections) == 1
    r = rejections[0]
    assert r.kwarg == "binds"
    assert r.level == 2
    assert "/var/data" in r.reason


def test_binds_with_allowed_path_passes() -> None:
    policy = KwargsPolicy(allowed_host_paths=("/mnt/datasets",))
    rejections = validate_kwargs(
        binds=["/mnt/datasets:/data:ro"], policy=policy,
    )
    assert rejections == []


# ─── Level 3 (never allowed; no policy override) ─────────────────────────────


def test_pid_mode_host_rejects_unconditionally() -> None:
    rejections = validate_kwargs(pid_mode="host")
    assert len(rejections) == 1
    r = rejections[0]
    assert r.kwarg == "pid_mode"
    assert r.level == 3
    assert r.hint is None


def test_pid_mode_host_rejects_even_with_permissive_policy() -> None:
    """No policy override should unlock Level 3."""
    policy = KwargsPolicy(
        allow_privileged=True,
        allow_host_network=True,
        allowed_host_paths=("/",),
    )
    rejections = validate_kwargs(pid_mode="host", policy=policy)
    assert len(rejections) == 1
    assert rejections[0].level == 3


def test_pid_mode_private_passes() -> None:
    """Default ``pid_mode`` is the container's own namespace; never trip."""
    assert validate_kwargs(pid_mode="private") == []


def test_ipc_mode_host_rejects_unconditionally() -> None:
    rejections = validate_kwargs(ipc_mode="host")
    assert len(rejections) == 1
    assert rejections[0].kwarg == "ipc_mode"
    assert rejections[0].level == 3


def test_cgroup_parent_rejects_unconditionally() -> None:
    rejections = validate_kwargs(cgroup_parent="/system.slice/custom")
    assert len(rejections) == 1
    assert rejections[0].kwarg == "cgroup_parent"
    assert rejections[0].level == 3


def test_cpuset_cpus_rejects_unconditionally() -> None:
    rejections = validate_kwargs(cpuset_cpus="0-3")
    assert len(rejections) == 1
    assert rejections[0].kwarg == "cpuset_cpus"
    assert rejections[0].level == 3


def test_cpuset_mems_rejects_unconditionally() -> None:
    rejections = validate_kwargs(cpuset_mems="0")
    assert len(rejections) == 1
    assert rejections[0].kwarg == "cpuset_mems"
    assert rejections[0].level == 3


def test_network_mode_container_rejects_unconditionally() -> None:
    rejections = validate_kwargs(network_mode="container:other")
    assert len(rejections) == 1
    r = rejections[0]
    assert r.kwarg == "network_mode"
    assert r.level == 3


def test_network_mode_container_rejects_even_with_host_opt_in() -> None:
    """``allow_host_network`` only unlocks Level 2 (mode=host), not
    Level 3 (mode=container:...)."""
    policy = KwargsPolicy(allow_host_network=True)
    rejections = validate_kwargs(network_mode="container:foo", policy=policy)
    assert len(rejections) == 1
    assert rejections[0].level == 3


# ─── Level 4 (architectural mismatch) ────────────────────────────────────────


def test_platform_rejects() -> None:
    rejections = validate_kwargs(platform="linux/x86_64")
    assert len(rejections) == 1
    r = rejections[0]
    assert r.kwarg == "platform"
    assert r.level == 4


def test_userns_mode_non_host_rejects() -> None:
    rejections = validate_kwargs(userns_mode="private")
    assert len(rejections) == 1
    r = rejections[0]
    assert r.kwarg == "userns_mode"
    assert r.level == 4


def test_userns_mode_host_passes() -> None:
    """``host`` is the docker-py default for userns; never trip."""
    assert validate_kwargs(userns_mode="host") == []
    assert validate_kwargs(userns_mode="") == []
    assert validate_kwargs(userns_mode=None) == []


# ─── Aggregation behaviour ───────────────────────────────────────────────────


def test_multiple_rejections_collected() -> None:
    """Bad kwargs across multiple tiers all surface in one call so the
    operator fixes everything in one pass."""
    rejections = validate_kwargs(
        devices=["/dev/sda"],          # Level 1
        privileged=True,                # Level 2
        pid_mode="host",                # Level 3
        platform="linux/arm64",         # Level 4
    )
    assert {r.kwarg for r in rejections} == {
        "devices", "privileged", "pid_mode", "platform",
    }
    assert {r.level for r in rejections} == {1, 2, 3, 4}


def test_violation_exception_carries_rejections() -> None:
    rejections = validate_kwargs(privileged=True, pid_mode="host")
    exc = KwargsPolicyViolation(rejections)
    assert len(exc.rejections) == 2
    assert "privileged" in str(exc)
    assert "pid_mode" in str(exc)


def test_rejection_format_includes_hint_when_present() -> None:
    r = KwargsRejection(
        kwarg="devices", level=1, reason="not in allowlist", hint="add it",
    )
    formatted = r.format()
    assert "devices" in formatted
    assert "level 1" in formatted
    assert "fix: add it" in formatted


def test_rejection_format_omits_hint_when_absent() -> None:
    r = KwargsRejection(kwarg="pid_mode", level=3, reason="namespace escape")
    formatted = r.format()
    assert "fix:" not in formatted


# ─── DEFAULT_POLICY shape sanity (regression guard) ──────────────────────────


def test_default_policy_unblocks_known_benchmarks() -> None:
    """The default policy must let these real-world kwarg sets through
    without operator config. Codifies the design intent so a future
    accidental tightening trips this test."""
    # SCUBA-style: KVM + NET_ADMIN.
    assert validate_kwargs(
        devices=["/dev/kvm"], cap_add=["NET_ADMIN"],
        policy=DEFAULT_POLICY,
    ) == []
    # swebench: cap_add for ptrace, user root.
    assert validate_kwargs(
        cap_add=["SYS_PTRACE"], policy=DEFAULT_POLICY,
    ) == []
    # FUSE-based filesystem grader.
    assert validate_kwargs(
        devices=["/dev/fuse"], cap_add=["SYS_ADMIN"],
        policy=DEFAULT_POLICY,
    ) == []


def test_default_policy_immutable() -> None:
    """Pydantic ``frozen=True`` — mutation raises ``ValidationError``.

    Catches the broad pydantic exception base via the typed
    ``pydantic.ValidationError`` to satisfy ruff's blind-exception
    check while still tolerating future pydantic-version error
    refinements (the exact subclass for frozen-set assignment has
    moved between v1 and v2)."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        DEFAULT_POLICY.allowed_devices = ()  # type: ignore[misc]


# ─── §5.2 — container_runtime (allowed_runtimes) ─────────────────────────────


def test_default_policy_rejects_runtime_override() -> None:
    """§10.x(3) — the default policy rejects a non-default runtime override;
    it's a Level-2 (operator opt-in) rejection on the ``container_runtime``
    kwarg."""
    rejections = validate_kwargs(runtime="sysbox-runc")
    assert len(rejections) == 1
    assert rejections[0].kwarg == "container_runtime"
    assert rejections[0].level == 2


def test_runtime_override_allowed_when_opted_in() -> None:
    """§5.2 — a runtime the operator listed in ``allowed_runtimes`` passes."""
    policy = KwargsPolicy(allowed_runtimes=("sysbox-runc",))
    assert validate_kwargs(runtime="sysbox-runc", policy=policy) == []


def test_runc_and_none_runtime_always_allowed() -> None:
    """§5.2 — ``runc`` (the daemon default) and ``None`` are never overrides,
    so they pass under the default (empty) policy."""
    assert validate_kwargs(runtime="runc") == []
    assert validate_kwargs(runtime=None) == []


def test_unlisted_runtime_still_rejected_under_opt_in() -> None:
    """§5.2 — opting into one runtime does NOT open the door to others."""
    policy = KwargsPolicy(allowed_runtimes=("sysbox-runc",))
    rejections = validate_kwargs(runtime="kata-runtime", policy=policy)
    assert len(rejections) == 1
    assert rejections[0].kwarg == "container_runtime"


# ─── allowed_host_paths directory-prefix matching (sysbox real-bind fix) ──────


def test_allowed_host_paths_allows_subtree() -> None:
    """An allowed_host_paths entry allows binds nested under it (dynamic
    per-run paths, e.g. the sysbox golden_cache mount)."""
    p = KwargsPolicy(allowed_host_paths=("/fsx/data/evoclaw-golden",))
    assert validate_kwargs(
        binds=["/fsx/data/evoclaw-golden/golden_cache/.mount/abc:/golden:ro"],
        policy=p,
    ) == []
    assert validate_kwargs(binds=["/fsx/data/evoclaw-golden:/x"], policy=p) == []


def test_allowed_host_paths_prefix_does_not_match_sibling() -> None:
    """The trailing-slash guard stops a prefix from matching a sibling dir."""
    p = KwargsPolicy(allowed_host_paths=("/mnt/data",))
    rej = validate_kwargs(binds=["/mnt/database/x:/y"], policy=p)
    assert len(rej) == 1 and rej[0].kwarg == "binds"


def test_allowed_host_paths_empty_rejects_all_binds() -> None:
    """Default (empty allowed_host_paths) still rejects every bind."""
    rej = validate_kwargs(binds=["/fsx/x:/y"], policy=DEFAULT_POLICY)
    assert len(rej) == 1 and rej[0].kwarg == "binds"
